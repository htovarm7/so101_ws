"""Detect drop-off zones (pink = ZONE_A, orange = ZONE_B) with the RealSense D435.

Sibling of ``object_classifier``.  For every synchronised colour+depth frame it
finds the largest pink contour and the largest orange contour, deprojects each
centroid to 3-D, transforms it to ``target_frame`` (default ``base_link``) and
publishes a ``PointStamped`` per zone.

A higher-level node can subscribe to the two zone topics and combine them with
``object_classifier``'s output to decide where to drop each object.
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.duration import Duration as RclDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transforms)

from geometry_msgs.msg import PointStamped, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

from message_filters import ApproximateTimeSynchronizer, Subscriber


class _ZoneSpec:
    __slots__ = ("name", "lower", "upper", "min_area", "wrap", "rgb")

    def __init__(self, name: str, lower, upper, min_area: int,
                 rgb: Tuple[int, int, int]) -> None:
        self.name = name
        self.lower = np.array(lower, dtype=np.uint8)
        self.upper = np.array(upper, dtype=np.uint8)
        self.min_area = int(min_area)
        self.wrap = bool(self.lower[0] > self.upper[0])
        self.rgb = rgb


class ZoneDetector(Node):

    def __init__(self) -> None:
        super().__init__("zone_detector")

        # ── Camera + frames ─────────────────────────────────────────────
        self.declare_parameter("color_image_topic",
                               "/camera/camera/color/image_raw")
        self.declare_parameter("depth_image_topic",
                               "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic",
                               "/camera/camera/color/camera_info")

        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("parent_link",  "moving_jaw_so101_v1_link")

        # Static TF gripper-link -> camera_color_optical_frame.
        # Same values as object_classifier / blue_object_detector — change
        # both if the mount changes.
        self.declare_parameter("mount_x",  0.0)
        self.declare_parameter("mount_y",  0.0)
        self.declare_parameter("mount_z", -0.02)
        self.declare_parameter("mount_qx", -0.5)
        self.declare_parameter("mount_qy",  0.5)
        self.declare_parameter("mount_qz", -0.5)
        self.declare_parameter("mount_qw", -0.5)

        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("max_depth_m", 3.0)
        self.declare_parameter("marker_lifetime_s", 1.0)

        # ── Zone A: PINK ────────────────────────────────────────────────
        self.declare_parameter("zone_a.hsv_lower", [140, 50, 50])
        self.declare_parameter("zone_a.hsv_upper", [170, 255, 255])
        self.declare_parameter("zone_a.min_contour_area", 5000)

        # ── Zone B: ORANGE ──────────────────────────────────────────────
        self.declare_parameter("zone_b.hsv_lower", [5, 100, 100])
        self.declare_parameter("zone_b.hsv_upper", [20, 255, 255])
        self.declare_parameter("zone_b.min_contour_area", 5000)

        # Broadcast the static gripper->camera TF, like the other
        # perception nodes do.  Disable this if another node in the launch
        # already publishes it (otherwise duplicate static TFs are harmless
        # but spammy).
        self.declare_parameter("publish_camera_tf", True)

        self._target_frame = self.get_parameter("target_frame").value
        self._parent_link  = self.get_parameter("parent_link").value
        self._mount = {k: self.get_parameter(f"mount_{k}").value
                       for k in ("x", "y", "z", "qx", "qy", "qz", "qw")}
        self._depth_scale = float(self.get_parameter("depth_scale").value)
        self._max_depth   = float(self.get_parameter("max_depth_m").value)
        self._publish_camera_tf_enabled = bool(
            self.get_parameter("publish_camera_tf").value
        )

        marker_lifetime = float(self.get_parameter("marker_lifetime_s").value)
        self._marker_lifetime = Duration(
            sec=int(marker_lifetime),
            nanosec=int((marker_lifetime % 1.0) * 1e9),
        )

        self._zones = [
            _ZoneSpec(
                "zone_a",
                self.get_parameter("zone_a.hsv_lower").value,
                self.get_parameter("zone_a.hsv_upper").value,
                self.get_parameter("zone_a.min_contour_area").value,
                rgb=(255, 0, 255),   # magenta
            ),
            _ZoneSpec(
                "zone_b",
                self.get_parameter("zone_b.hsv_lower").value,
                self.get_parameter("zone_b.hsv_upper").value,
                self.get_parameter("zone_b.min_contour_area").value,
                rgb=(255, 140, 0),   # orange
            ),
        ]

        self._fx = self._fy = self._cx = self._cy = None
        self._camera_frame: str = "camera_color_optical_frame"
        self._bridge = CvBridge()

        self._static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        color_topic = self.get_parameter("color_image_topic").value
        depth_topic = self.get_parameter("depth_image_topic").value
        info_topic  = self.get_parameter("camera_info_topic").value

        self._info_sub = self.create_subscription(
            CameraInfo, info_topic, self._camera_info_cb, sensor_qos
        )
        self._color_sub = Subscriber(self, Image, color_topic, qos_profile=sensor_qos)
        self._depth_sub = Subscriber(self, Image, depth_topic, qos_profile=sensor_qos)
        self._sync = ApproximateTimeSynchronizer(
            [self._color_sub, self._depth_sub], queue_size=5, slop=0.05
        )
        self._sync.registerCallback(self._image_cb)

        self._zone_pubs = {
            spec.name: self.create_publisher(
                PointStamped, f"~/{spec.name}", 10
            )
            for spec in self._zones
        }
        self._marker_pub = self.create_publisher(MarkerArray, "~/markers", 10)
        self._debug_pub  = self.create_publisher(Image, "~/debug_image", sensor_qos)

        self.get_logger().info(
            "ZoneDetector ready\n"
            f"  colour       : {color_topic}\n"
            f"  depth        : {depth_topic}\n"
            f"  parent_link  : {self._parent_link}\n"
            f"  target_frame : {self._target_frame}\n"
            f"  zones        : "
            f"A(pink) {self._zones[0].lower.tolist()}-{self._zones[0].upper.tolist()}, "
            f"B(orange) {self._zones[1].lower.tolist()}-{self._zones[1].upper.tolist()}"
        )

    # ── TF bridge ───────────────────────────────────────────────────────
    def _publish_camera_tf(self) -> None:
        if not self._publish_camera_tf_enabled:
            return
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self._parent_link
        tf.child_frame_id  = self._camera_frame
        tf.transform.translation.x = float(self._mount["x"])
        tf.transform.translation.y = float(self._mount["y"])
        tf.transform.translation.z = float(self._mount["z"])
        tf.transform.rotation.x    = float(self._mount["qx"])
        tf.transform.rotation.y    = float(self._mount["qy"])
        tf.transform.rotation.z    = float(self._mount["qz"])
        tf.transform.rotation.w    = float(self._mount["qw"])
        self._static_broadcaster.sendTransform(tf)

    # ── CameraInfo ──────────────────────────────────────────────────────
    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self._fx is not None:
            return
        k = msg.k
        self._fx, self._fy = float(k[0]), float(k[4])
        self._cx, self._cy = float(k[2]), float(k[5])
        if msg.header.frame_id:
            self._camera_frame = msg.header.frame_id
        self.get_logger().info(
            f"CameraInfo: fx={self._fx:.1f} fy={self._fy:.1f} "
            f"cx={self._cx:.1f} cy={self._cy:.1f} frame={self._camera_frame}"
        )
        self._publish_camera_tf()

    # ── Detection ───────────────────────────────────────────────────────
    @staticmethod
    def _mask_for(hsv: np.ndarray, spec: _ZoneSpec) -> np.ndarray:
        if not spec.wrap:
            mask = cv2.inRange(hsv, spec.lower, spec.upper)
        else:
            lo1 = np.array([0,             spec.lower[1], spec.lower[2]], dtype=np.uint8)
            hi1 = np.array([spec.upper[0], spec.upper[1], spec.upper[2]], dtype=np.uint8)
            lo2 = np.array([spec.lower[0], spec.lower[1], spec.lower[2]], dtype=np.uint8)
            hi2 = np.array([179,           spec.upper[1], spec.upper[2]], dtype=np.uint8)
            mask = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1),
                                  cv2.inRange(hsv, lo2, hi2))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _largest_contour(self, mask: np.ndarray, min_area: int):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0, None
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < min_area:
            return None, area, None
        M = cv2.moments(c)
        if M["m00"] <= 0:
            return None, area, None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return c, area, (cx, cy)

    def _image_cb(self, color_msg: Image, depth_msg: Image) -> None:
        if self._fx is None:
            return

        try:
            bgr   = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}")
            return

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        debug = bgr.copy()
        markers = MarkerArray()

        for idx, spec in enumerate(self._zones):
            mask = self._mask_for(hsv, spec)
            contour, area, centroid = self._largest_contour(mask, spec.min_area)
            if contour is None or centroid is None:
                cv2.putText(debug, f"{spec.name}: --",
                            (12, 28 + idx * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
                continue

            cx_px, cy_px = centroid
            r, g, b = spec.rgb
            bgr_colour = (b, g, r)
            cv2.drawContours(debug, [contour], -1, bgr_colour, 2)
            cv2.circle(debug, (cx_px, cy_px), 6, bgr_colour, -1)
            cv2.putText(debug, f"{spec.name} A={int(area)}",
                        (cx_px + 8, cy_px - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr_colour, 2)

            z_m = self._sample_depth(depth, cx_px, cy_px)
            if z_m is None or not (0.0 < z_m < self._max_depth):
                continue

            X = (cx_px - self._cx) * z_m / self._fx
            Y = (cy_px - self._cy) * z_m / self._fy
            Z = z_m

            tgt = self._point_in_target_frame(color_msg.header.stamp, X, Y, Z)
            if tgt is None:
                continue

            self._zone_pubs[spec.name].publish(tgt)

            m = Marker()
            m.header = tgt.header
            m.ns = "zone_detector"
            m.id = idx
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = tgt.point.x
            m.pose.position.y = tgt.point.y
            m.pose.position.z = tgt.point.z
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.05
            m.color.r = r / 255.0
            m.color.g = g / 255.0
            m.color.b = b / 255.0
            m.color.a = 0.9
            m.lifetime = self._marker_lifetime
            markers.markers.append(m)

        if markers.markers:
            self._marker_pub.publish(markers)

        try:
            dbg_msg = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            dbg_msg.header = color_msg.header
            self._debug_pub.publish(dbg_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish debug image: {e}")

    def _sample_depth(self, depth: np.ndarray, x: int, y: int) -> Optional[float]:
        h, w = depth.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        x0, x1 = max(0, x - 2), min(w, x + 3)
        y0, y1 = max(0, y - 2), min(h, y + 3)
        patch = depth[y0:y1, x0:x1].astype(np.float32)
        patch = patch[patch > 0]
        if patch.size == 0:
            return None
        return float(np.median(patch) * self._depth_scale)

    def _point_in_target_frame(self, stamp, X: float, Y: float, Z: float
                               ) -> Optional[PointStamped]:
        cam_pt = PointStamped()
        cam_pt.header.stamp = stamp
        cam_pt.header.frame_id = self._camera_frame
        cam_pt.point.x = float(X)
        cam_pt.point.y = float(Y)
        cam_pt.point.z = float(Z)
        try:
            return self._tf_buffer.transform(
                cam_pt, self._target_frame,
                timeout=RclDuration(seconds=0.1),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as e:
            self.get_logger().warn(
                f"TF {self._camera_frame} -> {self._target_frame} failed: {e}"
            )
            return None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ZoneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
