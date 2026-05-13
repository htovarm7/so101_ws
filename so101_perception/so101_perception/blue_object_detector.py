"""Blue-object detector using RealSense D435 colour + aligned depth streams.

TF bridge
---------
This node owns the static TF that connects the camera to the robot.
On startup it broadcasts:

    parent_link  ──(static)──>  camera_color_optical_frame

using a StaticTransformBroadcaster.  This is more reliable than an external
static_transform_publisher node because the transform is guaranteed to be
in this node's own TF buffer before the first image arrives.

Default transform values match the so101_cameras.xacro wrist-camera mount
composed with the standard optical-frame rotation.  Tune mount_* params if
your physical camera position differs.
"""

import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.duration import Duration as RclDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import numpy as np
from cv_bridge import CvBridge

import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped transform support

from geometry_msgs.msg import PointStamped, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration

from message_filters import ApproximateTimeSynchronizer, Subscriber


class BlueObjectDetector(Node):

    def __init__(self):
        super().__init__("blue_object_detector")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("color_image_topic",
                               "/camera/camera/color/image_raw")
        self.declare_parameter("depth_image_topic",
                               "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic",
                               "/camera/camera/color/camera_info")

        # Robot frame to publish detections in (and TF bridge parent).
        self.declare_parameter("target_frame", "base_link")

        # Parent link for the static TF bridge: parent_link → camera_optical_frame.
        # Must be a frame already in the robot TF tree.
        self.declare_parameter("parent_link", "moving_jaw_so101_v1_link")

        # Camera mount transform (parent_link → camera_color_optical_frame).
        # Quaternion is the composition of the wrist-camera URDF joint
        # (rpy=-π/2,0,-π/2) plus the standard optical-frame rotation (same rpy).
        # Translation: camera is 2 cm below the gripper tip along Z.
        self.declare_parameter("mount_x",  0.0)
        self.declare_parameter("mount_y",  0.0)
        self.declare_parameter("mount_z", -0.02)
        self.declare_parameter("mount_qx", -0.5)
        self.declare_parameter("mount_qy",  0.5)
        self.declare_parameter("mount_qz", -0.5)
        self.declare_parameter("mount_qw", -0.5)

        # HSV thresholds for blue
        self.declare_parameter("hsv_lower", [100, 80, 50])
        self.declare_parameter("hsv_upper", [130, 255, 255])
        self.declare_parameter("min_contour_area", 500)
        self.declare_parameter("marker_lifetime_s", 0.3)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("max_depth_m", 3.0)

        color_topic    = self.get_parameter("color_image_topic").value
        depth_topic    = self.get_parameter("depth_image_topic").value
        info_topic     = self.get_parameter("camera_info_topic").value
        self._target_frame = self.get_parameter("target_frame").value
        self._parent_link  = self.get_parameter("parent_link").value
        self._mount = {
            "x":  self.get_parameter("mount_x").value,
            "y":  self.get_parameter("mount_y").value,
            "z":  self.get_parameter("mount_z").value,
            "qx": self.get_parameter("mount_qx").value,
            "qy": self.get_parameter("mount_qy").value,
            "qz": self.get_parameter("mount_qz").value,
            "qw": self.get_parameter("mount_qw").value,
        }
        self.hsv_lower   = np.array(self.get_parameter("hsv_lower").value, dtype=np.uint8)
        self.hsv_upper   = np.array(self.get_parameter("hsv_upper").value, dtype=np.uint8)
        self.min_area    = self.get_parameter("min_contour_area").value
        self.depth_scale = self.get_parameter("depth_scale").value
        self.max_depth   = self.get_parameter("max_depth_m").value
        marker_lifetime  = self.get_parameter("marker_lifetime_s").value

        self._marker_lifetime = Duration(
            sec=int(marker_lifetime),
            nanosec=int((marker_lifetime % 1) * 1e9),
        )

        self._fx = self._fy = self._cx = self._cy = None
        self._camera_frame: str = "camera_color_optical_frame"
        self._bridge = CvBridge()

        # ── TF broadcaster + listener ──────────────────────────────────────────
        self._static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Subscriptions ─────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._info_sub = self.create_subscription(
            CameraInfo, info_topic, self._camera_info_cb, sensor_qos
        )
        self._color_sub = Subscriber(self, Image, color_topic, qos_profile=sensor_qos)
        self._depth_sub = Subscriber(self, Image, depth_topic, qos_profile=sensor_qos)
        self._sync = ApproximateTimeSynchronizer(
            [self._color_sub, self._depth_sub], queue_size=5, slop=0.05
        )
        self._sync.registerCallback(self._image_cb)

        # ── Publishers ────────────────────────────────────────────────────────
        self._point_pub  = self.create_publisher(PointStamped, "~/detected_point", 10)
        self._marker_pub = self.create_publisher(Marker,       "~/marker",         10)
        self._debug_pub  = self.create_publisher(Image,        "~/debug_image",    sensor_qos)

        self.get_logger().info(
            f"BlueObjectDetector ready\n"
            f"  colour       : {color_topic}\n"
            f"  depth        : {depth_topic}\n"
            f"  parent_link  : {self._parent_link}\n"
            f"  target_frame : {self._target_frame}\n"
            f"  HSV          : lower={self.hsv_lower.tolist()}  upper={self.hsv_upper.tolist()}"
        )

    # ── TF bridge ─────────────────────────────────────────────────────────────

    def _publish_camera_tf(self):
        """Broadcast parent_link → camera_frame as a static transform."""
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = self._parent_link
        t.child_frame_id  = self._camera_frame
        t.transform.translation.x = self._mount["x"]
        t.transform.translation.y = self._mount["y"]
        t.transform.translation.z = self._mount["z"]
        t.transform.rotation.x = self._mount["qx"]
        t.transform.rotation.y = self._mount["qy"]
        t.transform.rotation.z = self._mount["qz"]
        t.transform.rotation.w = self._mount["qw"]
        self._static_broadcaster.sendTransform(t)
        self.get_logger().info(
            f"Static TF published: {self._parent_link} → {self._camera_frame}"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo):
        if self._fx is not None:
            return
        self._fx = msg.k[0]
        self._fy = msg.k[4]
        self._cx = msg.k[2]
        self._cy = msg.k[5]
        self._camera_frame = msg.header.frame_id
        self.get_logger().info(
            f"Camera intrinsics — fx={self._fx:.1f} fy={self._fy:.1f} "
            f"cx={self._cx:.1f} cy={self._cy:.1f}  frame='{self._camera_frame}'"
        )
        # Now that we know the actual camera frame name, publish the static TF.
        self._publish_camera_tf()
        self.destroy_subscription(self._info_sub)

    def _image_cb(self, color_msg: Image, depth_msg: Image):
        if self._fx is None:
            self.get_logger().warn("Waiting for camera_info…",
                                   throttle_duration_sec=3.0)
            return

        # ── Decode ────────────────────────────────────────────────────────────
        color_bgr = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        depth_raw = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        # ── Blue mask ─────────────────────────────────────────────────────────
        hsv  = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        debug_img = color_bgr.copy()
        cv2.putText(debug_img, "Blue detector", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if not contours:
            self._publish_debug(debug_img, color_msg.header)
            return

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_area:
            self._publish_debug(debug_img, color_msg.header)
            return

        # ── Centroid ──────────────────────────────────────────────────────────
        M = cv2.moments(largest)
        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])

        # ── Depth: median over the entire blob ───────────────────────────────
        # Sampling the full contour area instead of just the centroid makes
        # the depth estimate robust against holes caused by moving objects.
        h, w = depth_raw.shape[:2]
        blob_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(blob_mask, [largest], -1, 255, cv2.FILLED)
        depths_in_blob = depth_raw[blob_mask > 0].astype(np.float32)
        valid = depths_in_blob[depths_in_blob > 0]

        if valid.size == 0:
            # Fallback: wider neighbourhood around the centroid
            patch = depth_raw[max(0, v-5):min(h, v+6),
                              max(0, u-5):min(w, u+6)].astype(np.float32)
            valid = patch[patch > 0]

        if valid.size == 0:
            self.get_logger().warn("Depth zero across blob — skipping.",
                                   throttle_duration_sec=1.0)
            self._publish_debug(debug_img, color_msg.header)
            return

        depth_m = float(np.median(valid)) * self.depth_scale
        if depth_m <= 0.0 or depth_m > self.max_depth:
            self.get_logger().warn(f"Depth {depth_m:.3f} m out of range — skipping.",
                                   throttle_duration_sec=1.0)
            self._publish_debug(debug_img, color_msg.header)
            return

        # ── Back-project to 3-D in camera frame ───────────────────────────────
        X_cam = (u - self._cx) * depth_m / self._fx
        Y_cam = (v - self._cy) * depth_m / self._fy
        Z_cam = depth_m

        pt_cam = PointStamped()
        pt_cam.header.stamp    = color_msg.header.stamp
        pt_cam.header.frame_id = self._camera_frame
        pt_cam.point.x = X_cam
        pt_cam.point.y = Y_cam
        pt_cam.point.z = Z_cam

        # ── Transform to target frame (base_link) using latest TF ─────────────
        publish_frame = self._camera_frame
        pt_out = pt_cam
        try:
            pt_latest = PointStamped()
            pt_latest.header.frame_id = self._camera_frame
            pt_latest.header.stamp    = rclpy.time.Time().to_msg()
            pt_latest.point           = pt_cam.point

            pt_out = self._tf_buffer.transform(
                pt_latest, self._target_frame,
                timeout=RclDuration(seconds=0.1),
            )
            pt_out.header.stamp = color_msg.header.stamp
            publish_frame = self._target_frame
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF {self._camera_frame} → {self._target_frame}: {e}",
                throttle_duration_sec=2.0,
            )

        X, Y, Z = pt_out.point.x, pt_out.point.y, pt_out.point.z

        # ── Publish PointStamped ──────────────────────────────────────────────
        self._point_pub.publish(pt_out)

        # ── Publish sphere Marker ─────────────────────────────────────────────
        marker = Marker()
        marker.header.stamp    = color_msg.header.stamp
        marker.header.frame_id = publish_frame
        marker.ns = "blue_object"
        marker.id = 0
        marker.type   = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = X
        marker.pose.position.y = Y
        marker.pose.position.z = Z
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 0.4
        marker.color.b = 1.0
        marker.color.a = 0.85
        marker.lifetime = self._marker_lifetime
        self._marker_pub.publish(marker)

        # ── Annotate debug image ──────────────────────────────────────────────
        cv2.drawContours(debug_img, [largest], -1, (0, 255, 0), 2)
        cv2.circle(debug_img, (u, v), 6, (0, 0, 255), -1)
        cam_label  = f"cam: ({X_cam:+.3f},{Y_cam:+.3f},{Z_cam:.3f})m"
        base_label = f"{publish_frame[:4]}: ({X:+.3f},{Y:+.3f},{Z:.3f})m"
        cv2.putText(debug_img, cam_label,  (u + 10, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
        cv2.putText(debug_img, base_label, (u + 10, v + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        self.get_logger().info(
            f"Blue object @ {publish_frame}: ({X:+.3f}, {Y:+.3f}, {Z:.3f}) m",
            throttle_duration_sec=0.5,
        )
        self._publish_debug(debug_img, color_msg.header)

    def _publish_debug(self, bgr_img: np.ndarray, header):
        msg = self._bridge.cv2_to_imgmsg(bgr_img, encoding="bgr8")
        msg.header = header
        self._debug_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BlueObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
