"""Multi-class HSV object classifier for the SO-101 robot arm.

This is the run-time counterpart of ``hsv_calibrator``.  It loads a list of
HSV-defined classes from parameters  then on every synchronised
colour+depth frame it finds the largest contour matching any enabled class, estimates its
3-D position, and publishes the results as a label, a TF point, and a visualization marker.
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.duration import Duration as RclDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge

import tf2_ros
import tf2_geometry_msgs 

from geometry_msgs.msg import PointStamped, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration

from message_filters import ApproximateTimeSynchronizer, Subscriber


# Distinct BGR colours for the debug overlay
DEBUG_COLOURS: List[Tuple[int, int, int]] = [
    (0,   0, 255),   # red
    (255, 0,   0),   # blue
    (0, 200, 255),   # amber
    (0, 200,   0),   # green
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
]


class _ClassSpec:
    """Compact in-memory view of one entry in the `classes` parameter list."""

    __slots__ = ("label", "enabled", "lower", "upper", "min_area", "wrap")

    def __init__(self, raw: Dict) -> None:
        self.label: str = str(raw.get("label", "unknown"))
        self.enabled: bool = bool(raw.get("enabled", True))
        self.lower = np.array(raw.get("hsv_lower", [0, 0, 0]),       dtype=np.uint8)
        self.upper = np.array(raw.get("hsv_upper", [179, 255, 255]), dtype=np.uint8)
        self.min_area: int = int(raw.get("min_contour_area", 500))
        self.wrap: bool = bool(self.lower[0] > self.upper[0])


class ObjectClassifier(Node):

    def __init__(self) -> None:
        super().__init__("object_classifier")

        self.declare_parameter("color_image_topic",
                               "/camera/camera/color/image_raw")
        self.declare_parameter("depth_image_topic",
                               "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic",
                               "/camera/camera/color/camera_info")

        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("parent_link",  "moving_jaw_so101_v1_link")

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
        self.declare_parameter("none_label", "none")

        self.declare_parameter("config_file", "")
        cfg_path = self.get_parameter("config_file").value or self._default_config_path()
        classes_raw = self._load_classes_from_yaml(cfg_path)
        if not classes_raw:
            self.get_logger().warn(
                f"No classes loaded from {cfg_path!r} — run hsv_calibrator first "
                "or pass config_file:=<path>. Detector will publish 'none'."
            )
        self._classes: List[_ClassSpec] = [_ClassSpec(c) for c in classes_raw]
        self._config_path = cfg_path

        color_topic = self.get_parameter("color_image_topic").value
        depth_topic = self.get_parameter("depth_image_topic").value
        info_topic  = self.get_parameter("camera_info_topic").value
        self._target_frame = self.get_parameter("target_frame").value
        self._parent_link  = self.get_parameter("parent_link").value
        self._mount = {k: self.get_parameter(f"mount_{k}").value
                       for k in ("x", "y", "z", "qx", "qy", "qz", "qw")}
        self._depth_scale = float(self.get_parameter("depth_scale").value)
        self._max_depth   = float(self.get_parameter("max_depth_m").value)
        self._none_label  = str(self.get_parameter("none_label").value)

        marker_lifetime = float(self.get_parameter("marker_lifetime_s").value)
        self._marker_lifetime = Duration(
            sec=int(marker_lifetime),
            nanosec=int((marker_lifetime % 1.0) * 1e9),
        )
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
        self._info_sub = self.create_subscription(
            CameraInfo, info_topic, self._camera_info_cb, sensor_qos
        )
        self._color_sub = Subscriber(self, Image, color_topic, qos_profile=sensor_qos)
        self._depth_sub = Subscriber(self, Image, depth_topic, qos_profile=sensor_qos)
        self._sync = ApproximateTimeSynchronizer(
            [self._color_sub, self._depth_sub], queue_size=5, slop=0.05
        )
        self._sync.registerCallback(self._image_cb)

        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._label_pub  = self.create_publisher(String,       "~/detected_label", 10)
        self._point_pub  = self.create_publisher(PointStamped, "~/detected_point", 10)
        self._marker_pub = self.create_publisher(Marker,       "~/marker",         10)
        self._debug_pub  = self.create_publisher(Image,        "~/debug_image",    pub_qos)

        enabled = [c.label for c in self._classes if c.enabled]
        self.get_logger().info(
            "ObjectClassifier ready\n"
            f"  colour       : {color_topic}\n"
            f"  depth        : {depth_topic}\n"
            f"  parent_link  : {self._parent_link}\n"
            f"  target_frame : {self._target_frame}\n"
            f"  config       : {self._config_path}\n"
            f"  classes      : {enabled if enabled else '(none — publishing none_label)'}"
        )

    # Config loading
    def _default_config_path(self) -> str:
        """Resolve the install-share config dir for so101_perception."""
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory("so101_perception")
            return os.path.join(share, "config", "objects_hsv.yaml")
        except Exception:
            return os.path.abspath("objects_hsv.yaml")

    def _load_classes_from_yaml(self, path: str) -> List[Dict]:
        """Load the `classes` list from a YAML file.
        """
        if not path or not os.path.isfile(path):
            self.get_logger().warn(f"Config file not found: {path!r}")
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            self.get_logger().error(f"Failed to read {path}: {e}")
            return []

        # Tolerate both the ROS-style nested format and a flat list.
        if isinstance(data, dict):
            # Look one or two levels deep for a `classes` key.
            for top_val in data.values():
                if isinstance(top_val, dict):
                    if "classes" in top_val:
                        return list(top_val["classes"] or [])
                    inner = top_val.get("ros__parameters")
                    if isinstance(inner, dict) and "classes" in inner:
                        return list(inner["classes"] or [])
            if "classes" in data:
                return list(data["classes"] or [])
        elif isinstance(data, list):
            return data

        self.get_logger().error(
            f"Unrecognised YAML layout in {path}: expected a `classes:` list "
            "somewhere in the tree."
        )
        return []

    # TF bridge

    def _publish_camera_tf(self) -> None:
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
        self.get_logger().info(
            f"Static TF broadcast: {self._parent_link} -> {self._camera_frame}"
        )

    # Camera intrinsics

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

    # Main pipeline
    def _detect_best(self, hsv: np.ndarray):
        """Run every enabled class and return (winner_index, contour, area, centroid).
        """
        best_idx = -1
        best_area = 0.0
        best_contour = None
        best_centroid: Optional[Tuple[int, int]] = None

        for i, spec in enumerate(self._classes):
            if not spec.enabled:
                continue
            mask = self._mask_for(hsv, spec)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area < spec.min_area:
                continue
            if area > best_area:
                M = cv2.moments(c)
                if M["m00"] <= 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best_idx     = i
                best_area    = area
                best_contour = c
                best_centroid = (cx, cy)
        return best_idx, best_contour, best_area, best_centroid

    @staticmethod
    def _mask_for(hsv: np.ndarray, spec: _ClassSpec) -> np.ndarray:
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
        idx, contour, area, centroid = self._detect_best(hsv)
        label = self._classes[idx].label if idx >= 0 else self._none_label
        self._label_pub.publish(String(data=label))

        debug = bgr.copy()

        if idx >= 0 and centroid is not None:
            cx_px, cy_px = centroid
            colour = DEBUG_COLOURS[idx % len(DEBUG_COLOURS)]
            cv2.drawContours(debug, [contour], -1, colour, 2)
            cv2.circle(debug, (cx_px, cy_px), 6, colour, -1)
            cv2.putText(debug, f"{label} (A={int(area)})",
                        (cx_px + 8, cy_px - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

            # Depth sample at the centroid
            z_m = self._sample_depth(depth, cx_px, cy_px)
            if z_m is not None and 0.0 < z_m < self._max_depth:
                X = (cx_px - self._cx) * z_m / self._fx
                Y = (cy_px - self._cy) * z_m / self._fy
                Z = z_m
                self._publish_point(color_msg.header.stamp, X, Y, Z, label)
        else:
            cv2.putText(debug, "none", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Always emit the debug image so RViz/rqt previews stay live.
        try:
            dbg_msg = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            dbg_msg.header = color_msg.header
            self._debug_pub.publish(dbg_msg)
        except Exception as e: 
            self.get_logger().warn(f"Failed to publish debug image: {e}")

    def _sample_depth(self, depth: np.ndarray, x: int, y: int) -> Optional[float]:
        """Median-filtered depth (in metres) in a 5 5 window around (x, y)."""
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

    def _publish_point(self, stamp, X: float, Y: float, Z: float, label: str) -> None:
        cam_pt = PointStamped()
        cam_pt.header.stamp = stamp
        cam_pt.header.frame_id = self._camera_frame
        cam_pt.point.x = float(X)
        cam_pt.point.y = float(Y)
        cam_pt.point.z = float(Z)

        try:
            tgt_pt = self._tf_buffer.transform(
                cam_pt,
                self._target_frame,
                timeout=RclDuration(seconds=0.1),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as e:
            self.get_logger().warn(
                f"TF {self._camera_frame} -> {self._target_frame} failed: {e}"
            )
            return

        self._point_pub.publish(tgt_pt)

        m = Marker()
        m.header = tgt_pt.header
        m.ns = "object_classifier"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = tgt_pt.point.x
        m.pose.position.y = tgt_pt.point.y
        m.pose.position.z = tgt_pt.point.z
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.04
        idx = next((i for i, s in enumerate(self._classes) if s.label == label), 0)
        b, g, r = DEBUG_COLOURS[idx % len(DEBUG_COLOURS)]
        m.color.r = r / 255.0
        m.color.g = g / 255.0
        m.color.b = b / 255.0
        m.color.a = 0.9
        m.lifetime = self._marker_lifetime
        self._marker_pub.publish(m)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectClassifier()
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
