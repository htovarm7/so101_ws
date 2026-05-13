"""Blue-object detector using RealSense D435 colour + aligned depth streams.

Pipeline
--------
1. Receive aligned colour + depth frames (same resolution, pixel-aligned).
2. Convert colour to HSV and threshold for blue (configurable).
3. Find the largest contour above a minimum area; compute its centroid.
4. Look up depth at the centroid pixel; convert to 3-D using the pinhole model.
5. Publish:
   - geometry_msgs/PointStamped   → ~/detected_point
   - visualization_msgs/Marker    → ~/marker        (sphere in camera frame)
   - sensor_msgs/Image            → ~/debug_image   (annotated colour frame)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
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

        # HSV thresholds for blue  (OpenCV: H∈[0,179], S/V∈[0,255])
        self.declare_parameter("hsv_lower", [100, 80, 50])
        self.declare_parameter("hsv_upper", [130, 255, 255])

        # Minimum contour area in pixels to avoid noise detections
        self.declare_parameter("min_contour_area", 500)

        # Marker lifetime in seconds (0 = forever until next update)
        self.declare_parameter("marker_lifetime_s", 0.2)

        # Depth value scale: RealSense returns millimetres → convert to metres
        self.declare_parameter("depth_scale", 0.001)

        # Maximum valid depth (metres) – discard readings beyond this
        self.declare_parameter("max_depth_m", 3.0)

        color_topic = self.get_parameter("color_image_topic").value
        depth_topic = self.get_parameter("depth_image_topic").value
        info_topic  = self.get_parameter("camera_info_topic").value
        self.hsv_lower = np.array(self.get_parameter("hsv_lower").value, dtype=np.uint8)
        self.hsv_upper = np.array(self.get_parameter("hsv_upper").value, dtype=np.uint8)
        self.min_area  = self.get_parameter("min_contour_area").value
        self.depth_scale = self.get_parameter("depth_scale").value
        self.max_depth   = self.get_parameter("max_depth_m").value
        marker_lifetime  = self.get_parameter("marker_lifetime_s").value

        self._marker_lifetime = Duration(
            sec=int(marker_lifetime),
            nanosec=int((marker_lifetime % 1) * 1e9),
        )

        # Camera intrinsics (filled on first CameraInfo message)
        self._fx = self._fy = self._cx = self._cy = None
        self._camera_frame: str = "camera_color_optical_frame"

        self._bridge = CvBridge()

        # ── Subscriptions ──────────────────────────────────────────────────────
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
            [self._color_sub, self._depth_sub],
            queue_size=5,
            slop=0.05,
        )
        self._sync.registerCallback(self._image_cb)

        # ── Publishers ─────────────────────────────────────────────────────────
        self._point_pub  = self.create_publisher(PointStamped, "~/detected_point", 10)
        self._marker_pub = self.create_publisher(Marker, "~/marker", 10)
        self._debug_pub  = self.create_publisher(Image, "~/debug_image", sensor_qos)

        self.get_logger().info(
            f"BlueObjectDetector ready\n"
            f"  colour : {color_topic}\n"
            f"  depth  : {depth_topic}\n"
            f"  HSV    : lower={self.hsv_lower.tolist()}  upper={self.hsv_upper.tolist()}"
        )

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo):
        if self._fx is not None:
            return  # already initialised
        self._fx = msg.k[0]
        self._fy = msg.k[4]
        self._cx = msg.k[2]
        self._cy = msg.k[5]
        self._camera_frame = msg.header.frame_id
        self.get_logger().info(
            f"Camera intrinsics loaded — fx={self._fx:.1f} fy={self._fy:.1f} "
            f"cx={self._cx:.1f} cy={self._cy:.1f}  frame='{self._camera_frame}'"
        )
        # Intrinsics received; camera_info subscriber is no longer needed
        self.destroy_subscription(self._info_sub)

    def _image_cb(self, color_msg: Image, depth_msg: Image):
        if self._fx is None:
            self.get_logger().warn("Waiting for camera_info…", throttle_duration_sec=3.0)
            return

        # ── Decode ─────────────────────────────────────────────────────────────
        color_bgr = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        # depth is uint16, values in millimetres
        depth_raw = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        # ── Blue mask ──────────────────────────────────────────────────────────
        hsv   = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask  = cv2.morphologyEx(mask, cv2.MORPH_DILATE,
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

        # ── Centroid ───────────────────────────────────────────────────────────
        M   = cv2.moments(largest)
        u   = int(M["m10"] / M["m00"])
        v   = int(M["m01"] / M["m00"])

        # ── Depth lookup (3×3 neighbourhood median for robustness) ────────────
        h, w = depth_raw.shape[:2]
        u_lo, u_hi = max(0, u - 1), min(w, u + 2)
        v_lo, v_hi = max(0, v - 1), min(h, v + 2)
        patch = depth_raw[v_lo:v_hi, u_lo:u_hi].astype(np.float32)
        valid = patch[patch > 0]
        if valid.size == 0:
            self.get_logger().warn("Depth is zero at centroid — skipping frame.",
                                   throttle_duration_sec=1.0)
            self._publish_debug(debug_img, color_msg.header)
            return

        depth_m = float(np.median(valid)) * self.depth_scale
        if depth_m > self.max_depth or depth_m <= 0.0:
            self.get_logger().warn(f"Depth {depth_m:.3f} m out of range — skipping.",
                                   throttle_duration_sec=1.0)
            self._publish_debug(debug_img, color_msg.header)
            return

        # ── Back-project to 3-D (pinhole model) ───────────────────────────────
        X = (u - self._cx) * depth_m / self._fx
        Y = (v - self._cy) * depth_m / self._fy
        Z = depth_m

        # ── Publish PointStamped ───────────────────────────────────────────────
        pt_msg = PointStamped()
        pt_msg.header.stamp    = color_msg.header.stamp
        pt_msg.header.frame_id = self._camera_frame
        pt_msg.point.x = X
        pt_msg.point.y = Y
        pt_msg.point.z = Z
        self._point_pub.publish(pt_msg)

        # ── Publish sphere Marker ──────────────────────────────────────────────
        marker = Marker()
        marker.header.stamp    = color_msg.header.stamp
        marker.header.frame_id = self._camera_frame
        marker.ns              = "blue_object"
        marker.id              = 0
        marker.type            = Marker.SPHERE
        marker.action          = Marker.ADD
        marker.pose.position.x = X
        marker.pose.position.y = Y
        marker.pose.position.z = Z
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.05  # 5 cm sphere
        marker.color.r = 0.0
        marker.color.g = 0.4
        marker.color.b = 1.0
        marker.color.a = 0.85
        marker.lifetime = self._marker_lifetime
        self._marker_pub.publish(marker)

        # ── Annotate debug image ───────────────────────────────────────────────
        cv2.drawContours(debug_img, [largest], -1, (0, 255, 0), 2)
        cv2.circle(debug_img, (u, v), 6, (0, 0, 255), -1)
        label = f"({X:+.3f}, {Y:+.3f}, {Z:.3f}) m"
        cv2.putText(debug_img, label, (u + 10, v),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        self.get_logger().info(f"Blue object @ {label}", throttle_duration_sec=0.5)

        self._publish_debug(debug_img, color_msg.header)

    # ── Helpers ────────────────────────────────────────────────────────────────

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
