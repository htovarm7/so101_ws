#!/usr/bin/env python3
"""Live camera-pose localiser using an ArUco fiducial as world anchor.

Use case
--------
The SO-101 has a wrist-mounted RealSense.  The camera-to-jaw mount is
imperfectly calibrated (rotation + translation error) and the joint
servos have backlash, so the URDF/TF chain that links the camera to
``base_link`` is unreliable.  An ArUco marker placed at a known position
on the workbench gives us a *visually-anchored* ground truth.

Math
----
Each frame we detect the marker we get ``T_cam_to_marker`` from
``cv2.solvePnP`` of the marker's 4 corners.  The marker is fixed at
``T_base_to_marker`` (measured once with a ruler).  So:

    T_base_to_cam = T_base_to_marker @ inv(T_cam_to_marker)

This is the camera's pose in ``base_link`` based purely on what the
camera sees — independent of FK, joint backlash, or mount calibration.
Downstream nodes (``object_classifier``) can multiply any deprojected
3-D point ``(X, Y, Z)`` in camera frame by this matrix and get the
correct position in ``base_link``.

Topics
------
Subscriptions:
  /camera/cam_static/color/image_raw   (sensor_msgs/Image)
  /camera/cam_static/color/camera_info (sensor_msgs/CameraInfo)

Publications:
  /camera_pose_in_base  (geometry_msgs/PoseStamped) — when marker detected
  /aruco_seen           (std_msgs/Bool)             — every frame
  /aruco/debug_image    (sensor_msgs/Image)         — with corners drawn
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import cv2.aruco as aruco
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = qx * qx + qy * qy + qz * qz + qw * qw
    s = 0.0 if n < 1e-12 else 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def rotmat_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0:
        s = 2.0 * (1.0 + tr) ** 0.5
        return (
            float((R[2, 1] - R[1, 2]) / s),
            float((R[0, 2] - R[2, 0]) / s),
            float((R[1, 0] - R[0, 1]) / s),
            float(0.25 * s),
        )
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5
        return (
            float(0.25 * s),
            float((R[0, 1] + R[1, 0]) / s),
            float((R[0, 2] + R[2, 0]) / s),
            float((R[2, 1] - R[1, 2]) / s),
        )
    if R[1, 1] > R[2, 2]:
        s = 2.0 * (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5
        return (
            float((R[0, 1] + R[1, 0]) / s),
            float(0.25 * s),
            float((R[1, 2] + R[2, 1]) / s),
            float((R[0, 2] - R[2, 0]) / s),
        )
    s = 2.0 * (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5
    return (
        float((R[0, 2] + R[2, 0]) / s),
        float((R[1, 2] + R[2, 1]) / s),
        float(0.25 * s),
        float((R[1, 0] - R[0, 1]) / s),
    )


class ArucoLocalizer(Node):
    def __init__(self):
        super().__init__("aruco_localizer")

        # ── Parameters ──────────────────────────────────────────────────
        d = self.declare_parameter
        d("color_image_topic", "/camera/cam_static/color/image_raw")
        d("camera_info_topic", "/camera/cam_static/color/camera_info")
        d("output_pose_topic", "/camera_pose_in_base")
        d("output_seen_topic", "/aruco_seen")
        d("debug_image_topic", "/aruco/debug_image")

        # Marker geometry & dictionary
        d("marker_id", 675)
        d("marker_size_m", 0.07)
        d("marker_dict", "DICT_ARUCO_ORIGINAL")

        # KNOWN position of the marker centre in base_link, measured ONCE
        # with a ruler.  Orientation: by default the marker lies FLAT on
        # the table, face-up (Z points up).  Override if you mount it at
        # an angle.
        d("marker_position_in_base", [0.56, 0.075, 0.035])
        d("marker_quat_in_base_xyzw", [0.0, 0.0, 0.0, 1.0])

        # Detection quality
        d("min_detection_score", 0.0)   # placeholder; cv2.aruco doesn't expose one

        g = self.get_parameter
        color_topic = g("color_image_topic").value
        info_topic = g("camera_info_topic").value
        pose_topic = g("output_pose_topic").value
        seen_topic = g("output_seen_topic").value
        dbg_topic = g("debug_image_topic").value

        self._marker_id = int(g("marker_id").value)
        self._marker_size = float(g("marker_size_m").value)
        dict_name = str(g("marker_dict").value)
        dict_attr = getattr(aruco, dict_name, None)
        if dict_attr is None:
            raise RuntimeError(f"Unknown ArUco dictionary: {dict_name}")
        self._aruco_dict = aruco.getPredefinedDictionary(dict_attr)
        # OpenCV ≤ 4.6: `DetectorParameters()` exists as a symbol but
        # SEGFAULTS when called (it's only a constructor from 4.7 on).
        # The try/except AttributeError doesn't catch native crashes, so
        # we have to gate on the version explicitly.
        cv_major, cv_minor = (int(x) for x in cv2.__version__.split(".")[:2])
        if (cv_major, cv_minor) <= (4, 6):
            self._detector_params = aruco.DetectorParameters_create()
        else:
            self._detector_params = aruco.DetectorParameters()

        # Marker pose in base_link (T_base_to_marker, 4x4)
        m_pos = np.array(g("marker_position_in_base").value, dtype=float)
        m_q = g("marker_quat_in_base_xyzw").value
        self._T_b2m = np.eye(4)
        self._T_b2m[:3, :3] = quat_to_rotmat(*m_q)
        self._T_b2m[:3, 3] = m_pos
        self.get_logger().info(
            f"Marker {self._marker_id} ({self._marker_size*1000:.0f} mm "
            f"{dict_name}) anchored at {tuple(m_pos)} in base_link"
        )

        # Camera intrinsics — populated by camera_info callback
        self._K: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._bridge = CvBridge()

        # 3-D marker corners in marker frame (counter-clockwise, marker
        # plane = XY, +Z normal to face).  solvePnP expects this order
        # to match how detectMarkers returns the 4 image corners.
        s = self._marker_size / 2.0
        self._marker_corners_3d = np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0],
        ], dtype=np.float32)

        # ── I/O ────────────────────────────────────────────────────────
        # BEST_EFFORT for the image stream — matches the sensor-data
        # QoS convention and connects to either RELIABLE or BEST_EFFORT
        # publishers (RELIABLE sub + BEST_EFFORT pub is incompatible
        # the other way around).  CameraInfo stays RELIABLE because it
        # latches once and we never want to miss it.
        info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        self.create_subscription(CameraInfo, info_topic, self._on_info, info_qos)
        self.create_subscription(Image, color_topic, self._on_image, image_qos)
        self._first_image_logged = False

        self._pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self._seen_pub = self.create_publisher(Bool, seen_topic, 10)
        self._dbg_pub = self.create_publisher(Image, dbg_topic, 1)

        self.get_logger().info(
            f"Subscribed to {color_topic}, publishing to {pose_topic}."
        )

    # ── Callbacks ───────────────────────────────────────────────────────

    def _on_info(self, msg: CameraInfo):
        if self._K is None:
            self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            # Realsense can publish an empty distortion array when its
            # in-driver rectification is on — but estimatePoseSingleMarkers
            # / solvePnP segfault on an empty D.  Substitute zeros so the
            # solver runs.
            if len(msg.d) == 0:
                self._D = np.zeros((5, 1), dtype=np.float64)
            else:
                self._D = np.array(msg.d, dtype=np.float64).reshape(-1, 1)
            self.get_logger().info(
                f"Got camera intrinsics: fx={self._K[0,0]:.1f} "
                f"fy={self._K[1,1]:.1f} cx={self._K[0,2]:.1f} cy={self._K[1,2]:.1f} "
                f"|D|={len(self._D)}"
            )

    def _on_image(self, msg: Image):
        if not self._first_image_logged:
            self._first_image_logged = True
            self.get_logger().info(
                f"First image received ({msg.width}x{msg.height} enc={msg.encoding})"
            )
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}", throttle_duration_sec=2.0)
            return

        # Step 1 — publish the raw image immediately, BEFORE any aruco
        # call.  This way the debug stream stays alive even if detection
        # segfaults later in the pipeline.
        self._publish_debug(bgr, None, None, None)

        if self._K is None:
            return
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Use the legacy aruco API exclusively — ArucoDetector + SOLVEPNP_
        # IPPE_SQUARE crash with a SIGSEGV on the OpenCV 4.6 that ships in
        # Jazzy.  estimatePoseSingleMarkers is deprecated in 4.7+ but still
        # works and is the most portable across versions.
        try:
            corners, ids, _ = aruco.detectMarkers(
                gray, self._aruco_dict, parameters=self._detector_params)
        except Exception as e:
            self.get_logger().warn(
                f"detectMarkers failed: {e}", throttle_duration_sec=2.0)
            # Even on failure, publish the raw frame so RViz/rqt can see
            # the live feed and the user can confirm the camera is alive.
            self._publish_debug(bgr, None, None, None)
            return

        seen = False
        chosen_idx: Optional[int] = None
        if ids is not None and len(ids) > 0:
            ids_flat = ids.flatten()
            matches = np.where(ids_flat == self._marker_id)[0]
            if matches.size > 0:
                chosen_idx = int(matches[0])
                seen = True

        # Publish "seen" flag every frame so subscribers can detect drop-out.
        seen_msg = Bool()
        seen_msg.data = seen
        self._seen_pub.publish(seen_msg)

        if not seen or chosen_idx is None:
            self._publish_debug(bgr, corners, ids, None)
            return

        try:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                [corners[chosen_idx]],
                self._marker_size,
                self._K, self._D,
            )
        except Exception as e:
            self.get_logger().warn(
                f"estimatePoseSingleMarkers failed: {e}", throttle_duration_sec=2.0)
            return
        rvec = rvecs[0]
        tvec = tvecs[0]

        R_cam_to_marker, _ = cv2.Rodrigues(rvec)
        t_cam_to_marker = tvec.flatten()

        # Compose: T_base_to_cam = T_base_to_marker @ inv(T_cam_to_marker)
        T_c2m = np.eye(4)
        T_c2m[:3, :3] = R_cam_to_marker
        T_c2m[:3, 3] = t_cam_to_marker
        T_m2c = np.linalg.inv(T_c2m)
        T_b2c = self._T_b2m @ T_m2c

        # Publish camera pose in base_link
        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = "base_link"
        pose.pose.position.x = float(T_b2c[0, 3])
        pose.pose.position.y = float(T_b2c[1, 3])
        pose.pose.position.z = float(T_b2c[2, 3])
        qx, qy, qz, qw = rotmat_to_quat(T_b2c[:3, :3])
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        self._pose_pub.publish(pose)

        self._publish_debug(bgr, corners, ids, (rvec, tvec))

    def _publish_debug(self, bgr, corners, ids, pose_opt):
        out = bgr.copy()
        if ids is not None and corners is not None:
            aruco.drawDetectedMarkers(out, corners, ids)
        if pose_opt is not None and self._K is not None:
            rvec, tvec = pose_opt
            cv2.drawFrameAxes(
                out, self._K, self._D, rvec, tvec,
                self._marker_size * 0.5,
            )
        try:
            msg = self._bridge.cv2_to_imgmsg(out, encoding="bgr8")
            self._dbg_pub.publish(msg)
        except Exception:
            pass


def main():
    rclpy.init()
    node = ArucoLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
