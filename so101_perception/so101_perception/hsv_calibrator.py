"""Live HSV calibrator for the SO-101 multi-object classifier.

Publishes an annotated debug image as a ROS topic (viewed in rqt_image_view).

How to use
----------
1. Launch this node alongside the camera.
2. View live preview: rqt --standalone rqt_image_view
   Select topic: /hsv_calibrator/debug_image
3. Adjust HSV ranges via ROS parameters (the debug image updates instantly):
       ros2 param set /hsv_calibrator active_class 0
       ros2 param set /hsv_calibrator h_min 170
       ros2 param set /hsv_calibrator h_max 10
       ros2 param set /hsv_calibrator s_min 120
       ros2 param set /hsv_calibrator v_min 70
4. Save / write actions are services (more reliable than param triggers):
       ros2 service call /hsv_calibrator/save_class std_srvs/srv/Trigger
       ros2 service call /hsv_calibrator/reset_class std_srvs/srv/Trigger
       ros2 service call /hsv_calibrator/write_yaml std_srvs/srv/Trigger
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger


DEFAULT_LABELS: List[str] = [
    "red_heart_bear",
    "blue_dragon",
    "purple_cocodrile",
    "banana",
    "green_platypus",
    "object_6",
]


class HSVCalibrator(Node):

    def __init__(self) -> None:
        super().__init__("hsv_calibrator")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("color_image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("class_labels", DEFAULT_LABELS)
        self.declare_parameter("output_path", "")
        self.declare_parameter("min_contour_area", 500)

        # Live trackbar-equivalent parameters.
        self.declare_parameter("active_class", 0)
        self.declare_parameter("h_min", 0)
        self.declare_parameter("h_max", 179)
        self.declare_parameter("s_min", 80)
        self.declare_parameter("s_max", 255)
        self.declare_parameter("v_min", 50)
        self.declare_parameter("v_max", 255)

        color_topic = self.get_parameter("color_image_topic").value
        labels = list(self.get_parameter("class_labels").value)
        if len(labels) != 6:
            labels = (labels + DEFAULT_LABELS)[:6]
        self._labels: List[str] = labels
        self._min_area: int = int(self.get_parameter("min_contour_area").value)

        out_param = self.get_parameter("output_path").value
        self._output_path = out_param if out_param else self._default_output_path()

        # ── State ─────────────────────────────────────────────────────────────
        self._captured: List[Optional[Dict]] = [None] * 6
        self._bridge = CvBridge()
        self._frame_count = 0  # for the periodic heartbeat log

        # ── Subscriptions ─────────────────────────────────────────────────────
        # RealSense in this container publishes RELIABLE; match it.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, color_topic, self._image_cb, sensor_qos)

        # ── Publishers ────────────────────────────────────────────────────────
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._debug_pub = self.create_publisher(Image, "~/debug_image", pub_qos)

        # ── Services (replaces the trigger-parameter mechanism) ───────────────
        self.create_service(Trigger, "~/save_class",  self._srv_save)
        self.create_service(Trigger, "~/reset_class", self._srv_reset)
        self.create_service(Trigger, "~/write_yaml",  self._srv_write)

        self.get_logger().info(
            "HSV Calibrator ready\n"
            f"  colour topic : {color_topic}\n"
            f"  debug image  : /hsv_calibrator/debug_image\n"
            f"  output path  : {self._output_path}\n"
            f"  classes      : {self._labels}\n"
            "\n"
            "  View preview: rqt --standalone rqt_image_view\n"
            "    -> select /hsv_calibrator/debug_image\n"
            "\n"
            "  Set HSV: ros2 param set /hsv_calibrator h_min 100   (etc.)\n"
            "  Switch:  ros2 param set /hsv_calibrator active_class 1\n"
            "  Save:    ros2 service call /hsv_calibrator/save_class std_srvs/srv/Trigger\n"
            "  Reset:   ros2 service call /hsv_calibrator/reset_class std_srvs/srv/Trigger\n"
            "  Write:   ros2 service call /hsv_calibrator/write_yaml std_srvs/srv/Trigger"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _default_output_path(self) -> str:
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory("so101_perception")
            return os.path.join(share, "config", "objects_hsv.yaml")
        except Exception:
            return os.path.abspath("objects_hsv.yaml")

    def _current_bounds(self):
        lo = (int(self.get_parameter("h_min").value),
              int(self.get_parameter("s_min").value),
              int(self.get_parameter("v_min").value))
        hi = (int(self.get_parameter("h_max").value),
              int(self.get_parameter("s_max").value),
              int(self.get_parameter("v_max").value))
        return lo, hi

    def _active_class_id(self) -> int:
        return max(0, min(5, int(self.get_parameter("active_class").value)))

    # ── Services ──────────────────────────────────────────────────────────────

    def _srv_save(self, req, resp):
        cid = self._active_class_id()
        lo, hi = self._current_bounds()
        self._captured[cid] = {
            "label":     self._labels[cid],
            "hsv_lower": [lo[0], lo[1], lo[2]],
            "hsv_upper": [hi[0], hi[1], hi[2]],
            "min_contour_area": int(self._min_area),
        }
        msg = f"Saved class {cid} ({self._labels[cid]}): lower={list(lo)} upper={list(hi)}"
        self.get_logger().info(msg)
        resp.success = True
        resp.message = msg
        return resp

    def _srv_reset(self, req, resp):
        cid = self._active_class_id()
        self._captured[cid] = None
        msg = f"Reset class {cid} ({self._labels[cid]})."
        self.get_logger().info(msg)
        resp.success = True
        resp.message = msg
        return resp

    def _srv_write(self, req, resp):
        classes_out = []
        for i, entry in enumerate(self._captured):
            if entry is None:
                classes_out.append({
                    "label":     self._labels[i],
                    "enabled":   False,
                    "hsv_lower": [0, 0, 0],
                    "hsv_upper": [179, 255, 255],
                    "min_contour_area": int(self._min_area),
                })
            else:
                classes_out.append({**entry, "enabled": True})

        payload = {
            "object_classifier": {
                "ros__parameters": {
                    "calibrated_at": datetime.now().isoformat(timespec="seconds"),
                    "classes": classes_out,
                }
            }
        }

        try:
            os.makedirs(os.path.dirname(self._output_path), exist_ok=True)
        except OSError:
            pass

        try:
            with open(self._output_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=None)
        except OSError as e:
            err = f"Failed to write {self._output_path}: {e}"
            self.get_logger().error(err)
            resp.success = False
            resp.message = err
            return resp

        n_ok = sum(1 for c in self._captured if c is not None)
        msg = f"Wrote {n_ok}/6 calibrated classes to {self._output_path}"
        self.get_logger().info(msg)
        resp.success = True
        resp.message = msg
        return resp

    # ── Image processing ──────────────────────────────────────────────────────

    def _compute_mask(self, hsv: np.ndarray, lo, hi) -> np.ndarray:
        h_lo, s_lo, v_lo = lo
        h_hi, s_hi, v_hi = hi
        if h_lo <= h_hi:
            mask = cv2.inRange(hsv,
                               np.array([h_lo, s_lo, v_lo], dtype=np.uint8),
                               np.array([h_hi, s_hi, v_hi], dtype=np.uint8))
        else:
            m1 = cv2.inRange(hsv,
                             np.array([0,    s_lo, v_lo], dtype=np.uint8),
                             np.array([h_hi, s_hi, v_hi], dtype=np.uint8))
            m2 = cv2.inRange(hsv,
                             np.array([h_lo, s_lo, v_lo], dtype=np.uint8),
                             np.array([179,  s_hi, v_hi], dtype=np.uint8))
            mask = cv2.bitwise_or(m1, m2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _annotate(self, bgr: np.ndarray, mask: np.ndarray,
                  cid: int, lo, hi) -> np.ndarray:
        out = bgr.copy()
        h, w = out.shape[:2]

        # Dim non-mask regions (fast OpenCV indexing).
        inv = cv2.bitwise_not(mask)
        out[inv > 0] = (out[inv > 0] * 0.35).astype(np.uint8)

        # Outline the largest contour.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) >= self._min_area:
                cv2.drawContours(out, [c], -1, (0, 255, 0), 2)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(out, (cx, cy), 6, (0, 255, 255), -1)

        # Status bar at the top.
        cv2.rectangle(out, (0, 0), (w, 70), (0, 0, 0), thickness=-1)
        cv2.putText(out, f"Class {cid}: {self._labels[cid]}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(
            out,
            f"HSV: [{lo[0]},{lo[1]},{lo[2]}] -> [{hi[0]},{hi[1]},{hi[2]}]",
            (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
        )

        # Capture status strip at the bottom.
        cv2.rectangle(out, (0, h - 30), (w, h), (0, 0, 0), thickness=-1)
        for i in range(6):
            colour = (0, 200, 0) if self._captured[i] is not None else (90, 90, 90)
            tag = "OK" if self._captured[i] is not None else "--"
            if i == cid:
                cv2.rectangle(out, (10 + i * 90, h - 25),
                              (95 + i * 90, h - 5), (0, 255, 255), 1)
            cv2.putText(out, f"{i}:{tag}",
                        (15 + i * 90, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        return out

    # ── ROS callback ──────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image) -> None:
        # Always log on the first frame and then every 60 frames, so we can
        # see in the terminal whether callbacks are firing.
        if self._frame_count == 0:
            self.get_logger().info(
                f"First frame received: {msg.width}x{msg.height} encoding={msg.encoding}"
            )
        self._frame_count += 1
        if self._frame_count % 60 == 0:
            self.get_logger().info(f"Processed {self._frame_count} frames")

        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        try:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            cid = self._active_class_id()
            lo, hi = self._current_bounds()
            mask = self._compute_mask(hsv, lo, hi)
            annotated = self._annotate(bgr, mask, cid, lo, hi)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Image processing failed: {e}")
            return

        try:
            out_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out_msg.header = msg.header
            self._debug_pub.publish(out_msg)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Failed to publish debug image: {e}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HSVCalibrator()
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