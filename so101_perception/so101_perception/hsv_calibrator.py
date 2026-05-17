"""Live HSV calibrator for the SO-101 multi-object classifier.

This version uses the same pattern as the working blue_object_detector:
it publishes an annotated debug image as a ROS topic instead of opening
OpenCV GUI windows.  This is needed because cv2.imshow does not render
inside this Docker container (OpenCV was built without GUI backend).

How to use
----------
1. Launch this node alongside the camera:
       ros2 launch so101_perception hsv_calibration.launch.py

2. View the live annotated feed in rqt_image_view:
       rqt --standalone rqt_image_view
   In the dropdown, pick:  /hsv_calibrator/debug_image

3. Adjust HSV ranges live via ROS parameters.  Examples:
       # Switch to a different class slot (0-5)
       ros2 param set /hsv_calibrator active_class 1

       # Set the HSV bounds for the currently active class
       ros2 param set /hsv_calibrator h_min 100
       ros2 param set /hsv_calibrator h_max 130
       ros2 param set /hsv_calibrator s_min 80
       ros2 param set /hsv_calibrator s_max 255
       ros2 param set /hsv_calibrator v_min 50
       ros2 param set /hsv_calibrator v_max 255

   The debug image updates instantly each time you set a parameter.

4. Capture the current trackbar values into the active slot:
       ros2 param set /hsv_calibrator save_now true
   (The parameter auto-resets to False after saving.)

5. Write the YAML file with all captured classes:
       ros2 param set /hsv_calibrator write_yaml true

6. Other handy parameter actions:
       ros2 param set /hsv_calibrator reset_active true   # clear current slot
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
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


DEFAULT_LABELS: List[str] = [
    "red_heart_bear",
    "blue_dragon",
    "cereal_box",
    "object_4",
    "object_5",
    "object_6",
]


class HSVCalibrator(Node):

    def __init__(self) -> None:
        super().__init__("hsv_calibrator")

        # ── Topic / file parameters ───────────────────────────────────────────
        self.declare_parameter(
            "color_image_topic", "/camera/camera/color/image_raw"
        )
        self.declare_parameter("class_labels", DEFAULT_LABELS)
        self.declare_parameter("output_path", "")
        self.declare_parameter("min_contour_area", 500)

        # ── Live trackbar-equivalent parameters ───────────────────────────────
        # active_class: which slot (0-5) the H/S/V values target right now.
        # h_min .. v_max: the HSV bounds you'd be adjusting on trackbars.
        self.declare_parameter("active_class", 0)
        self.declare_parameter("h_min", 0)
        self.declare_parameter("h_max", 179)
        self.declare_parameter("s_min", 80)
        self.declare_parameter("s_max", 255)
        self.declare_parameter("v_min", 50)
        self.declare_parameter("v_max", 255)

        # ── Action triggers (parameters used as buttons) ──────────────────────
        # Setting any of these to True triggers the action and the node
        # immediately resets the value back to False.
        self.declare_parameter("save_now",     False)
        self.declare_parameter("reset_active", False)
        self.declare_parameter("write_yaml",   False)

        # Cache values we care about.
        color_topic: str = self.get_parameter("color_image_topic").value
        labels = list(self.get_parameter("class_labels").value)
        if len(labels) != 6:
            self.get_logger().warn(
                f"class_labels has {len(labels)} entries; padding/truncating to 6."
            )
            labels = (labels + DEFAULT_LABELS)[:6]
        self._labels: List[str] = labels
        self._min_area: int = int(self.get_parameter("min_contour_area").value)

        out_param: str = self.get_parameter("output_path").value
        self._output_path: str = out_param if out_param else self._default_output_path()

        # ── State ─────────────────────────────────────────────────────────────
        self._captured: List[Optional[Dict]] = [None] * 6
        self._bridge = CvBridge()

        # React to parameter changes — this is what makes "buttons" work and
        # what makes the HSV preview update live without restarting the node.
        self.add_on_set_parameters_callback(self._on_param_change)

        # ── Subscriptions ─────────────────────────────────────────────────────
        # The RealSense driver in this container publishes RELIABLE; match it.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, color_topic, self._image_cb, sensor_qos)

        # ── Publishers ────────────────────────────────────────────────────────
        # Debug image: same pattern as blue_object_detector's debug_image.
        self._debug_pub = self.create_publisher(
            Image, "~/debug_image", sensor_qos
        )

        self.get_logger().info(
            "HSV Calibrator ready\n"
            f"  colour topic : {color_topic}\n"
            f"  debug image  : /hsv_calibrator/debug_image\n"
            f"  output path  : {self._output_path}\n"
            f"  classes      : {self._labels}\n"
            "\n"
            "  View live preview:\n"
            "    rqt --standalone rqt_image_view\n"
            "    (select /hsv_calibrator/debug_image)\n"
            "\n"
            "  Adjust HSV bounds:\n"
            "    ros2 param set /hsv_calibrator h_min 100\n"
            "    ros2 param set /hsv_calibrator h_max 130   (etc.)\n"
            "\n"
            "  Switch class slot:\n"
            "    ros2 param set /hsv_calibrator active_class 1\n"
            "\n"
            "  Capture current bounds into the active slot:\n"
            "    ros2 param set /hsv_calibrator save_now true\n"
            "\n"
            "  Write all captured classes to YAML:\n"
            "    ros2 param set /hsv_calibrator write_yaml true"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _default_output_path(self) -> str:
        """Resolve the install-share config dir for so101_perception."""
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
        cid = int(self.get_parameter("active_class").value)
        return max(0, min(5, cid))

    # ── Parameter change handler ──────────────────────────────────────────────

    def _on_param_change(self, params) -> SetParametersResult:
        """Called when any parameter is set via `ros2 param set`.

        HSV bounds and active_class are accepted as-is — the next image
        callback renders them automatically.  The three action-trigger
        params (save_now, reset_active, write_yaml) fire side effects
        immediately, then schedule themselves to reset back to False so
        the user can trigger them again next time.
        """
        for p in params:
            if p.name == "save_now" and bool(p.value) is True:
                self._save_current_class()
                self._schedule_reset("save_now")
            elif p.name == "reset_active" and bool(p.value) is True:
                self._reset_current_class()
                self._schedule_reset("reset_active")
            elif p.name == "write_yaml" and bool(p.value) is True:
                self._write_yaml()
                self._schedule_reset("write_yaml")
        return SetParametersResult(successful=True)

    def _schedule_reset(self, name: str) -> None:
        """Reset an action-trigger parameter back to False.

        Done from a one-shot timer so we don't recurse into the param
        callback we're currently inside.
        """
        # Keep a list of timers we can cancel after firing once.
        if not hasattr(self, "_pending_timers"):
            self._pending_timers = []

        def _do_reset():
            try:
                self.set_parameters([Parameter(name, Parameter.Type.BOOL, False)])
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"Failed to reset {name}: {e}")
            # Cancel ourselves so we only fire once.
            for t in list(self._pending_timers):
                if t.is_canceled():
                    self._pending_timers.remove(t)

        timer = self.create_timer(0.05, _do_reset)
        self._pending_timers.append(timer)

        # Cancel the timer after a tick so it really is one-shot.
        def _cancel():
            try:
                timer.cancel()
            except Exception:
                pass
        self.create_timer(0.1, _cancel)

    # ── Action handlers ───────────────────────────────────────────────────────

    def _save_current_class(self) -> None:
        cid = self._active_class_id()
        lo, hi = self._current_bounds()
        self._captured[cid] = {
            "label":     self._labels[cid],
            "hsv_lower": [lo[0], lo[1], lo[2]],
            "hsv_upper": [hi[0], hi[1], hi[2]],
            "min_contour_area": int(self._min_area),
        }
        self.get_logger().info(
            f"Saved class {cid} ({self._labels[cid]}): lower={list(lo)} upper={list(hi)}"
        )

    def _reset_current_class(self) -> None:
        cid = self._active_class_id()
        self._captured[cid] = None
        self.get_logger().info(f"Reset class {cid} ({self._labels[cid]}).")

    def _write_yaml(self) -> None:
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
            self.get_logger().error(f"Failed to write {self._output_path}: {e}")
            return

        n_ok = sum(1 for c in self._captured if c is not None)
        self.get_logger().info(
            f"Wrote {n_ok}/6 calibrated classes to {self._output_path}"
        )

    # ── Image processing ──────────────────────────────────────────────────────

    def _compute_mask(self, hsv: np.ndarray, lo, hi) -> np.ndarray:
        """HSV mask with hue wrap-around support (set H_min > H_max for red)."""
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
        """Produce the debug image: original BGR with mask overlay + status text."""
        out = bgr.copy()
        h, w = out.shape[:2]

        # Dim non-mask regions so the masked region stands out clearly.
        dim = (out * 0.35).astype(np.uint8)
        out = np.where(mask[..., None] > 0, out, dim)

        # Outline the largest contour for quick visual feedback.
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
        label = self._labels[cid]
        cv2.putText(out, f"Class {cid}: {label}", (10, 26),
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
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        cid = self._active_class_id()
        lo, hi = self._current_bounds()
        mask = self._compute_mask(hsv, lo, hi)
        annotated = self._annotate(bgr, mask, cid, lo, hi)

        try:
            out_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out_msg.header = msg.header
            self._debug_pub.publish(out_msg)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"Failed to publish debug image: {e}")


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