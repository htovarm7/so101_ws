"""Live HSV calibrator for the SO-101 multi-object classifier.

Subscribes to the RealSense colour stream and opens an OpenCV GUI with six
HSV trackbars plus a "class" selector.  Point the camera at one object at a
time, tune the sliders until the mask cleanly isolates that object, then
press ``s`` to save its HSV range into the selected class slot.  When all
six classes are captured, press ``w`` to write the resulting parameters to
``objects_hsv.yaml``.

Threading note
--------------
The OpenCV GUI loop calls ``cv2.waitKey(1)`` every tick, which blocks for
~1–30ms.  If the image subscription and the GUI timer share a single
executor thread, that block starves the image callback and the window
never receives a frame — you stay stuck on "Waiting for image…".

Fix: a MultiThreadedExecutor (in ``main``) plus two MutuallyExclusive
callback groups (one for the image subscription, one for the GUI timer)
so they can run in parallel.
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
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

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

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter(
            "color_image_topic", "/camera/camera/color/image_raw"
        )
        self.declare_parameter("class_labels", DEFAULT_LABELS)
        self.declare_parameter("output_path", "")
        self.declare_parameter("min_contour_area", 500)
        self.declare_parameter("window_width", 640)

        color_topic: str = self.get_parameter("color_image_topic").value
        labels = list(self.get_parameter("class_labels").value)
        if len(labels) != 6:
            self.get_logger().warn(
                f"class_labels has {len(labels)} entries; padding/truncating to 6."
            )
            labels = (labels + DEFAULT_LABELS)[:6]
        self._labels: List[str] = labels
        self._min_area: int = int(self.get_parameter("min_contour_area").value)
        self._win_w: int = int(self.get_parameter("window_width").value)

        out_param: str = self.get_parameter("output_path").value
        self._output_path: str = out_param if out_param else self._default_output_path()

        # ── State ─────────────────────────────────────────────────────────────
        self._captured: List[Optional[Dict[str, List[int]]]] = [None] * 6
        self._bridge = CvBridge()
        self._latest_bgr: Optional[np.ndarray] = None

        # ── Callback groups ───────────────────────────────────────────────────
        # Image callback and GUI timer go in SEPARATE groups so the
        # MultiThreadedExecutor can run them in parallel.  Without this the
        # GUI's blocking cv2.waitKey() prevents image messages from being
        # delivered, and the window stays on "Waiting for image…".
        self._cb_image = MutuallyExclusiveCallbackGroup()
        self._cb_gui   = MutuallyExclusiveCallbackGroup()

        # ── Subscriptions ─────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Image, color_topic, self._image_cb, sensor_qos,
            callback_group=self._cb_image,
        )

        # ── OpenCV window setup ───────────────────────────────────────────────
        self._win_main = "HSV Calibrator (image)"
        self._win_mask = "HSV Calibrator (mask)"
        cv2.namedWindow(self._win_main, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self._win_mask, cv2.WINDOW_NORMAL)

        cv2.createTrackbar("class",  self._win_main, 0, 5,   self._noop)
        cv2.createTrackbar("H min",  self._win_main, 0, 179, self._noop)
        cv2.createTrackbar("H max",  self._win_main, 179, 179, self._noop)
        cv2.createTrackbar("S min",  self._win_main, 80, 255, self._noop)
        cv2.createTrackbar("S max",  self._win_main, 255, 255, self._noop)
        cv2.createTrackbar("V min",  self._win_main, 50, 255, self._noop)
        cv2.createTrackbar("V max",  self._win_main, 255, 255, self._noop)

        # GUI tick runs on its own callback group so it can spin in parallel
        # with the image subscription.
        self.create_timer(1.0 / 30.0, self._gui_tick, callback_group=self._cb_gui)

        self.get_logger().info(
            "HSV Calibrator ready\n"
            f"  colour topic : {color_topic}\n"
            f"  output path  : {self._output_path}\n"
            f"  classes      : {self._labels}\n"
            "  Keys: [s]ave  [r]eset  [w]rite YAML  [n]ext  [p]rev  [q]uit"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _noop(_value: int) -> None:
        """OpenCV trackbar callback that does nothing — we poll values in the loop."""
        return None

    def _default_output_path(self) -> str:
        """Resolve the install-share config dir for so101_perception."""
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory("so101_perception")
            return os.path.join(share, "config", "objects_hsv.yaml")
        except Exception:
            return os.path.abspath("objects_hsv.yaml")

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image) -> None:
        try:
            self._latest_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"cv_bridge conversion failed: {e}")

    # ── GUI loop ──────────────────────────────────────────────────────────────

    def _read_trackbars(self):
        cid    = cv2.getTrackbarPos("class", self._win_main)
        h_lo   = cv2.getTrackbarPos("H min", self._win_main)
        h_hi   = cv2.getTrackbarPos("H max", self._win_main)
        s_lo   = cv2.getTrackbarPos("S min", self._win_main)
        s_hi   = cv2.getTrackbarPos("S max", self._win_main)
        v_lo   = cv2.getTrackbarPos("V min", self._win_main)
        v_hi   = cv2.getTrackbarPos("V max", self._win_main)
        return cid, (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi)

    def _compute_mask(self, hsv: np.ndarray, lo, hi) -> np.ndarray:
        """Compute an HSV mask, supporting wrap-around in the H channel."""
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

    def _draw_overlay(self, img: np.ndarray, cid: int, lo, hi) -> np.ndarray:
        """Annotate the live image with current class, HSV range, and capture status."""
        out = img.copy()
        h, w = out.shape[:2]

        # Status bar at the top.
        cv2.rectangle(out, (0, 0), (w, 70), (0, 0, 0), thickness=-1)
        label = self._labels[cid]
        cv2.putText(out, f"Class {cid}: {label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(
            out,
            f"HSV: [{lo[0]},{lo[1]},{lo[2]}]  ->  [{hi[0]},{hi[1]},{hi[2]}]",
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
        )

        # Per-class capture status strip at the bottom.
        cv2.rectangle(out, (0, h - 30), (w, h), (0, 0, 0), thickness=-1)
        for i in range(6):
            colour = (0, 200, 0) if self._captured[i] is not None else (60, 60, 60)
            if i == cid:
                # Highlight the active slot.
                cv2.rectangle(out, (10 + i * 90, h - 25),
                              (95 + i * 90, h - 5), (0, 255, 255), 1)
            cv2.putText(out, f"{i}:{'OK' if self._captured[i] else '--'}",
                        (15 + i * 90, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        return out

    def _maybe_resize(self, img: np.ndarray) -> np.ndarray:
        if self._win_w <= 0 or img.shape[1] == self._win_w:
            return img
        new_h = int(img.shape[0] * self._win_w / img.shape[1])
        return cv2.resize(img, (self._win_w, new_h), interpolation=cv2.INTER_AREA)

    def _gui_tick(self) -> None:
        if self._latest_bgr is None:
            # Still show a placeholder so the user knows the node is alive.
            placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for image...", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow(self._win_main, placeholder)
            cv2.imshow(self._win_mask, placeholder)
            cv2.waitKey(1)
            return

        bgr = self._latest_bgr
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        cid, lo, hi = self._read_trackbars()
        mask = self._compute_mask(hsv, lo, hi)
        masked = cv2.bitwise_and(bgr, bgr, mask=mask)

        annotated = self._draw_overlay(bgr, cid, lo, hi)

        cv2.imshow(self._win_main, self._maybe_resize(annotated))
        cv2.imshow(self._win_mask, self._maybe_resize(masked))
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return
        self._handle_key(key, cid, lo, hi)

    # ── Key handling ──────────────────────────────────────────────────────────

    def _handle_key(self, key: int, cid: int, lo, hi) -> None:
        ch = chr(key).lower() if key < 128 else ""
        if ch == "s":
            self._captured[cid] = {
                "label":     self._labels[cid],
                "hsv_lower": [int(lo[0]), int(lo[1]), int(lo[2])],
                "hsv_upper": [int(hi[0]), int(hi[1]), int(hi[2])],
                "min_contour_area": int(self._min_area),
            }
            self.get_logger().info(
                f"Saved class {cid} ({self._labels[cid]}): "
                f"lower={list(lo)} upper={list(hi)}"
            )
        elif ch == "r":
            self._captured[cid] = None
            self.get_logger().info(f"Reset class {cid} ({self._labels[cid]}).")
        elif ch == "n":
            new_cid = (cid + 1) % 6
            cv2.setTrackbarPos("class", self._win_main, new_cid)
        elif ch == "p":
            new_cid = (cid - 1) % 6
            cv2.setTrackbarPos("class", self._win_main, new_cid)
        elif ch == "w":
            self._write_yaml()
        elif ch == "q":
            self.get_logger().info("Quit requested. Shutting down.")
            rclpy.shutdown()

    # ── YAML output ───────────────────────────────────────────────────────────

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


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HSVCalibrator()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        try:
            executor.shutdown()
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()