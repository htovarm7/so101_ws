#!/usr/bin/env python3
"""Solve the camera-to-jaw mount offset from N (real, reported) samples.

Usage
-----
With the arm parked in ``scan_pose`` and the perception stack running:

    ros2 run so101_perception mount_calibrator

The script prompts for the real (x, y, z) of the object in ``base_link`` —
measure with a ruler — then captures the most recent
``/object_classifier/detected_point`` for that sample.  Repeat for at least
3 samples spread across the workspace (different X, Y, and one with Z on
the table; do NOT stack objects).  At the end it prints the corrected
``mount_x/y/z`` values to paste into ``objects_hsv.yaml`` *and*
``zones_hsv.yaml``.

Math
----
``detected_point`` in ``base_link`` =
    T_base_to_jaw @ T_jaw_to_cam @ point_in_cam_frame

``mount_x/y/z`` are the translation part of ``T_jaw_to_cam``.  Translating
that origin by ``delta`` (in the jaw frame) shifts every reported point by
``R_base_to_jaw @ delta`` in ``base_link``.  So:

    delta_in_jaw = R_jaw_to_base @ (real_in_base - reported_in_base)

The rotation ``R_jaw_to_base`` is the inverse of ``R_base_to_jaw``, which
we look up from TF the moment we capture each sample.  The script averages
``delta_in_jaw`` across samples; if the per-sample residuals are large,
the mount *rotation* is also off and a pure translation fit won't fix it
— that is reported as a warning.
"""
from __future__ import annotations

import sys
import time
from typing import List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


JAW_FRAME = "moving_jaw_so101_v1_link"
BASE_FRAME = "base_link"


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert geometry_msgs Quaternion (xyzw) to a 3x3 rotation matrix."""
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


class MountCalibrator(Node):
    def __init__(self) -> None:
        super().__init__("mount_calibrator")
        self._latest: PointStamped | None = None
        self.create_subscription(
            PointStamped,
            "/object_classifier/detected_point",
            self._on_point,
            10,
        )
        self._tf_buf = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)
        self.get_logger().info(
            "mount_calibrator ready.  Park the arm in scan_pose, put an "
            "object at a known position on the table, then enter its real "
            "coordinates below.  Type 'done' at the prompt to finish."
        )

    def _on_point(self, msg: PointStamped) -> None:
        self._latest = msg

    def latest_point(self) -> Tuple[float, float, float] | None:
        if self._latest is None:
            return None
        p = self._latest.point
        return (p.x, p.y, p.z)

    def jaw_rotation_in_base(self) -> np.ndarray | None:
        """Returns R_base_to_jaw (a 3x3 ndarray) or None if TF not ready."""
        try:
            tf = self._tf_buf.lookup_transform(
                BASE_FRAME, JAW_FRAME, rclpy.time.Time(),
                timeout=Duration(seconds=0.5),
            )
        except Exception as exc:
            self.get_logger().warn(f"TF lookup failed: {exc}")
            return None
        q = tf.transform.rotation
        return quat_to_rotmat(q.x, q.y, q.z, q.w)


def prompt_floats(prompt: str) -> Tuple[float, float, float] | None:
    raw = input(prompt).strip()
    if raw.lower() in ("done", "q", "quit", "exit"):
        return None
    parts = raw.replace(",", " ").split()
    if len(parts) != 3:
        print("  expected 3 numbers (x y z), try again")
        return prompt_floats(prompt)
    try:
        return tuple(float(p) for p in parts)  # type: ignore
    except ValueError:
        print("  could not parse numbers, try again")
        return prompt_floats(prompt)


def main() -> None:
    rclpy.init()
    node = MountCalibrator()

    # Spin in background so subscription + TF buffer fill up.
    import threading
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Wait until the first detection arrives.
    print("Waiting for the first /object_classifier/detected_point …")
    deadline = time.monotonic() + 30.0
    while node.latest_point() is None:
        if time.monotonic() > deadline:
            print("ERROR: no detection arrived in 30 s.  Is the perception "
                  "stack running and an object in view?", file=sys.stderr)
            rclpy.shutdown()
            sys.exit(1)
        time.sleep(0.2)
    print("Detection stream live.\n")

    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (real, reported, R_b2j)
    i = 0
    while True:
        i += 1
        print(f"── Sample {i} ──")
        print("  Place an object at a measured position on the table.")
        print("  Hold still until detected_point stabilises, then enter real "
              "coordinates as 'x y z' (metres, in base_link), or 'done'.")
        real = prompt_floats(f"  real xyz [{i}]: ")
        if real is None:
            break

        # Capture a stable detection: average the last N samples received
        # over a short window after the prompt finishes.
        window_s = 1.0
        end = time.monotonic() + window_s
        buf: List[Tuple[float, float, float]] = []
        while time.monotonic() < end:
            pt = node.latest_point()
            if pt is not None:
                buf.append(pt)
            time.sleep(0.05)
        if not buf:
            print("  no detections in window — skipping sample")
            i -= 1
            continue
        reported = np.mean(np.array(buf), axis=0)

        R_b2j = node.jaw_rotation_in_base()
        if R_b2j is None:
            print("  TF not available — skipping sample")
            i -= 1
            continue

        diff_in_base = np.array(real) - reported
        diff_in_jaw = R_b2j.T @ diff_in_base
        samples.append((np.array(real), reported, R_b2j))
        print(f"    reported = ({reported[0]:+.4f}, {reported[1]:+.4f}, {reported[2]:+.4f}) m")
        print(f"    diff (real-reported) in base = "
              f"({diff_in_base[0]:+.4f}, {diff_in_base[1]:+.4f}, {diff_in_base[2]:+.4f}) m")
        print(f"    diff transformed into jaw frame = "
              f"({diff_in_jaw[0]:+.4f}, {diff_in_jaw[1]:+.4f}, {diff_in_jaw[2]:+.4f}) m")
        print()

    if len(samples) < 1:
        print("No samples collected.")
        rclpy.shutdown()
        return

    # Average delta in jaw frame across samples; report residual std as a
    # health check (large std => rotation calibration is also off).
    deltas_in_jaw = np.array([s[2].T @ (s[0] - s[1]) for s in samples])
    mean_delta = deltas_in_jaw.mean(axis=0)
    std_delta = deltas_in_jaw.std(axis=0)

    print("══════════════════════════════════════════════════")
    print(f"Samples used: {len(samples)}")
    print(f"Mean delta in JAW frame: "
          f"({mean_delta[0]:+.4f}, {mean_delta[1]:+.4f}, {mean_delta[2]:+.4f}) m")
    print(f"Std  delta per axis    : "
          f"({std_delta[0]:.4f}, {std_delta[1]:.4f}, {std_delta[2]:.4f}) m")
    if np.linalg.norm(std_delta) > 0.02:
        print("WARNING: per-sample residual > 2 cm RMS — the camera mount "
              "rotation is probably wrong too.  Translation calibration "
              "alone will not fully correct it.  Re-measure and consider "
              "calibrating the quaternion next.")

    print()
    print("Add these to mount_x/y/z (objects_hsv.yaml AND zones_hsv.yaml):")
    print(f"    mount_x: {mean_delta[0]:+.6f}")
    print(f"    mount_y: {mean_delta[1]:+.6f}")
    print(f"    mount_z: {mean_delta[2]:+.6f}")
    print()
    print("These are ADDITIVE to whatever mount_x/y/z you currently have. "
          "If your YAML has mount_x: 0.0 today (and likewise for y/z), the "
          "values above are the final ones to paste.  Otherwise add them "
          "to the existing values.")
    print("══════════════════════════════════════════════════")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
