#!/usr/bin/env python3
"""Full mount calibration (rotation + translation) via SVD/Kabsch.

Replaces the earlier translation-only sketch.  Solves for the rigid
transform T_jaw_to_cam* that, applied to the camera's raw 3-D
detections, lands them at the user-measured physical positions.

Usage
-----
With the perception stack running (``./run.sh pick-and-place-servo``):

1. Park the arm at ``scan_pose`` (use Placo ``/go_to_joints`` once, or
   trigger ``/sort_by_class/trigger`` and abort before the pick).
   This calibrator does NOT move the arm — it relies on the arm
   staying still during sample capture.
2. Run::

       ros2 run so101_perception mount_calibrator

3. For each prompt, place the object at a measured (x, y, z) in
   ``base_link`` and type the coordinates as ``x y z``.  Hold for ~2 s
   while the script averages a window of detections.
4. Repeat for at least 4 samples (5–6 recommended) spread across the
   workspace — different X, Y, and at least one with different Z.
5. The script prints the new ``mount_x/y/z`` and ``mount_q*`` to paste
   into ``objects_hsv.yaml`` *and* ``zones_hsv.yaml``.

Math
----
We have the chain ``T_base_to_cam = T_base_to_jaw · T_jaw_to_cam``.
For each sample:

    real_in_base = T_base_to_jaw · T_jaw_to_cam* · p_cam

where ``p_cam`` is the raw 3-D point in the camera optical frame
(which we recover by undoing the current calibration from the
published ``base_link`` detection).

Rearranged, the unknowns ``R*, t*`` of ``T_jaw_to_cam*`` satisfy:

    R* · p_cam + t* = T_base_to_jaw^{-1} · real_in_base   ≡ real_in_jaw

This is exactly the Procrustes problem.  Kabsch / SVD gives the
optimal R* and t* in closed form across N ≥ 3 samples.

Why 4+ samples
--------------
3 generic non-collinear samples fully determine a rigid transform.
A 4th lets you spot configuration errors (large residuals).  More
samples ⇒ lower noise on R*, t*.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Quaternion
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


JAW_FRAME = "moving_jaw_so101_v1_link"
BASE_FRAME = "base_link"


# ---------------------------------------------------------------------------
#  Math helpers
# ---------------------------------------------------------------------------

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
    """Rotation matrix to quaternion (xyzw).  Handles all signs."""
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0:
        s = 2.0 * (1.0 + tr) ** 0.5
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Find rigid (R, t) minimising ||R·P_i + t - Q_i||² over rows.

    Returns (R, t, rms_residual_metres).
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    assert P.shape == Q.shape and P.shape[1] == 3
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    A = P - cP
    B = Q - cQ
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cQ - R @ cP
    residuals = (P @ R.T) + t - Q
    rms = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
    return R, t, rms


# ---------------------------------------------------------------------------
#  ROS node
# ---------------------------------------------------------------------------

class MountCalibrator(Node):
    def __init__(self):
        super().__init__("mount_calibrator")
        # Current mount in objects_hsv.yaml (used to undo and recover the raw
        # camera-frame detection from the published base_link point).  These
        # defaults match the un-calibrated values shipping in objects_hsv.yaml
        # — override via ROS parameters if your YAML differs.
        self.declare_parameter("current_mount_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("current_mount_quat_xyzw", [0.5, 0.5, -0.5, -0.5])

        self._cur_t = np.array(self.get_parameter("current_mount_xyz").value, dtype=float)
        cq = self.get_parameter("current_mount_quat_xyzw").value
        self._cur_R = quat_to_rotmat(*cq)

        self._latest_point: PointStamped | None = None
        self.create_subscription(
            PointStamped,
            "/object_classifier/detected_point",
            self._on_point,
            10,
        )
        self._tf_buf = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)

        self.get_logger().info(
            "MountCalibrator ready.  Park the arm at scan_pose, place an "
            "object at a known location, then enter its real (x, y, z) "
            "in base_link.  Type 'done' when finished (need >= 4 samples)."
        )

    def _on_point(self, msg: PointStamped):
        self._latest_point = msg

    def latest_point(self) -> np.ndarray | None:
        if self._latest_point is None:
            return None
        p = self._latest_point.point
        return np.array([p.x, p.y, p.z], dtype=float)

    def tf_base_to_jaw(self) -> Tuple[np.ndarray, np.ndarray] | None:
        """Returns (R_base_to_jaw, t_base_to_jaw)."""
        try:
            tf = self._tf_buf.lookup_transform(
                BASE_FRAME, JAW_FRAME, rclpy.time.Time(),
                timeout=Duration(seconds=0.5),
            )
        except Exception as exc:
            self.get_logger().warn(f"TF lookup failed: {exc}")
            return None
        q = tf.transform.rotation
        R = quat_to_rotmat(q.x, q.y, q.z, q.w)
        t = np.array(
            [tf.transform.translation.x,
             tf.transform.translation.y,
             tf.transform.translation.z], dtype=float)
        return R, t

    def recover_p_cam(self, reported_in_base: np.ndarray,
                      R_b2j: np.ndarray, t_b2j: np.ndarray) -> np.ndarray:
        """Undo the current calibration chain to recover the camera-frame point.

        reported_in_base = (T_base_to_jaw) · (R_cur · p_cam + t_cur)
        =>  R_cur · p_cam + t_cur = R_b2j^T · (reported - t_b2j)
        =>  p_cam = R_cur^T · ( R_b2j^T · (reported - t_b2j) - t_cur )
        """
        p_jaw = R_b2j.T @ (reported_in_base - t_b2j)
        p_cam = self._cur_R.T @ (p_jaw - self._cur_t)
        return p_cam


# ---------------------------------------------------------------------------
#  Interactive collection
# ---------------------------------------------------------------------------

def prompt_xyz(prompt: str) -> Tuple[float, float, float] | None:
    raw = input(prompt).strip()
    if raw.lower() in ("done", "q", "quit", "exit"):
        return None
    parts = raw.replace(",", " ").split()
    if len(parts) != 3:
        print("  expected 'x y z' in metres — try again")
        return prompt_xyz(prompt)
    try:
        return tuple(float(p) for p in parts)  # type: ignore
    except ValueError:
        print("  could not parse numbers — try again")
        return prompt_xyz(prompt)


def collect_one_sample(node: MountCalibrator, idx: int,
                       avg_window_s: float = 1.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Returns (p_cam, real_in_jaw, real_in_base) for one sample, or None."""
    real = prompt_xyz(f"  Sample {idx}: real xyz (m): ")
    if real is None:
        return None
    real = np.array(real, dtype=float)

    # Snapshot the TF immediately so any small wrist drift between TF and
    # detection lookup stays small.
    tf = node.tf_base_to_jaw()
    if tf is None:
        print("  TF not available — skipping sample")
        return None
    R_b2j, t_b2j = tf

    # Average detections for `avg_window_s` to denoise the depth.
    end = time.monotonic() + avg_window_s
    bag: List[np.ndarray] = []
    while time.monotonic() < end:
        pt = node.latest_point()
        if pt is not None:
            bag.append(pt)
        time.sleep(0.05)
    if not bag:
        print("  no detections in window — skipping")
        return None
    reported = np.mean(np.array(bag), axis=0)

    p_cam = node.recover_p_cam(reported, R_b2j, t_b2j)
    real_in_jaw = R_b2j.T @ (real - t_b2j)

    print(f"    reported = ({reported[0]:+.4f}, {reported[1]:+.4f}, {reported[2]:+.4f}) m")
    print(f"    p_cam    = ({p_cam[0]:+.4f}, {p_cam[1]:+.4f}, {p_cam[2]:+.4f}) m")
    print(f"    real     = ({real[0]:+.4f}, {real[1]:+.4f}, {real[2]:+.4f}) m base")
    print(f"               ({real_in_jaw[0]:+.4f}, {real_in_jaw[1]:+.4f}, {real_in_jaw[2]:+.4f}) m jaw")
    print()
    return p_cam, real_in_jaw, real


def main():
    rclpy.init()
    node = MountCalibrator()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("Waiting for the first detection on /object_classifier/detected_point …")
    deadline = time.monotonic() + 30.0
    while node.latest_point() is None:
        if time.monotonic() > deadline:
            print("ERROR: no detection in 30 s — is perception running?",
                  file=sys.stderr)
            rclpy.shutdown()
            sys.exit(1)
        time.sleep(0.2)
    print("Detection stream live.\n")

    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    idx = 0
    while True:
        idx += 1
        s = collect_one_sample(node, idx)
        if s is None:
            break
        samples.append(s)
        if len(samples) >= 4:
            print(f"  ({len(samples)} samples — enough for Kabsch.  "
                  f"Add more for robustness, or type 'done'.)\n")

    if len(samples) < 4:
        print(f"\nNeed at least 4 samples (got {len(samples)}). Aborting.")
        rclpy.shutdown()
        return

    P = np.array([s[0] for s in samples])  # camera-frame
    Q = np.array([s[1] for s in samples])  # jaw-frame (real)
    R_new, t_new, rms = kabsch(P, Q)
    qx, qy, qz, qw = rotmat_to_quat(R_new)

    print("══════════════════════════════════════════════════")
    print(f"Calibration result over {len(samples)} samples")
    print(f"RMS residual: {rms * 1000:.2f} mm")
    print()
    if rms > 0.02:
        print("WARNING: residual > 20 mm — calibration is shaky.  Check that")
        print("the arm stayed still in scan_pose, real coords were accurate,")
        print("and that samples span different X/Y/Z (not collinear).")
        print()
    print("New values for objects_hsv.yaml AND zones_hsv.yaml:")
    print()
    print(f"    mount_x:  {t_new[0]:+.6f}")
    print(f"    mount_y:  {t_new[1]:+.6f}")
    print(f"    mount_z:  {t_new[2]:+.6f}")
    print(f"    mount_qx: {qx:+.6f}")
    print(f"    mount_qy: {qy:+.6f}")
    print(f"    mount_qz: {qz:+.6f}")
    print(f"    mount_qw: {qw:+.6f}")
    print()
    print("Per-sample residuals (for diagnostics — should be sub-cm):")
    for i, (p_cam, real_jaw, real_base) in enumerate(samples, 1):
        predicted_in_jaw = R_new @ p_cam + t_new
        err = predicted_in_jaw - real_jaw
        print(f"  Sample {i}: error = "
              f"({err[0] * 1000:+.1f}, {err[1] * 1000:+.1f}, {err[2] * 1000:+.1f}) mm  "
              f"|err|={np.linalg.norm(err) * 1000:.1f} mm")
    print("══════════════════════════════════════════════════")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
