#!/usr/bin/env python3
"""Gaze-tracking visual servoing via /servo_target.

Publishes a Cartesian target that **constrains the gripper to a sphere
of constant radius around base_link**, rotated to face the detected
object's azimuth.  The gripper never reaches toward the object — it
just pans (and tilts a little in Z if you allow it).  Placo's
``cartesian_motion_node`` consumes the target and produces joint
commands, so we never fight any controller.

This is a different shape of servoing than the earlier joint-space
attempt: the published target is a single PoseStamped on
``/servo_target`` and the existing IK pipeline handles everything
downstream.  No direct controller publishing, no /go_to_joints during
the control loop.  Easier to layer on top of the running stack.

Trigger interface
-----------------
``/sort_by_class/trigger`` (std_msgs/Empty) toggles tracking.
  First press  → arm tracking.
  Second press → disarm.  We stop publishing; cartesian_motion_node's
                 own freshness gate (0.2 s) lets the arm hold position.
"""
from __future__ import annotations

import sys
import threading
import time
import traceback
from typing import Optional, Tuple

import numpy as np
import rclpy
import rclpy.time
import tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty
from tf_transformations import quaternion_multiply

from so101_kinematics_msgs.srv import GoToJoints


BASE_FRAME = "base_link"
# End-effector frame Placo uses (matches cartesian_motion_node.EE_FRAME
# and the manipulator group's tip_link in so101_arm.srdf).
EE_FRAME = "gripper_frame_link"

# scan_pose joints — matches the SRDF named state used elsewhere in the
# stack.  Kept here so this script is self-contained (no MoveIt config
# parsing needed at runtime).
SCAN_POSE_JOINTS = {
    "shoulder_pan":   0.0,
    "shoulder_lift": -1.0996,
    "elbow_flex":     0.1745,
    "wrist_flex":     1.3614,
    "wrist_roll":     1.5708,
}


def yaw_quat(dpsi: float) -> Tuple[float, float, float, float]:
    """Quaternion (xyzw) representing a rotation around +Z by ``dpsi``."""
    return (0.0, 0.0, float(np.sin(dpsi / 2.0)), float(np.cos(dpsi / 2.0)))


class GazeServo(Node):
    def __init__(self):
        super().__init__("gaze_servo")

        d = self.declare_parameter
        d("object_point_topic", "/object_classifier/detected_point")
        d("servo_target_topic", "/servo_target")
        d("trigger_topic",      "/sort_by_class/trigger")
        d("ee_frame",           EE_FRAME)
        d("servo_rate_hz",      20.0)

        # ── Optional small elevation follow ──────────────────────────
        # Lets target Z shift slightly with object Z so tall vs. flat
        # objects both get centred in the camera.  Set elev_gain = 0
        # to fully lock Z to scan height (recommended for first tests).
        d("elev_gain",      0.0)
        d("elev_z_ref_m",   0.05)
        d("elev_z_min_m",  -0.05)
        d("elev_z_max_m",   0.10)

        # ── Smoothing on the published reference ─────────────────────
        # EMA on the target azimuth (not on the detection itself);
        # the arm pursues a low-pass-filtered reference so a single
        # noisy detection frame can't yank the IK around.
        d("lpf_alpha",      0.20)

        # ── Stale-detection gate ─────────────────────────────────────
        # If the latest detected_point is older than this, we stop
        # publishing.  cartesian_motion_node will then hit its own
        # 0.2 s staleness gate and the arm holds the last commanded
        # pose — exactly the behaviour we want.
        d("detection_freshness_s", 1.0)

        g = self.get_parameter
        self._rate      = float(g("servo_rate_hz").value)
        self._ee_frame  = str(g("ee_frame").value)
        self._elev_gain = float(g("elev_gain").value)
        self._elev_ref  = float(g("elev_z_ref_m").value)
        self._elev_lo   = float(g("elev_z_min_m").value)
        self._elev_hi   = float(g("elev_z_max_m").value)
        self._alpha     = float(np.clip(g("lpf_alpha").value, 1e-3, 1.0))
        self._fresh_s   = float(g("detection_freshness_s").value)
        servo_topic     = str(g("servo_target_topic").value)
        point_topic     = str(g("object_point_topic").value)
        trigger_topic   = str(g("trigger_topic").value)

        # ── State ────────────────────────────────────────────────────
        # _scan_pos / _scan_quat / _scan_psi are CAPTURED from TF after
        # the parking move completes.  They define the canonical EE
        # pose at scan_pose; tracking rotates this pose around base +Z.
        self._scan_pos:   Optional[np.ndarray] = None       # (3,)
        self._scan_quat:  Optional[np.ndarray] = None       # xyzw
        self._scan_psi:   float = 0.0                       # rad
        self._filtered_psi: Optional[float] = None          # EMA
        self._last_obj:   Optional[np.ndarray] = None
        self._last_obj_t = 0.0
        self._tracking = False
        self._ready    = threading.Event()
        self._lock     = threading.Lock()

        # ── ROS I/O ──────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            PointStamped, point_topic, self._on_point, sensor_qos)
        self.create_subscription(
            Empty, trigger_topic, self._on_trigger, 10)
        self._target_pub = self.create_publisher(
            PoseStamped, servo_topic, 10)

        # TF listener for the scan-pose EE pose snapshot.
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Park at scan_pose via Placo's /go_to_joints, then idle until
        # trigger.  Done on a worker thread so __init__ doesn't block.
        self._goto = self.create_client(GoToJoints, "/go_to_joints")
        threading.Thread(target=self._park, daemon=True).start()

        self.create_timer(1.0 / self._rate, self._tick)
        self.get_logger().info(
            "GazeServo initialised — waiting for scan_pose init…"
        )

    # ── Init ──────────────────────────────────────────────────────────

    def _park(self):
        if not self._goto.wait_for_service(timeout_sec=20.0):
            self.get_logger().warn(
                "/go_to_joints unavailable — assuming arm is at scan_pose"
            )
        else:
            req = GoToJoints.Request()
            req.joint_names = list(SCAN_POSE_JOINTS.keys())
            req.positions = [float(v) for v in SCAN_POSE_JOINTS.values()]
            req.duration = 2.0
            future = self._goto.call_async(req)
            # Poll the future — DON'T spin here; main is already
            # spinning the node and rclpy.spin_until_future_complete
            # from a worker thread raises "Executor is already spinning".
            t0 = time.time()
            while not future.done() and time.time() - t0 < 10.0:
                time.sleep(0.05)
            time.sleep(0.5)

        # Snapshot the EE pose AT scan_pose.  This is the canonical pose
        # we'll rotate around base +Z to follow the object.  Doing this
        # via TF (instead of hardcoding XYZ + quat) guarantees the
        # published target lies on the SAME IK branch as scan_pose, so
        # Placo only needs to change shoulder_pan to reach it.
        if not self._capture_scan_pose():
            self.get_logger().error(
                "Failed to capture EE pose at scan_pose — gaze tracker disabled."
            )
            return
        self._ready.set()
        self.get_logger().info(
            f"Scan_pose captured (pos=({self._scan_pos[0]:+.3f}, "
            f"{self._scan_pos[1]:+.3f}, {self._scan_pos[2]:+.3f}), "
            f"ψ₀={self._scan_psi:+.3f} rad).  "
            "Publish /sort_by_class/trigger to start."
        )

    def _capture_scan_pose(self) -> bool:
        """Snapshot base_link → EE_FRAME after parking.

        Retries briefly because TF may not have caught up to the final
        joint state by the time we ask.
        """
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                ts = self._tf_buffer.lookup_transform(
                    BASE_FRAME, self._ee_frame, rclpy.time.Time())
                p = ts.transform.translation
                q = ts.transform.rotation
                self._scan_pos = np.array([p.x, p.y, p.z], dtype=float)
                self._scan_quat = np.array([q.x, q.y, q.z, q.w], dtype=float)
                self._scan_psi = float(np.arctan2(p.y, p.x))
                return True
            except (tf2_ros.LookupException,
                    tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException):
                time.sleep(0.1)
        return False

    # ── Callbacks ─────────────────────────────────────────────────────

    def _on_point(self, msg: PointStamped):
        with self._lock:
            self._last_obj = np.array(
                [msg.point.x, msg.point.y, msg.point.z], dtype=float)
            self._last_obj_t = time.time()

    def _on_trigger(self, _msg: Empty):
        if not self._ready.is_set():
            self.get_logger().warn("Trigger ignored: still parking.")
            return
        self._tracking = not self._tracking
        if self._tracking:
            # Reset the filtered azimuth so the EMA starts cleanly from
            # the next detection rather than slewing from a stale value.
            self._filtered_psi = None
            self.get_logger().info("Tracking ARMED.")
        else:
            self.get_logger().info("Tracking DISARMED — holding pose.")

    # ── Control loop ──────────────────────────────────────────────────

    def _tick(self):
        if not self._ready.is_set() or not self._tracking:
            return
        if self._scan_pos is None or self._scan_quat is None:
            return
        now = time.time()
        with self._lock:
            obj = None if self._last_obj is None else self._last_obj.copy()
            obj_t = self._last_obj_t
        if obj is None or now - obj_t > self._fresh_s:
            return  # stale or no detection — let cartesian node stale out

        # Desired azimuth pointing at the object.
        psi_obj = float(np.arctan2(obj[1], obj[0]))

        # EMA smoothing on the AZIMUTH (not on x/y) so the filter is
        # well-behaved around the ±π wrap.  Carry an unwrapped phase.
        if self._filtered_psi is None:
            self._filtered_psi = psi_obj
        else:
            # Shortest-arc step (handles the wraparound at ±π).
            err = ((psi_obj - self._filtered_psi + np.pi)
                   % (2.0 * np.pi)) - np.pi
            self._filtered_psi = float(self._filtered_psi + self._alpha * err)

        # Δψ = rotation we want to apply to the scan-pose EE pose
        # around base +Z so the gripper faces the object.
        dpsi = float(self._filtered_psi - self._scan_psi)
        cos_d, sin_d = float(np.cos(dpsi)), float(np.sin(dpsi))

        # Rotate the captured scan-pose position around base Z by Δψ.
        # This keeps |xy| and z identical to scan_pose — only azimuth
        # changes, so the IK branch stays on the scan_pose manifold.
        px, py, pz = self._scan_pos
        nx = px * cos_d - py * sin_d
        ny = px * sin_d + py * cos_d
        nz = pz + float(np.clip(
            self._elev_gain * (float(obj[2]) - self._elev_ref),
            self._elev_lo, self._elev_hi,
        ))

        # Rotate the captured orientation by the same Δψ around base Z:
        #   new_quat = q_yaw(dpsi) * scan_quat   (Hamilton product)
        new_quat = quaternion_multiply(yaw_quat(dpsi), self._scan_quat)

        ps = PoseStamped()
        ps.header.frame_id = BASE_FRAME
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(nx)
        ps.pose.position.y = float(ny)
        ps.pose.position.z = float(nz)
        ps.pose.orientation.x = float(new_quat[0])
        ps.pose.orientation.y = float(new_quat[1])
        ps.pose.orientation.z = float(new_quat[2])
        ps.pose.orientation.w = float(new_quat[3])
        self._target_pub.publish(ps)


def main():
    rclpy.init()
    try:
        node = GazeServo()
        rclpy.spin(node)
    except BaseException:
        sys.stderr.write("gaze_servo fatal:\n" + traceback.format_exc())
        sys.stderr.flush()
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
