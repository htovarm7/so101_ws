#!/usr/bin/env python3
"""Hover-track an object then pick it when it is stable and reachable.

State machine (single object, no place phase):

  TRACKING   — Continuously publish ``/servo_target`` ≈ detection + hover
               offset.  Placo's servo loop slews the arm so the gripper
               sits ``approach_height_m`` above wherever the camera last
               saw the object.

  STABILITY  — While tracking, watch the detection's spatial spread.  When
               the detection has stayed within ``stability_radius_m`` for
               ``stability_seconds`` AND the object is inside the pickable
               workspace, advance.

  PICK       — Stop tracking.  Publish a fixed target at the pick pose
               (lower Z by approach_height_m).  Once the arm has settled,
               close the gripper, then retreat back up by retreat_height_m.

  DONE       — Idle.  The node exits the state machine.  A new
               ``/sort_by_class/trigger`` rearms it from scratch.

Why this is different from sort_by_class_servo.py:
  - No detection-then-plan-then-execute.  The arm follows the object live.
  - No zones / no place phase.
  - Visual feedback closes the residual that mount-calibration error
    would otherwise force into pick_*_offset_m fudge factors.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from collections import deque
from typing import Optional, Tuple

import numpy as np
import rclpy
import rclpy.logging
from action_msgs.msg import GoalStatus
from control_msgs.action import ParallelGripperCommand
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, String
from tf_transformations import quaternion_from_matrix

from so101_kinematics_msgs.srv import GoToJoints


BASE_FRAME = "base_link"

# scan_pose joints — see SRDF
SCAN_POSE_JOINTS = {
    "shoulder_pan":   0.0,
    "shoulder_lift": -1.0996,
    "elbow_flex":     0.1745,
    "wrist_flex":     1.3614,
    "wrist_roll":     1.5708,
}

GRIPPER_OPEN = 1.5
GRIPPER_CLOSED = -0.16


# ---------------------------------------------------------------------------
#  Pose helpers
# ---------------------------------------------------------------------------

def topdown_quat_for(xyz_m) -> Tuple[float, float, float, float]:
    """Top-down quaternion with outward tilt at workspace edge.

    Identical to the formula validated in sort_by_class.py against tf2_echo
    of the real arm at scan_pose.
    """
    r = float(np.hypot(xyz_m[0], xyz_m[1]))
    theta = float(np.clip((r - 0.18) * 4.0, 0.0, 0.7))
    psi = float(np.arctan2(xyz_m[1], xyz_m[0]))
    c, s = float(np.cos(theta)), float(np.sin(theta))
    R0 = np.array([
        [0.0, -c,  s],
        [-1.0, 0.0, 0.0],
        [0.0, -s, -c],
    ])
    cz, sz = float(np.cos(psi)), float(np.sin(psi))
    Rpsi = np.array([
        [cz, -sz, 0.0],
        [sz,  cz, 0.0],
        [0.0, 0.0, 1.0],
    ])
    R = Rpsi @ R0
    T = np.eye(4)
    T[:3, :3] = R
    qx, qy, qz, qw = quaternion_from_matrix(T)
    return (float(qx), float(qy), float(qz), float(qw))


def make_pose_stamped(xyz_m, quat_xyzw, frame_id: str = BASE_FRAME) -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    ps.pose.position.x = float(xyz_m[0])
    ps.pose.position.y = float(xyz_m[1])
    ps.pose.position.z = float(xyz_m[2])
    ps.pose.orientation.x = float(quat_xyzw[0])
    ps.pose.orientation.y = float(quat_xyzw[1])
    ps.pose.orientation.z = float(quat_xyzw[2])
    ps.pose.orientation.w = float(quat_xyzw[3])
    return ps


# ---------------------------------------------------------------------------
#  Gripper action client (parallel_gripper_action_controller)
# ---------------------------------------------------------------------------

class GripperClient:
    ACTION_NAME = "/follower/gripper_controller/gripper_cmd"

    def __init__(self, node: Node):
        self._node = node
        self._client = ActionClient(node, ParallelGripperCommand, self.ACTION_NAME)
        node.get_logger().info(f"Waiting for gripper action at {self.ACTION_NAME}…")
        self._client.wait_for_server(timeout_sec=15.0)
        node.get_logger().info("Gripper action available.")

    def set_position(self, position: float, *, timeout_s: float = 4.0) -> bool:
        goal = ParallelGripperCommand.Goal()
        goal.command = JointState()
        goal.command.name = ["gripper"]
        goal.command.position = [float(position)]
        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self._node, send_future, timeout_sec=5.0)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return False
        res_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, res_future, timeout_sec=timeout_s)
        if res_future.result() is None:
            return False
        # parallel_gripper_action_controller marks stalled grasps as ABORTED;
        # we treat stall as success because the gripper has clamped.
        return True


# ---------------------------------------------------------------------------
#  Track-and-pick state machine
# ---------------------------------------------------------------------------

# State constants
S_IDLE = "IDLE"
S_TRACKING = "TRACKING"
S_PICK = "PICK"
S_DESCEND = "DESCEND"
S_CLOSE = "CLOSE"
S_RETREAT = "RETREAT"
S_DONE = "DONE"


class TrackAndPick(Node):
    def __init__(self):
        # Node name kept as `sort_by_class` so the existing pick_and_place.yaml
        # parameters (pick_*_offset_m, approach_height_m, retreat_height_m,
        # etc.) load without renaming the YAML.  New params below — like
        # stability_*, pick_zone_* — can be added under the same section.
        super().__init__("sort_by_class")

        # ── Parameters ────────────────────────────────────────────────
        d = self.declare_parameter
        d("object_label_topic", "/object_classifier/detected_label")
        d("object_point_topic", "/object_classifier/detected_point")
        d("servo_target_topic", "/servo_target")
        d("joint_states_topic", "/follower/joint_states")
        # which labels to track; empty => any label except 'none'
        d("accept_labels", [""])

        # Hover / pick geometry
        d("approach_height_m", 0.05)
        d("retreat_height_m",  0.05)
        d("pick_z_offset_m",   0.0)
        d("pick_x_offset_m",   0.0)
        d("pick_y_offset_m",   0.0)

        # Reachable pick zone (in base_link)
        d("pick_zone_radius_min_m", 0.10)
        d("pick_zone_radius_max_m", 0.27)
        d("pick_zone_z_min_m", -0.05)
        d("pick_zone_z_max_m",  0.20)

        # Stability gate
        d("stability_radius_m", 0.01)
        d("stability_seconds",  1.5)
        d("detection_freshness_s", 0.5)  # ignore detections older than this
        # If the detection drops for this many seconds while tracking,
        # return to scan_pose and disarm.  Prevents the arm from holding
        # a stale hover pose indefinitely when the object disappears.
        d("lost_detection_timeout_s", 3.0)

        # Loop / motion
        d("servo_rate_hz", 30.0)
        d("descend_time_s", 1.5)
        d("retreat_time_s", 1.5)
        d("settle_time_s",  0.5)

        # Tracking smoothness — applied to the *published* target, not to
        # the detection itself.  The arm pursues a low-pass-filtered and
        # velocity-clamped reference rather than chasing every detection
        # frame.  Two knobs (use either or both):
        #   tracking_lpf_alpha  — exponential-moving-average coefficient.
        #     0.0 = freeze, 1.0 = no filter.  Smaller = smoother + slower.
        #   tracking_max_speed_m_per_s — cap on how fast the reference can
        #     move toward a fresh detection (metres per second).
        d("tracking_lpf_alpha", 0.15)
        d("tracking_max_speed_m_per_s", 0.10)

        g = self.get_parameter

        self._accept = {n for n in g("accept_labels").value if n}
        self._approach = float(g("approach_height_m").value)
        self._retreat = float(g("retreat_height_m").value)
        self._pick_z_off = float(g("pick_z_offset_m").value)
        self._pick_x_off = float(g("pick_x_offset_m").value)
        self._pick_y_off = float(g("pick_y_offset_m").value)

        self._r_min = float(g("pick_zone_radius_min_m").value)
        self._r_max = float(g("pick_zone_radius_max_m").value)
        self._z_min = float(g("pick_zone_z_min_m").value)
        self._z_max = float(g("pick_zone_z_max_m").value)

        self._stab_radius = float(g("stability_radius_m").value)
        self._stab_seconds = float(g("stability_seconds").value)
        self._fresh_s = float(g("detection_freshness_s").value)
        self._lost_timeout = float(g("lost_detection_timeout_s").value)

        self._servo_rate = float(g("servo_rate_hz").value)
        self._descend_t = float(g("descend_time_s").value)
        self._retreat_t = float(g("retreat_time_s").value)
        self._settle_t = float(g("settle_time_s").value)
        self._lpf_alpha = float(np.clip(g("tracking_lpf_alpha").value, 1e-3, 1.0))
        self._max_speed = float(g("tracking_max_speed_m_per_s").value)

        # ── State ─────────────────────────────────────────────────────
        self._state = S_IDLE
        self._state_t0 = time.time()
        self._lock = threading.Lock()
        self._last_label: Optional[str] = None
        self._last_label_t = 0.0
        self._last_point: Optional[np.ndarray] = None  # (3,) xyz in base_link
        self._last_point_t = 0.0
        self._stab_buf: deque = deque(maxlen=64)   # (t, xyz)
        self._pick_xyz: Optional[np.ndarray] = None  # frozen at pick start
        self._tracked_xyz: Optional[np.ndarray] = None  # smoothed servo target
        self._last_tick_t: float = 0.0
        self._gripper: Optional[GripperClient] = None
        self._gripper_thread: Optional[threading.Thread] = None
        self._descend_done = threading.Event()
        self._close_done = threading.Event()
        self._retreat_done = threading.Event()

        # ── ROS I/O ───────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            String, str(g("object_label_topic").value), self._on_label, sensor_qos)
        self.create_subscription(
            PointStamped, str(g("object_point_topic").value),
            self._on_point, sensor_qos)
        self.create_subscription(
            Empty, "/sort_by_class/trigger", self._on_trigger, 10)

        self._pub_target = self.create_publisher(
            PoseStamped, str(g("servo_target_topic").value), 10)

        # Park the arm at scan_pose on startup via the Placo /go_to_joints
        # service.  This puts the camera in the canonical observation pose
        # so calibration / first pick begin from a known configuration.
        # Done in a thread so __init__ doesn't block on service availability.
        self._goto_joints = self.create_client(GoToJoints, "/go_to_joints")
        threading.Thread(target=self._park_at_scan_pose, daemon=True).start()

        self.create_timer(1.0 / self._servo_rate, self._tick)
        self.get_logger().info(
            f"TrackAndPick ready.  Publish to /sort_by_class/trigger to arm."
        )

    def _recover_to_scan_pose(self):
        """Sent on a worker thread when tracking is abandoned mid-pursuit."""
        if not self._goto_joints.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                "/go_to_joints unavailable — cannot recover to scan_pose"
            )
            return
        req = GoToJoints.Request()
        req.joint_names = list(SCAN_POSE_JOINTS.keys())
        req.positions = [float(v) for v in SCAN_POSE_JOINTS.values()]
        req.duration = 2.0
        self._goto_joints.call_async(req)
        # Don't block on the result — the timer is the primary control loop.

    def _park_at_scan_pose(self):
        if not self._goto_joints.wait_for_service(timeout_sec=20.0):
            self.get_logger().warn(
                "/go_to_joints unavailable — skipping initial scan_pose move"
            )
            return
        req = GoToJoints.Request()
        req.joint_names = list(SCAN_POSE_JOINTS.keys())
        req.positions = [float(v) for v in SCAN_POSE_JOINTS.values()]
        req.duration = 2.0
        future = self._goto_joints.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        result = future.result()
        if result is None or not result.success:
            self.get_logger().warn(
                f"Initial scan_pose move failed: "
                f"{(result.message if result else 'no response')}"
            )
        else:
            self.get_logger().info("Parked at scan_pose.")

    # ── Callbacks ────────────────────────────────────────────────────

    def _on_label(self, msg: String):
        now = time.time()
        with self._lock:
            self._last_label = msg.data
            self._last_label_t = now

    def _on_point(self, msg: PointStamped):
        now = time.time()
        xyz = np.array([msg.point.x, msg.point.y, msg.point.z])
        with self._lock:
            # bind to most recent label if accept_labels is non-empty
            if self._accept:
                if self._last_label is None or self._last_label == "none":
                    return
                if self._last_label not in self._accept:
                    return
                if now - self._last_label_t > 0.3:
                    return
            self._last_point = xyz
            self._last_point_t = now
            # Trim stability buffer to recent samples
            self._stab_buf.append((now, xyz))
            cutoff = now - self._stab_seconds * 2.0
            while self._stab_buf and self._stab_buf[0][0] < cutoff:
                self._stab_buf.popleft()

    def _on_trigger(self, _msg):
        with self._lock:
            if self._state not in (S_IDLE, S_DONE):
                self.get_logger().warn(
                    f"Trigger ignored: state={self._state}")
                return
            self._enter(S_TRACKING)
            self._pick_xyz = None
            self._tracked_xyz = None        # reset smoothed reference
            self._last_tick_t = time.time()
            self._stab_buf.clear()
            self._descend_done.clear()
            self._close_done.clear()
            self._retreat_done.clear()
            self.get_logger().info("ARMED — tracking…")

    # ── State machine tick ──────────────────────────────────────────

    def _enter(self, new_state: str):
        self.get_logger().info(f"State {self._state} → {new_state}")
        self._state = new_state
        self._state_t0 = time.time()

    def _tick(self):
        with self._lock:
            state = self._state

        if state == S_IDLE or state == S_DONE:
            return
        if state == S_TRACKING:
            self._tick_tracking()
        elif state == S_PICK:
            self._tick_pick_kickoff()
        elif state == S_DESCEND:
            self._tick_descend()
        elif state == S_CLOSE:
            self._tick_close()
        elif state == S_RETREAT:
            self._tick_retreat()

    # ── States ──────────────────────────────────────────────────────

    def _tick_tracking(self):
        now = time.time()
        with self._lock:
            pt = self._last_point
            pt_t = self._last_point_t
            buf = list(self._stab_buf)

        if pt is None or now - pt_t > self._fresh_s:
            # No fresh detection.  Decide between three behaviours based
            # on how long the detection has been missing:
            #   < lost_detection_timeout_s : hold the last servo target so
            #     the arm waits in place (object briefly occluded).
            #   ≥ lost_detection_timeout_s : abandon the pursuit, go back
            #     to scan_pose and disarm so the arm stops drifting.
            stale_for = (now - pt_t) if pt is not None else float("inf")
            if stale_for >= self._lost_timeout:
                self.get_logger().warn(
                    f"Detection lost for {stale_for:.1f}s — returning to "
                    "scan_pose and disarming."
                )
                self._tracked_xyz = None
                self._pick_xyz = None
                # Kick the recovery on a background thread so the timer
                # keeps running while the joint motion executes.
                threading.Thread(
                    target=self._recover_to_scan_pose, daemon=True).start()
                # Switch immediately so the timer stops publishing targets.
                self._enter(S_IDLE)
                return
            if self._tracked_xyz is not None:
                ps = make_pose_stamped(
                    self._tracked_xyz, topdown_quat_for(self._tracked_xyz))
                self._pub_target.publish(ps)
            self._last_tick_t = now
            return

        # Raw desired target (what we'd publish if we trusted every frame).
        # IMPORTANT: pick_z_offset_m is the grasp depth — it must NOT bleed
        # into the hover height, or the arm tries to sit *below* the
        # centroid while tracking.  Only X/Y offsets and approach_height
        # apply during hover; pick_z_off applies only at descent time
        # (see the stable-window snapshot below).
        raw = pt.copy()
        raw[0] += self._pick_x_off
        raw[1] += self._pick_y_off
        raw[2] += self._approach          # hover above the centroid

        # Initialise the tracked reference at the first valid detection.
        if self._tracked_xyz is None:
            self._tracked_xyz = raw.copy()
            self._last_tick_t = now
        else:
            # 1) Exponential-moving-average filter: smooths jitter.
            filtered = (self._lpf_alpha * raw
                        + (1.0 - self._lpf_alpha) * self._tracked_xyz)
            # 2) Velocity clamp: even if a frame is a big outlier, the
            #    reference cannot leap more than max_speed * dt per tick.
            dt = max(now - self._last_tick_t, 1e-3)
            self._last_tick_t = now
            delta = filtered - self._tracked_xyz
            step_norm = float(np.linalg.norm(delta))
            cap = self._max_speed * dt
            if step_norm > cap:
                delta = delta * (cap / step_norm)
            self._tracked_xyz = self._tracked_xyz + delta

        # Publish the smoothed reference so cartesian_motion_node servos
        # toward a pose that moves at most max_speed m/s.
        ref = self._tracked_xyz
        ps = make_pose_stamped(ref, topdown_quat_for(ref))
        self._pub_target.publish(ps)

        # Reachability check based on the object's reported position
        # (in base_link).  If the object is way outside the workspace
        # the arm should still hover at the best reachable pose Placo
        # picks, but we don't advance to PICK until the object is
        # inside our pickable zone.
        if not self._inside_pick_zone(pt):
            return

        # Stability over the last `stability_seconds`.
        if not self._is_stable(buf, now):
            return

        # Freeze the pick position at the centroid of the stable window
        # so the descent below is repeatable even if a stale frame comes in.
        recent = [b[1] for b in buf if now - b[0] <= self._stab_seconds]
        if len(recent) < 3:
            return
        frozen = np.mean(recent, axis=0)
        frozen[0] += self._pick_x_off
        frozen[1] += self._pick_y_off
        frozen[2] += self._pick_z_off
        with self._lock:
            self._pick_xyz = frozen
        self._enter(S_PICK)

    def _is_stable(self, buf, now: float) -> bool:
        recent = [(t, xyz) for (t, xyz) in buf
                  if now - t <= self._stab_seconds]
        if len(recent) < 5:
            return False
        # Must span at least stability_seconds
        if recent[-1][0] - recent[0][0] < 0.9 * self._stab_seconds:
            return False
        pts = np.array([xyz for _, xyz in recent])
        mean = pts.mean(axis=0)
        spread = float(np.max(np.linalg.norm(pts - mean, axis=1)))
        return spread <= self._stab_radius

    def _inside_pick_zone(self, xyz: np.ndarray) -> bool:
        r = float(np.hypot(xyz[0], xyz[1]))
        return (self._r_min <= r <= self._r_max and
                self._z_min <= xyz[2] <= self._z_max)

    def _tick_pick_kickoff(self):
        # Open gripper before descending.  Then move to S_DESCEND.
        self.get_logger().info("PICK: opening gripper, will descend…")
        threading.Thread(target=self._do_open_then_descend, daemon=True).start()

    def _do_open_then_descend(self):
        ok = self._gripper.set_position(GRIPPER_OPEN, timeout_s=2.0)
        if not ok:
            self.get_logger().warn("Gripper open returned non-success — continuing.")
        time.sleep(0.2)
        self._enter(S_DESCEND)

    def _tick_descend(self):
        # Publish a target at the pick pose (no hover offset) every tick;
        # let cartesian_motion_node servo there.  Move to CLOSE after a
        # fixed settle duration — we don't have FK locally to measure
        # convergence, so use time-based settle.
        with self._lock:
            pick = self._pick_xyz
        if pick is None:
            self._enter(S_TRACKING)
            return
        ps = make_pose_stamped(pick, topdown_quat_for(pick))
        self._pub_target.publish(ps)
        if time.time() - self._state_t0 >= self._descend_t + self._settle_t:
            self._enter(S_CLOSE)

    def _tick_close(self):
        # Fire gripper close once, then advance to RETREAT.
        with self._lock:
            already = self._close_done.is_set()
        if already:
            return
        self._close_done.set()
        threading.Thread(target=self._do_close_then_retreat, daemon=True).start()

    def _do_close_then_retreat(self):
        ok = self._gripper.set_position(GRIPPER_CLOSED, timeout_s=4.0)
        if not ok:
            self.get_logger().warn("Gripper close returned non-success — continuing.")
        time.sleep(0.3)
        self._enter(S_RETREAT)

    def _tick_retreat(self):
        with self._lock:
            pick = self._pick_xyz
        if pick is None:
            self._enter(S_DONE)
            return
        retreat = pick.copy()
        retreat[2] += self._retreat
        ps = make_pose_stamped(retreat, topdown_quat_for(retreat))
        self._pub_target.publish(ps)
        if time.time() - self._state_t0 >= self._retreat_t:
            self.get_logger().info("Cycle complete — back to IDLE on next trigger.")
            self._enter(S_DONE)

    # ── Wiring ──────────────────────────────────────────────────────

    def set_gripper(self, gripper: GripperClient):
        self._gripper = gripper


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    rclpy.init()
    try:
        node = TrackAndPick()
        gripper = GripperClient(node)
        node.set_gripper(gripper)
        rclpy.spin(node)
    except BaseException:
        sys.stderr.write("track_and_pick fatal:\n" + traceback.format_exc())
        sys.stderr.flush()
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
