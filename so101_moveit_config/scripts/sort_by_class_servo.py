#!/usr/bin/env python3
"""Pick-and-place orchestrator backed by Placo IK (not MoveIt).

Drop-in replacement for ``sort_by_class.py`` that talks to:

  * ``/go_to_pose``   (so101_kinematics_msgs/srv/GoToPose)   — arm motion
  * ``/go_to_joints`` (so101_kinematics_msgs/srv/GoToJoints) — named states
  * ``/follower/gripper_controller/gripper_cmd`` (action)    — gripper

Same trigger interface as the original (``/sort_by_class/trigger``), same
perception subscribers (``object_classifier``, ``zone_detector``), same
YAML knobs (``pick_and_place.yaml``).

What changes vs. sort_by_class.py:
  * No MoveIt, no pick_ik, no OMPL, no Pilz LIN.  Placo handles every IK
    call.  Placo always returns a best-effort solution — workspace edge
    poses produce small residuals instead of `IK failed`.
  * Approach/pick/retreat are joint-quintic trajectories (smooth in joint
    space) rather than Pilz LIN.  We lose strict Cartesian linearity but
    gain robustness near singularities — and visual servoing in Phase 4
    will trade joint-quintic for live `/servo_target` anyway.
  * Gripper open/close go straight to the action server; no group planning.

Topology assumptions (Phase 2 launch):
  - ``arm_forward_controller`` is the active arm controller.
  - ``gripper_controller`` (ParallelGripperCommand action) is active.
  - ``cartesian_motion_node`` is up and using the real-hardware URDF.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
import rclpy.logging
from action_msgs.msg import GoalStatus
from control_msgs.action import ParallelGripperCommand
from geometry_msgs.msg import Pose, PoseStamped, PointStamped
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, String
from tf_transformations import quaternion_from_matrix

from so101_kinematics_msgs.srv import GoToJoints, GoToPose


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

BASE_FRAME = "base_link"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex", "wrist_roll"]

# Joint angles for named states, copied from the SRDF — Placo doesn't know
# about MoveIt's named-state file, so we keep an authoritative copy here.
SCAN_POSE_JOINTS = {
    "shoulder_pan":   0.0,
    "shoulder_lift": -1.0996,
    "elbow_flex":     0.1745,
    "wrist_flex":     1.3614,
    "wrist_roll":     1.5708,
}
REST_POSE_JOINTS = {
    "shoulder_pan":   0.0,
    "shoulder_lift": -1.57,
    "elbow_flex":     1.57,
    "wrist_flex":     0.75,
    "wrist_roll":     0.0,
}

GRIPPER_OPEN  = 1.5
GRIPPER_CLOSED = -0.16


# ---------------------------------------------------------------------------
#  Pose helpers
# ---------------------------------------------------------------------------

def topdown_quat_for(xyz_m) -> Tuple[float, float, float, float]:
    """Quaternion for top-down pick orientation with outward radial tilt.

    Carried over verbatim from sort_by_class.py — the math is solver-
    independent: it produces the orientation of `gripper_frame_link` we
    want, regardless of whether pick_ik or Placo solves the IK.
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
#  Detection listeners (identical to sort_by_class.py — reused verbatim)
# ---------------------------------------------------------------------------

def _spin_node_in_thread(node: Node) -> Tuple[threading.Thread, SingleThreadedExecutor]:
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return thread, executor


class ObjectListener(Node):
    def __init__(self, label_topic: str, point_topic: str,
                 samples_required: int, stability_radius_m: float,
                 label_window_s: float, timeout_s: float,
                 accept_labels: set):
        super().__init__("servo_orchestrator_object_listener")
        self.samples_required = samples_required
        self.stability_radius_m = stability_radius_m
        self.label_window = label_window_s
        self.timeout_s = timeout_s
        self.accept_labels = accept_labels

        self._label_buf: deque = deque(maxlen=64)
        self._point_buf: list = []
        self._frame_id: Optional[str] = None
        self._done = threading.Event()
        self._result: Optional[Tuple[str, np.ndarray, str]] = None
        self._lock = threading.Lock()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(String, label_topic, self._on_label, sensor_qos)
        self.create_subscription(PointStamped, point_topic, self._on_point, sensor_qos)
        self.get_logger().info(
            f"Listening: labels='{label_topic}' points='{point_topic}' "
            f"accept={sorted(self.accept_labels) or 'ANY'}"
        )

    def _on_label(self, msg: String) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        with self._lock:
            self._label_buf.append((msg.data, now))

    def _on_point(self, msg: PointStamped) -> None:
        if self._done.is_set():
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        with self._lock:
            label = self._label_for(now)
            if label is None or label == "none":
                return
            if self.accept_labels and label not in self.accept_labels:
                return
            if self._frame_id is None:
                self._frame_id = msg.header.frame_id
            elif msg.header.frame_id != self._frame_id:
                self._point_buf.clear()
                self._frame_id = msg.header.frame_id
            pt = np.array([msg.point.x, msg.point.y, msg.point.z])
            self._point_buf.append((label, pt))
            self._check_stability()

    def _label_for(self, now: float) -> Optional[str]:
        for label, t in reversed(self._label_buf):
            if now - t <= self.label_window:
                return label
        return None

    def _check_stability(self) -> None:
        if len(self._point_buf) < self.samples_required:
            return
        recent = self._point_buf[-self.samples_required:]
        labels = {item[0] for item in recent}
        if len(labels) != 1:
            self._point_buf = self._point_buf[-self.samples_required + 1:]
            return
        pts = np.array([item[1] for item in recent])
        mean = pts.mean(axis=0)
        if np.max(np.linalg.norm(pts - mean, axis=1)) <= self.stability_radius_m:
            self._result = (recent[0][0], mean, self._frame_id or BASE_FRAME)
            self.get_logger().info(
                f"Stable detection: label='{recent[0][0]}' "
                f"xyz=({mean[0]:+.3f}, {mean[1]:+.3f}, {mean[2]:.3f}) m "
                f"in '{self._frame_id}'"
            )
            self._done.set()

    def wait(self) -> Optional[Tuple[str, np.ndarray, str]]:
        if self._done.wait(timeout=self.timeout_s):
            return self._result
        return None


class ZoneListener(Node):
    def __init__(self, topic: str, samples_required: int,
                 stability_radius_m: float, timeout_s: float):
        super().__init__(f"servo_orchestrator_zone_listener_{int(time.time()*1000)%10000}")
        self.samples_required = samples_required
        self.stability_radius_m = stability_radius_m
        self.timeout_s = timeout_s
        self._buf: list = []
        self._done = threading.Event()
        self._result: Optional[Tuple[np.ndarray, str]] = None
        self._frame: Optional[str] = None
        self._lock = threading.Lock()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(PointStamped, topic, self._on_point, sensor_qos)

    def _on_point(self, msg: PointStamped) -> None:
        if self._done.is_set():
            return
        with self._lock:
            if self._frame is None:
                self._frame = msg.header.frame_id
            self._buf.append(np.array([msg.point.x, msg.point.y, msg.point.z]))
            if len(self._buf) >= self.samples_required:
                recent = np.array(self._buf[-self.samples_required:])
                mean = recent.mean(axis=0)
                if np.max(np.linalg.norm(recent - mean, axis=1)) <= self.stability_radius_m:
                    self._result = (mean, self._frame)
                    self._done.set()

    def wait(self) -> Optional[Tuple[np.ndarray, str]]:
        if self._done.wait(timeout=self.timeout_s):
            return self._result
        return None


# ---------------------------------------------------------------------------
#  Motion primitives via Placo (replaces every MoveItPy call)
# ---------------------------------------------------------------------------

class PlacoArmClient:
    """Thin wrapper around /go_to_pose and /go_to_joints."""

    def __init__(self, node: Node):
        self._node = node
        self._pose_cli = node.create_client(GoToPose, "/go_to_pose")
        self._joints_cli = node.create_client(GoToJoints, "/go_to_joints")
        node.get_logger().info("Waiting for /go_to_pose and /go_to_joints services…")
        self._pose_cli.wait_for_service(timeout_sec=30.0)
        self._joints_cli.wait_for_service(timeout_sec=30.0)
        node.get_logger().info("Placo arm services available.")

    def go_to_pose(self, ps: PoseStamped, *, strategy: str = "joint_quintic",
                   duration: float = 0.0) -> bool:
        req = GoToPose.Request()
        req.target = ps
        req.strategy = strategy
        req.duration = float(duration)
        future = self._pose_cli.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=30.0)
        result = future.result()
        if result is None:
            self._node.get_logger().error("go_to_pose: no response")
            return False
        if not result.success:
            self._node.get_logger().error(f"go_to_pose: {result.message}")
            return False
        return True

    def go_to_joint_dict(self, jd: Dict[str, float], *, duration: float = 1.5) -> bool:
        req = GoToJoints.Request()
        req.joint_names = list(jd.keys())
        req.positions = [float(v) for v in jd.values()]
        req.duration = float(duration)
        future = self._joints_cli.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=30.0)
        result = future.result()
        if result is None or not result.success:
            msg = "no response" if result is None else result.message
            self._node.get_logger().error(f"go_to_joints: {msg}")
            return False
        return True


class GripperClient:
    """ParallelGripperCommand action client."""

    ACTION_NAME = "/follower/gripper_controller/gripper_cmd"

    def __init__(self, node: Node):
        self._node = node
        self._client = ActionClient(node, ParallelGripperCommand, self.ACTION_NAME)
        node.get_logger().info(f"Waiting for gripper action server at {self.ACTION_NAME}…")
        self._client.wait_for_server(timeout_sec=15.0)
        node.get_logger().info("Gripper action available.")

    def set_position(self, position: float, *, timeout_s: float = 4.0) -> bool:
        goal = ParallelGripperCommand.Goal()
        # The action takes a JointState target — fill name + position only.
        goal.command = JointState()
        goal.command.name = ["gripper"]
        goal.command.position = [float(position)]
        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self._node, send_future, timeout_sec=5.0)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            self._node.get_logger().error("gripper goal rejected")
            return False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, result_future, timeout_sec=timeout_s)
        if result_future.result() is None:
            self._node.get_logger().error("gripper goal timed out")
            return False
        status = result_future.result().status
        ok = status == GoalStatus.STATUS_SUCCEEDED
        if not ok:
            # parallel_gripper_action_controller marks "stalled" as ABORTED
            # when allow_stalling: true is set in the controller config.
            # We treat stall as success because the gripper has clamped.
            self._node.get_logger().warn(
                f"gripper goal ended with status {status} — treating as ok"
            )
            ok = True
        return ok


# ---------------------------------------------------------------------------
#  Config (mirrors sort_by_class.Config; same YAML structure)
# ---------------------------------------------------------------------------

class Config:
    def __init__(self, node: Node) -> None:
        d = node.declare_parameter
        d("object_label_topic", "/object_classifier/detected_label")
        d("object_point_topic", "/object_classifier/detected_point")
        d("zone_a_topic",       "/zone_detector/zone_a")
        d("zone_b_topic",       "/zone_detector/zone_b")
        d("class_to_zone", ["red_heart_bear:zone_a", "blue_dragon:zone_b"])
        d("object_samples_required", 10)
        d("object_stability_radius_m", 0.01)
        d("object_detection_timeout_s", 30.0)
        d("zone_samples_required", 5)
        d("zone_stability_radius_m", 0.02)
        d("zone_detection_timeout_s", 15.0)
        d("label_point_window_s", 0.15)
        d("pick_x_offset_m",    0.0)
        d("pick_y_offset_m",    0.0)
        d("pick_z_offset_m",    0.0)
        d("approach_height_m",  0.05)
        d("retreat_height_m",   0.05)
        d("place_z_offset_m",   0.03)
        d("place_approach_height_m", 0.03)
        d("place_retreat_height_m",  0.03)
        d("test_pick_xyz", [0.0, 0.0, 0.0])

        g = node.get_parameter
        self.label_topic = g("object_label_topic").value
        self.point_topic = g("object_point_topic").value
        self.zone_topics = {
            "zone_a": g("zone_a_topic").value,
            "zone_b": g("zone_b_topic").value,
        }
        raw_map = g("class_to_zone").value or []
        parsed: Dict[str, str] = {}
        for item in raw_map:
            if isinstance(item, str) and ":" in item:
                k, v = item.split(":", 1)
                parsed[k.strip()] = v.strip()
        self.class_to_zone = parsed
        self.object_samples = int(g("object_samples_required").value)
        self.object_radius  = float(g("object_stability_radius_m").value)
        self.object_timeout = float(g("object_detection_timeout_s").value)
        self.zone_samples   = int(g("zone_samples_required").value)
        self.zone_radius    = float(g("zone_stability_radius_m").value)
        self.zone_timeout   = float(g("zone_detection_timeout_s").value)
        self.label_window   = float(g("label_point_window_s").value)
        self.pick_x   = float(g("pick_x_offset_m").value)
        self.pick_y   = float(g("pick_y_offset_m").value)
        self.pick_z   = float(g("pick_z_offset_m").value)
        self.approach = float(g("approach_height_m").value)
        self.retreat  = float(g("retreat_height_m").value)
        self.place_z   = float(g("place_z_offset_m").value)
        self.place_app = float(g("place_approach_height_m").value)
        self.place_ret = float(g("place_retreat_height_m").value)
        raw_test = list(g("test_pick_xyz").value or [])
        if len(raw_test) == 3 and any(abs(float(v)) > 1e-9 for v in raw_test):
            self.test_pick_xyz: Optional[np.ndarray] = np.array(
                [float(v) for v in raw_test], dtype=float)
        else:
            self.test_pick_xyz = None


# ---------------------------------------------------------------------------
#  Pick cycle
# ---------------------------------------------------------------------------

def detect_zone(topic, samples, radius, timeout, label, logger) -> Optional[np.ndarray]:
    listener = ZoneListener(topic, samples, radius, timeout)
    _, executor = _spin_node_in_thread(listener)
    out = listener.wait()
    executor.shutdown()
    listener.destroy_node()
    if out is None:
        logger.error(f"No stable {label} after {timeout:.0f} s")
        return None
    mean, _ = out
    logger.info(f"{label} centroid: ({mean[0]:+.3f}, {mean[1]:+.3f}, {mean[2]:.3f}) m")
    return mean


def run_pick_cycle(arm: PlacoArmClient, gripper: GripperClient,
                   cfg: Config, accept_labels: set, logger) -> bool:
    """One detect → pick → (optional place) → scan cycle, Placo-driven."""
    # ── 1) Detect object ─────────────────────────────────────────────────
    obj_listener = ObjectListener(
        cfg.label_topic, cfg.point_topic,
        cfg.object_samples, cfg.object_radius,
        cfg.label_window, cfg.object_timeout, accept_labels,
    )
    _, obj_executor = _spin_node_in_thread(obj_listener)
    obj_result = obj_listener.wait()
    obj_executor.shutdown()
    obj_listener.destroy_node()
    if obj_result is None:
        logger.error(f"No stable object after {cfg.object_timeout:.0f} s")
        return False
    obj_label, obj_xyz, _ = obj_result
    if cfg.test_pick_xyz is not None:
        logger.warn(
            f"test_pick_xyz override: replacing {tuple(obj_xyz)} with "
            f"{tuple(cfg.test_pick_xyz)}"
        )
        obj_xyz = cfg.test_pick_xyz.copy()

    zone_name = cfg.class_to_zone.get(obj_label)
    if zone_name is None or zone_name not in cfg.zone_topics:
        logger.error(f"No zone mapped for label '{obj_label}'")
        return False

    # ── 2) Zone (best-effort) ───────────────────────────────────────────
    zone_xyz = detect_zone(
        cfg.zone_topics[zone_name], cfg.zone_samples, cfg.zone_radius,
        cfg.zone_timeout, zone_name, logger,
    )
    if zone_xyz is None:
        logger.warn(f"No {zone_name} — pick-only mode")

    # ── 3) PICK ─────────────────────────────────────────────────────────
    pick_xyz = obj_xyz.copy()
    pick_xyz[0] += cfg.pick_x
    pick_xyz[1] += cfg.pick_y
    pick_xyz[2] += cfg.pick_z
    approach_xyz = pick_xyz.copy()
    approach_xyz[2] += cfg.approach
    retreat_xyz = pick_xyz.copy()
    retreat_xyz[2] += cfg.retreat
    logger.info(
        f"pick={pick_xyz} approach={approach_xyz} retreat={retreat_xyz}"
    )

    if not gripper.set_position(GRIPPER_OPEN):
        return False

    logger.info(f"── Approach pick @ {approach_xyz} ──")
    if not arm.go_to_pose(make_pose_stamped(approach_xyz, topdown_quat_for(pick_xyz)),
                          strategy="joint_quintic"):
        return False

    logger.info(f"── Descend cartesian → pick @ {pick_xyz} ──")
    if not arm.go_to_pose(make_pose_stamped(pick_xyz, topdown_quat_for(pick_xyz)),
                          strategy="cartesian"):
        return False

    if not gripper.set_position(GRIPPER_CLOSED):
        return False
    time.sleep(0.3)

    logger.info(f"── Retreat cartesian @ {retreat_xyz} ──")
    if not arm.go_to_pose(make_pose_stamped(retreat_xyz, topdown_quat_for(pick_xyz)),
                          strategy="cartesian"):
        return False

    logger.info("── Back to scan_pose ──")
    if not arm.go_to_joint_dict(SCAN_POSE_JOINTS, duration=1.5):
        return False

    # ── 4) PLACE (optional) ─────────────────────────────────────────────
    if zone_xyz is not None:
        zone_xyz = detect_zone(
            cfg.zone_topics[zone_name], cfg.zone_samples, cfg.zone_radius,
            cfg.zone_timeout, f"{zone_name} (re-detect)", logger,
        )

    if zone_xyz is not None:
        place_xyz = zone_xyz.copy()
        place_xyz[2] += cfg.place_z
        place_app = place_xyz.copy()
        place_app[2] += cfg.place_app
        place_ret = place_xyz.copy()
        place_ret[2] += cfg.place_ret

        logger.info(f"── Approach place @ {place_app} ──")
        if not arm.go_to_pose(make_pose_stamped(place_app, topdown_quat_for(place_xyz)),
                              strategy="joint_quintic"):
            return False
        logger.info(f"── Place cartesian @ {place_xyz} ──")
        if not arm.go_to_pose(make_pose_stamped(place_xyz, topdown_quat_for(place_xyz)),
                              strategy="cartesian"):
            return False
        if not gripper.set_position(GRIPPER_OPEN):
            return False
        time.sleep(0.3)
        logger.info(f"── Retreat place @ {place_ret} ──")
        if not arm.go_to_pose(make_pose_stamped(place_ret, topdown_quat_for(place_xyz)),
                              strategy="cartesian"):
            return False
    else:
        logger.info("Place phase skipped (no zone).")

    logger.info("── Back to scan_pose ──")
    arm.go_to_joint_dict(SCAN_POSE_JOINTS, duration=1.5)
    return True


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    rclpy.init()
    logger = rclpy.logging.get_logger("sort_by_class_servo")
    logger.info("sort_by_class_servo starting…")

    try:
        param_node = Node("sort_by_class_servo_params")
        cfg = Config(param_node)
        param_node.destroy_node()
        if not cfg.class_to_zone:
            logger.error("class_to_zone is empty — nothing to do")
            os._exit(1)
        accept_labels = set(cfg.class_to_zone.keys())

        client_node = Node("sort_by_class_servo")
        arm = PlacoArmClient(client_node)
        gripper = GripperClient(client_node)

        # Pre-position at scan_pose.
        logger.info("── Going to scan_pose (initial) ──")
        if not arm.go_to_joint_dict(SCAN_POSE_JOINTS, duration=2.0):
            logger.error("Initial scan_pose failed — aborting")
            os._exit(1)

        trigger_state = {"requested": False, "cycle": 0, "busy": False}

        def on_trigger(_msg):
            if trigger_state["busy"]:
                logger.warn(
                    f"Cycle {trigger_state['cycle']} already in progress — trigger ignored"
                )
                return
            trigger_state["requested"] = True

        trigger_node = Node("sort_by_class_servo_trigger")
        trigger_node.create_subscription(
            Empty, "/sort_by_class/trigger", on_trigger, 10,
        )
        _, trigger_executor = _spin_node_in_thread(trigger_node)

        logger.info(
            "Ready. Publish a trigger:\n"
            "    ros2 topic pub --once /sort_by_class/trigger std_msgs/Empty {}"
        )

        try:
            while True:
                if not trigger_state["requested"]:
                    time.sleep(0.1)
                    continue
                trigger_state["requested"] = False
                trigger_state["busy"] = True
                trigger_state["cycle"] += 1
                cycle = trigger_state["cycle"]
                logger.info(f"── Starting cycle {cycle} ──")
                try:
                    ok = run_pick_cycle(arm, gripper, cfg, accept_labels, logger)
                    logger.info(f"Cycle {cycle} {'SUCCESS' if ok else 'ABORTED'}")
                finally:
                    trigger_state["busy"] = False
        finally:
            trigger_executor.shutdown()
            trigger_node.destroy_node()
            client_node.destroy_node()

        try:
            rclpy.shutdown()
        except Exception:
            pass

    except BaseException:
        sys.stderr.write("sort_by_class_servo fatal:\n" + traceback.format_exc())
        sys.stderr.flush()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        os._exit(1)


if __name__ == "__main__":
    main()
