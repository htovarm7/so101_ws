#!/usr/bin/env python3
"""Pick a classified object and drop it on the matching colour zone.

High-level sequence:

  1. Move the arm to ``scan_pose`` so the wrist-mounted RealSense can see
     the workspace.
  2. Wait for a stable object detection (label + 3-D point) on
     ``/object_classifier``.
  3. Wait for stable detections of both zones on ``/zone_detector``.
  4. Pick the object (open → approach → LIN descend → close → LIN retreat).
  5. Return to ``scan_pose`` and **re-read** the destination zone so the
     drop succeeds even if the sheet was moved while the arm was busy.
  6. Place (approach → LIN descend → open → LIN retreat).
  7. Go to ``rest``.

Most thresholds live in ``pick_and_place.yaml``.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from collections import deque
from typing import Dict, Optional

import numpy as np
import rclpy
import rclpy.logging
from geometry_msgs.msg import Pose, PoseStamped, PointStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf_transformations import quaternion_from_euler


def _spin_node_in_thread(node: Node) -> tuple[threading.Thread, SingleThreadedExecutor]:
    """Spin ``node`` on a private SingleThreadedExecutor in a daemon thread.

    Required after MoveItPy is constructed: MoveItPy installs its own global
    executor, so a plain ``rclpy.spin(node)`` raises
    'Executor is already spinning'.
    """
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return thread, executor

ARM_GROUP = "manipulator"
GRIPPER_GROUP = "gripper"
EE_FRAME = "gripper_frame_link"
BASE_FRAME = "base_link"
SCAN_STATE = "scan_pose"
REST_STATE = "rest"


# ---------------------------------------------------------------------------
#  Pose helpers
# ---------------------------------------------------------------------------

def topdown_rpy_for(xyz_m) -> tuple[float, float, float]:
    """Top-down orientation with yaw aimed radially at the target.

    The SO-101 is 5-DOF — fixing yaw=0 makes most top-down poses
    unreachable.  ``atan2(y, x)`` is the natural redundancy resolution.
    """
    yaw = float(np.arctan2(xyz_m[1], xyz_m[0]))
    return (0.0, float(np.pi), yaw)


def make_pose_stamped(xyz_m, rpy_rad, frame_id: str = BASE_FRAME) -> PoseStamped:
    q = quaternion_from_euler(*rpy_rad)
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    ps.pose.position.x = float(xyz_m[0])
    ps.pose.position.y = float(xyz_m[1])
    ps.pose.position.z = float(xyz_m[2])
    ps.pose.orientation.x = q[0]
    ps.pose.orientation.y = q[1]
    ps.pose.orientation.z = q[2]
    ps.pose.orientation.w = q[3]
    return ps


# ---------------------------------------------------------------------------
#  Detection listeners
# ---------------------------------------------------------------------------

class ObjectListener(Node):
    """Buffers the most recent label and waits for a stable classified point.

    Solves the cross-topic race between ``String`` and ``PointStamped`` by
    keeping a short ring buffer of ``(label, recv_time)`` and binding a
    point to the most recent label seen inside ``label_window_s``.
    """

    def __init__(self, label_topic: str, point_topic: str,
                 samples_required: int, stability_radius_m: float,
                 label_window_s: float, timeout_s: float,
                 accept_labels: set[str]):
        super().__init__("sort_by_class_object_listener")
        self.samples_required = samples_required
        self.stability_radius_m = stability_radius_m
        self.label_window = label_window_s
        self.timeout_s = timeout_s
        self.accept_labels = accept_labels

        self._label_buf: deque[tuple[str, float]] = deque(maxlen=64)
        self._point_buf: list[tuple[str, np.ndarray]] = []
        self._frame_id: Optional[str] = None
        self._done = threading.Event()
        self._result: Optional[tuple[str, np.ndarray, str]] = None
        self._lock = threading.Lock()

        # RELIABLE matches both `ros2 topic pub` defaults and the real
        # object_classifier publishers; with BEST_EFFORT we saw the sub
        # silently fail to match across containers under CycloneDDS.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(String, label_topic, self._on_label, sensor_qos)
        self.create_subscription(PointStamped, point_topic, self._on_point,
                                 sensor_qos)
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
                self.get_logger().info(
                    f"Ignoring point for unmapped label '{label}'"
                )
                return
            if self._frame_id is None:
                self._frame_id = msg.header.frame_id
            elif msg.header.frame_id != self._frame_id:
                self.get_logger().warn(
                    f"Detection frame changed {self._frame_id} -> "
                    f"{msg.header.frame_id}; resetting buffer"
                )
                self._point_buf.clear()
                self._frame_id = msg.header.frame_id

            p = np.array([msg.point.x, msg.point.y, msg.point.z])
            # Reset the buffer whenever the label changes — we want a stable
            # detection of one object, not an average over two.
            if self._point_buf and self._point_buf[-1][0] != label:
                self._point_buf.clear()
            self._point_buf.append((label, p))
            if len(self._point_buf) > self.samples_required:
                self._point_buf.pop(0)
            if len(self._point_buf) < self.samples_required:
                return
            arr = np.stack([p for _, p in self._point_buf])
            mean = arr.mean(axis=0)
            if np.linalg.norm(arr - mean, axis=1).max() > self.stability_radius_m:
                return
            label_locked = self._point_buf[-1][0]
            self._result = (label_locked, mean, self._frame_id)
            self._done.set()
            self.get_logger().info(
                f"Stable detection: label='{label_locked}' "
                f"xyz=({mean[0]:+.3f}, {mean[1]:+.3f}, {mean[2]:.3f}) m"
                f" in '{self._frame_id}'"
            )

    def _label_for(self, now_s: float) -> Optional[str]:
        # Walk newest-first; return the latest label within the window.
        for label, t in reversed(self._label_buf):
            if now_s - t <= self.label_window:
                return label
            break
        return None

    def wait(self) -> Optional[tuple[str, np.ndarray, str]]:
        if self._done.wait(timeout=self.timeout_s):
            return self._result
        return None


class ZoneListener(Node):
    """Waits for a stable detection on a single zone topic."""

    def __init__(self, topic: str, samples_required: int,
                 stability_radius_m: float, timeout_s: float):
        super().__init__(f"sort_by_class_zone_listener_{os.urandom(2).hex()}")
        self.samples_required = samples_required
        self.stability_radius_m = stability_radius_m
        self.timeout_s = timeout_s
        self._buf: list[np.ndarray] = []
        self._frame_id: Optional[str] = None
        self._done = threading.Event()
        self._result: Optional[tuple[np.ndarray, str]] = None
        self._lock = threading.Lock()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(PointStamped, topic, self._on_point, sensor_qos)
        self.topic = topic

    def _on_point(self, msg: PointStamped) -> None:
        if self._done.is_set():
            return
        with self._lock:
            if self._frame_id is None:
                self._frame_id = msg.header.frame_id
            p = np.array([msg.point.x, msg.point.y, msg.point.z])
            self._buf.append(p)
            if len(self._buf) > self.samples_required:
                self._buf.pop(0)
            if len(self._buf) < self.samples_required:
                return
            arr = np.stack(self._buf)
            mean = arr.mean(axis=0)
            if np.linalg.norm(arr - mean, axis=1).max() <= self.stability_radius_m:
                self._result = (mean, self._frame_id)
                self._done.set()

    def reset(self) -> None:
        with self._lock:
            self._buf.clear()
            self._done.clear()
            self._result = None

    def wait(self) -> Optional[tuple[np.ndarray, str]]:
        if self._done.wait(timeout=self.timeout_s):
            return self._result
        return None


# ---------------------------------------------------------------------------
#  MoveIt helpers
# ---------------------------------------------------------------------------

def plan_and_execute(robot, planning_component, logger,
                     multi_plan_parameters=None) -> bool:
    logger.info("Planning…")
    if multi_plan_parameters is not None:
        result = planning_component.plan(
            multi_plan_parameters=multi_plan_parameters)
    else:
        result = planning_component.plan()
    if not result:
        logger.error("Planning failed")
        return False
    logger.info("Executing")
    robot.execute(result.trajectory, controllers=[])
    return True


def solve_ik_and_plan(robot, arm, logger, pose: Pose,
                      plan_params, ik_timeout: float = 0.2) -> bool:
    from moveit.core.robot_state import RobotState
    robot_model = robot.get_robot_model()
    robot_state = RobotState(robot_model)
    with robot.get_planning_scene_monitor().read_only() as scene:
        robot_state.set_joint_group_positions(
            ARM_GROUP,
            scene.current_state.get_joint_group_positions(ARM_GROUP),
        )
    robot_state.update()
    if not robot_state.set_from_ik(ARM_GROUP, pose, EE_FRAME, ik_timeout):
        logger.error(f"IK failed for pose {pose.position}")
        return False
    robot_state.update()
    arm.set_start_state_to_current_state()
    arm.set_goal_state(robot_state=robot_state)
    return plan_and_execute(robot, arm, logger,
                            multi_plan_parameters=plan_params)


def plan_linear_cartesian(robot, arm, logger,
                          pose_stamped: PoseStamped, plan_params) -> bool:
    arm.set_start_state_to_current_state()
    arm.set_goal_state(pose_stamped_msg=pose_stamped, pose_link=EE_FRAME)
    return plan_and_execute(robot, arm, logger,
                            multi_plan_parameters=plan_params)


def goto_named_state(robot, arm, logger, name: str, plan_params) -> bool:
    arm.set_start_state_to_current_state()
    arm.set_goal_state(configuration_name=name)
    return plan_and_execute(robot, arm, logger,
                            multi_plan_parameters=plan_params)


def set_gripper(robot, gripper, logger, named_state: str) -> bool:
    logger.info(f"── Gripper -> '{named_state}' ──")
    gripper.set_start_state_to_current_state()
    gripper.set_goal_state(configuration_name=named_state)
    return plan_and_execute(robot, gripper, logger)


# ---------------------------------------------------------------------------
#  Main config + flow
# ---------------------------------------------------------------------------

class Config:
    """Pull every tunable knob from the ROS parameter server."""

    def __init__(self, node: Node) -> None:
        d = node.declare_parameter
        d("object_label_topic", "/object_classifier/detected_label")
        d("object_point_topic", "/object_classifier/detected_point")
        d("zone_a_topic",       "/zone_detector/zone_a")
        d("zone_b_topic",       "/zone_detector/zone_b")

        # ROS 2 parameters can't be dicts; declare a flat list of
        # "label:zone" strings and reassemble below.
        d("class_to_zone", ["red_heart_bear:zone_a",
                            "blue_dragon:zone_b"])

        d("object_samples_required", 10)
        d("object_stability_radius_m", 0.01)
        d("object_detection_timeout_s", 30.0)
        d("zone_samples_required", 5)
        d("zone_stability_radius_m", 0.02)
        d("zone_detection_timeout_s", 15.0)
        d("label_point_window_s", 0.15)

        d("pick_z_offset_m",    0.0)
        d("approach_height_m",  0.05)
        d("retreat_height_m",   0.08)
        d("place_z_offset_m",   0.04)
        d("place_approach_height_m", 0.08)
        d("place_retreat_height_m",  0.08)

        g = node.get_parameter
        self.label_topic = g("object_label_topic").value
        self.point_topic = g("object_point_topic").value
        self.zone_topics = {
            "zone_a": g("zone_a_topic").value,
            "zone_b": g("zone_b_topic").value,
        }
        raw_map = g("class_to_zone").value or {}
        # ROS 2 typed-dict parameters arrive as a flat list of "key: value"
        # strings if declared without an explicit type — handle both.
        if isinstance(raw_map, list):
            parsed: Dict[str, str] = {}
            for item in raw_map:
                if isinstance(item, str) and ":" in item:
                    k, v = item.split(":", 1)
                    parsed[k.strip()] = v.strip()
            raw_map = parsed
        self.class_to_zone: Dict[str, str] = dict(raw_map)

        self.object_samples = int(g("object_samples_required").value)
        self.object_radius  = float(g("object_stability_radius_m").value)
        self.object_timeout = float(g("object_detection_timeout_s").value)
        self.zone_samples   = int(g("zone_samples_required").value)
        self.zone_radius    = float(g("zone_stability_radius_m").value)
        self.zone_timeout   = float(g("zone_detection_timeout_s").value)
        self.label_window   = float(g("label_point_window_s").value)

        self.pick_z      = float(g("pick_z_offset_m").value)
        self.approach    = float(g("approach_height_m").value)
        self.retreat     = float(g("retreat_height_m").value)
        self.place_z     = float(g("place_z_offset_m").value)
        self.place_app   = float(g("place_approach_height_m").value)
        self.place_ret   = float(g("place_retreat_height_m").value)


def detect_zone(topic: str, samples: int, radius: float,
                timeout: float, label: str, logger) -> Optional[np.ndarray]:
    listener = ZoneListener(topic, samples, radius, timeout)
    _, executor = _spin_node_in_thread(listener)
    logger.info(f"Waiting for stable {label} on {topic}…")
    out = listener.wait()
    executor.shutdown()
    listener.destroy_node()
    if out is None:
        logger.error(f"No stable {label} after {timeout:.0f} s")
        return None
    mean, _ = out
    logger.info(
        f"{label} centroid: ({mean[0]:+.3f}, {mean[1]:+.3f}, {mean[2]:.3f}) m"
    )
    return mean


def main() -> None:
    rclpy.init()
    logger = rclpy.logging.get_logger("sort_by_class")
    logger.info("sort_by_class starting…")

    try:
        # Load parameters via a tiny throwaway node so we don't have to
        # construct MoveItPy yet.
        param_node = Node("sort_by_class_params")
        cfg = Config(param_node)
        param_node.destroy_node()

        if not cfg.class_to_zone:
            logger.error("class_to_zone is empty — nothing to do")
            os._exit(1)
        accept_labels = set(cfg.class_to_zone.keys())

        # ── 1) Detect object ─────────────────────────────────────────────
        obj_listener = ObjectListener(
            cfg.label_topic, cfg.point_topic,
            cfg.object_samples, cfg.object_radius,
            cfg.label_window, cfg.object_timeout,
            accept_labels,
        )
        _, obj_executor = _spin_node_in_thread(obj_listener)
        obj_result = obj_listener.wait()
        obj_executor.shutdown()
        obj_listener.destroy_node()
        if obj_result is None:
            logger.error(
                f"No stable object detection after "
                f"{cfg.object_timeout:.0f} s — aborting."
            )
            os._exit(1)
        obj_label, obj_xyz, obj_frame = obj_result
        if obj_frame != BASE_FRAME:
            logger.warn(
                f"Object frame '{obj_frame}' != expected '{BASE_FRAME}'"
            )
        zone_name = cfg.class_to_zone.get(obj_label)
        if zone_name is None or zone_name not in cfg.zone_topics:
            logger.error(
                f"No zone mapped for label '{obj_label}' "
                f"(known: {sorted(cfg.class_to_zone)})"
            )
            os._exit(1)
        zone_topic = cfg.zone_topics[zone_name]
        logger.info(f"Object '{obj_label}' -> {zone_name} ({zone_topic})")

        # ── MoveItPy ─────────────────────────────────────────────────────
        logger.info("Creating MoveItPy…")
        from moveit.planning import MoveItPy, MultiPipelinePlanRequestParameters
        robot = MoveItPy(
            node_name="moveit_py_sort",
            remappings={"joint_states": "/follower/joint_states"},
        )
        logger.info("MoveItPy ready")
        arm     = robot.get_planning_component(ARM_GROUP)
        gripper = robot.get_planning_component(GRIPPER_GROUP)
        ompl     = MultiPipelinePlanRequestParameters(robot, ["ompl_rrtc"])
        pilz_lin = MultiPipelinePlanRequestParameters(robot, ["pilz_lin"])

        # ── 2) Detect both zones BEFORE the pick (sanity check) ──────────
        zone_xyz = detect_zone(
            zone_topic, cfg.zone_samples, cfg.zone_radius,
            cfg.zone_timeout, zone_name, logger,
        )
        if zone_xyz is None:
            os._exit(1)

        # ── 3) PICK ──────────────────────────────────────────────────────
        pick_xyz = obj_xyz.copy()
        pick_xyz[2] += cfg.pick_z
        approach_xyz = pick_xyz.copy()
        approach_xyz[2] = pick_xyz[2] + cfg.approach
        retreat_xyz = pick_xyz.copy()
        retreat_xyz[2] = pick_xyz[2] + cfg.retreat

        if not set_gripper(robot, gripper, logger, "open"):
            os._exit(1)
        time.sleep(0.3)

        logger.info(f"── Approach pick @ {approach_xyz} ──")
        approach_pose = make_pose_stamped(approach_xyz, topdown_rpy_for(pick_xyz))
        if not solve_ik_and_plan(robot, arm, logger, approach_pose.pose, ompl):
            os._exit(1)
        time.sleep(0.4)

        logger.info(f"── Descend LIN @ {pick_xyz} ──")
        pick_pose = make_pose_stamped(pick_xyz, topdown_rpy_for(pick_xyz))
        if not plan_linear_cartesian(robot, arm, logger, pick_pose, pilz_lin):
            os._exit(1)
        time.sleep(0.2)

        if not set_gripper(robot, gripper, logger, "closed"):
            os._exit(1)
        time.sleep(0.4)

        logger.info(f"── Retreat LIN @ {retreat_xyz} ──")
        retreat_pose = make_pose_stamped(retreat_xyz, topdown_rpy_for(pick_xyz))
        if not plan_linear_cartesian(robot, arm, logger, retreat_pose, pilz_lin):
            os._exit(1)
        time.sleep(0.3)

        # ── 4) Back to scan_pose and RE-detect the destination zone ──────
        logger.info(f"── Going to '{SCAN_STATE}' for fresh zone read ──")
        if not goto_named_state(robot, arm, logger, SCAN_STATE, ompl):
            os._exit(1)
        time.sleep(0.5)

        zone_xyz = detect_zone(
            zone_topic, cfg.zone_samples, cfg.zone_radius,
            cfg.zone_timeout, f"{zone_name} (fresh)", logger,
        )
        if zone_xyz is None:
            os._exit(1)

        # ── 5) PLACE ─────────────────────────────────────────────────────
        place_xyz = zone_xyz.copy()
        place_xyz[2] += cfg.place_z
        place_app = place_xyz.copy()
        place_app[2] = place_xyz[2] + cfg.place_app
        place_ret = place_xyz.copy()
        place_ret[2] = place_xyz[2] + cfg.place_ret

        logger.info(f"── Approach place @ {place_app} ──")
        place_app_pose = make_pose_stamped(place_app, topdown_rpy_for(place_xyz))
        if not solve_ik_and_plan(robot, arm, logger, place_app_pose.pose, ompl):
            os._exit(1)
        time.sleep(0.4)

        logger.info(f"── Place LIN @ {place_xyz} ──")
        place_pose = make_pose_stamped(place_xyz, topdown_rpy_for(place_xyz))
        if not plan_linear_cartesian(robot, arm, logger, place_pose, pilz_lin):
            os._exit(1)
        time.sleep(0.2)

        if not set_gripper(robot, gripper, logger, "open"):
            os._exit(1)
        time.sleep(0.4)

        logger.info(f"── Retreat LIN @ {place_ret} ──")
        place_ret_pose = make_pose_stamped(place_ret, topdown_rpy_for(place_xyz))
        if not plan_linear_cartesian(robot, arm, logger, place_ret_pose, pilz_lin):
            os._exit(1)
        time.sleep(0.3)

        # ── 6) Rest ──────────────────────────────────────────────────────
        logger.info(f"── Going to '{REST_STATE}' ──")
        goto_named_state(robot, arm, logger, REST_STATE, ompl)

        logger.info("Pick-and-place SUCCESS")
        try:
            rclpy.shutdown()
        except Exception:
            pass
        os._exit(0)

    except SystemExit:
        raise
    except BaseException:
        # Make absolutely sure the traceback reaches stderr — silent crashes
        # before the first logger call were what bit us on feat/pick-blue.
        sys.stderr.write("sort_by_class fatal:\n" + traceback.format_exc())
        sys.stderr.flush()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        os._exit(1)


if __name__ == "__main__":
    main()
