# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ROS 2 **Jazzy** workspace for the **SO-101** arm. Provides MoveIt 2 motion planning, custom IK control, and RealSense-based blue-object perception. The canonical run environment is Docker (image `so101-moveit:latest` built from `Dockerfile.moveit`); the root directory itself is *not* a colcon workspace — packages are copied into `/ros2_ws/src/so101_ros/` inside the container at image build time.

## Common commands

Everything runs inside the Docker image. `run.sh` is a thin wrapper around `docker compose` that also runs `xhost +local:docker` for RViz.

```bash
./run.sh build            # build the Docker image (so101-moveit:latest)
./run.sh moveit           # MoveIt + RViz + real arm (no camera)
./run.sh perception       # arm + MoveIt + RealSense D435 + blue detector
./run.sh perception-only  # RealSense D435 + blue detector (no arm)
./run.sh pick-and-place   # full stack: MoveIt + RealSense + classifier + zone detector + sort_by_class
./run.sh shell            # interactive bash inside the container
```

The image build (`Dockerfile.moveit`) does the colcon build for all packages and pre-generates a URDF (`/ros2_ws/so101_follower_real.urdf`) for `hardware_type:=real`, `usb_port:=/dev/ttyACM0`, `joint_config_file:=/calibration/papu.json`. Any code change to a package requires rebuilding the image (`./run.sh build`) — there is no host-side colcon build wired up.

To rebuild a single package inside a running shell (`./run.sh shell`):

```bash
source /opt/ros/jazzy/setup.bash
cd /ros2_ws
colcon build --packages-select <pkg_name> --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

To override launch arguments without editing `docker-compose.yml`, pass a full `ros2 launch` command to `docker compose run --rm <service>` (see README "Customising launch arguments").

Hardware assumes a Feetech arm at `/dev/ttyACM0` and a LeRobot calibration JSON mounted at `/calibration/papu.json` (host file: `papu_ros2.json`).

## Architecture

The stack is layered top-down: a bringup launch composes hardware, MoveIt, perception, and IK control nodes; each layer lives in its own ament package.

- **`so101_description`** — URDF/Xacro. The single source of truth is `urdf/so101_arm.urdf.xacro`, parameterised by `variant`, `use_ros2_control`, `hardware_type` (`real` | `mujoco`), `usb_port`, `joint_config_file`. The pre-generated URDF in the Docker image bakes these in for real hardware.
- **`so101_bringup`** — Top-level orchestration. `follower_moveit_demo.launch.py` is the canonical entry point: it includes `follower_split.launch.py` (ros2_control + robot_state_publisher + spawners), optional `cameras.launch.py`, then MoveIt's `move_group.launch.py` and `moveit_rviz.launch.py`. Most launch files accept `hardware_type`, `usb_port`, `joint_config_file`, `use_rviz`.
- **`so101_perception`** — Vision nodes against the wrist-mounted RealSense D435. All share the same TF bridge (parent `moving_jaw_so101_v1_link` → `camera_color_optical_frame`) and publish detections in `base_link`.
  - `blue_object_detector` — legacy single-class blue HSV detector (kept for `perception` / `perception-only` services).
  - `object_classifier` — multi-class HSV detector. Iterates the `classes` list from `config/objects_hsv.yaml`, picks the biggest contour over `min_contour_area`, deprojects to 3-D. Publishes `~/detected_label` (`std_msgs/String`) and `~/detected_point` (`PointStamped`).
  - `hsv_calibrator` — OpenCV trackbar GUI; press `w` to overwrite `config/objects_hsv.yaml`.
  - `zone_detector` — finds the **pink (ZONE_A)** and **orange (ZONE_B)** drop-off sheets on every frame, publishes `~/zone_a` and `~/zone_b` (`PointStamped`) plus a `MarkerArray`. HSV thresholds in `config/zones_hsv.yaml`.
- **`so101_moveit_config`** — SRDF (groups `manipulator` 5-DOF + `gripper`; named states `rest`, `scan_pose`, `zero`, `extended`, gripper `open`/`closed`), kinematics (`pick-ik`), OMPL + Pilz, ros2_control controller config (`follower/arm_trajectory_controller` FJT + `follower/gripper_controller` ParallelGripperCommand). Provides `move_group.launch.py` and `moveit_rviz.launch.py` — these are *included* by `so101_bringup`, not run directly. Also ships `scripts/sort_by_class.py` (MoveItPy pick-and-place) and `config/pick_and_place.yaml` (tunables + class→zone mapping).
- **`so101_kinematics`** (ament_python) — IK / Cartesian control nodes that bypass MoveIt: `so101_ik_control_node` (live Viser gizmo → IK → joint commands), `so101_planned_control_node` (servo + planned-trajectory modes), `cartesian_motion_node` (service-based via `GoToPose`/`GoToJoints`). Uses **robokin** (Placo backend) — installed via pip, not apt. The `motion_planner.py` / `trajectory_executor.py` modules are reused libraries for these nodes.
- **`so101_kinematics_msgs`** (ament_cmake) — Holds `GoToPose.srv` and `GoToJoints.srv`. Lives separately because `rosidl_generate_interfaces` requires ament_cmake while `so101_kinematics` is ament_python.
- **`feetech_ros2_driver`** — Git submodule (branch `feat/joint-config-and-calibration`). The ros2_control hardware interface for the Feetech STS3215 servos. The Dockerfile clones it fresh during build rather than relying on `git submodule update`.

### Cross-cutting conventions

- `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` everywhere (set in `docker-compose.yml`).
- `SO101_URDF_FILE=/ros2_ws/so101_follower_real.urdf` is exported so launch files can use the pre-generated URDF instead of re-running xacro at runtime.
- Topics are typically under the `follower/` namespace, e.g. `/follower/joint_states` and `/follower/forward_controller/commands` — used by both the IK nodes and the perception TF bridge.
- The `perception` service runs `privileged: true` because the RealSense driver needs raw USB access; `moveit` does not.
