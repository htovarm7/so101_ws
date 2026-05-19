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
- **`so101_perception`** — Vision nodes against the wrist-mounted RealSense D435. All share the same TF bridge (parent `moving_jaw_so101_v1_link` → `camera_color_optical_frame`) and publish detections in `base_link`. The camera node is namespaced as `cam_static` (topics live under `/camera/cam_static/...`) — the prefix is legacy naming; the camera is physically wrist-mounted and moves with the arm.
  - `blue_object_detector` — legacy single-class blue HSV detector (kept for `perception` / `perception-only` services).
  - `object_classifier` — multi-class HSV detector. Iterates the `classes` list from `config/objects_hsv.yaml`, picks the biggest contour over `min_contour_area`, deprojects to 3-D. Publishes `~/detected_label` (`std_msgs/String`) and `~/detected_point` (`PointStamped`). TF lookup uses `rclpy.time.Time()` (latest available) — using the image stamp causes "extrapolation into the future" errors because `/joint_states` lags slightly behind camera frames.
  - `hsv_calibrator` — OpenCV trackbar GUI; press `w` to overwrite `config/objects_hsv.yaml`.
  - `zone_detector` — finds the **pink (ZONE_A)** and **orange (ZONE_B)** drop-off sheets on every frame, publishes `~/zone_a` and `~/zone_b` (`PointStamped`) plus a `MarkerArray`. HSV thresholds in `config/zones_hsv.yaml`. Same Time(0) TF lookup pattern as `object_classifier`.
- **`so101_moveit_config`** — SRDF (groups `manipulator` 5-DOF + `gripper`; named states `rest`, `scan_pose`, `zero`, `extended`, gripper `open`/`closed`), kinematics (`pick-ik`), OMPL + Pilz, ros2_control controller config (`follower/arm_trajectory_controller` FJT + `follower/gripper_controller` ParallelGripperCommand). Provides `move_group.launch.py` and `moveit_rviz.launch.py` — these are *included* by `so101_bringup`, not run directly. Also ships `scripts/sort_by_class.py` (MoveItPy pick-and-place) and `config/pick_and_place.yaml` (tunables + class→zone mapping).
- **`so101_kinematics`** (ament_python) — IK / Cartesian control nodes that bypass MoveIt: `so101_ik_control_node` (live Viser gizmo → IK → joint commands), `so101_planned_control_node` (servo + planned-trajectory modes), `cartesian_motion_node` (service-based via `GoToPose`/`GoToJoints`). Uses **robokin** (Placo backend) — installed via pip, not apt. The `motion_planner.py` / `trajectory_executor.py` modules are reused libraries for these nodes.
- **`so101_kinematics_msgs`** (ament_cmake) — Holds `GoToPose.srv` and `GoToJoints.srv`. Lives separately because `rosidl_generate_interfaces` requires ament_cmake while `so101_kinematics` is ament_python.
- **`feetech_ros2_driver`** — Git submodule (branch `feat/joint-config-and-calibration`). The ros2_control hardware interface for the Feetech STS3215 servos. The Dockerfile clones it fresh during build rather than relying on `git submodule update`.

### Cross-cutting conventions

- `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` everywhere (set in `docker-compose.yml`).
- `SO101_URDF_FILE=/ros2_ws/so101_follower_real.urdf` is exported so launch files can use the pre-generated URDF instead of re-running xacro at runtime.
- Topics are typically under the `follower/` namespace, e.g. `/follower/joint_states` and `/follower/forward_controller/commands` — used by both the IK nodes and the perception TF bridge.
- The `perception` and `pick-and-place` services run `privileged: true` because the RealSense driver needs raw USB access; `moveit` does not.

### Pick-and-place stack specifics

The `pick-and-place` Docker service (and the `pick_and_place.launch.py` it runs) is the integration point for the whole vision-driven sorting demo. A few non-obvious wiring details:

- **Camera topic namespace.** RealSense is launched as `camera_namespace:=camera` `camera_name:=cam_static`, so the active topics are `/camera/cam_static/color/image_raw`, `/camera/cam_static/aligned_depth_to_color/image_raw`, `/camera/cam_static/color/camera_info`. The "static" in the name is historical — the camera moves with the wrist. Both `objects_hsv.yaml` and `zones_hsv.yaml` must point at these exact topics.
- **TF tree ownership.** The RealSense driver is configured with `publish_tf: false` (`so101_bringup/config/cameras/so101_realsense2.yaml`). Letting it publish its own `camera_link` tree creates a disconnected secondary TF tree — the perception nodes (`object_classifier`, `zone_detector`) are the sole publishers of `moving_jaw_so101_v1_link` → `camera_color_optical_frame`. If you ever see "two or more unconnected trees" in TF, check `publish_tf` first.
- **Wrist-mount calibration lives in YAML, not URDF.** The `mount_x/y/z` + `mount_qx/qy/qz/qw` fields in both `objects_hsv.yaml` and `zones_hsv.yaml` define the gripper→camera transform and **must be kept in sync** — they are NOT loaded from a shared file. Recalibrating means editing both.
- **Trigger mechanism.** `sort_by_class.py` does not auto-loop and does not read stdin (ros2 launch detaches stdin). Instead, it subscribes to `/sort_by_class/trigger` (`std_msgs/Empty`) and runs one pick cycle per message, ignoring incoming triggers while a cycle is in progress. Send one with `ros2 topic pub --once /sort_by_class/trigger std_msgs/Empty {}`.
- **Pick-time offsets compensate for mount-calibration error.** `pick_x_offset_m` / `pick_y_offset_m` in `so101_moveit_config/config/pick_and_place.yaml` are added to the detected centroid before commanding the grasp. These are translational fudge factors — if the gripper is consistently off by the same vector regardless of object position, tune them; if the error magnitude or direction changes with position, the issue is rotational and the camera mount quaternion needs recalibration instead.
- **Class → zone mapping format.** ROS 2 params can't be dicts, so `class_to_zone` is a flat list of `"label:zone"` strings (e.g. `"red_heart_bear:zone_a"`). Unmapped labels are skipped.
- **Fast config iteration (no rebuild).** The `pick-and-place` service in `docker-compose.yml` bind-mounts the three runtime YAMLs (`pick_and_place.yaml`, `objects_hsv.yaml`, `zones_hsv.yaml`) into the container's `install/` tree, so edits on the host take effect on the next restart. Only Python/C++/launch-file changes still require `./run.sh build`. **Important:** the mount only applies to *freshly-created* containers — if you edited `docker-compose.yml` to add a new mount, you must run `docker compose down` before `./run.sh pick-and-place`, otherwise the old container without the mount is reused.
