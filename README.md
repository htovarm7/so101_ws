# SO-101 ROS 2 + MoveIt

ROS 2 Jazzy workspace for the **SO-101** robot arm. Provides motion planning with MoveIt 2 and vision-based blue-object detection using a RealSense D435 — all running inside Docker.

## Prerequisites

- Docker with Compose v2 (`docker compose`)
- An X server (for RViz GUI)
- **For arm services:** USB Feetech arm at `/dev/ttyACM0` and a LeRobot calibration file
- **For perception services:** RealSense D435 plugged in via USB

---

## Quick start

```bash
# 1. Build the image (once)
./run.sh build
# or:  docker compose build

# 2. Run what you need
./run.sh moveit            # MoveIt + RViz (real arm, no camera)
./run.sh perception        # MoveIt + RealSense D435 + blue-object detector
./run.sh perception-only   # RealSense D435 + blue-object detector (no arm needed)
./run.sh shell             # interactive shell inside the container
```

You can also call Docker Compose directly — `run.sh` is just a thin convenience wrapper that handles `xhost`:

```bash
docker compose up moveit
docker compose up perception
docker compose up perception-only
docker compose run --rm shell
```

---

## Services

| Service | What it runs | Requires |
|---|---|---|
| `moveit` | SO-101 follower arm + MoveIt move_group + RViz | arm on `/dev/ttyACM0`, calibration file |
| `perception` | Full arm stack + RealSense D435 + blue-object detector + combined RViz | arm + RealSense D435 |
| `perception-only` | RealSense D435 + blue-object detector + RViz | RealSense D435 only |
| `shell` | Interactive bash inside the container | — |

All services share the same Docker image (`so101-moveit:latest`) built from `Dockerfile.moveit`.

---

## Packages

| Package | Purpose |
|---|---|
| `so101_description` | URDF/Xacro robot description and meshes |
| `so101_moveit_config` | MoveIt 2 configuration (SRDF, kinematics, planners, controllers) |
| `so101_bringup` | Top-level launch files for the real arm (`follower_moveit_demo`, etc.) |
| `so101_perception` | RealSense D435 driver + HSV blue-object detector + TF bridge to gripper |
| `so101_kinematics` | Cartesian motion and IK control nodes |
| `so101_kinematics_msgs` | Custom service definitions (`GoToPose`, `GoToJoints`) |
| `feetech_ros2_driver` | ros2_control hardware interface for Feetech STS3215 servos |

---

## Hardware setup

### 1. Udev rules (stable device names)

Create `/etc/udev/rules.d/99-so101.rules` for stable symlinks. Query your device IDs first:

```bash
udevadm info --query=property --name=/dev/ttyACM0 | grep -E 'ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL_SHORT|ID_PATH'
```

Example rule:

```
SUBSYSTEM=="tty", ENV{ID_VENDOR_ID}=="XXXX", ENV{ID_MODEL_ID}=="YYYY", \
  ENV{ID_SERIAL_SHORT}=="YOUR_SERIAL", SYMLINK+="so101_follower", \
  GROUP="dialout", MODE="0660"
```

Reload:

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 2. User permissions

```bash
sudo usermod -aG dialout,video $USER
# log out and back in
```

### 3. Passing the device into Docker

Docker does not resolve udev symlinks. Pass the real device node:

```bash
# find the real node
ls -la /dev/so101_follower   # shows e.g. -> ttyACM0
# docker-compose.yml already uses /dev/ttyACM0
```

If your device node differs from `/dev/ttyACM0`, override it:

```bash
docker compose run --rm \
  -e "" \
  moveit \
  ros2 launch so101_bringup follower_moveit_demo.launch.py \
    hardware_type:=real usb_port:=/dev/ttyACM1 \
    joint_config_file:=/calibration/papu.json use_rviz:=true
```

---

## Customising launch arguments

All services accept additional ROS 2 launch arguments. Override the default command inline:

```bash
# MoveIt without RViz
docker compose run --rm moveit \
  ros2 launch so101_bringup follower_moveit_demo.launch.py \
    hardware_type:=real usb_port:=/dev/ttyACM0 \
    joint_config_file:=/calibration/papu.json use_rviz:=false

# Perception-only without RViz
docker compose run --rm perception-only \
  ros2 launch so101_perception blue_detector.launch.py use_rviz:=false
```

---

## License

See [LICENSE](LICENSE).
