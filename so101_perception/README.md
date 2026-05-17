# SO-101 Perception — HSV Object Classification

Color-based detection and 3D localization of up to 6 objects using a RealSense
D435i. The pipeline classifies each object by HSV color range and publishes its
3D centroid for a downstream pick node to consume.

## Overview

Two ROS 2 nodes plus a standalone host script:

| Component | Purpose | Where it runs |
|---|---|---|
| `hsv_calibrator_standalone.py` | Tune HSV ranges with live trackbars | Host (no Docker) |
| `hsv_calibrator` (ROS node) | Same job, in ROS — adjust via `ros2 param set` | Container |
| `object_classifier` (ROS node) | Live detection at runtime | Container |

Outputs:

| Topic | Type | Description |
|---|---|---|
| `/object_classifier/detected_label` | `std_msgs/String` | Class name or `none` (every frame) |
| `/object_classifier/detected_point` | `geometry_msgs/PointStamped` | 3D centroid in `base_link` (when detected) |
| `/object_classifier/marker` | `visualization_msgs/Marker` | RViz sphere |
| `/object_classifier/debug_image` | `sensor_msgs/Image` | Annotated camera feed |

---

## Workflow 1 — Standalone HSV calibration 

Runs purely on the host with `pyrealsense2`. No Docker, no ROS. Best for
finding good HSV ranges quickly with live OpenCV trackbars.

```bash
pip install --break-system-packages pyrealsense2 opencv-python numpy pyyaml
```

### Run

```bash

python3 hsv_calibrator_standalone.py \
    --output ~/Downloads/so101_ws/so101_perception/config/objects_hsv.yaml
```

## Workflow 2 — ROS-based HSV calibration

In **terminal 1** — camera:

```bash
docker compose run --rm perception-only bash

# Inside container:
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash
ros2 launch realsense2_camera rs_launch.py \
    enable_color:=true \
    rgb_camera.color_profile:='640x480x30'
```

Wait for `RealSense Node Is Up!`.

In **terminal 2** — calibrator:

```bash
docker exec -it $(docker ps -q --filter "ancestor=so101-moveit:latest") bash
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash
ros2 run so101_perception hsv_calibrator
```

In **terminal 3** — visual preview:

```bash
docker exec -it $(docker ps -q --filter "ancestor=so101-moveit:latest") bash
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash

# Install rqt if not present 
apt update && apt install -y ros-jazzy-rqt ros-jazzy-rqt-image-view

rqt --standalone rqt_image_view
```
---

## Workflow 3 — Run live detection

Reads the YAML from workflow 1 or 2 and starts classifying.

### Start the container

```bash
cd ~/Downloads/so101_ws
docker compose run --rm perception-only bash
```

### Terminal 1 — camera

```bash
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash
ros2 launch realsense2_camera rs_launch.py \
    enable_color:=true \
    enable_depth:=true \
    align_depth.enable:=true \
    rgb_camera.color_profile:='640x480x30'
```

Wait for `RealSense Node Is Up!`.

### Terminal 2 — classifier

```bash
docker exec -it $(docker ps -q --filter "ancestor=so101-moveit:latest") bash
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash

# If the robot is not running, target_frame:=camera_color_optical_frame
# avoids "base_link does not exist" TF warnings.
ros2 run so101_perception object_classifier --ros-args \
    -p config_file:=/ros2_ws/src/so101_ros/so101_perception/config/objects_hsv.yaml \
    -p target_frame:=camera_color_optical_frame
```

### Terminal 3 — watch detection

Pick whichever view you want:

```bash
docker exec -it $(docker ps -q --filter "ancestor=so101-moveit:latest") bash
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash

# Text — labels stream every frame
ros2 topic echo /object_classifier/detected_label

# Text — 3D centroids when detected
ros2 topic echo /object_classifier/detected_point

# Visual — annotated camera feed
rqt --standalone rqt_image_view
# In dropdown: /object_classifier/debug_image
```

---

## Consuming the data (pick node integration)

The downstream pick node subscribes to two topics, default ROS 2 QoS:

```python
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped

self.create_subscription(String,       '/object_classifier/detected_label',
                         self._label_cb, 10)
self.create_subscription(PointStamped, '/object_classifier/detected_point',
                         self._point_cb, 10)
```

- `detected_label` fires every frame; `data` is the class name or `none`.
- `detected_point` only fires when an object is detected.
- The point is already in `base_link` feed it straight into MoveIt.


---

### Fast commands to update

```bash
# On host
cd ~/Downloads/so101_ws
git pull

# In container — rebuild from new source
cd /ros2_ws
colcon build --packages-select so101_perception
source install/setup.bash
```

---

## File layout

```
so101_ws/
├── docker-compose.yml                       # mounts ./:/ros2_ws/src/so101_ros
├── so101_perception/
│   ├── package.xml
│   ├── setup.py                             # entry_points for hsv_calibrator and object_classifier
│   ├── so101_perception/
│   │   ├── hsv_calibrator.py                # ROS node
│   │   ├── object_classifier.py             # ROS node
│   │   └── blue_object_detector.py          # original, kept for reference
│   ├── config/
│   │   ├── objects_hsv.yaml                 # HSV ranges, written by calibrator, read by classifier
│   │   └── realsense_d435.yaml              # camera launch params
│   └── launch/
│       ├── hsv_calibration.launch.py        # camera + calibrator together
│       └── perception_classifier.launch.py  # camera + classifier together
└── hsv_calibrator_standalone.py             # host-only tuning helper
```

---

## ROS topic and parameter cheat sheet

### Topics

| Topic | Type | Producer |
|---|---|---|
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` | RealSense |
| `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | RealSense |
| `/camera/camera/color/camera_info` | `sensor_msgs/CameraInfo` | RealSense |
| `/hsv_calibrator/debug_image` | `sensor_msgs/Image` | calibrator |
| `/object_classifier/detected_label` | `std_msgs/String` | classifier |
| `/object_classifier/detected_point` | `geometry_msgs/PointStamped` | classifier |
| `/object_classifier/marker` | `visualization_msgs/Marker` | classifier |
| `/object_classifier/debug_image` | `sensor_msgs/Image` | classifier |

### Calibrator parameters (`/hsv_calibrator`)

| Parameter | Type | Notes |
|---|---|---|
| `active_class` | int | 0-5, which slot is being tuned |
| `h_min`, `h_max` | int | 0-179. Set min > max for hue wrap-around |
| `s_min`, `s_max` | int | 0-255 |
| `v_min`, `v_max` | int | 0-255 |
| `color_image_topic` | str | Default `/camera/camera/color/image_raw` |
| `output_path` | str | Where to write the YAML |
| `min_contour_area` | int | Default 500 |

Calibrator services:

| Service | Type | Action |
|---|---|---|
| `/hsv_calibrator/save_class` | `std_srvs/Trigger` | Save current bounds to active slot |
| `/hsv_calibrator/reset_class` | `std_srvs/Trigger` | Clear active slot |
| `/hsv_calibrator/write_yaml` | `std_srvs/Trigger` | Write YAML |

### Classifier parameters (`/object_classifier`)

| Parameter | Type | Default |
|---|---|---|
| `config_file` | str | Resolved from package share dir |
| `target_frame` | str | `base_link` |
| `parent_link` | str | `moving_jaw_so101_v1_link` |
| `color_image_topic` | str | `/camera/camera/color/image_raw` |
| `depth_image_topic` | str | `/camera/camera/aligned_depth_to_color/image_raw` |
| `camera_info_topic` | str | `/camera/camera/color/camera_info` |
| `mount_x/y/z`, `mount_qx/qy/qz/qw` | float | Camera mount on gripper |
| `depth_scale` | float | 0.001 (mm to m) |
| `max_depth_m` | float | 3.0 |
| `none_label` | str | `none` |