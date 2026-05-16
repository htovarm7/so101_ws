# Multi-class HSV object pipeline

Two nodes that extend `so101_perception` to
six-class colour classification.

## Files

```
so101_perception/
├── so101_perception/
│   ├── hsv_calibrator.py        
│   └── object_classifier.py    
├── launch/
│   ├── hsv_calibration.launch.py        
│   └── perception_classifier.launch.py 
├── config/
│   └── objects_hsv.yaml         
└── setup.py                     
```

## Workflow

### 1. Calibrate

```bash
ros2 launch so101_perception hsv_calibration.launch.py
```

Two OpenCV windows open: the live image with trackbars, and the mask
preview. Steps for each of the six objects:

1. Move the `class` trackbar to the slot you want to tune (0–5).
2. Hold the object in front of the camera.
3. Adjust H/S/V min/max trackbars until the mask cleanly covers the
   object and nothing else.
4. Press **s** to save that slot. The status bar at the bottom of the
   image flips that slot from `--` to `OK`.
5. Repeat for the other classes (press **n** / **p** to cycle, or just
   move the `class` trackbar).
6. Press **w** to write `config/objects_hsv.yaml`.
7. Press **q** to quit.

### 2. Detect

```bash
ros2 launch so101_perception perception_classifier.launch.py
```

This loads `config/objects_hsv.yaml`

### 3. Subscribe

```
/object_classifier/detected_label   std_msgs/String          (every frame)
/object_classifier/detected_point   geometry_msgs/PointStamped (when not "none")
/object_classifier/marker           visualization_msgs/Marker  (RViz)
/object_classifier/debug_image      sensor_msgs/Image          (annotated)
```

The label is always published exactly one of seven values per frame:
`red_heart_bear`, `blue_dragon`, `cereal_box`, `object_4`, `object_5`,
`object_6`, or `none`. The pick node can subscribe to
`detected_point` to get the 3-D centroid in `base_link`.

## Objects

| Class            | Largest contour area (px) | Mask coverage |
| ---------------- | ------------------------- | ------------- |
| `red_heart_bear` | 589,258                   | 29.4%         |
| `blue_dragon`    | 570,768                   | 29.3%         |


## Class-list customisation

To rename slots or change ordering, pass `class_labels` to the
calibrator:

```bash
ros2 run so101_perception hsv_calibrator --ros-args \
    -p class_labels:="['red_heart_bear','blue_dragon','green_apple', \
                       'yellow_duck','orange_block','purple_cup']"
```

The detector picks up whatever labels are in the YAML no code change
needed when you add the remaining four objects.
