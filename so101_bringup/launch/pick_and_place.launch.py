"""Pick-and-place by colour zone.

Layers:
  - follower_moveit_demo (bringup + MoveIt + optional RViz)
  - cameras.launch.py with the RealSense D435 config
  - object_classifier   (multi-class HSV detection of objects)
  - zone_detector       (PINK = ZONE_A, ORANGE = ZONE_B drop-off sheets)
  - sort_by_class       (the actual pick-and-place node, MoveItPy)

Run:
  ros2 launch so101_bringup pick_and_place.launch.py \\
      hardware_type:=real \\
      joint_config_file:=/calibration/papu.json \\
      use_rviz:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def _setup(context):
    hardware_type     = LaunchConfiguration("hardware_type").perform(context)
    namespace         = LaunchConfiguration("namespace").perform(context)
    usb_port          = LaunchConfiguration("usb_port").perform(context)
    joint_config_file = LaunchConfiguration("joint_config_file").perform(context)
    use_rviz          = LaunchConfiguration("use_rviz").perform(context)
    use_cameras       = LaunchConfiguration("use_cameras").perform(context)
    use_perception    = LaunchConfiguration("use_perception").perform(context)

    pkg_bringup    = get_package_share_directory("so101_bringup")
    pkg_perception = get_package_share_directory("so101_perception")
    pkg_moveit_cfg = get_package_share_directory("so101_moveit_config")

    # ── Bringup + MoveIt + cameras (uses the canonical pattern) ─────────────
    cameras_cfg = os.path.join(
        pkg_bringup, "config", "cameras", "so101_cameras_realsense.yaml"
    )
    follower_moveit_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, "launch", "follower_moveit_demo.launch.py")
        ),
        launch_arguments={
            "hardware_type": hardware_type,
            "namespace": namespace,
            "usb_port": usb_port,
            "joint_config_file": joint_config_file,
            "use_cameras": use_cameras,
            "cameras_config_file": cameras_cfg,
            "use_rviz": use_rviz,
        }.items(),
    )

    # ── Perception (skippable for hardware-free testing) ────────────────────
    objects_hsv = os.path.join(pkg_perception, "config", "objects_hsv.yaml")
    zones_hsv   = os.path.join(pkg_perception, "config", "zones_hsv.yaml")

    perception_nodes = []
    if use_perception.lower() == "true":
        perception_nodes.append(Node(
            package="so101_perception",
            executable="object_classifier",
            name="object_classifier",
            # Load the YAML as parameters (topics + thresholds) AND pass the
            # path again so the classifier's internal loader can read the
            # `classes:` list — the loader doesn't go through the param API.
            parameters=[objects_hsv, {"config_file": objects_hsv}],
            output="screen",
            emulate_tty=True,
        ))
        # Don't re-broadcast the gripper->camera TF — object_classifier already
        # does, and duplicate static TFs spam the log.
        perception_nodes.append(Node(
            package="so101_perception",
            executable="zone_detector",
            name="zone_detector",
            parameters=[zones_hsv, {"publish_camera_tf": False}],
            output="screen",
            emulate_tty=True,
        ))

    # ── Pick-and-place node ─────────────────────────────────────────────────
    # MoveItPy needs the full MoveIt config dict as parameters.  Re-build it
    # here from the URDF/SRDF so the script's internal MoveItCpp can plan.
    xacro_path = os.path.join(
        get_package_share_directory("so101_description"), "urdf", "so101_arm.urdf.xacro"
    )
    moveit_config = (
        MoveItConfigsBuilder("so101_arm", package_name="so101_moveit_config")
        .robot_description(
            file_path=xacro_path,
            mappings={"variant": "follower", "use_ros2_control": "false"},
        )
        .robot_description_semantic()
        .robot_description_kinematics()
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .pilz_cartesian_limits(file_path="config/pilz_cartesian_limits.yaml")
        .joint_limits()
        .trajectory_execution(
            file_path="config/moveit_controllers.yaml",
            moveit_manage_controllers=False,
        )
        .moveit_cpp(
            file_path=os.path.join(pkg_moveit_cfg, "config", "moveit_py_config.yaml")
        )
        # OctoMap disabled for now — default voxel size made the obstacles
        # too coarse and blocked otherwise-valid pick approach poses.  Re-enable
        # once the resolution / padding are tuned (see sensors_3d.yaml).
        # .sensors_3d(
        #     file_path=os.path.join(pkg_moveit_cfg, "config", "sensors_3d.yaml")
        # )
        .to_moveit_configs()
    )

    pick_place_cfg = os.path.join(pkg_moveit_cfg, "config", "pick_and_place.yaml")
    pick_place_node = Node(
        name="sort_by_class",
        package="so101_moveit_config",
        executable="sort_by_class.py",
        output="screen",
        emulate_tty=True,
        parameters=[moveit_config.to_dict(), pick_place_cfg],
    )

    return [follower_moveit_demo, *perception_nodes, pick_place_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("hardware_type", default_value="real"),
            DeclareLaunchArgument("namespace", default_value="follower"),
            DeclareLaunchArgument("usb_port", default_value="/dev/ttyACM0"),
            DeclareLaunchArgument("joint_config_file", default_value=""),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument(
                "use_cameras", default_value="true",
                description="Launch the RealSense D435 driver (false for testing without camera).",
            ),
            DeclareLaunchArgument(
                "use_perception", default_value="true",
                description="Launch object_classifier + zone_detector "
                "(false to mock detections via ros2 topic pub).",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
