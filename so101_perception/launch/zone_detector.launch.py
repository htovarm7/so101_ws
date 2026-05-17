"""Stand-alone launcher: RealSense D435 + zone_detector node.

Mostly useful for tuning HSV values against the real camera before plugging
the detector into the full pick-and-place pipeline.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    realsense_params = LaunchConfiguration("realsense_params")
    camera_namespace = LaunchConfiguration("camera_namespace")
    config_file      = LaunchConfiguration("config_file")

    default_rs_params = PathJoinSubstitution(
        [FindPackageShare("so101_perception"), "config", "realsense_d435.yaml"]
    )
    default_config = PathJoinSubstitution(
        [FindPackageShare("so101_perception"), "config", "zones_hsv.yaml"]
    )

    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="camera",
        namespace=camera_namespace,
        parameters=[realsense_params],
        output="screen",
        emulate_tty=True,
    )

    zone_node = Node(
        package="so101_perception",
        executable="zone_detector",
        name="zone_detector",
        parameters=[config_file],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([
        DeclareLaunchArgument("camera_namespace", default_value="camera"),
        DeclareLaunchArgument("realsense_params", default_value=default_rs_params),
        DeclareLaunchArgument("config_file",      default_value=default_config),
        realsense_node,
        zone_node,
    ])
