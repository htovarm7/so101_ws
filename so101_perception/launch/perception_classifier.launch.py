"""Launch the RealSense D435 + the multi-class object classifier.
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
        [FindPackageShare("so101_perception"), "config", "objects_hsv.yaml"]
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

    classifier_node = Node(
        package="so101_perception",
        executable="object_classifier",
        name="object_classifier",
        parameters=[{"config_file": config_file}],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([
        DeclareLaunchArgument("camera_namespace", default_value="camera"),
        DeclareLaunchArgument("realsense_params", default_value=default_rs_params),
        DeclareLaunchArgument("config_file",      default_value=default_config),
        realsense_node,
        classifier_node,
    ])
