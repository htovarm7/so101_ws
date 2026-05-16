"""Launch the RealSense D435 + the HSV calibrator GUI.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    realsense_params = LaunchConfiguration("realsense_params")
    camera_namespace = LaunchConfiguration("camera_namespace")
    output_path      = LaunchConfiguration("output_path")

    default_rs_params = PathJoinSubstitution(
        [FindPackageShare("so101_perception"), "config", "realsense_d435.yaml"]
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

    calibrator_node = Node(
        package="so101_perception",
        executable="hsv_calibrator",
        name="hsv_calibrator",
        parameters=[{"output_path": output_path}],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([
        DeclareLaunchArgument("camera_namespace", default_value="camera"),
        DeclareLaunchArgument("realsense_params", default_value=default_rs_params),
        DeclareLaunchArgument("output_path",      default_value=""),
        realsense_node,
        calibrator_node,
    ])
