"""Launch the RealSense D435 driver + blue-object detector + RViz."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_perception = get_package_share_directory("so101_perception")

    # ── Launch arguments ───────────────────────────────────────────────────────
    use_rviz        = LaunchConfiguration("use_rviz")
    camera_ns       = LaunchConfiguration("camera_namespace")
    realsense_params = LaunchConfiguration("realsense_params")
    detector_params  = LaunchConfiguration("detector_params")

    default_rs_params = PathJoinSubstitution(
        [FindPackageShare("so101_perception"), "config", "realsense_d435.yaml"]
    )
    default_det_params = PathJoinSubstitution(
        [FindPackageShare("so101_perception"), "config", "blue_detector_params.yaml"]
    )
    default_rviz = PathJoinSubstitution(
        [FindPackageShare("so101_perception"), "rviz", "blue_detector.rviz"]
    )

    # ── RealSense D435 driver ──────────────────────────────────────────────────
    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="camera",
        namespace=camera_ns,
        parameters=[realsense_params],
        output="screen",
        emulate_tty=True,
    )

    # ── Blue object detector ───────────────────────────────────────────────────
    detector_node = Node(
        package="so101_perception",
        executable="blue_object_detector",
        name="blue_object_detector",
        parameters=[detector_params],
        output="screen",
        emulate_tty=True,
    )

    # ── RViz ──────────────────────────────────────────────────────────────────
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", default_rviz],
        condition=IfCondition(use_rviz),
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_rviz",          default_value="true"),
        DeclareLaunchArgument("camera_namespace",  default_value="camera"),
        DeclareLaunchArgument("realsense_params",  default_value=default_rs_params),
        DeclareLaunchArgument("detector_params",   default_value=default_det_params),
        realsense_node,
        detector_node,
        rviz_node,
    ])
