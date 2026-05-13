"""Combined launch: SO-101 arm (real hw) + RealSense D435 + blue-object detector.

TF bridge
---------
The follower_moveit_demo stack (follower_split) runs robot_state_publisher
without a frame_prefix, so all robot frames are bare names (e.g.
``moving_jaw_so101_v1_link``).

We publish one static TF that connects the gripper to the RealSense body:

    moving_jaw_so101_v1_link  ──(static)──>  camera_link

The RealSense driver then publishes the rest of its chain automatically:
    camera_link → camera_color_frame → camera_color_optical_frame

Full path from robot base to the detected-object frame:
    base_link → … → moving_jaw_so101_v1_link → camera_link
                                              → camera_color_optical_frame

RViz shows the live robot model and the blue-sphere marker in the same scene.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_perception = get_package_share_directory("so101_perception")
    pkg_bringup    = get_package_share_directory("so101_bringup")

    # ── Launch arguments ───────────────────────────────────────────────────────
    hardware_type     = LaunchConfiguration("hardware_type")
    usb_port          = LaunchConfiguration("usb_port")
    joint_config_file = LaunchConfiguration("joint_config_file")
    use_rviz          = LaunchConfiguration("use_rviz")
    realsense_params  = LaunchConfiguration("realsense_params")
    detector_params   = LaunchConfiguration("detector_params")
    camera_namespace  = LaunchConfiguration("camera_namespace")

    # Camera mount offset — each value is a separate arg so they can be passed
    # cleanly to static_transform_publisher
    cam_x     = LaunchConfiguration("cam_x")
    cam_y     = LaunchConfiguration("cam_y")
    cam_z     = LaunchConfiguration("cam_z")
    cam_roll  = LaunchConfiguration("cam_roll")
    cam_pitch = LaunchConfiguration("cam_pitch")
    cam_yaw   = LaunchConfiguration("cam_yaw")

    default_rs_params  = PathJoinSubstitution(
        [FindPackageShare("so101_perception"), "config", "realsense_d435.yaml"]
    )
    default_det_params = PathJoinSubstitution(
        [FindPackageShare("so101_perception"), "config", "blue_detector_params.yaml"]
    )
    combined_rviz = os.path.join(pkg_perception, "rviz", "perception_with_robot.rviz")

    # ── 1. Robot arm + MoveIt (no RViz — we launch our own below) ─────────────
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, "launch", "follower_moveit_demo.launch.py")
        ),
        launch_arguments={
            "hardware_type":     hardware_type,
            "usb_port":          usb_port,
            "joint_config_file": joint_config_file,
            "use_rviz":          "false",
        }.items(),
    )

    # ── 2. RealSense D435 driver ───────────────────────────────────────────────
    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="camera",
        namespace=camera_namespace,
        parameters=[realsense_params],
        output="screen",
        emulate_tty=True,
    )

    # ── 3. Static TF: gripper link → RealSense body ────────────────────────────
    # Parent : moving_jaw_so101_v1_link  (gripper, no namespace prefix in this stack)
    # Child  : camera_link               (RealSense base frame published by driver)
    # Defaults: so101_cameras.xacro wrist-camera mount values
    camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="wrist_cam_to_camera_link_tf",
        arguments=[
            "--x",             cam_x,
            "--y",             cam_y,
            "--z",             cam_z,
            "--roll",          cam_roll,
            "--pitch",         cam_pitch,
            "--yaw",           cam_yaw,
            "--frame-id",      "moving_jaw_so101_v1_link",
            "--child-frame-id", "camera_link",
        ],
        output="screen",
    )

    # ── 4. Blue-object detector ────────────────────────────────────────────────
    detector_node = Node(
        package="so101_perception",
        executable="blue_object_detector",
        name="blue_object_detector",
        parameters=[detector_params],
        output="screen",
        emulate_tty=True,
    )

    # ── 5. RViz: robot model + perception markers ──────────────────────────────
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", combined_rviz],
        condition=IfCondition(use_rviz),
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("hardware_type",     default_value="real"),
        DeclareLaunchArgument("usb_port",          default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("joint_config_file", default_value=""),
        DeclareLaunchArgument("use_rviz",          default_value="true"),
        DeclareLaunchArgument("camera_namespace",  default_value="camera"),
        DeclareLaunchArgument("realsense_params",  default_value=default_rs_params),
        DeclareLaunchArgument("detector_params",   default_value=default_det_params),
        # Wrist-camera mount offset (so101_cameras.xacro defaults)
        DeclareLaunchArgument("cam_x",     default_value="0.0",
                              description="Camera X offset from moving_jaw link (m)"),
        DeclareLaunchArgument("cam_y",     default_value="0.0",
                              description="Camera Y offset from moving_jaw link (m)"),
        DeclareLaunchArgument("cam_z",     default_value="-0.02",
                              description="Camera Z offset from moving_jaw link (m)"),
        DeclareLaunchArgument("cam_roll",  default_value="-1.5708",
                              description="Camera roll  relative to gripper (rad)"),
        DeclareLaunchArgument("cam_pitch", default_value="0.0",
                              description="Camera pitch relative to gripper (rad)"),
        DeclareLaunchArgument("cam_yaw",   default_value="-1.5708",
                              description="Camera yaw   relative to gripper (rad)"),
        robot_launch,
        realsense_node,
        camera_tf,
        detector_node,
        rviz_node,
    ])
