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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ── Launch arguments ───────────────────────────────────────────────────────
    hardware_type     = LaunchConfiguration("hardware_type")
    usb_port          = LaunchConfiguration("usb_port")
    joint_config_file = LaunchConfiguration("joint_config_file")
    start_rviz        = LaunchConfiguration("start_rviz")
    realsense_params  = LaunchConfiguration("realsense_params")
    detector_params   = LaunchConfiguration("detector_params")
    camera_namespace  = LaunchConfiguration("camera_namespace")


    default_rs_params  = PathJoinSubstitution(
        [FindPackageShare("so101_perception"), "config", "realsense_d435.yaml"]
    )
    default_det_params = PathJoinSubstitution(
        [FindPackageShare("so101_perception"), "config", "blue_detector_params.yaml"]
    )
    combined_rviz = PathJoinSubstitution(
        [FindPackageShare("so101_perception"), "rviz", "perception_with_robot.rviz"]
    )

    # ── 1. Robot arm + MoveIt (no RViz — we launch our own below) ─────────────
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("so101_bringup"), "launch", "follower_moveit_demo.launch.py"])
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

    # ── 3. Blue-object detector ───────────────────────────────────────────────
    # The detector node itself broadcasts the static TF bridge
    # (moving_jaw_so101_v1_link → camera_color_optical_frame) via
    # StaticTransformBroadcaster once CameraInfo is received.
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
        condition=IfCondition(start_rviz),
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("hardware_type",     default_value="real"),
        DeclareLaunchArgument("usb_port",          default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("joint_config_file", default_value=""),
        DeclareLaunchArgument("start_rviz",        default_value="true"),
        DeclareLaunchArgument("camera_namespace",  default_value="camera"),
        DeclareLaunchArgument("realsense_params",  default_value=default_rs_params),
        DeclareLaunchArgument("detector_params",   default_value=default_det_params),
        robot_launch,
        realsense_node,
        detector_node,
        rviz_node,
    ])
