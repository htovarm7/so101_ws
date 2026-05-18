"""Pick-and-place stack wired for Placo-based visual servoing.

Replaces the MoveIt + pick_ik core of `pick_and_place.launch.py` with the
Placo solver from `so101_kinematics/cartesian_motion_node`.  Differences
vs. the original:

- `arm_forward_controller` is activated instead of `arm_trajectory_controller`
  so Placo can stream position commands continuously.
- `gripper_controller` (ParallelGripperCommand action) stays as before;
  the gripper still needs discrete open/close commands.
- No `move_group` / no `moveit_rviz` — pick_ik and OMPL are not used here.
- `cartesian_motion_node` is launched with `robot_description` pointing to
  the pre-generated real-hardware URDF so Placo plans against the same
  kinematics as the real arm.
- Perception nodes (object_classifier, zone_detector) and the cameras are
  identical to the original launch.

No orchestrator node is launched here (Phase 2 verification only — drive
manually via `ros2 service call /go_to_pose ...`).  The orchestrator
arrives in Phase 3.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Path of the URDF pre-generated at image build-time by Dockerfile.moveit.
# Kept identical to the rest of the stack so kinematics stay consistent.
SO101_URDF_FILE = os.environ.get(
    "SO101_URDF_FILE", "/ros2_ws/so101_follower_real.urdf"
)


def _launch_setup(context, *args, **kwargs):
    pkg_bringup = get_package_share_directory("so101_bringup")
    pkg_perception = get_package_share_directory("so101_perception")

    hardware_type = LaunchConfiguration("hardware_type").perform(context)
    namespace = LaunchConfiguration("namespace").perform(context)
    usb_port = LaunchConfiguration("usb_port").perform(context)
    joint_config_file = LaunchConfiguration("joint_config_file").perform(context)
    use_cameras = LaunchConfiguration("use_cameras").perform(context)
    use_perception = LaunchConfiguration("use_perception").perform(context)

    # ── Hardware + ros2_control with arm_forward_controller active ──────────
    follower_split = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, "launch", "follower_split.launch.py")
        ),
        launch_arguments={
            "hardware_type": hardware_type,
            "namespace": namespace,
            "usb_port": usb_port,
            "joint_config_file": joint_config_file,
            "arm_controller": "arm_forward_controller",
            "use_rviz": "false",
        }.items(),
    )

    # ── Cameras (RealSense) ─────────────────────────────────────────────────
    cameras_cfg = os.path.join(
        pkg_bringup, "config", "cameras", "so101_cameras_realsense.yaml"
    )
    cameras_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, "launch", "cameras.launch.py")
        ),
        launch_arguments={"cameras_config": cameras_cfg}.items(),
    )

    actions = [follower_split]
    if use_cameras.lower() == "true":
        actions.append(cameras_launch)

    # ── RViz (optional) — gives a visual of the arm + TF + detection markers
    use_rviz = LaunchConfiguration("use_rviz").perform(context)
    if use_rviz.lower() == "true":
        rviz_cfg = os.path.join(pkg_bringup, "rviz", "follower.rviz")
        actions.append(Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_cfg],
            output="screen",
        ))

    # ── Perception (same as the MoveIt-based stack) ─────────────────────────
    if use_perception.lower() == "true":
        objects_hsv = os.path.join(pkg_perception, "config", "objects_hsv.yaml")
        zones_hsv = os.path.join(pkg_perception, "config", "zones_hsv.yaml")
        actions.append(Node(
            package="so101_perception",
            executable="object_classifier",
            name="object_classifier",
            parameters=[objects_hsv, {"config_file": objects_hsv}],
            output="screen",
            emulate_tty=True,
        ))
        actions.append(Node(
            package="so101_perception",
            executable="zone_detector",
            name="zone_detector",
            parameters=[zones_hsv, {"publish_camera_tf": False}],
            output="screen",
            emulate_tty=True,
        ))

    # ── Placo-backed Cartesian motion node ──────────────────────────────────
    # Use the pre-generated real-hardware URDF (absolute path triggers the
    # filesystem-URDF code path patched into cartesian_motion_node.py).
    # arm_forward_controller owns only the 5 arm joints; the gripper stays
    # on the parallel_gripper_action_controller.  Tell the node to publish
    # only those 5 joints in the order the controller expects.
    cartesian = Node(
        package="so101_kinematics",
        executable="cartesian_motion_node",
        name="cartesian_motion_node",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "robot_description": SO101_URDF_FILE,
            "joints_topic": "/follower/joint_states",
            "cmd_topic": "/follower/arm_forward_controller/commands",
            "base_frame": "base_link",
            "command_joint_names": [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
        }],
    )
    actions.append(cartesian)

    # ── Servo-backed orchestrator (replaces MoveItPy sort_by_class) ────────
    use_orchestrator = LaunchConfiguration("use_orchestrator").perform(context)
    if use_orchestrator.lower() == "true":
        pkg_moveit_cfg = get_package_share_directory("so101_moveit_config")
        pick_place_cfg = os.path.join(pkg_moveit_cfg, "config", "pick_and_place.yaml")
        actions.append(Node(
            name="sort_by_class",
            package="so101_moveit_config",
            executable="sort_by_class_servo.py",
            output="screen",
            emulate_tty=True,
            parameters=[pick_place_cfg],
        ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("hardware_type", default_value="real"),
        DeclareLaunchArgument("namespace", default_value="follower"),
        DeclareLaunchArgument("usb_port", default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("joint_config_file", default_value="/calibration/papu.json"),
        DeclareLaunchArgument("use_cameras", default_value="true"),
        DeclareLaunchArgument("use_perception", default_value="true"),
        DeclareLaunchArgument("use_orchestrator", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        OpaqueFunction(function=_launch_setup),
    ])
