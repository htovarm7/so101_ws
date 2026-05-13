#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
fi

# Publish robot_description with TRANSIENT_LOCAL QoS in the background.
# ros2_control_node waits for this topic; RSP can race or fail silently in Docker.
if [ -f /ros2_ws/so101_follower_real.urdf ]; then
    python3 - <<'PYEOF' &
import rclpy, time, sys
from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

rclpy.init()
node = rclpy.create_node('robot_description_publisher')
qos = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)
pub = node.create_publisher(String, '/follower/robot_description', qos)
msg = String()
with open('/ros2_ws/so101_follower_real.urdf') as f:
    msg.data = f.read()
print(f'[rd_pub] Publishing robot_description ({len(msg.data)} bytes) on /follower/robot_description')
sys.stdout.flush()
while rclpy.ok():
    pub.publish(msg)
    time.sleep(30.0)
PYEOF
    # Give the publisher a moment to advertise before the launch starts
    sleep 1
fi

exec "$@"
