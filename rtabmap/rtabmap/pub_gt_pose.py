#!/usr/bin/env python3
# file: gazebo_gt_relay.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates, LinkStates

# ----- EDIT THESE CONSTANTS -----
MODEL_NAME       = "voyager"     # your model's name in Gazebo Classic
FIXED_FRAME      = "world"       # Gazebo world frame
POSE_TOPIC       = "/gazebo_gt_pose"
ODOM_TOPIC       = "/gazebo_gt_odom"
USE_LINK_STATES  = False         # True -> use /gazebo/link_states
LINK_NAME        = "base_link"   # used when USE_LINK_STATES is True
# --------------------------------

class GTRelay(Node):
    def __init__(self):
        super().__init__("gazebo_gt_relay")

        self.pose_pub = self.create_publisher(PoseStamped, POSE_TOPIC, 10)
        self.odom_pub = self.create_publisher(Odometry, ODOM_TOPIC, 10)

        if USE_LINK_STATES:
            self.create_subscription(LinkStates, "/gazebo/link_states", self.cb_link, 10)
            self.get_logger().info(f"Using /gazebo/link_states -> {MODEL_NAME}::{LINK_NAME}")
        else:
            self.create_subscription(ModelStates, "/model_states", self.cb_model, 10)
            self.get_logger().info(f"Using /model_states -> model '{MODEL_NAME}'")

        self.get_logger().info(
            f"Publishing {POSE_TOPIC} (PoseStamped) and {ODOM_TOPIC} (Odometry) in frame '{FIXED_FRAME}'"
        )

    def publish_pose_twist(self, pose, twist, child_frame_id: str):
        now = self.get_clock().now().to_msg()

        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = FIXED_FRAME
        ps.pose = pose
        self.pose_pub.publish(ps)

        od = Odometry()
        od.header = ps.header
        od.child_frame_id = child_frame_id
        od.pose.pose = pose
        od.twist.twist = twist
        self.odom_pub.publish(od)

    def cb_model(self, msg: ModelStates):
        try:
            idx = msg.name.index(MODEL_NAME)
        except ValueError:
            return  # model not present yet
        # ModelStates pose/twist are in world frame
        self.publish_pose_twist(msg.pose[idx], msg.twist[idx], "base_link")

    def cb_link(self, msg: LinkStates):
        full = f"{MODEL_NAME}::{LINK_NAME}"
        try:
            idx = msg.name.index(full)
        except ValueError:
            return
        # LinkStates pose/twist are also in world frame
        self.publish_pose_twist(msg.pose[idx], msg.twist[idx], LINK_NAME)

def main():
    rclpy.init()
    node = GTRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
