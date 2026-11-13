#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class TargetPosePublisher(Node):
    """Utility node to publish Cartesian reaching goals to /ur3_reach/target_pose."""

    def __init__(self, x: float, y: float, z: float):
        super().__init__("target_pose_publisher")
        self.pub = self.create_publisher(PoseStamped, "/ur3_reach/target_pose", 10)
        self.target = (x, y, z)
        # Timer to publish 5 times and then shutdown
        self.count = 0
        self.timer = self.create_timer(0.2, self._publish_target)

    def _publish_target(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.pose.position.x = float(self.target[0])
        msg.pose.position.y = float(self.target[1])
        msg.pose.position.z = float(self.target[2])

        # Pitch = pi/2: Wrist pointing downwards in base frame
        msg.pose.orientation.w = 0.7071068
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.7071068
        msg.pose.orientation.z = 0.0

        self.pub.publish(msg)
        self.get_logger().info(f"Published target: X={self.target[0]:.3f}, Y={self.target[1]:.3f}, Z={self.target[2]:.3f}")
        self.count += 1
        if self.count >= 3:
            self.get_logger().info("Target published successfully. Exiting.")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Publish reaching goal to UR3 RL Policy Node")
    parser.add_argument("--x", type=float, default=0.35, help="Target X coordinate in meters (default: 0.35)")
    parser.add_argument("--y", type=float, default=0.00, help="Target Y coordinate in meters (default: 0.00)")
    parser.add_argument("--z", type=float, default=0.22, help="Target Z coordinate in meters (default: 0.22)")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = TargetPosePublisher(args.x, args.y, args.z)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
