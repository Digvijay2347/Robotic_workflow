#!/usr/bin/env python3
"""
CLI Utility to Command Targets in base_link to UR3 BC Policy.
Validates table_robot_mount_frame geofence before publishing.

Usage:
    ros2 run ur3_bc send_target --x 0.25 --y 0.15 --z 0.10
    ros2 run ur3_bc send_target --center
    ros2 run ur3_bc send_target --ready
    ros2 run ur3_bc send_target
"""
import sys
import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

# Safety limits in table_robot_mount_frame
X_M_MIN, X_M_MAX = 0.070, 0.470   # 40 cm forward depth
Y_M_MIN, Y_M_MAX = -0.450, 0.450  # 90 cm lateral width
Z_M_MIN, Z_M_MAX = 0.020, 0.280   # 26 cm height
MIN_RADIUS = 0.160
MAX_RADIUS = 0.480

theta_m_b = -np.pi / 4.0
R_M_B = np.array([
    [ np.cos(theta_m_b), -np.sin(theta_m_b), 0.0],
    [ np.sin(theta_m_b),  np.cos(theta_m_b), 0.0],
    [ 0.0,                0.0,               1.0]
])

theta_b_m = np.pi / 4.0
R_B_M = np.array([
    [ np.cos(theta_b_m), -np.sin(theta_b_m), 0.0],
    [ np.sin(theta_b_m),  np.cos(theta_b_m), 0.0],
    [ 0.0,                0.0,               1.0]
])


class TargetPublisher(Node):
    def __init__(self):
        super().__init__("ur3_bc_send_target_cli")
        self.pub = self.create_publisher(PoseStamped, "/target_pose", 10)

    def publish_target(self, x_b, y_b, z_b):
        p_base = np.array([x_b, y_b, z_b], dtype=np.float32)
        p_mount = R_M_B @ p_base
        xm, ym, zm = p_mount[0], p_mount[1], p_mount[2]
        r = np.hypot(xm, ym)

        print("\nTarget Coordinate Check:")
        print(f"  In base_link              : [{x_b:.4f}, {y_b:.4f}, {z_b:.4f}] m")
        print(f"  In table_robot_mount_frame: [{xm:.4f}, {ym:.4f}, {zm:.4f}] m")
        print(f"  Radial distance from base : {r:.4f} m (allowed: [{MIN_RADIUS:.3f}, {MAX_RADIUS:.3f}] m)")

        # Detailed Safety Check with informative feedback
        violations = []
        if not (X_M_MIN <= xm <= X_M_MAX):
            violations.append(f"Forward reach X_m = {xm:.3f}m outside [{X_M_MIN:.3f}, {X_M_MAX:.3f}]m")
        if not (Y_M_MIN <= ym <= Y_M_MAX):
            violations.append(f"Lateral width Y_m = {ym:.3f}m outside [{Y_M_MIN:.3f}, {Y_M_MAX:.3f}]m")
        if not (Z_M_MIN <= zm <= Z_M_MAX):
            violations.append(f"Height Z = {zm:.3f}m outside [{Z_M_MIN:.3f}, {Z_M_MAX:.3f}]m ({zm*100:.1f}cm above table, max allowed: {Z_M_MAX*100:.0f}cm)")
        if r > MAX_RADIUS:
            violations.append(f"Radius r = {r:.3f}m exceeds max UR3 arm reach of {MAX_RADIUS:.3f}m")
        if r < MIN_RADIUS:
            violations.append(f"Radius r = {r:.3f}m too close to base pillar (< {MIN_RADIUS:.3f}m)")

        if violations:
            print("  --> [ERROR] Target is OUTSIDE the safe workspace geofence! Command ABORTED.")
            for v in violations:
                print(f"      * Violation: {v}")
            print(f"\n  [TIP] In base_link, Z is height above table plate (Z=0 is the table plate).")
            print(f"        Recommended height for tabletop tasks is Z in [0.05, 0.20] m.\n")
            return False

        print("  --> [SAFE] Target verified inside 90cm x 40cm x 26cm workspace geofence.")

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = float(x_b)
        msg.pose.position.y = float(y_b)
        msg.pose.position.z = float(z_b)
        msg.pose.orientation.w = 0.0
        msg.pose.orientation.x = 1.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0

        for _ in range(3):
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)

        print("  --> [PUBLISHED] Sent PoseStamped on /target_pose (frame_id: 'base_link')\n")
        return True


def main():
    parser = argparse.ArgumentParser(description="Send target pose to UR3 in base_link.")
    parser.add_argument("--x", type=float, help="X in base_link (m)")
    parser.add_argument("--y", type=float, help="Y in base_link (m)")
    parser.add_argument("--z", type=float, help="Z in base_link (m)")
    parser.add_argument("--center", action="store_true", help="Send workspace center target")
    parser.add_argument("--ready", action="store_true", help="Send Ready Pose TCP target")
    args = parser.parse_args()

    rclpy.init()
    node = TargetPublisher()

    try:
        if args.center:
            p_center = R_B_M @ np.array([0.270, 0.000, 0.150])
            node.publish_target(p_center[0], p_center[1], p_center[2])
        elif args.ready:
            node.publish_target(0.3439, 0.0625, 0.0444)
        elif args.x is not None and args.y is not None and args.z is not None:
            node.publish_target(args.x, args.y, args.z)
        else:
            print("\nInteractive Target Pose Input (strictly relative to base_link):")
            try:
                x = float(input("Enter X in base_link [default 0.25]: ") or "0.25")
                y = float(input("Enter Y in base_link [default 0.15]: ") or "0.15")
                z = float(input("Enter Z in base_link [default 0.10]: ") or "0.10")
                node.publish_target(x, y, z)
            except ValueError:
                print("Invalid input numbers.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
