#!/usr/bin/env python3
"""
Task 1: Closed-Loop Target Pose Reaching Node for UR3 in base_link.
Interfaces directly with 'ur_robot_driver' via 'forward_position_controller'.

Features:
  - Runs at 125 Hz continuous streaming.
  - Automatically verifies target against table_robot_mount_frame geofence.
  - Subscribes to /target_pose (PoseStamped in base_link).
  - Publishes commands to /forward_position_controller/commands.
  - Publishes 3D RViz visualization markers.
"""
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
import tf2_ros

from ur3_bc.policy_engine import PolicyEngine

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# base_link to table_robot_mount_frame: Rz(-45 deg)
theta_m_b = -np.pi / 4.0
R_M_B = np.array([
    [ np.cos(theta_m_b), -np.sin(theta_m_b), 0.0],
    [ np.sin(theta_m_b),  np.cos(theta_m_b), 0.0],
    [ 0.0,                0.0,               1.0]
])


class Task1ReachNode(Node):
    def __init__(self):
        super().__init__("ur3_bc_task1_reach_node")

        # Declare parameters
        self.declare_parameter("model_dir", "")
        self.declare_parameter("control_frequency_hz", 125.0)
        self.declare_parameter("command_topic", "/forward_position_controller/commands")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("target_pose_topic", "/target_pose")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tcp_frame", "rg2_gripper_tcp")

        # Geofence parameters in table_robot_mount_frame
        self.declare_parameter("geofence.x_min", 0.070)
        self.declare_parameter("geofence.x_max", 0.470)
        self.declare_parameter("geofence.y_min", -0.450)
        self.declare_parameter("geofence.y_max", 0.450)
        self.declare_parameter("geofence.z_min", 0.020)
        self.declare_parameter("geofence.z_max", 0.280)
        self.declare_parameter("geofence.r_min", 0.160)
        self.declare_parameter("geofence.r_max", 0.480)

        # Read parameters
        model_dir = self.get_parameter("model_dir").value or None
        self.freq = float(self.get_parameter("control_frequency_hz").value)
        self.dt = 1.0 / self.freq
        self.base_frame = self.get_parameter("base_frame").value
        self.tcp_frame = self.get_parameter("tcp_frame").value

        self.x_min = float(self.get_parameter("geofence.x_min").value)
        self.x_max = float(self.get_parameter("geofence.x_max").value)
        self.y_min = float(self.get_parameter("geofence.y_min").value)
        self.y_max = float(self.get_parameter("geofence.y_max").value)
        self.z_min = float(self.get_parameter("geofence.z_min").value)
        self.z_max = float(self.get_parameter("geofence.z_max").value)
        self.r_min = float(self.get_parameter("geofence.r_min").value)
        self.r_max = float(self.get_parameter("geofence.r_max").value)

        self.get_logger().info("=" * 65)
        self.get_logger().info("Initializing UR3 BC Task 1: Closed-Loop Reaching Node")
        self.get_logger().info(f"Target Frame     : {self.base_frame}")
        self.get_logger().info(f"TCP Frame        : {self.tcp_frame}")
        self.get_logger().info(f"Control Rate     : {self.freq:.1f} Hz (dt={self.dt*1000:.1f}ms)")
        self.get_logger().info("=" * 65)

        # Load Neural Policy
        self.policy = PolicyEngine(model_dir=model_dir)
        self.get_logger().info(f"Policy Engine active: Backend = '{self.policy.backend}'")

        # Robot State Cache
        self.q_actual = None

        # Default Target Pose in base_link (Ready Pose TCP)
        self.target_pos_base = np.array([0.3439, 0.0625, 0.0444], dtype=np.float32)
        self.target_received = False
        self.converged_reported = False

        # TF Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscriptions & Publishers
        self.sub_joints = self.create_subscription(
            JointState,
            self.get_parameter("joint_states_topic").value,
            self.joint_state_callback,
            10
        )
        self.sub_target = self.create_subscription(
            PoseStamped,
            self.get_parameter("target_pose_topic").value,
            self.target_pose_callback,
            10
        )
        self.pub_cmd = self.create_publisher(
            Float64MultiArray,
            self.get_parameter("command_topic").value,
            10
        )
        self.pub_marker = self.create_publisher(Marker, "/ur3_bc/target_marker", 10)

        # 125 Hz Control Loop Timer
        self.timer = self.create_timer(self.dt, self.control_cycle)

    def joint_state_callback(self, msg: JointState):
        name_to_pos = {name: pos for name, pos in zip(msg.name, msg.position)}
        if all(j in name_to_pos for j in JOINT_NAMES):
            self.q_actual = np.array([name_to_pos[j] for j in JOINT_NAMES], dtype=np.float32)

    def is_target_safe(self, p_base: np.ndarray) -> bool:
        """Validate target against table geofence."""
        p_m = R_M_B @ p_base
        xm, ym, zm = p_m[0], p_m[1], p_m[2]
        r = np.hypot(xm, ym)

        if not (self.x_min <= xm <= self.x_max and self.y_min <= ym <= self.y_max and self.z_min <= zm <= self.z_max):
            self.get_logger().warn(
                f"[GEOFENCE REJECT] Target {p_base} in mount [{xm:.3f}, {ym:.3f}, {zm:.3f}] "
                f"violates bounding box [{self.x_min}..{self.x_max}, {self.y_min}..{self.y_max}, {self.z_min}..{self.z_max}]!"
            )
            return False

        if r > self.r_max or r < self.r_min:
            self.get_logger().warn(f"[GEOFENCE REJECT] Radius {r:.3f}m outside [{self.r_min}, {self.r_max}]m!")
            return False

        return True

    def target_pose_callback(self, msg: PoseStamped):
        new_target = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ], dtype=np.float32)

        if self.is_target_safe(new_target):
            self.target_pos_base = new_target
            self.target_received = True
            self.converged_reported = False
            self.get_logger().info(f"[NEW TARGET ACCEPTED] base_link: {self.target_pos_base.tolist()}")
            self.publish_rviz_marker()
        else:
            self.get_logger().error(f"[TARGET DISCARDED] Unsafe coordinates: {new_target.tolist()}")

    def get_tcp_pos_base(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.tcp_frame, rclpy.time.Time()
            )
            t = transform.transform.translation
            return np.array([t.x, t.y, t.z], dtype=np.float32)
        except Exception:
            return None

    def publish_rviz_marker(self):
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.base_frame
        m.ns = "ur3_bc_target"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(self.target_pos_base[0])
        m.pose.position.y = float(self.target_pos_base[1])
        m.pose.position.z = float(self.target_pos_base[2])
        m.pose.orientation.w = 1.0
        m.scale.x = 0.03
        m.scale.y = 0.03
        m.scale.z = 0.03
        m.color.r = 1.0
        m.color.g = 0.85
        m.color.b = 0.0
        m.color.a = 0.95
        self.pub_marker.publish(m)

    def control_cycle(self):
        if self.q_actual is None:
            return  # Waiting for initial /joint_states

        tcp_b = self.get_tcp_pos_base()
        if tcp_b is None:
            return  # Waiting for TF

        rel_err = self.target_pos_base - tcp_b
        dist = np.linalg.norm(rel_err)

        # Convergence threshold: 2.5 mm
        if dist < 0.0025:
            if not self.converged_reported and self.target_received:
                self.get_logger().info(f"[TARGET REACHED] Position error: {dist*1000.0:.2f} mm | Holding position.")
                self.converged_reported = True
            return

        # Inference
        delta_q = self.policy.predict(self.q_actual, self.target_pos_base, tcp_b)

        # Next joint position command
        q_cmd = self.q_actual + delta_q

        cmd_msg = Float64MultiArray()
        cmd_msg.data = q_cmd.tolist()
        self.pub_cmd.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Task1ReachNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
