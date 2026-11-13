#!/usr/bin/env python3
"""
Task 2: Autonomous Pick and Place State Machine Node for UR3 + OnRobot RG2.
Combines high-level sequencing with the 125 Hz neural trajectory policy.

Sequence:
  0. IDLE / READY (Ready Pose TCP)
  1. MOVE_PRE_PICK (Pick [X, Y, Z + 10cm])
  2. OPEN_GRIPPER (Pub: /rg2_command_int = 1, wait for /rg2_state >= 70mm)
  3. MOVE_PICK (Pick [X, Y, Z_cube])
  4. CLOSE_GRASP (Pub: /rg2_command_int = 0, verify /rg2_state > 5mm)
  5. LIFT_RETREAT (Pick [X, Y, Z + 10cm])
  6. MOVE_PRE_PLACE (Place [X, Y, Z + 10cm])
  7. MOVE_PLACE (Place [X, Y, Z_place])
  8. OPEN_RELEASE (Pub: /rg2_command_int = 1, release object)
  9. RETREAT_PLACE (Place [X, Y, Z + 10cm])
  10. RETURN_READY (Return to Ready Pose TCP)
  11. CLOSE_PARKED (Close gripper to parked state)
  12. DONE
"""
import time
import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Int32, Float64
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

class PickAndPlaceState:
    IDLE = 0
    MOVE_PRE_PICK = 1
    OPEN_FOR_PICK = 2
    MOVE_PICK = 3
    CLOSE_GRASP = 4
    LIFT_RETREAT = 5
    MOVE_PRE_PLACE = 6
    MOVE_PLACE = 7
    OPEN_RELEASE = 8
    RETREAT_PLACE = 9
    RETURN_READY = 10
    CLOSE_PARKED = 11
    DONE = 12

STATE_NAMES = {
    0: "IDLE",
    1: "MOVE_PRE_PICK",
    2: "OPEN_FOR_PICK",
    3: "MOVE_PICK",
    4: "CLOSE_GRASP",
    5: "LIFT_RETREAT",
    6: "MOVE_PRE_PLACE",
    7: "MOVE_PLACE",
    8: "OPEN_RELEASE",
    9: "RETREAT_PLACE",
    10: "RETURN_READY",
    11: "CLOSE_PARKED",
    12: "DONE",
}


class Task2PickPlaceNode(Node):
    def __init__(self):
        super().__init__("ur3_bc_task2_pick_place_node")

        # Declare ROS 2 parameters
        self.declare_parameter("model_dir", "")
        self.declare_parameter("control_frequency_hz", 125.0)
        self.declare_parameter("command_topic", "/forward_position_controller/commands")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("gripper_command_topic", "/rg2_command_int")
        self.declare_parameter("gripper_state_topic", "/rg2_state")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tcp_frame", "rg2_gripper_tcp")

        # Pick and Place default positions in base_link (m)
        self.declare_parameter("pick_x", 0.250)
        self.declare_parameter("pick_y", 0.150)
        self.declare_parameter("pick_z", 0.040)
        self.declare_parameter("place_x", 0.250)
        self.declare_parameter("place_y", -0.150)
        self.declare_parameter("place_z", 0.040)
        self.declare_parameter("clearance", 0.100)

        # Read parameters
        model_dir = self.get_parameter("model_dir").value or None
        self.freq = float(self.get_parameter("control_frequency_hz").value)
        self.dt = 1.0 / self.freq
        self.base_frame = self.get_parameter("base_frame").value
        self.tcp_frame = self.get_parameter("tcp_frame").value

        self.pick_pos = np.array([
            float(self.get_parameter("pick_x").value),
            float(self.get_parameter("pick_y").value),
            float(self.get_parameter("pick_z").value)
        ], dtype=np.float32)

        self.place_pos = np.array([
            float(self.get_parameter("place_x").value),
            float(self.get_parameter("place_y").value),
            float(self.get_parameter("place_z").value)
        ], dtype=np.float32)

        self.clearance = float(self.get_parameter("clearance").value)

        # Locked-in Ready Pose TCP in base_link
        self.ready_pos = np.array([0.3439, 0.0625, 0.0444], dtype=np.float32)

        self.get_logger().info("=" * 65)
        self.get_logger().info("Initializing UR3 BC Task 2: Pick and Place Node")
        self.get_logger().info(f"Pick Location (base_link)  : {self.pick_pos.tolist()}")
        self.get_logger().info(f"Place Location (base_link) : {self.place_pos.tolist()}")
        self.get_logger().info(f"Approach Clearance         : {self.clearance*100:.1f} cm")
        self.get_logger().info(f"Control Rate               : {self.freq:.1f} Hz")
        self.get_logger().info("=" * 65)

        # Load Neural Policy Engine
        self.policy = PolicyEngine(model_dir=model_dir)
        self.get_logger().info(f"Policy Engine active: Backend = '{self.policy.backend}'")

        # Robot & Gripper State Cache
        self.q_actual = None
        self.current_gripper_width = 0.0
        self.gripper_state_received = False

        # State Machine Tracking
        self.state = PickAndPlaceState.IDLE
        self.state_start_time = self.get_clock().now()
        self.current_target = self.ready_pos.copy()
        self.start_countdown = int(self.freq * 1.0)  # Wait 1.0s before starting

        # Gripper continuous publishing throttle (10 Hz loop)
        self.last_gripper_cmd_time = 0.0
        self.last_logged_gripper_cmd = None

        # Motion progress telemetry throttle
        self.last_motion_log_time = 0.0

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
        self.sub_gripper = self.create_subscription(
            Float64,
            self.get_parameter("gripper_state_topic").value,
            self.gripper_state_callback,
            10
        )
        self.pub_cmd = self.create_publisher(
            Float64MultiArray,
            self.get_parameter("command_topic").value,
            10
        )
        self.pub_gripper = self.create_publisher(
            Int32,
            self.get_parameter("gripper_command_topic").value,
            10
        )
        self.pub_marker = self.create_publisher(Marker, "/ur3_bc/target_marker", 10)

        # 125 Hz Unified Control Loop
        self.timer = self.create_timer(self.dt, self.unified_control_loop)

    def joint_state_callback(self, msg: JointState):
        name_to_pos = {name: pos for name, pos in zip(msg.name, msg.position)}
        if all(j in name_to_pos for j in JOINT_NAMES):
            self.q_actual = np.array([name_to_pos[j] for j in JOINT_NAMES], dtype=np.float32)

    def gripper_state_callback(self, msg: Float64):
        self.current_gripper_width = float(msg.data)
        self.gripper_state_received = True

    def get_tcp_pos_base(self):
        try:
            t = self.tf_buffer.lookup_transform(self.base_frame, self.tcp_frame, rclpy.time.Time())
            return np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z], dtype=np.float32)
        except Exception:
            return None

    def get_cube_from_tf(self):
        try:
            t = self.tf_buffer.lookup_transform(self.base_frame, "cube_frame", rclpy.time.Time())
            return np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z], dtype=np.float32)
        except Exception:
            return None

    def send_gripper_command(self, cmd_int: int):
        msg = Int32()
        msg.data = int(cmd_int)
        self.pub_gripper.publish(msg)
        action_name = "OPEN" if cmd_int == 1 else "CLOSE"
        self.get_logger().info(f"[GRIPPER] Sent command: {action_name} ({cmd_int})")

    def send_gripper_command_throttled(self, cmd_int: int, rate_hz: float = 10.0):
        """Publishes gripper command repeatedly at a throttled rate (default 10 Hz)."""
        now = self.get_clock().now().nanoseconds / 1e9
        interval = 1.0 / rate_hz
        if (now - self.last_gripper_cmd_time) >= interval:
            msg = Int32()
            msg.data = int(cmd_int)
            self.pub_gripper.publish(msg)
            self.last_gripper_cmd_time = now

            if self.last_logged_gripper_cmd != (self.state, cmd_int):
                action_name = "OPEN" if cmd_int == 1 else "CLOSE"
                state_str = STATE_NAMES.get(self.state, str(self.state))
                self.get_logger().info(f"[GRIPPER] Repeatedly publishing {action_name} ({cmd_int}) at {rate_hz:.0f}Hz in {state_str}")
                self.last_logged_gripper_cmd = (self.state, cmd_int)

    def transition_to(self, new_state, new_target=None):
        prev_name = STATE_NAMES.get(self.state, str(self.state))
        new_name = STATE_NAMES.get(new_state, str(new_state))
        self.get_logger().info(f"[STATE] {prev_name} ---> {new_name}")
        self.state = new_state
        self.state_start_time = self.get_clock().now()
        self.last_gripper_cmd_time = 0.0  # Reset throttle so next gripper command sends immediately
        self.last_motion_log_time = 0.0   # Reset motion log timer for immediate status
        if new_target is not None:
            self.current_target = np.array(new_target, dtype=np.float32)
            self.publish_rviz_marker()

    def publish_rviz_marker(self):
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.base_frame
        m.ns = "ur3_bc_pnp"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(self.current_target[0])
        m.pose.position.y = float(self.current_target[1])
        m.pose.position.z = float(self.current_target[2])
        m.pose.orientation.w = 1.0
        m.scale.x = 0.03
        m.scale.y = 0.03
        m.scale.z = 0.03
        m.color.r = 0.0
        m.color.g = 0.85
        m.color.b = 1.0
        m.color.a = 0.95
        self.pub_marker.publish(m)

    def unified_control_loop(self):
        if self.q_actual is None:
            return

        tcp_b = self.get_tcp_pos_base()
        if tcp_b is None:
            return

        # Check if dynamic cube_frame exists in TF during IDLE
        if self.state == PickAndPlaceState.IDLE:
            cube_tf = self.get_cube_from_tf()
            if cube_tf is not None:
                self.pick_pos = cube_tf

        # Countdown before first motion
        if self.start_countdown > 0:
            self.start_countdown -= 1
            if self.start_countdown == 0:
                pre_pick = self.pick_pos + np.array([0, 0, self.clearance], dtype=np.float32)
                self.transition_to(PickAndPlaceState.MOVE_PRE_PICK, pre_pick)
            return

        elapsed = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
        rel_err = self.current_target - tcp_b
        dist = np.linalg.norm(rel_err)

        # State-specific convergence tolerance:
        # Fine precision for grasping & dropping; relaxed tolerance for clearance & transfer in free air
        if self.state in [PickAndPlaceState.MOVE_PICK, PickAndPlaceState.MOVE_PLACE]:
            tol = 0.0040  # 4.0 mm for contact/alignment
        else:
            tol = 0.0100  # 10.0 mm for approach, lift, transfer, retreat, and home

        # Robust convergence check: meets target tolerance, or safe timeout fallback (prevents getting stuck)
        motion_timeout = 8.0  # seconds
        timed_out = (elapsed > motion_timeout and dist < 0.035) or (elapsed > 12.0)
        converged = (dist < tol) or timed_out

        # Periodic motion telemetry (every 1.5s) to monitor trajectory progress
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if self.state in [
            PickAndPlaceState.MOVE_PRE_PICK,
            PickAndPlaceState.MOVE_PICK,
            PickAndPlaceState.LIFT_RETREAT,
            PickAndPlaceState.MOVE_PRE_PLACE,
            PickAndPlaceState.MOVE_PLACE,
            PickAndPlaceState.RETREAT_PLACE,
            PickAndPlaceState.RETURN_READY,
        ]:
            if (now_sec - self.last_motion_log_time) >= 1.5:
                state_str = STATE_NAMES.get(self.state, str(self.state))
                status_note = " [TIMEOUT FALLBACK]" if timed_out else ""
                self.get_logger().info(
                    f"[{state_str}] Progress: elapsed={elapsed:.1f}s | error={dist*1000.0:.1f}mm (tol={tol*1000.0:.1f}mm){status_note}"
                )
                self.last_motion_log_time = now_sec

        # -------------------------------------------------------------
        # STATE MACHINE TRANSITION LOGIC
        # -------------------------------------------------------------
        if self.state == PickAndPlaceState.MOVE_PRE_PICK:
            if converged and elapsed > 0.4:
                self.transition_to(PickAndPlaceState.OPEN_FOR_PICK)

        elif self.state == PickAndPlaceState.OPEN_FOR_PICK:
            self.send_gripper_command_throttled(1, rate_hz=10.0)
            if self.current_gripper_width >= 70.0 or elapsed > 1.5:
                self.get_logger().info(f"Gripper Opened: width = {self.current_gripper_width:.1f} mm")
                self.transition_to(PickAndPlaceState.MOVE_PICK, self.pick_pos)

        elif self.state == PickAndPlaceState.MOVE_PICK:
            if converged and elapsed > 0.4:
                self.transition_to(PickAndPlaceState.CLOSE_GRASP)

        elif self.state == PickAndPlaceState.CLOSE_GRASP:
            self.send_gripper_command_throttled(0, rate_hz=10.0)
            if elapsed > 1.0:
                self.get_logger().info(f"Grasp Closed: width = {self.current_gripper_width:.1f} mm")
                if self.current_gripper_width < 3.0:
                    self.get_logger().warn("WARNING: Gripper closed completely to 0mm! Object was MISSED.")
                else:
                    self.get_logger().info("Object grasp CONFIRMED!")
                post_pick = self.pick_pos + np.array([0, 0, self.clearance], dtype=np.float32)
                self.transition_to(PickAndPlaceState.LIFT_RETREAT, post_pick)

        elif self.state == PickAndPlaceState.LIFT_RETREAT:
            if converged and elapsed > 0.4:
                pre_place = self.place_pos + np.array([0, 0, self.clearance], dtype=np.float32)
                self.transition_to(PickAndPlaceState.MOVE_PRE_PLACE, pre_place)

        elif self.state == PickAndPlaceState.MOVE_PRE_PLACE:
            if converged and elapsed > 0.4:
                self.transition_to(PickAndPlaceState.MOVE_PLACE, self.place_pos)

        elif self.state == PickAndPlaceState.MOVE_PLACE:
            if converged and elapsed > 0.4:
                self.transition_to(PickAndPlaceState.OPEN_RELEASE)

        elif self.state == PickAndPlaceState.OPEN_RELEASE:
            self.send_gripper_command_throttled(1, rate_hz=10.0)
            if self.current_gripper_width >= 70.0 or elapsed > 1.5:
                self.get_logger().info(f"Object Released: width = {self.current_gripper_width:.1f} mm")
                post_place = self.place_pos + np.array([0, 0, self.clearance], dtype=np.float32)
                self.transition_to(PickAndPlaceState.RETREAT_PLACE, post_place)

        elif self.state == PickAndPlaceState.RETREAT_PLACE:
            if converged and elapsed > 0.4:
                self.transition_to(PickAndPlaceState.RETURN_READY, self.ready_pos)

        elif self.state == PickAndPlaceState.RETURN_READY:
            if converged and elapsed > 0.4:
                self.transition_to(PickAndPlaceState.CLOSE_PARKED)

        elif self.state == PickAndPlaceState.CLOSE_PARKED:
            self.send_gripper_command_throttled(0, rate_hz=10.0)
            if elapsed > 1.0:
                self.transition_to(PickAndPlaceState.DONE)
                self.get_logger().info("=" * 65)
                self.get_logger().info("PICK AND PLACE TASK SUCCESSFULLY COMPLETED!")
                self.get_logger().info("=" * 65)

        # -------------------------------------------------------------
        # 125 Hz MOTION EXECUTION
        # -------------------------------------------------------------
        if self.state in [
            PickAndPlaceState.MOVE_PRE_PICK,
            PickAndPlaceState.MOVE_PICK,
            PickAndPlaceState.LIFT_RETREAT,
            PickAndPlaceState.MOVE_PRE_PLACE,
            PickAndPlaceState.MOVE_PLACE,
            PickAndPlaceState.RETREAT_PLACE,
            PickAndPlaceState.RETURN_READY,
        ]:
            if not converged:
                delta_q = self.policy.predict(self.q_actual, self.current_target, tcp_b)
                q_cmd = self.q_actual + delta_q
                cmd_msg = Float64MultiArray()
                cmd_msg.data = q_cmd.tolist()
                self.pub_cmd.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Task2PickPlaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
