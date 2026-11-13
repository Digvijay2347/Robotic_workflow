#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import os
import time

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult


# Must match policy action order and forward controller joint order.
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Reference retained from your original code.
# Verify this matches the reference used during Isaac Lab training.
DEFAULT_JOINT_POS = np.array([
    0.0,
    -np.pi / 2,   # -1.5707963 rad (-90 deg)
    0.0,          #  0.0000000 rad (  0 deg)
    -np.pi / 2,   # -1.5707963 rad (-90 deg)
    0.0,          #  0.0000000 rad (  0 deg)
    0.0,          #  0.0000000 rad (  0 deg)
], dtype=np.float32)


class UR3RLPolicyNode(Node):
    """Load a reaching policy and optionally publish joint position targets."""

    def __init__(self):
        super().__init__("ur3_rl_policy_node")

        # Parameters are read at startup.
        self.declare_parameter("model_path", "")
        self.declare_parameter("action_scale", 0.05)
        self.declare_parameter("safety_max_joint_delta", 0.05)
        self.declare_parameter("control_frequency_hz", 30.0)
        self.declare_parameter("enable_execution", False)

        self.declare_parameter(
            "command_topic",
            "/forward_position_controller/commands",
        )
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("target_pose_topic", "/ur3_reach/target_pose")

        # Target acceptance bounds; these do not provide collision checking.
        self.declare_parameter("workspace_bounds.x_min", -0.15)
        self.declare_parameter("workspace_bounds.x_max", 0.60)
        self.declare_parameter("workspace_bounds.y_min", -0.60)
        self.declare_parameter("workspace_bounds.y_max", 0.60)
        self.declare_parameter("workspace_bounds.z_min", 0.005)
        self.declare_parameter("workspace_bounds.z_max", 0.70)
        self.declare_parameter("publish_workspace_marker", True)
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("box_marker_topic", "/ur3_reach/workspace_box_marker")

        model_path = self.get_parameter("model_path").value
        self.action_scale = float(
            self.get_parameter("action_scale").value
        )
        self.max_delta = float(
            self.get_parameter("safety_max_joint_delta").value
        )
        self.freq_hz = float(
            self.get_parameter("control_frequency_hz").value
        )
        self.enable_execution = bool(
            self.get_parameter("enable_execution").value
        )

        cmd_topic = self.get_parameter("command_topic").value
        js_topic = self.get_parameter("joint_states_topic").value
        target_topic = self.get_parameter("target_pose_topic").value

        self.x_bounds = (
            float(self.get_parameter("workspace_bounds.x_min").value),
            float(self.get_parameter("workspace_bounds.x_max").value),
        )
        self.y_bounds = (
            float(self.get_parameter("workspace_bounds.y_min").value),
            float(self.get_parameter("workspace_bounds.y_max").value),
        )
        self.z_bounds = (
            float(self.get_parameter("workspace_bounds.z_min").value),
            float(self.get_parameter("workspace_bounds.z_max").value),
        )
        self.add_on_set_parameters_callback(self._on_parameters_set)

        # Resolve model location.
        if not model_path:
            package_share = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            candidate = os.path.join(
                package_share, "models", "policy.onnx"
            )

            if os.path.exists(candidate):
                model_path = candidate
            else:
                model_path = "models/policy.onnx"

        if ort is None:
            raise ImportError(
                "onnxruntime is required. Install onnxruntime "
                "or onnxruntime-gpu in your Python environment."
            )

        # Initialize ONNX Runtime.
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.intra_op_num_threads = 2

        preferred_providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        available_providers = ort.get_available_providers()
        active_providers = [
            provider
            for provider in preferred_providers
            if provider in available_providers
        ]

        self.get_logger().info(
            f"Loading ONNX session with providers: {active_providers}"
        )

        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=active_providers,
        )
        self.input_name = self.session.get_inputs()[0].name

        self.get_logger().info(
            f"Successfully loaded policy from: {model_path}"
        )

        # State buffers.
        self.latest_joint_pos = None
        self.latest_joint_vel = None
        self.last_joint_state_time = 0.0

        # Model target order: [x, y, z, qw, qx, qy, qz].
        # This initial value is not used until a target is received.
        self.target_ee_pose = np.array(
            [0.35, 0.0, 0.225, 0.7071068, 0.0, 0.7071068, 0.0],
            dtype=np.float32,
        )
        self.last_action = np.zeros(6, dtype=np.float32)
        self.has_received_target = False

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            JointState,
            js_topic,
            self._joint_state_cb,
            sensor_qos,
        )
        self.create_subscription(
            PoseStamped,
            target_topic,
            self._target_pose_cb,
            10,
        )
        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            cmd_topic,
            1,
        )

        self.timer = self.create_timer(
            1.0 / self.freq_hz,
            self._control_loop,
        )

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.publish_marker = bool(
            self.get_parameter("publish_workspace_marker").value
        )
        if self.publish_marker:
            box_topic = str(self.get_parameter("box_marker_topic").value)
            self.box_pub = self.create_publisher(Marker, box_topic, 10)
            self.bounding_box_pub = self.create_publisher(
                Marker, "/ur3_reach/bounding_box_marker", 10
            )
            self.marker_timer = self.create_timer(
                1.0, self._publish_workspace_box_marker
            )

        status_str = (
            "COMMAND PUBLISHING ENABLED"
            if self.enable_execution
            else "DRY-RUN (NO COMMANDS)"
        )
        self.get_logger().info(
            f"UR3 RL Policy Node running at {self.freq_hz} Hz. "
            f"Mode: [{status_str}]."
        )

    def _joint_state_cb(self, msg: JointState):
        """Reorder feedback by joint name."""
        name_to_idx = {
            name: index for index, name in enumerate(msg.name)
        }

        if all(name in name_to_idx for name in JOINT_NAMES):
            self.latest_joint_pos = np.array(
                [
                    msg.position[name_to_idx[name]]
                    for name in JOINT_NAMES
                ],
                dtype=np.float32,
            )

            if len(msg.velocity) >= len(msg.name):
                self.latest_joint_vel = np.array(
                    [
                        msg.velocity[name_to_idx[name]]
                        for name in JOINT_NAMES
                    ],
                    dtype=np.float32,
                )
            else:
                self.latest_joint_vel = np.zeros(
                    6, dtype=np.float32
                )

            self.last_joint_state_time = time.time()

    def _target_pose_cb(self, msg: PoseStamped):
        """Accept a target already expressed in the training frame.

        No TF transformation is performed here.
        """
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        within_bounds = (
            self.x_bounds[0] <= x <= self.x_bounds[1]
            and self.y_bounds[0] <= y <= self.y_bounds[1]
            and self.z_bounds[0] <= z <= self.z_bounds[1]
        )

        if not within_bounds:
            self.get_logger().warning(
                f"Rejected target [{x:.2f}, {y:.2f}, {z:.2f}]: "
                f"Outside target workspace box. "
                f"X: {self.x_bounds}, "
                f"Y: {self.y_bounds}, "
                f"Z: {self.z_bounds}"
            )
            #return

        self.target_ee_pose[:] = [
            x,
            y,
            z,
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ]

        self.has_received_target = True

        self.get_logger().info(
            f"New reaching target received: "
            f"[{x:.3f}, {y:.3f}, {z:.3f}]"
        )

    def _control_loop(self):
        """Run inference and optionally publish joint targets."""

        # Wait until a target passes the workspace check.
        if not self.has_received_target:
            self.get_logger().info(
                "Waiting for a valid target pose...",
                throttle_duration_sec=2.0,
            )
            return

        # Skip inference if joint feedback is missing or stale.
        now = time.time()

        if (
            self.latest_joint_pos is None
            or (now - self.last_joint_state_time) > 0.10
        ):
            self.get_logger().warning(
                "Watchdog: Waiting for fresh /joint_states "
                "from UR driver...",
                throttle_duration_sec=2.0,
            )
            return

        # Build the 25-element observation.
        # [0:6]   Joint positions relative to the training reference
        # [6:12]  Joint velocities
        # [12:19] Target pose: x, y, z, qw, qx, qy, qz
        # [19:25] Previous clipped policy action
        joint_pos_rel = (
            self.latest_joint_pos - DEFAULT_JOINT_POS
        )
        joint_vel_rel = self.latest_joint_vel

        obs = np.concatenate(
            [
                joint_pos_rel,
                joint_vel_rel,
                self.target_ee_pose,
                self.last_action,
            ],
            axis=0,
        ).astype(np.float32).reshape(1, 25)

        # Policy inference.
        raw_action = self.session.run(
            None,
            {self.input_name: obs},
        )[0][0]

        action = np.clip(raw_action, -1.0, 1.0)
        self.last_action = action.copy()

        # Preserve the original action conversion.
        # Verify this matches your Isaac Lab action configuration.
        delta_q = np.clip(
            self.action_scale * action,
            -self.max_delta,
            self.max_delta,
        )

        # Forward position controller receives absolute joint angles.
        target_q = self.latest_joint_pos + delta_q

        if self.enable_execution:
            cmd_msg = Float64MultiArray()
            cmd_msg.data = target_q.tolist()
            self.cmd_pub.publish(cmd_msg)
        else:
            self.get_logger().info(
                f"[DRY-RUN] Current: "
                f"{np.round(self.latest_joint_pos, 2)} "
                f"-> Commanded Delta: {np.round(delta_q, 3)}",
                throttle_duration_sec=2.0,
            )

    def _publish_workspace_box_marker(self):
        """Publish 3D wireframe bounding box Marker (LINE_LIST) for RViz."""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.frame_id
        marker.ns = "ur3_workspace_box"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.005  # 5mm edge thickness
        marker.color.r = 0.0
        marker.color.g = 0.85
        marker.color.b = 1.0
        marker.color.a = 0.9

        x0, x1 = self.x_bounds
        y0, y1 = self.y_bounds
        z0, z1 = self.z_bounds

        corners = [
            Point(x=float(x0), y=float(y0), z=float(z0)),
            Point(x=float(x1), y=float(y0), z=float(z0)),
            Point(x=float(x1), y=float(y1), z=float(z0)),
            Point(x=float(x0), y=float(y1), z=float(z0)),
            Point(x=float(x0), y=float(y0), z=float(z1)),
            Point(x=float(x1), y=float(y0), z=float(z1)),
            Point(x=float(x1), y=float(y1), z=float(z1)),
            Point(x=float(x0), y=float(y1), z=float(z1)),
        ]

        edge_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical pillars
        ]

        marker.points = []
        for i1, i2 in edge_pairs:
            marker.points.append(corners[i1])
            marker.points.append(corners[i2])

        self.box_pub.publish(marker)
        self.bounding_box_pub.publish(marker)

    def _on_parameters_set(self, params):
        """Dynamically update workspace bounds whenever parameter values change."""
        for param in params:
            if param.name == "workspace_bounds.x_min":
                self.x_bounds = (float(param.value), self.x_bounds[1])
            elif param.name == "workspace_bounds.x_max":
                self.x_bounds = (self.x_bounds[0], float(param.value))
            elif param.name == "workspace_bounds.y_min":
                self.y_bounds = (float(param.value), self.y_bounds[1])
            elif param.name == "workspace_bounds.y_max":
                self.y_bounds = (self.y_bounds[0], float(param.value))
            elif param.name == "workspace_bounds.z_min":
                self.z_bounds = (float(param.value), self.z_bounds[1])
            elif param.name == "workspace_bounds.z_max":
                self.z_bounds = (self.z_bounds[0], float(param.value))

        self.get_logger().info(
            f"Workspace bounds dynamically updated from policy node parameters: "
            f"X=[{self.x_bounds[0]:.3f}, {self.x_bounds[1]:.3f}], "
            f"Y=[{self.y_bounds[0]:.3f}, {self.y_bounds[1]:.3f}], "
            f"Z=[{self.z_bounds[0]:.3f}, {self.z_bounds[1]:.3f}]"
        )
        if hasattr(self, "box_pub"):
            self._publish_workspace_box_marker()

        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = UR3RLPolicyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
