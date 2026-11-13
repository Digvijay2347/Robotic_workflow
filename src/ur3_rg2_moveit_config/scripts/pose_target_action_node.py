#!/usr/bin/env python3

# Copyright (c) 2026 ur3_rg2_moveit_config
# SPDX-License-Identifier: BSD-3-Clause

"""ROS 2 Node: Subscribes to /target_pose and dispatches MoveIt 2 Action Goals.

Features:
- Subscribes to /target_pose (geometry_msgs/msg/PoseStamped)
- Targets the combined gripper TCP frame: rg2_gripper_tcp
- Dispatches goal to MoveIt 2 action server: /move_action (MoveGroup)
- Streams real-time execution feedback (PLANNING -> MONITOR -> SUCCESS)
- Handles goal preemption: if a new target arrives mid-trajectory, smoothly cancels
  the existing goal and routes to the new target.
- Publishes execution status on /target_pose_status (std_msgs/msg/String)
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    MotionPlanRequest,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
)


ERROR_CODE_NAMES = {
    MoveItErrorCodes.SUCCESS: "SUCCESS",
    MoveItErrorCodes.FAILURE: "FAILURE",
    MoveItErrorCodes.PLANNING_FAILED: "PLANNING_FAILED",
    MoveItErrorCodes.INVALID_MOTION_PLAN: "INVALID_MOTION_PLAN",
    MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE: "ENVIRONMENT_CHANGED",
    MoveItErrorCodes.CONTROL_FAILED: "CONTROL_FAILED",
    MoveItErrorCodes.UNABLE_TO_AQUIRE_SENSOR_DATA: "UNABLE_TO_AQUIRE_SENSOR_DATA",
    MoveItErrorCodes.TIMED_OUT: "TIMED_OUT",
    MoveItErrorCodes.PREEMPTED: "PREEMPTED",
    MoveItErrorCodes.START_STATE_IN_COLLISION: "START_STATE_IN_COLLISION",
    MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
    MoveItErrorCodes.GOAL_IN_COLLISION: "GOAL_IN_COLLISION",
    MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS: "GOAL_VIOLATES_PATH_CONSTRAINTS",
    MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED: "GOAL_CONSTRAINTS_VIOLATED",
    MoveItErrorCodes.INVALID_GROUP_NAME: "INVALID_GROUP_NAME",
    MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS: "INVALID_GOAL_CONSTRAINTS",
    MoveItErrorCodes.INVALID_ROBOT_STATE: "INVALID_ROBOT_STATE",
    MoveItErrorCodes.INVALID_LINK_NAME: "INVALID_LINK_NAME",
    MoveItErrorCodes.NO_IK_SOLUTION: "NO_IK_SOLUTION",
}


class PoseTargetActionNode(Node):
    def __init__(self):
        super().__init__("pose_target_action_node")

        # -------------------------------------------------------------
        # Node Parameters
        # -------------------------------------------------------------
        self.declare_parameter("planning_group", "ur_manipulator")
        self.declare_parameter("tcp_link", "rg2_gripper_tcp")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("velocity_scaling", 0.25)
        self.declare_parameter("acceleration_scaling", 0.25)
        self.declare_parameter("planning_time", 5.0)
        self.declare_parameter("plan_only", False)
        self.declare_parameter("preempt_on_new_target", True)
        self.declare_parameter("pos_tolerance_m", 0.005)      # 5 mm
        self.declare_parameter("orient_tolerance_rad", 0.05)  # ~2.8 deg

        self.planning_group = self.get_parameter("planning_group").value
        self.tcp_link = self.get_parameter("tcp_link").value
        self.base_frame = self.get_parameter("base_frame").value
        self.velocity_scaling = self.get_parameter("velocity_scaling").value
        self.acceleration_scaling = self.get_parameter("acceleration_scaling").value
        self.planning_time = self.get_parameter("planning_time").value
        self.plan_only = self.get_parameter("plan_only").value
        self.preempt_on_new_target = self.get_parameter("preempt_on_new_target").value
        self.pos_tolerance = self.get_parameter("pos_tolerance_m").value
        self.orient_tolerance = self.get_parameter("orient_tolerance_rad").value

        # State management
        self.current_goal_handle = None
        self.is_executing = False
        self.pending_target_pose = None

        # -------------------------------------------------------------
        # ROS 2 Interfaces
        # -------------------------------------------------------------
        # Status Publisher
        self.status_pub = self.create_publisher(String, "/target_pose_status", 10)

        # Action Client to MoveIt
        self.action_client = ActionClient(self, MoveGroup, "/move_action")

        # Target Pose Subscriber
        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/target_pose",
            self.target_pose_callback,
            10,
        )

        self.publish_status("INITIALIZING: Waiting for MoveIt 2 action server...")
        self.get_logger().info("Connecting to MoveIt 2 '/move_action' action server...")
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().warn(
                "MoveIt action server not available yet. Will retry when targets arrive."
            )
        else:
            self.get_logger().info("Connected to MoveIt 2 action server!")
            self.publish_status("IDLE: Ready for target poses on /target_pose")

        self.get_logger().info(
            f"PoseTargetActionNode ready. Group: '{self.planning_group}', "
            f"TCP Link: '{self.tcp_link}', Frame: '{self.base_frame}'"
        )

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(f"[STATUS] {text}")

    # -----------------------------------------------------------------
    # Target Pose Callback
    # -----------------------------------------------------------------
    def target_pose_callback(self, msg: PoseStamped):
        pos = msg.pose.position
        ori = msg.pose.orientation
        frame = msg.header.frame_id if msg.header.frame_id else self.base_frame

        self.get_logger().info("=" * 60)
        self.get_logger().info(
            f"Received NEW target pose on /target_pose:\n"
            f"  Position:    X={pos.x:.4f}, Y={pos.y:.4f}, Z={pos.z:.4f} [m]\n"
            f"  Orientation: X={ori.x:.4f}, Y={ori.y:.4f}, Z={ori.z:.4f}, W={ori.w:.4f}\n"
            f"  Reference:   {frame} | Target Link: {self.tcp_link}"
        )

        # Handle busy state / active execution
        if self.is_executing and self.current_goal_handle is not None:
            if self.preempt_on_new_target:
                self.get_logger().warn(
                    "Robot is currently executing! Preempting active goal for new target..."
                )
                self.pending_target_pose = msg
                cancel_future = self.current_goal_handle.cancel_goal_async()
                cancel_future.add_done_callback(self.cancel_done_callback)
                return
            else:
                self.get_logger().warn(
                    "Robot is currently executing! Rejecting new target (preemption disabled)."
                )
                self.publish_status("REJECTED: Robot is busy with previous motion")
                return

        self.dispatch_goal(msg)

    # -----------------------------------------------------------------
    # Goal Construction & Dispatch
    # -----------------------------------------------------------------
    def dispatch_goal(self, target_pose: PoseStamped):
        if not self.action_client.server_is_ready():
            self.get_logger().error("Cannot send goal: MoveIt action server not ready!")
            self.publish_status("ERROR: MoveIt action server not available")
            return

        ref_frame = target_pose.header.frame_id if target_pose.header.frame_id else self.base_frame

        # 1. Motion Plan Request
        req = MotionPlanRequest()
        req.group_name = self.planning_group
        req.num_planning_attempts = 10
        req.allowed_planning_time = self.planning_time
        req.max_velocity_scaling_factor = self.velocity_scaling
        req.max_acceleration_scaling_factor = self.acceleration_scaling

        # Position Constraint
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = ref_frame
        pos_constraint.link_name = self.tcp_link

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self.pos_tolerance]

        pos_constraint.constraint_region.primitives.append(sphere)
        pos_constraint.constraint_region.primitive_poses.append(target_pose.pose)
        pos_constraint.weight = 1.0

        # Orientation Constraint
        orient_constraint = OrientationConstraint()
        orient_constraint.header.frame_id = ref_frame
        orient_constraint.link_name = self.tcp_link
        orient_constraint.orientation = target_pose.pose.orientation
        orient_constraint.absolute_x_axis_tolerance = self.orient_tolerance
        orient_constraint.absolute_y_axis_tolerance = self.orient_tolerance
        orient_constraint.absolute_z_axis_tolerance = self.orient_tolerance
        orient_constraint.weight = 1.0

        # Combine constraints into goal
        goal_constraints = Constraints()
        goal_constraints.position_constraints.append(pos_constraint)
        goal_constraints.orientation_constraints.append(orient_constraint)
        req.goal_constraints.append(goal_constraints)

        # 2. Planning Options
        goal_msg = MoveGroup.Goal()
        goal_msg.request = req
        goal_msg.planning_options = PlanningOptions()
        goal_msg.planning_options.plan_only = self.plan_only
        goal_msg.planning_options.replan = True
        goal_msg.planning_options.replan_attempts = 3

        # 3. Send Goal
        self.is_executing = True
        mode_str = "PLAN-ONLY" if self.plan_only else "PLAN & EXECUTE"
        self.publish_status(f"DISPATCHING ({mode_str}) to {self.tcp_link}...")

        send_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.action_feedback_callback,
        )
        send_future.add_done_callback(self.goal_response_callback)

    # -----------------------------------------------------------------
    # Action Callbacks
    # -----------------------------------------------------------------
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.is_executing = False
            self.publish_status("REJECTED: MoveIt rejected the goal request!")
            return

        self.current_goal_handle = goal_handle
        self.publish_status("ACCEPTED: MoveIt accepted goal. Planning path...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def action_feedback_callback(self, feedback_msg):
        state = feedback_msg.feedback.state
        self.get_logger().info(f"[MoveIt Feedback] State: {state}")
        self.publish_status(f"MOVING: MoveIt state '{state}'")

    def goal_result_callback(self, future):
        self.is_executing = False
        self.current_goal_handle = None

        result = future.result().result
        error_val = result.error_code.val
        error_name = ERROR_CODE_NAMES.get(error_val, f"CODE_{error_val}")

        if error_val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(
                f"Motion COMPLETED SUCCESSFULLY to target pose! Planning time: {result.planning_time:.3f}s"
            )
            self.publish_status("SUCCESS: Arrived at target pose!")
        else:
            self.get_logger().error(f"Motion FAILED with MoveIt Error: {error_name} ({error_val})")
            self.publish_status(f"FAILED: MoveIt error {error_name}")

        # Check if another target arrived while preempting
        if self.pending_target_pose is not None:
            next_pose = self.pending_target_pose
            self.pending_target_pose = None
            self.get_logger().info("Dispatching queued preemption target...")
            self.dispatch_goal(next_pose)

    def cancel_done_callback(self, future):
        self.get_logger().info("Previous goal cancellation acknowledged by MoveIt.")


def main(args=None):
    rclpy.init(args=args)
    node = PoseTargetActionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down PoseTargetActionNode...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
