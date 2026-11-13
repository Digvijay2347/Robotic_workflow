#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped, Point
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
    MarkerArray,
)
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler

from tf2_ros import Buffer, TransformListener
import tf2_ros


class InteractiveTargetPublisher(Node):
    """MoveIt RViz GUI-style interactive marker with Free 3D Drag, Auto-Sync, and Lock & Send workflow."""

    def __init__(self):
        super().__init__("interactive_target_publisher")

        # ---------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("ee_frame_id", "tool0")
        self.declare_parameter("target_pose_topic", "/ur3_reach/target_pose")
        self.declare_parameter("bounds_marker_topic", "/ur3_reach/workspace_bounds_marker")
        self.declare_parameter("box_marker_topic", "/ur3_reach/workspace_box_marker")

        # Fallback initial pose (used only if TF is not available)
        self.declare_parameter("initial_pose.x", 0.35)
        self.declare_parameter("initial_pose.y", 0.00)
        self.declare_parameter("initial_pose.z", 0.22)

        # 360 Workshell bounds (cylindrical radius + Z + Cartesian)
        self.declare_parameter("workspace_bounds.radius_min", 0.10)
        self.declare_parameter("workspace_bounds.radius_max", 0.48)
        self.declare_parameter("workspace_bounds.x_min", -0.15)
        self.declare_parameter("workspace_bounds.x_max", 0.48)
        self.declare_parameter("workspace_bounds.y_min", -0.48)
        self.declare_parameter("workspace_bounds.y_max", 0.48)
        self.declare_parameter("workspace_bounds.z_min", 0.05)
        self.declare_parameter("workspace_bounds.z_max", 0.40)

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.ee_frame = str(self.get_parameter("ee_frame_id").value)
        self.target_topic = str(self.get_parameter("target_pose_topic").value)
        self.bounds_topic = str(self.get_parameter("bounds_marker_topic").value)
        self.box_topic = str(self.get_parameter("box_marker_topic").value)

        self.r_bounds = (
            float(self.get_parameter("workspace_bounds.radius_min").value),
            float(self.get_parameter("workspace_bounds.radius_max").value),
        )
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

        # Default orientation: Wrist pointing downwards in base frame
        self.default_orientation = (0.7071068, 0.0, 0.7071068, 0.0)

        # State tracking
        self.current_pose = Pose()
        self.current_pose.position.x = float(self.get_parameter("initial_pose.x").value)
        self.current_pose.position.y = float(self.get_parameter("initial_pose.y").value)
        self.current_pose.position.z = float(self.get_parameter("initial_pose.z").value)
        self.current_pose.orientation.w = self.default_orientation[0]
        self.current_pose.orientation.x = self.default_orientation[1]
        self.current_pose.orientation.y = self.default_orientation[2]
        self.current_pose.orientation.z = self.default_orientation[3]

        self.has_synced_ee = False
        self.is_actively_dragging = False
        self.last_sent_time = 0.0

        # Publishers & TF Listener
        self.target_pub = self.create_publisher(PoseStamped, self.target_topic, 10)
        self.bounds_pub = self.create_publisher(MarkerArray, self.bounds_topic, 10)
        self.box_pub = self.create_publisher(Marker, self.box_topic, 10)
        self.bounding_box_pub = self.create_publisher(Marker, "/ur3_reach/bounding_box_marker", 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.ee_candidate_frames = [self.ee_frame, "rg2_tcp", "tool0", "wrist_3_link"]

        # Interactive Marker Server
        self.server = InteractiveMarkerServer(self, "target_pose_marker")

        # Right-click Context Menu
        self.menu_handler = MenuHandler()
        self.menu_handler.insert("🚀 Lock & Send Target Pose", callback=self._menu_lock_and_send)
        self.menu_handler.insert("📍 Snap Marker to Current Robot EE", callback=self._menu_snap_to_ee)
        self.menu_handler.insert("🔄 Reset Orientation (Downward Gripper)", callback=self._menu_reset_orientation)

        # Initialize full 6-DoF interactive marker
        self._init_marker()

        # Startup timer: repeatedly check TF to snap marker directly to robot EE on boot
        self.startup_sync_timer = self.create_timer(0.2, self._startup_sync_callback)

        # 1 Hz Heartbeat timer: keeps DDS topic alive so it appears in RViz dropdown
        self.heartbeat_timer = self.create_timer(1.0, self._heartbeat_callback)

        # 1 Hz Timer for 360 workshell RViz visualization
        self.bounds_timer = self.create_timer(1.0, self._publish_360_workshell_bounds)

        self.get_logger().info(
            f"InteractiveTargetPublisher initialized in LOCK & SEND mode.\n"
            f"  - Target Pose Topic: {self.target_topic}\n"
            f"  - Interactive Marker Update Topic: /target_pose_marker/update\n"
            f"  - 360 Workshell Limits: Radius=[{self.r_bounds[0]:.2f}, {self.r_bounds[1]:.2f}]m, Z=[{self.z_bounds[0]:.2f}, {self.z_bounds[1]:.2f}]m\n"
            f"  - Target publishing is LOCKED while moving. Click '[LOCK & SEND]' or right-click menu to send to robot."
        )

    def _init_marker(self):
        """Create multi-mode interactive marker (Free View Drag + Planar + Axes + Rotation + Send Button)."""
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.frame_id
        int_marker.header.stamp = self.get_clock().now().to_msg()
        int_marker.name = "ur3_reach_target"
        int_marker.description = "UR3 Target Pose\n(Drag sphere/arrows to adjust, then click [LOCK & SEND])"
        int_marker.scale = 0.18
        int_marker.pose = self.current_pose

        # -----------------------------------------------------
        # 1. Center Sphere with Free View-Plane Dragging
        # -----------------------------------------------------
        ctrl_center = InteractiveMarkerControl()
        ctrl_center.name = "center_free_drag"
        ctrl_center.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        ctrl_center.orientation_mode = InteractiveMarkerControl.VIEW_FACING
        ctrl_center.always_visible = True

        # Target sphere
        self.sphere_marker = Marker()
        self.sphere_marker.type = Marker.SPHERE
        self.sphere_marker.scale.x = 0.045
        self.sphere_marker.scale.y = 0.045
        self.sphere_marker.scale.z = 0.045
        self.sphere_marker.color.r = 0.1
        self.sphere_marker.color.g = 0.75
        self.sphere_marker.color.b = 0.95
        self.sphere_marker.color.a = 0.85
        ctrl_center.markers.append(self.sphere_marker)

        # Orientation arrow
        arrow = Marker()
        arrow.type = Marker.ARROW
        arrow.scale.x = 0.08
        arrow.scale.y = 0.012
        arrow.scale.z = 0.012
        arrow.color.r = 1.0
        arrow.color.g = 0.85
        arrow.color.b = 0.1
        arrow.color.a = 0.95
        ctrl_center.markers.append(arrow)

        int_marker.controls.append(ctrl_center)

        # -----------------------------------------------------
        # 2. Clickable 3D "[ LOCK & SEND ]" Button in RViz
        # -----------------------------------------------------
        ctrl_btn = InteractiveMarkerControl()
        ctrl_btn.name = "lock_send_button"
        ctrl_btn.interaction_mode = InteractiveMarkerControl.BUTTON
        ctrl_btn.orientation_mode = InteractiveMarkerControl.VIEW_FACING
        ctrl_btn.always_visible = True

        # Floating text above the marker
        self.btn_text = Marker()
        self.btn_text.type = Marker.TEXT_VIEW_FACING
        self.btn_text.text = "[ 🔒 LOCK & SEND ]"
        self.btn_text.scale.z = 0.035
        self.btn_text.pose.position.z = 0.08
        self.btn_text.color.r = 0.1
        self.btn_text.color.g = 1.0
        self.btn_text.color.b = 0.4
        self.btn_text.color.a = 1.0
        ctrl_btn.markers.append(self.btn_text)

        int_marker.controls.append(ctrl_btn)

        # -----------------------------------------------------
        # 3. 2D Planar Translation Handles (XY, XZ, YZ planes)
        # -----------------------------------------------------
        for name, q in [
            ("plane_xy", (0.7071, 0.0, 0.0, 0.7071)),
            ("plane_xz", (0.7071, 0.0, 0.7071, 0.0)),
            ("plane_yz", (0.7071, 0.7071, 0.0, 0.0)),
        ]:
            plane_ctrl = InteractiveMarkerControl()
            plane_ctrl.name = name
            plane_ctrl.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
            plane_ctrl.orientation.w, plane_ctrl.orientation.x, plane_ctrl.orientation.y, plane_ctrl.orientation.z = q
            int_marker.controls.append(plane_ctrl)

        # -----------------------------------------------------
        # 4. 1D Translation Arrows (X, Y, Z axes)
        # -----------------------------------------------------
        for axis, q in [
            ("x", (0.7071, 0.7071, 0.0, 0.0)),
            ("z", (0.7071, 0.0, 0.7071, 0.0)),
            ("y", (0.7071, 0.0, 0.0, 0.7071)),
        ]:
            move_ctrl = InteractiveMarkerControl()
            move_ctrl.name = f"move_{axis}"
            move_ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            move_ctrl.orientation.w, move_ctrl.orientation.x, move_ctrl.orientation.y, move_ctrl.orientation.z = q
            int_marker.controls.append(move_ctrl)

        # -----------------------------------------------------
        # 5. 1D Rotation Rings (Roll, Pitch, Yaw)
        # -----------------------------------------------------
        for axis, q in [
            ("x", (0.7071, 0.7071, 0.0, 0.0)),
            ("z", (0.7071, 0.0, 0.7071, 0.0)),
            ("y", (0.7071, 0.0, 0.0, 0.7071)),
        ]:
            rot_ctrl = InteractiveMarkerControl()
            rot_ctrl.name = f"rotate_{axis}"
            rot_ctrl.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            rot_ctrl.orientation.w, rot_ctrl.orientation.x, rot_ctrl.orientation.y, rot_ctrl.orientation.z = q
            int_marker.controls.append(rot_ctrl)

        self.server.insert(int_marker, feedback_callback=self._process_feedback)
        self.menu_handler.apply(self.server, "ur3_reach_target")
        self.server.applyChanges()

    def _startup_sync_callback(self):
        """Poll TF until the robot EE pose is available, then snap marker to robot EE."""
        if self._sync_marker_with_robot_ee(quiet=True):
            self.has_synced_ee = True
            if self.startup_sync_timer is not None:
                self.startup_sync_timer.cancel()
                self.startup_sync_timer = None
            self.get_logger().info(
                f"Marker placed at robot EE home pose: "
                f"X={self.current_pose.position.x:.3f}, "
                f"Y={self.current_pose.position.y:.3f}, "
                f"Z={self.current_pose.position.z:.3f} (Locked: Not publishing until you click [LOCK & SEND])"
            )

    def _heartbeat_callback(self):
        """Keep DDS topic active so RViz dropdown reliably detects /target_pose_marker/update."""
        if not self.is_actively_dragging:
            self.server.applyChanges()

    def _sync_marker_with_robot_ee(self, quiet: bool = False) -> bool:
        """Lookup current transform of the robot end-effector and place marker on it."""
        for frame in self.ee_candidate_frames:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.frame_id,
                    frame,
                    rclpy.time.Time(),
                )
                self.current_pose.position.x = t.transform.translation.x
                self.current_pose.position.y = t.transform.translation.y
                self.current_pose.position.z = t.transform.translation.z
                self.current_pose.orientation = t.transform.rotation

                self.server.setPose("ur3_reach_target", self.current_pose)
                self.server.applyChanges()
                if not quiet:
                    self.get_logger().info(
                        f"Snapped marker to robot EE link '{frame}': "
                        f"[{self.current_pose.position.x:.3f}, {self.current_pose.position.y:.3f}, {self.current_pose.position.z:.3f}]"
                    )
                return True
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue
            except Exception as e:
                if not quiet:
                    self.get_logger().warning(f"Error querying TF for EE sync: {e}")
                return False
        return False

    def _is_within_360_bounds(self, pos: Point) -> bool:
        """Verify if point lies inside the 360 degree reach shell."""
        radius_xy = math.hypot(pos.x, pos.y)
        within_radius = self.r_bounds[0] <= radius_xy <= self.r_bounds[1]
        within_z = self.z_bounds[0] <= pos.z <= self.z_bounds[1]
        within_x = self.x_bounds[0] <= pos.x <= self.x_bounds[1]
        within_y = self.y_bounds[0] <= pos.y <= self.y_bounds[1]
        return within_radius and within_z and within_x and within_y

    def _process_feedback(self, feedback: InteractiveMarkerFeedback):
        """Handle marker interaction from RViz."""
        self.current_pose = feedback.pose

        # Right-click context menu event
        if feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
            self.menu_handler.processFeedback(feedback)
            return

        # Click on 3D "[ LOCK & SEND ]" button
        if feedback.control_name == "lock_send_button" and feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            self._lock_and_send_target()
            return

        # User is dragging / adjusting the marker
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            self.is_actively_dragging = True
            # DO NOT PUBLISH TARGET WHILE MOVING.
            # Only update local position and bounds checking.
            return

        # Mouse released
        if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            self.is_actively_dragging = False
            # Still DO NOT PUBLISH on release. User will review pose and explicitly click [LOCK & SEND].
            r_xy = math.hypot(feedback.pose.position.x, feedback.pose.position.y)
            if self._is_within_360_bounds(feedback.pose.position):
                self.get_logger().info(
                    f"Pose adjusted: X={feedback.pose.position.x:.3f}, "
                    f"Y={feedback.pose.position.y:.3f}, Z={feedback.pose.position.z:.3f} "
                    f"(radius={r_xy:.3f}m) -> Ready. Click [LOCK & SEND] or right-click menu to dispatch."
                )
            else:
                self.get_logger().warning(
                    f"Pose [{feedback.pose.position.x:.2f}, {feedback.pose.position.y:.2f}, {feedback.pose.position.z:.2f}] "
                    f"(radius={r_xy:.2f}m) is OUTSIDE 360 reach limits [{self.r_bounds[0]:.2f}, {self.r_bounds[1]:.2f}]m!"
                )

    def _lock_and_send_target(self):
        """Explicitly lock current target and publish to /ur3_reach/target_pose."""
        r_xy = math.hypot(self.current_pose.position.x, self.current_pose.position.y)
        if not self._is_within_360_bounds(self.current_pose.position):
            self.get_logger().warning(
                f"CANNOT SEND: Target position [{self.current_pose.position.x:.2f}, "
                f"{self.current_pose.position.y:.2f}, {self.current_pose.position.z:.2f}] "
                f"(radius={r_xy:.2f}m) is outside the 360 workshell reach limits! "
                f"Radius must be between [{self.r_bounds[0]:.2f}, {self.r_bounds[1]:.2f}]m, "
                f"Z must be between [{self.z_bounds[0]:.2f}, {self.z_bounds[1]:.2f}]m."
            )
            return

        # Publish target pose
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose = self.current_pose
        self.target_pub.publish(msg)

        self.last_sent_time = time.time()
        self.get_logger().info(
            f"🚀 TARGET LOCKED & SENT TO UR3:\n"
            f"   Position: X={self.current_pose.position.x:.3f}, Y={self.current_pose.position.y:.3f}, Z={self.current_pose.position.z:.3f} (Radius={r_xy:.3f}m)\n"
            f"   Orientation: [w={self.current_pose.orientation.w:.4f}, x={self.current_pose.orientation.x:.4f}, y={self.current_pose.orientation.y:.4f}, z={self.current_pose.orientation.z:.4f}]"
        )

    # Menu Callbacks
    def _menu_lock_and_send(self, feedback: InteractiveMarkerFeedback):
        self._lock_and_send_target()

    def _menu_snap_to_ee(self, feedback: InteractiveMarkerFeedback):
        if self._sync_marker_with_robot_ee(quiet=False):
            self.get_logger().info("Marker snapped to robot EE position via menu.")
        else:
            self.get_logger().warning("Could not lookup robot EE transform from TF.")

    def _menu_reset_orientation(self, feedback: InteractiveMarkerFeedback):
        self.current_pose.orientation.w = self.default_orientation[0]
        self.current_pose.orientation.x = self.default_orientation[1]
        self.current_pose.orientation.y = self.default_orientation[2]
        self.current_pose.orientation.z = self.default_orientation[3]

        self.server.setPose("ur3_reach_target", self.current_pose)
        self.server.applyChanges()
        self.get_logger().info("Reset target orientation to downward pointing.")

    def _publish_360_workshell_bounds(self):
        """Publish full 360 degree cylindrical workshell markers to RViz."""
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        z_center = (self.z_bounds[0] + self.z_bounds[1]) / 2.0
        z_height = max(0.01, self.z_bounds[1] - self.z_bounds[0])

        # 1. Outer Reach Cylinder (Translucent Cyan)
        outer_cyl = Marker()
        outer_cyl.header.stamp = now
        outer_cyl.header.frame_id = self.frame_id
        outer_cyl.ns = "ur3_360_workshell"
        outer_cyl.id = 0
        outer_cyl.type = Marker.CYLINDER
        outer_cyl.action = Marker.ADD
        outer_cyl.pose.position.x = 0.0
        outer_cyl.pose.position.y = 0.0
        outer_cyl.pose.position.z = z_center
        outer_cyl.pose.orientation.w = 1.0
        outer_cyl.scale.x = 2.0 * self.r_bounds[1]
        outer_cyl.scale.y = 2.0 * self.r_bounds[1]
        outer_cyl.scale.z = z_height
        outer_cyl.color.r = 0.2
        outer_cyl.color.g = 0.7
        outer_cyl.color.b = 1.0
        outer_cyl.color.a = 0.08
        marker_array.markers.append(outer_cyl)

        # 2. Inner Base Exclusion Cylinder (Translucent Red)
        inner_cyl = Marker()
        inner_cyl.header.stamp = now
        inner_cyl.header.frame_id = self.frame_id
        inner_cyl.ns = "ur3_360_workshell"
        inner_cyl.id = 1
        inner_cyl.type = Marker.CYLINDER
        inner_cyl.action = Marker.ADD
        inner_cyl.pose.position.x = 0.0
        inner_cyl.pose.position.y = 0.0
        inner_cyl.pose.position.z = z_center
        inner_cyl.pose.orientation.w = 1.0
        inner_cyl.scale.x = 2.0 * self.r_bounds[0]
        inner_cyl.scale.y = 2.0 * self.r_bounds[0]
        inner_cyl.scale.z = z_height
        inner_cyl.color.r = 0.9
        inner_cyl.color.g = 0.2
        inner_cyl.color.b = 0.2
        inner_cyl.color.a = 0.12
        marker_array.markers.append(inner_cyl)

        # 3. Outer boundary rings (Line Strip at top and bottom)
        for idx, z_level in enumerate([self.z_bounds[0], self.z_bounds[1]]):
            ring = Marker()
            ring.header.stamp = now
            ring.header.frame_id = self.frame_id
            ring.ns = "ur3_360_workshell"
            ring.id = 2 + idx
            ring.type = Marker.LINE_STRIP
            ring.action = Marker.ADD
            ring.pose.orientation.w = 1.0
            ring.scale.x = 0.005  # Line width
            ring.color.r = 0.3
            ring.color.g = 0.8
            ring.color.b = 1.0
            ring.color.a = 0.5

            num_points = 48
            for i in range(num_points + 1):
                theta = 2.0 * math.pi * float(i) / float(num_points)
                pt = Point()
                pt.x = self.r_bounds[1] * math.cos(theta)
                pt.y = self.r_bounds[1] * math.sin(theta)
                pt.z = z_level
                ring.points.append(pt)

            marker_array.markers.append(ring)

        # 4. 3D Workspace Bounding Box Wireframe (12 edges via LINE_LIST)
        box_wireframe = self._build_bounding_box_wireframe(now)
        marker_array.markers.append(box_wireframe)

        # 5. Translucent Bounding Box Volume (CUBE)
        box_cube = self._build_bounding_box_cube(now)
        marker_array.markers.append(box_cube)

        # 6. Workspace Box Text Label
        box_label = Marker()
        box_label.header.stamp = now
        box_label.header.frame_id = self.frame_id
        box_label.ns = "ur3_workspace_box"
        box_label.id = 10
        box_label.type = Marker.TEXT_VIEW_FACING
        box_label.action = Marker.ADD
        box_label.pose.position.x = 0.0
        box_label.pose.position.y = 0.0
        box_label.pose.position.z = self.z_bounds[1] + 0.04
        box_label.pose.orientation.w = 1.0
        box_label.scale.z = 0.035
        box_label.color.r = 0.1
        box_label.color.g = 0.85
        box_label.color.b = 1.0
        box_label.color.a = 0.95
        box_label.text = (
            f"UR3 Workspace Box [X: {self.x_bounds[0]:.2f}..{self.x_bounds[1]:.2f}, "
            f"Y: {self.y_bounds[0]:.2f}..{self.y_bounds[1]:.2f}, "
            f"Z: {self.z_bounds[0]:.2f}..{self.z_bounds[1]:.2f}]"
        )
        marker_array.markers.append(box_label)

        # Publish full MarkerArray (for RViz "MarkerArray" display)
        self.bounds_pub.publish(marker_array)

        # Publish single Marker topics (for RViz "Marker" display dropdown)
        self.box_pub.publish(box_wireframe)
        self.bounding_box_pub.publish(box_wireframe)

    def _build_bounding_box_wireframe(self, now):
        """Construct a 12-edge 3D wireframe bounding box Marker (LINE_LIST)."""
        marker = Marker()
        marker.header.stamp = now
        marker.header.frame_id = self.frame_id
        marker.ns = "ur3_workspace_box"
        marker.id = 4
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
            Point(x=x0, y=y0, z=z0),  # 0: ---
            Point(x=x1, y=y0, z=z0),  # 1: +--
            Point(x=x1, y=y1, z=z0),  # 2: ++-
            Point(x=x0, y=y1, z=z0),  # 3: -+-
            Point(x=x0, y=y0, z=z1),  # 4: --+
            Point(x=x1, y=y0, z=z1),  # 5: +- +
            Point(x=x1, y=y1, z=z1),  # 6: +++
            Point(x=x0, y=y1, z=z1),  # 7: -++
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

        return marker

    def _build_bounding_box_cube(self, now):
        """Construct a translucent 3D solid bounding box volume Marker (CUBE)."""
        marker = Marker()
        marker.header.stamp = now
        marker.header.frame_id = self.frame_id
        marker.ns = "ur3_workspace_box"
        marker.id = 5
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        x0, x1 = self.x_bounds
        y0, y1 = self.y_bounds
        z0, z1 = self.z_bounds

        marker.pose.position.x = (x0 + x1) / 2.0
        marker.pose.position.y = (y0 + y1) / 2.0
        marker.pose.position.z = (z0 + z1) / 2.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = max(0.01, abs(x1 - x0))
        marker.scale.y = max(0.01, abs(y1 - y0))
        marker.scale.z = max(0.01, abs(z1 - z0))

        marker.color.r = 0.0
        marker.color.g = 0.75
        marker.color.b = 0.95
        marker.color.a = 0.05  # Subtle translucent highlight

        return marker


def main():
    rclpy.init()
    node = InteractiveTargetPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
