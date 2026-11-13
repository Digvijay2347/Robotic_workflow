# Copyright (c) 2026 ur3_rg2_moveit_config
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone interactive demo launch file for UR3 + OnRobot RG2 MoveIt 2.

Launches mock hardware (or physical robot), MoveIt 2 move_group with collision checking,
and RViz 2 with interactive marker. Both planning and trajectory execution work out-of-the-box.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="true",
            description="Run with mock hardware interface (true) or physical robot motors (false).",
        ),
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.1.10",
            description="IP address of physical UR3 controller or URSim virtual machine.",
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz with MoveIt visualizer.",
        ),
        DeclareLaunchArgument(
            "prefix",
            default_value="",
            description="Prefix for robot joint and link names.",
        ),
    ]

    # Include planning and execution launch file
    planning_execution_launch = IncludeLaunchDescription(
        PathJoinSubstitution(
            [
                FindPackageShare("ur3_rg2_moveit_config"),
                "launch",
                "ur3_rg2_planning_execution.launch.py",
            ]
        ),
        launch_arguments={
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "robot_ip": LaunchConfiguration("robot_ip"),
            "prefix": LaunchConfiguration("prefix"),
            "launch_rviz": LaunchConfiguration("launch_rviz"),
        }.items(),
    )

    return LaunchDescription(declared_arguments + [planning_execution_launch])
