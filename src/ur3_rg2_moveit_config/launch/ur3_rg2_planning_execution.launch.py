# Copyright (c) 2026 ur3_rg2_moveit_config
#
# SPDX-License-Identifier: BSD-3-Clause

"""Full planning and execution launch file connecting UR Driver and MoveIt 2.

Supports physical UR3 hardware or URSim emulator, with combined UR3 + RG2
robot description and collision-free MoveIt planning scene.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    # Resolve launch configurations early to prevent context pollution between inclusions
    use_fake_hardware_val = context.perform_substitution(
        LaunchConfiguration("use_fake_hardware")
    )
    robot_ip_val = context.perform_substitution(LaunchConfiguration("robot_ip"))
    launch_rviz_val = context.perform_substitution(LaunchConfiguration("launch_rviz"))
    prefix_val = context.perform_substitution(LaunchConfiguration("prefix"))

    # Include UR Driver (starts robot_state_publisher and ros2_control)
    # Note: we explicitly set driver launch_rviz to false so we only launch MoveIt's RViz
    ur_driver_launch = IncludeLaunchDescription(
        PathJoinSubstitution(
            [
                FindPackageShare("ur_robot_driver"),
                "launch",
                "ur_control.launch.py",
            ]
        ),
        launch_arguments={
            "ur_type": "ur3",
            "robot_ip": robot_ip_val,
            "use_fake_hardware": use_fake_hardware_val,
            "initial_joint_controller": "scaled_joint_trajectory_controller",
            "activate_joint_controller": "true",
            "launch_rviz": "false",
            "description_package": "ur3_rg2_description",
            "description_file": "ur3_rg2.urdf.xacro",
        }.items(),
    )

    # Include MoveIt 2 move_group and RViz with user's launch_rviz choice
    moveit_launch = IncludeLaunchDescription(
        PathJoinSubstitution(
            [
                FindPackageShare("ur3_rg2_moveit_config"),
                "launch",
                "ur3_rg2_moveit.launch.py",
            ]
        ),
        launch_arguments={
            "robot_ip": robot_ip_val,
            "use_fake_hardware": use_fake_hardware_val,
            "description_package": "ur3_rg2_description",
            "description_file": "ur3_rg2.urdf.xacro",
            "moveit_config_package": "ur3_rg2_moveit_config",
            "moveit_config_file": "ur3_rg2.srdf.xacro",
            "prefix": prefix_val,
            "launch_rviz": launch_rviz_val,
        }.items(),
    )

    return [ur_driver_launch, moveit_launch]


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
            description="IP address of UR3 controller or URSim virtual machine.",
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz visualizer with MoveIt motion planning display.",
        ),
        DeclareLaunchArgument(
            "prefix",
            default_value="",
            description="Prefix for robot joint and link names.",
        ),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
