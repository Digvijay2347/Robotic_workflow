# Copyright (c) 2021 PickNik, Inc.
# Modifications copyright (c) 2026 ur3_rg2_moveit_config
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from ur3_rg2_moveit_config.launch_common import load_yaml


def launch_setup(context, *args, **kwargs):
    # Initialize Arguments
    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    description_package = LaunchConfiguration("description_package")
    description_file = LaunchConfiguration("description_file")
    moveit_config_package = LaunchConfiguration("moveit_config_package")
    moveit_config_file = LaunchConfiguration("moveit_config_file")
    moveit_joint_limits_file = LaunchConfiguration("moveit_joint_limits_file")
    prefix = LaunchConfiguration("prefix")
    use_sim_time = LaunchConfiguration("use_sim_time")
    launch_rviz = LaunchConfiguration("launch_rviz")
    launch_servo = LaunchConfiguration("launch_servo")

    # Robot description from ur3_rg2_description
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare(description_package), "urdf", description_file]),
            " ",
            "robot_ip:=",
            robot_ip,
            " ",
            "use_fake_hardware:=",
            use_fake_hardware,
            " ",
            "tf_prefix:=",
            prefix,
            " ",
        ]
    )
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    # Semantic Robot Description (SRDF) from ur3_rg2_moveit_config
    robot_description_semantic_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare(moveit_config_package), "srdf", moveit_config_file]
            ),
            " ",
            "name:=",
            "ur3_rg2",
            " ",
            "prefix:=",
            prefix,
            " ",
        ]
    )
    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(
            robot_description_semantic_content, value_type=str
        )
    }

    # Kinematics
    robot_description_kinematics = PathJoinSubstitution(
        [FindPackageShare(moveit_config_package), "config", "kinematics.yaml"]
    )

    # Joint limits / Planning parameters
    robot_description_planning = {
        "robot_description_planning": load_yaml(
            str(moveit_config_package.perform(context)),
            os.path.join("config", str(moveit_joint_limits_file.perform(context))),
        )
    }

    # Planning Configuration (OMPL)
    ompl_planning_pipeline_config = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": (
                "default_planner_request_adapters/AddTimeOptimalParameterization "
                "default_planner_request_adapters/FixWorkspaceBounds "
                "default_planner_request_adapters/FixStartStateBounds "
                "default_planner_request_adapters/FixStartStateCollision "
                "default_planner_request_adapters/FixStartStatePathConstraints"
            ),
            "start_state_max_bounds_error": 0.1,
        }
    }
    ompl_planning_yaml = load_yaml("ur3_rg2_moveit_config", "config/ompl_planning.yaml")
    ompl_planning_pipeline_config["move_group"].update(ompl_planning_yaml)

    # Trajectory Execution Configuration
    controllers_yaml = load_yaml("ur3_rg2_moveit_config", "config/controllers.yaml")
    change_controllers = context.perform_substitution(use_sim_time)
    if change_controllers == "true":
        controllers_yaml["scaled_joint_trajectory_controller"]["default"] = False
        controllers_yaml["joint_trajectory_controller"]["default"] = True

    moveit_controllers = {
        "moveit_simple_controller_manager": controllers_yaml,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }

    trajectory_execution = {
        "moveit_manage_controllers": False,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
        "trajectory_execution.execution_duration_monitoring": False,
    }

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    # Move Group Node
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            {"publish_robot_description_semantic": True},
            robot_description_kinematics,
            robot_description_planning,
            ompl_planning_pipeline_config,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
            {"use_sim_time": use_sim_time},
        ],
    )

    # RViz with MoveIt visualizer
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare(moveit_config_package), "rviz", "view_robot.rviz"]
    )
    rviz_node = Node(
        package="rviz2",
        condition=IfCondition(launch_rviz),
        executable="rviz2",
        name="rviz2_moveit",
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            ompl_planning_pipeline_config,
            robot_description_kinematics,
            robot_description_planning,
            {"use_sim_time": use_sim_time},
        ],
    )

    # MoveIt Servo node (optional)
    servo_yaml = load_yaml("ur3_rg2_moveit_config", "config/ur_servo.yaml")
    servo_params = {"moveit_servo": servo_yaml}
    servo_node = Node(
        package="moveit_servo",
        condition=IfCondition(launch_servo),
        executable="servo_node_main",
        parameters=[
            servo_params,
            robot_description,
            robot_description_semantic,
        ],
        output="screen",
    )

    return [move_group_node, rviz_node, servo_node]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "robot_ip",
            default_value="0.0.0.0",
            description="IP address of physical UR3 robot controller.",
        ),
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description="Use mock hardware components for simulation/testing.",
        ),
        DeclareLaunchArgument(
            "description_package",
            default_value="ur3_rg2_description",
            description="Description package containing URDF/XACRO model.",
        ),
        DeclareLaunchArgument(
            "description_file",
            default_value="ur3_rg2.urdf.xacro",
            description="URDF/XACRO description file name.",
        ),
        DeclareLaunchArgument(
            "moveit_config_package",
            default_value="ur3_rg2_moveit_config",
            description="MoveIt config package name.",
        ),
        DeclareLaunchArgument(
            "moveit_config_file",
            default_value="ur3_rg2.srdf.xacro",
            description="SRDF description file name.",
        ),
        DeclareLaunchArgument(
            "moveit_joint_limits_file",
            default_value="joint_limits.yaml",
            description="MoveIt joint limits configuration YAML file.",
        ),
        DeclareLaunchArgument(
            "prefix",
            default_value="",
            description="Prefix for robot joint and link names.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation clock time if true.",
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz with MoveIt plugin.",
        ),
        DeclareLaunchArgument(
            "launch_servo",
            default_value="false",
            description="Launch MoveIt Servo node for real-time control.",
        ),
    ]

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
