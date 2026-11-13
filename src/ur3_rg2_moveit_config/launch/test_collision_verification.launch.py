# Copyright (c) 2026 ur3_rg2_moveit_config
# SPDX-License-Identifier: BSD-3-Clause

from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("ur3_rg2_description"), "urdf", "ur3_rg2.urdf.xacro"]
            ),
            " ",
            "use_fake_hardware:=true",
            " ",
        ]
    )
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    robot_description_semantic_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("ur3_rg2_moveit_config"), "srdf", "ur3_rg2.srdf.xacro"]
            ),
            " ",
            "name:=ur3_rg2",
            " ",
        ]
    )
    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(
            robot_description_semantic_content, value_type=str
        )
    }

    test_node = Node(
        package="ur3_rg2_moveit_config",
        executable="test_collision_matrix",
        name="test_collision_matrix",
        output="screen",
        parameters=[robot_description, robot_description_semantic],
    )

    return LaunchDescription([test_node])
