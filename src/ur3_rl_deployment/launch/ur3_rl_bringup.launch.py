
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    # ---------------------------------------------------------
    # Launch arguments
    # ---------------------------------------------------------

    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip",
        default_value="127.0.0.1",
        description="IP address of the UR robot controller",
    )

    use_fake_hardware_arg = DeclareLaunchArgument(
        "use_fake_hardware",
        default_value="true",
        description="Use simulated hardware (true) or real hardware (false)",
    )

    enable_exec_arg = DeclareLaunchArgument(
        "enable_execution",
        default_value="true",
        description="Enable policy execution and robot commands",
    )

    launch_rviz_arg = DeclareLaunchArgument(
        "launch_rviz",
        default_value="true",
        description="Launch RViz via ur_control",
    )

    launch_interactive_marker_arg = DeclareLaunchArgument(
        "launch_interactive_marker",
        default_value="true",
        description="Launch MoveIt-style 6-DoF RViz interactive marker node for target pose",
    )

    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=PathJoinSubstitution([
            FindPackageShare("ur3_rl_deployment"),
            "models",
            "policy.onnx",
        ]),
        description="Absolute path to the ONNX policy model",
    )

    action_scale_arg = DeclareLaunchArgument(
        "action_scale",
        default_value="0.05",
        description="Scale factor for policy actions",
    )

    safety_max_joint_delta_arg = DeclareLaunchArgument(
        "safety_max_joint_delta",
        default_value="0.05",
        description="Maximum allowed joint delta",
    )

    control_frequency_arg = DeclareLaunchArgument(
        "control_frequency_hz",
        default_value="30.0",
        description="Policy control frequency in Hz",
    )

    kinematics_arg = DeclareLaunchArgument(
        "kinematics_config",
        default_value=PathJoinSubstitution([
            FindPackageShare("ur3_rl_deployment"),
            "config",
            "my_ur3_calibration.yaml",
        ]),
        description="Kinematics calibration file for the UR3",
    )


    # ---------------------------------------------------------
    # Universal Robots ROS 2 driver with combined UR3 + RG2 URDF
    # ---------------------------------------------------------

    ur_driver_launch = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare("ur_robot_driver"),
            "launch",
            "ur_control.launch.py",
        ]),
        launch_arguments={
            "ur_type": "ur3",
            "robot_ip": LaunchConfiguration("robot_ip"),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),

            # Custom combined robot description
            "description_package": "ur3_rg2_description",
            "description_file": "ur3_rg2.urdf.xacro",
            #verify
            "verify_robot_model": "true",

            # UR3 calibration
            "kinematics_params_file": LaunchConfiguration("kinematics_config"),

            # Controllers
            "initial_joint_controller": "forward_position_controller",
            "activate_joint_controller": "true",

            # RViz
            "launch_rviz": LaunchConfiguration("launch_rviz"),
        }.items(),
    )


    # ---------------------------------------------------------
    # Policy parameter YAML file
    # ---------------------------------------------------------

    policy_params_file = PathJoinSubstitution([
        FindPackageShare("ur3_rl_deployment"),
        "config",
        "policy_params.yaml",
    ])

    # ---------------------------------------------------------
    # RL policy deployment node
    # ---------------------------------------------------------

    policy_node = Node(
        package="ur3_rl_deployment",
        executable="ur3_rl_policy_node",
        name="ur3_rl_policy_node",
        output="screen",
        parameters=[
            policy_params_file,
            {
                "model_path": LaunchConfiguration("model_path"),

                "action_scale": ParameterValue(
                    LaunchConfiguration("action_scale"),
                    value_type=float,
                ),

                "safety_max_joint_delta": ParameterValue(
                    LaunchConfiguration("safety_max_joint_delta"),
                    value_type=float,
                ),

                "control_frequency_hz": ParameterValue(
                    LaunchConfiguration("control_frequency_hz"),
                    value_type=float,
                ),

                "enable_execution": ParameterValue(
                    LaunchConfiguration("enable_execution"),
                    value_type=bool,
                ),
            },
        ],
    )

    # ---------------------------------------------------------
    # 6-DoF RViz Interactive Marker Node (MoveIt GUI style)
    # ---------------------------------------------------------

    interactive_marker_node = Node(
        package="ur3_rl_deployment",
        executable="interactive_target_publisher",
        name="interactive_target_publisher",
        output="screen",
        condition=IfCondition(LaunchConfiguration("launch_interactive_marker")),
        parameters=[policy_params_file],
    )

    # ---------------------------------------------------------
    # Launch description
    # ---------------------------------------------------------

    return LaunchDescription([
        robot_ip_arg,
        use_fake_hardware_arg,
        enable_exec_arg,
        launch_rviz_arg,
        launch_interactive_marker_arg,
        model_path_arg,
        action_scale_arg,
        safety_max_joint_delta_arg,
        control_frequency_arg,
        kinematics_arg,
        ur_driver_launch,
        policy_node,
        interactive_marker_node,
    ])
