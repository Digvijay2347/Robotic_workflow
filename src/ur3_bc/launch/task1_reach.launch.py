from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("ur3_bc")

    # -------------------------------------------------------------------------
    # Launch Arguments
    # -------------------------------------------------------------------------
    launch_driver_arg = DeclareLaunchArgument(
        "launch_driver",
        default_value="true",
        description="Whether to launch ur_robot_driver directly (true) or policy only (false)",
    )

    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip",
        default_value="192.168.50.50",
        description="IP address of the UR robot controller",
    )

    use_fake_hardware_arg = DeclareLaunchArgument(
        "use_fake_hardware",
        default_value="false",
        description="Use fake/simulated hardware (true) or real robot (false)",
    )

    launch_rviz_arg = DeclareLaunchArgument(
        "launch_rviz",
        default_value="false",
        description="Whether to launch RViz",
    )

    description_package_arg = DeclareLaunchArgument(
        "description_package",
        default_value="environment",
        description="Package containing cell URDF/xacro",
    )

    description_file_arg = DeclareLaunchArgument(
        "description_file",
        default_value="ur3_with_table.urdf.xacro",
        description="URDF/xacro file for the robot + table cell",
    )

    # -------------------------------------------------------------------------
    # Universal Robots ROS 2 Driver Include (Optional via launch_driver:=true)
    # -------------------------------------------------------------------------
    ur_driver_launch = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare("ur_robot_driver"),
            "launch",
            "ur_control.launch.py",
        ]),
        condition=IfCondition(LaunchConfiguration("launch_driver")),
        launch_arguments={
            "ur_type": "ur3",
            "robot_ip": LaunchConfiguration("robot_ip"),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "description_package": LaunchConfiguration("description_package"),
            "description_file": LaunchConfiguration("description_file"),
            "initial_joint_controller": "forward_position_controller",
            "activate_joint_controller": "true",
            "launch_rviz": LaunchConfiguration("launch_rviz"),
        }.items(),
    )

    # -------------------------------------------------------------------------
    # Policy Node
    # -------------------------------------------------------------------------
    config_file = PathJoinSubstitution([
        pkg_share,
        "config",
        "policy_params.yaml"
    ])

    policy_node = Node(
        package="ur3_bc",
        executable="task1_reach_target",
        name="ur3_bc_task1_reach_node",
        output="screen",
        parameters=[config_file]
    )

    return LaunchDescription([
        launch_driver_arg,
        robot_ip_arg,
        use_fake_hardware_arg,
        launch_rviz_arg,
        description_package_arg,
        description_file_arg,
        ur_driver_launch,
        policy_node,
    ])
