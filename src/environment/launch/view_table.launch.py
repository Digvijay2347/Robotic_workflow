import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = FindPackageShare('environment')

    # Configurable arguments
    include_gantry_arg = DeclareLaunchArgument(
        'include_overhead_gantry',
        default_value='true',
        description='Whether to include the overhead lighting and camera gantry'
    )
    include_enclosure_arg = DeclareLaunchArgument(
        'include_enclosure',
        default_value='true',
        description='Whether to include the transparent enclosure panels and cabinet doors'
    )
    include_controller_arg = DeclareLaunchArgument(
        'include_ur_controller',
        default_value='true',
        description='Whether to include the UR controller box in the lower shelf'
    )
    include_pendant_arg = DeclareLaunchArgument(
        'include_teach_pendant',
        default_value='true',
        description='Whether to include the UR teach pendant on the side bracket'
    )
    include_wall_arg = DeclareLaunchArgument(
        'include_wall',
        default_value='true',
        description='Whether to include the wall behind the table'
    )
    wall_gap_arg = DeclareLaunchArgument(
        'wall_gap',
        default_value='0.10',
        description='Distance (gap) in meters between table rear edge and wall'
    )

    xacro_file = PathJoinSubstitution([
        pkg_share,
        'urdf',
        'table.urdf.xacro'
    ])

    rviz_config_file = PathJoinSubstitution([
        pkg_share,
        'rviz',
        'view_environment.rviz'
    ])

    robot_description = ParameterValue(
        Command([
            'xacro ', xacro_file,
            ' include_overhead_gantry:=', LaunchConfiguration('include_overhead_gantry'),
            ' include_enclosure:=', LaunchConfiguration('include_enclosure'),
            ' include_ur_controller:=', LaunchConfiguration('include_ur_controller'),
            ' include_teach_pendant:=', LaunchConfiguration('include_teach_pendant'),
            ' include_wall:=', LaunchConfiguration('include_wall'),
            ' wall_gap:=', LaunchConfiguration('wall_gap')
        ]),
        value_type=str
    )

    # Robot State Publisher Node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    # Joint State Publisher GUI Node
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )

    # RViz2 Node
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    return LaunchDescription([
        include_gantry_arg,
        include_enclosure_arg,
        include_controller_arg,
        include_pendant_arg,
        include_wall_arg,
        wall_gap_arg,
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz2_node
    ])
