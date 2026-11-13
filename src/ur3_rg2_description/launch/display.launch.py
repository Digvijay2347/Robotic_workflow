
from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    robot_description_file = PathJoinSubstitution([
        FindPackageShare('ur3_rg2_description'),
        'urdf',
        'ur3_rg2.urdf.xacro'
    ])

    robot_description = ParameterValue(
        Command(['xacro ', robot_description_file]),
        value_type=str
    )

    return LaunchDescription([

        # Publish robot description and TF
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description
            }]
        ),

        # GUI joint controller - only one joint state publisher
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen'
        ),

        # RViz visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'
        )
    ])
