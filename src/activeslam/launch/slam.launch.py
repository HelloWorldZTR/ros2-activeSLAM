from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    turtlebot3_model = LaunchConfiguration('turtlebot3_model')
    planner_type = LaunchConfiguration('planner_type')

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('turtlebot3_gazebo'), 'launch',
                 'turtlebot3_world.launch.py']
            )
        )
    )

    slam_node = Node(
        package='slam_toolbox',
        executable='sync_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    config_path = PathJoinSubstitution(
        [FindPackageShare('activeslam'), 'config', 'exploration.yaml']
    )

    explorer_node = Node(
        package='activeslam',
        executable='exploration_coordinator',
        name='exploration_coordinator',
        output='screen',
        parameters=[
            config_path,
            {
                'use_sim_time': use_sim_time,
                'planner_type': planner_type,
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock from Gazebo.',
        ),
        DeclareLaunchArgument(
            'turtlebot3_model',
            default_value='burger',
            description='TurtleBot3 model to spawn in Gazebo.',
        ),
        DeclareLaunchArgument(
            'planner_type',
            default_value='astar',
            choices=['astar', 'rrt'],
            description='Path planning algorithm: astar or rrt.',
        ),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', turtlebot3_model),
        gazebo_launch,
        slam_node,
        explorer_node,
    ])
