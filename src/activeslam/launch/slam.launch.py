from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

import os

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    turtlebot3_model = LaunchConfiguration('turtlebot3_model')
    exploration_strategy = LaunchConfiguration('exploration_strategy')
    enable_evaluator = LaunchConfiguration('enable_evaluator')
    evaluator_log_root = LaunchConfiguration('evaluator_log_root')
    evaluator_sample_interval = LaunchConfiguration('evaluator_sample_interval')
    map_name = LaunchConfiguration('map')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')

    gazebo_world = PathJoinSubstitution([
        FindPackageShare('activeslam_resource'),
        'maps',
        PythonExpression(["'", map_name, ".world'"]),
    ])

    gzserver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('gazebo_ros'), 'launch', 'gzserver.launch.py']
            )
        ),
        launch_arguments={'world': gazebo_world}.items(),
    )

    gzclient_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('gazebo_ros'), 'launch', 'gzclient.launch.py']
            )
        ),
    )

    robot_state_publisher_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('turtlebot3_gazebo'),
                    'launch',
                    'robot_state_publisher.launch.py',
                ]
            )
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    spawn_turtlebot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('turtlebot3_gazebo'),
                    'launch',
                    'spawn_turtlebot3.launch.py',
                ]
            )
        ),
        launch_arguments={'x_pose': x_pose, 'y_pose': y_pose}.items(),
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

    nav2_params_path = PathJoinSubstitution(
        [FindPackageShare('activeslam'), 'config', 'nav2_params.yaml']
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('nav2_bringup'), 'launch', 'navigation_launch.py']
            )
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': nav2_params_path,
            'autostart': 'true',
        }.items(),
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
                'exploration_strategy': exploration_strategy,
            },
        ],
    )

    evaluator_node = Node(
        package='activeslam',
        executable='slam_evaluator',
        name='slam_evaluator',
        output='screen',
        condition=IfCondition(enable_evaluator),
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'world_name': map_name,
                'log_root': evaluator_log_root,
                'sample_interval': ParameterValue(
                    evaluator_sample_interval,
                    value_type=float,
                ),
            },
        ],
    )

    available_maps = [
        file.split('.')[0] for file in os.listdir(os.path.join(
            FindPackageShare('activeslam_resource').find('activeslam_resource'), 'maps'
        )) if file.endswith('.world')
    ]

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
            'exploration_strategy',
            default_value='frontier',
            choices=['frontier', 'graph'],
            description='Exploration target selection strategy.',
        ),
        DeclareLaunchArgument(
            'enable_evaluator',
            default_value='false',
            description='Start the SLAM evaluator node with the experiment.',
        ),
        DeclareLaunchArgument(
            'evaluator_log_root',
            default_value='logs',
            description='Directory where evaluator run logs are written.',
        ),
        DeclareLaunchArgument(
            'evaluator_sample_interval',
            default_value='1.0',
            description='Evaluator sampling interval in seconds.',
        ),
        DeclareLaunchArgument(
            'map',
            default_value='turtlebot3_world',
            choices=available_maps,
            description='Gazebo world name from the activeslam_resource maps directory.',
        ),
        DeclareLaunchArgument(
            'x_pose',
            default_value='-2.0',
            description='Initial TurtleBot3 x position in Gazebo.',
        ),
        DeclareLaunchArgument(
            'y_pose',
            default_value='-0.5',
            description='Initial TurtleBot3 y position in Gazebo.',
        ),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', turtlebot3_model),
        gzserver_launch,
        gzclient_launch,
        robot_state_publisher_launch,
        spawn_turtlebot_launch,
        slam_node,
        nav2_launch,
        explorer_node,
        evaluator_node,
    ])
