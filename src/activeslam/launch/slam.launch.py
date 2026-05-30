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
from launch_ros.substitutions import FindPackageShare

import os

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    turtlebot3_model = LaunchConfiguration('turtlebot3_model')
    exploration_strategy = LaunchConfiguration('exploration_strategy')
    nav2_params_file = LaunchConfiguration('nav2_params_file')
    map_name = LaunchConfiguration('map')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    gui = LaunchConfiguration('gui')
    run_evaluator = LaunchConfiguration('run_evaluator')
    log_root = LaunchConfiguration('log_root')
    plot_live = LaunchConfiguration('plot_live')
    save_plots = LaunchConfiguration('save_plots')

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
        condition=IfCondition(gui),
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

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('nav2_bringup'), 'launch', 'navigation_launch.py']
            )
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'autostart': 'true',
            'use_composition': 'False',
            'params_file': nav2_params_file,
        }.items(),
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
                'exploration_strategy': exploration_strategy,
            },
        ],
    )

    evaluator_node = Node(
        package='activeslam',
        executable='slam_evaluator',
        name='slam_evaluator',
        output='screen',
        condition=IfCondition(run_evaluator),
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'world_name': map_name,
                'log_root': log_root,
                'gt_model_name': turtlebot3_model,
                'plot_live': plot_live,
                'save_plots': save_plots,
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
            'nav2_params_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('activeslam'), 'config', 'nav2_params.yaml']
            ),
            description='Full path to the Nav2 parameters file.',
        ),
        DeclareLaunchArgument(
            'map',
            default_value='slam_rooms',
            choices=available_maps,
            description='Gazebo world name from the activeslam_resource maps directory.',
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Start Gazebo client GUI.',
        ),
        DeclareLaunchArgument(
            'run_evaluator',
            default_value='false',
            description='Start slam_evaluator in the same launch for experiment runs.',
        ),
        DeclareLaunchArgument(
            'log_root',
            default_value='logs',
            description='Directory where slam_evaluator writes run logs.',
        ),
        DeclareLaunchArgument(
            'plot_live',
            default_value='true',
            description='Show slam_evaluator live matplotlib windows.',
        ),
        DeclareLaunchArgument(
            'save_plots',
            default_value='true',
            description='Save slam_evaluator matplotlib plot images on shutdown.',
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
