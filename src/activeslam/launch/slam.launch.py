import os
from datetime import datetime
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _as_bool(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _launch_evaluator(context, *, evaluator_node, map_name, log_root, run_evaluator):
    """Start precise evaluation only for worlds with inline box geometry."""

    if not _as_bool(context.perform_substitution(run_evaluator)):
        return []

    world_name = context.perform_substitution(map_name)
    if world_name.startswith('slam_'):
        return [evaluator_node]

    reason = (
        f'Skipping slam_evaluator for map={world_name}: precise coverage and IoU '
        'require a slam_* world with inline box collisions. This world may use '
        'model:// includes, which the evaluator does not recursively parse.'
    )
    print(f'[slam.launch.py] WARNING: {reason}')
    try:
        root = Path(context.perform_substitution(log_root)).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        with (root / 'evaluator_skipped.log').open('a') as handle:
            timestamp = datetime.now().isoformat(timespec='seconds')
            handle.write(f'{timestamp} {reason}\n')
    except OSError as exc:
        print(f'[slam.launch.py] WARNING: Could not write evaluator skip log: {exc}')
    return []


def _launch_explorer(
    context,
    *,
    config_path,
    exploration_strategy,
    map_name,
    slam_mode,
    use_sim_time,
):
    """Start exploration while keeping YAML defaults unless launch overrides them."""

    mode = context.perform_substitution(slam_mode).strip()
    legacy_strategy = context.perform_substitution(exploration_strategy).strip()
    if mode and mode not in ('frontier', 'approx_graph', 'gbsae'):
        raise RuntimeError(f'Unsupported slam_mode={mode}.')
    if legacy_strategy and legacy_strategy not in ('frontier', 'graph', 'graph_based'):
        raise RuntimeError(f'Unsupported deprecated exploration_strategy={legacy_strategy}.')
    if mode and legacy_strategy:
        print(
            '[slam.launch.py] WARNING: Ignoring deprecated exploration_strategy '
            'because slam_mode was also provided.'
        )
    elif legacy_strategy:
        mode = 'approx_graph' if legacy_strategy in ('graph', 'graph_based') else 'frontier'
        print(
            '[slam.launch.py] WARNING: exploration_strategy is deprecated; '
            f'using slam_mode={mode}.'
        )

    overrides = {
        'use_sim_time': use_sim_time,
        'world_name': map_name,
    }
    if mode:
        overrides['slam_mode'] = mode
    return [
        Node(
            package='activeslam',
            executable='exploration_coordinator',
            name='exploration_coordinator',
            output='screen',
            parameters=[config_path, overrides],
        )
    ]


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    turtlebot3_model = LaunchConfiguration('turtlebot3_model')
    slam_mode = LaunchConfiguration('slam_mode')
    exploration_strategy = LaunchConfiguration('exploration_strategy')
    nav2_params_file = LaunchConfiguration('nav2_params_file')
    map_name = LaunchConfiguration('map')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    gui = LaunchConfiguration('gui')
    run_evaluator = LaunchConfiguration('run_evaluator')
    run_rviz = LaunchConfiguration('run_rviz')
    rviz_config_file = LaunchConfiguration('rviz_config_file')
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

    evaluator_node = Node(
        package='activeslam',
        executable='slam_evaluator',
        name='slam_evaluator',
        output='screen',
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

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(run_rviz),
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
            'slam_mode',
            default_value='',
            description='Optional exploration policy override: frontier, approx_graph, or gbsae.',
        ),
        DeclareLaunchArgument(
            'exploration_strategy',
            default_value='',
            description='Deprecated compatibility override: frontier or graph.',
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
            default_value='true',
            description='Start slam_evaluator for supported slam_* worlds.',
        ),
        DeclareLaunchArgument(
            'run_rviz',
            default_value='true',
            description='Start RViz with the Active SLAM debug view.',
        ),
        DeclareLaunchArgument(
            'rviz_config_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('activeslam'), 'rviz', 'activeslam.rviz']
            ),
            description='Full path to the RViz configuration file.',
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
        OpaqueFunction(
            function=_launch_explorer,
            kwargs={
                'config_path': config_path,
                'exploration_strategy': exploration_strategy,
                'map_name': map_name,
                'slam_mode': slam_mode,
                'use_sim_time': use_sim_time,
            },
        ),
        OpaqueFunction(
            function=_launch_evaluator,
            kwargs={
                'evaluator_node': evaluator_node,
                'map_name': map_name,
                'log_root': log_root,
                'run_evaluator': run_evaluator,
            },
        ),
        rviz_node,
    ])
