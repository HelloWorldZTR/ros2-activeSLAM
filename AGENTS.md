# ROS2 Active SLAM Project

## Project Design

This is a ROS 2 Humble workspace for TurtleBot3 active SLAM in Gazebo. The main package, `activeslam`, launches Gazebo, spawns a TurtleBot3, runs `slam_toolbox`, and starts an exploration controller that drives the robot from `/map`, `/scan`, and TF.

Exploration is coordinated by `activeslam.exploration_coordinator`. It detects frontiers on the occupancy grid, plans paths with A* or RRT, publishes `/cmd_vel`, and exposes RViz markers for goals, frontiers, planned paths, and the optional approximate pose graph. The `frontier` strategy chooses reachable frontiers directly; the `graph` strategy scores candidate paths with an approximate weighted pose graph and D-opt style information score.

Evaluation is handled separately by `activeslam.slam_evaluator`. It compares SLAM output against Gazebo ground truth and world geometry, then writes run logs under `logs/run_*` with estimated/ground-truth trajectories, coverage over time, coverage over path length, map IoU, and ATE metrics.

Simulation assets live in `activeslam_resource` and vendored TurtleBot3 simulation packages. World names passed to launches are basenames only, for example `map:=slam_rooms`, not `slam_rooms.world`.

## Important Files

- `src/activeslam/launch/slam.launch.py`: main launch file for Gazebo, TurtleBot3 spawn, `slam_toolbox`, and `exploration_coordinator`.
- `src/activeslam/config/exploration.yaml`: controller, planner, frontier, and graph-scoring parameters.
- `src/activeslam/activeslam/exploration_coordinator.py`: main active exploration ROS node.
- `src/activeslam/activeslam/frontier_detector.py`: frontier-cell detection and clustering.
- `src/activeslam/activeslam/path_planner.py`: A* and RRT planners plus obstacle inflation.
- `src/activeslam/activeslam/graph_exploration.py`: approximate pose graph tracking, graph scoring, and graph visualization.
- `src/activeslam/activeslam/slam_evaluator.py`: ROS node for SLAM coverage, trajectory, ATE, and IoU logging.
- `src/activeslam/activeslam/slam_evaluator_utils.py`: testable evaluator geometry and metric helpers.
- `src/activeslam_resource/maps/`: Gazebo worlds used by `slam.launch.py` and `slam_evaluator`.
- `src/activeslam_resource/models/`: custom Gazebo models referenced by project worlds.
- `src/setup.zsh`: bc01-style zsh setup with equivalent build/run aliases.


## Remote debug

The host machine is not capable of running the simulation and ROS2, basic debug and static check can be performed in the conda environment generic.
If you want to debug or run the project, please use the following command to sync the local project to the remote server.

```bash
rsync -avP . betail:/home/psirobot/projects/ros2_ws/
```

The target shell is zsh, use .zsh setup scripts. Use the following command to activate the ROS2 environment.

```bash
source /opt/ros/humble/setup.zsh
source /home/psirobot/projects/ros2_ws/src/setup2.zsh
cd /home/psirobot/projects/ros2_ws/src/
cb # alias for colcon build xxxxx
s # alias for source install/setup.zsh
# run your ros command
```

When debugging, try your best to use text only ways, e.g debug output file, tea to a log file etc. Rather than listening to the program outputs, as ROS command output can be quite lengthy and difficult to terminate.
When performing experiments, remember to kill existing processes.
For headless remote experiments, disable evaluator matplotlib explicitly instead of changing project defaults:

```bash
ros2 launch activeslam slam.launch.py map:=slam_rooms gui:=false run_evaluator:=true plot_live:=false save_plots:=false
```

For standalone evaluator runs, pass:

```bash
ros2 run activeslam slam_evaluator --ros-args -p plot_live:=false -p save_plots:=false
```

**Do not** perform any risky commands such as `rm -rf` or something that requires password.

## Coding rules

- add comments for critical functions and files
- commit when you think you have reached a temporaly milestone
- write your experiment log in `EXPERIMENT.MD`, with the corresponding link to the experiement products.
