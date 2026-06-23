# ROS2 Active SLAM Project

## Project Design

This is a ROS 2 Humble workspace for TurtleBot3 active SLAM in Gazebo. The main package, `activeslam`, launches Gazebo, spawns a TurtleBot3, runs `slam_toolbox`, starts Nav2, starts an exploration coordinator driven by `/map` and TF, and opens evaluator and RViz helpers by default.

Exploration is coordinated by `activeslam.exploration_coordinator`. It detects frontiers on the occupancy grid, asks Nav2 for candidate paths, sends selected goals to Nav2, and exposes RViz markers for goals, frontiers, and pose-graph helpers. Nav2 owns `/cmd_vel`, path following, and recovery behavior. The `slam_mode` parameter selects `frontier`, `approx_graph`, `gbsae`, `gvd_gbsae`, or `gvd_hierarchical`. `frontier` chooses reachable frontiers directly. `approx_graph` scores Nav2 candidate paths with an approximate weighted pose graph and D-opt style information score. `gbsae` loads a per-world topo-metric prior graph, follows a deterministic route, allocates frontiers to prior vertices, and inserts optional spectral loop revisits. `gvd_gbsae` first sweeps quickly along an obstacle-only online GVD skeleton with centerline-biased local A*. Its live graph uses a cached GVD-first, bidirectional-A*-fallback switching connection mechanism before dropping truly isolated components. Throughout bootstrap, if TF poses report no effective translation for too long, it uses bounded Nav2 `Spin` and `DriveOnHeading` recovery actions. It then derives a live graph for the existing GBSAE phase. `gvd_hierarchical` instead follows a throttled dynamic open-TSP walk over unexplored compressed live GVD vertices, treats explored vertices as transit, performs local flood-bounded frontier cleanup when traversal reaches an explored topological leaf, and finishes with global frontier cleanup. Each local Region owns an independent approximate pose graph and scores its Nav2-reachable cleanup paths with the D-opt heuristic. `slam_toolbox` itself is unchanged.

GVD skeleton construction, local GVD A*, and active-path invalidation retain robot-radius obstacle inflation. Only switching-connection fallback A* uses the permissive `gvd_reconnection_clearance` mask before Nav2 validates execution. Hierarchical macro traversal uses NetworkX `>=3.4,<4` open-TSP routing; cleared vertices remain usable as transit but leave the required target set.
RViz renders raw GVD skeleton edges in cyan and switching fallback A* bridges in orange so hypothesized reconnections remain visually distinct.
GVD nodes with a high local unknown ratio are marked unconfident. Each induced unconfident region is reduced to MST edges before macro planning, suppressing imagined loops without blocking unknown-space exploration. Hierarchical GVD rebuilds continue at the throttled rate during macro navigation, Region cleanup, and local probes. A changed TSP first step does not preempt an active macro Nav2 goal; local actions are also not preempted. Hierarchical GVD uses an expansion TSP over unexplored endpoints and necessary branch fallbacks, then a cleanup TSP over uncleared ordinary or degenerate leaves. Before replacing a live graph, it remembers the active-to-next-step direction. If the rebuilt active vertex is not cleanup-eligible, the most direction-aligned rebuilt neighbor becomes a forced first hop and only one open TSP is generated from that neighbor. Explored and cleared vertices remain available as shortest-path transit. Reaching an uncleared Region-bearing leaf triggers cleanup even if state migration has not marked it explored yet; native GVD and fallback A* edges count equally. Local Region cleanup is marked complete once observed cells reach the configurable `gvd_hierarchical_local_clear_progress_threshold` ratio, default `0.90`, or no Nav2-reachable Region frontier remains. If a live rebuild changes the active Region vertex into a non-leaf, cancel local cleanup without clearing it and return to macro traversal.

Frontier probes default on for `frontier`, `approx_graph`, and `gbsae`. Macro GVD navigation defaults off for `gvd_gbsae` and `gvd_hierarchical`, while `gvd_hierarchical` local flood cleanup defaults probes on. Keep mode-level overrides, the hierarchical-local override, and the unknown-frontier/open-edge type switches separate so ablation runs remain explicit.

There is no initial exploration Spin. After Nav2 and the first map are ready, every baseline shares the effective-translation watchdog and bounded Nav2 `Spin -> DriveOnHeading` random-walk recovery. The legacy `gvd_stuck_*` and `gvd_random_recovery_*` parameter prefixes remain for configuration compatibility. Recovery has priority over ordinary idle, selection, navigation, and frontier-probe actions, but must not recursively interrupt its own Spin or DriveOnHeading action.

Ordinary frontier clusters default to at least `10px`. Before downstream selection, the detector suppresses bounded unknown pockets exposed only by smaller clusters: neighboring free and occupied geometry votes fill them with low-confidence occupancy values. Large unknown components and components touching the `/map` array edge remain unknown, and low-confidence free cells are not valid safe goals.

Evaluation is handled by `activeslam.slam_evaluator`. It compares SLAM output against Gazebo ground truth and inline world geometry, then writes run logs under `logs/run_*` with estimated/ground-truth trajectories, coverage over time, coverage over path length, map IoU, and ATE metrics. Precise evaluation is limited to `slam_*` worlds with inline box collisions. The main launch skips evaluator for `turtlebot3_*` worlds that rely on unparsed `model://` includes.

Simulation assets live in `activeslam_resource` and vendored TurtleBot3 simulation packages. World names passed to launches are basenames only, for example `map:=slam_rooms`, not `slam_rooms.world`. `slam_office` is the IoU-friendly complex office benchmark generated from the upstream `Dataset-of-Gazebo-Worlds-Models-and-Maps` occupancy image.

## Important Files

- `src/activeslam/launch/slam.launch.py`: main launch file for Gazebo, TurtleBot3 spawn, `slam_toolbox`, and `exploration_coordinator`.
- `src/activeslam/config/exploration.yaml`: frontier selection and graph-scoring parameters.
- `src/activeslam/config/nav2_params.yaml`: Nav2 planner, controller, costmap, behavior, and velocity smoother parameters.
- `src/activeslam/rviz/activeslam.rviz`: default RViz view for the map, pruned hierarchical GVD state nodes, and directed TSP route. Other debugging displays remain available but disabled by default.
- `src/activeslam/activeslam/exploration_coordinator.py`: main active exploration ROS node.
- `src/activeslam/activeslam/nav2_backend.py`: asynchronous Nav2 action adapter.
- `src/activeslam/activeslam/frontier_detector.py`: frontier-cell detection and clustering.
- `src/activeslam/activeslam/graph_exploration.py`: approximate pose graph tracking, graph scoring, and graph visualization.
- `src/activeslam/activeslam/gbsae_exploration.py`: prior-graph loading, greedy routing, spectral loop insertion, GBSAE state, frontier allocation, and visualization.
- `src/activeslam/activeslam/gvd_exploration.py`: obstacle-only GVD topology, A* bootstrap goal ranking, trajectory-sweep tracking, and visualization.
- `src/activeslam/config/gvd_worlds.yaml`: coarse rectangular bounds for online GVD bootstrap. These bounds contain no wall structure.
- `src/activeslam/activeslam/slam_evaluator.py`: ROS node for SLAM coverage, trajectory, ATE, and IoU logging.
- `src/activeslam/activeslam/slam_evaluator_utils.py`: testable evaluator geometry and metric helpers.
- `src/activeslam_resource/maps/`: Gazebo worlds used by `slam.launch.py` and `slam_evaluator`.
- `src/activeslam_resource/maps/slam_rooms.gbsae.json`: initial GBSAE topo-metric prior graph. GBSAE fails early if the selected world lacks a matching asset.
- `src/activeslam_resource/maps/slam_office.gbsae.json`: generated reviewable Office GBSAE prior graph.
- `src/activeslam_resource/models/`: custom Gazebo models referenced by project worlds.
- `tools/generate_slam_office_world.py`: standard-library generator for the inline-box `slam_office` benchmark.
- `tools/generate_office_gbsae_prior.py`: standard-library generator for the initial Office GBSAE prior graph.
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
source /home/psirobot/projects/ros2_ws/src/setup.zsh
cd /home/psirobot/projects/ros2_ws/src/
cb # alias for colcon build xxxxx
s # alias for source install/setup.zsh
# run your ros command
```

When debugging, try your best to use text only ways, e.g debug output file, tea to a log file etc. Rather than listening to the program outputs, as ROS command output can be quite lengthy and difficult to terminate.
When performing experiments, remember to kill existing processes.
For headless remote experiments, disable evaluator matplotlib explicitly instead of changing project defaults:

```bash
ros2 launch activeslam slam.launch.py map:=slam_rooms gui:=false run_rviz:=false run_evaluator:=true plot_live:=false save_plots:=false
```

For a GBSAE headless smoke run:

```bash
MAP=slam_rooms SLAM_MODE=gbsae RUN_SECONDS=120 ./run.zsh
```

The exploration coordinator must not publish `/cmd_vel`; verify Nav2 ownership with:

```bash
ros2 topic info /cmd_vel --verbose
```

For standalone evaluator runs, pass:

```bash
ros2 run activeslam slam_evaluator --ros-args -p plot_live:=false -p save_plots:=false
```

**Do not** perform any risky commands such as `rm -rf` or something that requires password.

## Coding rules

- add comments for critical functions and files
- commit when you think you have reached a temporaly milestone
- write your experiment log in `experiments/experiment_log.md`, with the corresponding link to the experiement products.
