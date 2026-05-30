# Active SLAM Experiment Log

## 2026-05-27 - Frontier Baseline Evaluation Polish

Goal: make the frontier baseline easier to evaluate across `slam_*.world` maps and remove current blockers before comparing new active SLAM heuristics.

Changes in progress:

- Evaluator:
  - Ground-truth topic lookup now subscribes to ROS 2 `/model_states` and legacy `/gazebo/model_states`.
  - Added `/get_entity_state` service fallback for Gazebo launches that provide entity state through services instead of `ModelStates` topics.
  - Added simulator `/odom` as the final ground-truth fallback for this TurtleBot3 Gazebo launch, which exposes odometry but not model-state topics.
  - Ground-truth model lookup now accepts the Gazebo entity names used by the TurtleBot3 spawner, e.g. `burger`, while still accepting `turtlebot3_burger`.
  - Estimated pose lookup tries `base_footprint` and `base_link` under `map`, and logs the selected frame.
  - Live plotting and final metric plot generation remain enabled by default; agents should disable them explicitly for headless remote runs.
  - `metrics.json` is refreshed during sampling, so timeout-limited runs keep usable metrics even if final shutdown is interrupted.
  - Metrics now record selected TF/model names and sample counts.
- Frontier baseline:
  - A*/RRT planning treats unknown cells as blocked by default so paths stay in known free space.
  - If no known-space plan is reachable, the controller falls back to the previous permissive unknown-cell planner to keep the robot moving.
  - Frontier clusters retain their member grid cells.
  - Target selection tries multiple free cells per frontier cluster instead of only the centroid.
  - Baseline config now checks more frontier clusters and uses a practical `0.25 m` goal tolerance.
- Launch:
  - `slam.launch.py` can run headless with `gui:=false`.
  - `run_evaluator:=true` starts `slam_evaluator` in the same launch process.

Planned baseline command on the remote host after sync/build:

```bash
timeout 180s ros2 launch activeslam slam.launch.py map:=slam_rooms gui:=false run_evaluator:=true exploration_strategy:=frontier log_root:=logs/frontier_baseline
```

Repeat for:

- `slam_landmarks`
- `slam_loop`
- `slam_rooms`
- `slam_rooms_corridor`

Remote smoke result:

- Build passed on `betail` after sync.
- First 90 second `slam_rooms` run started Gazebo, SLAM, exploration, and evaluator.
- The run exposed that this ROS 2 setup publishes model states on `/model_states`, not `/gazebo/model_states`; evaluator topic fallback was added afterward.
- Follow-up runs showed no `ModelStates` topic or `/get_entity_state` service data was available; evaluator now falls back to simulator `/odom`.
- A 45 second smoke run reached `Evaluator is recording samples`, confirming estimated pose and ground-truth odometry are both available.
- A 35 second smoke run wrote usable metrics:
  - run: `logs/frontier_baseline/run_20260527_211537`
  - `gt_topic`: `/odom`
  - `estimated_samples`: 33
  - `ground_truth_samples`: 34
  - `ate_rmse`: 0.0117 m
  - `final_coverage`: 0.2210
- The runs also showed the strict known-space planner can reject all early frontiers; permissive fallback was added afterward while keeping known-space plans preferred.

Status: evaluator ground-truth and periodic metrics smoke test passed on `slam_rooms`.

## 2026-05-30 - Nav2 Backend Migration

Goal: replace the local A*/RRT and Pure Pursuit controller with the Nav2 Humble
navigation stack while retaining frontier and graph-based exploration strategies.

Implementation:

- Added an asynchronous Nav2 adapter for `/compute_path_to_pose`,
  `/navigate_to_pose`, and `/spin`.
- The coordinator now performs one initial Nav2 spin, scores reachable frontier
  goals using Nav2 paths, and sends the selected goal back to Nav2.
- Removed the local planner, direct `/cmd_vel` publishing, local stuck recovery,
  and random-walk fallback from the coordinator.
- Added TurtleBot3-style DWB, Navfn, costmap, behavior server, and velocity
  smoother parameters in `activeslam/config/nav2_params.yaml`.
- Configured the global costmap as a `20m x 20m` rolling window. Online SLAM
  initially publishes a tight map around the robot; a static-size costmap can
  leave the robot exactly one cell outside its boundary before the next map
  expansion.

Planned remote smoke commands after sync/build:

```bash
timeout 180s ros2 launch activeslam slam.launch.py map:=slam_rooms gui:=false run_evaluator:=true plot_live:=false save_plots:=false exploration_strategy:=frontier log_root:=logs/nav2_frontier
ros2 topic info /cmd_vel --verbose
timeout 180s ros2 launch activeslam slam.launch.py map:=slam_rooms gui:=false run_evaluator:=true plot_live:=false save_plots:=false exploration_strategy:=graph log_root:=logs/nav2_graph
```

Expected checks:

- `/compute_path_to_pose`, `/navigate_to_pose`, and `/spin` action servers exist.
- Startup logs report the initial Nav2 spin and subsequent frontier goals.
- `/cmd_vel` is published by Nav2's velocity smoother, not by
  `exploration_coordinator`.
- Graph mode continues reporting D-opt frontier scores.

Status: implementation complete locally; remote simulation smoke test pending.

## 2026-05-30 - Nav2 Safe Frontier Goals And Failure Cooldown

Goal: retain the event-driven Nav2 backend while improving frontier goal quality
and avoiding repeated retries of recently failed navigation targets.

Implementation:

- Replaced fixed approach-cell sampling with one safe standoff goal per frontier
  cluster. The search prefers known-free cells with obstacle clearance and a
  minimum forward advance toward the frontier.
- Added a 20 second cooldown for failed path checks, rejected or aborted
  navigation goals, and navigation timeouts. Nearby targets within 0.6m are
  excluded before frontier or graph selection.
- Added a 30 second NavigateToPose timeout. Active goals are no longer canceled
  merely because online map growth moves or removes the original frontier.
- Kept the grid frontier detector and Nav2-owned `/cmd_vel` pipeline. The
  direct-velocity open-boundary probe from `feat/Nav2` was intentionally not
  ported.
- Relaxed TurtleBot3 costmap clearance for narrow indoor passages while keeping
  the rolling global costmap, static layer, unknown tracking, and Navfn unknown
  traversal.

Status: Python compilation, focused helper tests, and diff checks pass locally.
Full `pytest` collection is blocked because the host is missing the ROS
`ament_copyright`, `ament_flake8`, and `ament_pep257` Python modules.
`colcon test` is also blocked until the local ROS workspace is rebuilt.
