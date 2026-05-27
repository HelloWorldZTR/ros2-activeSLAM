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
