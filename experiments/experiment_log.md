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
timeout 180s ros2 launch activeslam slam.launch.py map:=slam_rooms gui:=false run_rviz:=false run_evaluator:=true plot_live:=false save_plots:=false exploration_strategy:=frontier log_root:=logs/frontier_baseline
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
timeout 180s ros2 launch activeslam slam.launch.py map:=slam_rooms gui:=false run_rviz:=false run_evaluator:=true plot_live:=false save_plots:=false exploration_strategy:=frontier log_root:=logs/nav2_frontier
ros2 topic info /cmd_vel --verbose
timeout 180s ros2 launch activeslam slam.launch.py map:=slam_rooms gui:=false run_rviz:=false run_evaluator:=true plot_live:=false save_plots:=false exploration_strategy:=graph log_root:=logs/nav2_graph
```

Expected checks:

- `/compute_path_to_pose`, `/navigate_to_pose`, and `/spin` action servers exist.
- Startup logs report the initial Nav2 spin and subsequent frontier goals.
- `/cmd_vel` is published by Nav2's velocity smoother or Behavior Server, not
  by `exploration_coordinator`.
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

## 2026-05-30 - Open Map Edge Frontier Fallback

Goal: continue exploration when slam_toolbox publishes a tight occupancy-grid
rectangle and unexplored space outside that rectangle appears as RViz
background rather than in-map unknown cells.

Implementation:

- Added open-map-edge frontier clusters for free cells whose four-neighborhood
  crosses the `/map` array boundary. Ordinary free-to-unknown frontiers retain
  priority and both sources use independent eight-connected clustering.
- Initially kept open-edge goals as a fallback only: Nav2 path checks and graph
  scoring consumed ordinary frontiers first, then open-edge clusters only when
  needed. This was later replaced by shared information-gain ranking.
- Added a center-line obstacle check between each frontier seed and its
  standoff candidate. Unknown cells remain allowed on this short segment;
  complete reachability remains owned by Nav2.
- Kept `/frontier_markers` as the visualization interface. Ordinary clusters
  are green and open-edge fallback clusters are cyan.

Status: Python compilation, diff checks, and focused frontier/Nav2 helper tests
pass locally. Remote simulation smoke test pending.

## 2026-05-30 - Nav2-Owned Open Edge Probe

Goal: let open-map-edge fallback frontiers expand the SLAM occupancy-grid
rectangle without publishing velocity commands from the exploration node.

Implementation:

- Kept open-edge navigation goals inside the current map. After Nav2 reaches
  the safe standoff point, the coordinator estimates a local outward normal
  from nearby out-of-map neighbors, aligns with Nav2 `Spin`, and advances
  `0.40m` with Nav2 `DriveOnHeading`. This was later increased to `2.0m`.
- Added conservative probe speed and timeout parameters. Behavior Server owns
  the probe velocity command and uses its local costmap collision check.
- Added the existing failed-goal cooldown to alignment and probe failures so a
  blocked edge is not selected repeatedly.

Status: Python compilation and focused helper tests pass locally. Remote
simulation smoke test pending.

## 2026-05-30 - Shared Frontier Information Gain Ranking

Goal: let ordinary and open-map-edge frontiers compete fairly without allowing
long map-border clusters to dominate merely because they contain many cells.

Implementation:

- Replaced source-priority fallback groups with one shared candidate pool.
- Added a local potential-unknown-area utility inspired by `active_graph_slam`.
  Ordinary frontiers count in-map unknown cells; open-edge frontiers also treat
  out-of-map cells as potential unknown during candidate preselection only.
- Removed the cluster-size bonus from graph scoring. Graph mode now compares
  the real Nav2 paths of the shared top candidates using D-opt and path cost.
- Increased the Nav2-owned open-edge probe to `2.0m` at `0.12m/s` with a
  `20s` timeout.

Status: Python compilation, YAML parsing, diff checks, and focused helper tests
pass locally. Remote simulation smoke test pending.

## 2026-05-30 - Frontier Detection And Selection Cache Optimization

Goal: reduce frontier detection and local candidate-filter latency without
changing frontier definitions, candidate ordering, or Nav2 planning behavior.

Implementation:

- Vectorized ordinary and open-map-edge frontier masks while retaining the
  existing eight-connected cluster traversal.
- Cached each received occupancy grid as one NumPy array for detection,
  stability tracking, local candidate selection, and graph scoring.
- Added a prepared safe-goal grid built once per selection round. A summed-area
  table precomputes obstacle-clearance validity before clusters are evaluated.
- Vectorized local unknown-area and graph cell-information counting.
- Added `local_filter_ms` to shared frontier-pool logs for remote observation.

Local checks:

- Focused helper tests: `34 passed`.
- Randomized equivalence check: `200` generated grids matched the previous
  unknown-area and safe-goal helper results.
- Python compilation and `git diff --check`: passed.
- Full pytest collection remains blocked locally by missing ROS
  `ament_copyright`, `ament_flake8`, and `ament_pep257` Python modules.
- `colcon test --packages-select activeslam` remains blocked until the local
  workspace is rebuilt.

Synthetic local benchmark:

- Map size: `400 x 400`, resolution: `0.05m`.
- Previous Python four-neighbor mask scan: `414.45ms`.
- Vectorized detector including clustering: `17.97ms`.
- Detector speedup is at least `23.1x`; the old number excludes clustering.
- Prepared-grid construction once per selection round: `5.25ms`.
- Prepared safe-goal search for one long cluster: `5.68ms`.

Status: local implementation and focused tests pass. Remote frontier and graph
headless smoke tests pending.

## 2026-05-30 - One-Command Evaluator And RViz Launch

Goal: remove the need for separate terminals when running interactive Active
SLAM experiments.

Implementation:

- Enabled evaluator and RViz by default from `slam.launch.py`, with
  `run_evaluator` and `run_rviz` switches for headless execution.
- Added an Active SLAM RViz profile for `/map`, `/scan`, RobotModel, `/plan`,
  frontier markers, the current goal, pose graph markers, and a translucent
  global costmap overlay.
- Kept precise evaluator metrics limited to `slam_*` worlds with inline box
  collisions. Other worlds are allowed to run, but evaluator is skipped with a
  terminal warning and an entry in `logs/evaluator_skipped.log`.
- Removed the obsolete `planner_type` argument from the headless runner and
  made it explicitly disable RViz.

Status: Python compilation, diff checks, focused helper tests, RViz YAML
parsing, and evaluator launch routing checks pass locally. The local host lacks
the ROS Humble environment and built workspace dependencies, so interactive
and headless ROS smoke tests remain pending.

## 2026-05-31 - GBSAE Prior-Graph Exploration Mode

Goal: add a ROS 2 GBSAE adaptation while keeping `slam_toolbox`, the shared
safe-frontier pipeline, and Nav2 ownership of `/cmd_vel` unchanged.

Implementation:

- Replaced the coordinator-facing `exploration_strategy` setting with
  `slam_mode: frontier|approx_graph|gbsae`. Launch still accepts deprecated
  `exploration_strategy:=frontier|graph`, mapping `graph` to `approx_graph`.
- Added a `networkx` prior-graph planner with strict JSON validation,
  deterministic greedy route expansion, normalized Laplacian
  weighted-spanning-tree scoring, positive-objective spectral loop revisits,
  route progression, and frontier-to-prior-vertex allocation.
- Added `slam_rooms.gbsae.json`, with free-space vertices and traversable edges
  aligned to the room and doorway layout. Other worlds fail early in `gbsae`
  mode until they receive a matching `<world>.gbsae.json` asset.
- Reused Nav2 path checks and navigation actions for direct prior-vertex goals,
  allocated frontier goals, and optional loop revisits. Unreachable optional
  revisits are skipped with a warning to avoid deadlock.
- Updated `run.zsh` to use `SLAM_MODE`, while retaining legacy `STRATEGY`
  fallback.

Local checks:

- Focused helper tests: `51 passed`.
- Python compilation, YAML/JSON parsing, `zsh -n run.zsh`, manual line-length
  scan, and `git diff --check`: passed.
- Full local pytest collection remains blocked by missing ROS
  `ament_copyright`, `ament_flake8`, and `ament_pep257` Python modules.
- Implementation milestone commit: `cd6ad1d`.

Remote verification:

- Synced to `betail:/home/psirobot/projects/ros2_ws/` without `.external/`.
- The documented `src/setup2.zsh` was absent on `betail`; verification used the
  existing `src/setup.zsh`, and `AGENTS.md` now reflects that path.
- Remote dependency check: `networkx 3.4.2`.
- Remote `colcon build`: passed.
- Remote `colcon test --packages-select activeslam`:
  `54 tests, 0 errors, 0 failures, 1 skipped`.
- Unsupported-world check:
  `ros2 run activeslam exploration_coordinator --ros-args
  -p slam_mode:=gbsae -p world_name:=slam_loop` failed immediately with the
  expected missing `slam_loop.gbsae.json` error.

Remote smoke command:

```bash
MAP=slam_rooms SLAM_MODE=gbsae RUN_SECONDS=120 \
  LOG_ROOT=/home/psirobot/projects/ros2_ws/experiments/products/gbsae_smoke_20260531 \
  /home/psirobot/projects/ros2_ws/run.zsh
```

Smoke result: passed. The key-events log confirms prior-graph loading, route
creation with spectral loop edge `(0, 8)`, allocated frontier checks, direct
prior-vertex Nav2 goals, frontier goal dispatch, optional unreachable revisit
skips, and route completion without deadlock. The evaluator wrote `128`
estimated samples with `final_coverage=0.5817`, `ate_rmse=0.0371`, and
`free_iou=0.9599`. This accelerated-physics run exercised Nav2 recovery churn,
so it is a smoke test rather than a policy-quality benchmark.

`ros2 topic info /cmd_vel --verbose` reported Nav2 `behavior_server` endpoints
and `velocity_smoother` as the only publishers. `exploration_coordinator` was
not a publisher.

Products:

- [remote build and package tests](products/gbsae_remote_build_test_retry_20260531.log)
- [`/cmd_vel` publisher check](products/gbsae_cmd_vel_publishers_20260531.log)
- [unsupported-world missing-prior check](products/gbsae_missing_prior_check_20260531.log)
- [smoke key events](products/gbsae_smoke_key_events_20260531.log)
- [smoke metrics](products/gbsae_smoke_20260531/run_slam_rooms_gbsae_20260531_111227/run_20260531_111228/metrics.json)
- [final evaluator map](products/gbsae_smoke_20260531/run_slam_rooms_gbsae_20260531_111227/run_20260531_111228/final_map.pgm)

## 2026-05-31 - IoU-Friendly Office Benchmark

- Added `slam_office.world`, derived from the office occupancy image in
  [`Dataset-of-Gazebo-Worlds-Models-and-Maps`](https://github.com/mlherd/Dataset-of-Gazebo-Worlds-Models-and-Maps).
- Converted occupied pixels into merged inline Gazebo box collisions. This
  avoids the original ServiceSim mesh and `model://` hierarchy, allowing the
  existing evaluator to rasterize the same obstacle geometry used by Gazebo.
- Added `tools/generate_slam_office_world.py` so the generated benchmark remains
  reproducible without committing the upstream 96MB office archives.

Status: generated world XML contains `374` inline box collisions and only the
standard Gazebo `ground_plane` and `sun` model includes. The launch file uses
the mesh-aligned `PublicBathroomB` interior spawn position `(12.1, 1.5)` for
`slam_office`, while preserving `(-2.0, -0.5)` for the existing maps.
- Added `tools/generate_office_gbsae_prior.py` and generated the first
  reviewable `slam_office.gbsae.json`: `89` nodes and `143` visibility edges,
  connected to the `PublicBathroomB` seed. The generator checks `0.32m` node
  clearance and `0.24m` edge clearance before emitting the graph.

Status: local static validation passed. Remote ROS smoke test and RViz prior
graph review are pending.

### Office coordinate correction

- Fixed the PNG-row to ROS-world-y conversion in
  `tools/generate_slam_office_world.py`. The initial formula missed one image
  height term and shifted all generated Gazebo obstacles downward by `30m`.
- Regenerated `slam_office.world`. Its obstacle bounds now match the upstream
  occupancy geometry: `x=[-27.65, 21.25]`, `y=[-0.10, 22.60]`.
- Kept the GBSAE prior and spawn seed at the mesh-aligned `PublicBathroomB`
  interior point `(12.1, 1.5)`.

## 2026-05-31 - Relaxed Frontier Local Filtering

- Removed the hard `frontier_information_gain_min` filter. Information gain is
  still computed and used to rank candidates.
- Removed the hard `frontier_goal_min_advance` filter. Safe-goal scoring still
  prefers useful forward progress, while reach radius, known-free checks,
  obstacle clearance, segment checks, and failed-goal cooldown remain active.
- Kept the diagnostic Nav2 inflation adjustment: local `0.22m`, global `0.35m`.

Status: Python compilation, `git diff --check`, and the available pure Python
frontier, graph, Nav2 adapter, and evaluator helper tests passed (`40 passed`).
Full local pytest collection remains blocked by missing generic-environment
`ament_*` modules and `networkx`. Remote approx-graph smoke comparison is
pending.

### Unknown-frontier doorway probe

- Extended the Nav2-owned frontier probe to ordinary unknown frontiers. After
  reaching a known-free standoff goal, the coordinator estimates a local
  outward normal from adjacent unknown cells, aligns with Nav2 `Spin`, and
  advances with Nav2 `DriveOnHeading`.
- Kept ordinary probes conservative at `0.45m`, `0.08m/s`, and `8s`. Open map
  edges retain their existing `2.0m`, `0.12m/s`, and `20s` configuration.
- Behavior Server local-costmap collision checks remain active; the exploration
  coordinator still does not publish `/cmd_vel`.

Status: Python compilation, `git diff --check`, and the available pure Python
frontier, graph, Nav2 adapter, and evaluator helper tests passed (`42 passed`).

### Global planner clearance retune

- Increased global-costmap `inflation_radius` from `0.35m` to `0.45m` after
  observing U-shaped wall-following plans. Local-costmap inflation remains
  `0.22m` so DWB retains doorway maneuvering room.

## 2026-05-31 - Online GVD Bootstrap for GBSAE

- Added `slam_mode:=gvd_gbsae` without changing the existing static `gbsae`
  baseline. The first phase projects only laser-observed occupied cells into a
  coarse raster; unknown and free cells are both treated as traversable.
- Added pure NumPy Zhang-Suen thinning, skeleton compression, and a local A*
  selector. Candidate utility combines unswept rectangle-boundary direction,
  distance, historical trajectory-tube overlap, and straight-line heading.
- Nav2 still owns actual planning, control, and recovery. A new obstacle on the
  selected Nav2 path triggers immediate cancel and reselection.
- Added trajectory-sweep coverage switching. At `50%` swept coarse-bound area,
  the robot-connected live skeleton component initializes the existing GBSAE
  route planner.

Status: local Python compilation and focused GVD helper tests passed
(`7 passed`). Remote ROS smoke validation is pending.
