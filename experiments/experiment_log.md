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

### Centerline-biased paths and bounded stuck recovery

- Added a continuous A* penalty for distance from the estimated GVD centerline.
  The local bootstrap planner can now accept a modest detour instead of blindly
  preferring a shorter wall-hugging path.
- Added a TF-pose translation-progress watchdog for active GVD goals. After
  the configured timeout without enough positional displacement, the
  coordinator cancels navigation and performs up to three randomized Nav2-owned
  `Spin -> DriveOnHeading` attempts. In-place DWB rotation and tiny pose jitter
  do not postpone recovery.
- Kept the standalone direct-`/cmd_vel` `random_walker` as a debug entry point
  only. Online recovery uses Behavior Server actions so local-costmap collision
  checks remain active and the coordinator still does not publish velocity.

Status: local Python compilation, `git diff --check`, and available pure Python
helper regression tests passed (`64 passed`). Remote ROS smoke validation is
pending.

### GVD replanning-loop correction

- Fixed a state-machine regression introduced while adding stuck recovery:
  normal `NAVIGATING` cycles could fall through into `_start_selection()` and
  issue a new target every control tick.
- Tightened active-path invalidation. The coordinator now snapshots the
  obstacle-only raster when dispatching a goal, ignores walls already present
  at dispatch time, and checks only the forward suffix nearest to the robot.
  SLAM updates behind the robot no longer trigger unnecessary replanning.
- Replaced velocity-based stuck detection with TF-position progress anchors.
  Only translation of at least `0.15m` refreshes the watchdog; in-place
  rotation and small localization jitter no longer hide a U-shaped deadlock.
- Increased `gvd_centerline_distance_weight` from `2.0` to `5.0` so bootstrap
  A* more strongly prefers the estimated medial axis over shorter wall-hugging
  alternatives.
- Extended the translation-progress watchdog across GVD bootstrap `IDLE`,
  `SELECTING`, and `NAVIGATING` states. Any prolonged lack of effective
  displacement starts the same bounded random `Spin -> DriveOnHeading`
  escape sequence, including repeated empty selections and Nav2 failures.

Status: local Python compilation, `git diff --check`, and available pure Python
helper regression tests passed (`68 passed`). Remote ROS smoke validation is
pending.

### GVD-TG-inspired switching connections

- Added a topology repair pass inspired by the GVD-TG switching connection
  mechanism. Native compressed skeleton edges remain the first choice. When
  thinning leaves reachable components disconnected, the repair pass adds a
  collision-free bridge computed by deterministic bidirectional A*.
- Added a bounded LRU cache for connectivity queries. Cached positive paths are
  revalidated against the latest obstacle-only raster, including diagonal
  corner checks; negative results are scoped to the current map revision.
- Added `gvd_switching_connections_enabled`,
  `gvd_connection_neighbor_limit`, and `gvd_connection_cache_size` parameters
  for tuning and ablation studies.

Reference: [GVD-TG paper](https://arxiv.org/abs/2511.18708) and the authors'
[public implementation](https://github.com/littleBurgerrr/Hierarchical_GVD_Exploration).

Status: local Python compilation, `git diff --check`, and available pure Python
helper regression tests passed (`74 passed`). Remote ROS smoke validation is
pending.

## 2026-05-31 - One-pass hierarchical GVD exploration

- Added `slam_mode:=gvd_hierarchical` while preserving `gvd_gbsae` as a
  baseline. The new mode follows a three-phase state machine: macro live-GVD
  traversal, local frontier cleanup at exhausted branches, and global frontier
  tail cleanup.
- Added spatial migration for explored and locally-cleared macro vertices
  across live GVD rebuilds. Macro selection uses a simple local-unknown-area
  over topo-distance utility.
- Added rectangular known-free flood masks for micro cleanup. Local frontiers
  cannot leak through walls into adjacent rooms and are deliberately ranked by
  the baseline `cluster size / distance` rule.
- Added RViz markers for explored, locally cleared, and active macro vertices.
- Added a translucent cyan RViz cell overlay for the current local-cleanup
  flood mask. It visualizes the wall-constrained known-free region rather than
  only the configured bounding rectangle.

Status: local Python compilation, `git diff --check`, and available pure Python
helper regression tests passed (`81 passed`). Remote ROS smoke validation is
pending.

## 2026-06-01 - Mode-specific frontier probe defaults

- Enabled unknown-frontier and open-edge probes by default for `frontier`,
  `approx_graph`, and static `gbsae`.
- Disabled all frontier probes by default for `gvd_gbsae` and
  `gvd_hierarchical`, including hierarchical local cleanup and tail cleanup.
- Added `frontier_mode_probes_enabled` and `gvd_mode_probes_enabled` as
  mode-level ablation switches while retaining the existing per-probe-type
  switches.

Status: local Python compilation, `git diff --check`, and available pure Python
helper regression tests passed (`87 passed`). Remote ROS smoke validation is
pending.

## 2026-06-01 - Local cleanup rectangular Region constraints

- Replaced same-result four-edge flood attempts with one-edge greedy expansion
  strategies and selected the candidate closest to a square, with larger area
  as the tie-breaker.
- Limited each local-cleanup Region to the configured coarse world bounds.
- Stopped candidate expansion before it can include another live GVD vertex,
  keeping exhausted-node cleanup local to one macro region.
- Kept the existing local frontier ranking and Nav2 cleanup dispatch behavior.

Status: local Python compilation, `git diff --check`, focused GVD helper tests
(`28 passed`), and the available pure Python regression suite (`90 passed`)
passed. Remote ROS smoke validation is pending.

## 2026-06-01 - Robot-radius GVD connectivity

- Changed `gvd_obstacle_clearance` from the permissive `0.04m` value to the
  TurtleBot footprint radius `0.18m`.
- Kept one shared inflated obstacle raster for skeleton construction, native
  GVD connectivity, switching-connection bidirectional A* fallback, local GVD
  A*, and active-path invalidation.
- Added a regression showing that a one-pixel doorway is rejected after
  footprint inflation while a sufficiently wide doorway remains connected.

Status: local Python compilation, `git diff --check`, focused GVD helper tests
(`29 passed`), and the available pure Python regression suite (`91 passed`)
passed. Remote ROS smoke validation is pending.

## 2026-06-01 - Low-confidence fill for tiny frontier fragments

- Increased the default minimum ordinary frontier cluster size from `5px` to
  `10px`.
- Added a conservative detector pre-pass for smaller frontier fragments.
  Bounded adjacent unknown components use surrounding free and occupied
  geometry votes and are filled with low-confidence occupancy values `25` or
  `75` instead of remaining unknown.
- Kept large unknown components and components touching the `/map` array edge
  unchanged to avoid hiding genuine unexplored entrances.
- Cached the inferred grid in the coordinator so frontier selection, GVD,
  flood Regions, information gain, and explored-history updates share the same
  occupancy interpretation. Low-confidence free cells remain invalid as safe
  goals because safe-goal search still requires exact known-free value `0`.

Status: local Python compilation, workspace and index `git diff --check`,
focused detector helper tests (`14 passed`), and the available pure Python
regression suite (`104 passed`) passed. Remote ROS smoke validation is pending.

## 2026-06-01 - Hierarchical local cleanup goal freedom and probes

- Removed the requirement that a hierarchical local-cleanup safe goal itself
  lie inside the active flood Region. The selected frontier cluster must still
  touch the Region, so completion remains defined by clearing Region-local
  frontiers while Nav2 may choose a safer standoff cell just outside it.
- Enabled frontier `Spin -> DriveOnHeading` probes during
  `gvd_hierarchical` local flood cleanup without enabling probes for macro GVD
  vertex traversal or the global tail-cleanup phase.
- Added `gvd_hierarchical_local_probes_enabled: true` as an independent
  ablation switch. `gvd_mode_probes_enabled` remains the broader GVD-mode
  override.

Status: local Python compilation, workspace and index `git diff --check`,
focused frontier selection and safe-goal helper tests (`33 passed`), and the
available pure Python regression suite (`106 passed`) passed. Remote ROS smoke
validation is pending.

## 2026-06-01 - Permissive switching bridges and distance-only macro targets

- Kept `gvd_obstacle_clearance: 0.18` for skeleton construction, local GVD A*,
  and active-path invalidation.
- Added `gvd_reconnection_clearance: 0.04` only for switching-connection
  fallback bidirectional A*. This prevents coarse inflation from discarding
  narrow doorway hypotheses before Nav2 performs the real path check.
- Updated the topology connection cache to validate native GVD bridges against
  the strict mask and fallback A* bridges against the permissive mask.
- Changed `gvd_hierarchical` macro target selection to use topology distance
  only. Local unknown area no longer affects the ordering.

Status: local Python compilation, workspace and index `git diff --check`,
focused GVD helper tests (`33 passed`), and the available pure Python
regression suite (`107 passed`) passed. Remote ROS smoke validation is pending.

## 2026-06-01 - Hierarchical Region RViz snapshot rendering

- Stored the `/map` geometry snapshot together with the active local-cleanup
  mask instead of rendering an older mask using the newest dynamic map origin.
- Archived each completed rectangular Region as a compact world-space outline.
  Regions with no local frontier candidate now remain visible after the
  immediate local-clear transition instead of disappearing before RViz can
  publish them.
- Kept the active Region as a translucent cyan cell fill and render completed
  Regions as cyan rectangle outlines.

Status: local Python compilation, `git diff --check`, focused GVD helper tests
(`30 passed`), and the available pure Python regression suite (`92 passed`)
passed. Remote ROS smoke validation is pending.

## 2026-06-01 - Unknown-permissive Region flood and bounded local goals

- Changed hierarchical local Region expansion to stop on occupied cells,
  coarse bounds, the configured size limit, or another live GVD vertex.
  Unknown cells no longer stop Region growth.
- Kept safe goals conservative: they must still be known-free and now must
  also lie inside the active local Region.
- Passed the Region mask into safe-goal search so a cluster crossing the
  boundary can still select an interior fallback goal.

Status: local Python compilation, `git diff --check`, focused Region and
safe-goal helper tests (`51 passed`), and the available pure Python regression
suite (`95 passed`) passed. Remote ROS smoke validation is pending.

## 2026-06-01 - Leaf-centered weighted Region selection

- Replaced direction-priority rectangular growth with centered rectangle
  enumeration. The active leaf remains at the Region center instead of
  drifting toward an edge or corner.
- Select the Region using a weighted sum of normalized area and squareness.
- Added `gvd_hierarchical_region_area_weight` and
  `gvd_hierarchical_region_squareness_weight`, both defaulting to `1.0`.

Status: local Python compilation, `git diff --check`, focused GVD helper tests
(`32 passed`), and the available pure Python regression suite (`96 passed`)
passed. Remote ROS smoke validation is pending.

## 2026-06-01 - Per-Region approximate graph cleanup scoring

- Added an independent approximate pose-graph tracker for each hierarchical
  local-cleanup Region. Its trajectory state is discarded when the Region is
  complete instead of leaking across rooms.
- Changed local cleanup from first-reachable dispatch to serial Nav2 path
  checks followed by D-opt style comparison of every reachable Region-local
  candidate.
- Added `gvd_hierarchical_local_approx_graph_enabled: true` so the prior
  first-reachable behavior remains available for ablation.

Status: local Python compilation, workspace and index `git diff --check`,
focused graph and GVD helper tests (`36 passed`), and the available pure Python
regression suite (`98 passed`) passed. Remote ROS smoke validation is pending.

## 2026-06-01 - Relaxed hierarchical local-cleanup trigger

- Restored eager local cleanup for GVD leaves with graph degree `<= 1`.
- Also trigger local cleanup for branch vertices once at most one neighboring
  branch remains unexplored, so the Region sweep runs before macro traversal
  commits to the final exit.
- Kept the rectangular Region construction, local frontier ranking, and Nav2
  dispatch behavior unchanged.

Status: local Python compilation, `git diff --check`, focused GVD helper tests
(`29 passed`), and the available pure Python regression suite (`91 passed`)
passed. Remote ROS smoke validation is pending.

## 2026-06-01 - Dynamic open-TSP hierarchical GVD traversal

- Replaced nearest-unexplored macro selection in `gvd_hierarchical` with a
  NetworkX `3.4.x` open-TSP walk over uncleared live GVD vertices.
- Kept cleared vertices as shortest-path transit nodes and retained
  explored-but-uncleared vertices as revisit targets.
- Added explicit corner vertices, increased support spacing to `2.0m`, and
  replaced unrestricted pairwise merging with deterministic local GVD-chain
  clustering that preserves switching-bridge metadata.
- Added `0.5s` dirty-route throttling. Map updates refresh later TSP steps
  without preempting an active macro Nav2 goal merely because the first step
  changed. Region cleanup and bounded recovery remain uninterrupted.
- Remap an in-flight macro target by world position after a live graph rebuild,
  so the eventual Nav2 success callback still updates the corresponding new
  graph vertex without changing the dispatched geometric goal.
- Added an RViz marker for the remaining hierarchical TSP route.

Status: local Python compilation and `git diff --check` passed. Local pytest
is blocked because the generic host environment does not provide NetworkX.
Remote ROS smoke validation is pending.

## 2026-06-01 - RViz rendering for switching A* bridges

- Added an orange `gvd_astar_reconnections` marker that renders the stored
  world-space segments of fallback bidirectional-A* topology bridges.
- Kept raw thinned GVD skeleton edges cyan and the hierarchical TSP route
  yellow so connectivity hypotheses remain visually distinguishable.
- Added a pure helper regression test that excludes native GVD edges from the
  rendered A* bridge segment list.

Status: local Python compilation and `git diff --check` passed. Local pytest
remains blocked because the generic host environment does not provide
NetworkX. Remote RViz smoke validation is pending.

## 2026-06-01 - Unknown-loop suppression for live GVD topology

- Marked GVD vertices unconfident when the local `1.0m` disk contains at least
  `50%` unknown or out-of-map cells.
- Reduced each induced unconfident connected region to MST edges after
  switching repair and stable clustering. Retained edges return to the normal
  macro graph without a special TSP cost.
- Added purple RViz node markers and route-rebuild logs with unconfident vertex
  and removed-edge counts.
- Kept throttled GVD rebuild and TSP regeneration active during hierarchical
  Region cleanup, frontier navigation, and local probe actions without
  preempting those local actions.
- Added ablation parameters:
  `gvd_unknown_cycle_suppression_enabled`,
  `gvd_unconfident_unknown_radius`, and
  `gvd_unconfident_unknown_ratio`.

Status: local Python compilation and `git diff --check` passed. Local pytest
remains blocked because the generic host environment does not provide
NetworkX. Remote ROS and RViz smoke validation are pending.

## 2026-06-01 - History-aware hierarchical TSP replanning

- Preserved the previous remaining macro-route geometry across live GVD
  rebuilds and mapped it onto nearby vertices in the rebuilt graph.
- Compared fresh NetworkX TSP, unexplored-first, and history-preserving route
  candidates. Candidates outside a configurable `10%` shortest-route slack are
  rejected before continuity scoring.
- Added a geometric directed-segment continuity score so similarly short
  routes prefer the previous macro direction instead of changing branches on
  small map updates.
- Reduced explored-but-uncleared target priority to `0.25` while retaining
  those vertices as valid cleanup targets and shortest-path transit.
- Logged the selected route strategy, length, history distance, and unexplored
  priority score for remote tuning.
- Added bounded heuristic TSP refinement over the initial route pool. Each
  rebuild interleaves up to `30` deterministic `2-opt`, swap, and insertion
  mutations, expands target orders through cached weighted shortest paths, and
  rejects candidates outside the configured shortest-route slack.
- Added a normalized turn penalty so similarly short, history-compatible
  routes favor smoother macro motion through the live topology.
- Logged the selected route turn penalty and added tuning parameters:
  `gvd_hierarchical_turn_penalty_weight` and
  `gvd_hierarchical_tsp_local_search_iterations`.

Status: local Python compilation and `git diff --check` passed. Local pytest
remains blocked because the generic host environment does not provide
NetworkX. Remote ROS smoke validation is pending.

## 2026-06-02 - Refresh endpoint kinds after live GVD transforms

- Normalize compressed topology node kinds after connectivity repair,
  clustering, and unknown-heavy cycle pruning.
- Promote every final degree-0/1 vertex to `endpoint`, so newly exposed leaves
  enter the hierarchical expansion TSP instead of remaining stale
  `support`/`corner` transit-only nodes.
- Added a regression test for a pruned chain whose stale support labels must
  become endpoint labels at both ends.

Status: local Python compilation and `git diff --check` passed. Local pytest
remains blocked because the generic host environment does not provide
NetworkX. Remote ROS smoke validation is pending.

## 2026-06-02 - Direction-locked hierarchical TSP rebuilds

- Replaced route-candidate enumeration, length slack, turn scoring, and bounded
  `2-opt` / swap / insertion refinement with one open NetworkX TSP call.
- Before a live graph replacement, cache the old active-to-next-step direction.
  If the rebuilt active vertex remains non-leaf, mark it explored and force the
  most direction-aligned available neighbor as the next macro hop.
- If the rebuilt active vertex is cleanup-eligible, skip macro route generation
  so the coordinator immediately enters Region cleanup.
- Removed `gvd_hierarchical_route_length_slack`,
  `gvd_hierarchical_turn_penalty_weight`, and
  `gvd_hierarchical_tsp_local_search_iterations`.

Status: local Python compilation and `git diff --check` passed. Local pytest
remains blocked because the generic host environment does not provide
NetworkX. Remote ROS smoke validation is pending.

## 2026-06-02 - Layered hierarchical expansion and Region cleanup

- Replaced mixed unexplored-versus-cleanup TSP weighting with two explicit
  phases: expansion over unexplored endpoints and necessary branch fallbacks,
  followed by cleanup over uncleared ordinary and degenerate leaves.
- Removed `gvd_hierarchical_unexplored_priority_weight` and related telemetry.
  Within each route phase, length slack, turn penalty, and bounded local search
  remain active.
- Simplified Region eligibility to uncleared Region-bearing leaves. A branch is
  also eligible once none of its outgoing subtrees contains expansion work.
  Native GVD and fallback A* edges participate equally.
- Kept immediate cleanup when the robot is near an eligible active vertex and
  retained non-preemptive Nav2 macro navigation across live topology rebuilds.

Status: local Python compilation and `git diff --check` passed. Local pytest
remains blocked because the generic host environment does not provide
NetworkX. Remote ROS smoke validation is pending.

## 2026-06-02 - Remove historical-route similarity from hierarchical TSP

- Removed previous-route geometry migration, historical-route candidate
  generation, similarity scoring, telemetry, and the
  `gvd_hierarchical_route_history_weight` parameter.
- Restored an explicit unexplored-first preference. Among near-shortest route
  candidates, unexplored macro targets receive a larger score when visited
  earlier than reached-but-uncleared leaf Region revisits.
- Added `gvd_hierarchical_unexplored_priority_weight: 1.0`.
- Retained route-length slack, turn penalty, and bounded `2-opt`, swap, and
  insertion local search.

Status: local Python compilation and `git diff --check` passed. Local pytest
remains blocked because the generic host environment does not provide
NetworkX. Remote ROS smoke validation is pending.

## 2026-06-02 - Resolve completed macro goals against the latest GVD graph

- Kept the intended behavior that a live TSP refresh does not cancel an
  already dispatched Nav2 macro goal.
- Resolve a completed macro goal by world-space position before marking a GVD
  vertex reached. Rebuilt GVD graphs assign fresh numeric IDs, so trusting the
  pre-rebuild vertex ID could mark an unrelated node or skip leaf cleanup.
- If the completed goal no longer maps near any current graph vertex, return
  to macro replanning without clearing an unrelated Region.

Status: local Python compilation and `git diff --check` passed. Local pytest
remains blocked because the generic host environment does not provide
NetworkX. Remote ROS smoke validation is pending.

## 2026-06-02 - Region-bearing hierarchical GVD TSP targets

- Restricted hierarchical macro TSP targets to unexplored endpoints, necessary
  branch fallbacks for components without endpoint targets, and reached
  uncleared leaf Regions.
- Kept support and corner vertices in the compressed graph as shortest-path
  transit steps only.
- Added degree-based node-kind inference for legacy tests and graphs without
  explicit compressed-node metadata.

Status: local Python compilation and `git diff --check` passed. Local pytest
remains blocked because the generic host environment does not provide
NetworkX. Remote ROS smoke validation is pending.

## 2026-06-02 - Lean hierarchical startup and RViz topology view

- Removed the unconditional initial exploration Spin. Start selecting goals
  as soon as Nav2 and the first map are ready; recovery and probe Spin actions
  remain available.
- Added mutually exclusive pruned-topology node markers: purple unexplored,
  pink explored-but-uncleared, and orange cleared.
- Render each directed TSP traversal independently. Repeated use of the same
  undirected edge receives parallel offsets so reverse trips remain visible.
- Kept detailed GVD markers published for manual debugging, while the default
  RViz profile enables only `/map`, the three state-node namespaces, and the
  directed TSP route namespaces.
- Abort Region cleanup after a live rebuild if its active explored vertex is
  no longer a leaf. Preserve explored state and return to macro routing
  without marking the vertex cleared.

Status: local Python compilation and `git diff --check` passed. Local pytest
remains blocked because the generic host environment does not provide
NetworkX. Remote ROS and RViz smoke validation are pending.

## 2026-06-02 - Global high-priority random-walk recovery

- Extended the effective-translation watchdog from GVD macro phases to every
  baseline once Nav2 and the first map are ready.
- Keep the existing `gvd_stuck_*` and `gvd_random_recovery_*` parameter names
  for compatibility; `gvd_stuck_timeout` remains `10.0s`.
- Allow the bounded Nav2 `Spin -> DriveOnHeading` recovery to preempt idle,
  path selection, navigation, frontier-probe alignment, and frontier-probe
  drive states.
- Exclude recovery actions themselves from watchdog triggering to prevent
  recursive recovery.
- Fixed Region coverage projection by replacing the incorrect two-dimensional
  `np.flatnonzero()` unpack with `np.nonzero()`.

Status: local Python compilation and `git diff --check` passed. Local pytest
remains blocked because the generic host environment does not provide
NetworkX. Remote ROS smoke validation is pending.

## 2026-06-01 - Early completion for hierarchical local cleanup

- Added `gvd_hierarchical_local_clear_progress_threshold: 0.90`.
- Measure local cleanup progress as the fraction of non-unknown cells inside
  the current rectangular Region mask.
- Mark a Region locally cleared once its observed coverage reaches the
  threshold, without waiting for every residual frontier to disappear.
- Check progress from the `0.2s` coordinator loop and cancel an in-flight
  local path request, navigation goal, Spin, or DriveOnHeading probe once the
  threshold is reached.
- Kept the existing no-reachable-local-frontier completion path as a fallback
  and routed both completion reasons through one Region archival helper.

Status: local Python compilation and `git diff --check` passed. Local pytest
remains blocked because the generic host environment does not provide
NetworkX. Remote ROS smoke validation is pending.

## 2026-06-01 - Transit-only explored hierarchical GVD vertices

- Removed explored-but-uncleared vertices from the hierarchical open-TSP
  target set. They remain available as shortest-path transit steps.
- Replaced final-route-occurrence cleanup with a topological leaf rule:
  a reached vertex triggers Region cleanup only when its full graph degree is
  at most one.
- Counted native GVD and fallback A* reconnection edges equally for leaf
  detection.
- Removed the explored-target weight, unexplored-priority weight, and related
  route telemetry. History alignment, length slack, turn penalty, and bounded
  local search remain active.

Status: local Python compilation and `git diff --check` passed. Local pytest
remains blocked because the generic host environment does not provide
NetworkX. Remote ROS smoke validation is pending.
