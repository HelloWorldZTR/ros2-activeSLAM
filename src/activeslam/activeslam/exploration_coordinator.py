import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .frontier_detector import FrontierCluster, FrontierDetector
from .frontier_goal_utils import (
    FailedGoalCooldown,
    GridGeometry,
    PreparedSafeGoalGrid,
    SafeFrontierGoal,
    SafeGoalSearchConfig,
    navigation_timed_out,
    normalize_angle,
    open_edge_outward_normal,
    potential_unknown_area,
    prepare_safe_goal_grid,
    select_safe_frontier_goal,
    unknown_frontier_outward_normal,
)
from .frontier_selection import (
    OPEN_EDGE_FRONTIER,
    UNKNOWN_FRONTIER,
    FrontierCandidate,
    make_frontier_candidate,
    ranked_frontier_candidates,
)
from .graph_exploration import (
    ApproximatePoseGraphTracker,
    GraphBasedFrontierScorer,
    graph_to_marker_array,
    make_information_matrix,
)
from .gbsae_exploration import (
    GBSAEPlanner,
    gbsae_to_marker_array,
    load_prior_graph,
    online_bounds_marker_array,
    point_is_known_free,
    resolve_prior_graph_path,
    vertex_point,
)
from .nav2_backend import (
    GOAL_STATUS_SUCCEEDED,
    Nav2Backend,
    PlannedPath,
    heading_to_target,
)
from .online_gbsae import (
    BootstrapScore,
    BootstrapWeights,
    BranchHypothesis,
    WorldBounds,
    bootstrap_score,
    branch_path_has_no_known_obstacle,
    build_online_topology,
    directional_remaining_unknown,
    known_area_ratio,
    load_world_bounds,
    mark_branch_explored,
    normal_unknown_depth,
    path_known_ratio,
    record_branch_hypotheses,
    resolve_world_bounds_path,
    update_branch_failure,
)


def _yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class ExplorationCoordinator(Node):
    """Select exploration goals while Nav2 owns planning, control, and recovery."""

    WAITING_FOR_NAV2 = 'waiting_for_nav2'
    INITIAL_SPIN = 'initial_spin'
    IDLE = 'idle'
    SELECTING = 'selecting'
    NAVIGATING = 'navigating'
    ALIGNING_FRONTIER_PROBE = 'aligning_frontier_probe'
    PROBING_FRONTIER = 'probing_frontier'
    COMPLETE = 'complete'

    def __init__(self):
        super().__init__('exploration_coordinator')

        # --- Parameters ---
        self.stability_duration = self.declare_parameter('stability_duration', 10.0).value
        self.stability_threshold = self.declare_parameter('stability_threshold', 0.02).value
        self.min_frontier_size = self.declare_parameter('min_frontier_size', 5).value
        self.frontier_include_open_map_edges = self.declare_parameter(
            'frontier_include_open_map_edges', True
        ).value
        self.slam_mode = self.declare_parameter('slam_mode', 'frontier').value
        self.world_name = self.declare_parameter('world_name', 'slam_rooms').value
        self.frontier_planning_attempts = int(
            self.declare_parameter('frontier_planning_attempts', 3).value
        )
        self.frontier_goal_search_radius = self.declare_parameter(
            'frontier_goal_search_radius', 1.2
        ).value
        self.frontier_goal_clearance = self.declare_parameter(
            'frontier_goal_clearance', 0.20
        ).value
        self.frontier_goal_standoff = self.declare_parameter(
            'frontier_goal_standoff', 0.45
        ).value
        self.frontier_goal_map_edge_clearance = self.declare_parameter(
            'frontier_goal_map_edge_clearance', 0.0
        ).value
        self.frontier_goal_point_sample_limit = int(
            self.declare_parameter('frontier_goal_point_sample_limit', 40).value
        )
        self.frontier_information_gain_radius = self.declare_parameter(
            'frontier_information_gain_radius', 1.0
        ).value
        self.nav2_goal_reach_radius = self.declare_parameter(
            'nav2_goal_reach_radius', 0.25
        ).value
        self.failed_goal_cooldown = self.declare_parameter(
            'failed_goal_cooldown', 20.0
        ).value
        self.failed_goal_radius = self.declare_parameter(
            'failed_goal_radius', 0.6
        ).value
        self.frontier_retry_interval = self.declare_parameter(
            'frontier_retry_interval', 3.0
        ).value
        self.nav2_request_timeout = self.declare_parameter(
            'nav2_request_timeout', 5.0
        ).value
        self.initial_spin_yaw = self.declare_parameter(
            'initial_spin_yaw', 2.0 * math.pi
        ).value
        self.initial_spin_timeout = self.declare_parameter(
            'initial_spin_timeout', 30.0
        ).value
        self.nav2_goal_timeout = self.declare_parameter(
            'nav2_goal_timeout', 30.0
        ).value
        self.frontier_open_edge_probe_enabled = self.declare_parameter(
            'frontier_open_edge_probe_enabled', True
        ).value
        self.frontier_probe_normal_radius = self.declare_parameter(
            'frontier_probe_normal_radius', 0.35
        ).value
        self.frontier_probe_spin_tolerance = self.declare_parameter(
            'frontier_probe_spin_tolerance', 0.10
        ).value
        self.frontier_probe_spin_timeout = self.declare_parameter(
            'frontier_probe_spin_timeout', 8.0
        ).value
        self.frontier_open_edge_probe_distance = self.declare_parameter(
            'frontier_open_edge_probe_distance', 2.0
        ).value
        self.frontier_open_edge_probe_speed = self.declare_parameter(
            'frontier_open_edge_probe_speed', 0.12
        ).value
        self.frontier_open_edge_probe_timeout = self.declare_parameter(
            'frontier_open_edge_probe_timeout', 20.0
        ).value
        self.frontier_unknown_probe_enabled = self.declare_parameter(
            'frontier_unknown_probe_enabled', True
        ).value
        self.frontier_unknown_probe_distance = self.declare_parameter(
            'frontier_unknown_probe_distance', 0.45
        ).value
        self.frontier_unknown_probe_speed = self.declare_parameter(
            'frontier_unknown_probe_speed', 0.08
        ).value
        self.frontier_unknown_probe_timeout = self.declare_parameter(
            'frontier_unknown_probe_timeout', 8.0
        ).value
        self.graph_max_frontier_candidates = int(
            self.declare_parameter('graph_max_frontier_candidates', 8).value
        )
        self.graph_info_radius = self.declare_parameter('graph_info_radius', 1.5).value
        self.graph_node_spacing = self.declare_parameter('graph_node_spacing', 0.5).value
        self.graph_yaw_spacing = self.declare_parameter('graph_yaw_spacing', 0.35).value
        self.graph_hallucinated_node_spacing = self.declare_parameter(
            'graph_hallucinated_node_spacing', 0.75
        ).value
        self.graph_loop_closure_radius = self.declare_parameter(
            'graph_loop_closure_radius', 2.0
        ).value
        self.graph_loop_closure_min_separation = int(
            self.declare_parameter('graph_loop_closure_min_separation', 20).value
        )
        self.graph_loop_closure_occupied_threshold = self.declare_parameter(
            'graph_loop_closure_occupied_threshold', 0.03
        ).value
        self.graph_loop_closure_weight = self.declare_parameter(
            'graph_loop_closure_weight', 1.5
        ).value
        self.graph_max_loop_closures_per_node = int(
            self.declare_parameter('graph_max_loop_closures_per_node', 3).value
        )
        self.graph_path_cost_weight = self.declare_parameter(
            'graph_path_cost_weight', 0.05
        ).value
        self.graph_odom_cov_x = self.declare_parameter('graph_odom_cov_x', 0.04).value
        self.graph_odom_cov_y = self.declare_parameter('graph_odom_cov_y', 0.04).value
        self.graph_odom_cov_yaw = self.declare_parameter('graph_odom_cov_yaw', 0.008).value
        self.gbsae_loop_path_cost_weight = self.declare_parameter(
            'gbsae_loop_path_cost_weight', 0.01
        ).value
        self.online_gbsae_bootstrap_known_ratio = self.declare_parameter(
            'online_gbsae_bootstrap_known_ratio', 0.50
        ).value
        self.online_gbsae_bootstrap_shortlist = int(
            self.declare_parameter('online_gbsae_bootstrap_shortlist', 6).value
        )
        self.online_gbsae_unknown_depth_max = self.declare_parameter(
            'online_gbsae_unknown_depth_max', 2.0
        ).value
        self.online_gbsae_direction_half_angle = self.declare_parameter(
            'online_gbsae_direction_half_angle', math.pi / 4.0
        ).value
        self.online_gbsae_weights = BootstrapWeights(
            normal_unknown_depth=self.declare_parameter(
                'online_gbsae_normal_unknown_depth_weight', 5.0
            ).value,
            directional_remaining_unknown=self.declare_parameter(
                'online_gbsae_directional_remaining_unknown_weight', 1.5
            ).value,
            path_known_ratio_penalty=self.declare_parameter(
                'online_gbsae_path_known_ratio_penalty_weight', 100.0
            ).value,
        )
        self.online_gbsae_bootstrap_probe_enabled = self.declare_parameter(
            'online_gbsae_bootstrap_probe_enabled', True
        ).value
        self.online_gbsae_bootstrap_probe_distance = self.declare_parameter(
            'online_gbsae_bootstrap_probe_distance', 2.0
        ).value
        self.online_gbsae_bootstrap_probe_speed = self.declare_parameter(
            'online_gbsae_bootstrap_probe_speed', 0.2
        ).value
        self.online_gbsae_bootstrap_probe_timeout = self.declare_parameter(
            'online_gbsae_bootstrap_probe_timeout', 20.0
        ).value
        self.online_gbsae_directional_prior_enabled = self.declare_parameter(
            'online_gbsae_directional_prior_enabled', True
        ).value
        self.online_gbsae_branch_hypotheses_enabled = self.declare_parameter(
            'online_gbsae_branch_hypotheses_enabled', True
        ).value
        self.online_gbsae_explored_migration_enabled = self.declare_parameter(
            'online_gbsae_explored_migration_enabled', True
        ).value
        self.online_gbsae_branch_score_ratio = self.declare_parameter(
            'online_gbsae_branch_score_ratio', 0.70
        ).value
        self.online_gbsae_branch_min_angle = self.declare_parameter(
            'online_gbsae_branch_min_angle', math.pi / 4.0
        ).value
        self.online_gbsae_branch_merge_radius = self.declare_parameter(
            'online_gbsae_branch_merge_radius', 1.0
        ).value
        self.online_gbsae_branch_projection_distance = self.declare_parameter(
            'online_gbsae_branch_projection_distance', 1.0
        ).value
        self.online_gbsae_branch_failure_limit = int(
            self.declare_parameter('online_gbsae_branch_failure_limit', 3).value
        )
        self.online_gbsae_topology_clearance = self.declare_parameter(
            'online_gbsae_topology_clearance', 0.20
        ).value
        self.online_gbsae_spur_prune_length = self.declare_parameter(
            'online_gbsae_spur_prune_length', 0.50
        ).value
        self.online_gbsae_support_vertex_spacing = self.declare_parameter(
            'online_gbsae_support_vertex_spacing', 2.0
        ).value
        self.online_gbsae_migration_radius = self.declare_parameter(
            'online_gbsae_migration_radius', 1.0
        ).value
        self.online_gbsae_migration_overlap = self.declare_parameter(
            'online_gbsae_migration_overlap', 0.50
        ).value

        if self.slam_mode not in ('frontier', 'approx_graph', 'gbsae', 'online_gbsae'):
            self.get_logger().warn(
                f'Unknown slam_mode={self.slam_mode}. '
                'Falling back to frontier.'
            )
            self.slam_mode = 'frontier'

        # --- Publishers and subscribers ---
        self.goal_pub = self.create_publisher(Marker, '/goal_point', 10)
        self.frontier_pub = self.create_publisher(MarkerArray, '/frontier_markers', 10)
        self.pose_graph_pub = self.create_publisher(MarkerArray, '/pose_graph_markers', 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self._map_callback, 10)

        # --- TF and components ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.frontier_detector = FrontierDetector(
            min_frontier_size=self.min_frontier_size,
            include_open_map_edges=self.frontier_include_open_map_edges,
        )
        self.nav2 = Nav2Backend(self)

        graph_odom_information = make_information_matrix(
            self.graph_odom_cov_x,
            self.graph_odom_cov_y,
            self.graph_odom_cov_yaw,
        )
        self.pose_graph_tracker = ApproximatePoseGraphTracker(
            node_spacing=self.graph_node_spacing,
            yaw_spacing=self.graph_yaw_spacing,
            loop_closure_radius=self.graph_loop_closure_radius,
            loop_closure_min_separation=self.graph_loop_closure_min_separation,
            loop_closure_weight=self.graph_loop_closure_weight,
            max_loop_closures_per_node=self.graph_max_loop_closures_per_node,
            odom_information=graph_odom_information,
        )
        self.graph_scorer = GraphBasedFrontierScorer(
            info_radius=self.graph_info_radius,
            hallucinated_node_spacing=self.graph_hallucinated_node_spacing,
            loop_closure_radius=self.graph_loop_closure_radius,
            loop_closure_min_separation=self.graph_loop_closure_min_separation,
            loop_closure_occupied_threshold=self.graph_loop_closure_occupied_threshold,
            loop_closure_weight=self.graph_loop_closure_weight,
            max_loop_closures_per_node=self.graph_max_loop_closures_per_node,
            path_cost_weight=self.graph_path_cost_weight,
            odom_information=graph_odom_information,
        )
        self.gbsae_prior_graph = None
        self.gbsae_planner: Optional[GBSAEPlanner] = None
        self.online_gbsae_bounds: Optional[WorldBounds] = None
        if self.slam_mode == 'gbsae':
            prior_graph_path = resolve_prior_graph_path(self.world_name)
            self.gbsae_prior_graph = load_prior_graph(prior_graph_path, self.world_name)
            self.get_logger().info(
                f'Loaded GBSAE prior graph for world={self.world_name}: '
                f'{self.gbsae_prior_graph.number_of_nodes()} nodes, '
                f'{self.gbsae_prior_graph.number_of_edges()} edges from {prior_graph_path}.'
            )
        elif self.slam_mode == 'online_gbsae':
            self.online_gbsae_bounds = load_world_bounds(
                resolve_world_bounds_path(),
                self.world_name,
            )
            self.get_logger().info(
                f'Loaded coarse online GBSAE bounds for world={self.world_name}: '
                f'[{self.online_gbsae_bounds.min_x:.1f}, '
                f'{self.online_gbsae_bounds.max_x:.1f}] x '
                f'[{self.online_gbsae_bounds.min_y:.1f}, '
                f'{self.online_gbsae_bounds.max_y:.1f}].'
            )

        # --- Exploration state ---
        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_grid: Optional[np.ndarray] = None
        self.frontier_clusters: List[FrontierCluster] = []
        self.current_goal: Optional[Tuple[float, float]] = None
        self.current_safe_goal: Optional[SafeFrontierGoal] = None
        self.target_cluster: Optional[FrontierCluster] = None
        self.frontier_probe_normal: Optional[Tuple[float, float]] = None
        self.explored_history = deque()
        self.state = self.WAITING_FOR_NAV2
        self.next_retry_wall_time = 0.0
        self.selection_generation = 0
        self.selection_start_xy = (0.0, 0.0)
        self.selection_candidates: List[FrontierCandidate] = []
        self.selection_cluster_index = 0
        self.graph_candidates = []
        self.selection_request_wall_time: Optional[float] = None
        self.selection_request_goal: Optional[Tuple[float, float]] = None
        self.selection_kind = 'frontier'
        self.current_navigation_kind = 'frontier'
        self.current_navigation_vertex_id: Optional[int] = None
        self.navigation_start_wall_time: Optional[float] = None
        self.frontier_probe_action_start_wall_time: Optional[float] = None
        self.failed_goals = FailedGoalCooldown(
            self.failed_goal_cooldown,
            self.failed_goal_radius,
        )
        self.last_wait_log_wall_time = 0.0
        self.online_gbsae_phase = 'bootstrap'
        self.online_gbsae_branches: List[BranchHypothesis] = []
        self.online_gbsae_topology_executor = ThreadPoolExecutor(max_workers=1)
        self.online_gbsae_topology_future: Optional[Future] = None
        self.online_gbsae_topology_generation = 0
        self.online_gbsae_future_generation = 0
        self.online_gbsae_topology_version = 0
        self.online_gbsae_build_pose = (0.0, 0.0)

        self.control_timer = self.create_timer(0.2, self._control_loop)

        self.get_logger().info(
            f'Exploration coordinator started with Nav2 backend. '
            f'SLAM mode: {self.slam_mode}'
        )
        if self.slam_mode == 'approx_graph':
            self.get_logger().info(
                'Pose graph source: approximate TF trajectory. slam_toolbox can serialize '
                'its pose graph, but this node does not receive live nodes/edges/FIM from it.'
            )

    # ------------------------------------------------------------------
    # Main callbacks
    # ------------------------------------------------------------------

    def _map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg
        self.latest_grid = np.asarray(msg.data, dtype=np.int8).reshape(
            msg.info.height,
            msg.info.width,
        )
        self.frontier_clusters, _ = self.frontier_detector.detect(msg, self.latest_grid)
        self._update_explored_history(self.latest_grid)

    def _control_loop(self):
        self._publish_visualizations()
        pose = self._get_robot_pose()
        if pose is not None:
            self.pose_graph_tracker.update(pose)
            self._initialize_gbsae(pose)
            self._poll_online_topology_build(pose)

        if self.state == self.COMPLETE:
            return
        if not self.nav2.servers_ready():
            self._log_waiting_for_nav2()
            return
        if self.state == self.WAITING_FOR_NAV2:
            if self.latest_map is not None:
                self._start_initial_spin()
            return
        if self.state == self.INITIAL_SPIN:
            return
        if self.state == self.NAVIGATING:
            if navigation_timed_out(
                self.navigation_start_wall_time,
                self.nav2_goal_timeout,
                time.monotonic(),
            ):
                self.get_logger().warn(
                    f'Nav2 goal timed out after {self.nav2_goal_timeout:.1f}s; '
                    'canceling and selecting another target.'
                )
                self.nav2.cancel_navigation()
                self._handle_navigation_failure('Nav2 navigation timed out')
                self._clear_navigation()
                self._schedule_retry(0.0)
            return
        if self.state == self.ALIGNING_FRONTIER_PROBE:
            if navigation_timed_out(
                self.frontier_probe_action_start_wall_time,
                self.frontier_probe_spin_timeout,
                time.monotonic(),
            ):
                self.get_logger().warn('Frontier probe alignment spin timed out; retrying.')
                self.nav2.cancel_spin()
                self._frontier_probe_failed()
            return
        if self.state == self.PROBING_FRONTIER:
            settings = self._frontier_probe_settings()
            timeout = 0.0 if settings is None else settings[2]
            if navigation_timed_out(
                self.frontier_probe_action_start_wall_time,
                timeout,
                time.monotonic(),
            ):
                self.get_logger().warn('Frontier DriveOnHeading probe timed out; retrying.')
                self.nav2.cancel_drive_on_heading()
                self._frontier_probe_failed()
            return
        if self.state == self.SELECTING:
            if (
                self.selection_request_wall_time is not None
                and time.monotonic() - self.selection_request_wall_time > self.nav2_request_timeout
            ):
                self.get_logger().warn('Nav2 path request timed out; retrying frontier selection.')
                self._mark_goal_failed(self.selection_request_goal)
                self.nav2.cancel_path_batch()
                self.selection_request_goal = None
                if self._online_active_branch() is not None:
                    self._record_online_branch_failure('Nav2 branch path request timed out')
                self._skip_unreachable_loop_revisit('Nav2 path request timed out')
                self._schedule_retry(0.0)
            return
        if time.monotonic() < self.next_retry_wall_time or pose is None:
            return
        self._start_selection(pose[0], pose[1])

    # ------------------------------------------------------------------
    # Nav2 action orchestration
    # ------------------------------------------------------------------

    def _initialize_gbsae(self, pose: Tuple[float, float, float]):
        if self.slam_mode != 'gbsae' or self.gbsae_planner is not None:
            return
        assert self.gbsae_prior_graph is not None
        self.gbsae_planner = GBSAEPlanner(
            self.gbsae_prior_graph,
            (pose[0], pose[1]),
            self.gbsae_loop_path_cost_weight,
        )
        route = [step.vertex_id for step in self.gbsae_planner.route]
        self.get_logger().info(
            f'Created GBSAE route from prior vertex {self.gbsae_planner.start_vertex}: '
            f'{route}; loop_edges={self.gbsae_planner.loop_edges}.'
        )

    def _start_initial_spin(self):
        self.state = self.INITIAL_SPIN
        self.get_logger().info('Nav2 is ready. Starting initial 360 degree scan.')
        self.nav2.spin_once(
            self.initial_spin_yaw,
            self.initial_spin_timeout,
            self._initial_spin_finished,
        )

    def _initial_spin_finished(self, status: int):
        if self.state != self.INITIAL_SPIN:
            return
        if status == GOAL_STATUS_SUCCEEDED:
            self.get_logger().info('Initial Nav2 spin completed.')
        else:
            self.get_logger().warn(
                f'Initial Nav2 spin ended with status={status}; continuing exploration.'
            )
        self._schedule_retry(0.0)

    def _start_selection(self, rx: float, ry: float):
        if self.slam_mode == 'gbsae' and self.gbsae_planner is not None:
            self._start_gbsae_selection(rx, ry)
            return
        if self.slam_mode == 'online_gbsae':
            if self.gbsae_planner is not None and self.online_gbsae_phase == 'topology':
                self._start_gbsae_selection(rx, ry)
                return
            self._maybe_start_online_topology_build((rx, ry), 'bootstrap coverage threshold')
            self._start_online_bootstrap_selection(rx, ry)
            return
        self._start_standard_selection(rx, ry)

    # ------------------------------------------------------------------
    # Online bootstrap and topology orchestration
    # ------------------------------------------------------------------

    def _maybe_start_online_topology_build(
        self,
        pose_xy: Tuple[float, float],
        reason: str,
    ) -> bool:
        if (
            self.slam_mode != 'online_gbsae'
            or self.latest_grid is None
            or self.online_gbsae_bounds is None
            or self.online_gbsae_topology_future is not None
        ):
            return False
        geometry = self._grid_geometry()
        ratio = known_area_ratio(self.latest_grid, geometry, self.online_gbsae_bounds)
        if ratio < self.online_gbsae_bootstrap_known_ratio:
            return False
        old_graph = None if self.gbsae_planner is None else self.gbsae_planner.graph.copy()
        old_explored = (
            set()
            if self.gbsae_planner is None or not self.online_gbsae_explored_migration_enabled
            else self.gbsae_planner.completed_vertices
        )
        self.online_gbsae_topology_generation += 1
        self.online_gbsae_future_generation = self.online_gbsae_topology_generation
        self.online_gbsae_build_pose = pose_xy
        self.online_gbsae_topology_future = self.online_gbsae_topology_executor.submit(
            build_online_topology,
            self.latest_grid.copy(),
            geometry,
            pose_xy,
            tuple(self.online_gbsae_branches),
            old_graph,
            old_explored,
            clearance=self.online_gbsae_topology_clearance,
            spur_prune_length=self.online_gbsae_spur_prune_length,
            support_vertex_spacing=self.online_gbsae_support_vertex_spacing,
            migration_radius=self.online_gbsae_migration_radius,
            migration_overlap=self.online_gbsae_migration_overlap,
        )
        self.get_logger().info(
            f'Starting online GBSAE topology build generation='
            f'{self.online_gbsae_future_generation}, reason={reason}, '
            f'known_ratio={ratio:.3f}.'
        )
        return True

    def _poll_online_topology_build(self, pose: Tuple[float, float, float]):
        future = self.online_gbsae_topology_future
        if self.slam_mode != 'online_gbsae' or future is None or not future.done():
            return
        generation = self.online_gbsae_future_generation
        self.online_gbsae_topology_future = None
        if generation != self.online_gbsae_topology_generation:
            return
        try:
            topology = future.result()
        except Exception as exc:
            self.get_logger().warn(f'Online GBSAE topology build failed: {exc}')
            self.online_gbsae_phase = 'bootstrap'
            return
        if topology.graph.number_of_nodes() < 2 or not topology.target_vertices:
            self.get_logger().warn(
                'Online GBSAE topology is not actionable yet; continuing bootstrap.'
            )
            self.gbsae_planner = None
            self.online_gbsae_phase = 'bootstrap'
            return
        planner = GBSAEPlanner(
            topology.graph,
            (pose[0], pose[1]),
            self.gbsae_loop_path_cost_weight,
            target_vertices=topology.target_vertices,
            explored_vertices=topology.inherited_explored,
        )
        if planner.active_step is None:
            self.get_logger().warn(
                'Online GBSAE rebuild produced no pending route; continuing bootstrap.'
            )
            self.gbsae_planner = None
            self.online_gbsae_phase = 'bootstrap'
            return
        self.gbsae_planner = planner
        self.online_gbsae_topology_version += 1
        self.online_gbsae_phase = 'topology'
        self.get_logger().info(
            f'Installed online GBSAE topology version={self.online_gbsae_topology_version}: '
            f'nodes={planner.graph.number_of_nodes()}, '
            f'edges={planner.graph.number_of_edges()}, '
            f'targets={len(planner.target_vertices)}, '
            f'inherited_explored={len(topology.inherited_explored)}.'
        )

    def _invalidate_online_route(self, pose_xy: Tuple[float, float], reason: str):
        if self.slam_mode != 'online_gbsae':
            return
        self.get_logger().warn(f'Online GBSAE route invalid: {reason}.')
        self._maybe_start_online_topology_build(pose_xy, reason)
        self.gbsae_planner = None
        self.online_gbsae_phase = 'bootstrap'

    def _start_online_bootstrap_selection(self, rx: float, ry: float):
        if not self.frontier_clusters:
            if (
                self.online_gbsae_topology_future is not None
                or any(
                    not branch.blocked and not branch.explored
                    for branch in self.online_gbsae_branches
                )
            ):
                self._schedule_retry()
                return
            self._start_standard_selection(rx, ry)
            return
        now = time.monotonic()
        self.failed_goals.expire(now)
        candidates = self._ranked_frontier_candidates(
            rx,
            ry,
            now,
            len(self.frontier_clusters),
        )
        self.selection_start_xy = (rx, ry)
        candidates.sort(
            key=lambda candidate: self._score_online_bootstrap_candidate(
                candidate,
                None,
            ).total,
            reverse=True,
        )
        self.selection_candidates = candidates[:self.online_gbsae_bootstrap_shortlist]
        self.selection_generation = self.nav2.start_path_batch()
        self.selection_cluster_index = 0
        self.graph_candidates = []
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        self.selection_kind = 'online_bootstrap'
        self.state = self.SELECTING
        ratio = self._online_known_ratio()
        self.get_logger().info(
            f'Online GBSAE bootstrap is checking {len(self.selection_candidates)} '
            f'frontiers, known_ratio={ratio:.3f}.'
        )
        self._request_next_online_bootstrap_path()

    def _request_next_online_bootstrap_path(self):
        if self.selection_cluster_index < len(self.selection_candidates):
            candidate = self.selection_candidates[self.selection_cluster_index]
            goal_xy = candidate.safe_goal.point
            self.selection_request_wall_time = time.monotonic()
            self.selection_request_goal = goal_xy
            self.nav2.compute_path(
                self.selection_generation,
                self.selection_start_xy,
                goal_xy,
                self._online_bootstrap_path_computed,
            )
            return
        if not self.graph_candidates:
            self.get_logger().info('Online GBSAE bootstrap found no reachable frontier.')
            self.nav2.cancel_path_batch()
            self._schedule_retry()
            return
        self.graph_candidates.sort(key=lambda item: item[0].total, reverse=True)
        score, candidate, planned_path = self.graph_candidates[0]
        self._record_online_branches(self.graph_candidates[1:], score.total)
        self.get_logger().info(
            f'Online GBSAE bootstrap selected source={candidate.cluster.source}, '
            f'score={score.total:.3f}, depth={score.normal_unknown_depth:.3f}, '
            f'directional_unknown={score.directional_remaining_unknown:.3f}, '
            f'path_known_ratio={score.path_known_ratio:.3f}.'
        )
        self._dispatch_navigation(candidate, planned_path)

    def _online_bootstrap_path_computed(self, planned_path: Optional[PlannedPath]):
        if self.state != self.SELECTING or self.selection_kind != 'online_bootstrap':
            return
        self.selection_request_wall_time = None
        candidate = self.selection_candidates[self.selection_cluster_index]
        if planned_path is None:
            self._mark_goal_failed(self.selection_request_goal)
        else:
            score = self._score_online_bootstrap_candidate(candidate, planned_path)
            self.graph_candidates.append((score, candidate, planned_path))
            self.get_logger().info(
                f'Online GBSAE bootstrap candidate source={candidate.cluster.source}, '
                f'score={score.total:.3f}, depth={score.normal_unknown_depth:.3f}, '
                f'directional_unknown={score.directional_remaining_unknown:.3f}, '
                f'path_known_ratio={score.path_known_ratio:.3f}.'
            )
        self.selection_request_goal = None
        self.selection_cluster_index += 1
        self._request_next_online_bootstrap_path()

    def _score_online_bootstrap_candidate(
        self,
        candidate: FrontierCandidate,
        planned_path: Optional[PlannedPath],
    ) -> BootstrapScore:
        geometry = self._grid_geometry()
        seed_xy = self._grid_to_world(candidate.safe_goal.seed, geometry)
        unknown_depth = normal_unknown_depth(
            self.latest_grid,
            geometry,
            seed_xy,
            candidate.safe_goal.outward_normal,
            self.online_gbsae_unknown_depth_max,
        )
        directional_unknown = 0.0
        if self.online_gbsae_directional_prior_enabled:
            directional_unknown = directional_remaining_unknown(
                self.latest_grid,
                geometry,
                self.online_gbsae_bounds,
                self.selection_start_xy,
                candidate.safe_goal.point,
                self.online_gbsae_direction_half_angle,
            )
        known_ratio = 0.0
        if planned_path is not None:
            known_ratio = path_known_ratio(
                planned_path.points,
                self.latest_grid,
                geometry,
            )
        return bootstrap_score(
            unknown_depth=unknown_depth,
            directional_unknown=directional_unknown,
            path_known_ratio=known_ratio,
            weights=self.online_gbsae_weights,
        )

    def _record_online_branches(self, alternatives, best_score: float):
        if not self.online_gbsae_branch_hypotheses_enabled:
            return
        branch_inputs = []
        for score, candidate, _ in alternatives:
            normal = candidate.safe_goal.outward_normal
            if normal is None:
                continue
            branch_inputs.append((
                (candidate.cluster.centroid_x, candidate.cluster.centroid_y),
                normal,
                score.total,
            ))
        previous_count = len(self.online_gbsae_branches)
        self.online_gbsae_branches = record_branch_hypotheses(
            self.online_gbsae_branches,
            branch_inputs,
            best_score=best_score,
            score_ratio=self.online_gbsae_branch_score_ratio,
            min_angle=self.online_gbsae_branch_min_angle,
            merge_radius=self.online_gbsae_branch_merge_radius,
            projection_distance=self.online_gbsae_branch_projection_distance,
            bounds=self.online_gbsae_bounds,
        )
        added = len(self.online_gbsae_branches) - previous_count
        if added:
            self.get_logger().info(f'Recorded {added} online GBSAE branch hypotheses.')

    def _online_known_ratio(self) -> float:
        if self.latest_grid is None or self.online_gbsae_bounds is None:
            return 0.0
        return known_area_ratio(
            self.latest_grid,
            self._grid_geometry(),
            self.online_gbsae_bounds,
        )

    def _start_standard_selection(self, rx: float, ry: float):
        if not self.frontier_clusters:
            if self._is_map_stable():
                self.state = self.COMPLETE
                self.get_logger().info(
                    f'Exploration complete. Map stable for {self.stability_duration}s.'
                )
            else:
                self._schedule_retry()
            return

        limit = (
            self.graph_max_frontier_candidates
            if self.slam_mode == 'approx_graph'
            else self.frontier_planning_attempts
        )
        self.selection_start_xy = (rx, ry)
        now = time.monotonic()
        self.failed_goals.expire(now)
        local_filter_start = time.monotonic()
        self.selection_candidates = self._ranked_frontier_candidates(rx, ry, now, limit)
        local_filter_ms = (time.monotonic() - local_filter_start) * 1000.0
        self.selection_generation = self.nav2.start_path_batch()
        self.selection_cluster_index = 0
        self.graph_candidates = []
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        self.selection_kind = 'frontier'
        self.state = self.SELECTING
        source_counts = {
            source: sum(
                candidate.cluster.source == source
                for candidate in self.selection_candidates
            )
            for source in ('unknown', OPEN_EDGE_FRONTIER)
        }
        self.get_logger().info(
            f'Shared frontier pool has {len(self.selection_candidates)} candidates: '
            f'unknown={source_counts["unknown"]}, open_edge={source_counts[OPEN_EDGE_FRONTIER]}, '
            f'local_filter_ms={local_filter_ms:.1f}.'
        )
        self._request_next_path()

    def _start_gbsae_selection(self, rx: float, ry: float):
        planner = self.gbsae_planner
        assert planner is not None
        for step in planner.advance_reached_steps((rx, ry), self.nav2_goal_reach_radius):
            self._mark_online_step_complete(step.vertex_id)
            self.get_logger().info(
                f'GBSAE vertex {step.vertex_id} already reached'
                f'{" during loop revisit" if step.loop_revisit else ""}.'
            )

        step = planner.active_step
        if step is None:
            if self.slam_mode == 'online_gbsae':
                self._invalidate_online_route((rx, ry), 'route exhausted')
                self._start_online_bootstrap_selection(rx, ry)
                return
            self.get_logger().info(
                'GBSAE prior route completed; continuing with frontier coverage.'
            )
            self._start_standard_selection(rx, ry)
            return

        self.selection_start_xy = (rx, ry)
        self.failed_goals.expire(time.monotonic())
        target = vertex_point(planner.graph, step.vertex_id)
        branch = self._online_active_branch()
        if branch is not None:
            if self.failed_goals.contains(target, time.monotonic()):
                self.get_logger().info(
                    f'Online GBSAE branch {branch.branch_id} is cooling down; '
                    f'retrying after {self.failed_goal_cooldown:.1f}s.'
                )
                self._schedule_retry(self.failed_goal_cooldown)
                return
            if (
                self.latest_grid is None
                or not self.online_gbsae_bounds.contains(target)
                or not branch_path_has_no_known_obstacle(
                    self.latest_grid,
                    self._grid_geometry(),
                    branch,
                )
            ):
                self._record_online_branch_failure(
                    f'branch {branch.branch_id} ray crosses a known obstacle or prior boundary'
                )
                self._schedule_retry(self.failed_goal_cooldown)
                return
            self._request_gbsae_vertex_path(step.vertex_id, target, branch.branch_id)
            return
        if (
            self.latest_map is not None
            and self.latest_grid is not None
            and point_is_known_free(self.latest_map, self.latest_grid, target)
            and not self.failed_goals.contains(target, time.monotonic())
        ):
            self._request_gbsae_vertex_path(step.vertex_id, target)
            return
        if step.loop_revisit:
            self._skip_unreachable_loop_revisit('prior vertex is not currently known-free')
            self._schedule_retry(0.0)
            return
        if self.slam_mode == 'online_gbsae':
            self._invalidate_online_route(
                (rx, ry),
                f'skeleton vertex {step.vertex_id} is no longer known-free',
            )
            self._start_online_bootstrap_selection(rx, ry)
            return
        self._start_gbsae_frontier_selection(rx, ry)

    def _request_gbsae_vertex_path(
        self,
        vertex_id: int,
        target: Tuple[float, float],
        branch_id: Optional[int] = None,
    ):
        step = self.gbsae_planner.active_step
        self.selection_generation = self.nav2.start_path_batch()
        self.selection_kind = 'gbsae_vertex'
        self.selection_request_wall_time = time.monotonic()
        self.selection_request_goal = target
        self.state = self.SELECTING
        label = f'online branch {branch_id}' if branch_id is not None else f'vertex {vertex_id}'
        self.get_logger().info(
            f'Checking GBSAE {label}'
            f'{" loop revisit" if step is not None and step.loop_revisit else ""} '
            f'at ({target[0]:.2f}, {target[1]:.2f}).'
        )
        self.nav2.compute_path(
            self.selection_generation,
            self.selection_start_xy,
            target,
            self._gbsae_vertex_path_computed,
        )

    def _start_gbsae_frontier_selection(self, rx: float, ry: float):
        planner = self.gbsae_planner
        assert planner is not None
        local_filter_start = time.monotonic()
        candidates = self._ranked_frontier_candidates(
            rx,
            ry,
            time.monotonic(),
            len(self.frontier_clusters),
        )
        self.selection_candidates = planner.frontiers_for_active(candidates)[
            :self.frontier_planning_attempts
        ]
        local_filter_ms = (time.monotonic() - local_filter_start) * 1000.0
        self.selection_generation = self.nav2.start_path_batch()
        self.selection_cluster_index = 0
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        self.selection_kind = 'gbsae_frontier'
        self.state = self.SELECTING
        step = planner.active_step
        assert step is not None
        self.get_logger().info(
            f'GBSAE prior vertex {step.vertex_id} has {len(self.selection_candidates)} '
            f'allocated safe frontier candidates, local_filter_ms={local_filter_ms:.1f}.'
        )
        self._request_next_gbsae_frontier_path()

    def _gbsae_vertex_path_computed(self, planned_path: Optional[PlannedPath]):
        if self.state != self.SELECTING or self.selection_kind != 'gbsae_vertex':
            return
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        planner = self.gbsae_planner
        assert planner is not None
        step = planner.active_step
        if step is None:
            self._schedule_retry(0.0)
            return
        if planned_path is not None:
            self._dispatch_gbsae_vertex_navigation(planned_path, step.vertex_id)
            return
        if self._online_active_branch() is not None:
            self._record_online_branch_failure('Nav2 could not plan the virtual branch')
            self._schedule_retry(self.failed_goal_cooldown)
            return
        if step.loop_revisit:
            self._skip_unreachable_loop_revisit('Nav2 could not plan the optional revisit')
            self._schedule_retry(0.0)
            return
        self._mark_goal_failed(vertex_point(planner.graph, step.vertex_id))
        if self.slam_mode == 'online_gbsae':
            self._invalidate_online_route(
                self.selection_start_xy,
                f'Nav2 could not plan skeleton vertex {step.vertex_id}',
            )
            self._start_online_bootstrap_selection(*self.selection_start_xy)
            return
        self._start_gbsae_frontier_selection(*self.selection_start_xy)

    def _request_next_gbsae_frontier_path(self):
        if self.selection_cluster_index < len(self.selection_candidates):
            candidate = self.selection_candidates[self.selection_cluster_index]
            goal_xy = candidate.safe_goal.point
            self.selection_request_wall_time = time.monotonic()
            self.selection_request_goal = goal_xy
            self.get_logger().info(
                f'Checking GBSAE allocated frontier source={candidate.cluster.source}, '
                f'information_gain={candidate.information_gain:.2f}m^2.'
            )
            self.nav2.compute_path(
                self.selection_generation,
                self.selection_start_xy,
                goal_xy,
                self._gbsae_frontier_path_computed,
            )
            return

        planner = self.gbsae_planner
        assert planner is not None
        step = planner.active_step
        if step is not None:
            self.get_logger().warn(
                f'GBSAE prior vertex {step.vertex_id} has no reachable allocated '
                'frontier; advancing the route.'
            )
            planner.advance_active_step()
        self.nav2.cancel_path_batch()
        self._schedule_retry()

    def _gbsae_frontier_path_computed(self, planned_path: Optional[PlannedPath]):
        if self.state != self.SELECTING or self.selection_kind != 'gbsae_frontier':
            return
        self.selection_request_wall_time = None
        candidate = self.selection_candidates[self.selection_cluster_index]
        if planned_path is not None:
            self._dispatch_navigation(candidate, planned_path)
            return
        self._mark_goal_failed(self.selection_request_goal)
        self.selection_request_goal = None
        self.selection_cluster_index += 1
        self._request_next_gbsae_frontier_path()

    def _request_next_path(self):
        if self.selection_cluster_index < len(self.selection_candidates):
            candidate = self.selection_candidates[self.selection_cluster_index]
            goal_xy = candidate.safe_goal.point
            self.selection_request_wall_time = time.monotonic()
            self.selection_request_goal = goal_xy
            self.get_logger().info(
                f'Checking shared frontier candidate source={candidate.cluster.source}, '
                f'information_gain={candidate.information_gain:.2f}m^2, '
                f'utility={candidate.utility:.3f}.'
            )
            self.nav2.compute_path(
                self.selection_generation,
                self.selection_start_xy,
                goal_xy,
                self._path_computed,
            )
            return

        if self.graph_candidates:
            self.graph_candidates.sort(key=lambda item: item[0], reverse=True)
            score, candidate, planned_path = self.graph_candidates[0]
            self.get_logger().info(
                f'Graph-based selected frontier score={score:.3f}, '
                f'information_gain={candidate.information_gain:.2f}m^2, '
                f'source={candidate.cluster.source}'
            )
            self._dispatch_navigation(candidate, planned_path)
            return

        self.get_logger().info(
            f'No reachable frontier among {len(self.frontier_clusters)} clusters. '
            f'Retrying in {self.frontier_retry_interval:.1f}s.'
        )
        self.nav2.cancel_path_batch()
        self._schedule_retry()

    def _path_computed(self, planned_path: Optional[PlannedPath]):
        if self.state != self.SELECTING:
            return
        self.selection_request_wall_time = None
        candidate = self.selection_candidates[self.selection_cluster_index]
        if planned_path is None:
            self._mark_goal_failed(self.selection_request_goal)
        elif self.slam_mode != 'approx_graph':
            self._dispatch_navigation(candidate, planned_path)
            return
        else:
            score = self.graph_scorer.score(
                self.pose_graph_tracker.graph,
                self.latest_map,
                planned_path.points,
                self.latest_grid,
            )
            if np.isfinite(score):
                self.graph_candidates.append((score, candidate, planned_path))
        self.selection_request_goal = None
        self.selection_cluster_index += 1
        self._request_next_path()

    def _dispatch_navigation(
        self,
        candidate: FrontierCandidate,
        planned_path: PlannedPath,
    ):
        self.nav2.cancel_path_batch()
        cluster = candidate.cluster
        self.current_goal = planned_path.goal_xy
        self.current_safe_goal = candidate.safe_goal
        self.target_cluster = cluster
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        self.current_navigation_kind = (
            'online_bootstrap_frontier'
            if self.slam_mode == 'online_gbsae' and self.selection_kind == 'online_bootstrap'
            else 'frontier'
        )
        self.state = self.NAVIGATING
        self.navigation_start_wall_time = time.monotonic()
        yaw = heading_to_target(planned_path.goal_xy, (cluster.centroid_x, cluster.centroid_y))
        self.get_logger().info(
            f'Navigating to frontier goal=({planned_path.goal_xy[0]:.2f}, '
            f'{planned_path.goal_xy[1]:.2f}), size={cluster.size}, '
            f'cost={planned_path.cost:.2f}, source={cluster.source}, '
            f'information_gain={candidate.information_gain:.2f}m^2'
        )
        self.nav2.navigate(planned_path.goal_xy, yaw, self._navigation_finished)

    def _dispatch_gbsae_vertex_navigation(self, planned_path: PlannedPath, vertex_id: int):
        self.nav2.cancel_path_batch()
        self.current_goal = planned_path.goal_xy
        self.current_safe_goal = None
        self.target_cluster = None
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        branch = self._online_active_branch()
        self.current_navigation_kind = (
            'online_gbsae_branch' if branch is not None else 'gbsae_vertex'
        )
        self.current_navigation_vertex_id = vertex_id
        self.state = self.NAVIGATING
        self.navigation_start_wall_time = time.monotonic()
        planner = self.gbsae_planner
        assert planner is not None
        step = planner.active_step
        revisit = step is not None and step.loop_revisit
        self.get_logger().info(
            f'Navigating directly to GBSAE '
            f'{"virtual branch" if branch is not None else "vertex"} {vertex_id}'
            f'{" loop revisit" if revisit else ""}: '
            f'goal=({planned_path.goal_xy[0]:.2f}, {planned_path.goal_xy[1]:.2f}), '
            f'cost={planned_path.cost:.2f}.'
        )
        self.nav2.navigate(planned_path.goal_xy, 0.0, self._navigation_finished)

    def _navigation_finished(self, status: int):
        if self.state != self.NAVIGATING:
            return
        if self.current_navigation_kind in ('gbsae_vertex', 'online_gbsae_branch'):
            if status == GOAL_STATUS_SUCCEEDED:
                planner = self.gbsae_planner
                assert planner is not None
                step = planner.advance_active_step()
                assert step is not None
                self._mark_online_step_complete(step.vertex_id)
                self.get_logger().info(
                    f'Nav2 reached GBSAE vertex {step.vertex_id}'
                    f'{" during loop revisit" if step.loop_revisit else ""}.'
                )
            elif self.current_navigation_kind == 'online_gbsae_branch':
                self._record_online_branch_failure(
                    f'Nav2 virtual branch navigation ended with status={status}'
                )
            else:
                self._handle_navigation_failure(
                    f'Nav2 GBSAE navigation ended with status={status}'
                )
            self._clear_navigation()
            self._schedule_retry(0.0)
            return
        if status == GOAL_STATUS_SUCCEEDED:
            self.get_logger().info('Nav2 reached the active frontier goal.')
            if self._frontier_probe_settings() is not None:
                self._start_frontier_probe()
                return
        else:
            self.get_logger().warn(f'Nav2 navigation ended with status={status}; retrying.')
            self._mark_goal_failed(self.current_goal)
        self._clear_navigation()
        self._schedule_retry(0.0)

    def _start_frontier_probe(self):
        if (
            self.latest_map is None
            or self.target_cluster is None
            or self.current_safe_goal is None
        ):
            self._frontier_probe_failed()
            return

        normal = self.current_safe_goal.outward_normal
        pose = self._get_robot_pose()
        if normal is None or pose is None:
            self.get_logger().warn('Cannot estimate frontier probe direction; retrying.')
            self._frontier_probe_failed()
            return

        self.frontier_probe_normal = normal
        target_yaw = math.atan2(normal[1], normal[0])
        yaw_delta = normalize_angle(target_yaw - pose[2])
        if abs(yaw_delta) <= self.frontier_probe_spin_tolerance:
            self._start_frontier_drive()
            return

        self.state = self.ALIGNING_FRONTIER_PROBE
        self.frontier_probe_action_start_wall_time = time.monotonic()
        self.get_logger().info(
            f'Aligning with {self.target_cluster.source} frontier normal using '
            f'Nav2 Spin: delta={yaw_delta:.2f} rad.'
        )
        self.nav2.spin_once(
            yaw_delta,
            self.frontier_probe_spin_timeout,
            self._frontier_probe_spin_finished,
        )

    def _frontier_probe_spin_finished(self, status: int):
        if self.state != self.ALIGNING_FRONTIER_PROBE:
            return
        if status != GOAL_STATUS_SUCCEEDED:
            self.get_logger().warn(
                f'Frontier probe alignment spin ended with status={status}; retrying.'
            )
            self._frontier_probe_failed()
            return
        self._start_frontier_drive()

    def _start_frontier_drive(self):
        settings = self._frontier_probe_settings()
        if settings is None:
            self._frontier_probe_failed()
            return
        distance, speed, timeout = settings
        self.state = self.PROBING_FRONTIER
        self.frontier_probe_action_start_wall_time = time.monotonic()
        self.get_logger().info(
            f'Probing beyond {self.target_cluster.source} frontier with '
            f'Nav2 DriveOnHeading: distance={distance:.2f}m, speed={speed:.2f}m/s.'
        )
        self.nav2.drive_on_heading(
            distance,
            speed,
            timeout,
            self._frontier_probe_drive_finished,
        )

    def _frontier_probe_drive_finished(self, status: int):
        if self.state != self.PROBING_FRONTIER:
            return
        if status != GOAL_STATUS_SUCCEEDED:
            self.get_logger().warn(f'Frontier probe ended with status={status}; retrying.')
            self._frontier_probe_failed()
            return
        self.get_logger().info('Frontier Nav2 probe completed.')
        self._clear_navigation()
        self._schedule_retry(0.0)

    def _frontier_probe_settings(self) -> Optional[Tuple[float, float, float]]:
        if self.target_cluster is None:
            return None
        if (
            self.current_navigation_kind == 'online_bootstrap_frontier'
            and self.online_gbsae_bootstrap_probe_enabled
        ):
            return (
                self.online_gbsae_bootstrap_probe_distance,
                self.online_gbsae_bootstrap_probe_speed,
                self.online_gbsae_bootstrap_probe_timeout,
            )
        if (
            self.target_cluster.source == OPEN_EDGE_FRONTIER
            and self.frontier_open_edge_probe_enabled
        ):
            return (
                self.frontier_open_edge_probe_distance,
                self.frontier_open_edge_probe_speed,
                self.frontier_open_edge_probe_timeout,
            )
        if (
            self.target_cluster.source == UNKNOWN_FRONTIER
            and self.frontier_unknown_probe_enabled
        ):
            return (
                self.frontier_unknown_probe_distance,
                self.frontier_unknown_probe_speed,
                self.frontier_unknown_probe_timeout,
            )
        return None

    def _frontier_probe_failed(self):
        self._mark_goal_failed(self.current_goal)
        self._clear_navigation()
        self._schedule_retry(0.0)

    def _clear_navigation(self):
        self.current_goal = None
        self.current_safe_goal = None
        self.target_cluster = None
        self.navigation_start_wall_time = None
        self.frontier_probe_normal = None
        self.frontier_probe_action_start_wall_time = None
        self.current_navigation_kind = 'frontier'
        self.current_navigation_vertex_id = None

    def _handle_navigation_failure(self, reason: str):
        if self.current_navigation_kind == 'online_gbsae_branch':
            self._record_online_branch_failure(reason)
            return
        if self._skip_unreachable_loop_revisit(reason):
            return
        if self.slam_mode == 'online_gbsae' and self.current_navigation_kind == 'gbsae_vertex':
            pose = self._get_robot_pose()
            self._invalidate_online_route(
                self.selection_start_xy if pose is None else (pose[0], pose[1]),
                reason,
            )
            return
        self.get_logger().warn(f'{reason}; retrying.')
        self._mark_goal_failed(self.current_goal)

    def _skip_unreachable_loop_revisit(self, reason: str) -> bool:
        if self.slam_mode not in ('gbsae', 'online_gbsae') or self.gbsae_planner is None:
            return False
        step = self.gbsae_planner.active_step
        if step is None or not step.loop_revisit:
            return False
        skipped = self.gbsae_planner.skip_active_loop_revisit()
        self.get_logger().warn(
            f'Skipping optional GBSAE loop revisit to prior vertex '
            f'{skipped.vertex_id}: {reason}.'
        )
        return True

    def _online_active_branch(self) -> Optional[BranchHypothesis]:
        if self.slam_mode != 'online_gbsae' or self.gbsae_planner is None:
            return None
        step = self.gbsae_planner.active_step
        if step is None:
            return None
        attributes = self.gbsae_planner.graph.nodes[step.vertex_id]
        branch_id = attributes.get('branch_id')
        if attributes.get('kind') != 'branch' or branch_id is None:
            return None
        return next(
            (branch for branch in self.online_gbsae_branches if branch.branch_id == branch_id),
            None,
        )

    def _record_online_branch_failure(self, reason: str):
        branch = self._online_active_branch()
        if branch is None:
            return
        self._mark_goal_failed(branch.point)
        self.online_gbsae_branches = update_branch_failure(
            self.online_gbsae_branches,
            branch.branch_id,
            self.online_gbsae_branch_failure_limit,
        )
        updated = next(
            item for item in self.online_gbsae_branches if item.branch_id == branch.branch_id
        )
        if self.gbsae_planner is not None:
            for _, attributes in self.gbsae_planner.graph.nodes(data=True):
                if attributes.get('branch_id') == branch.branch_id:
                    attributes['failures'] = updated.failures
                    attributes['blocked'] = updated.blocked
        self.get_logger().warn(
            f'Online GBSAE branch {branch.branch_id} failed '
            f'({updated.failures}/{self.online_gbsae_branch_failure_limit}): {reason}.'
        )
        if updated.blocked:
            pose = self._get_robot_pose()
            self._invalidate_online_route(
                self.selection_start_xy if pose is None else (pose[0], pose[1]),
                f'branch {branch.branch_id} reached bounded retry limit',
            )

    def _mark_online_step_complete(self, vertex_id: int):
        if self.slam_mode != 'online_gbsae' or self.gbsae_planner is None:
            return
        attributes = self.gbsae_planner.graph.nodes[vertex_id]
        branch_id = attributes.get('branch_id')
        if attributes.get('kind') != 'branch' or branch_id is None:
            return
        self.online_gbsae_branches = mark_branch_explored(
            self.online_gbsae_branches,
            branch_id,
        )
        attributes['explored'] = True

    def _schedule_retry(self, delay: Optional[float] = None):
        self.state = self.IDLE
        self.next_retry_wall_time = time.monotonic() + (
            self.frontier_retry_interval if delay is None else delay
        )

    # ------------------------------------------------------------------
    # Frontier candidate generation
    # ------------------------------------------------------------------

    def _ranked_frontier_candidates(
        self,
        rx: float,
        ry: float,
        now: float,
        limit: int,
    ) -> List[FrontierCandidate]:
        if self.latest_grid is None:
            return []
        info = self.latest_map.info
        data = self.latest_grid
        geometry = GridGeometry(
            origin_x=info.origin.position.x,
            origin_y=info.origin.position.y,
            resolution=info.resolution,
            width=info.width,
            height=info.height,
        )
        search_config = SafeGoalSearchConfig(
            search_radius=self.frontier_goal_search_radius,
            clearance=self.frontier_goal_clearance,
            standoff=self.frontier_goal_standoff,
            map_edge_clearance=self.frontier_goal_map_edge_clearance,
            reach_radius=self.nav2_goal_reach_radius,
            point_sample_limit=self.frontier_goal_point_sample_limit,
        )
        prepared_grid = prepare_safe_goal_grid(data, geometry, search_config)
        candidates = [
            candidate
            for cluster in self.frontier_clusters
            if (
                candidate := self._candidate_for_cluster(
                    cluster,
                    rx,
                    ry,
                    now,
                    data,
                    geometry,
                    search_config,
                    prepared_grid,
                )
            ) is not None
        ]
        return ranked_frontier_candidates(candidates, limit)

    def _candidate_for_cluster(
        self,
        cluster: FrontierCluster,
        rx: float,
        ry: float,
        now: float,
        data: np.ndarray,
        geometry: GridGeometry,
        search_config: SafeGoalSearchConfig,
        prepared_grid: PreparedSafeGoalGrid,
    ) -> Optional[FrontierCandidate]:
        goal = select_safe_frontier_goal(
            data,
            geometry,
            cluster.cells,
            (rx, ry),
            search_config,
            prepared_grid,
        )
        if goal is None:
            return None
        goal = replace(
            goal,
            outward_normal=self._frontier_outward_normal(
                data,
                geometry,
                cluster,
                goal.seed,
            ),
        )
        information_gain = potential_unknown_area(
            data,
            geometry,
            goal.seed,
            self.frontier_information_gain_radius,
            include_outside_map=cluster.source == OPEN_EDGE_FRONTIER,
        )
        return make_frontier_candidate(
            cluster,
            goal,
            information_gain,
            rx,
            ry,
            on_cooldown=self.failed_goals.contains(goal.point, now),
        )

    def _frontier_outward_normal(
        self,
        data: np.ndarray,
        geometry: GridGeometry,
        cluster: FrontierCluster,
        seed: Tuple[int, int],
    ) -> Optional[Tuple[float, float]]:
        if cluster.source == OPEN_EDGE_FRONTIER:
            return open_edge_outward_normal(
                cluster.cells,
                seed,
                geometry,
                self.frontier_probe_normal_radius,
            )
        return unknown_frontier_outward_normal(
            data,
            cluster.cells,
            seed,
            geometry,
            self.frontier_probe_normal_radius,
        )

    def _mark_goal_failed(self, goal: Optional[Tuple[float, float]]):
        if goal is not None:
            self.failed_goals.mark(goal, time.monotonic())

    # ------------------------------------------------------------------
    # Map stability and visualization
    # ------------------------------------------------------------------

    def _update_explored_history(self, data: np.ndarray):
        now = self.get_clock().now()
        known = int(np.count_nonzero(data != -1))
        self.explored_history.append((now, known))
        stability_window = Duration(seconds=self.stability_duration)
        if now.nanoseconds < stability_window.nanoseconds:
            return
        cutoff = now - stability_window
        while self.explored_history and self.explored_history[0][0] < cutoff:
            self.explored_history.popleft()

    def _is_map_stable(self) -> bool:
        if len(self.explored_history) < 2:
            return False
        earliest_time = self.explored_history[0][0]
        latest_time = self.explored_history[-1][0]
        duration = (latest_time - earliest_time).nanoseconds / 1e9
        if duration < self.stability_duration * 0.8:
            return False
        earliest_known = self.explored_history[0][1]
        latest_known = self.explored_history[-1][1]
        if earliest_known == 0:
            return False
        return abs(latest_known - earliest_known) / earliest_known < self.stability_threshold

    def _publish_visualizations(self):
        if self.latest_map is None:
            return
        self._publish_frontier_markers()
        self._publish_goal_marker()
        if self.slam_mode == 'approx_graph':
            markers = graph_to_marker_array(
                self.pose_graph_tracker.graph,
                'map',
                self.get_clock().now().to_msg(),
            )
            self.pose_graph_pub.publish(markers)
        elif self.slam_mode in ('gbsae', 'online_gbsae') and self.gbsae_planner is not None:
            markers = gbsae_to_marker_array(
                self.gbsae_planner,
                'map',
                self.get_clock().now().to_msg(),
                self.online_gbsae_bounds if self.slam_mode == 'online_gbsae' else None,
            )
            self.pose_graph_pub.publish(markers)
        elif self.slam_mode == 'online_gbsae' and self.online_gbsae_bounds is not None:
            markers = online_bounds_marker_array(
                self.online_gbsae_bounds,
                'map',
                self.get_clock().now().to_msg(),
            )
            self.pose_graph_pub.publish(markers)

    def _publish_frontier_markers(self):
        marker_array = MarkerArray()
        delete = Marker()
        delete.action = Marker.DELETEALL
        marker_array.markers.append(delete)

        for index, cluster in enumerate(self.frontier_clusters):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'frontiers'
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = cluster.centroid_x
            marker.pose.position.y = cluster.centroid_y
            marker.pose.position.z = 0.1
            scale = min(1.2, max(0.15, cluster.size * 0.002))
            marker.scale.x = marker.scale.y = marker.scale.z = scale
            marker.color.a = 0.6
            marker.color.g = 1.0
            marker.color.b = 1.0 if cluster.source == OPEN_EDGE_FRONTIER else 0.3
            marker_array.markers.append(marker)

        self.frontier_pub.publish(marker_array)

    def _publish_goal_marker(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'goal'
        marker.id = 0
        marker.type = Marker.SPHERE

        if self.current_goal is not None:
            marker.action = Marker.ADD
            marker.pose.position.x = self.current_goal[0]
            marker.pose.position.y = self.current_goal[1]
            marker.pose.position.z = 0.2
            marker.scale.x = marker.scale.y = marker.scale.z = 0.25
            marker.color.a = 1.0
            marker.color.r = 1.0
        else:
            marker.action = Marker.DELETE

        self.goal_pub.publish(marker)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_footprint',
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5),
            )
            return (
                transform.transform.translation.x,
                transform.transform.translation.y,
                _yaw_from_quaternion(transform.transform.rotation),
            )
        except Exception:
            return None

    def _grid_geometry(self) -> GridGeometry:
        info = self.latest_map.info
        return GridGeometry(
            origin_x=info.origin.position.x,
            origin_y=info.origin.position.y,
            resolution=info.resolution,
            width=info.width,
            height=info.height,
        )

    @staticmethod
    def _grid_to_world(cell: Tuple[int, int], geometry: GridGeometry):
        return (
            geometry.origin_x + (cell[1] + 0.5) * geometry.resolution,
            geometry.origin_y + (cell[0] + 0.5) * geometry.resolution,
        )

    def _log_waiting_for_nav2(self):
        now = time.monotonic()
        if now - self.last_wait_log_wall_time > 5.0:
            self.get_logger().info('Waiting for Nav2 action servers.')
            self.last_wait_log_wall_time = now

    def destroy_node(self):
        self.nav2.destroy()
        self.online_gbsae_topology_executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationCoordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
