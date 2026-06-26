import math
import random
import time
from collections import deque
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
    frontier_range_gains,
    frontier_probes_enabled_for_mode,
    make_frontier_candidate,
    ranked_frontier_candidates,
    ranked_local_cleanup_candidates,
    ranked_range_gain_candidates,
)
from .graph_exploration import (
    ApproximatePoseGraphTracker,
    GraphBasedFrontierScorer,
    best_graph_candidate,
    graph_to_marker_array,
    make_information_matrix,
)
from .gbsae_exploration import (
    GBSAEPlanner,
    gbsae_to_marker_array,
    load_prior_graph,
    point_is_known_free,
    resolve_prior_graph_path,
    vertex_point,
)
from .gvd_exploration import (
    GVDGoal,
    GVDGuidePlanStep,
    GVDGuidePlanner,
    GVDTopology,
    GVDWeights,
    HierarchicalGVDTarget,
    HierarchicalGVDTracker,
    TopologyConnectionCache,
    TrajectorySweepTracker,
    _graph_node_point,
    astar_path,
    build_obstacle_gvd_topology,
    build_obstacle_traversability,
    cluster_touches_mask,
    grid_to_world,
    gvd_guide_edge_blocked_run,
    gvd_to_marker_array,
    load_world_bounds,
    local_free_flood_mask,
    local_region_known_ratio,
    off_graph_new_free_area,
    path_crosses_new_obstacle,
    path_suffix_from_nearest,
    progress_watchdog_expired,
    rank_gvd_goals,
    rectangle_mask_outline,
    resolve_gvd_bounds_path,
    robot_component_graph,
    route_replan_due,
    sample_random_recovery_motion,
    update_translation_progress,
    world_to_grid,
)
from .nav2_backend import (
    GOAL_STATUS_SUCCEEDED,
    Nav2Backend,
    PlannedPath,
    heading_to_target,
)


def _yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class ExplorationCoordinator(Node):
    """Select exploration goals while Nav2 owns planning, control, and recovery."""

    WAITING_FOR_NAV2 = 'waiting_for_nav2'
    IDLE = 'idle'
    SELECTING = 'selecting'
    NAVIGATING = 'navigating'
    ALIGNING_FRONTIER_PROBE = 'aligning_frontier_probe'
    PROBING_FRONTIER = 'probing_frontier'
    RANDOM_RECOVERY_SPIN = 'random_recovery_spin'
    RANDOM_RECOVERY_DRIVE = 'random_recovery_drive'
    COMPLETE = 'complete'

    def __init__(self):
        super().__init__('exploration_coordinator')

        # --- Parameters ---
        self.stability_duration = self.declare_parameter('stability_duration', 10.0).value
        self.stability_threshold = self.declare_parameter('stability_threshold', 0.02).value
        self.min_frontier_size = self.declare_parameter('min_frontier_size', 10).value
        self.frontier_include_open_map_edges = self.declare_parameter(
            'frontier_include_open_map_edges', True
        ).value
        self.frontier_low_confidence_fill_enabled = self.declare_parameter(
            'frontier_low_confidence_fill_enabled', True
        ).value
        self.frontier_low_confidence_fill_max_unknown_cells = int(
            self.declare_parameter(
                'frontier_low_confidence_fill_max_unknown_cells',
                64,
            ).value
        )
        self.frontier_low_confidence_free_value = int(
            self.declare_parameter('frontier_low_confidence_free_value', 25).value
        )
        self.frontier_low_confidence_occupied_value = int(
            self.declare_parameter('frontier_low_confidence_occupied_value', 75).value
        )
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
        self.frontier_nav2_segment_distance = self.declare_parameter(
            'frontier_nav2_segment_distance', 6.0
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
        self.nav2_goal_timeout = self.declare_parameter(
            'nav2_goal_timeout', 60.0
        ).value
        self.frontier_mode_probes_enabled = self.declare_parameter(
            'frontier_mode_probes_enabled', True
        ).value
        self.gvd_mode_probes_enabled = self.declare_parameter(
            'gvd_mode_probes_enabled', False
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
        self.gvd_guide_expansion_distance = self.declare_parameter(
            'gvd_guide_expansion_distance', 0.50
        ).value
        self.gvd_guide_expansion_area_threshold = self.declare_parameter(
            'gvd_guide_expansion_area_threshold', 3.0
        ).value
        self.gvd_guide_edge_blocked_run_threshold = self.declare_parameter(
            'gvd_guide_edge_blocked_run_threshold', 0.28
        ).value
        self.gvd_guide_loop_path_cost_weight = self.declare_parameter(
            'gvd_guide_loop_path_cost_weight', 0.01
        ).value
        self.gvd_guide_frontier_detour_weight = self.declare_parameter(
            'gvd_guide_frontier_detour_weight', 0.5
        ).value
        self.gvd_guide_frontier_detour_max_extra_distance = self.declare_parameter(
            'gvd_guide_frontier_detour_max_extra_distance', 4.0
        ).value
        self.gvd_guide_frontier_detour_min_gain = self.declare_parameter(
            'gvd_guide_frontier_detour_min_gain', 0.5
        ).value
        self.gvd_guide_rebuild_cooldown = self.declare_parameter(
            'gvd_guide_rebuild_cooldown', 5.0
        ).value
        self.gvd_guide_warmup_distance = self.declare_parameter(
            'gvd_guide_warmup_distance', 20.0
        ).value
        self.gvd_guide_explored_migration_radius = self.declare_parameter(
            'gvd_guide_explored_migration_radius', 1.25
        ).value
        self.gvd_guide_waypoint_sample_count = int(
            self.declare_parameter('gvd_guide_waypoint_sample_count', 8).value
        )
        self.gvd_guide_waypoint_sample_sigma = self.declare_parameter(
            'gvd_guide_waypoint_sample_sigma', 0.30
        ).value
        self.gvd_guide_waypoint_sample_max_radius = self.declare_parameter(
            'gvd_guide_waypoint_sample_max_radius', 0.75
        ).value
        self.gvd_sweep_switch_ratio = self.declare_parameter(
            'gvd_sweep_switch_ratio', 0.50
        ).value
        self.gvd_raster_resolution = self.declare_parameter(
            'gvd_raster_resolution', 0.15
        ).value
        self.gvd_sweep_radius = self.declare_parameter('gvd_sweep_radius', 2.0).value
        self.gvd_overlap_radius = self.declare_parameter('gvd_overlap_radius', 0.35).value
        self.gvd_obstacle_clearance = self.declare_parameter(
            'gvd_obstacle_clearance', 0.14
        ).value
        self.gvd_boundary_margin = self.declare_parameter('gvd_boundary_margin', 0.20).value
        self.gvd_support_vertex_spacing = self.declare_parameter(
            'gvd_support_vertex_spacing', 2.0
        ).value
        self.gvd_corner_turn_threshold = math.radians(
            self.declare_parameter('gvd_corner_turn_threshold_deg', 45.0).value
        )
        self.gvd_min_vertex_spacing = self.declare_parameter(
            'gvd_min_vertex_spacing', 1.0
        ).value
        self.gvd_min_goal_distance = self.declare_parameter(
            'gvd_min_goal_distance', 1.0
        ).value
        self.gvd_max_goal_distance = self.declare_parameter(
            'gvd_max_goal_distance', 6.0
        ).value
        self.gvd_candidate_limit = int(self.declare_parameter('gvd_candidate_limit', 24).value)
        self.gvd_nav2_planning_attempts = int(
            self.declare_parameter('gvd_nav2_planning_attempts', 8).value
        )
        self.gvd_skeleton_cost = self.declare_parameter('gvd_skeleton_cost', 1.0).value
        self.gvd_off_skeleton_cost = self.declare_parameter(
            'gvd_off_skeleton_cost', 2.5
        ).value
        self.gvd_weights = GVDWeights(
            boundary_unknown=self.declare_parameter('gvd_boundary_unknown_weight', 2.0).value,
            goal_distance=self.declare_parameter('gvd_goal_distance_weight', 1.0).value,
            path_overlap_penalty=self.declare_parameter(
                'gvd_path_overlap_penalty_weight', 2.0
            ).value,
            straightness=self.declare_parameter('gvd_straightness_weight', 1.0).value,
        )
        self.gvd_centerline_distance_weight = self.declare_parameter(
            'gvd_centerline_distance_weight', 5.0
        ).value
        self.gvd_switching_connections_enabled = self.declare_parameter(
            'gvd_switching_connections_enabled', True
        ).value
        self.gvd_connection_neighbor_limit = int(
            self.declare_parameter('gvd_connection_neighbor_limit', 10).value
        )
        self.gvd_connection_cache_size = int(
            self.declare_parameter('gvd_connection_cache_size', 4096).value
        )
        self.gvd_reconnection_clearance = self.declare_parameter(
            'gvd_reconnection_clearance', 0.04
        ).value
        self.gvd_unknown_cycle_suppression_enabled = self.declare_parameter(
            'gvd_unknown_cycle_suppression_enabled', True
        ).value
        self.gvd_unconfident_unknown_radius = self.declare_parameter(
            'gvd_unconfident_unknown_radius', 1.0
        ).value
        self.gvd_unconfident_unknown_ratio = self.declare_parameter(
            'gvd_unconfident_unknown_ratio', 0.5
        ).value
        self.gvd_hierarchical_min_vertex_spacing = self.declare_parameter(
            'gvd_hierarchical_min_vertex_spacing', 2.0
        ).value
        self.gvd_hierarchical_state_migration_radius = self.declare_parameter(
            'gvd_hierarchical_state_migration_radius', 0.75
        ).value
        self.gvd_hierarchical_local_half_extent = self.declare_parameter(
            'gvd_hierarchical_local_half_extent', 2.5
        ).value
        self.gvd_hierarchical_region_area_weight = self.declare_parameter(
            'gvd_hierarchical_region_area_weight', 1.0
        ).value
        self.gvd_hierarchical_region_squareness_weight = self.declare_parameter(
            'gvd_hierarchical_region_squareness_weight', 1.0
        ).value
        self.gvd_hierarchical_local_approx_graph_enabled = self.declare_parameter(
            'gvd_hierarchical_local_approx_graph_enabled', True
        ).value
        self.gvd_hierarchical_local_probes_enabled = self.declare_parameter(
            'gvd_hierarchical_local_probes_enabled', True
        ).value
        self.gvd_hierarchical_local_clear_progress_threshold = max(
            0.0,
            min(
                1.0,
                float(
                    self.declare_parameter(
                        'gvd_hierarchical_local_clear_progress_threshold',
                        0.90,
                    ).value
                ),
            ),
        )
        self.gvd_hierarchical_route_replan_interval = self.declare_parameter(
            'gvd_hierarchical_route_replan_interval', 0.5
        ).value
        self.gvd_stuck_recovery_enabled = self.declare_parameter(
            'gvd_stuck_recovery_enabled', True
        ).value
        self.gvd_stuck_min_progress_distance = self.declare_parameter(
            'gvd_stuck_min_progress_distance', 0.15
        ).value
        self.gvd_stuck_timeout = self.declare_parameter('gvd_stuck_timeout', 10.0).value
        self.gvd_random_recovery_attempts = int(
            self.declare_parameter('gvd_random_recovery_attempts', 3).value
        )
        self.gvd_random_recovery_min_abs_yaw = self.declare_parameter(
            'gvd_random_recovery_min_abs_yaw', 0.6
        ).value
        self.gvd_random_recovery_max_abs_yaw = self.declare_parameter(
            'gvd_random_recovery_max_abs_yaw', 2.4
        ).value
        self.gvd_random_recovery_spin_timeout = self.declare_parameter(
            'gvd_random_recovery_spin_timeout', 5.0
        ).value
        self.gvd_random_recovery_distance = self.declare_parameter(
            'gvd_random_recovery_distance', 0.45
        ).value
        self.gvd_random_recovery_speed = self.declare_parameter(
            'gvd_random_recovery_speed', 0.10
        ).value
        self.gvd_random_recovery_drive_timeout = self.declare_parameter(
            'gvd_random_recovery_drive_timeout', 6.0
        ).value

        if self.slam_mode not in (
            'frontier',
            'approx_graph',
            'gbsae',
            'gvd_gbsae',
            'gvd_hierarchical',
            'gvd_guide',
        ):
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
            low_confidence_fill_enabled=self.frontier_low_confidence_fill_enabled,
            low_confidence_fill_max_unknown_cells=(
                self.frontier_low_confidence_fill_max_unknown_cells
            ),
            low_confidence_free_value=self.frontier_low_confidence_free_value,
            low_confidence_occupied_value=self.frontier_low_confidence_occupied_value,
        )
        self.nav2 = Nav2Backend(self)

        self.graph_odom_information = make_information_matrix(
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
            odom_information=self.graph_odom_information,
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
            odom_information=self.graph_odom_information,
        )
        self.gbsae_prior_graph = None
        self.gbsae_planner: Optional[GBSAEPlanner] = None
        self.gvd_bounds = None
        self.gvd_sweep: Optional[TrajectorySweepTracker] = None
        if self.slam_mode == 'gbsae':
            prior_graph_path = resolve_prior_graph_path(self.world_name)
            self.gbsae_prior_graph = load_prior_graph(prior_graph_path, self.world_name)
            self.get_logger().info(
                f'Loaded GBSAE prior graph for world={self.world_name}: '
                f'{self.gbsae_prior_graph.number_of_nodes()} nodes, '
                f'{self.gbsae_prior_graph.number_of_edges()} edges from {prior_graph_path}.'
            )
        elif self.slam_mode in ('gvd_gbsae', 'gvd_hierarchical', 'gvd_guide'):
            self.gvd_bounds = load_world_bounds(resolve_gvd_bounds_path(), self.world_name)
            self.gvd_sweep = TrajectorySweepTracker(
                self.gvd_bounds,
                self.gvd_raster_resolution,
                self.gvd_sweep_radius,
                self.gvd_overlap_radius,
            )
            self.get_logger().info(
                f'Loaded coarse GVD bounds for world={self.world_name}: '
                f'[{self.gvd_bounds.min_x:.1f}, {self.gvd_bounds.max_x:.1f}] x '
                f'[{self.gvd_bounds.min_y:.1f}, {self.gvd_bounds.max_y:.1f}].'
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
        self.selection_request_final_goal: Optional[Tuple[float, float]] = None
        self.selection_request_is_frontier_staging = False
        self.selection_kind = 'frontier'
        self.current_navigation_kind = 'frontier'
        self.current_navigation_path_cost = 0.0
        self.frontier_probe_parent_kind = 'frontier'
        self.frontier_probe_drive_distance = 0.0
        self.frontier_staging_final_goal: Optional[Tuple[float, float]] = None
        self.navigation_start_wall_time: Optional[float] = None
        self.frontier_probe_action_start_wall_time: Optional[float] = None
        self.failed_goals = FailedGoalCooldown(
            self.failed_goal_cooldown,
            self.failed_goal_radius,
        )
        self.last_wait_log_wall_time = 0.0
        self.gvd_phase = 'macro' if self.slam_mode == 'gvd_hierarchical' else 'bootstrap'
        self.gvd_topology: Optional[GVDTopology] = None
        self.gvd_connection_cache = TopologyConnectionCache(
            self.gvd_connection_cache_size
        )
        self.gvd_hierarchical_tracker = HierarchicalGVDTracker(
            self.gvd_hierarchical_state_migration_radius,
        )
        self.gvd_hierarchical_target: Optional[HierarchicalGVDTarget] = None
        self.gvd_hierarchical_last_route_replan_wall_time = -math.inf
        self.gvd_hierarchical_local_mask: Optional[np.ndarray] = None
        self.gvd_hierarchical_local_geometry: Optional[GridGeometry] = None
        self.gvd_hierarchical_cleared_region_outlines: List[
            Tuple[Tuple[float, float], ...]
        ] = []
        self.gvd_guide_planner: Optional[GVDGuidePlanner] = None
        self.gvd_guide_active_step: Optional[GVDGuidePlanStep] = None
        self.gvd_guide_pending_rebuild_reason: Optional[str] = None
        self.gvd_guide_rebuild_start_hint_xy: Optional[Tuple[float, float]] = None
        self.gvd_guide_last_rebuild_wall_time = -math.inf
        self.gvd_guide_last_expansion_area = 0.0
        self.gvd_guide_explored_points: List[Tuple[float, float]] = []
        self.gvd_guide_warmup_travel = 0.0
        self.gvd_guide_warmup_started = False
        self.gvd_guide_warmup_complete_logged = False
        self.gvd_hierarchical_local_graph_tracker: Optional[
            ApproximatePoseGraphTracker
        ] = None
        self.gvd_hierarchical_local_graph_candidates = []
        self.gvd_candidates: List[GVDGoal] = []
        self.gvd_candidate_index = 0
        self.gvd_active_path: Tuple[Tuple[float, float], ...] = ()
        self.gvd_active_traversability: Optional[np.ndarray] = None
        self.gvd_map_generation = 0
        self.gvd_checked_map_generation = -1
        self.gvd_progress_anchor_xy: Optional[Tuple[float, float]] = None
        self.gvd_last_progress_wall_time: Optional[float] = None
        self.gvd_random_recovery_attempt = 0
        self.gvd_random_recovery_action_start_wall_time: Optional[float] = None
        self.gvd_random_recovery_motion = None
        self.gvd_random = random.Random()
        self._gvd_obstruction_original_goal = None
        self._gvd_obstruction_path = ()
        self._gvd_obstruction_start = None
        self._gvd_obstruction_checkpoints = []
        self._gvd_obstruction_checkpoint_idx = 0

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
        raw_grid = np.asarray(msg.data, dtype=np.int8).reshape(
            msg.info.height,
            msg.info.width,
        )
        self.latest_grid = self.frontier_detector.fill_small_unknown_regions(raw_grid)
        self.frontier_clusters, _ = self.frontier_detector.detect(msg, self.latest_grid)
        self._update_explored_history(self.latest_grid)
        if self.slam_mode in ('gvd_gbsae', 'gvd_hierarchical', 'gvd_guide'):
            self.gvd_map_generation += 1
        if self.slam_mode == 'gvd_hierarchical':
            self.gvd_hierarchical_tracker.mark_route_dirty()

    def _control_loop(self):
        self._publish_visualizations()
        pose = self._get_robot_pose()
        if pose is not None:
            self.pose_graph_tracker.update(pose)
            if self.gvd_hierarchical_local_graph_tracker is not None:
                self.gvd_hierarchical_local_graph_tracker.update(pose)
            self._initialize_gbsae(pose)
            if self.gvd_sweep is not None:
                self.gvd_sweep.mark_pose((pose[0], pose[1]))

        if self.state == self.COMPLETE:
            return
        if not self.nav2.servers_ready():
            self._log_waiting_for_nav2()
            return
        if self.state == self.WAITING_FOR_NAV2:
            if self.latest_map is not None:
                self.get_logger().info(
                    'Nav2 and the first map are ready; starting exploration.'
                )
                self._schedule_retry(0.0)
            return
        if self._exploration_stuck(pose):
            self._start_gvd_stuck_recovery()
            return
        self._maybe_refresh_hierarchical_background_route(pose)
        if self._maybe_abort_hierarchical_local_cleanup_if_not_leaf():
            return
        if self._maybe_finish_hierarchical_local_cleanup_by_progress():
            return
        if self._maybe_trigger_gvd_guide_rebuild(pose):
            return
        if self.state == self.NAVIGATING:
            if self._maybe_replan_hierarchical_navigation(pose):
                return
            if self.current_navigation_kind == 'gvd_goal' and self._gvd_path_obstructed():
                self._handle_gvd_path_obstructed()
                return
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
                if self.current_navigation_kind in (
                    'frontier_staging',
                    'warmup_frontier_staging',
                ):
                    self._mark_goal_failed(self.current_goal)
                    self._mark_goal_failed(self.frontier_staging_final_goal)
                elif (
                    self.current_navigation_kind.startswith('gvd_guide_')
                    and self.gvd_guide_active_step is not None
                ):
                    self._handle_gvd_guide_step_failure(
                        self.gvd_guide_active_step,
                        'Nav2 navigation timed out',
                    )
                else:
                    self._handle_navigation_failure('Nav2 navigation timed out')
                self._clear_navigation()
                self._schedule_retry(0.0)
                return
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
        if self.state == self.RANDOM_RECOVERY_SPIN:
            if navigation_timed_out(
                self.gvd_random_recovery_action_start_wall_time,
                self.gvd_random_recovery_spin_timeout,
                time.monotonic(),
            ):
                self.nav2.cancel_spin()
                self._gvd_random_recovery_attempt_failed('random recovery spin timed out')
            return
        if self.state == self.RANDOM_RECOVERY_DRIVE:
            if navigation_timed_out(
                self.gvd_random_recovery_action_start_wall_time,
                self.gvd_random_recovery_drive_timeout,
                time.monotonic(),
            ):
                self.nav2.cancel_drive_on_heading()
                self._gvd_random_recovery_attempt_failed('random recovery drive timed out')
            return
        if self.state == self.SELECTING:
            if (
                self.selection_request_wall_time is not None
                and time.monotonic() - self.selection_request_wall_time > self.nav2_request_timeout
            ):
                self.get_logger().warn('Nav2 path request timed out; retrying frontier selection.')
                self._mark_goal_failed(self.selection_request_goal)
                if self.selection_request_is_frontier_staging:
                    self._mark_goal_failed(self.selection_request_final_goal)
                if self.selection_kind == 'hierarchical_gvd_vertex':
                    self.gvd_hierarchical_tracker.mark_route_dirty()
                elif self.selection_kind == 'gvd_guide_step':
                    planner = self.gvd_guide_planner
                    step = None if planner is None else planner.active_step
                    if step is not None:
                        self._handle_gvd_guide_step_failure(
                            step,
                            'Nav2 path request timed out',
                        )
                self.nav2.cancel_path_batch()
                self.selection_request_goal = None
                self.selection_request_final_goal = None
                self.selection_request_is_frontier_staging = False
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

    def _start_selection(self, rx: float, ry: float):
        if self.slam_mode == 'gvd_guide':
            if not self._gvd_guide_warmup_complete():
                self._start_gvd_guide_warmup_selection(rx, ry)
                return
            self._start_gvd_guide_selection(rx, ry)
            return
        if self.slam_mode == 'gvd_hierarchical':
            self._start_hierarchical_selection(rx, ry)
            return
        if self.slam_mode == 'gvd_gbsae':
            if self.gvd_phase == 'bootstrap':
                if (
                    self.gvd_sweep is not None
                    and self.gvd_sweep.ratio >= self.gvd_sweep_switch_ratio
                    and self._switch_gvd_to_gbsae(rx, ry)
                ):
                    self._start_gbsae_selection(rx, ry)
                else:
                    self._start_gvd_selection(rx, ry)
                return
            if self.gbsae_planner is not None:
                self._start_gbsae_selection(rx, ry)
                return
        if self.slam_mode == 'gbsae' and self.gbsae_planner is not None:
            self._start_gbsae_selection(rx, ry)
            return
        self._start_standard_selection(rx, ry)

    def _start_hierarchical_selection(self, rx: float, ry: float):
        if self.gvd_phase == 'tail_cleanup':
            self._start_standard_selection(rx, ry)
            return
        if self.gvd_phase == 'local_clear':
            self._start_hierarchical_local_selection(rx, ry)
            return
        self._start_hierarchical_macro_selection(rx, ry)

    def _make_pose_graph_tracker(self) -> ApproximatePoseGraphTracker:
        """Create an independent approximate graph tracker for one local Region."""
        return ApproximatePoseGraphTracker(
            node_spacing=self.graph_node_spacing,
            yaw_spacing=self.graph_yaw_spacing,
            loop_closure_radius=self.graph_loop_closure_radius,
            loop_closure_min_separation=self.graph_loop_closure_min_separation,
            loop_closure_weight=self.graph_loop_closure_weight,
            max_loop_closures_per_node=self.graph_max_loop_closures_per_node,
            odom_information=self.graph_odom_information,
        )

    def _start_hierarchical_macro_selection(self, rx: float, ry: float):
        self.gvd_hierarchical_local_mask = None
        self.gvd_hierarchical_local_geometry = None
        self.gvd_hierarchical_local_graph_tracker = None
        self.gvd_hierarchical_local_graph_candidates = []
        if self.latest_map is None or self.latest_grid is None:
            self._schedule_retry()
            return
        tracker = self.gvd_hierarchical_tracker
        if not self._refresh_hierarchical_route(
            rx,
            ry,
            force=self.gvd_topology is None,
        ):
            remaining = max(
                0.0,
                self.gvd_hierarchical_route_replan_interval
                - (
                    time.monotonic()
                    - self.gvd_hierarchical_last_route_replan_wall_time
                ),
            )
            self._schedule_retry(remaining)
            return
        topology = self.gvd_topology
        assert topology is not None
        component = tracker.graph

        if self._start_hierarchical_local_clear_if_ready(rx, ry):
            return

        now = time.monotonic()
        self.failed_goals.expire(now)
        target = tracker.select_macro_target(
            (rx, ry),
            failed=lambda point: self.failed_goals.contains(point, now),
            arrival_radius=self.nav2_goal_reach_radius,
        )
        if self._start_hierarchical_local_clear_if_ready(rx, ry):
            return
        if target is None:
            if tracker.has_pending_macro_targets:
                tracker.mark_route_dirty()
                self.get_logger().info(
                    'Hierarchical GVD has pending macro vertices, but all are '
                    f'temporarily unavailable; retrying in {self.frontier_retry_interval:.1f}s.'
                )
                self._schedule_retry()
                return
            self.gvd_phase = 'tail_cleanup'
            self.gvd_hierarchical_local_mask = None
            self.gvd_hierarchical_local_geometry = None
            self.get_logger().info(
                'Hierarchical GVD macro traversal completed; starting global frontier cleanup.'
            )
            self._start_standard_selection(rx, ry)
            return

        self.gvd_hierarchical_target = target
        self.selection_start_xy = (rx, ry)
        self.selection_generation = self.nav2.start_path_batch()
        self.selection_kind = 'hierarchical_gvd_vertex'
        self.selection_request_wall_time = time.monotonic()
        self.selection_request_goal = target.point
        self.state = self.SELECTING
        self.get_logger().info(
            f'Checking hierarchical GVD macro vertex={target.vertex_id} '
            f'goal=({target.point[0]:.2f}, {target.point[1]:.2f}), '
            f'travel={target.travel_cost:.2f}m, utility={target.utility:.3f}; '
            f'{self._gvd_repair_log(topology)}.'
        )
        self.nav2.compute_path(
            self.selection_generation,
            self.selection_start_xy,
            target.point,
            self._hierarchical_gvd_path_computed,
        )

    def _start_hierarchical_local_clear_if_ready(self, rx: float, ry: float) -> bool:
        """Interject Region cleanup when the active reached vertex is a leaf."""
        tracker = self.gvd_hierarchical_tracker
        active = tracker.active_vertex
        if active is None or active not in tracker.graph:
            return False
        active_point = _graph_node_point(tracker.graph, active)
        if (
            not tracker.should_clear_local(active)
            or math.dist((rx, ry), active_point)
            > self.gvd_hierarchical_local_half_extent
        ):
            return False
        self.get_logger().info(
            f'Hierarchical GVD vertex={active} is a reached leaf; '
            'triggering local clearance.'
        )
        self.gvd_phase = 'local_clear'
        self._start_hierarchical_local_selection(rx, ry)
        return True

    def _refresh_hierarchical_route(
        self,
        rx: float,
        ry: float,
        *,
        force: bool = False,
    ) -> bool:
        """Rebuild the live component and its TSP route at a bounded rate."""
        tracker = self.gvd_hierarchical_tracker
        if not tracker.route_dirty and self.gvd_topology is not None:
            return True
        now = time.monotonic()
        if not force and not route_replan_due(
            self.gvd_hierarchical_last_route_replan_wall_time,
            self.gvd_hierarchical_route_replan_interval,
            now,
        ):
            return False
        started = time.monotonic()
        topology = self._build_gvd_topology()
        component = robot_component_graph(topology.graph, (rx, ry))
        self.failed_goals.expire(now)
        tracker.update_graph(component)
        tracker.rebuild_route(
            (rx, ry),
            failed=lambda point: self.failed_goals.contains(point, now),
        )
        self.gvd_topology = topology
        self.gvd_hierarchical_last_route_replan_wall_time = now
        elapsed_ms = (time.monotonic() - started) * 1000.0
        self.get_logger().info(
            f'Rebuilt hierarchical GVD TSP route version={tracker.graph_version}: '
            f'{component.number_of_nodes()} nodes/{component.number_of_edges()} edges, '
            f'route_phase={tracker.route_phase}, '
            f'expansion_targets={list(tracker.expansion_targets)}, '
            f'cleanup_targets={list(tracker.cleanup_targets)}, '
            f'targets={list(tracker.route_targets)}, '
            f'transit={list(tracker.remaining_route)}, '
            f'continuation_successor={tracker.continuation_successor}, '
            f'length={tracker.route_length:.2f}m, '
            f'replan_ms={elapsed_ms:.1f}; {self._gvd_compression_log(topology)}, '
            f'{self._gvd_unknown_cycle_log(topology)}, '
            f'{self._gvd_repair_log(topology)}.'
        )
        return True

    def _maybe_replan_hierarchical_navigation(
        self,
        pose: Optional[Tuple[float, float, float]],
    ) -> bool:
        """Refresh a dirty macro route without replacing an active Nav2 macro goal."""
        if (
            self.slam_mode != 'gvd_hierarchical'
            or self.gvd_phase != 'macro'
            or self.current_navigation_kind != 'hierarchical_gvd_vertex'
            or pose is None
            or not self.gvd_hierarchical_tracker.route_dirty
            or not self._refresh_hierarchical_route(pose[0], pose[1])
        ):
            return False
        tracker = self.gvd_hierarchical_tracker
        self.gvd_hierarchical_target = tracker.remap_target(
            self.gvd_hierarchical_target
        )
        now = time.monotonic()
        desired = tracker.select_macro_target(
            (pose[0], pose[1]),
            failed=lambda point: self.failed_goals.contains(point, now),
            arrival_radius=self.nav2_goal_reach_radius,
        )
        if self._hierarchical_local_clear_ready(pose[0], pose[1]):
            self.get_logger().info(
                'Hierarchical GVD map update exposed a reached leaf vertex; '
                'preempting macro navigation for local clearance.'
            )
            self.nav2.cancel_navigation()
            self.gvd_hierarchical_target = None
            self._clear_navigation()
            self.gvd_phase = 'local_clear'
            self._schedule_retry(0.0)
            return True
        if (
            desired is not None
            and self.current_goal is not None
            and math.dist(desired.point, self.current_goal)
            <= self.nav2_goal_reach_radius
        ):
            self.gvd_hierarchical_target = desired
            return False
        self.get_logger().info(
            'Hierarchical GVD TSP first step changed after map update; '
            'keeping the active Nav2 macro goal until it finishes.'
        )
        return False

    def _maybe_refresh_hierarchical_background_route(
        self,
        pose: Optional[Tuple[float, float, float]],
    ):
        """Keep macro topology live during local cleanup without interrupting its actions."""
        if (
            self.slam_mode != 'gvd_hierarchical'
            or self.gvd_phase != 'local_clear'
            or pose is None
            or not self.gvd_hierarchical_tracker.route_dirty
            or self.latest_map is None
            or self.latest_grid is None
        ):
            return
        if self._refresh_hierarchical_route(pose[0], pose[1]):
            self.get_logger().info(
                f'Refreshed hierarchical GVD TSP route during local cleanup; '
                f'state={self.state}.'
            )

    def _hierarchical_local_clear_ready(self, rx: float, ry: float) -> bool:
        tracker = self.gvd_hierarchical_tracker
        active = tracker.active_vertex
        return (
            active is not None
            and active in tracker.graph
            and tracker.should_clear_local(active)
            and math.dist((rx, ry), _graph_node_point(tracker.graph, active))
            <= self.gvd_hierarchical_local_half_extent
        )

    def _hierarchical_gvd_path_computed(self, planned_path: Optional[PlannedPath]):
        if self.state != self.SELECTING or self.selection_kind != 'hierarchical_gvd_vertex':
            return
        self.selection_request_wall_time = None
        target = self.gvd_hierarchical_target
        if planned_path is not None and target is not None:
            self.nav2.cancel_path_batch()
            self.current_goal = planned_path.goal_xy
            self.current_safe_goal = None
            self.target_cluster = None
            self.selection_request_goal = None
            self.current_navigation_kind = 'hierarchical_gvd_vertex'
            self.state = self.NAVIGATING
            self.navigation_start_wall_time = time.monotonic()
            self._start_gvd_progress_watchdog(self.selection_start_xy)
            self.get_logger().info(
                f'Navigating to hierarchical GVD macro vertex={target.vertex_id}: '
                f'goal=({planned_path.goal_xy[0]:.2f}, {planned_path.goal_xy[1]:.2f}), '
                f'cost={planned_path.cost:.2f}.'
            )
            self.nav2.navigate(planned_path.goal_xy, 0.0, self._navigation_finished)
            return
        self._mark_goal_failed(self.selection_request_goal)
        self.gvd_hierarchical_tracker.mark_route_dirty()
        self.selection_request_goal = None
        self.gvd_hierarchical_target = None
        self.nav2.cancel_path_batch()
        self._schedule_retry()

    def _start_hierarchical_local_selection(self, rx: float, ry: float):
        tracker = self.gvd_hierarchical_tracker
        if self._maybe_abort_hierarchical_local_cleanup_if_not_leaf():
            return
        if (
            self.latest_map is None
            or self.latest_grid is None
            or tracker.active_point is None
        ):
            self.gvd_phase = 'macro'
            self.gvd_hierarchical_local_mask = None
            self.gvd_hierarchical_local_geometry = None
            self.gvd_hierarchical_local_graph_tracker = None
            self.gvd_hierarchical_local_graph_candidates = []
            self._schedule_retry(0.0)
            return
        local_geometry = self._latest_map_geometry()
        local_mask = local_free_flood_mask(
            self.latest_grid,
            local_geometry,
            tracker.active_point,
            self.gvd_hierarchical_local_half_extent,
            bounds=self.gvd_bounds,
            excluded_points=tuple(
                (float(attributes['x']), float(attributes['y']))
                for vertex_id, attributes in tracker.graph.nodes(data=True)
                if vertex_id != tracker.active_vertex
            ),
            area_weight=self.gvd_hierarchical_region_area_weight,
            squareness_weight=self.gvd_hierarchical_region_squareness_weight,
        )
        self.gvd_hierarchical_local_mask = local_mask
        self.gvd_hierarchical_local_geometry = local_geometry
        progress = self._hierarchical_local_cleanup_progress()
        if self._maybe_finish_hierarchical_local_cleanup_by_progress():
            return
        if self.gvd_hierarchical_local_graph_tracker is None:
            self.gvd_hierarchical_local_graph_tracker = self._make_pose_graph_tracker()
            pose = self._get_robot_pose()
            self.gvd_hierarchical_local_graph_tracker.update(
                (rx, ry, 0.0 if pose is None else pose[2])
            )
        self.gvd_hierarchical_local_graph_candidates = []
        clusters = [
            cluster
            for cluster in self.frontier_clusters
            if cluster_touches_mask(cluster.cells, local_mask)
        ]
        self.selection_start_xy = (rx, ry)
        now = time.monotonic()
        self.failed_goals.expire(now)
        self.selection_candidates = self._ranked_frontier_candidates(
            rx,
            ry,
            now,
            self.frontier_planning_attempts,
            clusters=clusters,
            local_cleanup=True,
        )
        self.selection_generation = self.nav2.start_path_batch()
        self.selection_cluster_index = 0
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        self.selection_request_final_goal = None
        self.selection_request_is_frontier_staging = False
        self.selection_kind = 'hierarchical_local_frontier'
        self.state = self.SELECTING
        self.get_logger().info(
            f'Hierarchical GVD local cleanup around vertex={tracker.active_vertex}: '
            f'known_coverage={progress:.1%}, '
            f'{len(clusters)} clusters, {len(self.selection_candidates)} safe candidates.'
        )
        self._request_next_hierarchical_local_path()

    def _request_next_hierarchical_local_path(self):
        if self.selection_cluster_index < len(self.selection_candidates):
            candidate = self.selection_candidates[self.selection_cluster_index]
            goal_xy, is_staging = self._frontier_request_goal(candidate)
            self.selection_request_wall_time = time.monotonic()
            self.selection_request_goal = goal_xy
            self.nav2.compute_path(
                self.selection_generation,
                self.selection_start_xy,
                goal_xy,
                self._hierarchical_local_path_computed,
            )
            if is_staging:
                self.get_logger().info(
                    'Checking hierarchical local frontier staging goal before '
                    'the final cleanup frontier.'
                )
            return
        best = best_graph_candidate(self.gvd_hierarchical_local_graph_candidates)
        if best is not None:
            score, candidate, planned_path = best
            self.get_logger().info(
                f'Hierarchical GVD local approx-graph selected frontier '
                f'score={score:.3f}, size={candidate.cluster.size}, '
                f'source={candidate.cluster.source}.'
            )
            self._dispatch_navigation(candidate, planned_path)
            return
        self._finish_hierarchical_local_cleanup(
            reason='no Nav2-reachable Region-local frontier remains',
        )

    def _hierarchical_local_cleanup_progress(self) -> float:
        """Return observed coverage for the active Region snapshot."""
        if (
            self.latest_grid is None
            or self.gvd_hierarchical_local_mask is None
        ):
            return 0.0
        return local_region_known_ratio(
            self.latest_grid,
            self.gvd_hierarchical_local_mask,
            grid_geometry=self._latest_map_geometry(),
            region_geometry=self.gvd_hierarchical_local_geometry,
        )

    def _maybe_abort_hierarchical_local_cleanup_if_not_leaf(self) -> bool:
        """Return to macro traversal if a live rebuild invalidates the active leaf."""
        if self.slam_mode != 'gvd_hierarchical' or self.gvd_phase != 'local_clear':
            return False
        tracker = self.gvd_hierarchical_tracker
        active = tracker.active_vertex
        if (
            active is not None
            and active in tracker.graph
            and tracker.should_clear_local(active)
        ):
            return False
        if self.state == self.NAVIGATING:
            self.nav2.cancel_navigation()
        elif self.state == self.ALIGNING_FRONTIER_PROBE:
            self.nav2.cancel_spin()
        elif self.state == self.PROBING_FRONTIER:
            self.nav2.cancel_drive_on_heading()
        self.nav2.cancel_path_batch()
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        self._clear_navigation()
        self.gvd_phase = 'macro'
        self.gvd_hierarchical_local_mask = None
        self.gvd_hierarchical_local_geometry = None
        self.gvd_hierarchical_local_graph_tracker = None
        self.gvd_hierarchical_local_graph_candidates = []
        tracker.mark_route_dirty()
        self.get_logger().info(
            f'Hierarchical GVD vertex={active} no longer maps to a reached leaf; '
            'aborting local cleanup and returning to macro traversal.'
        )
        self._schedule_retry(0.0)
        return True

    def _maybe_finish_hierarchical_local_cleanup_by_progress(self) -> bool:
        """Stop an active local action once the Region is sufficiently observed."""
        if self.slam_mode != 'gvd_hierarchical' or self.gvd_phase != 'local_clear':
            return False
        progress = self._hierarchical_local_cleanup_progress()
        if progress < self.gvd_hierarchical_local_clear_progress_threshold:
            return False
        if self.state == self.NAVIGATING:
            self.nav2.cancel_navigation()
        elif self.state == self.ALIGNING_FRONTIER_PROBE:
            self.nav2.cancel_spin()
        elif self.state == self.PROBING_FRONTIER:
            self.nav2.cancel_drive_on_heading()
        elif self.state == self.SELECTING:
            self.nav2.cancel_path_batch()
        self._clear_navigation()
        self._finish_hierarchical_local_cleanup(
            reason=(
                f'Region known coverage {progress:.1%} reached '
                f'{self.gvd_hierarchical_local_clear_progress_threshold:.1%}'
            ),
        )
        return True

    def _finish_hierarchical_local_cleanup(self, *, reason: str):
        """Archive one completed local Region and resume macro traversal."""
        tracker = self.gvd_hierarchical_tracker
        cleared_vertex = tracker.active_vertex
        tracker.mark_local_cleared(cleared_vertex)
        outline = (
            rectangle_mask_outline(
                self.gvd_hierarchical_local_mask,
                self.gvd_hierarchical_local_geometry,
            )
            if (
                self.gvd_hierarchical_local_mask is not None
                and self.gvd_hierarchical_local_geometry is not None
            )
            else ()
        )
        if outline and outline not in self.gvd_hierarchical_cleared_region_outlines:
            self.gvd_hierarchical_cleared_region_outlines.append(outline)
        self.nav2.cancel_path_batch()
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        self.gvd_phase = 'macro'
        self.gvd_hierarchical_local_mask = None
        self.gvd_hierarchical_local_geometry = None
        self.gvd_hierarchical_local_graph_tracker = None
        self.gvd_hierarchical_local_graph_candidates = []
        self.get_logger().info(
            f'Hierarchical GVD local cleanup completed for vertex={cleared_vertex}: '
            f'{reason}; returning to macro traversal.'
        )
        self._schedule_retry(0.0)

    def _hierarchical_local_path_computed(self, planned_path: Optional[PlannedPath]):
        if self.state != self.SELECTING or self.selection_kind != 'hierarchical_local_frontier':
            return
        self.selection_request_wall_time = None
        candidate = self.selection_candidates[self.selection_cluster_index]
        if planned_path is not None:
            if self.selection_request_is_frontier_staging:
                self._dispatch_frontier_staging(candidate, planned_path)
                return
            if not self.gvd_hierarchical_local_approx_graph_enabled:
                self._dispatch_navigation(candidate, planned_path)
                return
            tracker = self.gvd_hierarchical_local_graph_tracker
            if tracker is not None:
                score = self.graph_scorer.score(
                    tracker.graph,
                    self.latest_map,
                    planned_path.points,
                    self.latest_grid,
                )
                if np.isfinite(score):
                    self.gvd_hierarchical_local_graph_candidates.append(
                        (score, candidate, planned_path)
                    )
        else:
            self._mark_goal_failed(self.selection_request_goal)
            if self.selection_request_is_frontier_staging:
                self._mark_goal_failed(self.selection_request_final_goal)
        self.selection_request_goal = None
        self.selection_cluster_index += 1
        self._request_next_hierarchical_local_path()

    def _gvd_guide_warmup_complete(self) -> bool:
        return (
            self.gvd_guide_warmup_distance <= 0.0
            or self.gvd_guide_warmup_travel >= self.gvd_guide_warmup_distance
        )

    def _start_gvd_guide_warmup_selection(self, rx: float, ry: float):
        if not self.frontier_clusters:
            self._finish_gvd_guide_warmup_early(
                'no frontiers available for greedy warmup'
            )
            self._schedule_retry(0.0)
            return
        if not self.gvd_guide_warmup_started:
            self.gvd_guide_warmup_started = True
            self.get_logger().info(
                f'GVD guide warmup started: greedily selecting max-unknown '
                f'frontiers for {self.gvd_guide_warmup_distance:.1f}m before '
                'sparse GVD planning.'
            )
        self.selection_start_xy = (rx, ry)
        now = time.monotonic()
        self.failed_goals.expire(now)
        local_filter_start = time.monotonic()
        self.selection_candidates = self._ranked_frontier_candidates(
            rx,
            ry,
            now,
            max(self.frontier_planning_attempts, len(self.frontier_clusters)),
            greedy_range_gain=True,
        )
        local_filter_ms = (time.monotonic() - local_filter_start) * 1000.0
        if not self.selection_candidates:
            self._finish_gvd_guide_warmup_early(
                'no safe frontier candidates for greedy warmup'
            )
            self._schedule_retry(0.0)
            return
        self.selection_generation = self.nav2.start_path_batch()
        self.selection_cluster_index = 0
        self.graph_candidates = []
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        self.selection_request_final_goal = None
        self.selection_request_is_frontier_staging = False
        self.selection_kind = 'gvd_guide_warmup'
        self.state = self.SELECTING
        remaining = max(0.0, self.gvd_guide_warmup_distance - self.gvd_guide_warmup_travel)
        best = self.selection_candidates[0]
        self.get_logger().info(
            f'GVD guide warmup frontier pool has {len(self.selection_candidates)} '
            f'candidates, remaining={remaining:.2f}m, '
            f'best_range_gain={best.range_gain:.2f}m^2, '
            f'local_filter_ms={local_filter_ms:.1f}.'
        )
        self._request_next_path()

    def _finish_gvd_guide_warmup_early(self, reason: str):
        self.gvd_guide_warmup_travel = max(
            self.gvd_guide_warmup_travel,
            self.gvd_guide_warmup_distance,
        )
        if not self.gvd_guide_warmup_complete_logged:
            self.gvd_guide_warmup_complete_logged = True
            self.get_logger().warn(
                f'GVD guide warmup ended early: {reason}; starting sparse GVD planning.'
            )

    def _start_gvd_guide_selection(self, rx: float, ry: float):
        if self.latest_map is None or self.latest_grid is None:
            self._schedule_retry()
            return
        if (
            self.gvd_guide_planner is None
            or self.gvd_guide_planner.is_complete
            or self.gvd_guide_pending_rebuild_reason is not None
        ):
            reason = self.gvd_guide_pending_rebuild_reason or 'plan queue empty'
            if not self._rebuild_gvd_guide_plan(rx, ry, reason=reason):
                self._start_standard_selection(rx, ry)
                return
        planner = self.gvd_guide_planner
        if planner is None or planner.active_step is None:
            self.get_logger().info(
                'GVD guide plan has no queued sparse waypoints; falling back to frontier cleanup.'
            )
            self._start_standard_selection(rx, ry)
            return
        self._request_gvd_guide_active_path(rx, ry)

    def _rebuild_gvd_guide_plan(self, rx: float, ry: float, *, reason: str) -> bool:
        if self.latest_map is None or self.latest_grid is None:
            return False
        started = time.monotonic()
        topology = self._build_gvd_topology()
        now = time.monotonic()
        start_hint = self._gvd_guide_rebuild_start_hint()
        self.failed_goals.expire(now)
        candidates = self._ranked_frontier_candidates(
            rx,
            ry,
            now,
            len(self.frontier_clusters),
        )
        planner = GVDGuidePlanner.build(
            topology,
            self.latest_grid,
            (rx, ry),
            candidates,
            loop_path_cost_weight=self.gvd_guide_loop_path_cost_weight,
            frontier_detour_weight=self.gvd_guide_frontier_detour_weight,
            frontier_detour_max_extra_distance=(
                self.gvd_guide_frontier_detour_max_extra_distance
            ),
            frontier_detour_min_gain=self.gvd_guide_frontier_detour_min_gain,
            explored_points=tuple(self.gvd_guide_explored_points),
            migration_radius=self.gvd_guide_explored_migration_radius,
            max_waypoint_distance=self.gvd_max_goal_distance,
            start_hint=start_hint,
        )
        self.gvd_topology = topology
        self.gvd_guide_planner = planner
        self.gvd_guide_pending_rebuild_reason = None
        self.gvd_guide_rebuild_start_hint_xy = None
        self.gvd_guide_last_rebuild_wall_time = now
        self.gvd_guide_last_expansion_area = 0.0
        elapsed_ms = (time.monotonic() - started) * 1000.0
        detours = sum(step.kind == 'frontier_detour' for step in planner.plan_queue)
        loop_steps = sum(step.kind == 'loop_revisit' for step in planner.plan_queue)
        self.get_logger().info(
            f'Rebuilt GVD guide plan reason="{reason}": '
            f'sparse={planner.graph.number_of_nodes()} nodes/'
            f'{planner.graph.number_of_edges()} edges, '
            f'explored={len(planner.explored_vertices)}/'
            f'{len(self.gvd_guide_explored_points)}, '
            f'start_hint={None if start_hint is None else (round(start_hint[0], 2), round(start_hint[1], 2))}, '
            f'route={list(planner.route_vertices)}, '
            f'route_shortcuts={planner.route_shortcuts}, '
            f'queue={len(planner.plan_queue)}, detours={detours}, '
            f'loop_revisits={loop_steps}, loop_edges={list(planner.loop_edges)}, '
            f'rebuild_ms={elapsed_ms:.1f}; {self._gvd_compression_log(topology)}, '
            f'{self._gvd_unknown_cycle_log(topology)}, {self._gvd_repair_log(topology)}.'
        )
        return planner.graph.number_of_nodes() > 0 and bool(planner.plan_queue)

    def _request_gvd_guide_active_path(self, rx: float, ry: float):
        planner = self.gvd_guide_planner
        if planner is None:
            self._schedule_retry()
            return
        step = planner.active_step
        if step is None:
            self._schedule_retry(0.0)
            return
        if math.dist((rx, ry), step.goal_xy) <= self.nav2_goal_reach_radius:
            explored_vertex = self._record_gvd_guide_explored_step(step)
            if self.gvd_guide_pending_rebuild_reason is not None:
                self.gvd_guide_rebuild_start_hint_xy = self._gvd_guide_step_target_hint(step)
            reached = planner.advance_active_step()
            skipped = (
                self._skip_gvd_guide_steps_to_explored_vertex(explored_vertex)
                if reached is not None and reached.kind == 'frontier_detour'
                else 0
            )
            self._refresh_gvd_guide_online_detours(reached)
            self.get_logger().info(
                f'GVD guide waypoint already reached: kind={reached.kind}, '
                f'goal=({reached.goal_xy[0]:.2f}, {reached.goal_xy[1]:.2f})'
                f'{", skipped_explored_steps=" + str(skipped) if skipped else ""}.'
            )
            self._schedule_retry(0.0)
            return
        self.selection_start_xy = (rx, ry)
        self.selection_generation = self.nav2.start_path_batch()
        self.selection_kind = 'gvd_guide_step'
        self.selection_request_wall_time = time.monotonic()
        step_goal = step.goal_xy
        self.selection_request_final_goal = step.goal_xy
        self.selection_request_is_frontier_staging = False
        if step.kind in ('gvd_vertex', 'loop_revisit'):
            step_goal = self._sample_gvd_guide_waypoint_goal(step.goal_xy)
        elif step.kind == 'frontier_detour':
            staging_goal = self._frontier_staging_goal(self.selection_start_xy, step.goal_xy)
            if staging_goal is not None:
                step_goal = staging_goal
                self.selection_request_is_frontier_staging = True
        self.selection_request_goal = step_goal
        self.state = self.SELECTING
        self.get_logger().info(
            f'Checking GVD guide step kind={step.kind}, '
            f'goal=({step_goal[0]:.2f}, {step_goal[1]:.2f}), '
            f'final=({step.goal_xy[0]:.2f}, {step.goal_xy[1]:.2f}), '
            f'expected_cost={step.expected_cost:.2f}.'
        )
        self.nav2.compute_path(
            self.selection_generation,
            self.selection_start_xy,
            step_goal,
            self._gvd_guide_path_computed,
        )

    def _gvd_guide_path_computed(self, planned_path: Optional[PlannedPath]):
        if self.state != self.SELECTING or self.selection_kind != 'gvd_guide_step':
            return
        self.selection_request_wall_time = None
        planner = self.gvd_guide_planner
        step = None if planner is None else planner.active_step
        if planner is None or step is None:
            self.selection_request_goal = None
            self.selection_request_final_goal = None
            self.selection_request_is_frontier_staging = False
            self.nav2.cancel_path_batch()
            self._schedule_retry(0.0)
            return
        if planned_path is not None:
            dispatch_step = step
            if self.selection_request_is_frontier_staging:
                dispatch_step = replace(
                    step,
                    kind='frontier_staging',
                    goal_xy=planned_path.goal_xy,
                    expected_cost=planned_path.cost,
                )
            self._dispatch_gvd_guide_navigation(dispatch_step, planned_path)
            return
        self._handle_gvd_guide_step_failure(step, 'Nav2 could not plan to waypoint')
        self.selection_request_goal = None
        self.selection_request_final_goal = None
        self.selection_request_is_frontier_staging = False
        self.nav2.cancel_path_batch()
        self._schedule_retry(0.0)

    def _dispatch_gvd_guide_navigation(
        self,
        step: GVDGuidePlanStep,
        planned_path: PlannedPath,
    ):
        self.nav2.cancel_path_batch()
        self.gvd_guide_active_step = step
        self.current_goal = planned_path.goal_xy
        self.current_safe_goal = (
            getattr(step.frontier, 'safe_goal', None)
            if step.kind == 'frontier_detour'
            else None
        )
        self.target_cluster = (
            getattr(step.frontier, 'cluster', None)
            if step.kind in ('frontier_detour', 'frontier_staging')
            else None
        )
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        self.selection_request_final_goal = None
        self.selection_request_is_frontier_staging = False
        self.current_navigation_kind = f'gvd_guide_{step.kind}'
        self.gvd_active_path = tuple(planned_path.points)
        _, self.gvd_active_traversability = self._current_gvd_traversability()
        self.gvd_checked_map_generation = self.gvd_map_generation
        self.state = self.NAVIGATING
        self.navigation_start_wall_time = time.monotonic()
        self._start_gvd_progress_watchdog(self.selection_start_xy)
        yaw_target = planned_path.goal_xy
        if self.target_cluster is not None:
            yaw_target = (self.target_cluster.centroid_x, self.target_cluster.centroid_y)
        yaw = (
            heading_to_target(planned_path.points[-2], yaw_target)
            if len(planned_path.points) >= 2
            else 0.0
        )
        self.get_logger().info(
            f'Navigating along GVD guide step kind={step.kind}: '
            f'goal=({planned_path.goal_xy[0]:.2f}, {planned_path.goal_xy[1]:.2f}), '
            f'cost={planned_path.cost:.2f}.'
        )
        self.nav2.navigate(planned_path.goal_xy, yaw, self._navigation_finished)

    def _handle_gvd_guide_step_failure(self, step: GVDGuidePlanStep, reason: str):
        planner = self.gvd_guide_planner
        if step.kind == 'loop_revisit':
            if planner is not None:
                planner.skip_active_step()
            self.get_logger().warn(
                f'Skipping optional GVD guide loop revisit: {reason}.'
            )
            return
        if step.kind in ('frontier_detour', 'frontier_staging'):
            self._mark_goal_failed(step.goal_xy)
            safe_goal = getattr(getattr(step, 'frontier', None), 'safe_goal', None)
            if safe_goal is not None:
                self._mark_goal_failed(safe_goal.point)
            if planner is not None:
                planner.skip_active_step()
            self.get_logger().warn(
                f'Skipping GVD guide frontier detour after failure: {reason}.'
            )
            return
        self._mark_goal_failed(step.goal_xy)
        self.gvd_guide_rebuild_start_hint_xy = self._gvd_guide_step_target_hint(step)
        self.gvd_guide_pending_rebuild_reason = reason
        self.gvd_guide_planner = None
        self.get_logger().warn(
            f'GVD guide macro waypoint failed; forcing sparse rebuild: {reason}.'
        )

    def _finish_gvd_guide_navigation(self, status: int) -> bool:
        if not self.current_navigation_kind.startswith('gvd_guide_'):
            return False
        step = self.gvd_guide_active_step
        planner = self.gvd_guide_planner
        if step is None:
            self._clear_navigation()
            self._schedule_retry(0.0)
            return True
        if status == GOAL_STATUS_SUCCEEDED:
            if step.kind == 'frontier_staging':
                self.get_logger().info(
                    'Nav2 reached a GVD guide frontier staging waypoint; '
                    'continuing the same detour on the next selection pass.'
                )
            else:
                explored_vertex = self._record_gvd_guide_explored_step(step)
                skipped = 0
                if planner is not None:
                    if self.gvd_guide_pending_rebuild_reason is not None:
                        self.gvd_guide_rebuild_start_hint_xy = (
                            self._gvd_guide_step_target_hint(step)
                        )
                    planner.advance_active_step()
                    skipped = (
                        self._skip_gvd_guide_steps_to_explored_vertex(explored_vertex)
                        if step.kind == 'frontier_detour'
                        else 0
                    )
                    self._refresh_gvd_guide_online_detours(step)
                self.get_logger().info(
                    f'Nav2 reached GVD guide waypoint kind={step.kind}'
                    f'{", skipped_explored_steps=" + str(skipped) if skipped else ""}.'
                )
            if step.kind == 'frontier_detour' and self._frontier_probe_settings() is not None:
                self._start_frontier_probe()
                return True
        else:
            self._handle_gvd_guide_step_failure(
                step,
                f'Nav2 navigation ended with status={status}',
            )
        self.gvd_guide_active_step = None
        self._clear_navigation()
        self._schedule_retry(0.0)
        return True

    def _record_gvd_guide_explored_step(
        self,
        step: GVDGuidePlanStep,
    ) -> Optional[int]:
        """Persist reached sparse guide vertices across future rebuilds."""
        if step.kind == 'gvd_vertex' and step.vertex_id is not None:
            node_id = step.vertex_id
        elif step.kind == 'frontier_detour' and step.target_vertex is not None:
            node_id = step.target_vertex
        else:
            return None
        point = step.goal_xy
        if (
            self.gvd_guide_planner is not None
            and node_id in self.gvd_guide_planner.graph
        ):
            self.gvd_guide_planner.graph.nodes[node_id]['explored'] = True
            point = _graph_node_point(self.gvd_guide_planner.graph, node_id)
        radius = max(0.0, self.gvd_guide_explored_migration_radius)
        if any(math.dist(point, existing) <= radius for existing in self.gvd_guide_explored_points):
            return node_id
        self.gvd_guide_explored_points.append(point)
        return node_id

    def _skip_gvd_guide_steps_to_explored_vertex(self, vertex_id: Optional[int]) -> int:
        planner = self.gvd_guide_planner
        if planner is None or vertex_id is None:
            return 0
        skipped = 0
        while (
            planner.active_step is not None
            and planner.active_step.kind == 'gvd_vertex'
            and (
                planner.active_step.vertex_id == vertex_id
                or planner.active_step.target_vertex == vertex_id
            )
        ):
            planner.skip_active_step()
            skipped += 1
        return skipped

    def _gvd_guide_rebuild_start_hint(self) -> Optional[Tuple[float, float]]:
        if self.gvd_guide_rebuild_start_hint_xy is not None:
            return self.gvd_guide_rebuild_start_hint_xy
        planner = self.gvd_guide_planner
        step = self.gvd_guide_active_step
        if step is None and planner is not None:
            step = planner.active_step
        return self._gvd_guide_step_target_hint(step)

    def _gvd_guide_step_target_hint(
        self,
        step: Optional[GVDGuidePlanStep],
    ) -> Optional[Tuple[float, float]]:
        planner = self.gvd_guide_planner
        if step is None or planner is None:
            return None
        node_id = step.vertex_id if step.vertex_id is not None else step.target_vertex
        if node_id is not None and node_id in planner.graph:
            return _graph_node_point(planner.graph, node_id)
        if step.vertex_id is not None:
            return step.goal_xy
        return None

    def _refresh_gvd_guide_online_detours(self, step: Optional[GVDGuidePlanStep]):
        """Refresh the next local frontier detour after a sparse guide vertex is reached."""
        planner = self.gvd_guide_planner
        if (
            planner is None
            or step is None
            or step.vertex_id is None
            or self.latest_grid is None
            or self.latest_map is None
        ):
            return
        pose = self._get_robot_pose()
        rx, ry = (step.goal_xy if pose is None else (pose[0], pose[1]))
        now = time.monotonic()
        self.failed_goals.expire(now)
        candidates = self._ranked_frontier_candidates(
            rx,
            ry,
            now,
            len(self.frontier_clusters),
        )
        update = planner.refresh_online_frontier_detour(
            step.vertex_id,
            candidates,
            frontier_detour_weight=self.gvd_guide_frontier_detour_weight,
            frontier_detour_max_extra_distance=(
                self.gvd_guide_frontier_detour_max_extra_distance
            ),
            frontier_detour_min_gain=self.gvd_guide_frontier_detour_min_gain,
        )
        if update.target_vertex is None:
            self.get_logger().info(
                f'GVD guide online detour analysis at vertex={step.vertex_id}: '
                f'assigned_frontiers={update.assigned_frontiers}, '
                f'removed_stale_detours={update.removed_detours}, no next local segment.'
            )
            return
        selected = (
            'none'
            if update.selected_goal is None
            else f'({update.selected_goal[0]:.2f}, {update.selected_goal[1]:.2f})'
        )
        self.get_logger().info(
            f'GVD guide online detour analysis {update.source_vertex}->'
            f'{update.target_vertex}: assigned_frontiers={update.assigned_frontiers}, '
            f'target_frontiers={update.target_frontiers}, '
            f'removed_stale_detours={update.removed_detours}, '
            f'inserted={update.inserted}, selected={selected}, '
            f'extra_cost={update.expected_extra_cost:.2f}.'
        )

    def _maybe_trigger_gvd_guide_rebuild(
        self,
        pose: Optional[Tuple[float, float, float]],
    ) -> bool:
        if (
            self.slam_mode != 'gvd_guide'
            or self.latest_map is None
            or self.latest_grid is None
            or self.gvd_guide_planner is None
        ):
            return False
        if (
            self.state == self.NAVIGATING
            and self.current_navigation_kind.startswith('gvd_guide_')
            and self.gvd_guide_active_step is not None
            and self.gvd_guide_active_step.edge is not None
            and self.gvd_checked_map_generation != self.gvd_map_generation
        ):
            blocked_run = self._gvd_guide_active_edge_blocked_run()
            self.gvd_checked_map_generation = self.gvd_map_generation
            if blocked_run > self.gvd_guide_edge_blocked_run_threshold:
                self.get_logger().warn(
                    f'GVD guide active sparse edge inflated_obstacle_blocked_run='
                    f'{blocked_run:.2f}m '
                    f'exceeds {self.gvd_guide_edge_blocked_run_threshold:.2f}m; '
                    'canceling active goal and rebuilding.'
                )
                self.nav2.cancel_navigation()
                self.gvd_guide_rebuild_start_hint_xy = self._gvd_guide_step_target_hint(
                    self.gvd_guide_active_step
                )
                self.gvd_guide_pending_rebuild_reason = (
                    f'active edge inflated_obstacle_blocked_run={blocked_run:.2f}m'
                )
                self.gvd_guide_planner = None
                self.gvd_guide_active_step = None
                self._clear_navigation()
                self._schedule_retry(0.0)
                return True
        area = self._gvd_guide_off_graph_new_free_area()
        self.gvd_guide_last_expansion_area = area
        if (
            area > self.gvd_guide_expansion_area_threshold
            and self.gvd_guide_pending_rebuild_reason is None
            and time.monotonic() - self.gvd_guide_last_rebuild_wall_time
            >= self.gvd_guide_rebuild_cooldown
        ):
            self.gvd_guide_pending_rebuild_reason = (
                f'off_graph_new_free_area={area:.2f}m^2'
            )
            self.get_logger().info(
                f'GVD guide queued sparse rebuild at next safe point: '
                f'off_graph_new_free_area={area:.2f}m^2.'
            )
        return False

    def _gvd_guide_active_edge_blocked_run(self) -> float:
        planner = self.gvd_guide_planner
        if planner is None:
            return 0.0
        pose = self._get_robot_pose()
        polyline = (
            path_suffix_from_nearest(self.gvd_active_path, (pose[0], pose[1]))
            if pose is not None and self.gvd_active_path
            else planner.active_sparse_edge_polyline()
        )
        if not polyline:
            return 0.0
        geometry, inflated_traversable = self._current_gvd_traversability()
        return gvd_guide_edge_blocked_run(polyline, geometry, inflated_traversable)

    def _gvd_guide_off_graph_new_free_area(self) -> float:
        planner = self.gvd_guide_planner
        if planner is None or self.latest_grid is None:
            return 0.0
        return off_graph_new_free_area(
            planner.rebuild_grid,
            self.latest_grid,
            self._latest_map_geometry(),
            planner.graph,
            self.gvd_guide_expansion_distance,
        )

    def _start_gvd_selection(self, rx: float, ry: float):
        if self.latest_map is None or self.latest_grid is None or self.gvd_sweep is None:
            self._schedule_retry()
            return
        pose = self._get_robot_pose()
        if pose is None:
            self._schedule_retry()
            return
        topology = self._build_gvd_topology()
        self.gvd_topology = topology
        now = time.monotonic()
        self.failed_goals.expire(now)
        self.gvd_candidates = rank_gvd_goals(
            topology,
            (rx, ry),
            pose[2],
            self.gvd_sweep,
            self.gvd_weights,
            min_goal_distance=self.gvd_min_goal_distance,
            max_goal_distance=self.gvd_max_goal_distance,
            candidate_limit=self.gvd_candidate_limit,
            skeleton_cost=self.gvd_skeleton_cost,
            off_skeleton_cost=self.gvd_off_skeleton_cost,
            centerline_distance_weight=self.gvd_centerline_distance_weight,
            failed=lambda goal: self.failed_goals.contains(goal, now),
        )[:self.gvd_nav2_planning_attempts]
        self.selection_start_xy = (rx, ry)
        self.selection_generation = self.nav2.start_path_batch()
        self.gvd_candidate_index = 0
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        self.selection_kind = 'gvd_goal'
        self.state = self.SELECTING
        self.get_logger().info(
            f'GVD bootstrap swept={self.gvd_sweep.ratio:.1%}; '
            f'topology={topology.graph.number_of_nodes()} nodes/'
            f'{topology.graph.number_of_edges()} edges; '
            f'{self._gvd_repair_log(topology)}; '
            f'checking {len(self.gvd_candidates)} A*-reachable candidates.'
        )
        self._request_next_gvd_path()

    def _request_next_gvd_path(self):
        if self.gvd_candidate_index >= len(self.gvd_candidates):
            self.nav2.cancel_path_batch()
            self.get_logger().info(
                f'GVD bootstrap found no reachable goal; retrying in '
                f'{self.frontier_retry_interval:.1f}s.'
            )
            self._schedule_retry()
            return
        candidate = self.gvd_candidates[self.gvd_candidate_index]
        self.selection_request_wall_time = time.monotonic()
        self.selection_request_goal = candidate.point
        self.get_logger().info(
            f'Checking GVD goal=({candidate.point[0]:.2f}, {candidate.point[1]:.2f}), '
            f'utility={candidate.utility:.3f}, border={candidate.boundary_unknown:.2f}, '
            f'distance={candidate.goal_distance:.2f}, overlap={candidate.path_overlap:.2f}, '
            f'straight={candidate.straightness:.2f}.'
        )
        self.nav2.compute_path(
            self.selection_generation,
            self.selection_start_xy,
            candidate.point,
            self._gvd_path_computed,
        )

    def _gvd_path_computed(self, planned_path: Optional[PlannedPath]):
        if self.state != self.SELECTING or self.selection_kind != 'gvd_goal':
            return
        self.selection_request_wall_time = None
        candidate = self.gvd_candidates[self.gvd_candidate_index]
        if planned_path is not None:
            self._dispatch_gvd_navigation(candidate, planned_path)
            return
        self._mark_goal_failed(self.selection_request_goal)
        self.selection_request_goal = None
        self.gvd_candidate_index += 1
        self._request_next_gvd_path()

    def _dispatch_gvd_navigation(self, candidate: GVDGoal, planned_path: PlannedPath):
        self.nav2.cancel_path_batch()
        self.current_goal = planned_path.goal_xy
        self.current_safe_goal = None
        self.target_cluster = None
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        self.current_navigation_kind = 'gvd_goal'
        self.gvd_active_path = tuple(planned_path.points)
        _, self.gvd_active_traversability = self._current_gvd_traversability()
        self.gvd_checked_map_generation = self.gvd_map_generation
        self.state = self.NAVIGATING
        self.navigation_start_wall_time = time.monotonic()
        self._start_gvd_progress_watchdog(self.selection_start_xy)
        yaw = heading_to_target(planned_path.points[-2], planned_path.goal_xy) if (
            len(planned_path.points) >= 2
        ) else 0.0
        self.get_logger().info(
            f'Navigating to GVD bootstrap goal=({planned_path.goal_xy[0]:.2f}, '
            f'{planned_path.goal_xy[1]:.2f}), cost={planned_path.cost:.2f}, '
            f'utility={candidate.utility:.3f}.'
        )
        self.nav2.navigate(planned_path.goal_xy, yaw, self._navigation_finished)

    def _switch_gvd_to_gbsae(self, rx: float, ry: float) -> bool:
        topology = self._build_gvd_topology()
        component = robot_component_graph(topology.graph, (rx, ry))
        self.gvd_topology = topology
        if component.number_of_nodes() < 2 or component.number_of_edges() == 0:
            self.get_logger().warn(
                'GVD sweep threshold reached, but live skeleton graph is too small; '
                'continuing bootstrap.'
            )
            return False
        self.gbsae_planner = GBSAEPlanner(
            component,
            (rx, ry),
            self.gbsae_loop_path_cost_weight,
        )
        self.gvd_phase = 'gbsae'
        route = [step.vertex_id for step in self.gbsae_planner.route]
        self.get_logger().info(
            f'Switching GVD bootstrap to live GBSAE at swept={self.gvd_sweep.ratio:.1%}: '
            f'{component.number_of_nodes()} nodes, {component.number_of_edges()} edges, '
            f'{self._gvd_repair_log(topology)}, '
            f'route={route}.'
        )
        return True

    def _gvd_min_vertex_spacing_for_mode(self) -> float:
        if self.slam_mode == 'gvd_hierarchical':
            return self.gvd_hierarchical_min_vertex_spacing
        return self.gvd_min_vertex_spacing

    def _build_gvd_topology(self) -> GVDTopology:
        assert self.latest_map is not None
        assert self.latest_grid is not None
        assert self.gvd_bounds is not None
        return build_obstacle_gvd_topology(
            self.latest_grid,
            self._latest_map_geometry(),
            self.gvd_bounds,
            resolution=self.gvd_raster_resolution,
            clearance=self.gvd_obstacle_clearance,
            boundary_margin=self.gvd_boundary_margin,
            support_vertex_spacing=self.gvd_support_vertex_spacing,
            corner_turn_threshold=self.gvd_corner_turn_threshold,
            min_vertex_spacing=self._gvd_min_vertex_spacing_for_mode(),
            suppress_unknown_cycles=self.gvd_unknown_cycle_suppression_enabled,
            unconfident_unknown_radius=self.gvd_unconfident_unknown_radius,
            unconfident_unknown_ratio=self.gvd_unconfident_unknown_ratio,
            reconnection_clearance=self.gvd_reconnection_clearance,
            connection_cache=self.gvd_connection_cache,
            repair_connectivity=self.gvd_switching_connections_enabled,
            connection_neighbor_limit=self.gvd_connection_neighbor_limit,
            map_revision=self.gvd_map_generation,
        )

    def _gvd_repair_log(self, topology: GVDTopology) -> str:
        stats = topology.repair_stats
        if stats is None:
            return 'switching_connections=unavailable'
        return (
            f'switching_connections=gvd:{stats.gvd_edges},'
            f'astar:{stats.astar_edges},'
            f'components:{stats.unresolved_components},'
            f'cache:{self.gvd_connection_cache.hits}/'
            f'{self.gvd_connection_cache.misses}'
        )

    def _gvd_compression_log(self, topology: GVDTopology) -> str:
        stats = topology.compression_stats
        if stats is None:
            return 'vertex_clustering=unavailable'
        return f'vertex_clustering={stats.before_vertices}->{stats.after_vertices}'

    def _gvd_unknown_cycle_log(self, topology: GVDTopology) -> str:
        stats = topology.cycle_suppression_stats
        if stats is None:
            return 'unknown_cycle_suppression=unavailable'
        return (
            f'unknown_cycle_suppression=vertices:{stats.unconfident_vertices},'
            f'removed_edges:{stats.removed_edges}'
        )

    def _gvd_path_obstructed(self) -> bool:
        if (
            not self.gvd_active_path
            or self.latest_map is None
            or self.latest_grid is None
            or self.gvd_checked_map_generation == self.gvd_map_generation
        ):
            return False
        assert self.gvd_bounds is not None
        geometry, traversable = self._current_gvd_traversability()
        self.gvd_checked_map_generation = self.gvd_map_generation
        if self.gvd_active_traversability is None:
            self.gvd_active_traversability = traversable
            return False
        pose = self._get_robot_pose()
        if pose is None:
            return False
        forward_path = path_suffix_from_nearest(self.gvd_active_path, (pose[0], pose[1]))
        return path_crosses_new_obstacle(
            forward_path,
            geometry,
            self.gvd_active_traversability,
            traversable,
        )

    def _current_gvd_traversability(self):
        """Return the GVD raster after obstacle inflation with shared clearance."""
        assert self.latest_grid is not None
        assert self.gvd_bounds is not None
        return build_obstacle_traversability(
            self.latest_grid,
            self._latest_map_geometry(),
            self.gvd_bounds,
            resolution=self.gvd_raster_resolution,
            clearance=self.gvd_obstacle_clearance,
            boundary_margin=self.gvd_boundary_margin,
        )

    def _exploration_stuck(self, pose: Optional[Tuple[float, float, float]]) -> bool:
        """Return whether an active Nav2 motion has stopped translating."""
        if (
            not self.gvd_stuck_recovery_enabled
            or self.state not in self._translation_watchdog_states()
            or pose is None
        ):
            return False
        self._record_gvd_navigation_progress((pose[0], pose[1]))
        return progress_watchdog_expired(
            self.gvd_last_progress_wall_time,
            self.gvd_stuck_timeout,
            time.monotonic(),
        )

    def _translation_watchdog_states(self):
        return (
            self.NAVIGATING,
            self.ALIGNING_FRONTIER_PROBE,
            self.PROBING_FRONTIER,
        )

    def _start_gvd_stuck_recovery(self):
        self.get_logger().warn(
            f'Exploration has made no effective translation for '
            f'{self.gvd_stuck_timeout:.1f}s; starting bounded random-walk recovery.'
        )
        if self.state == self.NAVIGATING:
            self.nav2.cancel_navigation()
            self._mark_goal_failed(self.current_goal)
            self._clear_navigation()
        elif self.state == self.SELECTING:
            self.nav2.cancel_path_batch()
            self.selection_request_wall_time = None
            self.selection_request_goal = None
        elif self.state == self.ALIGNING_FRONTIER_PROBE:
            self.nav2.cancel_spin()
            self._mark_goal_failed(self.current_goal)
            self._clear_navigation()
        elif self.state == self.PROBING_FRONTIER:
            self.nav2.cancel_drive_on_heading()
            self._mark_goal_failed(self.current_goal)
            self._clear_navigation()
        self._start_gvd_random_recovery()

    def _start_gvd_progress_watchdog(self, anchor_xy: Tuple[float, float]):
        self.gvd_progress_anchor_xy = anchor_xy
        self.gvd_last_progress_wall_time = time.monotonic()

    def _record_gvd_navigation_progress(self, robot_xy: Tuple[float, float]):
        (
            self.gvd_progress_anchor_xy,
            self.gvd_last_progress_wall_time,
            _,
        ) = update_translation_progress(
            self.gvd_progress_anchor_xy,
            self.gvd_last_progress_wall_time,
            robot_xy,
            self.gvd_stuck_min_progress_distance,
            time.monotonic(),
        )

    def _start_gvd_random_recovery(self):
        self.gvd_random_recovery_attempt = 0
        self._start_next_gvd_random_recovery_attempt()

    def _start_next_gvd_random_recovery_attempt(self):
        if self.gvd_random_recovery_attempt >= self.gvd_random_recovery_attempts:
            self.get_logger().warn(
                'GVD random-walk recovery exhausted its bounded attempts; reselecting.'
            )
            self._clear_gvd_random_recovery()
            self._schedule_retry(0.0)
            return
        motion = sample_random_recovery_motion(
            self.gvd_random,
            min_abs_yaw=self.gvd_random_recovery_min_abs_yaw,
            max_abs_yaw=self.gvd_random_recovery_max_abs_yaw,
            distance=self.gvd_random_recovery_distance,
            speed=self.gvd_random_recovery_speed,
        )
        self.gvd_random_recovery_motion = motion
        self.gvd_random_recovery_action_start_wall_time = time.monotonic()
        self.state = self.RANDOM_RECOVERY_SPIN
        self.get_logger().info(
            f'GVD random-walk recovery attempt '
            f'{self.gvd_random_recovery_attempt + 1}/{self.gvd_random_recovery_attempts}: '
            f'Nav2 Spin delta={motion.yaw_delta:.2f}rad.'
        )
        self.nav2.spin_once(
            motion.yaw_delta,
            self.gvd_random_recovery_spin_timeout,
            self._gvd_random_recovery_spin_finished,
        )

    def _gvd_random_recovery_spin_finished(self, status: int):
        if self.state != self.RANDOM_RECOVERY_SPIN:
            return
        if status != GOAL_STATUS_SUCCEEDED:
            self._gvd_random_recovery_attempt_failed(
                f'random recovery spin ended with status={status}'
            )
            return
        assert self.gvd_random_recovery_motion is not None
        motion = self.gvd_random_recovery_motion
        self.state = self.RANDOM_RECOVERY_DRIVE
        self.gvd_random_recovery_action_start_wall_time = time.monotonic()
        self.get_logger().info(
            f'GVD random-walk recovery driving {motion.distance:.2f}m at '
            f'{motion.speed:.2f}m/s with Nav2 collision checking.'
        )
        self.nav2.drive_on_heading(
            motion.distance,
            motion.speed,
            self.gvd_random_recovery_drive_timeout,
            self._gvd_random_recovery_drive_finished,
        )

    def _gvd_random_recovery_drive_finished(self, status: int):
        if self.state != self.RANDOM_RECOVERY_DRIVE:
            return
        if status != GOAL_STATUS_SUCCEEDED:
            self._gvd_random_recovery_attempt_failed(
                f'random recovery drive ended with status={status}'
            )
            return
        self.get_logger().info('GVD random-walk recovery completed; reselecting.')
        self._clear_gvd_random_recovery()
        self._schedule_retry(0.0)

    def _gvd_random_recovery_attempt_failed(self, reason: str):
        self.get_logger().warn(f'GVD {reason}; trying another direction.')
        self.gvd_random_recovery_attempt += 1
        self._start_next_gvd_random_recovery_attempt()

    def _clear_gvd_random_recovery(self):
        self.gvd_random_recovery_action_start_wall_time = None
        self.gvd_random_recovery_motion = None
        self.gvd_progress_anchor_xy = None
        self.gvd_last_progress_wall_time = None

    def _handle_gvd_path_obstructed(self):
        """Replan to same goal first, then backtrack along original path.

        When the active GVD path is obstructed, instead of immediately
        abandoning the goal and selecting a new one, we first try to replan
        to the same goal from the current robot pose.  If that fails we
        walk backwards along the original planned path and navigate to the
        farthest checkpoint that Nav2 can still reach.
        """
        self.get_logger().warn(
            'Observed a new obstacle on the active GVD path; attempting replan.'
        )
        self.nav2.cancel_navigation()

        original_goal = self.current_goal
        original_path = self.gvd_active_path
        self._clear_navigation()

        pose = self._get_robot_pose()
        if pose is None or not original_path:
            if original_goal is not None:
                self._mark_goal_failed(original_goal)
            self._schedule_retry(0.0)
            return

        # Sample evenly-spaced checkpoints from the goal backwards towards
        # the start.  The first checkpoint (goal) gives the same-goal replan
        # attempt; subsequent ones implement the backtracking fallback.
        num_checkpoints = min(8, len(original_path))
        step = (
            max(1, (len(original_path) - 1) // (num_checkpoints - 1))
            if num_checkpoints > 1
            else 1
        )
        # Indices from goal (last) down to start (first)
        checkpoints = list(range(len(original_path) - 1, -1, -step))

        self._gvd_obstruction_original_goal = original_goal
        self._gvd_obstruction_path = original_path
        self._gvd_obstruction_start = (pose[0], pose[1])
        self._gvd_obstruction_checkpoints = checkpoints
        self._gvd_obstruction_checkpoint_idx = 0

        self.state = self.SELECTING
        self.selection_kind = 'gvd_obstruction_replan'
        self._gvd_obstruction_gen = self.nav2.start_path_batch()

        self._try_gvd_obstruction_checkpoint()

    def _try_gvd_obstruction_checkpoint(self):
        """Issue a Nav2 path request for the current backtrack checkpoint."""
        idx = self._gvd_obstruction_checkpoint_idx
        checkpoints = self._gvd_obstruction_checkpoints
        if idx >= len(checkpoints):
            self._finish_gvd_obstruction_fallback()
            return

        path_idx = checkpoints[idx]
        goal = self._gvd_obstruction_path[path_idx]

        self.selection_request_wall_time = time.monotonic()
        self.selection_request_goal = goal
        self.get_logger().info(
            f'GVD obstruction replan: checkpoint {idx + 1}/{len(checkpoints)} '
            f'(path index {path_idx}/{len(self._gvd_obstruction_path) - 1}): '
            f'({goal[0]:.2f}, {goal[1]:.2f})'
        )
        self.nav2.compute_path(
            self._gvd_obstruction_gen,
            self._gvd_obstruction_start,
            goal,
            self._gvd_obstruction_replan_computed,
        )

    def _gvd_obstruction_replan_computed(
        self, planned_path: Optional[PlannedPath]
    ):
        """Callback for obstruction-replan path requests.

        The first successful path terminates the search and the robot
        navigates there.  Failures advance to the next checkpoint
        (closer to the robot).  If every checkpoint fails we fall back
        to the original full-reselection behaviour.
        """
        if (
            self.state != self.SELECTING
            or self.selection_kind != 'gvd_obstruction_replan'
        ):
            return

        if planned_path is not None:
            checkpoint_idx = self._gvd_obstruction_checkpoint_idx
            checkpoints = self._gvd_obstruction_checkpoints
            path_idx = checkpoints[checkpoint_idx]
            self.get_logger().info(
                f'GVD obstruction replan: checkpoint {checkpoint_idx + 1}/'
                f'{len(checkpoints)} (path index {path_idx}) is reachable; '
                'navigating there.'
            )
            self._dispatch_gvd_obstruction_fallback(planned_path)
            return

        # This checkpoint is unreachable — try the next one (closer to robot).
        self._gvd_obstruction_checkpoint_idx += 1
        self._try_gvd_obstruction_checkpoint()

    def _dispatch_gvd_obstruction_fallback(self, planned_path: PlannedPath):
        """Navigate to the fallback point found during obstruction replan."""
        self.nav2.cancel_path_batch()
        self.current_goal = planned_path.goal_xy
        self.current_safe_goal = None
        self.target_cluster = None
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        self.current_navigation_kind = 'gvd_goal'
        self.gvd_active_path = tuple(planned_path.points)
        _, self.gvd_active_traversability = self._current_gvd_traversability()
        self.gvd_checked_map_generation = self.gvd_map_generation
        self.state = self.NAVIGATING
        self.navigation_start_wall_time = time.monotonic()
        anchor_xy = self._gvd_obstruction_start or planned_path.points[0]
        self._start_gvd_progress_watchdog(anchor_xy)
        yaw = (
            heading_to_target(planned_path.points[-2], planned_path.goal_xy)
            if len(planned_path.points) >= 2
            else 0.0
        )
        self.get_logger().info(
            f'Navigating to GVD obstruction fallback goal='
            f'({planned_path.goal_xy[0]:.2f}, {planned_path.goal_xy[1]:.2f}), '
            f'cost={planned_path.cost:.2f}.'
        )
        self._cleanup_gvd_obstruction_state()
        self.nav2.navigate(planned_path.goal_xy, yaw, self._navigation_finished)

    def _finish_gvd_obstruction_fallback(self):
        """No checkpoint on the original path is reachable — full reselection."""
        self.get_logger().warn(
            'GVD obstruction replan: no checkpoint on original path is '
            'reachable; falling back to full reselection.'
        )
        self.nav2.cancel_path_batch()
        if self._gvd_obstruction_original_goal is not None:
            self._mark_goal_failed(self._gvd_obstruction_original_goal)
        self._cleanup_gvd_obstruction_state()
        self._schedule_retry(0.0)

    def _cleanup_gvd_obstruction_state(self):
        """Clear the transient obstruction-replan bookkeeping."""
        self._gvd_obstruction_original_goal = None
        self._gvd_obstruction_path = ()
        self._gvd_obstruction_start = None
        self._gvd_obstruction_checkpoints = []
        self._gvd_obstruction_checkpoint_idx = 0

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
        self.selection_request_final_goal = None
        self.selection_request_is_frontier_staging = False
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
            self.get_logger().info(
                f'GBSAE prior vertex {step.vertex_id} already reached'
                f'{" during loop revisit" if step.loop_revisit else ""}.'
            )

        step = planner.active_step
        if step is None:
            self.get_logger().info(
                'GBSAE prior route completed; continuing with frontier coverage.'
            )
            self._start_standard_selection(rx, ry)
            return

        self.selection_start_xy = (rx, ry)
        self.failed_goals.expire(time.monotonic())
        target = vertex_point(planner.graph, step.vertex_id)
        if (
            self.latest_map is not None
            and self.latest_grid is not None
            and point_is_known_free(self.latest_map, self.latest_grid, target)
            and not self.failed_goals.contains(target, time.monotonic())
        ):
            self.selection_generation = self.nav2.start_path_batch()
            self.selection_kind = 'gbsae_vertex'
            self.selection_request_wall_time = time.monotonic()
            self.selection_request_goal = target
            self.state = self.SELECTING
            self.get_logger().info(
                f'Checking GBSAE prior vertex {step.vertex_id}'
                f'{" loop revisit" if step.loop_revisit else ""} '
                f'at ({target[0]:.2f}, {target[1]:.2f}).'
            )
            self.nav2.compute_path(
                self.selection_generation,
                self.selection_start_xy,
                target,
                self._gbsae_vertex_path_computed,
            )
            return
        if step.loop_revisit:
            self._skip_unreachable_loop_revisit('prior vertex is not currently known-free')
            self._schedule_retry(0.0)
            return
        self._start_gbsae_frontier_selection(rx, ry)

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
        self.selection_request_final_goal = None
        self.selection_request_is_frontier_staging = False
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
        if step.loop_revisit:
            self._skip_unreachable_loop_revisit('Nav2 could not plan the optional revisit')
            self._schedule_retry(0.0)
            return
        self._mark_goal_failed(vertex_point(planner.graph, step.vertex_id))
        self._start_gbsae_frontier_selection(*self.selection_start_xy)

    def _request_next_gbsae_frontier_path(self):
        if self.selection_cluster_index < len(self.selection_candidates):
            candidate = self.selection_candidates[self.selection_cluster_index]
            goal_xy, is_staging = self._frontier_request_goal(candidate)
            self.selection_request_wall_time = time.monotonic()
            self.selection_request_goal = goal_xy
            self.get_logger().info(
                f'Checking GBSAE allocated frontier source={candidate.cluster.source}, '
                f'information_gain={candidate.information_gain:.2f}m^2'
                f'{", staging first" if is_staging else ""}.'
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
            if self.selection_request_is_frontier_staging:
                self._dispatch_frontier_staging(candidate, planned_path)
                return
            self._dispatch_navigation(candidate, planned_path)
            return
        self._mark_goal_failed(self.selection_request_goal)
        if self.selection_request_is_frontier_staging:
            self._mark_goal_failed(self.selection_request_final_goal)
        self.selection_request_goal = None
        self.selection_cluster_index += 1
        self._request_next_gbsae_frontier_path()

    def _request_next_path(self):
        if self.selection_cluster_index < len(self.selection_candidates):
            candidate = self.selection_candidates[self.selection_cluster_index]
            goal_xy, is_staging = self._frontier_request_goal(candidate)
            self.selection_request_wall_time = time.monotonic()
            self.selection_request_goal = goal_xy
            if self.selection_kind == 'gvd_guide_warmup':
                self.get_logger().info(
                    f'Checking GVD guide warmup frontier source={candidate.cluster.source}, '
                    f'range_gain={candidate.range_gain:.2f}m^2, '
                    f'information_gain={candidate.information_gain:.2f}m^2, '
                    f'progress={self.gvd_guide_warmup_travel:.2f}/'
                    f'{self.gvd_guide_warmup_distance:.2f}m'
                    f'{", staging first" if is_staging else ""}.'
                )
            else:
                self.get_logger().info(
                    f'Checking shared frontier candidate source={candidate.cluster.source}, '
                    f'information_gain={candidate.information_gain:.2f}m^2, '
                    f'utility={candidate.utility:.3f}'
                    f'{", staging first" if is_staging else ""}.'
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

        if self.selection_kind == 'gvd_guide_warmup':
            self._finish_gvd_guide_warmup_early(
                'no reachable warmup frontier after Nav2 path checks'
            )
            self.nav2.cancel_path_batch()
            self._schedule_retry(0.0)
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
            if self.selection_request_is_frontier_staging:
                self._mark_goal_failed(self.selection_request_final_goal)
        elif self.selection_request_is_frontier_staging:
            self._dispatch_frontier_staging(candidate, planned_path)
            return
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
        self.selection_request_final_goal = None
        self.selection_request_is_frontier_staging = False
        warmup = self.selection_kind == 'gvd_guide_warmup'
        self.current_navigation_kind = 'warmup_frontier' if warmup else 'frontier'
        self.current_navigation_path_cost = planned_path.cost
        self.state = self.NAVIGATING
        self.navigation_start_wall_time = time.monotonic()
        self._start_gvd_progress_watchdog(self.selection_start_xy)
        yaw = heading_to_target(planned_path.goal_xy, (cluster.centroid_x, cluster.centroid_y))
        prefix = 'GVD guide warmup frontier' if warmup else 'frontier'
        range_note = f', range_gain={candidate.range_gain:.2f}m^2' if warmup else ''
        self.get_logger().info(
            f'Navigating to {prefix} goal=({planned_path.goal_xy[0]:.2f}, '
            f'{planned_path.goal_xy[1]:.2f}), size={cluster.size}, '
            f'cost={planned_path.cost:.2f}, source={cluster.source}, '
            f'information_gain={candidate.information_gain:.2f}m^2{range_note}'
        )
        self.nav2.navigate(planned_path.goal_xy, yaw, self._navigation_finished)

    def _dispatch_frontier_staging(
        self,
        candidate: FrontierCandidate,
        planned_path: PlannedPath,
    ):
        self.nav2.cancel_path_batch()
        final_goal = self.selection_request_final_goal or candidate.safe_goal.point
        self.current_goal = planned_path.goal_xy
        self.current_safe_goal = None
        self.target_cluster = candidate.cluster
        self.frontier_staging_final_goal = final_goal
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        self.selection_request_final_goal = None
        self.selection_request_is_frontier_staging = False
        warmup = self.selection_kind == 'gvd_guide_warmup'
        self.current_navigation_kind = (
            'warmup_frontier_staging' if warmup else 'frontier_staging'
        )
        self.current_navigation_path_cost = planned_path.cost
        self.state = self.NAVIGATING
        self.navigation_start_wall_time = time.monotonic()
        self._start_gvd_progress_watchdog(self.selection_start_xy)
        yaw = heading_to_target(planned_path.goal_xy, final_goal)
        prefix = 'GVD guide warmup frontier staging' if warmup else 'frontier staging'
        self.get_logger().info(
            f'Navigating to {prefix} goal=({planned_path.goal_xy[0]:.2f}, '
            f'{planned_path.goal_xy[1]:.2f}) toward final=({final_goal[0]:.2f}, '
            f'{final_goal[1]:.2f}), cost={planned_path.cost:.2f}, '
            f'source={candidate.cluster.source}.'
        )
        self.nav2.navigate(planned_path.goal_xy, yaw, self._navigation_finished)

    def _dispatch_gbsae_vertex_navigation(self, planned_path: PlannedPath, vertex_id: int):
        self.nav2.cancel_path_batch()
        self.current_goal = planned_path.goal_xy
        self.current_safe_goal = None
        self.target_cluster = None
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        self.current_navigation_kind = 'gbsae_vertex'
        self.state = self.NAVIGATING
        self.navigation_start_wall_time = time.monotonic()
        self._start_gvd_progress_watchdog(self.selection_start_xy)
        planner = self.gbsae_planner
        assert planner is not None
        step = planner.active_step
        revisit = step is not None and step.loop_revisit
        self.get_logger().info(
            f'Navigating directly to GBSAE prior vertex {vertex_id}'
            f'{" loop revisit" if revisit else ""}: '
            f'goal=({planned_path.goal_xy[0]:.2f}, {planned_path.goal_xy[1]:.2f}), '
            f'cost={planned_path.cost:.2f}.'
        )
        self.nav2.navigate(planned_path.goal_xy, 0.0, self._navigation_finished)

    def _navigation_finished(self, status: int):
        if self.state != self.NAVIGATING:
            return
        if self._finish_gvd_guide_navigation(status):
            return
        if self.current_navigation_kind == 'hierarchical_gvd_vertex':
            target = self.gvd_hierarchical_target
            if status == GOAL_STATUS_SUCCEEDED and target is not None:
                tracker = self.gvd_hierarchical_tracker
                reached_vertex = tracker.mark_reached_point(target.point)
                if reached_vertex is None:
                    tracker.mark_route_dirty()
                    self.gvd_phase = 'macro'
                    self.get_logger().warn(
                        'Nav2 reached a hierarchical GVD macro goal that no longer '
                        'maps to the rebuilt topology; replanning.'
                    )
                else:
                    self.gvd_phase = (
                        'local_clear'
                        if tracker.should_clear_local(reached_vertex)
                        else 'macro'
                    )
                    self.get_logger().info(
                        f'Nav2 reached hierarchical GVD macro vertex={reached_vertex}; '
                        f'degree={tracker.graph.degree(reached_vertex)}, '
                        f'next_phase={self.gvd_phase}.'
                    )
            else:
                self.get_logger().warn(
                    f'Nav2 hierarchical GVD macro navigation ended with status={status}; '
                    'retrying.'
                )
                self._mark_goal_failed(self.current_goal)
                self.gvd_hierarchical_tracker.mark_route_dirty()
            self.gvd_hierarchical_target = None
            self._clear_navigation()
            self._schedule_retry(0.0)
            return
        if self.current_navigation_kind == 'gvd_goal':
            if status == GOAL_STATUS_SUCCEEDED:
                self.get_logger().info('Nav2 reached the active GVD bootstrap goal.')
            else:
                self.get_logger().warn(
                    f'Nav2 GVD bootstrap navigation ended with status={status}; retrying.'
                )
                self._mark_goal_failed(self.current_goal)
            self._clear_navigation()
            self._schedule_retry(0.0)
            return
        if self.current_navigation_kind == 'gbsae_vertex':
            if status == GOAL_STATUS_SUCCEEDED:
                planner = self.gbsae_planner
                assert planner is not None
                step = planner.advance_active_step()
                assert step is not None
                self.get_logger().info(
                    f'Nav2 reached GBSAE prior vertex {step.vertex_id}'
                    f'{" during loop revisit" if step.loop_revisit else ""}.'
                )
            else:
                self._handle_navigation_failure(
                    f'Nav2 GBSAE navigation ended with status={status}'
                )
            self._clear_navigation()
            self._schedule_retry(0.0)
            return
        if self.current_navigation_kind in ('frontier_staging', 'warmup_frontier_staging'):
            if status == GOAL_STATUS_SUCCEEDED:
                if self.current_navigation_kind == 'warmup_frontier_staging':
                    self._record_gvd_guide_warmup_progress()
                self.get_logger().info(
                    'Nav2 reached a frontier staging waypoint; continuing '
                    'toward the final frontier on the next selection pass.'
                )
            else:
                self.get_logger().warn(
                    f'Nav2 frontier staging navigation ended with status={status}; '
                    'temporarily cooling the final frontier.'
                )
                self._mark_goal_failed(self.current_goal)
                self._mark_goal_failed(self.frontier_staging_final_goal)
            self._clear_navigation()
            self._schedule_retry(0.0)
            return
        if status == GOAL_STATUS_SUCCEEDED:
            if self.current_navigation_kind == 'warmup_frontier':
                self._record_gvd_guide_warmup_progress()
            self.get_logger().info('Nav2 reached the active frontier goal.')
            if self._frontier_probe_settings() is not None:
                self._start_frontier_probe()
                return
        else:
            self.get_logger().warn(f'Nav2 navigation ended with status={status}; retrying.')
            self._mark_goal_failed(self.current_goal)
        self._clear_navigation()
        self._schedule_retry(0.0)

    def _record_gvd_guide_warmup_progress(self, distance: Optional[float] = None):
        if self.slam_mode != 'gvd_guide' or self.gvd_guide_warmup_distance <= 0.0:
            return
        before = self.gvd_guide_warmup_travel
        traveled = self.current_navigation_path_cost if distance is None else distance
        self.gvd_guide_warmup_travel += max(0.0, float(traveled))
        target = self.gvd_guide_warmup_distance
        if before < target <= self.gvd_guide_warmup_travel:
            if not self.gvd_guide_warmup_complete_logged:
                self.gvd_guide_warmup_complete_logged = True
                self.get_logger().info(
                    f'GVD guide warmup completed after '
                    f'{self.gvd_guide_warmup_travel:.2f}m; starting sparse GVD planning.'
                )
        else:
            self.get_logger().info(
                f'GVD guide warmup progress '
                f'{self.gvd_guide_warmup_travel:.2f}/{target:.2f}m.'
            )

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

        self.frontier_probe_parent_kind = self.current_navigation_kind
        self.frontier_probe_normal = normal
        target_yaw = math.atan2(normal[1], normal[0])
        yaw_delta = normalize_angle(target_yaw - pose[2])
        if abs(yaw_delta) <= self.frontier_probe_spin_tolerance:
            self._start_frontier_drive()
            return

        self.state = self.ALIGNING_FRONTIER_PROBE
        self.frontier_probe_action_start_wall_time = time.monotonic()
        self._start_gvd_progress_watchdog((pose[0], pose[1]))
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
        self.frontier_probe_drive_distance = distance
        self.state = self.PROBING_FRONTIER
        self.frontier_probe_action_start_wall_time = time.monotonic()
        pose = self._get_robot_pose()
        if pose is not None:
            self._start_gvd_progress_watchdog((pose[0], pose[1]))
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
        if self.frontier_probe_parent_kind == 'warmup_frontier':
            self._record_gvd_guide_warmup_progress(
                self.frontier_probe_drive_distance
            )
        self._clear_navigation()
        self._schedule_retry(0.0)

    def _frontier_probe_settings(self) -> Optional[Tuple[float, float, float]]:
        warmup_frontier_probe = (
            self.slam_mode == 'gvd_guide'
            and self.current_navigation_kind == 'warmup_frontier'
        )
        if (
            self.target_cluster is None
            or not frontier_probes_enabled_for_mode(
                self.slam_mode,
                frontier_modes_enabled=self.frontier_mode_probes_enabled,
                gvd_modes_enabled=self.gvd_mode_probes_enabled,
                hierarchical_local_cleanup=(
                    self.slam_mode == 'gvd_hierarchical'
                    and self.gvd_phase == 'local_clear'
                ),
                hierarchical_local_cleanup_enabled=(
                    self.gvd_hierarchical_local_probes_enabled
                ),
                gvd_guide_warmup=warmup_frontier_probe,
            )
        ):
            return None
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
        self.frontier_staging_final_goal = None
        self.navigation_start_wall_time = None
        self.frontier_probe_normal = None
        self.frontier_probe_action_start_wall_time = None
        self.frontier_probe_parent_kind = 'frontier'
        self.frontier_probe_drive_distance = 0.0
        self.gvd_guide_active_step = None
        self.gvd_active_path = ()
        self.gvd_active_traversability = None
        self.gvd_progress_anchor_xy = None
        self.gvd_last_progress_wall_time = None
        self.current_navigation_kind = 'frontier'
        self.current_navigation_path_cost = 0.0

    def _handle_navigation_failure(self, reason: str):
        if self._skip_unreachable_loop_revisit(reason):
            return
        self.get_logger().warn(f'{reason}; retrying.')
        self._mark_goal_failed(self.current_goal)

    def _skip_unreachable_loop_revisit(self, reason: str) -> bool:
        if self.slam_mode not in ('gbsae', 'gvd_gbsae') or self.gbsae_planner is None:
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

    def _schedule_retry(self, delay: Optional[float] = None):
        self.state = self.IDLE
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        self.selection_request_final_goal = None
        self.selection_request_is_frontier_staging = False
        self.gvd_progress_anchor_xy = None
        self.gvd_last_progress_wall_time = None
        self.next_retry_wall_time = time.monotonic() + (
            self.frontier_retry_interval if delay is None else delay
        )

    def _frontier_request_goal(
        self,
        candidate: FrontierCandidate,
    ) -> Tuple[Tuple[float, float], bool]:
        """Return the next Nav2 goal, splitting far frontier targets into stages."""
        final_goal = candidate.safe_goal.point
        staging_goal = self._frontier_staging_goal(self.selection_start_xy, final_goal)
        is_staging = staging_goal is not None
        self.selection_request_final_goal = final_goal
        self.selection_request_is_frontier_staging = is_staging
        return (staging_goal if staging_goal is not None else final_goal), is_staging

    def _sample_gvd_guide_waypoint_goal(
        self,
        original_goal: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Randomly jitter a guide waypoint onto a nearby Nav2-friendlier cell."""
        sample_count = max(0, int(self.gvd_guide_waypoint_sample_count))
        sigma = max(0.0, float(self.gvd_guide_waypoint_sample_sigma))
        max_radius = max(0.0, float(self.gvd_guide_waypoint_sample_max_radius))
        if (
            sample_count <= 0
            or sigma <= 0.0
            or max_radius <= 0.0
            or self.latest_map is None
            or self.latest_grid is None
        ):
            return original_goal

        geometry = self._latest_map_geometry()
        if geometry.resolution <= 0.0:
            return original_goal
        search_config = SafeGoalSearchConfig(
            search_radius=self.frontier_goal_search_radius,
            clearance=self.frontier_goal_clearance,
            standoff=self.frontier_goal_standoff,
            map_edge_clearance=self.frontier_goal_map_edge_clearance,
            reach_radius=self.nav2_goal_reach_radius,
            point_sample_limit=self.frontier_goal_point_sample_limit,
        )
        valid_mask = prepare_safe_goal_grid(
            self.latest_grid,
            geometry,
            search_config,
        ).valid_goal_mask

        candidates = []
        attempts = max(sample_count * 3, sample_count)
        for _ in range(attempts):
            if len(candidates) >= sample_count:
                break
            dx = random.gauss(0.0, sigma)
            dy = random.gauss(0.0, sigma)
            distance = math.hypot(dx, dy)
            if distance > max_radius:
                scale = max_radius / distance
                dx *= scale
                dy *= scale
            point = (original_goal[0] + dx, original_goal[1] + dy)
            if not self._point_is_valid_goal_cell(point, valid_mask, geometry):
                continue
            if any(math.dist(point, existing) <= geometry.resolution for existing in candidates):
                continue
            candidates.append(point)

        if not candidates:
            return original_goal
        selected = random.choice(candidates)
        if math.dist(selected, original_goal) > 1e-6:
            self.get_logger().info(
                f'GVD guide sampled waypoint around ({original_goal[0]:.2f}, '
                f'{original_goal[1]:.2f}): selected=({selected[0]:.2f}, '
                f'{selected[1]:.2f}), candidates={len(candidates)}.'
            )
        return selected

    def _point_is_valid_goal_cell(
        self,
        point: Tuple[float, float],
        valid_mask: np.ndarray,
        geometry: GridGeometry,
    ) -> bool:
        cell = world_to_grid(point, geometry)
        return cell is not None and bool(valid_mask[cell])

    def _frontier_staging_goal(
        self,
        start_xy: Tuple[float, float],
        final_goal: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        max_distance = float(self.frontier_nav2_segment_distance)
        if (
            max_distance <= 0.0
            or not math.isfinite(max_distance)
            or math.dist(start_xy, final_goal) <= max_distance
            or self.latest_map is None
            or self.latest_grid is None
        ):
            return None

        geometry = self._latest_map_geometry()
        if geometry.resolution <= 0.0:
            return None
        search_config = SafeGoalSearchConfig(
            search_radius=self.frontier_goal_search_radius,
            clearance=self.frontier_goal_clearance,
            standoff=self.frontier_goal_standoff,
            map_edge_clearance=self.frontier_goal_map_edge_clearance,
            reach_radius=self.nav2_goal_reach_radius,
            point_sample_limit=self.frontier_goal_point_sample_limit,
        )
        prepared_grid = prepare_safe_goal_grid(
            self.latest_grid,
            geometry,
            search_config,
        )
        valid_mask = prepared_grid.valid_goal_mask
        start_cell = self._nearest_valid_goal_cell(
            valid_mask,
            geometry,
            start_xy,
            max(
                self.frontier_goal_search_radius,
                self.nav2_goal_reach_radius + geometry.resolution,
                self.frontier_goal_clearance + 2.0 * geometry.resolution,
            ),
        )
        goal_cell = self._nearest_valid_goal_cell(
            valid_mask,
            geometry,
            final_goal,
            max(
                self.frontier_goal_search_radius,
                self.frontier_goal_clearance + 2.0 * geometry.resolution,
            ),
        )
        if start_cell is None or goal_cell is None:
            return None

        path_cells = astar_path(
            valid_mask,
            np.zeros_like(valid_mask, dtype=bool),
            start_cell,
            goal_cell,
            1.0,
            1.0,
        )
        if len(path_cells) < 2:
            return None
        selected = self._frontier_staging_point_from_path(
            path_cells,
            geometry,
            start_xy,
            final_goal,
            max_distance,
        )
        if selected is None:
            return None
        self.get_logger().info(
            f'Splitting far frontier target: final=({final_goal[0]:.2f}, '
            f'{final_goal[1]:.2f}), staging=({selected[0]:.2f}, '
            f'{selected[1]:.2f}), straight_distance={math.dist(start_xy, final_goal):.2f}m, '
            f'segment_limit={max_distance:.2f}m.'
        )
        return selected

    def _frontier_staging_point_from_path(
        self,
        path_cells,
        geometry: GridGeometry,
        start_xy: Tuple[float, float],
        final_goal: Tuple[float, float],
        max_distance: float,
    ) -> Optional[Tuple[float, float]]:
        path_points = [grid_to_world(cell, geometry) for cell in path_cells]
        cumulative = 0.0
        previous = path_points[0]
        selected = None
        for point in path_points[1:]:
            cumulative += math.dist(previous, point)
            previous = point
            if cumulative > max_distance:
                break
            if math.dist(start_xy, point) > self.nav2_goal_reach_radius:
                selected = point
        if selected is None:
            return None
        if math.dist(selected, final_goal) <= self.nav2_goal_reach_radius:
            return None
        return selected

    def _nearest_valid_goal_cell(
        self,
        valid_mask: np.ndarray,
        geometry: GridGeometry,
        point: Tuple[float, float],
        max_radius: float,
    ):
        cell = world_to_grid(point, geometry)
        if cell is None:
            return None
        if valid_mask[cell]:
            return cell
        radius_cells = max(1, int(math.ceil(max_radius / geometry.resolution)))
        row, col = cell
        row0 = max(0, row - radius_cells)
        row1 = min(geometry.height, row + radius_cells + 1)
        col0 = max(0, col - radius_cells)
        col1 = min(geometry.width, col + radius_cells + 1)
        local = np.argwhere(valid_mask[row0:row1, col0:col1])
        if local.size == 0:
            return None
        candidates = [
            (int(row0 + local_row), int(col0 + local_col))
            for local_row, local_col in local
        ]
        return min(
            candidates,
            key=lambda candidate: (
                math.dist(grid_to_world(candidate, geometry), point),
                candidate[0],
                candidate[1],
            ),
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
        *,
        clusters=None,
        local_cleanup: bool = False,
        allowed_goal_mask: Optional[np.ndarray] = None,
        greedy_range_gain: bool = False,
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
        cluster_list = list(self.frontier_clusters if clusters is None else clusters)
        range_gains = (
            frontier_range_gains(data, geometry.resolution, cluster_list)
            if greedy_range_gain
            else [0.0 for _ in cluster_list]
        )
        candidates = []
        for cluster, range_gain in zip(cluster_list, range_gains):
            candidate = self._candidate_for_cluster(
                cluster,
                rx,
                ry,
                now,
                data,
                geometry,
                search_config,
                prepared_grid,
                allowed_goal_mask,
                range_gain=range_gain,
            )
            if candidate is not None:
                candidates.append(candidate)
        if greedy_range_gain:
            return ranked_range_gain_candidates(candidates, rx, ry, limit)
        if local_cleanup:
            return ranked_local_cleanup_candidates(candidates, rx, ry, limit)
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
        allowed_goal_mask: Optional[np.ndarray] = None,
        range_gain: float = 0.0,
    ) -> Optional[FrontierCandidate]:
        goal = select_safe_frontier_goal(
            data,
            geometry,
            cluster.cells,
            (rx, ry),
            search_config,
            prepared_grid,
            allowed_goal_mask,
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
            range_gain=range_gain,
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
        elif (
            (
                self.slam_mode == 'gvd_hierarchical'
                or self.slam_mode == 'gvd_guide'
                or (
                    self.slam_mode == 'gvd_gbsae'
                    and self.gvd_phase == 'bootstrap'
                )
            )
            and self.gvd_topology is not None
            and self.gvd_bounds is not None
        ):
            markers = gvd_to_marker_array(
                self.gvd_topology,
                self.gvd_bounds,
                self.gvd_active_path,
                'map',
                self.get_clock().now().to_msg(),
                self.gvd_hierarchical_tracker
                if self.slam_mode == 'gvd_hierarchical'
                else None,
                self.gvd_hierarchical_local_mask
                if self.slam_mode == 'gvd_hierarchical'
                else None,
                self.gvd_hierarchical_local_geometry
                if self.slam_mode == 'gvd_hierarchical'
                else None,
                self.gvd_hierarchical_cleared_region_outlines
                if self.slam_mode == 'gvd_hierarchical'
                else (),
                self.gvd_guide_planner
                if self.slam_mode == 'gvd_guide'
                else None,
            )
            self.pose_graph_pub.publish(markers)
        elif self.slam_mode in ('gbsae', 'gvd_gbsae') and self.gbsae_planner is not None:
            markers = gbsae_to_marker_array(
                self.gbsae_planner,
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

    def _latest_map_geometry(self) -> GridGeometry:
        info = self.latest_map.info
        return GridGeometry(
            origin_x=info.origin.position.x,
            origin_y=info.origin.position.y,
            resolution=info.resolution,
            width=info.width,
            height=info.height,
        )

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

    def _log_waiting_for_nav2(self):
        now = time.monotonic()
        if now - self.last_wait_log_wall_time > 5.0:
            self.get_logger().info('Waiting for Nav2 action servers.')
            self.last_wait_log_wall_time = now

    def destroy_node(self):
        self.nav2.destroy()
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
