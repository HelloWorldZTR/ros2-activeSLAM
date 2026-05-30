import math
import time
from collections import deque
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
)
from .frontier_selection import (
    OPEN_EDGE_FRONTIER,
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
    INITIAL_SPIN = 'initial_spin'
    IDLE = 'idle'
    SELECTING = 'selecting'
    NAVIGATING = 'navigating'
    ALIGNING_OPEN_EDGE = 'aligning_open_edge'
    PROBING_OPEN_EDGE = 'probing_open_edge'
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
        self.exploration_strategy = self.declare_parameter(
            'exploration_strategy', 'frontier'
        ).value
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
        self.frontier_goal_min_advance = self.declare_parameter(
            'frontier_goal_min_advance', 0.35
        ).value
        self.frontier_goal_point_sample_limit = int(
            self.declare_parameter('frontier_goal_point_sample_limit', 40).value
        )
        self.frontier_information_gain_radius = self.declare_parameter(
            'frontier_information_gain_radius', 1.0
        ).value
        self.frontier_information_gain_min = self.declare_parameter(
            'frontier_information_gain_min', 0.45
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
        self.frontier_open_edge_normal_radius = self.declare_parameter(
            'frontier_open_edge_normal_radius', 0.35
        ).value
        self.frontier_open_edge_spin_tolerance = self.declare_parameter(
            'frontier_open_edge_spin_tolerance', 0.10
        ).value
        self.frontier_open_edge_spin_timeout = self.declare_parameter(
            'frontier_open_edge_spin_timeout', 8.0
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

        if self.exploration_strategy not in ('frontier', 'graph', 'graph_based'):
            self.get_logger().warn(
                f'Unknown exploration_strategy={self.exploration_strategy}. '
                'Falling back to frontier.'
            )
            self.exploration_strategy = 'frontier'
        if self.exploration_strategy == 'graph_based':
            self.exploration_strategy = 'graph'

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

        # --- Exploration state ---
        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_grid: Optional[np.ndarray] = None
        self.frontier_clusters: List[FrontierCluster] = []
        self.current_goal: Optional[Tuple[float, float]] = None
        self.current_safe_goal: Optional[SafeFrontierGoal] = None
        self.target_cluster: Optional[FrontierCluster] = None
        self.open_edge_normal: Optional[Tuple[float, float]] = None
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
        self.navigation_start_wall_time: Optional[float] = None
        self.open_edge_action_start_wall_time: Optional[float] = None
        self.failed_goals = FailedGoalCooldown(
            self.failed_goal_cooldown,
            self.failed_goal_radius,
        )
        self.last_wait_log_wall_time = 0.0

        self.control_timer = self.create_timer(0.2, self._control_loop)

        self.get_logger().info(
            f'Exploration coordinator started with Nav2 backend. '
            f'Strategy: {self.exploration_strategy}'
        )
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
                    'canceling and selecting another frontier.'
                )
                self._mark_goal_failed(self.current_goal)
                self.nav2.cancel_navigation()
                self._clear_navigation()
                self._schedule_retry(0.0)
            return
        if self.state == self.ALIGNING_OPEN_EDGE:
            if navigation_timed_out(
                self.open_edge_action_start_wall_time,
                self.frontier_open_edge_spin_timeout,
                time.monotonic(),
            ):
                self.get_logger().warn('Open-edge alignment spin timed out; retrying.')
                self.nav2.cancel_spin()
                self._open_edge_probe_failed()
            return
        if self.state == self.PROBING_OPEN_EDGE:
            if navigation_timed_out(
                self.open_edge_action_start_wall_time,
                self.frontier_open_edge_probe_timeout,
                time.monotonic(),
            ):
                self.get_logger().warn('Open-edge DriveOnHeading timed out; retrying.')
                self.nav2.cancel_drive_on_heading()
                self._open_edge_probe_failed()
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
                self._schedule_retry(0.0)
            return
        if time.monotonic() < self.next_retry_wall_time or pose is None:
            return
        self._start_selection(pose[0], pose[1])

    # ------------------------------------------------------------------
    # Nav2 action orchestration
    # ------------------------------------------------------------------

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
            if self.exploration_strategy == 'graph'
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
        elif self.exploration_strategy == 'frontier':
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

    def _navigation_finished(self, status: int):
        if self.state != self.NAVIGATING:
            return
        if status == GOAL_STATUS_SUCCEEDED:
            self.get_logger().info('Nav2 reached the active frontier goal.')
            if (
                self.frontier_open_edge_probe_enabled
                and self.target_cluster is not None
                and self.target_cluster.source == OPEN_EDGE_FRONTIER
            ):
                self._start_open_edge_probe()
                return
        else:
            self.get_logger().warn(f'Nav2 navigation ended with status={status}; retrying.')
            self._mark_goal_failed(self.current_goal)
        self._clear_navigation()
        self._schedule_retry(0.0)

    def _start_open_edge_probe(self):
        if (
            self.latest_map is None
            or self.target_cluster is None
            or self.current_safe_goal is None
        ):
            self._open_edge_probe_failed()
            return

        info = self.latest_map.info
        normal = open_edge_outward_normal(
            self.target_cluster.cells,
            self.current_safe_goal.seed,
            GridGeometry(
                origin_x=info.origin.position.x,
                origin_y=info.origin.position.y,
                resolution=info.resolution,
                width=info.width,
                height=info.height,
            ),
            self.frontier_open_edge_normal_radius,
        )
        pose = self._get_robot_pose()
        if normal is None or pose is None:
            self.get_logger().warn('Cannot estimate open-edge probe direction; retrying.')
            self._open_edge_probe_failed()
            return

        self.open_edge_normal = normal
        target_yaw = math.atan2(normal[1], normal[0])
        yaw_delta = normalize_angle(target_yaw - pose[2])
        if abs(yaw_delta) <= self.frontier_open_edge_spin_tolerance:
            self._start_open_edge_drive()
            return

        self.state = self.ALIGNING_OPEN_EDGE
        self.open_edge_action_start_wall_time = time.monotonic()
        self.get_logger().info(
            f'Aligning with open map edge normal using Nav2 Spin: delta={yaw_delta:.2f} rad.'
        )
        self.nav2.spin_once(
            yaw_delta,
            self.frontier_open_edge_spin_timeout,
            self._open_edge_spin_finished,
        )

    def _open_edge_spin_finished(self, status: int):
        if self.state != self.ALIGNING_OPEN_EDGE:
            return
        if status != GOAL_STATUS_SUCCEEDED:
            self.get_logger().warn(
                f'Open-edge alignment spin ended with status={status}; retrying.'
            )
            self._open_edge_probe_failed()
            return
        self._start_open_edge_drive()

    def _start_open_edge_drive(self):
        self.state = self.PROBING_OPEN_EDGE
        self.open_edge_action_start_wall_time = time.monotonic()
        self.get_logger().info(
            f'Probing beyond open map edge with Nav2 DriveOnHeading: '
            f'distance={self.frontier_open_edge_probe_distance:.2f}m, '
            f'speed={self.frontier_open_edge_probe_speed:.2f}m/s.'
        )
        self.nav2.drive_on_heading(
            self.frontier_open_edge_probe_distance,
            self.frontier_open_edge_probe_speed,
            self.frontier_open_edge_probe_timeout,
            self._open_edge_drive_finished,
        )

    def _open_edge_drive_finished(self, status: int):
        if self.state != self.PROBING_OPEN_EDGE:
            return
        if status != GOAL_STATUS_SUCCEEDED:
            self.get_logger().warn(f'Open-edge probe ended with status={status}; retrying.')
            self._open_edge_probe_failed()
            return
        self.get_logger().info('Open-edge Nav2 probe completed.')
        self._clear_navigation()
        self._schedule_retry(0.0)

    def _open_edge_probe_failed(self):
        self._mark_goal_failed(self.current_goal)
        self._clear_navigation()
        self._schedule_retry(0.0)

    def _clear_navigation(self):
        self.current_goal = None
        self.current_safe_goal = None
        self.target_cluster = None
        self.navigation_start_wall_time = None
        self.open_edge_normal = None
        self.open_edge_action_start_wall_time = None

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
            min_advance=self.frontier_goal_min_advance,
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
            self.frontier_information_gain_min,
            on_cooldown=self.failed_goals.contains(goal.point, now),
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
        if self.exploration_strategy == 'graph':
            markers = graph_to_marker_array(
                self.pose_graph_tracker.graph,
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
