import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .frontier_detector import FrontierCluster, FrontierDetector
from .graph_exploration import (
    ApproximatePoseGraphTracker,
    GraphBasedFrontierScorer,
    graph_to_marker_array,
    make_information_matrix,
)


def _yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _quaternion_from_yaw(yaw: float):
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class ExplorationCoordinator(Node):
    def __init__(self):
        super().__init__('exploration_coordinator')

        # --- Exploration parameters ---
        self.replan_interval = self.declare_parameter('replan_interval', 3.0).value
        self.goal_reached_distance = self.declare_parameter('goal_reached_distance', 0.35).value
        self.stability_duration = self.declare_parameter('stability_duration', 10.0).value
        self.stability_threshold = self.declare_parameter('stability_threshold', 0.02).value
        self.min_frontier_size = self.declare_parameter('min_frontier_size', 5).value
        self.frontier_detection_interval = self.declare_parameter(
            'frontier_detection_interval', 1.0
        ).value
        self.rrt_frontier_iterations = int(
            self.declare_parameter('rrt_frontier_iterations', 800).value
        )
        self.rrt_frontier_step_size = self.declare_parameter(
            'rrt_frontier_step_size', 1.0
        ).value
        self.rrt_frontier_cluster_radius = self.declare_parameter(
            'rrt_frontier_cluster_radius', 0.6
        ).value
        self.rrt_frontier_max_points = int(
            self.declare_parameter('rrt_frontier_max_points', 120).value
        )
        self.rrt_frontier_map_padding = self.declare_parameter(
            'rrt_frontier_map_padding', 2.0
        ).value
        self.exploration_strategy = self.declare_parameter('exploration_strategy', 'frontier').value
        self.frontier_goal_candidates = int(
            self.declare_parameter('frontier_goal_candidates', 5).value
        )
        self.frontier_goal_search_radius = self.declare_parameter(
            'frontier_goal_search_radius', 1.2
        ).value
        self.frontier_goal_clearance = self.declare_parameter(
            'frontier_goal_clearance', 0.2
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
        self.failed_goal_cooldown = self.declare_parameter(
            'failed_goal_cooldown', 20.0
        ).value
        self.failed_goal_radius = self.declare_parameter('failed_goal_radius', 0.6).value
        self.nav2_goal_timeout = self.declare_parameter('nav2_goal_timeout', 30.0).value
        self.nav2_wait_log_interval = self.declare_parameter(
            'nav2_wait_log_interval', 5.0
        ).value
        self.completion_frontierless_scans = int(
            self.declare_parameter('completion_frontierless_scans', 5).value
        )
        self.completion_grace_period = self.declare_parameter(
            'completion_grace_period', 20.0
        ).value
        self.completion_open_edge_min_cells = int(
            self.declare_parameter('completion_open_edge_min_cells', 3).value
        )
        self.frontier_probe_enabled = self.declare_parameter(
            'frontier_probe_enabled', True
        ).value
        self.frontier_probe_speed = self.declare_parameter(
            'frontier_probe_speed', 0.08
        ).value
        self.frontier_probe_obstacle_distance = self.declare_parameter(
            'frontier_probe_obstacle_distance', 0.35
        ).value
        self.frontier_probe_timeout = self.declare_parameter(
            'frontier_probe_timeout', 12.0
        ).value
        self.frontier_probe_map_growth_cells = int(
            self.declare_parameter('frontier_probe_map_growth_cells', 30).value
        )
        self.frontier_probe_start_max_map_growth_cells = int(
            self.declare_parameter('frontier_probe_start_max_map_growth_cells', 15).value
        )
        self.frontier_probe_kp_angular = self.declare_parameter(
            'frontier_probe_kp_angular', 1.0
        ).value
        self.frontier_probe_max_angular_speed = self.declare_parameter(
            'frontier_probe_max_angular_speed', 0.6
        ).value

        # --- Graph-based scoring parameters ---
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
        self.graph_frontier_weight = self.declare_parameter(
            'graph_frontier_weight', 0.001
        ).value
        self.graph_odom_cov_x = self.declare_parameter('graph_odom_cov_x', 0.04).value
        self.graph_odom_cov_y = self.declare_parameter('graph_odom_cov_y', 0.04).value
        self.graph_odom_cov_yaw = self.declare_parameter('graph_odom_cov_yaw', 0.008).value

        if self.exploration_strategy not in ('frontier', 'graph', 'graph_based'):
            self.get_logger().warn(
                f'Unknown exploration_strategy={self.exploration_strategy}. Falling back to frontier.'
            )
            self.exploration_strategy = 'frontier'
        if self.exploration_strategy == 'graph_based':
            self.exploration_strategy = 'graph'

        # --- Publishers ---
        self.goal_pub = self.create_publisher(Marker, '/goal_point', 10)
        self.frontier_pub = self.create_publisher(MarkerArray, '/frontier_markers', 10)
        self.pose_graph_pub = self.create_publisher(MarkerArray, '/pose_graph_markers', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- Subscribers ---
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self._map_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self._scan_callback, 10)

        # --- TF and Nav2 action client ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.plan_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')

        # --- Components ---
        self.frontier_detector = FrontierDetector(
            min_frontier_size=self.min_frontier_size,
            iterations=self.rrt_frontier_iterations,
            step_size=self.rrt_frontier_step_size,
            cluster_radius=self.rrt_frontier_cluster_radius,
            max_frontier_points=self.rrt_frontier_max_points,
            map_padding=self.rrt_frontier_map_padding,
        )

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
            frontier_weight=self.graph_frontier_weight,
            odom_information=graph_odom_information,
        )

        # --- State ---
        self.latest_map: Optional[OccupancyGrid] = None
        self.frontier_clusters: List[FrontierCluster] = []
        self.last_frontier_detection_time = self.get_clock().now()
        self.frontiers_initialized = False
        self.current_goal: Optional[Tuple[float, float]] = None
        self.target_cluster: Optional[FrontierCluster] = None
        self.last_replan_time = self.get_clock().now()
        self.nav_goal_handle = None
        self.nav_goal_pending = False
        self.nav_cancel_requested = False
        self.nav_goal_sent_time = self.get_clock().now()
        self.nav_goal_start_known_cells = 0
        self.plan_check_pending = False
        self.plan_check_candidates = []
        self.plan_check_index = 0
        self.failed_goals: List[Tuple[float, float, object]] = []
        self.explored_history = deque()
        self.exploration_complete = False
        self.last_nav_wait_log_time = self.get_clock().now()
        self.replan_requested = True
        self.frontierless_scan_count = 0
        self.last_frontier_seen_time = self.get_clock().now()
        self.front_obstacle_distance = float('inf')
        self.probe_active = False
        self.probe_yaw = 0.0
        self.probe_start: Optional[Tuple[float, float]] = None
        self.probe_deadline = self.get_clock().now()
        self.probe_start_known_cells = 0

        # --- Timers ---
        self.control_timer = self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f'Exploration coordinator started. Execution: Nav2 NavigateToPose, '
            f'strategy: {self.exploration_strategy}'
        )
        self.get_logger().info(
            'Pose graph source: approximate TF trajectory. slam_toolbox can serialize '
            'its pose graph, but this node does not receive live nodes/edges/FIM from it.'
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg
        self._update_explored_history(msg)

    def _scan_callback(self, msg: LaserScan):
        front_min = float('inf')
        for i, value in enumerate(msg.ranges):
            if not math.isfinite(value):
                continue
            if value < msg.range_min or value > msg.range_max:
                continue
            angle = msg.angle_min + i * msg.angle_increment
            if abs(_wrap_angle(angle)) <= math.radians(25.0):
                front_min = min(front_min, value)
        self.front_obstacle_distance = front_min

    # ------------------------------------------------------------------
    # Main control loop (10 Hz)
    # ------------------------------------------------------------------

    def _control_loop(self):
        if self.exploration_complete:
            self._cancel_current_goal()
            return

        if self.latest_map is None:
            return

        pose = self._get_robot_pose()
        if pose is None:
            return
        rx, ry, ryaw = pose
        self.pose_graph_tracker.update((rx, ry, ryaw))
        frontiers_refreshed = self._update_frontiers(rx, ry)
        self._expire_failed_goals()
        self._check_nav_timeout()

        self._publish_frontier_markers()
        self._publish_goal_marker()
        if self.exploration_strategy == 'graph':
            self._publish_pose_graph_markers()

        if self.probe_active:
            self._run_frontier_probe(rx, ry, ryaw)
            return

        if self._navigation_active():
            return

        now = self.get_clock().now()
        elapsed = (now - self.last_replan_time).nanoseconds / 1e9
        if not frontiers_refreshed and not self.replan_requested and elapsed < self.replan_interval:
            return

        self.replan_requested = False
        self._dispatch_next_goal(rx, ry)

    # ------------------------------------------------------------------
    # Exploration logic
    # ------------------------------------------------------------------

    def _dispatch_next_goal(self, rx: float, ry: float):
        if not self.nav_client.server_is_ready() or not self.plan_client.server_is_ready():
            self.replan_requested = True
            self._log_waiting_for_nav2()
            return

        self.last_replan_time = self.get_clock().now()
        if len(self.frontier_clusters) == 0:
            if self._can_finish_exploration(rx, ry):
                self.exploration_complete = True
                self.get_logger().info(
                    f'Exploration complete. Map stable for {self.stability_duration}s.'
                )
            self.current_goal = None
            self.target_cluster = None
            return

        if self.exploration_strategy == 'graph':
            candidates = self._graph_based_goal_candidates(rx, ry)
        else:
            candidates = self._frontier_goal_candidates(rx, ry)

        if not candidates:
            self.current_goal = None
            self.target_cluster = None
            self.get_logger().info(
                f'No Nav2-safe frontier goal among {len(self.frontier_clusters)} clusters.'
            )
            return

        self._start_plan_check(candidates)

    def _update_frontiers(self, rx: float, ry: float) -> bool:
        now = self.get_clock().now()
        elapsed = (now - self.last_frontier_detection_time).nanoseconds / 1e9
        if self.frontiers_initialized and elapsed < self.frontier_detection_interval:
            return False
        self.frontier_clusters, _ = self.frontier_detector.detect(
            self.latest_map,
            (rx, ry),
        )
        self.last_frontier_detection_time = now
        self.frontiers_initialized = True
        if self.frontier_clusters:
            self.frontierless_scan_count = 0
            self.last_frontier_seen_time = now
        else:
            self.frontierless_scan_count += 1
        return True

    def _score_frontiers_by_size_distance(self, rx: float, ry: float):
        scored = []
        for c in self.frontier_clusters:
            dist = math.hypot(c.centroid_x - rx, c.centroid_y - ry)
            utility = c.size / (dist + 0.1)
            scored.append((utility, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _frontier_goal_candidates(
        self, rx: float, ry: float
    ) -> List[Tuple[Tuple[float, float], FrontierCluster, float]]:
        scored = self._score_frontiers_by_size_distance(rx, ry)
        candidates = []
        for utility, cluster in scored[:self.frontier_goal_candidates]:
            goal = self._make_nav_goal_for_cluster(cluster, rx, ry)
            if goal is None or self._goal_on_cooldown(goal):
                continue
            if math.hypot(goal[0] - rx, goal[1] - ry) < self.goal_reached_distance:
                continue
            candidates.append((goal, cluster, utility))
        return candidates

    def _graph_based_goal_candidates(
        self, rx: float, ry: float
    ) -> List[Tuple[Tuple[float, float], FrontierCluster, float]]:
        scored = self._score_frontiers_by_size_distance(rx, ry)
        candidates = []
        for _, cluster in scored[:self.graph_max_frontier_candidates]:
            goal = self._make_nav_goal_for_cluster(cluster, rx, ry)
            if goal is None or self._goal_on_cooldown(goal):
                continue
            if math.hypot(goal[0] - rx, goal[1] - ry) < self.goal_reached_distance:
                continue
            path = self._straight_line_path((rx, ry), goal)
            graph_score = self.graph_scorer.score(
                self.pose_graph_tracker.graph,
                self.latest_map,
                path,
                cluster.size,
            )
            if np.isfinite(graph_score):
                candidates.append((graph_score, goal, cluster))

        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [(goal, cluster, score) for score, goal, cluster in candidates]

    def _make_nav_goal_for_cluster(
        self, cluster: FrontierCluster, rx: float, ry: float
    ) -> Optional[Tuple[float, float]]:
        grid = self._occupancy_array()
        if grid is None:
            return None

        resolution = self.latest_map.info.resolution
        search_cells = max(1, int(math.ceil(self.frontier_goal_search_radius / resolution)))
        clearance_cells = max(0, int(math.ceil(self.frontier_goal_clearance / resolution)))
        map_edge_clearance_cells = max(
            0,
            int(math.ceil(self.frontier_goal_map_edge_clearance / resolution)),
        )
        width = self.latest_map.info.width
        height = self.latest_map.info.height

        best = None
        for frontier_x, frontier_y in self._frontier_goal_seed_points(cluster, rx, ry):
            direction_x = frontier_x - rx
            direction_y = frontier_y - ry
            frontier_distance = math.hypot(direction_x, direction_y)
            if frontier_distance < 1e-6:
                continue

            direction_x /= frontier_distance
            direction_y /= frontier_distance
            nominal_x = frontier_x - direction_x * self.frontier_goal_standoff
            nominal_y = frontier_y - direction_y * self.frontier_goal_standoff
            start_cell = self._world_to_grid(nominal_x, nominal_y)
            if start_cell is None:
                continue
            sx, sy = start_cell

            for dy in range(-search_cells, search_cells + 1):
                for dx in range(-search_cells, search_cells + 1):
                    gx = sx + dx
                    gy = sy + dy
                    if gx < 0 or gy < 0 or gx >= width or gy >= height:
                        continue
                    if not self._inside_map_with_margin(gx, gy, map_edge_clearance_cells):
                        continue
                    if grid[gy, gx] != 0:
                        continue
                    if not self._has_obstacle_clearance(grid, gx, gy, clearance_cells):
                        continue
                    wx, wy = self._grid_to_world(gx, gy)
                    advance = (wx - rx) * direction_x + (wy - ry) * direction_y
                    if advance < self.frontier_goal_min_advance:
                        continue
                    dist_to_frontier = math.hypot(wx - frontier_x, wy - frontier_y)
                    dist_to_robot = math.hypot(wx - rx, wy - ry)
                    standoff_error = abs(dist_to_frontier - self.frontier_goal_standoff)
                    lateral_error = abs((wx - rx) * direction_y - (wy - ry) * direction_x)
                    score = standoff_error + 0.1 * lateral_error - 0.02 * advance + 0.005 * dist_to_robot
                    if best is None or score < best[0]:
                        best = (score, wx, wy)

        if best is None:
            return None
        return best[1], best[2]

    def _frontier_goal_seed_points(
        self, cluster: FrontierCluster, rx: float, ry: float
    ) -> List[Tuple[float, float]]:
        points = list(cluster.points) if cluster.points else []
        points.append((cluster.centroid_x, cluster.centroid_y))
        points.sort(key=lambda p: math.hypot(p[0] - rx, p[1] - ry), reverse=True)
        limit = max(1, self.frontier_goal_point_sample_limit)
        if len(points) <= limit:
            return points

        sampled = []
        denom = max(1, limit - 1)
        for i in range(limit):
            index = round(i * (len(points) - 1) / denom)
            sampled.append(points[index])
        return sampled

    def _straight_line_path(
        self, start: Tuple[float, float], goal: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        dist = math.hypot(goal[0] - start[0], goal[1] - start[1])
        spacing = max(0.1, self.graph_hallucinated_node_spacing)
        steps = max(1, int(math.ceil(dist / spacing)))
        path = []
        for i in range(steps + 1):
            t = i / steps
            x = start[0] + (goal[0] - start[0]) * t
            y = start[1] + (goal[1] - start[1]) * t
            path.append((x, y))
        return path

    def _make_goal_pose(
        self,
        goal: Tuple[float, float],
        cluster: FrontierCluster,
    ) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = goal[0]
        pose.pose.position.y = goal[1]
        pose.pose.position.z = 0.0

        yaw = math.atan2(cluster.centroid_y - goal[1], cluster.centroid_x - goal[0])
        pose.pose.orientation = _quaternion_from_yaw(yaw)
        return pose

    # ------------------------------------------------------------------
    # Nav2 path precheck
    # ------------------------------------------------------------------

    def _start_plan_check(
        self,
        candidates: List[Tuple[Tuple[float, float], FrontierCluster, float]],
    ):
        self.plan_check_candidates = candidates
        self.plan_check_index = 0
        self._send_next_plan_check()

    def _send_next_plan_check(self):
        if self.plan_check_index >= len(self.plan_check_candidates):
            self.plan_check_pending = False
            self.plan_check_candidates = []
            self.current_goal = None
            self.target_cluster = None
            self.last_replan_time = self.get_clock().now()
            self.replan_requested = False
            self.get_logger().info('No Nav2-plannable frontier candidate. Waiting before retry.')
            return

        goal, cluster, _ = self.plan_check_candidates[self.plan_check_index]
        self.current_goal = goal
        self.target_cluster = cluster

        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal = self._make_goal_pose(goal, cluster)
        goal_msg.planner_id = ''
        goal_msg.use_start = False

        self.plan_check_pending = True
        future = self.plan_client.send_goal_async(goal_msg)
        future.add_done_callback(self._plan_check_response_callback)

    def _plan_check_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f'Failed to request Nav2 path precheck: {exc}')
            self._reject_current_plan_candidate()
            return

        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 rejected path precheck for frontier candidate.')
            self._reject_current_plan_candidate()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._plan_check_result_callback)

    def _plan_check_result_callback(self, future):
        try:
            result_msg = future.result()
            status = result_msg.status
            path = result_msg.result.path
        except Exception as exc:
            self.get_logger().warn(f'Nav2 path precheck failed: {exc}')
            self._reject_current_plan_candidate()
            return

        if status == GoalStatus.STATUS_SUCCEEDED and len(path.poses) >= 2:
            goal, cluster, score = self.plan_check_candidates[self.plan_check_index]
            self.plan_check_pending = False
            self.plan_check_candidates = []
            self._send_nav_goal(goal, cluster, score)
            return

        self._reject_current_plan_candidate()

    def _reject_current_plan_candidate(self):
        if self.plan_check_index < len(self.plan_check_candidates):
            goal, _, _ = self.plan_check_candidates[self.plan_check_index]
            self._mark_goal_failed(goal)
        self.plan_check_index += 1
        self.plan_check_pending = False
        self._send_next_plan_check()

    # ------------------------------------------------------------------
    # Open-boundary probing
    # ------------------------------------------------------------------

    def _try_start_frontier_probe(
        self,
        rx: float,
        ry: float,
        cluster: FrontierCluster,
    ) -> bool:
        if not self.frontier_probe_enabled or self.probe_active:
            return False
        if self.nav_goal_handle is not None or self.nav_goal_pending:
            return False

        if cluster is None:
            return False
        if not self._front_is_open_edge_not_obstacle(cluster):
            return False

        target_x, target_y = self._probe_target_point(cluster, rx, ry)
        dx = target_x - rx
        dy = target_y - ry
        if math.hypot(dx, dy) < 1e-3:
            return False

        self.probe_yaw = math.atan2(dy, dx)
        self.probe_start = (rx, ry)
        self.probe_deadline = self.get_clock().now() + Duration(
            seconds=self.frontier_probe_timeout
        )
        self.probe_start_known_cells = self._known_cell_count()
        self.probe_active = True
        self.current_goal = (target_x, target_y)
        self.target_cluster = cluster
        self.replan_requested = False
        self.get_logger().info(
            f'Frontier reached with little map growth; probing open edge toward '
            f'({target_x:.2f}, {target_y:.2f}) until the map grows.'
        )
        return True

    def _try_start_probe_after_frontier_reached(
        self,
        goal: Optional[Tuple[float, float]],
        cluster: Optional[FrontierCluster],
        map_growth_cells: int,
    ) -> bool:
        if goal is None or cluster is None:
            return False
        if map_growth_cells > self.frontier_probe_start_max_map_growth_cells:
            return False
        pose = self._get_robot_pose()
        if pose is None:
            return False
        rx, ry, _ = pose
        if math.hypot(goal[0] - rx, goal[1] - ry) > max(0.6, self.goal_reached_distance * 2.0):
            return False
        return self._try_start_frontier_probe(rx, ry, cluster)

    def _front_is_open_edge_not_obstacle(self, cluster: FrontierCluster) -> bool:
        return (
            self._cluster_is_open_boundary(cluster)
            and self.front_obstacle_distance >= self.frontier_probe_obstacle_distance
        )

    def _probe_target_point(
        self,
        cluster: FrontierCluster,
        rx: float,
        ry: float,
    ) -> Tuple[float, float]:
        points = list(cluster.points) if cluster.points else []
        if points:
            points.sort(key=lambda p: math.hypot(p[0] - rx, p[1] - ry), reverse=True)
            return points[0]
        return cluster.centroid_x, cluster.centroid_y

    def _cluster_is_open_boundary(self, cluster: FrontierCluster) -> bool:
        points = list(cluster.points) if cluster.points else []
        points.append((cluster.centroid_x, cluster.centroid_y))
        for wx, wy in points:
            cell = self._world_to_grid(wx, wy)
            if cell is None:
                return True
            if self._grid_cell_on_map_edge(cell[0], cell[1], margin_cells=1):
                return True
        return False

    def _run_frontier_probe(self, rx: float, ry: float, ryaw: float):
        if self.probe_start is None:
            self._finish_frontier_probe('missing start pose')
            return

        now = self.get_clock().now()
        known_growth = self._known_cell_count() - self.probe_start_known_cells

        if self.front_obstacle_distance < self.frontier_probe_obstacle_distance:
            self._finish_frontier_probe(
                f'obstacle at {self.front_obstacle_distance:.2f}m'
            )
            return
        if (
            self.frontier_probe_map_growth_cells > 0
            and known_growth >= self.frontier_probe_map_growth_cells
        ):
            self._finish_frontier_probe(f'map grew by {known_growth} cells')
            return
        if now > self.probe_deadline:
            self._finish_frontier_probe('timeout')
            return

        angle_error = _wrap_angle(self.probe_yaw - ryaw)
        cmd = Twist()
        cmd.angular.z = max(
            -self.frontier_probe_max_angular_speed,
            min(
                self.frontier_probe_max_angular_speed,
                self.frontier_probe_kp_angular * angle_error,
            ),
        )
        if abs(angle_error) < 0.7:
            cmd.linear.x = self.frontier_probe_speed
        self.cmd_vel_pub.publish(cmd)

    def _finish_frontier_probe(self, reason: str):
        self.cmd_vel_pub.publish(Twist())
        self.probe_active = False
        self.probe_start = None
        self.current_goal = None
        self.target_cluster = None
        self.frontiers_initialized = False
        self.last_replan_time = self.get_clock().now()
        self.replan_requested = True
        self.get_logger().info(f'Open-boundary probe finished: {reason}.')

    # ------------------------------------------------------------------
    # Nav2 action handling
    # ------------------------------------------------------------------

    def _send_nav_goal(
        self,
        goal: Tuple[float, float],
        cluster: FrontierCluster,
        score: float,
    ):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self._make_goal_pose(goal, cluster)

        self.current_goal = goal
        self.target_cluster = cluster
        self.nav_goal_pending = True
        self.nav_cancel_requested = False
        self.nav_goal_sent_time = self.get_clock().now()
        self.nav_goal_start_known_cells = self._known_cell_count()

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._nav_goal_response_callback)
        self.get_logger().info(
            f'Sent Nav2 goal ({goal[0]:.2f}, {goal[1]:.2f}), '
            f'frontier_size={cluster.size}, score={score:.3f}'
        )

    def _nav_goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f'Failed to send Nav2 goal: {exc}')
            self._mark_current_goal_failed()
            self._clear_navigation_state()
            return

        self.nav_goal_pending = False
        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 rejected the exploration goal.')
            self._mark_current_goal_failed()
            self._clear_navigation_state()
            return

        self.nav_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav_result_callback)

    def _nav_result_callback(self, future):
        try:
            result = future.result()
            status = result.status
        except Exception as exc:
            self.get_logger().warn(f'Nav2 goal result failed: {exc}')
            self._mark_current_goal_failed()
            self._clear_navigation_state()
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Nav2 reached exploration goal.')
            goal = self.current_goal
            cluster = self.target_cluster
            map_growth_cells = self._known_cell_count() - self.nav_goal_start_known_cells
            self.nav_goal_handle = None
            self.nav_goal_pending = False
            self.nav_cancel_requested = False
            if self._try_start_probe_after_frontier_reached(
                goal,
                cluster,
                map_growth_cells,
            ):
                return
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn('Nav2 exploration goal was canceled.')
            self._mark_current_goal_failed()
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn('Nav2 aborted exploration goal.')
            self._mark_current_goal_failed()
        else:
            self.get_logger().warn(f'Nav2 finished with status={status}.')
            self._mark_current_goal_failed()

        self._clear_navigation_state()

    def _navigation_active(self) -> bool:
        return (
            self.probe_active
            or self.plan_check_pending
            or self.nav_goal_pending
            or self.nav_goal_handle is not None
        )

    def _check_nav_timeout(self):
        if self.nav2_goal_timeout <= 0.0:
            return
        if self.nav_goal_handle is None or self.nav_cancel_requested:
            return
        elapsed = (self.get_clock().now() - self.nav_goal_sent_time).nanoseconds / 1e9
        if elapsed < self.nav2_goal_timeout:
            return
        self.get_logger().warn(
            f'Nav2 goal timed out after {elapsed:.1f}s; canceling and trying another frontier.'
        )
        self._mark_current_goal_failed()
        self._cancel_current_goal()

    def _cancel_current_goal(self):
        if self.nav_goal_handle is None or self.nav_cancel_requested:
            return
        self.nav_cancel_requested = True
        self.nav_goal_handle.cancel_goal_async()

    def _clear_navigation_state(self):
        self.current_goal = None
        self.target_cluster = None
        self.nav_goal_handle = None
        self.nav_goal_pending = False
        self.nav_cancel_requested = False
        self.last_replan_time = self.get_clock().now()
        self.replan_requested = True

    def _mark_current_goal_failed(self):
        if self.current_goal is None:
            return
        self._mark_goal_failed(self.current_goal)

    def _mark_goal_failed(self, goal: Tuple[float, float]):
        if self.failed_goal_cooldown <= 0.0:
            return
        deadline = self.get_clock().now() + Duration(seconds=self.failed_goal_cooldown)
        self.failed_goals.append((goal[0], goal[1], deadline))

    def _expire_failed_goals(self):
        now = self.get_clock().now()
        self.failed_goals = [
            (x, y, deadline)
            for x, y, deadline in self.failed_goals
            if deadline > now
        ]

    def _goal_on_cooldown(self, goal: Tuple[float, float]) -> bool:
        for x, y, _ in self.failed_goals:
            if math.hypot(goal[0] - x, goal[1] - y) < self.failed_goal_radius:
                return True
        return False

    def _log_waiting_for_nav2(self):
        now = self.get_clock().now()
        elapsed = (now - self.last_nav_wait_log_time).nanoseconds / 1e9
        if elapsed < self.nav2_wait_log_interval:
            return
        self.last_nav_wait_log_time = now
        self.get_logger().info('Waiting for Nav2 planner/navigation action servers...')

    # ------------------------------------------------------------------
    # Map stability
    # ------------------------------------------------------------------

    def _update_explored_history(self, msg: OccupancyGrid):
        now = self.get_clock().now()
        known = int(np.sum(np.array(msg.data, dtype=np.int8) != -1))
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

    def _can_finish_exploration(self, rx: float, ry: float) -> bool:
        if self.frontierless_scan_count < self.completion_frontierless_scans:
            return False
        since_frontier = (
            self.get_clock().now() - self.last_frontier_seen_time
        ).nanoseconds / 1e9
        if since_frontier < self.completion_grace_period:
            return False
        if not self._robot_inside_latest_map(rx, ry):
            return False
        if self._has_open_map_edge():
            return False
        return self._is_map_stable()

    # ------------------------------------------------------------------
    # Visualization publishers
    # ------------------------------------------------------------------

    def _publish_frontier_markers(self):
        ma = MarkerArray()
        delete = Marker()
        delete.action = Marker.DELETEALL
        ma.markers.append(delete)

        for i, c in enumerate(self.frontier_clusters):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'frontiers'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = c.centroid_x
            m.pose.position.y = c.centroid_y
            m.pose.position.z = 0.1
            s = min(1.2, max(0.15, c.size * 0.002))
            m.scale.x = m.scale.y = m.scale.z = s
            m.color.a = 0.6
            m.color.g = 1.0
            m.color.b = 0.3
            ma.markers.append(m)

        self.frontier_pub.publish(ma)

    def _publish_goal_marker(self):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'goal'
        m.id = 0
        m.type = Marker.SPHERE

        if self.current_goal is not None:
            m.action = Marker.ADD
            m.pose.position.x = self.current_goal[0]
            m.pose.position.y = self.current_goal[1]
            m.pose.position.z = 0.2
            m.scale.x = m.scale.y = m.scale.z = 0.25
            m.color.a = 1.0
            m.color.r = 1.0
        else:
            m.action = Marker.DELETE

        self.goal_pub.publish(m)

    def _publish_pose_graph_markers(self):
        markers = graph_to_marker_array(
            self.pose_graph_tracker.graph,
            'map',
            self.get_clock().now().to_msg(),
        )
        self.pose_graph_pub.publish(markers)

    # ------------------------------------------------------------------
    # Map helpers
    # ------------------------------------------------------------------

    def _occupancy_array(self):
        if self.latest_map is None:
            return None
        return np.asarray(self.latest_map.data, dtype=np.int16).reshape(
            self.latest_map.info.height,
            self.latest_map.info.width,
        )

    def _known_cell_count(self) -> int:
        grid = self._occupancy_array()
        if grid is None:
            return 0
        return int(np.sum(grid != -1))

    def _world_to_grid(self, wx: float, wy: float) -> Optional[Tuple[int, int]]:
        info = self.latest_map.info
        gx = int(math.floor((wx - info.origin.position.x) / info.resolution))
        gy = int(math.floor((wy - info.origin.position.y) / info.resolution))
        if gx < 0 or gy < 0 or gx >= info.width or gy >= info.height:
            return None
        return gx, gy

    def _grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        info = self.latest_map.info
        wx = info.origin.position.x + (gx + 0.5) * info.resolution
        wy = info.origin.position.y + (gy + 0.5) * info.resolution
        return wx, wy

    def _inside_map_with_margin(self, gx: int, gy: int, margin_cells: int) -> bool:
        if margin_cells <= 0:
            return True
        info = self.latest_map.info
        return (
            margin_cells <= gx < info.width - margin_cells
            and margin_cells <= gy < info.height - margin_cells
        )

    def _grid_cell_on_map_edge(self, gx: int, gy: int, margin_cells: int = 0) -> bool:
        info = self.latest_map.info
        return (
            gx <= margin_cells
            or gy <= margin_cells
            or gx >= info.width - 1 - margin_cells
            or gy >= info.height - 1 - margin_cells
        )

    def _has_obstacle_clearance(self, grid, gx: int, gy: int, clearance_cells: int) -> bool:
        if clearance_cells <= 0:
            return True
        y0 = max(0, gy - clearance_cells)
        y1 = min(grid.shape[0], gy + clearance_cells + 1)
        x0 = max(0, gx - clearance_cells)
        x1 = min(grid.shape[1], gx + clearance_cells + 1)
        return not np.any(grid[y0:y1, x0:x1] > 50)

    def _has_open_map_edge(self) -> bool:
        grid = self._occupancy_array()
        if grid is None or self.completion_open_edge_min_cells <= 0:
            return False
        height, width = grid.shape
        if height == 0 or width == 0:
            return False

        free_edge_cells = 0
        for j in range(width):
            if grid[0, j] == 0:
                free_edge_cells += 1
            if height > 1 and grid[height - 1, j] == 0:
                free_edge_cells += 1
            if free_edge_cells >= self.completion_open_edge_min_cells:
                return True

        for i in range(1, max(1, height - 1)):
            if grid[i, 0] == 0:
                free_edge_cells += 1
            if width > 1 and grid[i, width - 1] == 0:
                free_edge_cells += 1
            if free_edge_cells >= self.completion_open_edge_min_cells:
                return True

        return False

    def _robot_inside_latest_map(self, rx: float, ry: float) -> bool:
        if self.latest_map is None:
            return False
        return self._world_to_grid(rx, ry) is not None

    # ------------------------------------------------------------------
    # TF helpers
    # ------------------------------------------------------------------

    def _get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=0.5),
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            yaw = _yaw_from_quaternion(t.transform.rotation)
            return x, y, yaw
        except Exception:
            return None

    def destroy_node(self):
        self._cancel_current_goal()
        self.cmd_vel_pub.publish(Twist())
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
