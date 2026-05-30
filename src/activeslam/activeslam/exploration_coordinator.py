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
    SafeGoalSearchConfig,
    navigation_timed_out,
    select_safe_frontier_goal,
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
    COMPLETE = 'complete'

    def __init__(self):
        super().__init__('exploration_coordinator')

        # --- Parameters ---
        self.stability_duration = self.declare_parameter('stability_duration', 10.0).value
        self.stability_threshold = self.declare_parameter('stability_threshold', 0.02).value
        self.min_frontier_size = self.declare_parameter('min_frontier_size', 5).value
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
        self.frontier_detector = FrontierDetector(min_frontier_size=self.min_frontier_size)
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
            frontier_weight=self.graph_frontier_weight,
            odom_information=graph_odom_information,
        )

        # --- Exploration state ---
        self.latest_map: Optional[OccupancyGrid] = None
        self.frontier_clusters: List[FrontierCluster] = []
        self.current_goal: Optional[Tuple[float, float]] = None
        self.target_cluster: Optional[FrontierCluster] = None
        self.explored_history = deque()
        self.state = self.WAITING_FOR_NAV2
        self.next_retry_wall_time = 0.0
        self.selection_generation = 0
        self.selection_start_xy = (0.0, 0.0)
        self.selection_clusters = []
        self.selection_cluster_index = 0
        self.selection_goal_index = 0
        self.selection_best: Optional[PlannedPath] = None
        self.graph_candidates = []
        self.selection_request_wall_time: Optional[float] = None
        self.selection_request_goal: Optional[Tuple[float, float]] = None
        self.navigation_start_wall_time: Optional[float] = None
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
        self.frontier_clusters, _ = self.frontier_detector.detect(msg)
        self._update_explored_history(msg)

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

        scored = self._score_frontiers_by_size_distance(rx, ry)
        limit = (
            self.graph_max_frontier_candidates
            if self.exploration_strategy == 'graph'
            else self.frontier_planning_attempts
        )
        self.selection_start_xy = (rx, ry)
        now = time.monotonic()
        self.failed_goals.expire(now)
        self.selection_clusters = [
            (cluster, self._selectable_goal_for_cluster(cluster, rx, ry, now))
            for _, cluster in scored[:limit]
        ]
        self.selection_generation = self.nav2.start_path_batch()
        self.selection_cluster_index = 0
        self.selection_goal_index = 0
        self.selection_best = None
        self.graph_candidates = []
        self.selection_request_wall_time = None
        self.state = self.SELECTING
        self._request_next_path()

    def _request_next_path(self):
        while self.selection_cluster_index < len(self.selection_clusters):
            cluster, goal_candidates = self.selection_clusters[self.selection_cluster_index]
            if self.selection_goal_index < len(goal_candidates):
                goal_xy = goal_candidates[self.selection_goal_index]
                self.selection_request_wall_time = time.monotonic()
                self.selection_request_goal = goal_xy
                self.nav2.compute_path(
                    self.selection_generation,
                    self.selection_start_xy,
                    goal_xy,
                    self._path_computed,
                )
                return
            if self.selection_best is not None:
                if self.exploration_strategy == 'frontier':
                    self._dispatch_navigation(cluster, self.selection_best)
                    return
                score = self.graph_scorer.score(
                    self.pose_graph_tracker.graph,
                    self.latest_map,
                    self.selection_best.points,
                    cluster.size,
                )
                if np.isfinite(score):
                    self.graph_candidates.append((score, cluster, self.selection_best))
            self.selection_cluster_index += 1
            self.selection_goal_index = 0
            self.selection_best = None

        if self.graph_candidates:
            self.graph_candidates.sort(key=lambda item: item[0], reverse=True)
            score, cluster, planned_path = self.graph_candidates[0]
            self.get_logger().info(
                f'Graph-based selected frontier score={score:.3f}, size={cluster.size}'
            )
            self._dispatch_navigation(cluster, planned_path)
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
        if planned_path is None:
            self._mark_goal_failed(self.selection_request_goal)
        self.selection_request_goal = None
        if (
            planned_path is not None
            and (self.selection_best is None or planned_path.cost < self.selection_best.cost)
        ):
            self.selection_best = planned_path
        self.selection_goal_index += 1
        self._request_next_path()

    def _dispatch_navigation(self, cluster: FrontierCluster, planned_path: PlannedPath):
        self.nav2.cancel_path_batch()
        self.current_goal = planned_path.goal_xy
        self.target_cluster = cluster
        self.selection_request_wall_time = None
        self.selection_request_goal = None
        self.state = self.NAVIGATING
        self.navigation_start_wall_time = time.monotonic()
        yaw = heading_to_target(planned_path.goal_xy, (cluster.centroid_x, cluster.centroid_y))
        self.get_logger().info(
            f'Navigating to frontier goal=({planned_path.goal_xy[0]:.2f}, '
            f'{planned_path.goal_xy[1]:.2f}), size={cluster.size}, cost={planned_path.cost:.2f}'
        )
        self.nav2.navigate(planned_path.goal_xy, yaw, self._navigation_finished)

    def _navigation_finished(self, status: int):
        if self.state != self.NAVIGATING:
            return
        if status == GOAL_STATUS_SUCCEEDED:
            self.get_logger().info('Nav2 reached the active frontier goal.')
        else:
            self.get_logger().warn(f'Nav2 navigation ended with status={status}; retrying.')
            self._mark_goal_failed(self.current_goal)
        self._clear_navigation()
        self._schedule_retry(0.0)

    def _clear_navigation(self):
        self.current_goal = None
        self.target_cluster = None
        self.navigation_start_wall_time = None

    def _schedule_retry(self, delay: Optional[float] = None):
        self.state = self.IDLE
        self.next_retry_wall_time = time.monotonic() + (
            self.frontier_retry_interval if delay is None else delay
        )

    # ------------------------------------------------------------------
    # Frontier candidate generation
    # ------------------------------------------------------------------

    def _score_frontiers_by_size_distance(self, rx: float, ry: float):
        scored = []
        for cluster in self.frontier_clusters:
            dist = math.hypot(cluster.centroid_x - rx, cluster.centroid_y - ry)
            scored.append((cluster.size / (dist + 0.1), cluster))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _selectable_goal_for_cluster(
        self,
        cluster: FrontierCluster,
        rx: float,
        ry: float,
        now: float,
    ) -> List[Tuple[float, float]]:
        info = self.latest_map.info
        data = np.array(self.latest_map.data, dtype=np.int8).reshape(
            info.height,
            info.width,
        )
        goal = select_safe_frontier_goal(
            data,
            GridGeometry(
                origin_x=info.origin.position.x,
                origin_y=info.origin.position.y,
                resolution=info.resolution,
                width=info.width,
                height=info.height,
            ),
            cluster.cells,
            (rx, ry),
            SafeGoalSearchConfig(
                search_radius=self.frontier_goal_search_radius,
                clearance=self.frontier_goal_clearance,
                standoff=self.frontier_goal_standoff,
                map_edge_clearance=self.frontier_goal_map_edge_clearance,
                min_advance=self.frontier_goal_min_advance,
                reach_radius=self.nav2_goal_reach_radius,
                point_sample_limit=self.frontier_goal_point_sample_limit,
            ),
        )
        if goal is None or self.failed_goals.contains(goal, now):
            return []
        return [goal]

    def _mark_goal_failed(self, goal: Optional[Tuple[float, float]]):
        if goal is not None:
            self.failed_goals.mark(goal, time.monotonic())

    # ------------------------------------------------------------------
    # Map stability and visualization
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
            marker.color.b = 0.3
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
