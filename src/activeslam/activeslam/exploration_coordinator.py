import math
from collections import deque
from random import random as _random
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .frontier_detector import FrontierCluster, FrontierDetector
from .path_planner import create_planner


def _yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class ExplorationCoordinator(Node):
    def __init__(self):
        super().__init__('exploration_coordinator')

        # --- Parameters ---
        self.planner_type = self.declare_parameter('planner_type', 'astar').value
        self.target_linear_speed = self.declare_parameter('target_linear_speed', 0.08).value
        self.lookahead_distance = self.declare_parameter('lookahead_distance', 0.5).value
        self.kp_angular = self.declare_parameter('kp_angular', 1.2).value
        self.max_angular_speed = self.declare_parameter('max_angular_speed', 0.6).value
        self.replan_interval = self.declare_parameter('replan_interval', 3.0).value
        self.goal_tolerance = self.declare_parameter('goal_tolerance', 0.5).value
        self.stability_duration = self.declare_parameter('stability_duration', 10.0).value
        self.stability_threshold = self.declare_parameter('stability_threshold', 0.02).value
        self.min_frontier_size = self.declare_parameter('min_frontier_size', 5).value
        self.obstacle_distance = self.declare_parameter('obstacle_distance', 0.4).value
        self.obstacle_inflation = self.declare_parameter('obstacle_inflation', 0.3).value
        self.rrt_step_size = self.declare_parameter('rrt_step_size', 0.3).value

        # --- Publishers ---
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.goal_pub = self.create_publisher(Marker, '/goal_point', 10)
        self.frontier_pub = self.create_publisher(MarkerArray, '/frontier_markers', 10)

        # --- Subscribers ---
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self._map_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self._scan_callback, 10)

        # --- TF ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Components ---
        self.frontier_detector = FrontierDetector(min_frontier_size=self.min_frontier_size)
        self.planner = create_planner(
            self.planner_type,
            goal_tolerance=self.goal_tolerance,
            obstacle_inflation=self.obstacle_inflation,
            rrt_step_size=self.rrt_step_size,
        )

        # --- State ---
        self.latest_map: Optional[OccupancyGrid] = None
        self.frontier_clusters: List[FrontierCluster] = []
        self.current_path: List[Tuple[float, float]] = []
        self.current_goal: Optional[Tuple[float, float]] = None
        self.target_cluster: Optional[FrontierCluster] = None
        self.last_replan_time = self.get_clock().now()
        self.front_obstacle_distance = float('inf')
        self.safety_turn_deadline = self.get_clock().now()
        self.safety_turn_direction = 0.0
        self.explored_history = deque()
        self.exploration_complete = False
        self.random_walk_deadline = self.get_clock().now()

        # --- Timers ---
        self.control_timer = self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f'Exploration coordinator started. Planner: {self.planner_type}'
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg
        self.frontier_clusters, _ = self.frontier_detector.detect(msg)
        self._update_explored_history(msg)

    def _scan_callback(self, msg: LaserScan):
        front = list(msg.ranges[:20]) + list(msg.ranges[-20:])
        valid = [d for d in front if msg.range_min < d < msg.range_max]
        self.front_obstacle_distance = min(valid) if valid else float('inf')

    # ------------------------------------------------------------------
    # Main control loop (10 Hz)
    # ------------------------------------------------------------------

    def _control_loop(self):
        if self.exploration_complete:
            self._publish_cmd_vel(0.0, 0.0)
            return

        if self.latest_map is None:
            self._publish_cmd_vel(0.0, 0.1)  # rotate to build initial map
            return

        # --- Safety override ---
        if self._handle_safety():
            return

        # --- Get robot pose ---
        pose = self._get_robot_pose()
        if pose is None:
            self._publish_cmd_vel(0.0, 0.1)
            return
        rx, ry, ryaw = pose

        # --- Publish visualizations ---
        self._publish_frontier_markers()
        self._publish_goal_marker()

        # --- Check if replan needed ---
        now = self.get_clock().now()
        elapsed = (now - self.last_replan_time).nanoseconds / 1e9
        near_goal = (
            self.current_goal is not None
            and math.hypot(rx - self.current_goal[0], ry - self.current_goal[1])
            < self.goal_tolerance
        )
        frontier_shrunk = self._target_frontier_shrunk()

        need_replan = (
            len(self.current_path) == 0
            or near_goal
            or frontier_shrunk
            or elapsed > self.replan_interval
        )

        if need_replan and elapsed > 1.0:
            self._replan(rx, ry)

        # --- Execute path ---
        if len(self.current_path) > 0:
            self._follow_path(rx, ry, ryaw)
        else:
            self._random_walk()

    # ------------------------------------------------------------------
    # Safety (ported from random_walker)
    # ------------------------------------------------------------------

    def _handle_safety(self) -> bool:
        if self.front_obstacle_distance < self.obstacle_distance:
            now = self.get_clock().now()
            if now > self.safety_turn_deadline:
                self.safety_turn_direction = -1.0 if self.safety_turn_direction >= 0 else 1.0
                duration = 0.8 + np.random.random() * 1.0
                self.safety_turn_deadline = now + Duration(seconds=duration)
            self._publish_cmd_vel(0.0, self.safety_turn_direction * 0.8)
            self.current_path.clear()
            self.current_goal = None
            self.target_cluster = None
            self.last_replan_time = now
            return True
        return False

    # ------------------------------------------------------------------
    # Exploration logic
    # ------------------------------------------------------------------

    def _replan(self, rx: float, ry: float):
        if len(self.frontier_clusters) == 0:
            if self._is_map_stable():
                self.exploration_complete = True
                self.get_logger().info(
                    f'Exploration complete. Map stable for {self.stability_duration}s.'
                )
            self.current_path.clear()
            self.current_goal = None
            self.target_cluster = None
            return

        scored = []
        for c in self.frontier_clusters:
            dist = math.hypot(c.centroid_x - rx, c.centroid_y - ry)
            utility = c.size / (dist + 0.1)
            scored.append((utility, c))
        scored.sort(key=lambda x: x[0], reverse=True)

        for _, cluster in scored[:3]:
            path, cost, success = self.planner.plan(
                self.latest_map,
                (rx, ry),
                (cluster.centroid_x, cluster.centroid_y),
            )
            if success and len(path) >= 2:
                self.current_path = path
                self.current_goal = (cluster.centroid_x, cluster.centroid_y)
                self.target_cluster = cluster
                self.last_replan_time = self.get_clock().now()
                self._publish_path()
                return

        self.current_path.clear()
        self.current_goal = None
        self.target_cluster = None
        self.last_replan_time = self.get_clock().now()
        self.get_logger().info(
            f'No reachable frontier among {len(scored)} clusters. Random walking.'
        )

    def _target_frontier_shrunk(self) -> bool:
        if self.target_cluster is None:
            return False
        for c in self.frontier_clusters:
            d = math.hypot(
                c.centroid_x - self.target_cluster.centroid_x,
                c.centroid_y - self.target_cluster.centroid_y,
            )
            if d < 0.5 and c.size < self.target_cluster.size * 0.5:
                return True
        return False

    # ------------------------------------------------------------------
    # Path following (Pure Pursuit)
    # ------------------------------------------------------------------

    def _follow_path(self, rx: float, ry: float, ryaw: float):
        path = self.current_path

        closest_idx = 0
        closest_dist = float('inf')
        for i, (wx, wy) in enumerate(path):
            d = math.hypot(wx - rx, wy - ry)
            if d < closest_dist:
                closest_dist = d
                closest_idx = i

        lookahead_idx = closest_idx
        forward_found = False
        for i in range(closest_idx, len(path)):
            dx = path[i][0] - rx
            dy = path[i][1] - ry
            forward_proj = dx * math.cos(ryaw) + dy * math.sin(ryaw)
            if forward_proj < 0.0:
                continue
            if math.hypot(dx, dy) >= self.lookahead_distance:
                lookahead_idx = i
                forward_found = True
                break
        if not forward_found:
            lookahead_idx = len(path) - 1

        tx, ty = path[lookahead_idx]
        target_angle = math.atan2(ty - ry, tx - rx)
        angle_error = target_angle - ryaw
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        angular_z = self.kp_angular * angle_error
        angular_z = max(-self.max_angular_speed, min(self.max_angular_speed, angular_z))
        linear_x = self.target_linear_speed * max(0.0, 1.0 - abs(angular_z) / self.max_angular_speed)

        dist_to_end = math.hypot(path[-1][0] - rx, path[-1][1] - ry)
        if dist_to_end < 0.5:
            linear_x *= max(0.1, dist_to_end / 0.5)

        self._publish_cmd_vel(linear_x, angular_z)

    # ------------------------------------------------------------------
    # Random walk (fallback when no path available)
    # ------------------------------------------------------------------

    def _random_walk(self):
        now = self.get_clock().now()
        if now > self.random_walk_deadline:
            linear_x = 0.03 + _random() * 0.05
            angular_z = -0.25 + _random() * 0.5
            duration = 1.5 + _random() * 2.5
            self.random_walk_deadline = now + Duration(seconds=duration)
            self._publish_cmd_vel(linear_x, angular_z)
        # On non-deadline ticks, repeat the last cmd_vel to keep moving

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

    def _publish_path(self):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.current_path:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.path_pub.publish(path_msg)

    # ------------------------------------------------------------------
    # Helpers
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

    def _publish_cmd_vel(self, linear_x: float, angular_z: float):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)

    def destroy_node(self):
        self._publish_cmd_vel(0.0, 0.0)
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
