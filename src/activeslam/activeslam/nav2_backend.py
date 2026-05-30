import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple


GOAL_STATUS_ABORTED = 6
GOAL_STATUS_SUCCEEDED = 4


@dataclass
class PlannedPath:
    goal_xy: Tuple[float, float]
    points: List[Tuple[float, float]]
    cost: float


class GenerationGuard:
    """Invalidate callbacks from action requests superseded by newer work."""

    def __init__(self):
        self._generation = 0

    def advance(self) -> int:
        self._generation += 1
        return self._generation

    def is_current(self, generation: int) -> bool:
        return generation == self._generation


def path_to_xy(path_message) -> List[Tuple[float, float]]:
    return [
        (pose.pose.position.x, pose.pose.position.y)
        for pose in path_message.poses
    ]


def path_length(points: Sequence[Tuple[float, float]]) -> float:
    return sum(
        math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        for i in range(1, len(points))
    )


def heading_to_target(
    origin_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
) -> float:
    return math.atan2(target_xy[1] - origin_xy[1], target_xy[0] - origin_xy[0])


class Nav2Backend:
    """Small asynchronous adapter around the Nav2 actions used by exploration."""

    def __init__(self, node):
        from rclpy.action import ActionClient
        from nav2_msgs.action import ComputePathToPose, NavigateToPose, Spin

        self._node = node
        self._compute_action = ComputePathToPose
        self._navigate_action = NavigateToPose
        self._spin_action = Spin
        self._compute_client = ActionClient(node, ComputePathToPose, '/compute_path_to_pose')
        self._navigate_client = ActionClient(node, NavigateToPose, '/navigate_to_pose')
        self._spin_client = ActionClient(node, Spin, '/spin')
        self._path_guard = GenerationGuard()
        self._navigation_guard = GenerationGuard()
        self._spin_guard = GenerationGuard()
        self._path_goal_handle = None
        self._navigation_goal_handle = None
        self._spin_goal_handle = None

    def servers_ready(self) -> bool:
        return (
            self._compute_client.wait_for_server(timeout_sec=0.0)
            and self._navigate_client.wait_for_server(timeout_sec=0.0)
            and self._spin_client.wait_for_server(timeout_sec=0.0)
        )

    def start_path_batch(self) -> int:
        generation = self._path_guard.advance()
        self._cancel_handle(self._path_goal_handle)
        self._path_goal_handle = None
        return generation

    def cancel_path_batch(self):
        self.start_path_batch()

    def compute_path(
        self,
        generation: int,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        callback: Callable[[Optional[PlannedPath]], None],
    ):
        if not self._path_guard.is_current(generation):
            return

        goal = self._compute_action.Goal()
        goal.start = self._pose_stamped(start_xy[0], start_xy[1], 0.0)
        goal.goal = self._pose_stamped(goal_xy[0], goal_xy[1], 0.0)
        goal.planner_id = 'GridBased'
        goal.use_start = True
        future = self._compute_client.send_goal_async(goal)
        future.add_done_callback(
            lambda done: self._path_goal_response(done, generation, goal_xy, callback)
        )

    def navigate(
        self,
        goal_xy: Tuple[float, float],
        yaw: float,
        callback: Callable[[int], None],
    ) -> int:
        generation = self.cancel_navigation()
        goal = self._navigate_action.Goal()
        goal.pose = self._pose_stamped(goal_xy[0], goal_xy[1], yaw)
        future = self._navigate_client.send_goal_async(goal)
        future.add_done_callback(
            lambda done: self._navigation_goal_response(done, generation, callback)
        )
        return generation

    def cancel_navigation(self) -> int:
        generation = self._navigation_guard.advance()
        self._cancel_handle(self._navigation_goal_handle)
        self._navigation_goal_handle = None
        return generation

    def spin_once(
        self,
        target_yaw: float,
        timeout_seconds: float,
        callback: Callable[[int], None],
    ) -> int:
        generation = self.cancel_spin()
        goal = self._spin_action.Goal()
        goal.target_yaw = float(target_yaw)
        goal.time_allowance = self._duration(timeout_seconds)
        future = self._spin_client.send_goal_async(goal)
        future.add_done_callback(
            lambda done: self._spin_goal_response(done, generation, callback)
        )
        return generation

    def cancel_spin(self) -> int:
        generation = self._spin_guard.advance()
        self._cancel_handle(self._spin_goal_handle)
        self._spin_goal_handle = None
        return generation

    def destroy(self):
        self.cancel_path_batch()
        self.cancel_navigation()
        self.cancel_spin()
        self._compute_client.destroy()
        self._navigate_client.destroy()
        self._spin_client.destroy()

    def _path_goal_response(self, future, generation, goal_xy, callback):
        goal_handle = self._future_result(future)
        if goal_handle is None:
            if self._path_guard.is_current(generation):
                callback(None)
            return
        if not self._path_guard.is_current(generation):
            self._cancel_handle(goal_handle)
            return
        if not goal_handle.accepted:
            callback(None)
            return
        self._path_goal_handle = goal_handle
        future = goal_handle.get_result_async()
        future.add_done_callback(
            lambda done: self._path_result(done, generation, goal_xy, callback)
        )

    def _path_result(self, future, generation, goal_xy, callback):
        if not self._path_guard.is_current(generation):
            return
        self._path_goal_handle = None
        wrapped_result = self._future_result(future)
        if wrapped_result is None or wrapped_result.status != GOAL_STATUS_SUCCEEDED:
            callback(None)
            return
        points = path_to_xy(wrapped_result.result.path)
        if len(points) < 2:
            callback(None)
            return
        callback(PlannedPath(goal_xy=goal_xy, points=points, cost=path_length(points)))

    def _navigation_goal_response(self, future, generation, callback):
        goal_handle = self._future_result(future)
        if goal_handle is None:
            if self._navigation_guard.is_current(generation):
                callback(GOAL_STATUS_ABORTED)
            return
        if not self._navigation_guard.is_current(generation):
            self._cancel_handle(goal_handle)
            return
        if not goal_handle.accepted:
            callback(GOAL_STATUS_ABORTED)
            return
        self._navigation_goal_handle = goal_handle
        future = goal_handle.get_result_async()
        future.add_done_callback(
            lambda done: self._navigation_result(done, generation, callback)
        )

    def _navigation_result(self, future, generation, callback):
        if not self._navigation_guard.is_current(generation):
            return
        self._navigation_goal_handle = None
        wrapped_result = self._future_result(future)
        status = GOAL_STATUS_ABORTED if wrapped_result is None else wrapped_result.status
        callback(status)

    def _spin_goal_response(self, future, generation, callback):
        goal_handle = self._future_result(future)
        if goal_handle is None:
            if self._spin_guard.is_current(generation):
                callback(GOAL_STATUS_ABORTED)
            return
        if not self._spin_guard.is_current(generation):
            self._cancel_handle(goal_handle)
            return
        if not goal_handle.accepted:
            callback(GOAL_STATUS_ABORTED)
            return
        self._spin_goal_handle = goal_handle
        future = goal_handle.get_result_async()
        future.add_done_callback(
            lambda done: self._spin_result(done, generation, callback)
        )

    def _spin_result(self, future, generation, callback):
        if not self._spin_guard.is_current(generation):
            return
        self._spin_goal_handle = None
        wrapped_result = self._future_result(future)
        status = GOAL_STATUS_ABORTED if wrapped_result is None else wrapped_result.status
        callback(status)

    def _pose_stamped(self, x: float, y: float, yaw: float):
        from geometry_msgs.msg import PoseStamped

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self._node.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw * 0.5)
        pose.pose.orientation.w = math.cos(yaw * 0.5)
        return pose

    @staticmethod
    def _duration(seconds: float):
        from builtin_interfaces.msg import Duration

        duration = Duration()
        duration.sec = int(seconds)
        duration.nanosec = int((seconds - duration.sec) * 1e9)
        return duration

    @staticmethod
    def _cancel_handle(goal_handle):
        if goal_handle is not None:
            goal_handle.cancel_goal_async()

    def _future_result(self, future):
        try:
            return future.result()
        except Exception as exc:
            self._node.get_logger().warn(f'Nav2 action request failed: {exc}')
            return None
