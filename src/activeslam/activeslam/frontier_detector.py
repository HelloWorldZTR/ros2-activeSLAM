import math
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from nav_msgs.msg import OccupancyGrid


@dataclass
class FrontierCluster:
    centroid_x: float
    centroid_y: float
    size: int
    points: Optional[List[Tuple[float, float]]] = None


class FrontierDetector:
    """RRT-based frontier detector.

    The tree grows only through known free space. When an attempted edge first
    touches unknown space, that first unknown point is emitted as a frontier
    candidate. This mirrors the active_graph_slam/rrt_exploration detector and
    is better suited to large open areas than scanning only adjacent grid cells.
    """

    def __init__(
        self,
        min_frontier_size: int = 5,
        iterations: int = 800,
        step_size: float = 1.0,
        cluster_radius: float = 0.6,
        max_frontier_points: int = 120,
        map_padding: float = 2.0,
        seed: Optional[int] = None,
    ):
        self.min_frontier_size = min_frontier_size
        self.iterations = iterations
        self.step_size = step_size
        self.cluster_radius = cluster_radius
        self.max_frontier_points = max_frontier_points
        self.map_padding = map_padding
        self.rng = random.Random(seed)

    def detect(
        self,
        grid_msg: OccupancyGrid,
        start_xy: Tuple[float, float],
    ) -> Tuple[List[FrontierCluster], np.ndarray]:
        width = grid_msg.info.width
        height = grid_msg.info.height
        resolution = grid_msg.info.resolution
        origin_x = grid_msg.info.origin.position.x
        origin_y = grid_msg.info.origin.position.y

        data = np.array(grid_msg.data, dtype=np.int8).reshape(height, width)
        frontier_mask = np.zeros((height, width), dtype=bool)

        start = self._valid_start(start_xy, data, origin_x, origin_y, resolution)
        if start is None:
            return [], frontier_mask

        tree = [start]
        frontier_points: List[Tuple[float, float]] = []
        for _ in range(max(1, int(self.iterations))):
            sample_padding = max(0.0, self.map_padding)
            sample = (
                origin_x
                - sample_padding
                + self.rng.random() * (width * resolution + 2.0 * sample_padding),
                origin_y
                - sample_padding
                + self.rng.random() * (height * resolution + 2.0 * sample_padding),
            )
            nearest = min(tree, key=lambda p: _distance(p, sample))
            new_point = self._steer(nearest, sample, self.step_size)
            status, hit = self._trace_edge(
                nearest,
                new_point,
                data,
                origin_x,
                origin_y,
                resolution,
            )
            if status == 'free':
                tree.append(new_point)
            elif status == 'unknown' and hit is not None:
                frontier_points.append(hit)
                cell = self._world_to_grid(hit[0], hit[1], origin_x, origin_y, resolution)
                if cell is not None:
                    i, j = cell
                    if 0 <= i < height and 0 <= j < width:
                        frontier_mask[i, j] = True
                if len(frontier_points) >= self.max_frontier_points:
                    break

        frontier_points.extend(
            self._map_edge_frontier_points(
                data,
                origin_x,
                origin_y,
                resolution,
                self.max_frontier_points,
            )
        )
        clusters = self._cluster_points(frontier_points, self.cluster_radius)
        result = []
        for points in clusters:
            if len(points) < self.min_frontier_size:
                continue
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            result.append(
                FrontierCluster(
                    centroid_x=cx,
                    centroid_y=cy,
                    size=len(points),
                    points=list(points),
                )
            )

        return result, frontier_mask

    def _valid_start(
        self,
        start_xy: Tuple[float, float],
        data: np.ndarray,
        origin_x: float,
        origin_y: float,
        resolution: float,
    ) -> Optional[Tuple[float, float]]:
        cell = self._world_to_grid(start_xy[0], start_xy[1], origin_x, origin_y, resolution)
        if cell is None:
            return None
        i, j = cell
        height, width = data.shape
        if not (0 <= i < height and 0 <= j < width):
            i = min(max(i, 0), height - 1)
            j = min(max(j, 0), width - 1)
        if data[i, j] == 0:
            if 0 <= cell[0] < height and 0 <= cell[1] < width:
                return start_xy

        nearest = self._nearest_free_cell(data, i, j)
        if nearest is None:
            return None
        ni, nj = nearest
        return (
            origin_x + (nj + 0.5) * resolution,
            origin_y + (ni + 0.5) * resolution,
        )

    def _nearest_free_cell(self, data: np.ndarray, start_i: int, start_j: int):
        height, width = data.shape
        visited = np.zeros_like(data, dtype=bool)
        queue = deque([(start_i, start_j)])
        visited[start_i, start_j] = True
        while queue:
            i, j = queue.popleft()
            if data[i, j] == 0:
                return i, j
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width and not visited[ni, nj]:
                    visited[ni, nj] = True
                    queue.append((ni, nj))
        return None

    def _trace_edge(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        data: np.ndarray,
        origin_x: float,
        origin_y: float,
        resolution: float,
    ) -> Tuple[str, Optional[Tuple[float, float]]]:
        height, width = data.shape
        step = max(resolution * 0.5, 1e-3)
        steps = max(1, int(math.ceil(_distance(start, end) / step)))

        for k in range(1, steps + 1):
            frac = k / steps
            x = start[0] + (end[0] - start[0]) * frac
            y = start[1] + (end[1] - start[1]) * frac
            cell = self._world_to_grid(x, y, origin_x, origin_y, resolution)
            if cell is None:
                return 'blocked', None
            i, j = cell
            if not (0 <= i < height and 0 <= j < width):
                return 'unknown', (x, y)
            value = data[i, j]
            if value > 50:
                return 'blocked', None
            if value == -1:
                return 'unknown', (x, y)

        return 'free', None

    def _map_edge_frontier_points(
        self,
        data: np.ndarray,
        origin_x: float,
        origin_y: float,
        resolution: float,
        limit: int,
    ) -> List[Tuple[float, float]]:
        height, width = data.shape
        if height == 0 or width == 0:
            return []

        stride = max(1, int(round(max(self.cluster_radius, resolution) / resolution)))
        cells = []
        for j in range(0, width, stride):
            cells.append((0, j, 0.0, -1.0))
            cells.append((height - 1, j, 0.0, 1.0))
        for i in range(0, height, stride):
            cells.append((i, 0, -1.0, 0.0))
            cells.append((i, width - 1, 1.0, 0.0))

        points = []
        for i, j, normal_x, normal_y in cells:
            if data[i, j] != 0:
                continue
            wx = origin_x + (j + 0.5) * resolution + normal_x * resolution
            wy = origin_y + (i + 0.5) * resolution + normal_y * resolution
            points.append((wx, wy))

        if limit <= 0 or len(points) <= limit:
            return points

        sampled = []
        denom = max(1, limit - 1)
        for k in range(limit):
            index = round(k * (len(points) - 1) / denom)
            sampled.append(points[index])
        return sampled

    @staticmethod
    def _steer(
        start: Tuple[float, float],
        target: Tuple[float, float],
        max_distance: float,
    ) -> Tuple[float, float]:
        distance = _distance(start, target)
        if distance <= max_distance or distance < 1e-9:
            return target
        ratio = max_distance / distance
        return (
            start[0] + (target[0] - start[0]) * ratio,
            start[1] + (target[1] - start[1]) * ratio,
        )

    @staticmethod
    def _world_to_grid(
        x: float,
        y: float,
        origin_x: float,
        origin_y: float,
        resolution: float,
    ) -> Optional[Tuple[int, int]]:
        if resolution <= 0.0:
            return None
        j = int(math.floor((x - origin_x) / resolution))
        i = int(math.floor((y - origin_y) / resolution))
        return i, j

    @staticmethod
    def _cluster_points(
        points: Sequence[Tuple[float, float]],
        radius: float,
    ) -> List[List[Tuple[float, float]]]:
        visited = [False] * len(points)
        clusters = []
        for index in range(len(points)):
            if visited[index]:
                continue
            cluster = []
            queue = deque([index])
            visited[index] = True
            while queue:
                current = queue.popleft()
                cluster.append(points[current])
                for other in range(len(points)):
                    if visited[other]:
                        continue
                    if _distance(points[current], points[other]) <= radius:
                        visited[other] = True
                        queue.append(other)
            clusters.append(cluster)
        return clusters


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
