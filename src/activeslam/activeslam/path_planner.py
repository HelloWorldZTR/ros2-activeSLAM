import heapq
import math
import random
from typing import List, Tuple

import numpy as np
from nav_msgs.msg import OccupancyGrid

SQRT2 = math.sqrt(2)


def inflate_obstacles(data: np.ndarray, resolution: float, inflation_radius: float) -> np.ndarray:
    h, w = data.shape
    r = int(inflation_radius / resolution)
    if r < 1:
        return data
    result = data.copy()
    for i in range(h):
        for j in range(w):
            if data[i, j] <= 50:
                continue
            ilo, ihi = max(0, i - r), min(h, i + r + 1)
            jlo, jhi = max(0, j - r), min(w, j + r + 1)
            result[ilo:ihi, jlo:jhi] = 100
    return result


class AStarPlanner:
    def __init__(
        self,
        goal_tolerance: float = 0.3,
        obstacle_inflation: float = 0.3,
        allow_unknown: bool = False,
    ):
        self.goal_tolerance = goal_tolerance
        self.obstacle_inflation = obstacle_inflation
        self.allow_unknown = allow_unknown

    def plan(
        self,
        grid_msg: OccupancyGrid,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
    ) -> Tuple[List[Tuple[float, float]], float, bool]:
        width = grid_msg.info.width
        height = grid_msg.info.height
        res = grid_msg.info.resolution
        ox = grid_msg.info.origin.position.x
        oy = grid_msg.info.origin.position.y

        raw = np.array(grid_msg.data, dtype=np.int8).reshape(height, width)
        data = inflate_obstacles(raw, res, self.obstacle_inflation)

        si, sj = self._w2g(start_xy[0], start_xy[1], ox, oy, res, width, height)
        gi, gj = self._w2g(goal_xy[0], goal_xy[1], ox, oy, res, width, height)

        if data[si, sj] > 50 or (data[si, sj] < 0 and not self.allow_unknown):
            return [], float('inf'), False

        goal_thresh = self.goal_tolerance / res

        open_heap = [(0.0, 0, si, sj)]
        counter = 1
        g_score = {(si, sj): 0.0}
        came_from = {}
        closed = set()

        while open_heap:
            _, _, ci, cj = heapq.heappop(open_heap)
            if (ci, cj) in closed:
                continue
            closed.add((ci, cj))

            if math.hypot(ci - gi, cj - gj) <= goal_thresh:
                path = self._reconstruct(came_from, (ci, cj))
                path_w = self._to_world(path, ox, oy, res)
                path_w = self._simplify(path_w)
                return path_w, g_score[(ci, cj)], True

            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < height and 0 <= nj < width):
                    continue
                val = data[ni, nj]
                if val > 50 or (val < 0 and not self.allow_unknown):
                    continue
                base = 1.0 if val == 0 else 2.0
                step_cost = base * (SQRT2 if di != 0 and dj != 0 else 1.0)
                ng = g_score[(ci, cj)] + step_cost

                if (ni, nj) not in g_score or ng < g_score[(ni, nj)]:
                    g_score[(ni, nj)] = ng
                    h = math.hypot(ni - gi, nj - gj)
                    heapq.heappush(open_heap, (ng + h, counter, ni, nj))
                    counter += 1
                    came_from[(ni, nj)] = (ci, cj)

        return [], float('inf'), False

    @staticmethod
    def _w2g(wx, wy, ox, oy, res, w, h):
        j = int((wx - ox) / res)
        i = int((wy - oy) / res)
        return max(0, min(h - 1, i)), max(0, min(w - 1, j))

    @staticmethod
    def _reconstruct(came_from, node):
        path = []
        while node is not None:
            path.append(node)
            node = came_from.get(node)
        return path[::-1]

    @staticmethod
    def _to_world(path, ox, oy, res):
        return [(ox + (j + 0.5) * res, oy + (i + 0.5) * res) for i, j in path]

    @staticmethod
    def _simplify(path):
        if len(path) <= 2:
            return path
        out = [path[0]]
        for i in range(1, len(path) - 1):
            px, py = out[-1]
            cx, cy = path[i]
            nx, ny = path[i + 1]
            if abs((cx - px) * (ny - py) - (cy - py) * (nx - px)) > 1e-6:
                out.append(path[i])
        out.append(path[-1])
        return out


class RRTPlanner:
    def __init__(
        self,
        max_iterations: int = 2000,
        step_size: float = 0.3,
        goal_bias: float = 0.1,
        goal_tolerance: float = 0.3,
        obstacle_inflation: float = 0.3,
        allow_unknown: bool = False,
    ):
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.goal_tolerance = goal_tolerance
        self.obstacle_inflation = obstacle_inflation
        self.allow_unknown = allow_unknown

    def plan(
        self,
        grid_msg: OccupancyGrid,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
    ) -> Tuple[List[Tuple[float, float]], float, bool]:
        width = grid_msg.info.width
        height = grid_msg.info.height
        res = grid_msg.info.resolution
        ox = grid_msg.info.origin.position.x
        oy = grid_msg.info.origin.position.y

        raw = np.array(grid_msg.data, dtype=np.int8).reshape(height, width)
        data = inflate_obstacles(raw, res, self.obstacle_inflation)

        start = (start_xy[0], start_xy[1])
        goal = (goal_xy[0], goal_xy[1])

        nodes = [start]
        parents = {start: None}

        for _ in range(self.max_iterations):
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sx = ox + random.random() * width * res
                sy = oy + random.random() * height * res
                sample = (sx, sy)

            nearest = min(nodes, key=lambda n: math.hypot(n[0] - sample[0], n[1] - sample[1]))

            dist = math.hypot(sample[0] - nearest[0], sample[1] - nearest[1])
            if dist < 1e-6:
                continue
            ratio = self.step_size / dist
            new = (
                nearest[0] + (sample[0] - nearest[0]) * ratio,
                nearest[1] + (sample[1] - nearest[1]) * ratio,
            )

            if self._collision_free(nearest, new, data, ox, oy, res, width, height):
                nodes.append(new)
                parents[new] = nearest

                if math.hypot(new[0] - goal[0], new[1] - goal[1]) < self.goal_tolerance:
                    path_w = self._extract_path(parents, new)
                    path_w = self._smooth(path_w, data, ox, oy, res, width, height)
                    cost = self._path_cost(path_w)
                    return path_w, cost, True

        # Return the closest branch for diagnostics, but do not mark it as a
        # valid plan because it did not reach the goal tolerance.
        best = min(nodes, key=lambda n: math.hypot(n[0] - goal[0], n[1] - goal[1]))
        path_w = self._extract_path(parents, best)
        path_w = self._smooth(path_w, data, ox, oy, res, width, height)
        cost = self._path_cost(path_w)
        return path_w, cost, False

    def _collision_free(self, a, b, data, ox, oy, res, width, height):
        steps = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / res))
        for k in range(steps + 1):
            frac = k / steps
            x = a[0] + (b[0] - a[0]) * frac
            y = a[1] + (b[1] - a[1]) * frac
            j = int((x - ox) / res)
            i = int((y - oy) / res)
            if not (0 <= i < height and 0 <= j < width):
                return False
            if data[i, j] > 50 or (data[i, j] < 0 and not self.allow_unknown):
                return False
        return True

    def _smooth(self, path, data, ox, oy, res, width, height):
        if len(path) <= 2:
            return path
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            for j in range(len(path) - 1, i, -1):
                if self._collision_free(path[i], path[j], data, ox, oy, res, width, height):
                    out.append(path[j])
                    i = j
                    break
        return out

    @staticmethod
    def _extract_path(parents, node):
        path = []
        while node is not None:
            path.append(node)
            node = parents.get(node)
        return path[::-1]

    @staticmethod
    def _path_cost(path):
        return sum(
            math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            for i in range(1, len(path))
        )


def create_planner(planner_type: str, **kwargs):
    inflation = kwargs.get('obstacle_inflation', 0.3)
    allow_unknown = kwargs.get('allow_unknown', False)
    if planner_type == 'astar':
        return AStarPlanner(
            goal_tolerance=kwargs.get('goal_tolerance', 0.5),
            obstacle_inflation=inflation,
            allow_unknown=allow_unknown,
        )
    elif planner_type == 'rrt':
        return RRTPlanner(
            max_iterations=kwargs.get('max_iterations', 2000),
            step_size=kwargs.get('rrt_step_size', 0.3),
            goal_bias=kwargs.get('goal_bias', 0.1),
            goal_tolerance=kwargs.get('goal_tolerance', 0.5),
            obstacle_inflation=inflation,
            allow_unknown=allow_unknown,
        )
    else:
        raise ValueError(f'Unknown planner type: {planner_type}')
