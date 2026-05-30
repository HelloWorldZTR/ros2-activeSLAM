import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


GridCell = Tuple[int, int]
Point = Tuple[float, float]


@dataclass(frozen=True)
class GridGeometry:
    origin_x: float
    origin_y: float
    resolution: float
    width: int
    height: int


@dataclass(frozen=True)
class SafeGoalSearchConfig:
    search_radius: float
    clearance: float
    standoff: float
    map_edge_clearance: float
    min_advance: float
    reach_radius: float
    point_sample_limit: int


@dataclass(frozen=True)
class SafeFrontierGoal:
    point: Point
    seed: GridCell


@dataclass(frozen=True)
class FailedGoal:
    x: float
    y: float
    deadline: float


class FailedGoalCooldown:
    """Temporarily reject goals near locations which recently failed."""

    def __init__(self, duration: float, radius: float):
        self.duration = duration
        self.radius = radius
        self._goals: List[FailedGoal] = []

    def mark(self, goal_xy: Point, now: float):
        if self.duration <= 0.0:
            return
        self._goals.append(FailedGoal(goal_xy[0], goal_xy[1], now + self.duration))

    def expire(self, now: float):
        self._goals = [goal for goal in self._goals if goal.deadline > now]

    def contains(self, goal_xy: Point, now: float) -> bool:
        self.expire(now)
        return any(
            math.hypot(goal_xy[0] - goal.x, goal_xy[1] - goal.y) < self.radius
            for goal in self._goals
        )

    @property
    def goals(self) -> Tuple[FailedGoal, ...]:
        return tuple(self._goals)


def is_goal_outside_reach_radius(
    origin_xy: Point,
    goal_xy: Point,
    reach_radius: float,
) -> bool:
    """Return whether Nav2 would need to move to reach this goal."""

    return math.hypot(goal_xy[0] - origin_xy[0], goal_xy[1] - origin_xy[1]) > reach_radius


def navigation_timed_out(started_at: Optional[float], timeout: float, now: float) -> bool:
    return started_at is not None and timeout > 0.0 and now - started_at >= timeout


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def open_edge_outward_normal(
    frontier_cells: Sequence[GridCell],
    seed: GridCell,
    geometry: GridGeometry,
    radius: float,
) -> Optional[Point]:
    """Estimate the local outward normal from map-edge neighbor directions."""

    if geometry.resolution <= 0.0 or radius < 0.0:
        return None

    seed_xy = _grid_to_world(seed, geometry)
    normal_x = 0.0
    normal_y = 0.0
    for cell in frontier_cells:
        cell_xy = _grid_to_world(cell, geometry)
        if math.hypot(cell_xy[0] - seed_xy[0], cell_xy[1] - seed_xy[1]) > radius:
            continue
        i, j = cell
        if i == 0:
            normal_y -= 1.0
        if i == geometry.height - 1:
            normal_y += 1.0
        if j == 0:
            normal_x -= 1.0
        if j == geometry.width - 1:
            normal_x += 1.0

    normal_x = math.copysign(1.0, normal_x) if abs(normal_x) >= 1e-6 else 0.0
    normal_y = math.copysign(1.0, normal_y) if abs(normal_y) >= 1e-6 else 0.0
    magnitude = math.hypot(normal_x, normal_y)
    if magnitude < 1e-6:
        return None
    return normal_x / magnitude, normal_y / magnitude


def potential_unknown_area(
    grid: np.ndarray,
    geometry: GridGeometry,
    center: GridCell,
    radius: float,
    include_outside_map: bool = False,
) -> float:
    """Return local unknown area, optionally treating out-of-map cells as unknown."""

    if geometry.resolution <= 0.0 or radius < 0.0:
        return 0.0

    center_i, center_j = center
    radius_cells = int(math.ceil(radius / geometry.resolution))
    unknown_cells = 0
    for i in range(center_i - radius_cells, center_i + radius_cells + 1):
        for j in range(center_j - radius_cells, center_j + radius_cells + 1):
            if math.hypot(i - center_i, j - center_j) * geometry.resolution > radius:
                continue
            if i < 0 or j < 0 or i >= geometry.height or j >= geometry.width:
                if include_outside_map:
                    unknown_cells += 1
            elif grid[i, j] == -1:
                unknown_cells += 1
    return unknown_cells * geometry.resolution * geometry.resolution


def select_safe_frontier_goal(
    grid: np.ndarray,
    geometry: GridGeometry,
    frontier_cells: Sequence[GridCell],
    robot_xy: Point,
    config: SafeGoalSearchConfig,
) -> Optional[SafeFrontierGoal]:
    """Find the best known-free standoff cell for a frontier cluster."""

    if geometry.resolution <= 0.0 or not frontier_cells:
        return None

    search_cells = max(1, int(math.ceil(config.search_radius / geometry.resolution)))
    clearance_cells = max(0, int(math.ceil(config.clearance / geometry.resolution)))
    edge_cells = max(0, int(math.ceil(config.map_edge_clearance / geometry.resolution)))
    seeds = _sample_frontier_cells(frontier_cells, robot_xy, geometry, config.point_sample_limit)
    best = None

    for seed in seeds:
        frontier_xy = _grid_to_world(seed, geometry)
        direction_x = frontier_xy[0] - robot_xy[0]
        direction_y = frontier_xy[1] - robot_xy[1]
        frontier_distance = math.hypot(direction_x, direction_y)
        if frontier_distance < 1e-6:
            continue

        direction_x /= frontier_distance
        direction_y /= frontier_distance
        nominal_xy = (
            frontier_xy[0] - direction_x * config.standoff,
            frontier_xy[1] - direction_y * config.standoff,
        )
        nominal_cell = _world_to_grid(nominal_xy, geometry)
        if nominal_cell is None:
            continue
        nominal_i, nominal_j = nominal_cell

        for di in range(-search_cells, search_cells + 1):
            for dj in range(-search_cells, search_cells + 1):
                cell = nominal_i + di, nominal_j + dj
                if not _is_known_free_with_clearance(grid, cell, clearance_cells):
                    continue
                if not _inside_map_with_margin(cell, geometry, edge_cells):
                    continue

                candidate_xy = _grid_to_world(cell, geometry)
                if not segment_is_obstacle_free(grid, geometry, frontier_xy, candidate_xy):
                    continue
                if not is_goal_outside_reach_radius(
                    robot_xy,
                    candidate_xy,
                    config.reach_radius,
                ):
                    continue
                advance = (
                    (candidate_xy[0] - robot_xy[0]) * direction_x
                    + (candidate_xy[1] - robot_xy[1]) * direction_y
                )
                if advance < config.min_advance:
                    continue

                distance_to_frontier = math.hypot(
                    candidate_xy[0] - frontier_xy[0],
                    candidate_xy[1] - frontier_xy[1],
                )
                distance_to_robot = math.hypot(
                    candidate_xy[0] - robot_xy[0],
                    candidate_xy[1] - robot_xy[1],
                )
                standoff_error = abs(distance_to_frontier - config.standoff)
                lateral_error = abs(
                    (candidate_xy[0] - robot_xy[0]) * direction_y
                    - (candidate_xy[1] - robot_xy[1]) * direction_x
                )
                score = (
                    standoff_error
                    + 0.1 * lateral_error
                    - 0.02 * advance
                    + 0.005 * distance_to_robot
                )
                if best is None or score < best[0]:
                    best = score, SafeFrontierGoal(point=candidate_xy, seed=seed)

    return None if best is None else best[1]


def segment_is_obstacle_free(
    grid: np.ndarray,
    geometry: GridGeometry,
    start_xy: Point,
    end_xy: Point,
) -> bool:
    """Return whether a map-bounded segment avoids occupied cells."""

    distance = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
    step = geometry.resolution * 0.5
    if step <= 0.0:
        return False
    sample_count = max(1, int(math.ceil(distance / step)))
    for index in range(sample_count + 1):
        ratio = index / sample_count
        cell = _world_to_grid(
            (
                start_xy[0] + (end_xy[0] - start_xy[0]) * ratio,
                start_xy[1] + (end_xy[1] - start_xy[1]) * ratio,
            ),
            geometry,
        )
        if cell is None or grid[cell] > 50:
            return False
    return True


def _sample_frontier_cells(
    frontier_cells: Iterable[GridCell],
    robot_xy: Point,
    geometry: GridGeometry,
    limit: int,
) -> List[GridCell]:
    cells = sorted(
        set(frontier_cells),
        key=lambda cell: math.hypot(
            _grid_to_world(cell, geometry)[0] - robot_xy[0],
            _grid_to_world(cell, geometry)[1] - robot_xy[1],
        ),
        reverse=True,
    )
    limit = max(1, limit)
    if len(cells) <= limit:
        return cells
    denominator = max(1, limit - 1)
    return [
        cells[round(index * (len(cells) - 1) / denominator)]
        for index in range(limit)
    ]


def _world_to_grid(point: Point, geometry: GridGeometry) -> Optional[GridCell]:
    i = int(math.floor((point[1] - geometry.origin_y) / geometry.resolution))
    j = int(math.floor((point[0] - geometry.origin_x) / geometry.resolution))
    if i < 0 or j < 0 or i >= geometry.height or j >= geometry.width:
        return None
    return i, j


def _grid_to_world(cell: GridCell, geometry: GridGeometry) -> Point:
    i, j = cell
    return (
        geometry.origin_x + (j + 0.5) * geometry.resolution,
        geometry.origin_y + (i + 0.5) * geometry.resolution,
    )


def _inside_map_with_margin(cell: GridCell, geometry: GridGeometry, margin: int) -> bool:
    i, j = cell
    return (
        margin <= i < geometry.height - margin
        and margin <= j < geometry.width - margin
    )


def _is_known_free_with_clearance(grid: np.ndarray, cell: GridCell, clearance: int) -> bool:
    i, j = cell
    height, width = grid.shape
    if i < 0 or j < 0 or i >= height or j >= width or grid[i, j] != 0:
        return False
    i0, i1 = max(0, i - clearance), min(height, i + clearance + 1)
    j0, j1 = max(0, j - clearance), min(width, j + clearance + 1)
    return not np.any(grid[i0:i1, j0:j1] > 50)
