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
    reach_radius: float
    point_sample_limit: int


@dataclass(frozen=True)
class SafeFrontierGoal:
    point: Point
    seed: GridCell
    outward_normal: Optional[Point] = None


@dataclass(frozen=True)
class PreparedSafeGoalGrid:
    """Cache cells which can be used as known-free frontier goals."""

    valid_goal_mask: np.ndarray
    clearance_cells: int
    edge_cells: int


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


def unknown_frontier_outward_normal(
    grid: np.ndarray,
    frontier_cells: Sequence[GridCell],
    seed: GridCell,
    geometry: GridGeometry,
    radius: float,
) -> Optional[Point]:
    """Estimate a local frontier normal from adjacent unknown-cell votes."""
    if (
        geometry.resolution <= 0.0
        or radius < 0.0
        or grid.shape != (geometry.height, geometry.width)
    ):
        return None

    seed_xy = _grid_to_world(seed, geometry)
    normal_x = 0.0
    normal_y = 0.0
    for cell in frontier_cells:
        cell_xy = _grid_to_world(cell, geometry)
        if math.hypot(cell_xy[0] - seed_xy[0], cell_xy[1] - seed_xy[1]) > radius:
            continue
        i, j = cell
        for di, dj in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            ni, nj = i + di, j + dj
            if (
                0 <= ni < geometry.height
                and 0 <= nj < geometry.width
                and grid[ni, nj] == -1
            ):
                normal_x += dj
                normal_y += di

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

    radius_cells = int(math.ceil(radius / geometry.resolution))
    center_i, center_j = center
    rows = np.arange(center_i - radius_cells, center_i + radius_cells + 1)
    cols = np.arange(center_j - radius_cells, center_j + radius_cells + 1)
    circle_mask = (
        np.hypot(rows[:, None] - center_i, cols[None, :] - center_j)
        * geometry.resolution
        <= radius
    )
    inside_rows = np.logical_and(rows >= 0, rows < geometry.height)
    inside_cols = np.logical_and(cols >= 0, cols < geometry.width)
    inside_mask = np.logical_and(inside_rows[:, None], inside_cols[None, :])
    unknown_cells = 0
    if np.any(inside_rows) and np.any(inside_cols):
        row_indices = np.flatnonzero(inside_rows)
        col_indices = np.flatnonzero(inside_cols)
        local_grid = grid[np.ix_(rows[row_indices], cols[col_indices])]
        local_circle = circle_mask[np.ix_(row_indices, col_indices)]
        unknown_cells = int(np.count_nonzero(np.logical_and(local_circle, local_grid == -1)))
    if include_outside_map:
        unknown_cells += int(np.count_nonzero(np.logical_and(circle_mask, ~inside_mask)))
    return unknown_cells * geometry.resolution * geometry.resolution


def prepare_safe_goal_grid(
    grid: np.ndarray,
    geometry: GridGeometry,
    config: SafeGoalSearchConfig,
) -> PreparedSafeGoalGrid:
    """Precompute known-free cells with the requested obstacle clearance."""
    clearance_cells = max(0, int(math.ceil(config.clearance / geometry.resolution)))
    edge_cells = max(0, int(math.ceil(config.map_edge_clearance / geometry.resolution)))
    occupied = grid > 50
    prefix = np.pad(occupied.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    rows = np.arange(geometry.height)
    cols = np.arange(geometry.width)
    row0 = np.maximum(0, rows - clearance_cells)
    row1 = np.minimum(geometry.height, rows + clearance_cells + 1)
    col0 = np.maximum(0, cols - clearance_cells)
    col1 = np.minimum(geometry.width, cols + clearance_cells + 1)
    occupied_counts = (
        prefix[row1[:, None], col1[None, :]]
        - prefix[row0[:, None], col1[None, :]]
        - prefix[row1[:, None], col0[None, :]]
        + prefix[row0[:, None], col0[None, :]]
    )
    inside_margin = np.logical_and(
        np.logical_and(rows[:, None] >= edge_cells, rows[:, None] < geometry.height - edge_cells),
        np.logical_and(cols[None, :] >= edge_cells, cols[None, :] < geometry.width - edge_cells),
    )
    return PreparedSafeGoalGrid(
        valid_goal_mask=np.logical_and.reduce((grid == 0, occupied_counts == 0, inside_margin)),
        clearance_cells=clearance_cells,
        edge_cells=edge_cells,
    )


def select_safe_frontier_goal(
    grid: np.ndarray,
    geometry: GridGeometry,
    frontier_cells: Sequence[GridCell],
    robot_xy: Point,
    config: SafeGoalSearchConfig,
    prepared_grid: Optional[PreparedSafeGoalGrid] = None,
    allowed_goal_mask: Optional[np.ndarray] = None,
) -> Optional[SafeFrontierGoal]:
    """Find the best known-free standoff cell for a frontier cluster."""
    if geometry.resolution <= 0.0 or not frontier_cells:
        return None
    if allowed_goal_mask is not None and allowed_goal_mask.shape != grid.shape:
        raise ValueError('Allowed goal mask shape does not match occupancy grid.')

    search_cells = max(1, int(math.ceil(config.search_radius / geometry.resolution)))
    clearance_cells = max(0, int(math.ceil(config.clearance / geometry.resolution)))
    edge_cells = max(0, int(math.ceil(config.map_edge_clearance / geometry.resolution)))
    if (
        prepared_grid is None
        or prepared_grid.valid_goal_mask.shape != grid.shape
        or prepared_grid.clearance_cells != clearance_cells
        or prepared_grid.edge_cells != edge_cells
    ):
        prepared_grid = prepare_safe_goal_grid(grid, geometry, config)
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

        min_i = max(0, nominal_i - search_cells)
        max_i = min(geometry.height, nominal_i + search_cells + 1)
        min_j = max(0, nominal_j - search_cells)
        max_j = min(geometry.width, nominal_j + search_cells + 1)
        local_valid = prepared_grid.valid_goal_mask[min_i:max_i, min_j:max_j]
        if allowed_goal_mask is not None:
            local_valid = np.logical_and(
                local_valid,
                allowed_goal_mask[min_i:max_i, min_j:max_j],
            )
        local_rows, local_cols = np.nonzero(local_valid)
        if local_rows.size == 0:
            continue

        rows = local_rows + min_i
        cols = local_cols + min_j
        xs = geometry.origin_x + (cols + 0.5) * geometry.resolution
        ys = geometry.origin_y + (rows + 0.5) * geometry.resolution
        robot_dx = xs - robot_xy[0]
        robot_dy = ys - robot_xy[1]
        distance_to_robot = np.hypot(robot_dx, robot_dy)
        advance = robot_dx * direction_x + robot_dy * direction_y
        valid = distance_to_robot > config.reach_radius
        if not np.any(valid):
            continue

        rows = rows[valid]
        cols = cols[valid]
        xs = xs[valid]
        ys = ys[valid]
        distance_to_robot = distance_to_robot[valid]
        advance = advance[valid]
        scores = (
            np.abs(np.hypot(xs - frontier_xy[0], ys - frontier_xy[1]) - config.standoff)
            + 0.1 * np.abs(robot_dx[valid] * direction_y - robot_dy[valid] * direction_x)
            - 0.02 * advance
            + 0.005 * distance_to_robot
        )
        for index in np.argsort(scores, kind='stable'):
            candidate_xy = float(xs[index]), float(ys[index])
            if segment_is_obstacle_free(grid, geometry, frontier_xy, candidate_xy):
                score = float(scores[index])
                if best is None or score < best[0]:
                    best = score, SafeFrontierGoal(point=candidate_xy, seed=seed)
                break

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
