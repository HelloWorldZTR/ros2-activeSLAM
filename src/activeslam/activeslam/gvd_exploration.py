"""Obstacle-only GVD helpers for fast online bootstrap exploration."""

import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .frontier_goal_utils import GridGeometry, normalize_angle


GridCell = Tuple[int, int]
Point = Tuple[float, float]


@dataclass(frozen=True)
class WorldBounds:
    """A coarse rectangular prior without wall or room structure."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @property
    def area(self) -> float:
        return max(0.0, self.max_x - self.min_x) * max(0.0, self.max_y - self.min_y)


@dataclass(frozen=True)
class GVDWeights:
    boundary_unknown: float = 2.0
    goal_distance: float = 1.0
    path_overlap_penalty: float = 2.0
    straightness: float = 1.0


@dataclass(frozen=True)
class GVDGoal:
    """One locally reachable skeleton goal with its A* path and score terms."""

    point: Point
    path: Tuple[Point, ...]
    utility: float
    boundary_unknown: float
    goal_distance: float
    path_overlap: float
    straightness: float


@dataclass(frozen=True)
class GVDTopology:
    """Obstacle-only medial skeleton and the corresponding A* traversability mask."""

    graph: nx.Graph
    geometry: GridGeometry
    skeleton: np.ndarray
    traversable: np.ndarray
    centerline_distance: Optional[np.ndarray] = None


@dataclass(frozen=True)
class RandomRecoveryMotion:
    """One bounded Nav2-owned random-walk recovery attempt."""

    yaw_delta: float
    distance: float
    speed: float


class TrajectorySweepTracker:
    """Rasterize swept scan area and historical trajectory tubes inside coarse bounds."""

    def __init__(
        self,
        bounds: WorldBounds,
        resolution: float,
        sweep_radius: float,
        overlap_radius: float,
    ):
        self.geometry = bounds_geometry(bounds, resolution)
        self.sweep_radius = max(0.0, sweep_radius)
        self.overlap_radius = max(0.0, overlap_radius)
        shape = self.geometry.height, self.geometry.width
        self.swept_mask = np.zeros(shape, dtype=bool)
        self.trajectory_mask = np.zeros(shape, dtype=bool)

    def mark_pose(self, point: Point):
        self.swept_mask |= disk_mask(self.geometry, point, self.sweep_radius)
        self.trajectory_mask |= disk_mask(self.geometry, point, self.overlap_radius)

    @property
    def ratio(self) -> float:
        return float(np.count_nonzero(self.swept_mask)) / float(self.swept_mask.size)


def resolve_gvd_bounds_path() -> Path:
    """Resolve the installed per-world coarse-bound configuration."""
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory('activeslam')) / 'config' / 'gvd_worlds.yaml'


def load_world_bounds(path: Path, world_name: str) -> WorldBounds:
    """Load one coarse rectangle from YAML."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError('gvd_gbsae requires python3-yaml or PyYAML.') from exc

    payload = yaml.safe_load(Path(path).read_text())
    values = None if not isinstance(payload, dict) else payload.get(world_name)
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f'Missing GVD bounds for world={world_name}.')
    bounds = WorldBounds(*(float(value) for value in values))
    if bounds.area <= 0.0:
        raise ValueError(f'Invalid GVD bounds for world={world_name}: {values!r}.')
    return bounds


def build_obstacle_gvd_topology(
    grid: np.ndarray,
    map_geometry: GridGeometry,
    bounds: WorldBounds,
    *,
    resolution: float,
    clearance: float,
    boundary_margin: float,
    support_vertex_spacing: float,
) -> GVDTopology:
    """Build a thinned medial skeleton while treating unknown and free as flat ground."""
    geometry, traversable = build_obstacle_traversability(
        grid,
        map_geometry,
        bounds,
        resolution=resolution,
        clearance=clearance,
        boundary_margin=boundary_margin,
    )
    skeleton = zhang_suen_thinning(traversable)
    graph = skeleton_to_graph(skeleton, geometry, support_vertex_spacing)
    return GVDTopology(
        graph,
        geometry,
        skeleton,
        traversable,
        distance_to_mask(skeleton, geometry.resolution),
    )


def build_obstacle_traversability(
    grid: np.ndarray,
    map_geometry: GridGeometry,
    bounds: WorldBounds,
    *,
    resolution: float,
    clearance: float,
    boundary_margin: float,
) -> Tuple[GridGeometry, np.ndarray]:
    """Build the cheap obstacle-only raster used for active-path invalidation."""
    geometry = bounds_geometry(bounds, resolution)
    obstacles = project_observed_obstacles(grid, map_geometry, geometry)
    blocked = inflate_square(obstacles, geometry, clearance)
    margin_cells = max(0, int(math.ceil(boundary_margin / geometry.resolution)))
    if margin_cells:
        blocked[:margin_cells, :] = True
        blocked[-margin_cells:, :] = True
        blocked[:, :margin_cells] = True
        blocked[:, -margin_cells:] = True
    traversable = np.logical_not(blocked)
    return geometry, traversable


def rank_gvd_goals(
    topology: GVDTopology,
    robot_xy: Point,
    robot_yaw: float,
    sweep: TrajectorySweepTracker,
    weights: GVDWeights,
    *,
    min_goal_distance: float,
    max_goal_distance: float,
    candidate_limit: int,
    skeleton_cost: float,
    off_skeleton_cost: float,
    centerline_distance_weight: float = 0.0,
    failed: Optional[Callable[[Point], bool]] = None,
) -> List[GVDGoal]:
    """Rank A*-reachable skeleton goals using border, distance, overlap, and heading terms."""
    candidates = _cheap_candidate_cells(
        topology,
        robot_xy,
        sweep,
        min_goal_distance,
        max_goal_distance,
        candidate_limit,
        failed,
    )
    start = nearest_traversable_cell(topology.traversable, topology.geometry, robot_xy)
    if start is None:
        return []
    goals = []
    for cell, boundary_unknown, distance_score in candidates:
        cells = astar_path(
            topology.traversable,
            topology.skeleton,
            start,
            cell,
            skeleton_cost,
            off_skeleton_cost,
            topology.centerline_distance,
            centerline_distance_weight,
        )
        if not cells:
            continue
        path = tuple(grid_to_world(item, topology.geometry) for item in cells)
        overlap = path_overlap_ratio(path, topology.geometry, sweep)
        straightness = initial_straightness(path, robot_yaw)
        utility = (
            weights.boundary_unknown * boundary_unknown
            + weights.goal_distance * distance_score
            - weights.path_overlap_penalty * overlap
            + weights.straightness * straightness
        )
        goals.append(
            GVDGoal(
                point=path[-1],
                path=path,
                utility=utility,
                boundary_unknown=boundary_unknown,
                goal_distance=distance_score,
                path_overlap=overlap,
                straightness=straightness,
            )
        )
    return sorted(
        goals,
        key=lambda goal: (
            -goal.utility,
            -goal.boundary_unknown,
            -goal.goal_distance,
            goal.path_overlap,
            -goal.straightness,
            goal.point,
        ),
    )


def path_crosses_obstacle(path: Sequence[Point], topology: GVDTopology) -> bool:
    """Return whether an updated obstacle raster invalidates a selected local path."""
    return path_crosses_traversability(path, topology.geometry, topology.traversable)


def path_crosses_traversability(
    path: Sequence[Point],
    geometry: GridGeometry,
    traversable: np.ndarray,
) -> bool:
    """Return whether a path leaves the raster or intersects its blocked cells."""
    return any(
        (cell := world_to_grid(point, geometry)) is None
        or not traversable[cell]
        for point in path
    )


def path_crosses_new_obstacle(
    path: Sequence[Point],
    geometry: GridGeometry,
    previous_traversable: np.ndarray,
    current_traversable: np.ndarray,
) -> bool:
    """Return whether a path intersects cells newly blocked since dispatch."""
    newly_blocked = np.logical_and(previous_traversable, np.logical_not(current_traversable))
    return path_crosses_traversability(path, geometry, np.logical_not(newly_blocked))


def path_suffix_from_nearest(
    path: Sequence[Point],
    robot_xy: Point,
    *,
    lookbehind_points: int = 1,
) -> Tuple[Point, ...]:
    """Trim already-traversed path points while retaining a small local overlap."""
    if not path:
        return ()
    nearest = min(
        range(len(path)),
        key=lambda index: (math.dist(path[index], robot_xy), index),
    )
    start = max(0, nearest - max(0, lookbehind_points))
    return tuple(path[start:])


def path_overlap_ratio(
    path: Sequence[Point],
    geometry: GridGeometry,
    sweep: TrajectorySweepTracker,
) -> float:
    """Measure historical trajectory-tube overlap, including path crossings."""
    if not path:
        return 0.0
    path_mask = np.zeros((geometry.height, geometry.width), dtype=bool)
    for point in path:
        path_mask |= disk_mask(geometry, point, sweep.overlap_radius)
    area = int(np.count_nonzero(path_mask))
    if area == 0:
        return 0.0
    return float(np.count_nonzero(np.logical_and(path_mask, sweep.trajectory_mask))) / area


def initial_straightness(path: Sequence[Point], robot_yaw: float) -> float:
    """Score whether the first non-trivial A* segment continues the robot heading."""
    if len(path) < 2:
        return 0.0
    origin = path[0]
    for target in path[1:]:
        if math.dist(origin, target) <= 1e-9:
            continue
        path_yaw = math.atan2(target[1] - origin[1], target[0] - origin[0])
        return 0.5 * (1.0 + math.cos(normalize_angle(path_yaw - robot_yaw)))
    return 0.0


def progress_watchdog_expired(
    last_progress_wall_time: Optional[float],
    timeout: float,
    now: float,
) -> bool:
    """Return whether navigation has made no effective translation for too long."""
    return (
        last_progress_wall_time is not None
        and timeout > 0.0
        and now - last_progress_wall_time > timeout
    )


def update_translation_progress(
    anchor_xy: Optional[Point],
    last_progress_wall_time: Optional[float],
    robot_xy: Point,
    min_distance: float,
    now: float,
) -> Tuple[Point, float, bool]:
    """Refresh the progress timestamp only after a meaningful position change."""
    if (
        anchor_xy is None
        or last_progress_wall_time is None
        or math.dist(anchor_xy, robot_xy) >= max(0.0, min_distance)
    ):
        return robot_xy, now, True
    return anchor_xy, last_progress_wall_time, False


def sample_random_recovery_motion(
    rng,
    *,
    min_abs_yaw: float,
    max_abs_yaw: float,
    distance: float,
    speed: float,
) -> RandomRecoveryMotion:
    """Sample a short collision-checked turn-and-drive recovery motion."""
    low = max(0.0, min(min_abs_yaw, max_abs_yaw))
    high = max(low, max(min_abs_yaw, max_abs_yaw))
    yaw = rng.uniform(low, high)
    if rng.random() < 0.5:
        yaw = -yaw
    return RandomRecoveryMotion(yaw, max(0.0, distance), max(0.0, speed))


def boundary_unknown_score(
    point: Point,
    robot_xy: Point,
    geometry: GridGeometry,
    sweep: TrajectorySweepTracker,
) -> float:
    """Score whether the goal direction reaches an unswept part of the coarse rectangle."""
    dx = point[0] - robot_xy[0]
    dy = point[1] - robot_xy[1]
    magnitude = math.hypot(dx, dy)
    if magnitude <= 1e-9:
        return 0.0
    hit = ray_rectangle_intersection(
        robot_xy,
        (dx / magnitude, dy / magnitude),
        geometry,
    )
    if hit is None:
        return 0.0
    disk = disk_mask(geometry, hit, sweep.sweep_radius)
    cells = int(np.count_nonzero(disk))
    if cells == 0:
        return 0.0
    return float(np.count_nonzero(np.logical_and(disk, np.logical_not(sweep.swept_mask)))) / cells


def ray_rectangle_intersection(
    origin: Point,
    direction: Point,
    geometry: GridGeometry,
) -> Optional[Point]:
    """Return the first positive ray hit on the coarse rectangular boundary."""
    min_x = geometry.origin_x
    max_x = geometry.origin_x + geometry.width * geometry.resolution
    min_y = geometry.origin_y
    max_y = geometry.origin_y + geometry.height * geometry.resolution
    hits = []
    if abs(direction[0]) > 1e-9:
        for x in (min_x, max_x):
            scale = (x - origin[0]) / direction[0]
            y = origin[1] + scale * direction[1]
            if scale > 1e-9 and min_y - 1e-9 <= y <= max_y + 1e-9:
                hits.append((scale, (x, y)))
    if abs(direction[1]) > 1e-9:
        for y in (min_y, max_y):
            scale = (y - origin[1]) / direction[1]
            x = origin[0] + scale * direction[0]
            if scale > 1e-9 and min_x - 1e-9 <= x <= max_x + 1e-9:
                hits.append((scale, (x, y)))
    if not hits:
        return None
    return min(hits)[1]


def astar_path(
    traversable: np.ndarray,
    skeleton: np.ndarray,
    start: GridCell,
    goal: GridCell,
    skeleton_cost: float,
    off_skeleton_cost: float,
    centerline_distance: Optional[np.ndarray] = None,
    centerline_distance_weight: float = 0.0,
) -> List[GridCell]:
    """Find a deterministic obstacle-avoiding path while preferring the GVD skeleton."""
    if not traversable[start] or not traversable[goal]:
        return []
    queue = [(0.0, 0.0, start)]
    costs = {start: 0.0}
    previous = {}
    while queue:
        _, cost, cell = heapq.heappop(queue)
        if cost > costs.get(cell, math.inf) + 1e-9:
            continue
        if cell == goal:
            return _reconstruct_path(previous, cell)
        for neighbor, distance in _neighbors(cell, traversable.shape):
            if not traversable[neighbor]:
                continue
            terrain = skeleton_cost if skeleton[neighbor] else off_skeleton_cost
            if centerline_distance is not None:
                terrain += centerline_distance_weight * centerline_distance[neighbor]
            next_cost = cost + distance * terrain
            if next_cost + 1e-9 >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = next_cost
            previous[neighbor] = cell
            heuristic = math.dist(neighbor, goal) * min(skeleton_cost, off_skeleton_cost)
            heapq.heappush(queue, (next_cost + heuristic, next_cost, neighbor))
    return []


def nearest_traversable_cell(
    traversable: np.ndarray,
    geometry: GridGeometry,
    point: Point,
) -> Optional[GridCell]:
    """Return the nearest passable cell to a world-space pose."""
    cell = world_to_grid(point, geometry)
    if cell is not None and traversable[cell]:
        return cell
    rows, cols = np.nonzero(traversable)
    if rows.size == 0:
        return None
    x = geometry.origin_x + (cols + 0.5) * geometry.resolution
    y = geometry.origin_y + (rows + 0.5) * geometry.resolution
    index = int(np.argmin(np.hypot(x - point[0], y - point[1])))
    return int(rows[index]), int(cols[index])


def bounds_geometry(bounds: WorldBounds, resolution: float) -> GridGeometry:
    """Create a fixed raster over the coarse rectangular prior."""
    if resolution <= 0.0 or bounds.area <= 0.0:
        raise ValueError('GVD raster resolution and prior area must be positive.')
    width = max(1, int(math.ceil((bounds.max_x - bounds.min_x) / resolution)))
    height = max(1, int(math.ceil((bounds.max_y - bounds.min_y) / resolution)))
    return GridGeometry(bounds.min_x, bounds.min_y, resolution, width, height)


def project_observed_obstacles(
    grid: np.ndarray,
    map_geometry: GridGeometry,
    target_geometry: GridGeometry,
) -> np.ndarray:
    """Project SLAM occupied cells into the coarse GVD raster."""
    projected = np.zeros((target_geometry.height, target_geometry.width), dtype=bool)
    occupied = np.argwhere(grid > 50)
    if occupied.size == 0:
        return projected
    xs = map_geometry.origin_x + (occupied[:, 1] + 0.5) * map_geometry.resolution
    ys = map_geometry.origin_y + (occupied[:, 0] + 0.5) * map_geometry.resolution
    cols = np.floor((xs - target_geometry.origin_x) / target_geometry.resolution).astype(int)
    rows = np.floor((ys - target_geometry.origin_y) / target_geometry.resolution).astype(int)
    inside = (
        (rows >= 0)
        & (rows < target_geometry.height)
        & (cols >= 0)
        & (cols < target_geometry.width)
    )
    projected[rows[inside], cols[inside]] = True
    return projected


def inflate_square(mask: np.ndarray, geometry: GridGeometry, radius: float) -> np.ndarray:
    """Inflate observed obstacles with a cheap square prefix-sum window."""
    cells = max(0, int(math.ceil(radius / geometry.resolution)))
    if cells == 0:
        return mask.copy()
    prefix = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    rows = np.arange(geometry.height)
    cols = np.arange(geometry.width)
    row0 = np.maximum(0, rows - cells)
    row1 = np.minimum(geometry.height, rows + cells + 1)
    col0 = np.maximum(0, cols - cells)
    col1 = np.minimum(geometry.width, cols + cells + 1)
    counts = (
        prefix[row1[:, None], col1[None, :]]
        - prefix[row0[:, None], col1[None, :]]
        - prefix[row1[:, None], col0[None, :]]
        + prefix[row0[:, None], col0[None, :]]
    )
    return counts > 0


def zhang_suen_thinning(mask: np.ndarray) -> np.ndarray:
    """Return a one-cell-wide skeleton using vectorized Zhang-Suen deletion masks."""
    image = np.asarray(mask, dtype=bool).copy()
    if image.ndim != 2 or image.size == 0:
        return image
    changed = True
    while changed:
        changed = False
        for first_step in (True, False):
            padded = np.pad(image, 1)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(neighbor.astype(np.uint8) for neighbor in neighbors)
            transitions = sum(
                np.logical_and(~neighbors[index], neighbors[(index + 1) % 8])
                for index in range(8)
            )
            if first_step:
                side_a = ~(p2 & p4 & p6)
                side_b = ~(p4 & p6 & p8)
            else:
                side_a = ~(p2 & p4 & p8)
                side_b = ~(p2 & p6 & p8)
            remove = image & (count >= 2) & (count <= 6) & (transitions == 1) & side_a & side_b
            if np.any(remove):
                image[remove] = False
                changed = True
    return image


def distance_to_mask(mask: np.ndarray, resolution: float) -> np.ndarray:
    """Approximate metric distance to a mask with deterministic eight-neighbor Dijkstra."""
    distances = np.full(mask.shape, np.inf, dtype=np.float64)
    queue = []
    for row, col in np.argwhere(mask):
        cell = int(row), int(col)
        distances[cell] = 0.0
        heapq.heappush(queue, (0.0, cell))
    while queue:
        distance, cell = heapq.heappop(queue)
        if distance > distances[cell] + 1e-9:
            continue
        for neighbor, step in _neighbors(cell, mask.shape):
            next_distance = distance + step * resolution
            if next_distance + 1e-9 >= distances[neighbor]:
                continue
            distances[neighbor] = next_distance
            heapq.heappush(queue, (next_distance, neighbor))
    return distances


def skeleton_to_graph(
    skeleton: np.ndarray,
    geometry: GridGeometry,
    support_vertex_spacing: float,
) -> nx.Graph:
    """Compress skeleton chains into a topo-metric graph for the GBSAE phase."""
    adjacency = skeleton_adjacency(skeleton)
    graph = nx.Graph()
    if not adjacency:
        return graph
    key_cells = {cell for cell, neighbors in adjacency.items() if len(neighbors) != 2}
    if not key_cells:
        key_cells.add(min(adjacency))
    nodes = {}

    def ensure_node(cell: GridCell) -> int:
        if cell not in nodes:
            nodes[cell] = len(nodes)
            x, y = grid_to_world(cell, geometry)
            graph.add_node(nodes[cell], x=x, y=y, kind='gvd')
        return nodes[cell]

    for cell in sorted(key_cells):
        ensure_node(cell)
    visited = set()
    spacing = max(geometry.resolution, support_vertex_spacing)
    for start in sorted(key_cells):
        for neighbor in adjacency[start]:
            edge = cell_edge(start, neighbor)
            if edge in visited:
                continue
            chain = [start]
            previous, current = start, neighbor
            visited.add(edge)
            while current not in key_cells:
                chain.append(current)
                choices = [cell for cell in adjacency[current] if cell != previous]
                if not choices:
                    break
                next_cell = choices[0]
                visited.add(cell_edge(current, next_cell))
                previous, current = current, next_cell
            chain.append(current)
            indices = _split_chain_indices(chain, geometry.resolution, spacing)
            for source_index, target_index in zip(indices, indices[1:]):
                source = ensure_node(chain[source_index])
                target = ensure_node(chain[target_index])
                if source == target:
                    continue
                distance = _chain_length(chain[source_index:target_index + 1], geometry.resolution)
                graph.add_edge(
                    source,
                    target,
                    weight=distance,
                    information_weight=1.0 / max(distance, 1e-6),
                    virtual=False,
                )
    return graph


def robot_component_graph(graph: nx.Graph, robot_xy: Point) -> nx.Graph:
    """Keep only the live skeleton component closest to the robot."""
    if graph.number_of_nodes() == 0:
        return graph.copy()
    start = min(
        graph.nodes,
        key=lambda node_id: (
            math.dist(
                robot_xy,
                (float(graph.nodes[node_id]['x']), float(graph.nodes[node_id]['y'])),
            ),
            node_id,
        ),
    )
    component = nx.node_connected_component(graph, start)
    return graph.subgraph(component).copy()


def gvd_to_marker_array(
    topology: GVDTopology,
    bounds: WorldBounds,
    active_path: Sequence[Point],
    frame_id: str,
    stamp,
):
    """Build compact RViz markers for bootstrap bounds, skeleton, and active path."""
    from geometry_msgs.msg import Point as MarkerPoint
    from visualization_msgs.msg import Marker, MarkerArray

    markers = MarkerArray()
    delete = Marker()
    delete.action = Marker.DELETEALL
    markers.markers.append(delete)

    boundary = _marker(Marker, frame_id, stamp, 'gvd_bounds', 0, Marker.LINE_STRIP)
    boundary.scale.x = 0.05
    boundary.color.a = 0.85
    boundary.color.r = 0.95
    boundary.color.g = 0.7
    for point in (
        (bounds.min_x, bounds.min_y),
        (bounds.max_x, bounds.min_y),
        (bounds.max_x, bounds.max_y),
        (bounds.min_x, bounds.max_y),
        (bounds.min_x, bounds.min_y),
    ):
        boundary.points.append(_marker_point(MarkerPoint, point, 0.08))
    markers.markers.append(boundary)

    skeleton = _marker(Marker, frame_id, stamp, 'gvd_skeleton', 1, Marker.LINE_LIST)
    skeleton.scale.x = 0.025
    skeleton.color.a = 0.75
    skeleton.color.g = 0.8
    skeleton.color.b = 1.0
    adjacency = skeleton_adjacency(topology.skeleton)
    for source, neighbors in adjacency.items():
        for target in neighbors:
            if source >= target:
                continue
            source_point = grid_to_world(source, topology.geometry)
            target_point = grid_to_world(target, topology.geometry)
            skeleton.points.append(_marker_point(MarkerPoint, source_point, 0.06))
            skeleton.points.append(_marker_point(MarkerPoint, target_point, 0.06))
    markers.markers.append(skeleton)

    path = _marker(Marker, frame_id, stamp, 'gvd_active_path', 2, Marker.LINE_STRIP)
    path.scale.x = 0.07
    path.color.a = 0.95
    path.color.r = 1.0
    path.color.g = 0.35
    for point in active_path:
        path.points.append(_marker_point(MarkerPoint, point, 0.12))
    markers.markers.append(path)
    return markers


def disk_mask(geometry: GridGeometry, point: Point, radius: float) -> np.ndarray:
    """Rasterize a disk while computing distances only inside its local window."""
    mask = np.zeros((geometry.height, geometry.width), dtype=bool)
    center = (
        int(math.floor((point[1] - geometry.origin_y) / geometry.resolution)),
        int(math.floor((point[0] - geometry.origin_x) / geometry.resolution)),
    )
    cells = max(0, int(math.ceil(radius / geometry.resolution)))
    row0 = max(0, center[0] - cells)
    row1 = min(geometry.height, center[0] + cells + 1)
    col0 = max(0, center[1] - cells)
    col1 = min(geometry.width, center[1] + cells + 1)
    if row0 >= row1 or col0 >= col1:
        return mask
    rows = np.arange(row0, row1)
    cols = np.arange(col0, col1)
    xs = geometry.origin_x + (cols + 0.5) * geometry.resolution
    ys = geometry.origin_y + (rows + 0.5) * geometry.resolution
    mask[row0:row1, col0:col1] = (
        np.hypot(xs[None, :] - point[0], ys[:, None] - point[1]) <= radius
    )
    return mask


def world_to_grid(point: Point, geometry: GridGeometry) -> Optional[GridCell]:
    row = int(math.floor((point[1] - geometry.origin_y) / geometry.resolution))
    col = int(math.floor((point[0] - geometry.origin_x) / geometry.resolution))
    if row < 0 or col < 0 or row >= geometry.height or col >= geometry.width:
        return None
    return row, col


def grid_to_world(cell: GridCell, geometry: GridGeometry) -> Point:
    return (
        geometry.origin_x + (cell[1] + 0.5) * geometry.resolution,
        geometry.origin_y + (cell[0] + 0.5) * geometry.resolution,
    )


def skeleton_adjacency(skeleton: np.ndarray):
    cells = {tuple(cell) for cell in np.argwhere(skeleton)}
    return {
        cell: sorted(
            (cell[0] + di, cell[1] + dj)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
            if (di or dj) and (cell[0] + di, cell[1] + dj) in cells
        )
        for cell in cells
    }


def _cheap_candidate_cells(
    topology: GVDTopology,
    robot_xy: Point,
    sweep: TrajectorySweepTracker,
    min_goal_distance: float,
    max_goal_distance: float,
    candidate_limit: int,
    failed: Optional[Callable[[Point], bool]],
):
    cells = np.argwhere(topology.skeleton)
    ranked = []
    for row, col in cells:
        cell = int(row), int(col)
        point = grid_to_world(cell, topology.geometry)
        distance = math.dist(robot_xy, point)
        if distance <= min_goal_distance or distance > max_goal_distance:
            continue
        if failed is not None and failed(point):
            continue
        boundary = boundary_unknown_score(point, robot_xy, topology.geometry, sweep)
        ranked.append((boundary + distance / max(max_goal_distance, 1e-6), cell, boundary))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [
        (cell, boundary, math.dist(robot_xy, grid_to_world(cell, topology.geometry))
         / max(max_goal_distance, 1e-6))
        for _, cell, boundary in ranked[:candidate_limit]
    ]


def _neighbors(cell: GridCell, shape):
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if not (di or dj):
                continue
            neighbor = cell[0] + di, cell[1] + dj
            if 0 <= neighbor[0] < shape[0] and 0 <= neighbor[1] < shape[1]:
                yield neighbor, math.hypot(di, dj)


def _reconstruct_path(previous, cell: GridCell) -> List[GridCell]:
    path = [cell]
    while cell in previous:
        cell = previous[cell]
        path.append(cell)
    return list(reversed(path))


def _split_chain_indices(chain: Sequence[GridCell], resolution: float, spacing: float):
    indices = [0]
    accumulated = 0.0
    for index in range(1, len(chain)):
        step = math.dist(chain[index - 1], chain[index]) * resolution
        if accumulated + step > spacing and index - 1 > indices[-1]:
            indices.append(index - 1)
            accumulated = 0.0
        accumulated += step
    if indices[-1] != len(chain) - 1:
        indices.append(len(chain) - 1)
    return indices


def _chain_length(chain: Sequence[GridCell], resolution: float) -> float:
    return sum(math.dist(source, target) * resolution for source, target in zip(chain, chain[1:]))


def cell_edge(source: GridCell, target: GridCell):
    return tuple(sorted((source, target)))


def _marker(marker_class, frame_id: str, stamp, namespace: str, marker_id: int, marker_type: int):
    marker = marker_class()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = namespace
    marker.id = marker_id
    marker.type = marker_type
    marker.action = marker.ADD
    marker.pose.orientation.w = 1.0
    return marker


def _marker_point(point_class, point: Point, z: float):
    marker_point = point_class()
    marker_point.x = point[0]
    marker_point.y = point[1]
    marker_point.z = z
    return marker_point
