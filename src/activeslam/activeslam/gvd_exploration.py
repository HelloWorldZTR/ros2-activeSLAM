"""Obstacle-only GVD helpers for fast online bootstrap exploration."""

import heapq
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Set, Tuple

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
    repair_stats: Optional['TopologyRepairStats'] = None
    compression_stats: Optional['TopologyCompressionStats'] = None
    cycle_suppression_stats: Optional['UnknownCycleSuppressionStats'] = None


@dataclass(frozen=True)
class TopologyConnection:
    """A collision-free bridge supplied by the GVD layer or grid A* fallback."""

    mode: str
    path: Tuple[GridCell, ...]
    length: float


@dataclass(frozen=True)
class TopologyRepairStats:
    """Summary of switching connections added while repairing a topology."""

    gvd_edges: int = 0
    astar_edges: int = 0
    unresolved_components: int = 0


@dataclass(frozen=True)
class TopologyCompressionStats:
    """Summary of deterministic nearby-vertex clustering."""

    before_vertices: int = 0
    after_vertices: int = 0


@dataclass(frozen=True)
class UnknownCycleSuppressionStats:
    """Summary of MST pruning inside unknown-heavy topology regions."""

    unconfident_vertices: int = 0
    removed_edges: int = 0


@dataclass(frozen=True)
class _CachedTopologyConnection:
    connection: Optional[TopologyConnection]
    revision: object


class TopologyConnectionCache:
    """LRU cache for GVD-first, bidirectional-A*-fallback connectivity queries."""

    def __init__(self, maxsize: int = 4096):
        self.maxsize = max(1, int(maxsize))
        self._entries = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    def revision_for(
        self,
        traversable: np.ndarray,
        revision=None,
        astar_traversable: Optional[np.ndarray] = None,
    ):
        """Return a stable revision token for negative cache entries."""
        if revision is not None:
            return revision
        fallback = traversable if astar_traversable is None else astar_traversable
        return (
            traversable.shape,
            hash(np.asarray(traversable, dtype=bool).tobytes()),
            fallback.shape,
            hash(np.asarray(fallback, dtype=bool).tobytes()),
        )

    def connection(
        self,
        traversable: np.ndarray,
        skeleton: np.ndarray,
        start: GridCell,
        goal: GridCell,
        *,
        resolution: float,
        revision=None,
        astar_traversable: Optional[np.ndarray] = None,
    ) -> Optional[TopologyConnection]:
        """Return a cached collision-free connection, preferring the GVD layer."""
        key = cell_edge(start, goal)
        reverse = key != (start, goal)
        fallback = traversable if astar_traversable is None else astar_traversable
        if fallback.shape != traversable.shape:
            raise ValueError('A* fallback mask shape does not match traversability mask.')
        revision = self.revision_for(traversable, revision, fallback)
        cached = self._entries.get(key)
        if cached is not None and self._cache_entry_valid(
            cached,
            traversable,
            skeleton,
            revision,
            fallback,
        ):
            self.hits += 1
            self._entries.move_to_end(key)
            return self._orient_connection(cached.connection, reverse)
        if cached is not None:
            self.invalidations += 1
            del self._entries[key]

        self.misses += 1
        path = bidirectional_astar_path(
            np.logical_and(traversable, skeleton),
            key[0],
            key[1],
            prevent_corner_cutting=False,
        )
        mode = 'gvd'
        if not path:
            path = bidirectional_astar_path(fallback, key[0], key[1])
            mode = 'astar'
        connection = None
        if path:
            connection = TopologyConnection(
                mode,
                tuple(path),
                _chain_length(path, resolution),
            )
        self._entries[key] = _CachedTopologyConnection(connection, revision)
        self._entries.move_to_end(key)
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)
        return self._orient_connection(connection, reverse)

    @staticmethod
    def _cache_entry_valid(
        cached: _CachedTopologyConnection,
        traversable: np.ndarray,
        skeleton: np.ndarray,
        revision,
        astar_traversable: np.ndarray,
    ) -> bool:
        connection = cached.connection
        if connection is None:
            return cached.revision == revision
        mask = astar_traversable if connection.mode == 'astar' else np.logical_and(
            traversable,
            skeleton,
        )
        return _path_cells_traversable(
            connection.path,
            mask,
            prevent_corner_cutting=connection.mode == 'astar',
        )

    @staticmethod
    def _orient_connection(
        connection: Optional[TopologyConnection],
        reverse: bool,
    ) -> Optional[TopologyConnection]:
        if connection is None or not reverse:
            return connection
        return TopologyConnection(
            connection.mode,
            tuple(reversed(connection.path)),
            connection.length,
        )


@dataclass(frozen=True)
class RandomRecoveryMotion:
    """One bounded Nav2-owned random-walk recovery attempt."""

    yaw_delta: float
    distance: float
    speed: float


@dataclass(frozen=True)
class HierarchicalGVDTarget:
    """One unexplored macro GVD vertex selected for a coarse exploration step."""

    vertex_id: int
    point: Point
    travel_cost: float
    utility: float


class HierarchicalGVDTracker:
    """Track macro exploration and local-cleanup state across live GVD rebuilds."""

    def __init__(self, migration_radius: float):
        self.migration_radius = max(0.0, migration_radius)
        self.graph = nx.Graph()
        self.explored_points: List[Point] = []
        self.cleared_points: List[Point] = []
        self.explored_vertices: Set[int] = set()
        self.cleared_vertices: Set[int] = set()
        self.previous_vertex: Optional[int] = None
        self.active_vertex: Optional[int] = None
        self.previous_point: Optional[Point] = None
        self.active_point: Optional[Point] = None
        self.graph_version = 0
        self.route_dirty = True
        self.route_targets: Tuple[int, ...] = ()
        self.transit_route: Tuple[int, ...] = ()
        self.route_index = 0

    def update_graph(self, graph: nx.Graph):
        """Install a rebuilt graph while migrating states by nearby world positions."""
        self.graph = graph.copy()
        self.graph_version += 1
        self.explored_vertices = self._matching_vertices(self.explored_points)
        self.cleared_vertices = self._matching_vertices(self.cleared_points)
        self.previous_vertex = self._nearest_vertex(self.previous_point)
        self.active_vertex = self._nearest_vertex(self.active_point)
        self.route_targets = ()
        self.transit_route = ()
        self.route_index = 0
        self.route_dirty = True

    def mark_route_dirty(self):
        """Record that the latest SLAM map requires a fresh macro route."""
        self.route_dirty = True

    def remap_target(self, target: Optional[HierarchicalGVDTarget]):
        """Migrate an in-flight target to a nearby vertex after a live graph rebuild."""
        if target is None or self.graph.number_of_nodes() == 0:
            return target
        vertex_id = self._nearest_graph_vertex(target.point)
        point = _graph_node_point(self.graph, vertex_id)
        if math.dist(point, target.point) > self.migration_radius:
            return target
        return HierarchicalGVDTarget(
            vertex_id,
            point,
            target.travel_cost,
            target.utility,
        )

    def rebuild_route(
        self,
        robot_xy: Point,
        failed: Optional[Callable[[Point], bool]] = None,
    ) -> Tuple[int, ...]:
        """Create an open TSP walk over uncleared vertices and expand graph transit."""
        self.route_dirty = False
        self.route_index = 0
        if self.graph.number_of_nodes() == 0:
            self.route_targets = ()
            self.transit_route = ()
            return self.transit_route
        start = self._nearest_graph_vertex(robot_xy)
        targets = tuple(
            node_id
            for node_id in sorted(set(self.graph.nodes) - self.cleared_vertices)
            if failed is None or not failed(_graph_node_point(self.graph, node_id))
        )
        self.route_targets = targets
        if not targets:
            self.transit_route = ()
            return self.transit_route
        if len(targets) == 1:
            tsp_route = list(targets)
        elif len(targets) == 2:
            tsp_route = nx.shortest_path(
                self.graph,
                targets[0],
                targets[1],
                weight='weight',
            )
        else:
            tsp_route = nx.approximation.traveling_salesman_problem(
                self.graph,
                nodes=targets,
                cycle=False,
                weight='weight',
            )
        if self._route_endpoint_cost(start, tsp_route[-1]) < self._route_endpoint_cost(
            start,
            tsp_route[0],
        ):
            tsp_route.reverse()
        prefix = nx.shortest_path(self.graph, start, tsp_route[0], weight='weight')
        self.transit_route = tuple(_deduplicate_adjacent(prefix[:-1] + tsp_route))
        return self.transit_route

    def select_macro_target(
        self,
        robot_xy: Point,
        failed: Optional[Callable[[Point], bool]] = None,
        arrival_radius: float = 0.25,
    ) -> Optional[HierarchicalGVDTarget]:
        """Return the next step along the expanded open-TSP macro route."""
        if self.graph.number_of_nodes() == 0 or not self.transit_route:
            return None
        while self.route_index < len(self.transit_route):
            node_id = self.transit_route[self.route_index]
            point = _graph_node_point(self.graph, node_id)
            if math.dist(robot_xy, point) <= max(0.0, arrival_radius):
                self._mark_vertex_reached(node_id)
                self.route_index += 1
                continue
            if failed is not None and failed(point):
                return None
            start = self._nearest_graph_vertex(robot_xy)
            try:
                travel_cost = nx.shortest_path_length(
                    self.graph,
                    start,
                    node_id,
                    weight='weight',
                )
            except nx.NetworkXNoPath:
                return None
            travel_cost += math.dist(robot_xy, _graph_node_point(self.graph, start))
            utility = 1.0 / (travel_cost + 0.1)
            return HierarchicalGVDTarget(
                node_id,
                point,
                travel_cost,
                utility,
            )
        return None

    def mark_reached(self, vertex_id: int):
        """Mark a reached macro vertex and its incoming edge as explored."""
        if vertex_id not in self.graph:
            return
        self._mark_vertex_reached(vertex_id)
        if (
            self.route_index < len(self.transit_route)
            and self.transit_route[self.route_index] == vertex_id
        ):
            self.route_index += 1

    def _mark_vertex_reached(self, vertex_id: int):
        """Record one reached transit step without changing the route cursor."""
        if self.previous_vertex in self.graph and self.graph.has_edge(
            self.previous_vertex,
            vertex_id,
        ):
            self.graph.edges[self.previous_vertex, vertex_id]['explored'] = True
        self.graph.nodes[vertex_id]['explored'] = True
        self.explored_vertices.add(vertex_id)
        self._append_unique_point(self.explored_points, _graph_node_point(self.graph, vertex_id))
        self.previous_vertex = vertex_id
        self.active_vertex = vertex_id
        self.previous_point = _graph_node_point(self.graph, vertex_id)
        self.active_point = self.previous_point

    def should_clear_local(self, vertex_id: int) -> bool:
        """Return whether the expanded route will never revisit *vertex_id*."""
        if vertex_id not in self.graph or vertex_id in self.cleared_vertices:
            return False
        return vertex_id not in self.transit_route[self.route_index:]

    def mark_local_cleared(self, vertex_id: Optional[int]):
        if vertex_id is None or vertex_id not in self.graph:
            return
        self.graph.nodes[vertex_id]['cleared'] = True
        self.cleared_vertices.add(vertex_id)
        self._append_unique_point(self.cleared_points, _graph_node_point(self.graph, vertex_id))
        self.route_dirty = True

    @property
    def has_unexplored_vertices(self) -> bool:
        return bool(set(self.graph.nodes) - self.explored_vertices)

    @property
    def has_uncleared_vertices(self) -> bool:
        return bool(set(self.graph.nodes) - self.cleared_vertices)

    @property
    def remaining_route(self) -> Tuple[int, ...]:
        return self.transit_route[self.route_index:]

    @property
    def route_points(self) -> Tuple[Point, ...]:
        return tuple(
            _graph_node_point(self.graph, node_id)
            for node_id in self.remaining_route
            if node_id in self.graph
        )

    def _nearest_graph_vertex(self, point: Point) -> int:
        return min(
            self.graph.nodes,
            key=lambda node_id: (
                math.dist(_graph_node_point(self.graph, node_id), point),
                node_id,
            ),
        )

    def _route_endpoint_cost(self, start: int, endpoint: int) -> float:
        return float(nx.shortest_path_length(self.graph, start, endpoint, weight='weight'))

    def _matching_vertices(self, points: Sequence[Point]) -> Set[int]:
        return {
            node_id
            for node_id in self.graph.nodes
            if any(
                math.dist(_graph_node_point(self.graph, node_id), point)
                <= self.migration_radius
                for point in points
            )
        }

    def _nearest_vertex(self, point: Optional[Point]) -> Optional[int]:
        if point is None or self.graph.number_of_nodes() == 0:
            return None
        vertex_id = min(
            self.graph.nodes,
            key=lambda node_id: (
                math.dist(_graph_node_point(self.graph, node_id), point),
                node_id,
            ),
        )
        if math.dist(_graph_node_point(self.graph, vertex_id), point) <= self.migration_radius:
            return vertex_id
        return None

    def _append_unique_point(self, points: List[Point], point: Point):
        if not any(math.dist(existing, point) <= self.migration_radius for existing in points):
            points.append(point)


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
    corner_turn_threshold: float = math.pi / 4.0,
    min_vertex_spacing: float = 1.0,
    suppress_unknown_cycles: bool = False,
    unconfident_unknown_radius: float = 1.0,
    unconfident_unknown_ratio: float = 0.5,
    reconnection_clearance: Optional[float] = None,
    connection_cache: Optional[TopologyConnectionCache] = None,
    repair_connectivity: bool = True,
    connection_neighbor_limit: int = 10,
    map_revision=None,
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
    graph = skeleton_to_graph(
        skeleton,
        geometry,
        support_vertex_spacing,
        corner_turn_threshold,
    )
    astar_traversable = traversable
    if reconnection_clearance is not None and reconnection_clearance != clearance:
        _, astar_traversable = build_obstacle_traversability(
            grid,
            map_geometry,
            bounds,
            resolution=resolution,
            clearance=reconnection_clearance,
            boundary_margin=boundary_margin,
        )
    repair_stats = TopologyRepairStats(
        unresolved_components=nx.number_connected_components(graph)
        if graph.number_of_nodes()
        else 0
    )
    if repair_connectivity and graph.number_of_nodes() > 1:
        graph, repair_stats = repair_topology_connectivity(
            graph,
            skeleton,
            traversable,
            geometry,
            connection_cache or TopologyConnectionCache(),
            astar_traversable=astar_traversable,
            neighbor_limit=connection_neighbor_limit,
            map_revision=map_revision,
        )
    before_clustering = graph.number_of_nodes()
    if min_vertex_spacing > 0.0 and graph.number_of_nodes() > 1:
        graph = _cluster_close_vertices(graph, min_vertex_spacing)
    compression_stats = TopologyCompressionStats(before_clustering, graph.number_of_nodes())
    cycle_suppression_stats = UnknownCycleSuppressionStats()
    if suppress_unknown_cycles and graph.number_of_nodes() > 0:
        graph, cycle_suppression_stats = suppress_unconfident_cycles(
            graph,
            grid,
            map_geometry,
            radius=unconfident_unknown_radius,
            ratio_threshold=unconfident_unknown_ratio,
        )
    return GVDTopology(
        graph,
        geometry,
        skeleton,
        traversable,
        distance_to_mask(skeleton, geometry.resolution),
        repair_stats,
        compression_stats,
        cycle_suppression_stats,
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


def local_unknown_area(
    grid: np.ndarray,
    geometry: GridGeometry,
    point: Point,
    radius: float,
    *,
    include_outside_map: bool = False,
) -> float:
    """Measure unknown SLAM-map area inside a metric disk around a macro vertex."""
    if (
        radius < 0.0
        or geometry.resolution <= 0.0
        or grid.shape != (geometry.height, geometry.width)
    ):
        return 0.0
    cell = (
        int(math.floor((point[1] - geometry.origin_y) / geometry.resolution)),
        int(math.floor((point[0] - geometry.origin_x) / geometry.resolution)),
    )
    radius_cells = int(math.ceil(radius / geometry.resolution))
    rows = np.arange(cell[0] - radius_cells, cell[0] + radius_cells + 1)
    cols = np.arange(cell[1] - radius_cells, cell[1] + radius_cells + 1)
    inside = (
        np.hypot(rows[:, None] - cell[0], cols[None, :] - cell[1])
        * geometry.resolution
        <= radius
    )
    inside_rows = np.logical_and(rows >= 0, rows < geometry.height)
    inside_cols = np.logical_and(cols >= 0, cols < geometry.width)
    inside_map = np.logical_and(inside_rows[:, None], inside_cols[None, :])
    unknown_cells = 0
    if np.any(inside_rows) and np.any(inside_cols):
        row_indices = np.flatnonzero(inside_rows)
        col_indices = np.flatnonzero(inside_cols)
        local_grid = grid[np.ix_(rows[row_indices], cols[col_indices])]
        local_disk = inside[np.ix_(row_indices, col_indices)]
        unknown_cells = int(np.count_nonzero(np.logical_and(local_disk, local_grid == -1)))
    if include_outside_map:
        unknown_cells += int(np.count_nonzero(np.logical_and(inside, ~inside_map)))
    return unknown_cells * geometry.resolution * geometry.resolution


def local_unknown_ratio(
    grid: np.ndarray,
    geometry: GridGeometry,
    point: Point,
    radius: float,
    *,
    include_outside_map: bool = True,
) -> float:
    """Measure the fraction of a local disk that remains unknown."""
    if (
        radius < 0.0
        or geometry.resolution <= 0.0
        or grid.shape != (geometry.height, geometry.width)
    ):
        return 0.0
    cell = (
        int(math.floor((point[1] - geometry.origin_y) / geometry.resolution)),
        int(math.floor((point[0] - geometry.origin_x) / geometry.resolution)),
    )
    radius_cells = int(math.ceil(radius / geometry.resolution))
    rows = np.arange(cell[0] - radius_cells, cell[0] + radius_cells + 1)
    cols = np.arange(cell[1] - radius_cells, cell[1] + radius_cells + 1)
    disk = (
        np.hypot(rows[:, None] - cell[0], cols[None, :] - cell[1])
        * geometry.resolution
        <= radius
    )
    total = int(np.count_nonzero(disk))
    if total == 0:
        return 0.0
    inside_rows = np.logical_and(rows >= 0, rows < geometry.height)
    inside_cols = np.logical_and(cols >= 0, cols < geometry.width)
    inside_map = np.logical_and(inside_rows[:, None], inside_cols[None, :])
    unknown = 0
    if np.any(inside_rows) and np.any(inside_cols):
        row_indices = np.flatnonzero(inside_rows)
        col_indices = np.flatnonzero(inside_cols)
        local_grid = grid[np.ix_(rows[row_indices], cols[col_indices])]
        local_disk = disk[np.ix_(row_indices, col_indices)]
        unknown = int(np.count_nonzero(np.logical_and(local_disk, local_grid == -1)))
    if include_outside_map:
        unknown += int(np.count_nonzero(np.logical_and(disk, ~inside_map)))
    return float(unknown) / float(total)


def local_free_flood_mask(
    grid: np.ndarray,
    geometry: GridGeometry,
    center: Point,
    half_extent: float,
    bounds: Optional[WorldBounds] = None,
    excluded_points: Sequence[Point] = (),
    area_weight: float = 1.0,
    squareness_weight: float = 1.0,
) -> np.ndarray:
    """Grow a room-approximating rectangle from *center*.

    Candidate rectangles remain centered on the leaf vertex while expanding
    through free or unknown cells.  Occupied cells, map bounds, the coarse
    prior, and other GVD vertices limit expansion.  The selected Region
    balances normalized area and squareness using configurable weights.
    """
    mask = np.zeros((geometry.height, geometry.width), dtype=bool)
    if (
        half_extent < 0.0
        or geometry.resolution <= 0.0
        or grid.shape != mask.shape
    ):
        return mask

    allowed = grid <= 50
    if bounds is not None:
        allowed = np.logical_and(allowed, _cells_inside_world_bounds(geometry, bounds))
    excluded_vertices = np.zeros(mask.shape, dtype=bool)
    for point in excluded_points:
        cell = world_to_grid(point, geometry)
        if cell is not None:
            excluded_vertices[cell] = True

    # Locate the seed cell nearest to the active vertex without borrowing
    # another macro vertex's cell.
    seed = nearest_value_cell(
        np.logical_and(allowed, ~excluded_vertices),
        geometry,
        center,
    )
    if seed is None:
        return mask
    cr, cc = seed

    max_radius = int(math.ceil(half_extent / geometry.resolution))
    best = _best_centered_room_rect(
        allowed,
        excluded_vertices,
        cr,
        cc,
        max_radius,
        area_weight=max(0.0, float(area_weight)),
        squareness_weight=max(0.0, float(squareness_weight)),
    )
    top, bottom, left, right = best
    mask[top:bottom + 1, left:right + 1] = True
    return mask


def _best_centered_room_rect(
    allowed: np.ndarray,
    excluded_vertices: np.ndarray,
    cr: int,
    cc: int,
    max_radius: int,
    *,
    area_weight: float,
    squareness_weight: float,
) -> Tuple[int, int, int, int]:
    """Select a centered Region using area and square-shape utility."""
    blocked = np.logical_or(~allowed, excluded_vertices)
    prefix = np.pad(blocked.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    max_area = float((2 * max_radius + 1) ** 2)
    candidates = []
    for row_radius in range(max_radius + 1):
        top = cr - row_radius
        bottom = cr + row_radius
        if top < 0 or bottom >= allowed.shape[0]:
            break
        for col_radius in range(max_radius + 1):
            left = cc - col_radius
            right = cc + col_radius
            if left < 0 or right >= allowed.shape[1]:
                break
            blocked_count = (
                prefix[bottom + 1, right + 1]
                - prefix[top, right + 1]
                - prefix[bottom + 1, left]
                + prefix[top, left]
            )
            if blocked_count:
                continue
            height = float(bottom - top + 1)
            width = float(right - left + 1)
            area = height * width
            squareness = min(height, width) / max(height, width)
            utility = (
                area_weight * area / max_area
                + squareness_weight * squareness
            )
            candidates.append((utility, area, squareness, (top, bottom, left, right)))
    if not candidates:
        return cr, cr, cc, cc
    return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def _cells_inside_world_bounds(
    geometry: GridGeometry,
    bounds: WorldBounds,
) -> np.ndarray:
    """Return cells whose complete footprint stays inside the coarse prior."""
    rows = np.arange(geometry.height, dtype=float)
    cols = np.arange(geometry.width, dtype=float)
    min_x = geometry.origin_x + cols * geometry.resolution
    min_y = geometry.origin_y + rows * geometry.resolution
    max_x = min_x + geometry.resolution
    max_y = min_y + geometry.resolution
    epsilon = max(1e-9, geometry.resolution * 1e-9)
    inside_x = np.logical_and(
        min_x >= bounds.min_x - epsilon,
        max_x <= bounds.max_x + epsilon,
    )
    inside_y = np.logical_and(
        min_y >= bounds.min_y - epsilon,
        max_y <= bounds.max_y + epsilon,
    )
    return np.logical_and(inside_y[:, None], inside_x[None, :])


def cluster_touches_mask(cluster_cells: Sequence[GridCell], mask: np.ndarray) -> bool:
    """Return whether a frontier cluster belongs to a flooded local region."""
    return any(
        _cell_in_shape(cell, mask.shape) and mask[cell]
        for cell in cluster_cells
    )


def rectangle_mask_outline(
    mask: np.ndarray,
    geometry: GridGeometry,
) -> Tuple[Point, ...]:
    """Return a closed world-space outline for one rectangular Region mask."""
    cells = np.argwhere(mask)
    if cells.size == 0:
        return ()
    top = int(np.min(cells[:, 0]))
    bottom = int(np.max(cells[:, 0])) + 1
    left = int(np.min(cells[:, 1]))
    right = int(np.max(cells[:, 1])) + 1
    min_x = geometry.origin_x + left * geometry.resolution
    max_x = geometry.origin_x + right * geometry.resolution
    min_y = geometry.origin_y + top * geometry.resolution
    max_y = geometry.origin_y + bottom * geometry.resolution
    return (
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (min_x, min_y),
    )


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


def route_replan_due(last_replan_wall_time: float, interval: float, now: float) -> bool:
    """Return whether a dirty hierarchical route may be rebuilt without thrashing."""
    return now - last_replan_wall_time >= max(0.0, interval)


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


def bidirectional_astar_path(
    traversable: np.ndarray,
    start: GridCell,
    goal: GridCell,
    *,
    prevent_corner_cutting: bool = True,
) -> List[GridCell]:
    """Find a deterministic collision-free bridge with two-ended A* search."""
    if (
        not _cell_in_shape(start, traversable.shape)
        or not _cell_in_shape(goal, traversable.shape)
        or not traversable[start]
        or not traversable[goal]
    ):
        return []
    if start == goal:
        return [start]

    forward_queue = [(math.dist(start, goal), 0.0, start)]
    backward_queue = [(math.dist(goal, start), 0.0, goal)]
    forward_costs = {start: 0.0}
    backward_costs = {goal: 0.0}
    forward_previous = {}
    backward_previous = {}

    while forward_queue and backward_queue:
        meeting = _expand_bidirectional_frontier(
            forward_queue,
            forward_costs,
            forward_previous,
            backward_costs,
            goal,
            traversable,
            prevent_corner_cutting,
        )
        if meeting is not None:
            return _reconstruct_bidirectional_path(
                forward_previous,
                backward_previous,
                meeting,
            )
        meeting = _expand_bidirectional_frontier(
            backward_queue,
            backward_costs,
            backward_previous,
            forward_costs,
            start,
            traversable,
            prevent_corner_cutting,
        )
        if meeting is not None:
            return _reconstruct_bidirectional_path(
                forward_previous,
                backward_previous,
                meeting,
            )
    return []


def nearest_traversable_cell(
    traversable: np.ndarray,
    geometry: GridGeometry,
    point: Point,
) -> Optional[GridCell]:
    """Return the nearest passable cell to a world-space pose."""
    return nearest_value_cell(traversable, geometry, point)


def nearest_value_cell(
    mask: np.ndarray,
    geometry: GridGeometry,
    point: Point,
) -> Optional[GridCell]:
    """Return the nearest true cell in a raster mask."""
    cell = world_to_grid(point, geometry)
    if cell is not None and mask[cell]:
        return cell
    rows, cols = np.nonzero(mask)
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
    corner_turn_threshold: float = math.pi / 4.0,
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

    def ensure_node(cell: GridCell, kind: str = 'support') -> int:
        if cell not in nodes:
            nodes[cell] = len(nodes)
            x, y = grid_to_world(cell, geometry)
            graph.add_node(nodes[cell], x=x, y=y, kind=kind)
        elif _vertex_kind_priority(kind) < _vertex_kind_priority(
            graph.nodes[nodes[cell]]['kind']
        ):
            graph.nodes[nodes[cell]]['kind'] = kind
        return nodes[cell]

    for cell in sorted(key_cells):
        ensure_node(cell, _structural_vertex_kind(cell, adjacency))
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
            vertices = _split_chain_vertices(
                chain,
                geometry.resolution,
                spacing,
                corner_turn_threshold,
            )
            for source_vertex, target_vertex in zip(vertices, vertices[1:]):
                source_index, source_kind = source_vertex
                target_index, target_kind = target_vertex
                source = ensure_node(chain[source_index], source_kind)
                target = ensure_node(chain[target_index], target_kind)
                if source == target:
                    continue
                distance = _chain_length(chain[source_index:target_index + 1], geometry.resolution)
                graph.add_edge(
                    source,
                    target,
                    weight=distance,
                    information_weight=1.0 / max(distance, 1e-6),
                    virtual=False,
                    connection_mode='gvd',
                )
    return graph


def _cluster_close_vertices(graph: nx.Graph, min_spacing: float) -> nx.Graph:
    """Cluster directly connected nearby vertices without bridging unrelated chains."""
    if graph.number_of_nodes() < 2:
        return graph.copy()
    ordered = sorted(
        graph.nodes,
        key=lambda node_id: (
            _vertex_kind_priority(graph.nodes[node_id].get('kind', 'support')),
            -graph.degree(node_id),
            float(graph.nodes[node_id]['x']),
            float(graph.nodes[node_id]['y']),
            node_id,
        ),
    )
    assigned = {}
    clusters = []
    for representative in ordered:
        if representative in assigned:
            continue
        cluster = {representative}
        assigned[representative] = representative
        queue = [representative]
        origin = _graph_node_point(graph, representative)
        while queue:
            current = queue.pop(0)
            for neighbor in sorted(graph.neighbors(current)):
                if neighbor in assigned:
                    continue
                attributes = graph.edges[current, neighbor]
                if attributes.get('connection_mode', 'gvd') != 'gvd':
                    continue
                if math.dist(origin, _graph_node_point(graph, neighbor)) >= min_spacing:
                    continue
                assigned[neighbor] = representative
                cluster.add(neighbor)
                queue.append(neighbor)
        clusters.append((representative, cluster))

    clustered = nx.Graph()
    members = {}
    for representative, cluster in clusters:
        clustered.add_node(representative, **dict(graph.nodes[representative]))
        members[representative] = cluster
    for source, target, attributes in graph.edges(data=True):
        clustered_source = assigned[source]
        clustered_target = assigned[target]
        if clustered_source == clustered_target:
            continue
        candidate = dict(attributes)
        candidate['weight'] = (
            _cluster_internal_distance(
                graph,
                members[clustered_source],
                clustered_source,
                source,
            )
            + float(attributes.get('weight', 0.0))
            + _cluster_internal_distance(
                graph,
                members[clustered_target],
                target,
                clustered_target,
            )
        )
        candidate['information_weight'] = 1.0 / max(candidate['weight'], 1e-6)
        existing = clustered.get_edge_data(clustered_source, clustered_target)
        if existing is None or float(candidate.get('weight', math.inf)) < float(
            existing.get('weight', math.inf)
        ):
            clustered.add_edge(clustered_source, clustered_target, **candidate)
    return clustered


def _cluster_internal_distance(
    graph: nx.Graph,
    cluster,
    source: int,
    target: int,
) -> float:
    if source == target:
        return 0.0
    return float(
        nx.shortest_path_length(
            graph.subgraph(cluster),
            source,
            target,
            weight='weight',
        )
    )


def suppress_unconfident_cycles(
    graph: nx.Graph,
    grid: np.ndarray,
    map_geometry: GridGeometry,
    *,
    radius: float,
    ratio_threshold: float,
) -> Tuple[nx.Graph, UnknownCycleSuppressionStats]:
    """Replace unknown-heavy induced regions with MST edges before macro planning."""
    pruned = graph.copy()
    threshold = min(1.0, max(0.0, float(ratio_threshold)))
    unconfident = set()
    for node_id in pruned.nodes:
        ratio = local_unknown_ratio(
            grid,
            map_geometry,
            _graph_node_point(pruned, node_id),
            max(0.0, radius),
        )
        pruned.nodes[node_id]['unknown_ratio'] = ratio
        pruned.nodes[node_id]['unconfident'] = ratio >= threshold
        if ratio >= threshold:
            unconfident.add(node_id)
    removed_edges = 0
    induced = pruned.subgraph(unconfident)
    for component in nx.connected_components(induced):
        local = induced.subgraph(component)
        if local.number_of_edges() <= max(0, local.number_of_nodes() - 1):
            continue
        tree_edges = {
            cell_edge(source, target)
            for source, target in nx.minimum_spanning_edges(
                local,
                algorithm='kruskal',
                weight='weight',
                data=False,
            )
        }
        for source, target in list(local.edges):
            if cell_edge(source, target) in tree_edges:
                continue
            pruned.remove_edge(source, target)
            removed_edges += 1
    return pruned, UnknownCycleSuppressionStats(len(unconfident), removed_edges)


def repair_topology_connectivity(
    graph: nx.Graph,
    skeleton: np.ndarray,
    traversable: np.ndarray,
    geometry: GridGeometry,
    cache: TopologyConnectionCache,
    *,
    astar_traversable: Optional[np.ndarray] = None,
    neighbor_limit: int,
    map_revision=None,
) -> Tuple[nx.Graph, TopologyRepairStats]:
    """Repair reachable skeleton breaks with GVD-first, grid-A*-fallback bridges."""
    repaired = graph.copy()
    gvd_edges = 0
    astar_edges = 0
    fallback = traversable if astar_traversable is None else astar_traversable
    if fallback.shape != traversable.shape:
        raise ValueError('A* fallback mask shape does not match traversability mask.')
    revision = cache.revision_for(traversable, map_revision, fallback)
    while repaired.number_of_nodes() > 1:
        components = sorted(
            nx.connected_components(repaired),
            key=lambda component: (len(component), min(component)),
        )
        if len(components) <= 1:
            break
        added = False
        all_nodes = set(repaired.nodes)
        for component in components:
            outside = all_nodes - component
            bridge = _best_component_bridge(
                repaired,
                component,
                outside,
                skeleton,
                traversable,
                geometry,
                cache,
                astar_traversable=fallback,
                neighbor_limit=max(1, int(neighbor_limit)),
                revision=revision,
            )
            if bridge is None:
                continue
            source, target, connection = bridge
            repaired.add_edge(
                source,
                target,
                weight=connection.length,
                information_weight=1.0 / max(connection.length, 1e-6),
                virtual=False,
                connection_mode=connection.mode,
                path=tuple(grid_to_world(cell, geometry) for cell in connection.path),
            )
            if connection.mode == 'gvd':
                gvd_edges += 1
            else:
                astar_edges += 1
            added = True
            break
        if not added:
            break
    unresolved = (
        nx.number_connected_components(repaired)
        if repaired.number_of_nodes()
        else 0
    )
    return repaired, TopologyRepairStats(gvd_edges, astar_edges, unresolved)


def _best_component_bridge(
    graph: nx.Graph,
    component,
    outside,
    skeleton: np.ndarray,
    traversable: np.ndarray,
    geometry: GridGeometry,
    cache: TopologyConnectionCache,
    *,
    astar_traversable: np.ndarray,
    neighbor_limit: int,
    revision,
):
    candidates = []
    for source in sorted(component):
        source_point = _graph_node_point(graph, source)
        nearest = sorted(
            outside,
            key=lambda target: (
                math.dist(source_point, _graph_node_point(graph, target)),
                target,
            ),
        )[:neighbor_limit]
        for target in nearest:
            start = world_to_grid(source_point, geometry)
            goal = world_to_grid(_graph_node_point(graph, target), geometry)
            if start is None or goal is None:
                continue
            connection = cache.connection(
                traversable,
                skeleton,
                start,
                goal,
                resolution=geometry.resolution,
                revision=revision,
                astar_traversable=astar_traversable,
            )
            if connection is not None:
                candidates.append(
                    (
                        connection.mode != 'gvd',
                        connection.length,
                        source,
                        target,
                        connection,
                    )
                )
    if not candidates:
        return None
    _, _, source, target, connection = min(candidates)
    return source, target, connection


def _graph_node_point(graph: nx.Graph, node_id) -> Point:
    return float(graph.nodes[node_id]['x']), float(graph.nodes[node_id]['y'])


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


def astar_reconnection_segments(graph: nx.Graph) -> Tuple[Tuple[Point, Point], ...]:
    """Return world-space segments for switching-connection fallback A* bridges."""
    segments = []
    for _, _, attributes in graph.edges(data=True):
        if attributes.get('connection_mode') != 'astar':
            continue
        path = tuple(attributes.get('path', ()))
        segments.extend(zip(path, path[1:]))
    return tuple(segments)


def gvd_to_marker_array(
    topology: GVDTopology,
    bounds: WorldBounds,
    active_path: Sequence[Point],
    frame_id: str,
    stamp,
    hierarchical_tracker: Optional[HierarchicalGVDTracker] = None,
    local_cleanup_mask: Optional[np.ndarray] = None,
    local_cleanup_geometry: Optional[GridGeometry] = None,
    cleared_region_outlines: Sequence[Sequence[Point]] = (),
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
    if hierarchical_tracker is not None:
        explored = _marker(Marker, frame_id, stamp, 'gvd_explored_nodes', 3, Marker.SPHERE_LIST)
        explored.scale.x = explored.scale.y = explored.scale.z = 0.16
        explored.color.a = 0.9
        explored.color.g = 0.7
        for node_id in sorted(hierarchical_tracker.explored_vertices):
            if node_id in hierarchical_tracker.graph:
                explored.points.append(
                    _marker_point(MarkerPoint, _graph_node_point(hierarchical_tracker.graph, node_id), 0.10)
                )
        markers.markers.append(explored)

        cleared = _marker(Marker, frame_id, stamp, 'gvd_cleared_nodes', 4, Marker.SPHERE_LIST)
        cleared.scale.x = cleared.scale.y = cleared.scale.z = 0.20
        cleared.color.a = 0.95
        cleared.color.g = 1.0
        cleared.color.b = 0.35
        for node_id in sorted(hierarchical_tracker.cleared_vertices):
            if node_id in hierarchical_tracker.graph:
                cleared.points.append(
                    _marker_point(MarkerPoint, _graph_node_point(hierarchical_tracker.graph, node_id), 0.13)
                )
        markers.markers.append(cleared)

        if hierarchical_tracker.active_vertex in hierarchical_tracker.graph:
            active = _marker(Marker, frame_id, stamp, 'gvd_active_macro_node', 5, Marker.SPHERE)
            active.pose.position.x, active.pose.position.y = _graph_node_point(
                hierarchical_tracker.graph,
                hierarchical_tracker.active_vertex,
            )
            active.pose.position.z = 0.17
            active.scale.x = active.scale.y = active.scale.z = 0.28
            active.color.a = 1.0
            active.color.r = 1.0
            active.color.g = 0.2
            markers.markers.append(active)
    if (
        local_cleanup_mask is not None
        and local_cleanup_geometry is not None
        and local_cleanup_mask.shape
        == (local_cleanup_geometry.height, local_cleanup_geometry.width)
    ):
        flood = _marker(Marker, frame_id, stamp, 'gvd_local_cleanup_flood', 6, Marker.CUBE_LIST)
        flood.scale.x = flood.scale.y = local_cleanup_geometry.resolution
        flood.scale.z = 0.025
        flood.color.a = 0.22
        flood.color.g = 0.9
        flood.color.b = 1.0
        for row, column in np.argwhere(local_cleanup_mask):
            flood.points.append(
                _marker_point(
                    MarkerPoint,
                    grid_to_world((int(row), int(column)), local_cleanup_geometry),
                    0.025,
                )
            )
        markers.markers.append(flood)
    cleared_regions = _marker(
        Marker,
        frame_id,
        stamp,
        'gvd_cleared_region_outlines',
        7,
        Marker.LINE_LIST,
    )
    cleared_regions.scale.x = 0.045
    cleared_regions.color.a = 0.75
    cleared_regions.color.g = 0.95
    cleared_regions.color.b = 1.0
    for outline in cleared_region_outlines:
        for source, target in zip(outline, outline[1:]):
            cleared_regions.points.append(_marker_point(MarkerPoint, source, 0.035))
            cleared_regions.points.append(_marker_point(MarkerPoint, target, 0.035))
    markers.markers.append(cleared_regions)
    if hierarchical_tracker is not None:
        tsp_route = _marker(
            Marker,
            frame_id,
            stamp,
            'gvd_hierarchical_tsp_route',
            8,
            Marker.LINE_STRIP,
        )
        tsp_route.scale.x = 0.065
        tsp_route.color.a = 0.9
        tsp_route.color.r = 1.0
        tsp_route.color.g = 0.75
        for point in hierarchical_tracker.route_points:
            tsp_route.points.append(_marker_point(MarkerPoint, point, 0.11))
        markers.markers.append(tsp_route)
    astar_bridges = _marker(
        Marker,
        frame_id,
        stamp,
        'gvd_astar_reconnections',
        9,
        Marker.LINE_LIST,
    )
    astar_bridges.scale.x = 0.085
    astar_bridges.color.a = 0.95
    astar_bridges.color.r = 1.0
    astar_bridges.color.g = 0.45
    for source, target in astar_reconnection_segments(topology.graph):
        astar_bridges.points.append(_marker_point(MarkerPoint, source, 0.14))
        astar_bridges.points.append(_marker_point(MarkerPoint, target, 0.14))
    markers.markers.append(astar_bridges)
    unconfident = _marker(
        Marker,
        frame_id,
        stamp,
        'gvd_unconfident_nodes',
        10,
        Marker.SPHERE_LIST,
    )
    unconfident.scale.x = unconfident.scale.y = unconfident.scale.z = 0.14
    unconfident.color.a = 0.9
    unconfident.color.r = 0.75
    unconfident.color.b = 1.0
    for node_id, attributes in topology.graph.nodes(data=True):
        if attributes.get('unconfident', False):
            unconfident.points.append(
                _marker_point(MarkerPoint, _graph_node_point(topology.graph, node_id), 0.16)
            )
    markers.markers.append(unconfident)
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


def _cell_in_shape(cell: GridCell, shape) -> bool:
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _collision_free_neighbors(
    cell: GridCell,
    traversable: np.ndarray,
    prevent_corner_cutting: bool,
):
    for neighbor, distance in _neighbors(cell, traversable.shape):
        if not traversable[neighbor]:
            continue
        if (
            prevent_corner_cutting
            and cell[0] != neighbor[0]
            and cell[1] != neighbor[1]
            and (
                not traversable[cell[0], neighbor[1]]
                or not traversable[neighbor[0], cell[1]]
            )
        ):
            continue
        yield neighbor, distance


def _path_cells_traversable(
    path: Sequence[GridCell],
    traversable: np.ndarray,
    *,
    prevent_corner_cutting: bool,
) -> bool:
    for source, target in zip(path, path[1:]):
        if not traversable[source] or not traversable[target]:
            return False
        if (
            prevent_corner_cutting
            and source[0] != target[0]
            and source[1] != target[1]
            and (
                not traversable[source[0], target[1]]
                or not traversable[target[0], source[1]]
            )
        ):
            return False
    return not path or bool(traversable[path[-1]])


def _expand_bidirectional_frontier(
    queue,
    own_costs,
    own_previous,
    other_costs,
    target: GridCell,
    traversable: np.ndarray,
    prevent_corner_cutting: bool,
) -> Optional[GridCell]:
    while queue:
        _, cost, cell = heapq.heappop(queue)
        if cost <= own_costs.get(cell, math.inf) + 1e-9:
            break
    else:
        return None
    if cell in other_costs:
        return cell
    for neighbor, distance in _collision_free_neighbors(
        cell,
        traversable,
        prevent_corner_cutting,
    ):
        next_cost = cost + distance
        if next_cost + 1e-9 >= own_costs.get(neighbor, math.inf):
            continue
        own_costs[neighbor] = next_cost
        own_previous[neighbor] = cell
        heapq.heappush(
            queue,
            (next_cost + math.dist(neighbor, target), next_cost, neighbor),
        )
        if neighbor in other_costs:
            return neighbor
    return None


def _reconstruct_bidirectional_path(
    forward_previous,
    backward_previous,
    meeting: GridCell,
) -> List[GridCell]:
    path = _reconstruct_path(forward_previous, meeting)
    cell = meeting
    while cell in backward_previous:
        cell = backward_previous[cell]
        path.append(cell)
    return path


def _reconstruct_path(previous, cell: GridCell) -> List[GridCell]:
    path = [cell]
    while cell in previous:
        cell = previous[cell]
        path.append(cell)
    return list(reversed(path))


def _split_chain_vertices(
    chain: Sequence[GridCell],
    resolution: float,
    spacing: float,
    corner_turn_threshold: float,
):
    """Split one skeleton chain at structural ends, supports, and visible corners."""
    vertices = [(0, 'support')]
    accumulated = 0.0
    accumulated_turn = 0.0
    previous_heading = None
    for index in range(1, len(chain)):
        step = math.dist(chain[index - 1], chain[index]) * resolution
        heading = math.atan2(
            chain[index][0] - chain[index - 1][0],
            chain[index][1] - chain[index - 1][1],
        )
        if previous_heading is not None:
            accumulated_turn += abs(normalize_angle(heading - previous_heading))
        split_kind = None
        if (
            corner_turn_threshold > 0.0
            and accumulated_turn + 1e-9 >= corner_turn_threshold
            and index - 1 > vertices[-1][0]
        ):
            split_kind = 'corner'
        elif accumulated + step > spacing and index - 1 > vertices[-1][0]:
            split_kind = 'support'
        if split_kind is not None:
            vertices.append((index - 1, split_kind))
            accumulated = 0.0
            accumulated_turn = 0.0
        accumulated += step
        previous_heading = heading
    if vertices[-1][0] != len(chain) - 1:
        vertices.append((len(chain) - 1, 'support'))
    return vertices


def _split_chain_indices(chain: Sequence[GridCell], resolution: float, spacing: float):
    """Compatibility wrapper returning support-only split indices."""
    return [
        index
        for index, _ in _split_chain_vertices(chain, resolution, spacing, math.inf)
    ]


def _chain_length(chain: Sequence[GridCell], resolution: float) -> float:
    return sum(math.dist(source, target) * resolution for source, target in zip(chain, chain[1:]))


def _structural_vertex_kind(cell: GridCell, adjacency) -> str:
    degree = len(adjacency[cell])
    if degree <= 1:
        return 'endpoint'
    if degree >= 3:
        return 'branch'
    return 'support'


def _vertex_kind_priority(kind: str) -> int:
    return {
        'branch': 0,
        'endpoint': 1,
        'corner': 2,
        'support': 3,
    }.get(kind, 4)


def _deduplicate_adjacent(values: Sequence[int]) -> List[int]:
    deduplicated = []
    for value in values:
        if not deduplicated or deduplicated[-1] != value:
            deduplicated.append(value)
    return deduplicated


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
