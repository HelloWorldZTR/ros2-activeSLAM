"""Obstacle-only GVD helpers for fast online bootstrap exploration."""

import heapq
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
class SparseGVDEdge:
    """One compressed GVD chain between structural skeleton vertices."""

    source: int
    target: int
    polyline: Tuple[Point, ...]
    length: float
    blocked_run: float = 0.0
    stale: bool = False


@dataclass(frozen=True)
class GVDGuidePlanStep:
    """One queued sparse-guide waypoint or optional local detour."""

    kind: str
    goal_xy: Point
    vertex_id: Optional[int] = None
    source_vertex: Optional[int] = None
    target_vertex: Optional[int] = None
    frontier: Any = None
    edge: Optional[Tuple[int, int]] = None
    expected_cost: float = 0.0
    optional: bool = False


@dataclass(frozen=True)
class GVDGuideOnlineDetourUpdate:
    """Summary of one online frontier-detour refresh around the reached vertex."""

    source_vertex: Optional[int]
    target_vertex: Optional[int]
    assigned_frontiers: int = 0
    target_frontiers: int = 0
    removed_detours: int = 0
    inserted: bool = False
    selected_goal: Optional[Point] = None
    expected_extra_cost: float = 0.0


@dataclass(frozen=True)
class _GVDGuideRouteVertex:
    vertex_id: int
    loop_revisit: bool = False


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

    def __init__(
        self,
        migration_radius: float,
    ):
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
        self.expansion_targets: Tuple[int, ...] = ()
        self.cleanup_targets: Tuple[int, ...] = ()
        self.route_phase = 'empty'
        self.transit_route: Tuple[int, ...] = ()
        self.route_index = 0
        self.route_length = 0.0
        self.continuation_successor: Optional[int] = None
        self._continuation_direction: Optional[Point] = None
        self._shortest_path_cache = {}

    def update_graph(self, graph: nx.Graph):
        """Install a rebuilt graph while migrating states by nearby world positions."""
        self._continuation_direction = self._next_route_direction()
        self.graph = graph.copy()
        self.graph_version += 1
        self.explored_vertices = self._matching_vertices(self.explored_points)
        self.cleared_vertices = self._matching_vertices(self.cleared_points)
        self.previous_vertex = self._nearest_vertex(self.previous_point)
        self.active_vertex = self._nearest_vertex(self.active_point)
        self.route_targets = ()
        self.expansion_targets = ()
        self.cleanup_targets = ()
        self.route_phase = 'empty'
        self.transit_route = ()
        self.route_index = 0
        self.continuation_successor = None
        self.route_dirty = True
        self._shortest_path_cache = {}

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
        """Create an expansion-first open TSP walk over Region-bearing vertices."""
        self.route_dirty = False
        self.route_index = 0
        if self.graph.number_of_nodes() == 0:
            self.route_targets = ()
            self.expansion_targets = ()
            self.cleanup_targets = ()
            self.route_phase = 'empty'
            self.transit_route = ()
            self.route_length = 0.0
            self.continuation_successor = None
            return self.transit_route
        start = self._nearest_graph_vertex(robot_xy)
        if (
            self.active_vertex is not None
            and self.active_vertex in self.graph
            and self.should_clear_local(self.active_vertex)
        ):
            self.route_targets = ()
            self.expansion_targets = ()
            self.cleanup_targets = ()
            self.route_phase = 'cleanup_ready'
            self.transit_route = ()
            self.route_length = 0.0
            self.continuation_successor = None
            self._continuation_direction = None
            return self.transit_route
        forced_successor = self._select_continuation_successor(failed)
        if forced_successor is not None and self.active_vertex is not None:
            self._mark_vertex_reached(self.active_vertex)
            start = forced_successor
        self.expansion_targets = self._expansion_target_vertices()
        self.cleanup_targets = self._cleanup_target_vertices(self.expansion_targets)
        self.route_phase = 'expansion' if self.expansion_targets else 'cleanup'
        phase_targets = (
            self.expansion_targets
            if self.route_phase == 'expansion'
            else self.cleanup_targets
        )
        targets = tuple(
            node_id
            for node_id in phase_targets
            if failed is None or not failed(_graph_node_point(self.graph, node_id))
        )
        self.route_targets = targets
        if not targets:
            self.transit_route = ()
            self.route_length = 0.0
            self.continuation_successor = forced_successor
            return self.transit_route
        self.transit_route = self._fresh_open_tsp_walk(start, targets)
        self.route_length = self._walk_length(self.transit_route)
        self.continuation_successor = forced_successor
        return self.transit_route

    def _fresh_open_tsp_walk(self, start: int, targets: Sequence[int]) -> Tuple[int, ...]:
        """Return one NetworkX open-TSP walk joined to *start*."""
        if not targets:
            return ()
        if len(targets) == 1:
            tsp_route = list(targets)
        elif len(targets) == 2:
            tsp_route = list(self._shortest_path(targets[0], targets[1]))
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
        prefix = list(self._shortest_path(start, tsp_route[0]))
        return tuple(_deduplicate_adjacent(prefix[:-1] + tsp_route))

    def _shortest_path(self, source: int, target: int) -> Tuple[int, ...]:
        """Return one cached weighted shortest path inside the current graph version."""
        key = source, target
        if key not in self._shortest_path_cache:
            path = tuple(nx.shortest_path(self.graph, source, target, weight='weight'))
            self._shortest_path_cache[key] = path
            self._shortest_path_cache[target, source] = tuple(reversed(path))
        return self._shortest_path_cache[key]

    def _walk_length(self, route: Sequence[int]) -> float:
        return sum(
            float(self.graph.edges[source, target].get('weight', 1.0))
            for source, target in zip(route, route[1:])
        )

    def _next_route_direction(self) -> Optional[Point]:
        """Capture the old active-to-next-step direction before replacing the graph."""
        if self.active_point is None:
            return None
        for node_id in self.remaining_route:
            if node_id not in self.graph:
                continue
            point = _graph_node_point(self.graph, node_id)
            direction = point[0] - self.active_point[0], point[1] - self.active_point[1]
            if math.hypot(*direction) > 1e-9:
                return direction
        return None

    def _select_continuation_successor(
        self,
        failed: Optional[Callable[[Point], bool]],
    ) -> Optional[int]:
        """Pick the rebuilt active neighbor closest to the pre-rebuild direction."""
        direction = self._continuation_direction
        self._continuation_direction = None
        active = self.active_vertex
        if direction is None or active is None or active not in self.graph:
            return None
        direction_length = math.hypot(*direction)
        candidates = []
        active_point = _graph_node_point(self.graph, active)
        for neighbor in self.graph.neighbors(active):
            point = _graph_node_point(self.graph, neighbor)
            if failed is not None and failed(point):
                continue
            offset = point[0] - active_point[0], point[1] - active_point[1]
            offset_length = math.hypot(*offset)
            if offset_length <= 1e-9:
                continue
            cosine = (
                direction[0] * offset[0] + direction[1] * offset[1]
            ) / (direction_length * offset_length)
            candidates.append((cosine, -neighbor, neighbor))
        return None if not candidates else max(candidates)[2]

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

    def mark_reached_point(self, point: Point) -> Optional[int]:
        """Mark the latest graph vertex near a completed world-space Nav2 goal."""
        vertex_id = self._nearest_vertex(point)
        if vertex_id is None:
            return None
        self.mark_reached(vertex_id)
        return vertex_id

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
        """Return whether one uncleared Region-bearing vertex is locally completeable."""
        if (
            vertex_id not in self.graph
            or vertex_id in self.cleared_vertices
            or self._macro_vertex_kind(vertex_id) not in ('endpoint', 'branch')
        ):
            return False
        return self.graph.degree(vertex_id) <= 1 or self._is_degenerate_leaf(vertex_id)

    def _expansion_target_vertices(self) -> Tuple[int, ...]:
        """Return unexplored endpoints, with branches as component fallbacks."""
        targets = set()
        unexplored = set(self.graph.nodes) - self.explored_vertices - self.cleared_vertices
        for component in nx.connected_components(self.graph):
            component = set(component)
            component_unexplored = component & unexplored
            endpoints = {
                node_id
                for node_id in component_unexplored
                if self._macro_vertex_kind(node_id) == 'endpoint'
            }
            targets.update(endpoints)
            # A branch normally becomes explored while traveling to an endpoint. Keep
            # one as a target only when this component has no endpoint representative.
            if not endpoints:
                targets.update(
                    node_id
                    for node_id in component_unexplored
                    if self._macro_vertex_kind(node_id) == 'branch'
                )
        return tuple(sorted(targets))

    def _cleanup_target_vertices(
        self,
        expansion_targets: Sequence[int],
    ) -> Tuple[int, ...]:
        """Return uncleared leaves after expansion routing has been exhausted."""
        return tuple(
            node_id
            for node_id in sorted(self.graph.nodes)
            if self.should_clear_local(node_id)
            and node_id not in expansion_targets
        )

    def _is_degenerate_leaf(self, vertex_id: int) -> bool:
        """Return whether every branch beyond one branch vertex lacks expansion work."""
        if self._macro_vertex_kind(vertex_id) != 'branch':
            return False
        expansion_targets = set(self._expansion_target_vertices()) - {vertex_id}
        for neighbor in self.graph.neighbors(vertex_id):
            reachable = nx.node_connected_component(
                nx.restricted_view(self.graph, (vertex_id,), ()),
                neighbor,
            )
            if expansion_targets.intersection(reachable):
                return False
        return True

    def _macro_vertex_kind(self, vertex_id: int) -> str:
        """Read compressed-node kind with a degree-based fallback for legacy graphs."""
        kind = self.graph.nodes[vertex_id].get('kind')
        if kind is not None:
            return kind
        degree = self.graph.degree(vertex_id)
        if degree <= 1:
            return 'endpoint'
        if degree > 2:
            return 'branch'
        return 'support'

    def mark_local_cleared(self, vertex_id: Optional[int]):
        if vertex_id is None or vertex_id not in self.graph:
            return
        self.graph.nodes[vertex_id]['cleared'] = True
        self.cleared_vertices.add(vertex_id)
        self._append_unique_point(self.cleared_points, _graph_node_point(self.graph, vertex_id))
        self.route_dirty = True

    @property
    def has_pending_macro_targets(self) -> bool:
        return bool(self.expansion_targets or self.cleanup_targets)

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


class GVDGuidePlanner:
    """Sparse GVD route epoch with optional loops and local frontier detours."""

    def __init__(
        self,
        graph: nx.Graph,
        rebuild_grid: np.ndarray,
        route_vertices: Sequence[int],
        plan_queue: Sequence[GVDGuidePlanStep],
        *,
        loop_edges: Sequence[Tuple[int, int]] = (),
        frontier_assignments: Optional[Dict[int, List[Any]]] = None,
        explored_vertices: Sequence[int] = (),
        expansion_area: float = 0.0,
        route_shortcuts: int = 0,
        start_hint: Optional[Point] = None,
    ):
        self.graph = graph.copy()
        self.rebuild_grid = np.asarray(rebuild_grid, dtype=np.int8).copy()
        self.route_vertices = tuple(route_vertices)
        self.plan_queue = tuple(plan_queue)
        self.loop_edges = tuple(loop_edges)
        self.explored_vertices = tuple(sorted(explored_vertices))
        self.route_shortcuts = int(route_shortcuts)
        self.start_hint = start_hint
        self.frontier_assignments = (
            {key: list(value) for key, value in frontier_assignments.items()}
            if frontier_assignments is not None
            else {}
        )
        self.expansion_area = float(expansion_area)
        self.active_index = 0
        self.online_detour_updates = 0
        self.online_detour_insertions = 0
        self.online_detour_last_update: Optional[GVDGuideOnlineDetourUpdate] = None

    @classmethod
    def build(
        cls,
        topology: GVDTopology,
        rebuild_grid: np.ndarray,
        robot_xy: Point,
        frontiers: Iterable[Any],
        *,
        loop_path_cost_weight: float,
        frontier_detour_weight: float,
        frontier_detour_max_extra_distance: float,
        frontier_detour_min_gain: float,
        explored_points: Sequence[Point] = (),
        migration_radius: float = 0.0,
        max_waypoint_distance: float = math.inf,
        start_hint: Optional[Point] = None,
    ) -> 'GVDGuidePlanner':
        """Create one sparse-guide plan epoch from the current GVD and frontiers."""
        sparse = build_sparse_gvd_graph_from_topology(topology.graph, topology.geometry)
        if sparse.number_of_nodes() > 0:
            sparse = robot_component_graph(sparse, robot_xy)
        explored_vertices = guide_matching_vertices(
            sparse,
            explored_points,
            migration_radius,
        )
        for node_id in explored_vertices:
            if node_id in sparse:
                sparse.nodes[node_id]['explored'] = True
        assignments = assign_frontiers_to_sparse_vertices(sparse, frontiers)
        start = (
            nearest_graph_vertex(sparse, robot_xy)
            if start_hint is None
            else nearest_graph_vertex_by_path_estimate(
                sparse,
                start_hint,
                topology.geometry,
                topology.traversable,
            )
        )
        if start is None:
            return cls(
                sparse,
                rebuild_grid,
                (),
                (),
                frontier_assignments=assignments,
                explored_vertices=explored_vertices,
                start_hint=start_hint,
            )
        targets = sorted(
            (set(sparse_leaf_vertices(sparse)) - set(explored_vertices))
            | {
                node_id
                for node_id, values in assignments.items()
                if values and node_id not in explored_vertices
            }
        )
        if not targets:
            targets = [start]
        route = sparse_open_tsp_route(sparse, start, targets)
        route, route_shortcuts = shortcut_gvd_guide_route(
            sparse,
            route,
            topology.geometry,
            topology.traversable,
        )
        route_steps, loop_edges = insert_gvd_guide_loop_revisits(
            sparse,
            route,
            loop_path_cost_weight,
        )
        queue = gvd_guide_plan_steps(
            sparse,
            route_steps,
            assignments,
            frontier_detour_weight=frontier_detour_weight,
            frontier_detour_max_extra_distance=frontier_detour_max_extra_distance,
            frontier_detour_min_gain=frontier_detour_min_gain,
            max_waypoint_distance=max_waypoint_distance,
        )
        return cls(
            sparse,
            rebuild_grid,
            route,
            queue,
            loop_edges=loop_edges,
            frontier_assignments=assignments,
            explored_vertices=explored_vertices,
            route_shortcuts=route_shortcuts,
            start_hint=start_hint,
        )

    @property
    def active_step(self) -> Optional[GVDGuidePlanStep]:
        if self.active_index >= len(self.plan_queue):
            return None
        return self.plan_queue[self.active_index]

    @property
    def remaining_steps(self) -> Tuple[GVDGuidePlanStep, ...]:
        return self.plan_queue[self.active_index:]

    @property
    def is_complete(self) -> bool:
        return self.active_step is None

    def advance_active_step(self) -> Optional[GVDGuidePlanStep]:
        step = self.active_step
        if step is not None:
            self.active_index += 1
        return step

    def skip_active_step(self) -> Optional[GVDGuidePlanStep]:
        return self.advance_active_step()

    def refresh_online_frontier_detour(
        self,
        current_vertex: Optional[int],
        frontiers: Iterable[Any],
        *,
        frontier_detour_weight: float,
        frontier_detour_max_extra_distance: float,
        frontier_detour_min_gain: float,
    ) -> GVDGuideOnlineDetourUpdate:
        """Recompute the optional frontier detour for the next local guide segment."""
        assignments = assign_frontiers_to_sparse_vertices(self.graph, frontiers)
        self.frontier_assignments = assignments
        assigned_count = sum(len(values) for values in assignments.values())
        if current_vertex is None or current_vertex not in self.graph:
            return self._record_online_detour_update(
                GVDGuideOnlineDetourUpdate(
                    current_vertex,
                    None,
                    assigned_frontiers=assigned_count,
                )
            )

        prefix = list(self.plan_queue[: self.active_index])
        remaining = list(self.plan_queue[self.active_index :])
        removed = 0
        while remaining and remaining[0].kind == 'frontier_detour':
            remaining.pop(0)
            removed += 1

        next_step = next(
            (
                step
                for step in remaining
                if (
                    step.source_vertex == current_vertex
                    and step.target_vertex is not None
                    and step.kind in ('gvd_vertex', 'loop_revisit')
                )
            ),
            None,
        )
        if next_step is None:
            self.plan_queue = tuple(prefix + remaining)
            self.active_index = len(prefix)
            return self._record_online_detour_update(
                GVDGuideOnlineDetourUpdate(
                    current_vertex,
                    None,
                    assigned_frontiers=assigned_count,
                    removed_detours=removed,
                )
            )

        target = next_step.target_vertex
        target_frontiers = len(assignments.get(target, ()))
        detour = None
        if next_step.kind == 'gvd_vertex':
            used_frontiers = {
                id(step.frontier)
                for step in remaining
                if step.kind == 'frontier_detour' and step.frontier is not None
            }
            detour = _best_frontier_detour(
                self.graph,
                current_vertex,
                target,
                assignments.get(target, ()),
                used_frontiers,
                frontier_detour_weight=frontier_detour_weight,
                frontier_detour_max_extra_distance=frontier_detour_max_extra_distance,
                frontier_detour_min_gain=frontier_detour_min_gain,
            )
            if detour is not None:
                remaining.insert(0, detour)

        self.plan_queue = tuple(prefix + remaining)
        self.active_index = len(prefix)
        return self._record_online_detour_update(
            GVDGuideOnlineDetourUpdate(
                current_vertex,
                target,
                assigned_frontiers=assigned_count,
                target_frontiers=target_frontiers,
                removed_detours=removed,
                inserted=detour is not None,
                selected_goal=None if detour is None else detour.goal_xy,
                expected_extra_cost=0.0 if detour is None else detour.expected_cost,
            )
        )

    def _record_online_detour_update(
        self,
        update: GVDGuideOnlineDetourUpdate,
    ) -> GVDGuideOnlineDetourUpdate:
        self.online_detour_updates += 1
        if update.inserted:
            self.online_detour_insertions += 1
        self.online_detour_last_update = update
        return update

    def active_sparse_edge_polyline(self) -> Tuple[Point, ...]:
        step = self.active_step
        if (
            step is None
            or step.edge is None
            or not self.graph.has_edge(*step.edge)
        ):
            return ()
        return tuple(self.graph.edges[step.edge].get('polyline', ()))


def build_sparse_gvd_graph(skeleton: np.ndarray, geometry: GridGeometry) -> nx.Graph:
    """Compress a skeleton to branch/leaf nodes with edge polylines."""
    adjacency = skeleton_adjacency(skeleton)
    graph = nx.Graph()
    if not adjacency:
        return graph
    key_cells = set()
    for component in _skeleton_components(adjacency):
        structural = {cell for cell in component if len(adjacency[cell]) != 2}
        if structural:
            key_cells.update(structural)
            continue
        first = min(component)
        second = max(
            component,
            key=lambda cell: (math.dist(first, cell), cell),
        )
        key_cells.update((first, second))
    nodes: Dict[GridCell, int] = {}

    def ensure_node(cell: GridCell) -> int:
        if cell not in nodes:
            node_id = len(nodes)
            nodes[cell] = node_id
            x, y = grid_to_world(cell, geometry)
            graph.add_node(
                node_id,
                x=x,
                y=y,
                cell=cell,
                kind=_sparse_vertex_kind(cell, adjacency, key_cells),
            )
        return nodes[cell]

    for cell in sorted(key_cells):
        ensure_node(cell)
    visited_edges = set()
    for start in sorted(key_cells):
        for neighbor in adjacency[start]:
            edge = cell_edge(start, neighbor)
            if edge in visited_edges:
                continue
            chain = [start]
            previous, current = start, neighbor
            visited_edges.add(edge)
            while current not in key_cells:
                chain.append(current)
                choices = [cell for cell in adjacency[current] if cell != previous]
                if not choices:
                    break
                next_cell = choices[0]
                visited_edges.add(cell_edge(current, next_cell))
                previous, current = current, next_cell
            chain.append(current)
            source = ensure_node(chain[0])
            target = ensure_node(chain[-1])
            if source == target:
                continue
            length = _chain_length(chain, geometry.resolution)
            polyline = tuple(grid_to_world(cell, geometry) for cell in chain)
            existing = graph.get_edge_data(source, target)
            if existing is not None and float(existing.get('weight', math.inf)) <= length:
                continue
            graph.add_edge(
                source,
                target,
                weight=length,
                information_weight=1.0 / max(length, 1e-6),
                polyline=polyline,
                cells=tuple(chain),
                connection_mode='gvd_guide',
                blocked_run=0.0,
                stale=False,
            )
    return graph


def build_sparse_gvd_graph_from_topology(
    topology_graph: nx.Graph,
    geometry: GridGeometry,
) -> nx.Graph:
    """Compress the repaired topology graph to branch/leaf guide vertices.

    The input graph is the same repaired, clustered, and cycle-pruned graph used by
    hierarchical GVD.  This final pass only removes degree-2 transit vertices while
    preserving their edge polylines for blockage checks and RViz rendering.
    """
    sparse = nx.Graph()
    if topology_graph.number_of_nodes() == 0:
        return sparse
    key_nodes = set()
    for component in nx.connected_components(topology_graph):
        component = set(component)
        structural = {
            node_id
            for node_id in component
            if topology_graph.degree(node_id) != 2
        }
        if structural:
            key_nodes.update(structural)
            continue
        first = min(component)
        second = max(
            component,
            key=lambda node_id: (
                math.dist(
                    _graph_node_point(topology_graph, first),
                    _graph_node_point(topology_graph, node_id),
                ),
                node_id,
            ),
        )
        key_nodes.update((first, second))

    for node_id in sorted(key_nodes):
        attributes = dict(topology_graph.nodes[node_id])
        attributes['kind'] = _sparse_topology_vertex_kind(
            topology_graph,
            node_id,
            key_nodes,
        )
        sparse.add_node(
            node_id,
            **attributes,
        )

    visited_edges = set()
    for start in sorted(key_nodes):
        for neighbor in sorted(topology_graph.neighbors(start)):
            edge = _node_edge_key(start, neighbor)
            if edge in visited_edges:
                continue
            chain = [start]
            previous, current = start, neighbor
            visited_edges.add(edge)
            while current not in key_nodes:
                chain.append(current)
                choices = [
                    node_id
                    for node_id in sorted(topology_graph.neighbors(current))
                    if node_id != previous
                ]
                if not choices:
                    break
                next_node = choices[0]
                visited_edges.add(_node_edge_key(current, next_node))
                previous, current = current, next_node
            chain.append(current)
            source, target = chain[0], chain[-1]
            if source == target or source not in sparse or target not in sparse:
                continue
            length = _topology_chain_length(topology_graph, chain)
            polyline = _topology_chain_polyline(topology_graph, chain, geometry)
            connection_modes = tuple(
                topology_graph.edges[u, v].get('connection_mode', 'gvd')
                for u, v in zip(chain, chain[1:])
            )
            connection_mode = (
                connection_modes[0]
                if len(set(connection_modes)) == 1
                else 'mixed'
            )
            candidate = {
                'weight': length,
                'information_weight': 1.0 / max(length, 1e-6),
                'polyline': polyline,
                'source_graph_nodes': tuple(chain),
                'connection_mode': connection_mode,
                'connection_modes': connection_modes,
                'blocked_run': 0.0,
                'stale': False,
            }
            existing = sparse.get_edge_data(source, target)
            if existing is None or length < float(existing.get('weight', math.inf)):
                sparse.add_edge(source, target, **candidate)
    return sparse


def sparse_leaf_vertices(graph: nx.Graph) -> Tuple[int, ...]:
    """Return sparse route targets, with cycle anchors as fallback leaves."""
    leaves = tuple(sorted(node_id for node_id in graph.nodes if graph.degree(node_id) <= 1))
    if leaves:
        return leaves
    return tuple(
        sorted(
            node_id
            for node_id, attributes in graph.nodes(data=True)
            if attributes.get('kind') == 'cycle_anchor'
        )
    )


def nearest_graph_vertex(graph: nx.Graph, point: Point) -> Optional[int]:
    if graph.number_of_nodes() == 0:
        return None
    return min(
        graph.nodes,
        key=lambda node_id: (math.dist(_graph_node_point(graph, node_id), point), node_id),
    )


def nearest_graph_vertex_by_path_estimate(
    graph: nx.Graph,
    point: Point,
    geometry: GridGeometry,
    traversable: np.ndarray,
    *,
    candidate_limit: int = 3,
) -> Optional[int]:
    """Pick the nearest sparse vertex by a bounded obstacle-aware A* estimate."""
    fallback = nearest_graph_vertex(graph, point)
    if fallback is None:
        return None
    start = world_to_grid(point, geometry)
    if (
        start is None
        or traversable.shape != (geometry.height, geometry.width)
        or not traversable[start]
    ):
        return fallback
    candidates = sorted(
        graph.nodes,
        key=lambda node_id: (math.dist(_graph_node_point(graph, node_id), point), node_id),
    )
    limit = max(1, int(candidate_limit))
    candidates = candidates[:limit]
    scored = []
    for node_id in candidates:
        node_point = _graph_node_point(graph, node_id)
        goal = world_to_grid(node_point, geometry)
        if goal is None or not traversable[goal]:
            continue
        path = bidirectional_astar_path(traversable, start, goal)
        if not path:
            continue
        scored.append(
            (
                _grid_path_length(path, geometry.resolution),
                math.dist(node_point, point),
                node_id,
            )
        )
    if not scored:
        return fallback
    return min(scored)[2]


def _grid_path_length(path: Sequence[GridCell], resolution: float) -> float:
    return sum(
        math.dist(source, target) * resolution
        for source, target in zip(path, path[1:])
    )


def assign_frontiers_to_sparse_vertices(
    graph: nx.Graph,
    frontiers: Iterable[Any],
) -> Dict[int, List[Any]]:
    """Assign each safe frontier candidate to its nearest sparse GVD vertex."""
    assignments: Dict[int, List[Any]] = {node_id: [] for node_id in graph.nodes}
    if graph.number_of_nodes() == 0:
        return assignments
    for frontier in frontiers:
        point = getattr(getattr(frontier, 'safe_goal', None), 'point', None)
        if point is None:
            cluster = getattr(frontier, 'cluster', None)
            if cluster is None:
                continue
            point = (
                float(getattr(cluster, 'centroid_x')),
                float(getattr(cluster, 'centroid_y')),
            )
        node_id = nearest_graph_vertex(graph, point)
        if node_id is not None:
            assignments[node_id].append(frontier)
    return assignments


def guide_matching_vertices(
    graph: nx.Graph,
    points: Sequence[Point],
    migration_radius: float,
) -> Tuple[int, ...]:
    """Return sparse vertices matching previously reached world-space points."""
    radius = max(0.0, float(migration_radius))
    if graph.number_of_nodes() == 0 or not points:
        return ()
    return tuple(
        sorted(
            node_id
            for node_id in graph.nodes
            if any(
                math.dist(_graph_node_point(graph, node_id), point) <= radius
                for point in points
            )
        )
    )


def sparse_open_tsp_route(
    graph: nx.Graph,
    start: int,
    targets: Sequence[int],
) -> Tuple[int, ...]:
    """Return a deterministic open-TSP walk expanded onto the sparse graph."""
    if start not in graph:
        return ()
    targets = tuple(sorted({target for target in targets if target in graph}))
    if not targets:
        return (start,)
    if len(targets) == 1:
        return tuple(_shortest_graph_path(graph, start, targets[0]))

    metric = nx.Graph()
    metric.add_nodes_from(targets)
    for index, source in enumerate(targets):
        for target in targets[index + 1:]:
            try:
                distance = nx.shortest_path_length(graph, source, target, weight='weight')
            except nx.NetworkXNoPath:
                continue
            metric.add_edge(source, target, weight=float(distance))
    if not nx.is_connected(metric):
        reachable = [
            node_id
            for node_id in targets
            if nx.has_path(graph, start, node_id)
        ]
        if not reachable:
            return (start,)
        return tuple(_greedy_sparse_route(graph, start, reachable))
    tsp_route = list(
        nx.approximation.traveling_salesman_problem(
            metric,
            cycle=False,
            weight='weight',
        )
    )
    if _route_endpoint_cost(graph, start, tsp_route[-1]) < _route_endpoint_cost(
        graph,
        start,
        tsp_route[0],
    ):
        tsp_route.reverse()
    expanded = list(_shortest_graph_path(graph, start, tsp_route[0]))
    for source, target in zip(tsp_route, tsp_route[1:]):
        expanded.extend(_shortest_graph_path(graph, source, target)[1:])
    return tuple(_deduplicate_adjacent(expanded))


def shortcut_gvd_guide_route(
    graph: nx.Graph,
    route: Sequence[int],
    geometry: GridGeometry,
    traversable: np.ndarray,
) -> Tuple[Tuple[int, ...], int]:
    """Remove repeated interior transit vertices when a clear local shortcut exists."""
    smoothed = list(_deduplicate_adjacent(route))
    if len(smoothed) < 3:
        return tuple(smoothed), 0
    shortcuts = 0
    changed = True
    while changed:
        changed = False
        counts = {}
        for vertex_id in smoothed:
            counts[vertex_id] = counts.get(vertex_id, 0) + 1
        for index in range(1, len(smoothed) - 1):
            middle = smoothed[index]
            if counts.get(middle, 0) <= 1:
                continue
            source = smoothed[index - 1]
            target = smoothed[index + 1]
            if source == target or source not in graph or target not in graph:
                continue
            if not _gvd_guide_shortcut_clear(graph, source, target, geometry, traversable):
                continue
            _ensure_gvd_guide_shortcut_edge(graph, source, target)
            del smoothed[index]
            smoothed = list(_deduplicate_adjacent(smoothed))
            shortcuts += 1
            changed = True
            break
    return tuple(smoothed), shortcuts


def insert_gvd_guide_loop_revisits(
    graph: nx.Graph,
    route: Sequence[int],
    path_cost_weight: float,
) -> Tuple[Tuple[_GVDGuideRouteVertex, ...], Tuple[Tuple[int, int], ...]]:
    """Insert optional direct-edge revisits using the GBSAE D-opt objective."""
    if not route:
        return (), ()
    from .gbsae_exploration import weighted_spanning_tree_d_opt

    steps = [_GVDGuideRouteVertex(route[0])]
    observed = nx.Graph()
    observed.add_node(route[0])
    visited = {route[0]}
    loop_edges = []
    for source, target in zip(route, route[1:]):
        _copy_weighted_edge(graph, observed, source, target)
        observed.add_node(target)
        visited.add(target)
        steps.append(_GVDGuideRouteVertex(target))
        baseline = weighted_spanning_tree_d_opt(observed)
        candidates = []
        for revisit in sorted(visited):
            if revisit == target or not graph.has_edge(target, revisit):
                continue
            edge = _node_edge_key(target, revisit)
            if observed.has_edge(*edge):
                continue
            expected = observed.copy()
            _copy_weighted_edge(graph, expected, *edge)
            gain = weighted_spanning_tree_d_opt(expected) - baseline
            extra_distance = 2.0 * float(graph.edges[edge].get('weight', 0.0))
            objective = gain - max(0.0, path_cost_weight) * extra_distance
            if objective > 1e-9:
                candidates.append((objective, -extra_distance, -revisit, revisit, edge))
        if not candidates:
            continue
        _, _, _, revisit, edge = max(candidates)
        _copy_weighted_edge(graph, observed, *edge)
        loop_edges.append(edge)
        steps.append(_GVDGuideRouteVertex(revisit, loop_revisit=True))
        steps.append(_GVDGuideRouteVertex(target, loop_revisit=True))
    return tuple(steps), tuple(loop_edges)


def gvd_guide_plan_steps(
    graph: nx.Graph,
    route_steps: Sequence[_GVDGuideRouteVertex],
    frontier_assignments: Dict[int, List[Any]],
    *,
    frontier_detour_weight: float,
    frontier_detour_max_extra_distance: float,
    frontier_detour_min_gain: float,
    max_waypoint_distance: float = math.inf,
) -> Tuple[GVDGuidePlanStep, ...]:
    """Convert route vertices into executable steps with at most one detour per segment."""
    if len(route_steps) < 2:
        return ()
    queue: List[GVDGuidePlanStep] = []
    used_frontiers = set()
    previous = route_steps[0].vertex_id
    for route_step in route_steps[1:]:
        target = route_step.vertex_id
        if not route_step.loop_revisit:
            detour = _best_frontier_detour(
                graph,
                previous,
                target,
                frontier_assignments.get(target, ()),
                used_frontiers,
                frontier_detour_weight=frontier_detour_weight,
                frontier_detour_max_extra_distance=frontier_detour_max_extra_distance,
                frontier_detour_min_gain=frontier_detour_min_gain,
            )
            if detour is not None:
                queue.append(detour)
                used_frontiers.add(id(detour.frontier))
        edge = _node_edge_key(previous, target) if graph.has_edge(previous, target) else None
        queue.extend(
            _split_gvd_guide_waypoints(
                graph,
                previous,
                target,
                kind='loop_revisit' if route_step.loop_revisit else 'gvd_vertex',
                edge=edge,
                optional=route_step.loop_revisit,
                max_waypoint_distance=max_waypoint_distance,
            )
        )
        previous = target
    return tuple(queue)


def gvd_guide_edge_blocked_run(
    polyline: Sequence[Point],
    geometry: GridGeometry,
    traversable: np.ndarray,
) -> float:
    """Return the longest continuous blocked run along a sparse edge polyline."""
    if len(polyline) < 2 or traversable.shape != (geometry.height, geometry.width):
        return 0.0
    longest = 0.0
    current = 0.0
    previous_blocked = _point_blocked(polyline[0], geometry, traversable)
    for source, target in zip(polyline, polyline[1:]):
        target_blocked = _point_blocked(target, geometry, traversable)
        length = math.dist(source, target)
        if previous_blocked or target_blocked:
            current += length
            longest = max(longest, current)
        else:
            current = 0.0
        previous_blocked = target_blocked
    return longest


def off_graph_new_free_area(
    rebuild_grid: np.ndarray,
    current_grid: np.ndarray,
    geometry: GridGeometry,
    graph: nx.Graph,
    distance_threshold: float,
) -> float:
    """Measure newly free area that is not represented by the old sparse GVD."""
    if (
        rebuild_grid.shape != current_grid.shape
        or current_grid.shape != (geometry.height, geometry.width)
        or geometry.resolution <= 0.0
    ):
        return 0.0
    new_free = np.logical_and(rebuild_grid == -1, current_grid == 0)
    if not np.any(new_free):
        return 0.0
    graph_mask = sparse_graph_mask(graph, geometry)
    if np.any(graph_mask):
        distances = distance_to_mask(graph_mask, geometry.resolution)
        off_graph = distances > max(0.0, distance_threshold)
    else:
        off_graph = np.ones_like(new_free, dtype=bool)
    cells = int(np.count_nonzero(np.logical_and(new_free, off_graph)))
    return cells * geometry.resolution * geometry.resolution


def sparse_graph_mask(graph: nx.Graph, geometry: GridGeometry) -> np.ndarray:
    """Rasterize sparse graph edges and nodes into a map-aligned mask."""
    mask = np.zeros((geometry.height, geometry.width), dtype=bool)
    for node_id in graph.nodes:
        cell = world_to_grid(_graph_node_point(graph, node_id), geometry)
        if cell is not None:
            mask[cell] = True
    for _, _, attributes in graph.edges(data=True):
        polyline = tuple(attributes.get('polyline', ()))
        for point in polyline:
            cell = world_to_grid(point, geometry)
            if cell is not None:
                mask[cell] = True
        for source, target in zip(polyline, polyline[1:]):
            _rasterize_segment(mask, geometry, source, target)
    return mask


def _skeleton_components(adjacency: Dict[GridCell, List[GridCell]]):
    remaining = set(adjacency)
    while remaining:
        start = min(remaining)
        stack = [start]
        component = set()
        remaining.remove(start)
        while stack:
            cell = stack.pop()
            component.add(cell)
            for neighbor in adjacency[cell]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        yield component


def _sparse_vertex_kind(
    cell: GridCell,
    adjacency: Dict[GridCell, List[GridCell]],
    key_cells: Set[GridCell],
) -> str:
    degree = len(adjacency[cell])
    if degree <= 1:
        return 'endpoint'
    if degree >= 3:
        return 'branch'
    return 'cycle_anchor' if cell in key_cells else 'support'


def _sparse_topology_vertex_kind(
    graph: nx.Graph,
    node_id: int,
    key_nodes: Set[int],
) -> str:
    degree = graph.degree(node_id)
    if degree <= 1:
        return 'endpoint'
    if degree >= 3:
        return 'branch'
    return 'cycle_anchor' if node_id in key_nodes else 'support'


def _topology_chain_length(graph: nx.Graph, chain: Sequence[int]) -> float:
    return sum(
        float(
            graph.edges[source, target].get(
                'weight',
                math.dist(
                    _graph_node_point(graph, source),
                    _graph_node_point(graph, target),
                ),
            )
        )
        for source, target in zip(chain, chain[1:])
    )


def _topology_chain_polyline(
    graph: nx.Graph,
    chain: Sequence[int],
    geometry: GridGeometry,
) -> Tuple[Point, ...]:
    if not chain:
        return ()
    points: List[Point] = [_graph_node_point(graph, chain[0])]
    for source, target in zip(chain, chain[1:]):
        segment = _topology_edge_polyline(graph, source, target, geometry)
        if points and segment and math.dist(points[-1], segment[0]) <= 1e-9:
            points.extend(segment[1:])
        else:
            points.extend(segment)
    return tuple(points)


def _topology_edge_polyline(
    graph: nx.Graph,
    source: int,
    target: int,
    geometry: GridGeometry,
) -> Tuple[Point, ...]:
    attributes = graph.edges[source, target]
    values = attributes.get('polyline') or attributes.get('path') or ()
    if not values and attributes.get('cells'):
        values = tuple(grid_to_world(cell, geometry) for cell in attributes['cells'])
    points = tuple((float(point[0]), float(point[1])) for point in values)
    source_point = _graph_node_point(graph, source)
    target_point = _graph_node_point(graph, target)
    if not points:
        points = (source_point, target_point)
    forward_error = math.dist(points[0], source_point) + math.dist(points[-1], target_point)
    reverse_error = math.dist(points[-1], source_point) + math.dist(points[0], target_point)
    if reverse_error < forward_error:
        points = tuple(reversed(points))
    if math.dist(points[0], source_point) > 1e-9:
        points = (source_point,) + points
    if math.dist(points[-1], target_point) > 1e-9:
        points = points + (target_point,)
    return points


def _greedy_sparse_route(graph: nx.Graph, start: int, targets: Sequence[int]) -> List[int]:
    route = [start]
    remaining = set(targets)
    while remaining:
        current = route[-1]
        target = min(
            remaining,
            key=lambda node_id: (
                nx.shortest_path_length(graph, current, node_id, weight='weight'),
                node_id,
            ),
        )
        segment = _shortest_graph_path(graph, current, target)
        route.extend(segment[1:])
        remaining.discard(target)
    return _deduplicate_adjacent(route)


def _shortest_graph_path(graph: nx.Graph, source: int, target: int) -> List[int]:
    paths = nx.all_shortest_paths(graph, source, target, weight='weight')
    return list(min(tuple(path) for path in paths))


def _route_endpoint_cost(graph: nx.Graph, start: int, endpoint: int) -> float:
    return float(nx.shortest_path_length(graph, start, endpoint, weight='weight'))


def _copy_weighted_edge(source_graph: nx.Graph, target_graph: nx.Graph, source: int, target: int):
    target_graph.add_edge(source, target, **source_graph.edges[source, target])


def _node_edge_key(source: int, target: int) -> Tuple[int, int]:
    return (source, target) if source <= target else (target, source)


def _gvd_guide_shortcut_clear(
    graph: nx.Graph,
    source: int,
    target: int,
    geometry: GridGeometry,
    traversable: np.ndarray,
) -> bool:
    if traversable.shape != (geometry.height, geometry.width):
        return False
    source_point = _graph_node_point(graph, source)
    target_point = _graph_node_point(graph, target)
    distance = math.dist(source_point, target_point)
    steps = max(1, int(math.ceil(distance / max(geometry.resolution * 0.5, 1e-9))))
    for index in range(steps + 1):
        ratio = index / steps
        point = (
            source_point[0] + (target_point[0] - source_point[0]) * ratio,
            source_point[1] + (target_point[1] - source_point[1]) * ratio,
        )
        cell = world_to_grid(point, geometry)
        if cell is None or not traversable[cell]:
            return False
    return True


def _ensure_gvd_guide_shortcut_edge(graph: nx.Graph, source: int, target: int):
    source_point = _graph_node_point(graph, source)
    target_point = _graph_node_point(graph, target)
    length = max(math.dist(source_point, target_point), 1e-6)
    edge = _node_edge_key(source, target)
    candidate = {
        'weight': length,
        'information_weight': 1.0 / length,
        'polyline': (source_point, target_point),
        'source_graph_nodes': edge,
        'connection_mode': 'shortcut',
        'connection_modes': ('shortcut',),
        'blocked_run': 0.0,
        'stale': False,
    }
    existing = graph.get_edge_data(*edge)
    if existing is None or length < float(existing.get('weight', math.inf)):
        graph.add_edge(*edge, **candidate)


def _best_frontier_detour(
    graph: nx.Graph,
    source: int,
    target: int,
    candidates: Iterable[Any],
    used_frontiers: Set[int],
    *,
    frontier_detour_weight: float,
    frontier_detour_max_extra_distance: float,
    frontier_detour_min_gain: float,
) -> Optional[GVDGuidePlanStep]:
    if source not in graph or target not in graph:
        return None
    try:
        direct = float(nx.shortest_path_length(graph, source, target, weight='weight'))
    except nx.NetworkXNoPath:
        direct = math.dist(_graph_node_point(graph, source), _graph_node_point(graph, target))
    source_point = _graph_node_point(graph, source)
    target_point = _graph_node_point(graph, target)
    scored = []
    for frontier in candidates:
        if id(frontier) in used_frontiers:
            continue
        goal = getattr(getattr(frontier, 'safe_goal', None), 'point', None)
        if goal is None:
            continue
        gain = float(getattr(frontier, 'information_gain', 0.0))
        if gain < max(0.0, frontier_detour_min_gain):
            continue
        detour_cost = math.dist(source_point, goal) + math.dist(goal, target_point)
        extra = detour_cost - direct
        if extra < 0.0:
            extra = 0.0
        if extra > max(0.0, frontier_detour_max_extra_distance):
            continue
        score = gain - max(0.0, frontier_detour_weight) * extra
        if score <= 0.0:
            continue
        scored.append((score, gain, -extra, goal, frontier))
    if not scored:
        return None
    score, gain, neg_extra, goal, frontier = max(scored)
    return GVDGuidePlanStep(
        'frontier_detour',
        goal,
        source_vertex=source,
        target_vertex=target,
        frontier=frontier,
        expected_cost=-neg_extra,
        optional=True,
    )


def _split_gvd_guide_waypoints(
    graph: nx.Graph,
    source: int,
    target: int,
    *,
    kind: str,
    edge: Optional[Tuple[int, int]],
    optional: bool,
    max_waypoint_distance: float,
) -> Tuple[GVDGuidePlanStep, ...]:
    source_point = _graph_node_point(graph, source)
    target_point = _graph_node_point(graph, target)
    expected_cost = (
        float(graph.edges[edge].get('weight', 0.0))
        if edge is not None
        else math.dist(source_point, target_point)
    )
    polyline = (
        tuple(graph.edges[edge].get('polyline', ()))
        if edge is not None and graph.has_edge(*edge)
        else ()
    )
    if len(polyline) < 2:
        polyline = (source_point, target_point)
    if math.dist(polyline[0], source_point) > math.dist(polyline[-1], source_point):
        polyline = tuple(reversed(polyline))
    path_length = _polyline_length(polyline)
    if path_length <= 1e-9:
        path_length = expected_cost
    split_distance = max(0.0, float(max_waypoint_distance))
    if not math.isfinite(split_distance) or split_distance <= 0.0:
        split_distance = math.inf
    distances = []
    if math.isfinite(split_distance) and path_length > split_distance:
        next_distance = split_distance
        while next_distance < path_length - 1e-9:
            distances.append(next_distance)
            next_distance += split_distance
    points = [
        _point_along_polyline(polyline, distance)
        for distance in distances
    ]
    points.append(target_point)
    steps = []
    previous_distance = 0.0
    for index, point in enumerate(points):
        is_final = index == len(points) - 1
        distance = distances[index] if not is_final else path_length
        steps.append(
            GVDGuidePlanStep(
                kind,
                point,
                vertex_id=target if is_final else None,
                source_vertex=source,
                target_vertex=target,
                edge=edge,
                expected_cost=max(0.0, distance - previous_distance),
                optional=optional,
            )
        )
        previous_distance = distance
    return tuple(steps)


def _point_along_polyline(polyline: Sequence[Point], distance: float) -> Point:
    if not polyline:
        return (0.0, 0.0)
    remaining = max(0.0, float(distance))
    for source, target in zip(polyline, polyline[1:]):
        segment = math.dist(source, target)
        if segment <= 1e-9:
            continue
        if remaining <= segment:
            ratio = remaining / segment
            return (
                source[0] + (target[0] - source[0]) * ratio,
                source[1] + (target[1] - source[1]) * ratio,
            )
        remaining -= segment
    return polyline[-1]


def _polyline_length(polyline: Sequence[Point]) -> float:
    return sum(math.dist(source, target) for source, target in zip(polyline, polyline[1:]))


def _point_blocked(point: Point, geometry: GridGeometry, traversable: np.ndarray) -> bool:
    cell = world_to_grid(point, geometry)
    return cell is None or not traversable[cell]


def _rasterize_segment(
    mask: np.ndarray,
    geometry: GridGeometry,
    source: Point,
    target: Point,
):
    distance = math.dist(source, target)
    steps = max(1, int(math.ceil(distance / max(geometry.resolution * 0.5, 1e-9))))
    for index in range(steps + 1):
        ratio = index / steps
        point = (
            source[0] + (target[0] - source[0]) * ratio,
            source[1] + (target[1] - source[1]) * ratio,
        )
        cell = world_to_grid(point, geometry)
        if cell is not None:
            mask[cell] = True


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
    graph = normalize_topology_vertex_kinds(graph)
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


def local_region_known_ratio(
    grid: np.ndarray,
    region_mask: np.ndarray,
    *,
    grid_geometry: Optional[GridGeometry] = None,
    region_geometry: Optional[GridGeometry] = None,
) -> float:
    """Return the observed-cell fraction inside one local cleanup Region."""
    if (grid_geometry is None) != (region_geometry is None):
        return 0.0
    region_cells = int(np.count_nonzero(region_mask))
    if region_cells == 0:
        return 0.0
    if grid_geometry is None:
        if grid.shape != region_mask.shape:
            return 0.0
        known_cells = int(np.count_nonzero(np.logical_and(region_mask, grid != -1)))
        return float(known_cells) / float(region_cells)
    if (
        grid_geometry.resolution <= 0.0
        or region_geometry.resolution <= 0.0
        or grid.shape != (grid_geometry.height, grid_geometry.width)
        or region_mask.shape != (region_geometry.height, region_geometry.width)
    ):
        return 0.0
    region_rows, region_cols = np.nonzero(region_mask)
    world_x = (
        region_geometry.origin_x
        + (region_cols.astype(float) + 0.5) * region_geometry.resolution
    )
    world_y = (
        region_geometry.origin_y
        + (region_rows.astype(float) + 0.5) * region_geometry.resolution
    )
    grid_cols = np.floor(
        (world_x - grid_geometry.origin_x) / grid_geometry.resolution
    ).astype(int)
    grid_rows = np.floor(
        (world_y - grid_geometry.origin_y) / grid_geometry.resolution
    ).astype(int)
    inside = np.logical_and.reduce(
        (
            grid_rows >= 0,
            grid_rows < grid_geometry.height,
            grid_cols >= 0,
            grid_cols < grid_geometry.width,
        )
    )
    known_cells = int(
        np.count_nonzero(
            grid[grid_rows[inside], grid_cols[inside]] != -1
        )
    )
    return float(known_cells) / float(region_cells)


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
                segment_cells = tuple(chain[source_index:target_index + 1])
                distance = _chain_length(segment_cells, geometry.resolution)
                polyline = tuple(grid_to_world(cell, geometry) for cell in segment_cells)
                graph.add_edge(
                    source,
                    target,
                    weight=distance,
                    information_weight=1.0 / max(distance, 1e-6),
                    virtual=False,
                    connection_mode='gvd',
                    cells=segment_cells,
                    polyline=polyline,
                    path=polyline,
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
    """Globally prune prunable edges that form cycles with confident structure.

    1. **Confident–confident** edges are unconditionally kept (even when
       they form cycles among themselves).
    2. Every edge that touches at least one unconfident endpoint is
       *prunable* and competes in a single Union-Find pass ordered by
       descending confidence.  An edge that would close a cycle is
       removed so that fake loops through unknown-heavy territory cannot
       distort the TSP.
    """
    pruned = graph.copy()
    threshold = min(1.0, max(0.0, float(ratio_threshold)))

    # ---- tag every vertex ----
    unconfident_ids = set()
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
            unconfident_ids.add(node_id)

    # ---- classify edges ----
    # Confidence = 1 – effective_unknown_ratio.
    # Pure unconfident edges use the worse endpoint directly.
    # Cross edges (one confident endpoint) are strongly preferred:
    # the confident side contributes −0.5 instead of 0, giving them
    # an effective bonus of 0.5 in confidence space.
    prunable = []  # (-confidence, u_tie, v_tie, u, v)
    for u, v in pruned.edges:
        u_unconf = pruned.nodes[u]['unconfident']
        v_unconf = pruned.nodes[v]['unconfident']
        if not u_unconf and not v_unconf:
            continue  # confident–confident  → never removed
        if u_unconf and v_unconf:
            # Both unconfident — full penalty.
            effective = max(
                pruned.nodes[u]['unknown_ratio'],
                pruned.nodes[v]['unknown_ratio'],
            )
        else:
            # Cross edge — confident side gives a bonus.
            unconf_ratio = (
                pruned.nodes[u]['unknown_ratio'] if u_unconf
                else pruned.nodes[v]['unknown_ratio']
            )
            effective = max(0.0, unconf_ratio - 0.5)
        confidence = 1.0 - effective
        prunable.append((-confidence, u, v, u, v))

    # ---- Union-Find ----
    parent = {n: n for n in pruned.nodes}

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x, y):
        rx, ry = _find(x), _find(y)
        if rx != ry:
            parent[rx] = ry
            return True
        return False

    # Confident–confident edges establish the initial skeleton.
    for u, v in pruned.edges:
        if (
            not pruned.nodes[u]['unconfident']
            and not pruned.nodes[v]['unconfident']
        ):
            _union(u, v)

    # All prunable edges compete by descending confidence.
    prunable.sort()
    removed_edges = 0
    for _neg_conf, _tu, _tv, u, v in prunable:
        if _union(u, v):
            continue
        pruned.remove_edge(u, v)
        removed_edges += 1

    return pruned, UnknownCycleSuppressionStats(
        len(unconfident_ids), removed_edges,
    )


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


def offset_repeated_route_segments(
    route_points: Sequence[Point],
    spacing: float = 0.10,
) -> Tuple[Tuple[Point, Point], ...]:
    """Offset repeated undirected route segments so every traversal stays visible."""
    segments = tuple(zip(route_points, route_points[1:]))
    counts = {}
    for source, target in segments:
        key = tuple(sorted((source, target)))
        counts[key] = counts.get(key, 0) + 1
    occurrences = {}
    shifted = []
    for source, target in segments:
        key = tuple(sorted((source, target)))
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        offset = (occurrence - 0.5 * (counts[key] - 1)) * max(0.0, spacing)
        canonical_source, canonical_target = key
        dx = canonical_target[0] - canonical_source[0]
        dy = canonical_target[1] - canonical_source[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            shifted.append((source, target))
            continue
        ox = -dy * offset / length
        oy = dx * offset / length
        shifted.append(
            (
                (source[0] + ox, source[1] + oy),
                (target[0] + ox, target[1] + oy),
            )
        )
    return tuple(shifted)


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
    guide_planner: Optional[GVDGuidePlanner] = None,
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

    # Topological graph edges — this is what the TSP actually sees after
    # clustering, repair, and cycle suppression.
    topo_graph = _marker(
        Marker, frame_id, stamp, 'gvd_topo_graph', 11, Marker.LINE_LIST,
    )
    topo_graph.scale.x = 0.04
    topo_graph.color.a = 0.8
    topo_graph.color.r = 0.2
    topo_graph.color.g = 0.9
    topo_graph.color.b = 0.5
    for source, target in topology.graph.edges:
        topo_graph.points.append(
            _marker_point(MarkerPoint, _graph_node_point(topology.graph, source), 0.0),
        )
        topo_graph.points.append(
            _marker_point(MarkerPoint, _graph_node_point(topology.graph, target), 0.0),
        )
    markers.markers.append(topo_graph)

    path = _marker(Marker, frame_id, stamp, 'gvd_active_path', 2, Marker.LINE_STRIP)
    path.scale.x = 0.07
    path.color.a = 0.95
    path.color.r = 1.0
    path.color.g = 0.35
    for point in active_path:
        path.points.append(_marker_point(MarkerPoint, point, 0.12))
    markers.markers.append(path)
    if hierarchical_tracker is not None:
        unexplored = _marker(
            Marker,
            frame_id,
            stamp,
            'gvd_unexplored_nodes',
            12,
            Marker.SPHERE_LIST,
        )
        unexplored.scale.x = unexplored.scale.y = unexplored.scale.z = 0.17
        unexplored.color.a = 0.95
        unexplored.color.r = 0.62
        unexplored.color.b = 0.95
        for node_id in sorted(
            set(topology.graph.nodes)
            - hierarchical_tracker.explored_vertices
            - hierarchical_tracker.cleared_vertices
        ):
            unexplored.points.append(
                _marker_point(MarkerPoint, _graph_node_point(topology.graph, node_id), 0.10)
            )
        markers.markers.append(unexplored)

        explored = _marker(Marker, frame_id, stamp, 'gvd_explored_nodes', 3, Marker.SPHERE_LIST)
        explored.scale.x = explored.scale.y = explored.scale.z = 0.16
        explored.color.a = 0.95
        explored.color.r = 1.0
        explored.color.g = 0.42
        explored.color.b = 0.68
        for node_id in sorted(
            hierarchical_tracker.explored_vertices
            - hierarchical_tracker.cleared_vertices
        ):
            if node_id in topology.graph:
                explored.points.append(
                    _marker_point(MarkerPoint, _graph_node_point(topology.graph, node_id), 0.10)
                )
        markers.markers.append(explored)

        cleared = _marker(Marker, frame_id, stamp, 'gvd_cleared_nodes', 4, Marker.SPHERE_LIST)
        cleared.scale.x = cleared.scale.y = cleared.scale.z = 0.20
        cleared.color.a = 0.95
        cleared.color.r = 1.0
        cleared.color.g = 0.52
        cleared.color.b = 0.05
        for node_id in sorted(hierarchical_tracker.cleared_vertices):
            if node_id in topology.graph:
                cleared.points.append(
                    _marker_point(MarkerPoint, _graph_node_point(topology.graph, node_id), 0.13)
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
        route_points = hierarchical_tracker.route_points
        route_segments = offset_repeated_route_segments(route_points)
        tsp_route = _marker(
            Marker,
            frame_id,
            stamp,
            'gvd_hierarchical_tsp_route',
            8,
            Marker.LINE_LIST,
        )
        tsp_route.scale.x = 0.065
        tsp_route.color.a = 0.9
        tsp_route.color.r = 1.0
        tsp_route.color.g = 0.75
        for source, target in route_segments:
            tsp_route.points.append(_marker_point(MarkerPoint, source, 0.11))
            tsp_route.points.append(_marker_point(MarkerPoint, target, 0.11))
        markers.markers.append(tsp_route)

        # Draw every traversal independently, including repeated reverse segments.
        for index, (source, target) in enumerate(route_segments):
            arrow = _marker(
                Marker,
                frame_id,
                stamp,
                'gvd_hierarchical_tsp_arrows',
                100 + index,
                Marker.ARROW,
            )
            arrow.scale.x = 0.04   # shaft diameter
            arrow.scale.y = 0.08   # head diameter
            arrow.scale.z = 0.12   # head length
            arrow.color.a = 0.85
            arrow.color.r = 1.0
            arrow.color.g = 0.45
            arrow.color.b = 0.0
            arrow.points.append(_marker_point(MarkerPoint, source, 0.12))
            arrow.points.append(_marker_point(MarkerPoint, target, 0.12))
            markers.markers.append(arrow)
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
    if guide_planner is not None:
        _append_gvd_guide_markers(markers, Marker, MarkerPoint, guide_planner, frame_id, stamp)
    return markers


def _append_gvd_guide_markers(
    markers,
    marker_class,
    point_class,
    guide_planner: GVDGuidePlanner,
    frame_id: str,
    stamp,
):
    sparse_nodes = _marker(
        marker_class,
        frame_id,
        stamp,
        'gvd_guide_sparse_nodes',
        30,
        marker_class.SPHERE_LIST,
    )
    sparse_nodes.scale.x = sparse_nodes.scale.y = sparse_nodes.scale.z = 0.18
    sparse_nodes.color.a = 0.95
    sparse_nodes.color.r = 1.0
    sparse_nodes.color.g = 0.95
    for node_id in sorted(guide_planner.graph.nodes):
        if guide_planner.graph.nodes[node_id].get('explored'):
            continue
        sparse_nodes.points.append(
            _marker_point(point_class, _graph_node_point(guide_planner.graph, node_id), 0.18)
        )
    markers.markers.append(sparse_nodes)

    explored_nodes = _marker(
        marker_class,
        frame_id,
        stamp,
        'gvd_guide_explored_sparse_nodes',
        35,
        marker_class.SPHERE_LIST,
    )
    explored_nodes.scale.x = explored_nodes.scale.y = explored_nodes.scale.z = 0.20
    explored_nodes.color.a = 0.95
    explored_nodes.color.r = 0.15
    explored_nodes.color.g = 0.45
    explored_nodes.color.b = 1.0
    for node_id in sorted(guide_planner.graph.nodes):
        if not guide_planner.graph.nodes[node_id].get('explored'):
            continue
        explored_nodes.points.append(
            _marker_point(point_class, _graph_node_point(guide_planner.graph, node_id), 0.20)
        )
    markers.markers.append(explored_nodes)

    sparse_edges = _marker(
        marker_class,
        frame_id,
        stamp,
        'gvd_guide_sparse_edges',
        31,
        marker_class.LINE_LIST,
    )
    sparse_edges.scale.x = 0.06
    sparse_edges.color.a = 0.9
    sparse_edges.color.r = 0.1
    sparse_edges.color.g = 1.0
    sparse_edges.color.b = 0.7
    for _, _, attributes in guide_planner.graph.edges(data=True):
        polyline = tuple(attributes.get('polyline', ()))
        for source, target in zip(polyline, polyline[1:]):
            sparse_edges.points.append(_marker_point(point_class, source, 0.13))
            sparse_edges.points.append(_marker_point(point_class, target, 0.13))
    markers.markers.append(sparse_edges)

    route = _marker(
        marker_class,
        frame_id,
        stamp,
        'gvd_guide_planned_route',
        32,
        marker_class.LINE_LIST,
    )
    route.scale.x = 0.075
    route.color.a = 0.9
    route.color.r = 1.0
    route.color.g = 0.75
    points = [step.goal_xy for step in guide_planner.remaining_steps]
    for source, target in zip(points, points[1:]):
        route.points.append(_marker_point(point_class, source, 0.22))
        route.points.append(_marker_point(point_class, target, 0.22))
    markers.markers.append(route)

    loops = _marker(
        marker_class,
        frame_id,
        stamp,
        'gvd_guide_loop_revisits',
        33,
        marker_class.LINE_LIST,
    )
    loops.scale.x = 0.11
    loops.color.a = 0.95
    loops.color.r = 1.0
    for source, target in guide_planner.loop_edges:
        if source in guide_planner.graph and target in guide_planner.graph:
            loops.points.append(
                _marker_point(point_class, _graph_node_point(guide_planner.graph, source), 0.24)
            )
            loops.points.append(
                _marker_point(point_class, _graph_node_point(guide_planner.graph, target), 0.24)
            )
    markers.markers.append(loops)

    detours = _marker(
        marker_class,
        frame_id,
        stamp,
        'gvd_guide_frontier_detours',
        34,
        marker_class.SPHERE_LIST,
    )
    detours.scale.x = detours.scale.y = detours.scale.z = 0.20
    detours.color.a = 0.95
    detours.color.g = 1.0
    detours.color.b = 0.2
    for step in guide_planner.remaining_steps:
        if step.kind == 'frontier_detour':
            detours.points.append(_marker_point(point_class, step.goal_xy, 0.24))
    markers.markers.append(detours)


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


def normalize_topology_vertex_kinds(graph: nx.Graph) -> nx.Graph:
    """Refresh structural kinds after clustering, repair, and cycle pruning."""
    normalized = graph.copy()
    for node_id in normalized.nodes:
        degree = normalized.degree(node_id)
        if degree <= 1:
            kind = 'endpoint'
        elif degree >= 3:
            kind = 'branch'
        elif normalized.nodes[node_id].get('kind') == 'corner':
            kind = 'corner'
        else:
            kind = 'support'
        normalized.nodes[node_id]['kind'] = kind
    return normalized


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
