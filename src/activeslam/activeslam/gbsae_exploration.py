"""Prior-graph route planning helpers for the ROS 2 GBSAE adaptation."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


Point = Tuple[float, float]
Edge = Tuple[int, int]


@dataclass(frozen=True)
class RouteStep:
    """One prior-graph waypoint, including optional spectral loop revisits."""

    vertex_id: int
    loop_revisit: bool = False


def load_prior_graph(path: Path, expected_world: Optional[str] = None) -> nx.Graph:
    """Load and validate a topo-metric prior graph from a JSON asset."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'GBSAE prior graph is missing: {path}')
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid GBSAE prior graph JSON in {path}: {exc}') from exc

    if not isinstance(payload, dict):
        raise ValueError('GBSAE prior graph JSON root must be an object.')
    world = payload.get('world')
    if not isinstance(world, str) or not world:
        raise ValueError('GBSAE prior graph must define a non-empty world string.')
    if expected_world is not None and world != expected_world:
        raise ValueError(
            f'GBSAE prior graph world mismatch: expected {expected_world}, got {world}.'
        )

    nodes = payload.get('nodes')
    edges = payload.get('edges')
    if not isinstance(nodes, list) or len(nodes) < 2:
        raise ValueError('GBSAE prior graph requires at least two nodes.')
    if not isinstance(edges, list):
        raise ValueError('GBSAE prior graph edges must be a list.')

    graph = nx.Graph(world=world, source_path=str(path))
    for raw_node in nodes:
        if not isinstance(raw_node, dict):
            raise ValueError('GBSAE prior graph nodes must be objects.')
        node_id = raw_node.get('id')
        x = raw_node.get('x')
        y = raw_node.get('y')
        if not isinstance(node_id, int) or isinstance(node_id, bool) or node_id < 0:
            raise ValueError(f'Invalid GBSAE prior graph node id: {node_id!r}.')
        if node_id in graph:
            raise ValueError(f'Duplicate GBSAE prior graph node id: {node_id}.')
        if (
            not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
            or not math.isfinite(x)
            or not math.isfinite(y)
        ):
            raise ValueError(f'Node {node_id} must have finite numeric x and y coordinates.')
        graph.add_node(node_id, x=float(x), y=float(y))

    seen_edges = set()
    for raw_edge in edges:
        if not isinstance(raw_edge, list) or len(raw_edge) != 2:
            raise ValueError(f'Invalid GBSAE prior graph edge: {raw_edge!r}.')
        source, target = raw_edge
        if (
            not isinstance(source, int)
            or isinstance(source, bool)
            or not isinstance(target, int)
            or isinstance(target, bool)
        ):
            raise ValueError(f'GBSAE prior graph edge IDs must be integers: {raw_edge!r}.')
        if source not in graph or target not in graph:
            raise ValueError(f'GBSAE prior graph edge references an unknown node: {raw_edge!r}.')
        if source == target:
            raise ValueError(f'GBSAE prior graph self-edge is not navigable: {raw_edge!r}.')
        edge = _edge_key(source, target)
        if edge in seen_edges:
            raise ValueError(f'Duplicate GBSAE prior graph edge: {raw_edge!r}.')
        seen_edges.add(edge)
        distance = math.dist(vertex_point(graph, source), vertex_point(graph, target))
        if distance <= 1e-6:
            raise ValueError(f'GBSAE prior graph edge has zero navigable length: {raw_edge!r}.')
        graph.add_edge(
            source,
            target,
            weight=distance,
            information_weight=1.0 / distance,
        )

    if not nx.is_connected(graph):
        raise ValueError('GBSAE prior graph must be connected.')
    if any(graph.degree[node_id] == 0 for node_id in graph):
        raise ValueError('GBSAE prior graph contains an isolated node.')
    return graph


def resolve_prior_graph_path(world_name: str) -> Path:
    """Resolve the installed per-world GBSAE graph asset."""

    from ament_index_python.packages import get_package_share_directory

    resource_dir = Path(get_package_share_directory('activeslam_resource'))
    return resource_dir / 'maps' / f'{world_name}.gbsae.json'


def vertex_point(graph: nx.Graph, vertex_id: int) -> Point:
    node = graph.nodes[vertex_id]
    return float(node['x']), float(node['y'])


def nearest_vertex(
    graph: nx.Graph,
    point: Point,
    candidates: Optional[Iterable[int]] = None,
) -> int:
    """Return the closest prior vertex with a stable ID tie break."""

    node_ids = graph.nodes if candidates is None else candidates
    return min(
        node_ids,
        key=lambda node_id: (math.dist(point, vertex_point(graph, node_id)), node_id),
    )


def shortest_path_expansion(graph: nx.Graph, vertices: Sequence[int]) -> List[int]:
    """Expand sparse route vertices into deterministic connected graph steps."""

    if not vertices:
        return []
    expanded = [vertices[0]]
    for source, target in zip(vertices, vertices[1:]):
        expanded.extend(_shortest_path(graph, source, target)[1:])
    return expanded


def greedy_visit_route(graph: nx.Graph, start_vertex: int) -> List[int]:
    """Build a deterministic nearest-unvisited route over the prior graph."""

    if start_vertex not in graph:
        raise ValueError(f'Unknown GBSAE start vertex: {start_vertex}.')
    route = [start_vertex]
    unvisited = set(graph.nodes) - {start_vertex}
    while unvisited:
        current = route[-1]
        target = min(
            unvisited,
            key=lambda node_id: (
                nx.shortest_path_length(graph, current, node_id, weight='weight'),
                node_id,
            ),
        )
        segment = _shortest_path(graph, current, target)
        route.extend(segment[1:])
        unvisited.difference_update(segment)
    return route


def weighted_spanning_tree_d_opt(graph: nx.Graph) -> float:
    """Return the normalized weighted-spanning-tree D-opt objective."""

    if graph.number_of_nodes() < 2 or not nx.is_connected(graph):
        return 0.0
    node_ids = sorted(graph.nodes)
    index = {node_id: offset for offset, node_id in enumerate(node_ids)}
    laplacian = np.zeros((len(node_ids), len(node_ids)), dtype=np.float64)
    for source, target, attributes in graph.edges(data=True):
        weight = float(attributes.get('information_weight', 1.0))
        if weight <= 0.0 or not math.isfinite(weight):
            continue
        i = index[source]
        j = index[target]
        laplacian[i, i] += weight
        laplacian[j, j] += weight
        laplacian[i, j] -= weight
        laplacian[j, i] -= weight
    reduced = laplacian[1:, 1:]
    sign, logdet = np.linalg.slogdet(reduced)
    if sign <= 0 or not math.isfinite(logdet):
        return 0.0
    return float(math.exp(logdet / reduced.shape[0]))


def insert_spectral_loop_revisits(
    graph: nx.Graph,
    route: Sequence[int],
    path_cost_weight: float,
) -> Tuple[List[RouteStep], List[Edge]]:
    """Insert optional revisits when spectral gain outweighs extra travel."""

    if not route:
        return [], []
    steps = [RouteStep(route[0])]
    observed = nx.Graph()
    observed.add_node(route[0])
    visited = {route[0]}
    loop_edges = []

    for source, target in zip(route, route[1:]):
        _copy_edge(graph, observed, source, target)
        observed.add_node(target)
        visited.add(target)
        steps.append(RouteStep(target))

        baseline = weighted_spanning_tree_d_opt(observed)
        candidates = []
        for revisit in sorted(visited):
            if revisit == target or not graph.has_edge(target, revisit):
                continue
            edge = _edge_key(target, revisit)
            if observed.has_edge(*edge):
                continue
            expected = observed.copy()
            _copy_edge(graph, expected, *edge)
            gain = weighted_spanning_tree_d_opt(expected) - baseline
            extra_distance = 2.0 * float(graph.edges[edge]['weight'])
            objective = gain - path_cost_weight * extra_distance
            if objective > 1e-9:
                candidates.append((objective, -extra_distance, -revisit, revisit, edge))
        if not candidates:
            continue

        _, _, _, revisit, edge = max(candidates)
        _copy_edge(graph, observed, *edge)
        loop_edges.append(edge)
        steps.append(RouteStep(revisit, loop_revisit=True))
        steps.append(RouteStep(target, loop_revisit=True))
    return steps, loop_edges


class GBSAEPlanner:
    """Track the active topo-metric route and frontier allocation state."""

    def __init__(
        self,
        graph: nx.Graph,
        initial_pose: Point,
        loop_path_cost_weight: float,
    ):
        self.graph = graph
        self.start_vertex = nearest_vertex(graph, initial_pose)
        greedy_route = greedy_visit_route(graph, self.start_vertex)
        self.route, self.loop_edges = insert_spectral_loop_revisits(
            graph,
            greedy_route,
            loop_path_cost_weight,
        )
        self.active_index = 0
        self.visited_prefix: List[int] = []
        self.frontier_assignments: Dict[int, List[object]] = {}

    @property
    def active_step(self) -> Optional[RouteStep]:
        if self.active_index >= len(self.route):
            return None
        return self.route[self.active_index]

    @property
    def completed_vertices(self):
        return set(self.visited_prefix)

    def advance_active_step(self) -> Optional[RouteStep]:
        step = self.active_step
        if step is not None:
            self.visited_prefix.append(step.vertex_id)
            self.active_index += 1
        return step

    def advance_reached_steps(self, robot_xy: Point, reach_radius: float) -> List[RouteStep]:
        advanced = []
        while self.active_step is not None:
            target = vertex_point(self.graph, self.active_step.vertex_id)
            if math.dist(robot_xy, target) > reach_radius:
                break
            step = self.advance_active_step()
            assert step is not None
            advanced.append(step)
        return advanced

    def skip_active_loop_revisit(self) -> RouteStep:
        step = self.active_step
        if step is None or not step.loop_revisit:
            raise ValueError('The active GBSAE route step is not an optional loop revisit.')
        self.advance_active_step()
        return step

    def allocate_frontiers(self, frontiers: Iterable[object]) -> Dict[int, List[object]]:
        """Assign frontiers to nearest uncompleted prior vertices."""

        active = self.active_step
        candidates = set(self.graph.nodes) - self.completed_vertices
        if active is not None:
            candidates.add(active.vertex_id)
        assignments: Dict[int, List[object]] = {node_id: [] for node_id in candidates}
        for frontier in frontiers:
            if not candidates:
                break
            node_id = nearest_vertex(
                self.graph,
                (frontier.cluster.centroid_x, frontier.cluster.centroid_y),
                candidates,
            )
            assignments[node_id].append(frontier)
        self.frontier_assignments = assignments
        return assignments

    def frontiers_for_active(self, frontiers: Iterable[object]) -> List[object]:
        active = self.active_step
        if active is None:
            return []
        assignments = self.allocate_frontiers(frontiers)
        target = vertex_point(self.graph, active.vertex_id)
        return sorted(
            assignments.get(active.vertex_id, []),
            key=lambda frontier: (
                math.dist(
                    (frontier.cluster.centroid_x, frontier.cluster.centroid_y),
                    target,
                ),
                -frontier.utility,
            ),
        )


def point_is_known_free(grid_msg, data: np.ndarray, point: Point) -> bool:
    """Return whether a world point currently maps to a known free grid cell."""

    info = grid_msg.info
    if info.resolution <= 0.0:
        return False
    column = math.floor((point[0] - info.origin.position.x) / info.resolution)
    row = math.floor((point[1] - info.origin.position.y) / info.resolution)
    return (
        0 <= row < info.height
        and 0 <= column < info.width
        and data[row, column] == 0
    )


def gbsae_to_marker_array(planner: GBSAEPlanner, frame_id: str, stamp):
    """Build RViz markers for the prior graph, route, active vertex, and loops."""

    from geometry_msgs.msg import Point as MarkerPoint
    from visualization_msgs.msg import Marker, MarkerArray

    markers = MarkerArray()
    delete = Marker()
    delete.action = Marker.DELETEALL
    markers.markers.append(delete)

    nodes = _marker(Marker, frame_id, stamp, 'gbsae_prior_nodes', 0, Marker.SPHERE_LIST)
    nodes.scale.x = nodes.scale.y = nodes.scale.z = 0.14
    nodes.color.a = 0.9
    nodes.color.b = 1.0
    for node_id in sorted(planner.graph.nodes):
        nodes.points.append(_marker_point(MarkerPoint, vertex_point(planner.graph, node_id), 0.08))
    markers.markers.append(nodes)

    edges = _marker(Marker, frame_id, stamp, 'gbsae_prior_edges', 1, Marker.LINE_LIST)
    edges.scale.x = 0.035
    edges.color.a = 0.55
    edges.color.g = 0.7
    edges.color.b = 1.0
    for source, target in planner.graph.edges:
        edges.points.append(_marker_point(MarkerPoint, vertex_point(planner.graph, source), 0.04))
        edges.points.append(_marker_point(MarkerPoint, vertex_point(planner.graph, target), 0.04))
    markers.markers.append(edges)

    route = _marker(Marker, frame_id, stamp, 'gbsae_route', 2, Marker.LINE_STRIP)
    route.scale.x = 0.06
    route.color.a = 0.85
    route.color.r = 0.9
    route.color.g = 0.75
    for step in planner.route:
        route.points.append(
            _marker_point(MarkerPoint, vertex_point(planner.graph, step.vertex_id), 0.10)
        )
    markers.markers.append(route)

    loops = _marker(Marker, frame_id, stamp, 'gbsae_loop_edges', 3, Marker.LINE_LIST)
    loops.scale.x = 0.11
    loops.color.a = 0.95
    loops.color.r = 1.0
    for source, target in planner.loop_edges:
        loops.points.append(_marker_point(MarkerPoint, vertex_point(planner.graph, source), 0.14))
        loops.points.append(_marker_point(MarkerPoint, vertex_point(planner.graph, target), 0.14))
    markers.markers.append(loops)

    active = planner.active_step
    if active is not None:
        marker = _marker(Marker, frame_id, stamp, 'gbsae_active_vertex', 4, Marker.SPHERE)
        marker.pose.position.x, marker.pose.position.y = vertex_point(
            planner.graph,
            active.vertex_id,
        )
        marker.pose.position.z = 0.18
        marker.scale.x = marker.scale.y = marker.scale.z = 0.28
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.2
        markers.markers.append(marker)
    return markers


def _shortest_path(graph: nx.Graph, source: int, target: int) -> List[int]:
    paths = nx.all_shortest_paths(graph, source, target, weight='weight')
    return list(min(tuple(path) for path in paths))


def _edge_key(source: int, target: int) -> Edge:
    return min(source, target), max(source, target)


def _copy_edge(source_graph: nx.Graph, target_graph: nx.Graph, source: int, target: int):
    target_graph.add_edge(source, target, **source_graph.edges[source, target])


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
