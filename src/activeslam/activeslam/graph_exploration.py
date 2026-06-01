import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class PoseGraphNode:
    node_id: int
    x: float
    y: float
    yaw: float


@dataclass
class PoseGraphEdge:
    source: int
    target: int
    edge_type: int
    information: np.ndarray


class WeightedPoseGraph:
    def __init__(self):
        self.nodes: List[PoseGraphNode] = []
        self.edges: List[PoseGraphEdge] = []
        self._edge_keys = set()

    def copy(self) -> 'WeightedPoseGraph':
        graph = WeightedPoseGraph()
        graph.nodes = [
            PoseGraphNode(n.node_id, n.x, n.y, n.yaw)
            for n in self.nodes
        ]
        graph.edges = [
            PoseGraphEdge(e.source, e.target, e.edge_type, e.information.copy())
            for e in self.edges
        ]
        graph._edge_keys = set(self._edge_keys)
        return graph

    def add_node(self, x: float, y: float, yaw: float) -> PoseGraphNode:
        node = PoseGraphNode(len(self.nodes), x, y, yaw)
        self.nodes.append(node)
        return node

    def add_edge(self, source: int, target: int, edge_type: int, information: np.ndarray) -> bool:
        if source == target:
            return False
        key = (min(source, target), max(source, target), edge_type)
        if key in self._edge_keys:
            return False
        self._edge_keys.add(key)
        self.edges.append(PoseGraphEdge(source, target, edge_type, information.copy()))
        return True

    def has_edges(self) -> bool:
        return len(self.edges) > 0

    def latest_node(self) -> Optional[PoseGraphNode]:
        if not self.nodes:
            return None
        return self.nodes[-1]

    def find_loop_candidates(
        self,
        node: PoseGraphNode,
        radius: float,
        min_separation: int,
        max_candidates: int,
    ) -> List[PoseGraphNode]:
        candidates = []
        for other in self.nodes:
            if abs(node.node_id - other.node_id) < min_separation:
                continue
            dist = math.hypot(node.x - other.x, node.y - other.y)
            if dist <= radius:
                candidates.append((dist, other))
        candidates.sort(key=lambda item: item[0])
        return [other for _, other in candidates[:max_candidates]]

    def d_opt_score(self) -> float:
        n_nodes = len(self.nodes)
        if n_nodes < 2 or not self.edges:
            return 0.0

        laplacian = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        for edge in self.edges:
            weight = _information_weight(edge.information)
            if weight <= 0.0 or not np.isfinite(weight):
                continue
            i = edge.source
            j = edge.target
            laplacian[i, i] += weight
            laplacian[j, j] += weight
            laplacian[i, j] -= weight
            laplacian[j, i] -= weight

        if np.count_nonzero(laplacian) == 0:
            return 0.0

        anchored = laplacian[1:, 1:]
        sign, logdet = np.linalg.slogdet(anchored)
        if sign <= 0 or not np.isfinite(logdet):
            return 0.0
        n = float(n_nodes)
        return float((n ** (1.0 / n)) * math.exp(logdet / n))


class ApproximatePoseGraphTracker:
    def __init__(
        self,
        node_spacing: float,
        yaw_spacing: float,
        loop_closure_radius: float,
        loop_closure_min_separation: int,
        loop_closure_weight: float,
        max_loop_closures_per_node: int,
        odom_information: np.ndarray,
    ):
        self.graph = WeightedPoseGraph()
        self.node_spacing = node_spacing
        self.yaw_spacing = yaw_spacing
        self.loop_closure_radius = loop_closure_radius
        self.loop_closure_min_separation = loop_closure_min_separation
        self.loop_closure_weight = loop_closure_weight
        self.max_loop_closures_per_node = max_loop_closures_per_node
        self.odom_information = odom_information

    def update(self, pose: Tuple[float, float, float]):
        x, y, yaw = pose
        latest = self.graph.latest_node()
        if latest is None:
            self.graph.add_node(x, y, yaw)
            return

        distance = math.hypot(x - latest.x, y - latest.y)
        yaw_delta = abs(_normalize_angle(yaw - latest.yaw))
        if distance < self.node_spacing and yaw_delta < self.yaw_spacing:
            return

        new_node = self.graph.add_node(x, y, yaw)
        self.graph.add_edge(
            latest.node_id,
            new_node.node_id,
            0,
            self.odom_information,
        )

        loop_info = self.odom_information * self.loop_closure_weight
        for candidate in self.graph.find_loop_candidates(
            new_node,
            self.loop_closure_radius,
            self.loop_closure_min_separation,
            self.max_loop_closures_per_node,
        ):
            self.graph.add_edge(candidate.node_id, new_node.node_id, 1, loop_info)


class GraphBasedFrontierScorer:
    def __init__(
        self,
        info_radius: float,
        hallucinated_node_spacing: float,
        loop_closure_radius: float,
        loop_closure_min_separation: int,
        loop_closure_occupied_threshold: float,
        loop_closure_weight: float,
        max_loop_closures_per_node: int,
        path_cost_weight: float,
        odom_information: np.ndarray,
    ):
        self.info_radius = info_radius
        self.hallucinated_node_spacing = hallucinated_node_spacing
        self.loop_closure_radius = loop_closure_radius
        self.loop_closure_min_separation = loop_closure_min_separation
        self.loop_closure_occupied_threshold = loop_closure_occupied_threshold
        self.loop_closure_weight = loop_closure_weight
        self.max_loop_closures_per_node = max_loop_closures_per_node
        self.path_cost_weight = path_cost_weight
        self.odom_information = odom_information

    def score(
        self,
        base_graph: WeightedPoseGraph,
        grid_msg: OccupancyGrid,
        path: Sequence[Tuple[float, float]],
        grid: Optional[np.ndarray] = None,
    ) -> float:
        if len(path) < 2:
            return -float('inf')

        graph = base_graph.copy()
        if not graph.nodes:
            first = path[0]
            yaw = _path_yaw(path, 0)
            graph.add_node(first[0], first[1], yaw)

        previous = graph.latest_node()
        for x, y in self._sample_path(path):
            assert previous is not None
            yaw = math.atan2(y - previous.y, x - previous.x)
            unknown_ratio, occupied_ratio = cell_information(
                grid_msg,
                x,
                y,
                self.info_radius,
                grid,
            )
            node = graph.add_node(x, y, yaw)
            odom_info = self.odom_information * (1.0 + unknown_ratio)
            graph.add_edge(previous.node_id, node.node_id, 0, odom_info)

            if occupied_ratio >= self.loop_closure_occupied_threshold:
                loop_info = self.odom_information * (
                    self.loop_closure_weight * max(occupied_ratio, 1e-3)
                )
                for candidate in graph.find_loop_candidates(
                    node,
                    self.loop_closure_radius,
                    self.loop_closure_min_separation,
                    self.max_loop_closures_per_node,
                ):
                    graph.add_edge(candidate.node_id, node.node_id, 1, loop_info)
            previous = node

        path_length = _path_length(path)
        return graph.d_opt_score() - self.path_cost_weight * path_length

    def _sample_path(self, path: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        sampled = []
        prev_x, prev_y = path[0]
        accumulated = 0.0
        for x, y in path[1:]:
            step = math.hypot(x - prev_x, y - prev_y)
            accumulated += step
            if accumulated >= self.hallucinated_node_spacing:
                sampled.append((x, y))
                accumulated = 0.0
            prev_x, prev_y = x, y
        if sampled and sampled[-1] == path[-1]:
            return sampled
        sampled.append(path[-1])
        return sampled


def best_graph_candidate(candidates):
    """Return the highest-scoring reachable frontier candidate, if any."""
    return max(candidates, key=lambda item: item[0], default=None)


def make_information_matrix(cov_x: float, cov_y: float, cov_yaw: float) -> np.ndarray:
    cov = np.diag([
        max(cov_x, 1e-6),
        max(cov_y, 1e-6),
        max(cov_yaw, 1e-6),
    ])
    return np.linalg.inv(cov) / 2.0


def cell_information(
    grid_msg: OccupancyGrid,
    x: float,
    y: float,
    radius: float,
    data: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    width = grid_msg.info.width
    height = grid_msg.info.height
    resolution = grid_msg.info.resolution
    origin_x = grid_msg.info.origin.position.x
    origin_y = grid_msg.info.origin.position.y

    if width == 0 or height == 0 or resolution <= 0.0:
        return 0.0, 0.0

    if data is None:
        data = np.asarray(grid_msg.data, dtype=np.int8).reshape(height, width)
    elif data.shape != (height, width):
        raise ValueError('Cached occupancy grid shape does not match OccupancyGrid info.')
    center_i = int((y - origin_y) / resolution)
    center_j = int((x - origin_x) / resolution)
    radius_cells = max(1, int(math.ceil(radius / resolution)))

    min_i = max(0, center_i - radius_cells)
    max_i = min(height, center_i + radius_cells + 1)
    min_j = max(0, center_j - radius_cells)
    max_j = min(width, center_j + radius_cells + 1)
    rows = np.arange(min_i, max_i)
    cols = np.arange(min_j, max_j)
    wx = origin_x + (cols + 0.5) * resolution
    wy = origin_y + (rows + 0.5) * resolution
    circle_mask = np.hypot(wx[None, :] - x, wy[:, None] - y) <= radius
    local_grid = data[min_i:max_i, min_j:max_j]
    total = int(np.count_nonzero(circle_mask))
    unknown = int(np.count_nonzero(np.logical_and(circle_mask, local_grid == -1)))
    occupied = int(np.count_nonzero(np.logical_and(circle_mask, local_grid > 50)))

    if total == 0:
        return 0.0, 0.0
    return float(unknown) / total, float(occupied) / total


def graph_to_marker_array(graph: WeightedPoseGraph, frame_id: str, stamp) -> MarkerArray:
    marker_array = MarkerArray()
    delete = Marker()
    delete.action = Marker.DELETEALL
    marker_array.markers.append(delete)

    node_marker = Marker()
    node_marker.header.frame_id = frame_id
    node_marker.header.stamp = stamp
    node_marker.ns = 'approx_pose_graph_nodes'
    node_marker.id = 0
    node_marker.type = Marker.SPHERE_LIST
    node_marker.action = Marker.ADD
    node_marker.scale.x = 0.08
    node_marker.scale.y = 0.08
    node_marker.scale.z = 0.08
    node_marker.color.a = 0.8
    node_marker.color.r = 0.1
    node_marker.color.g = 0.4
    node_marker.color.b = 1.0

    for node in graph.nodes:
        point = Point()
        point.x = node.x
        point.y = node.y
        point.z = 0.05
        node_marker.points.append(point)
    marker_array.markers.append(node_marker)

    odom_edges = _edge_marker(frame_id, stamp, 'approx_pose_graph_odom', 1, 0.0, 0.9, 0.3)
    loop_edges = _edge_marker(frame_id, stamp, 'approx_pose_graph_loop', 2, 1.0, 0.6, 0.0)
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    for edge in graph.edges:
        marker = loop_edges if edge.edge_type == 1 else odom_edges
        source = nodes_by_id.get(edge.source)
        target = nodes_by_id.get(edge.target)
        if source is None or target is None:
            continue
        for node in (source, target):
            point = Point()
            point.x = node.x
            point.y = node.y
            point.z = 0.03
            marker.points.append(point)

    marker_array.markers.append(odom_edges)
    marker_array.markers.append(loop_edges)
    return marker_array


def _edge_marker(
    frame_id: str,
    stamp,
    ns: str,
    marker_id: int,
    r: float,
    g: float,
    b: float,
) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.025
    marker.color.a = 0.7
    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    return marker


def _information_weight(information: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(information)
    positive = eigvals[eigvals > 1e-8]
    if len(positive) == 0:
        return 0.0
    return float(math.exp(np.mean(np.log(positive))))


def _path_length(path: Sequence[Tuple[float, float]]) -> float:
    return sum(
        math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        for i in range(1, len(path))
    )


def _path_yaw(path: Sequence[Tuple[float, float]], index: int) -> float:
    if len(path) < 2:
        return 0.0
    next_index = min(index + 1, len(path) - 1)
    return math.atan2(path[next_index][1] - path[index][1], path[next_index][0] - path[index][0])


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
