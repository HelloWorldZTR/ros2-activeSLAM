"""Online bootstrap and skeleton-topology helpers for GBSAE exploration."""

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

from .frontier_goal_utils import GridGeometry


Point = Tuple[float, float]
GridCell = Tuple[int, int]


@dataclass(frozen=True)
class WorldBounds:
    """A coarse rectangular prior without obstacle or room structure."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @property
    def area(self) -> float:
        return max(0.0, self.max_x - self.min_x) * max(0.0, self.max_y - self.min_y)

    def contains(self, point: Point) -> bool:
        return (
            self.min_x <= point[0] <= self.max_x
            and self.min_y <= point[1] <= self.max_y
        )


@dataclass(frozen=True)
class BootstrapWeights:
    normal_unknown_depth: float = 5.0
    directional_remaining_unknown: float = 1.5
    path_known_ratio_penalty: float = 100.0


@dataclass(frozen=True)
class BootstrapScore:
    """Normalized bootstrap features and their weighted objective."""

    normal_unknown_depth: float
    directional_remaining_unknown: float
    path_known_ratio: float
    total: float


@dataclass(frozen=True)
class BranchHypothesis:
    """A virtual leaf projected into unknown space from a frontier."""

    branch_id: int
    frontier_point: Point
    point: Point
    normal: Point
    score: float
    failures: int = 0
    blocked: bool = False
    explored: bool = False


@dataclass(frozen=True)
class OnlineTopology:
    """A skeleton transit graph plus the targets which still need visits."""

    graph: nx.Graph
    target_vertices: Set[int]
    inherited_explored: Set[int]


def resolve_world_bounds_path() -> Path:
    """Resolve the installed per-world online GBSAE bounds asset."""
    from ament_index_python.packages import get_package_share_directory

    return (
        Path(get_package_share_directory('activeslam'))
        / 'config'
        / 'online_gbsae_worlds.yaml'
    )


def load_world_bounds(path: Path, world_name: str) -> WorldBounds:
    """Load one coarse world rectangle from YAML."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError('online_gbsae requires python3-yaml or PyYAML.') from exc

    payload = yaml.safe_load(Path(path).read_text())
    values = None if not isinstance(payload, dict) else payload.get(world_name)
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f'Missing online GBSAE bounds for world={world_name}.')
    bounds = WorldBounds(*(float(value) for value in values))
    if bounds.area <= 0.0:
        raise ValueError(f'Invalid online GBSAE bounds for world={world_name}: {values!r}.')
    return bounds


def known_area_ratio(grid: np.ndarray, geometry: GridGeometry, bounds: WorldBounds) -> float:
    """Return known OccupancyGrid area divided by the coarse prior area."""
    if geometry.resolution <= 0.0 or bounds.area <= 0.0:
        return 0.0
    rows, cols = _cells_inside_bounds(geometry, bounds)
    if rows.size == 0 or cols.size == 0:
        return 0.0
    known = int(np.count_nonzero(grid[np.ix_(rows, cols)] != -1))
    return min(1.0, known * geometry.resolution * geometry.resolution / bounds.area)


def normal_unknown_depth(
    grid: np.ndarray,
    geometry: GridGeometry,
    start: Point,
    normal: Optional[Point],
    max_depth: float,
) -> float:
    """Measure contiguous unknown depth beyond a frontier along its normal."""
    if normal is None or max_depth <= 0.0 or geometry.resolution <= 0.0:
        return 0.0
    step = max(geometry.resolution, 0.01)
    depth = 0.0
    distance = step
    while distance <= max_depth + 1e-9:
        point = start[0] + normal[0] * distance, start[1] + normal[1] * distance
        cell = _world_to_grid(point, geometry)
        if cell is None:
            depth = distance
        elif grid[cell] == -1:
            depth = distance
        else:
            break
        distance += step
    return min(1.0, depth / max_depth)


def directional_remaining_unknown(
    grid: np.ndarray,
    geometry: GridGeometry,
    bounds: WorldBounds,
    origin: Point,
    target: Point,
    half_angle: float,
) -> float:
    """Return the unknown fraction inside the prior rectangle and goal-facing sector."""
    if geometry.resolution <= 0.0:
        return 0.0
    xs = np.arange(
        bounds.min_x + geometry.resolution * 0.5,
        bounds.max_x,
        geometry.resolution,
    )
    ys = np.arange(
        bounds.min_y + geometry.resolution * 0.5,
        bounds.max_y,
        geometry.resolution,
    )
    if xs.size == 0 or ys.size == 0:
        return 0.0
    dx = xs[None, :] - origin[0]
    dy = ys[:, None] - origin[1]
    target_angle = math.atan2(target[1] - origin[1], target[0] - origin[0])
    angle_delta = np.arctan2(np.sin(np.arctan2(dy, dx) - target_angle),
                             np.cos(np.arctan2(dy, dx) - target_angle))
    sector = np.abs(angle_delta) <= half_angle
    if not np.any(sector):
        return 0.0
    cols = np.floor((xs - geometry.origin_x) / geometry.resolution).astype(int)
    rows = np.floor((ys - geometry.origin_y) / geometry.resolution).astype(int)
    inside_rows = (rows >= 0) & (rows < geometry.height)
    inside_cols = (cols >= 0) & (cols < geometry.width)
    inside = inside_rows[:, None] & inside_cols[None, :]
    unknown = np.ones_like(sector, dtype=bool)
    if np.any(inside_rows) and np.any(inside_cols):
        row_indices = np.flatnonzero(inside_rows)
        col_indices = np.flatnonzero(inside_cols)
        local = grid[np.ix_(rows[row_indices], cols[col_indices])] == -1
        unknown[np.ix_(row_indices, col_indices)] = local
    unknown |= ~inside
    return float(np.count_nonzero(np.logical_and(sector, unknown))) / float(
        np.count_nonzero(sector)
    )


def path_known_ratio(
    path: Sequence[Point],
    grid: np.ndarray,
    geometry: GridGeometry,
) -> float:
    """Return known-space travel divided by the full Nav2 path length."""
    if len(path) < 2 or geometry.resolution <= 0.0:
        return 0.0
    known_distance = 0.0
    path_distance = 0.0
    sample_spacing = max(geometry.resolution * 0.5, 1e-6)
    for start, end in zip(path, path[1:]):
        segment_length = math.dist(start, end)
        if segment_length <= 1e-9:
            continue
        path_distance += segment_length
        samples = max(1, int(math.ceil(segment_length / sample_spacing)))
        step = segment_length / samples
        for index in range(samples):
            along_segment = (index + 0.5) * step
            ratio = along_segment / segment_length
            point = (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
            cell = _world_to_grid(point, geometry)
            if cell is not None and grid[cell] != -1:
                known_distance += step
    return 0.0 if path_distance <= 1e-9 else known_distance / path_distance


def bootstrap_score(
    *,
    unknown_depth: float,
    directional_unknown: float,
    path_known_ratio: float,
    weights: BootstrapWeights,
) -> BootstrapScore:
    """Combine normalized fast-expansion features into one objective."""
    values = (
        min(1.0, max(0.0, unknown_depth)),
        min(1.0, max(0.0, directional_unknown)),
        min(1.0, max(0.0, path_known_ratio)),
    )
    total = (
        weights.normal_unknown_depth * values[0]
        + weights.directional_remaining_unknown * values[1]
        - weights.path_known_ratio_penalty * values[2]
    )
    return BootstrapScore(*values, total)


def record_branch_hypotheses(
    existing: Sequence[BranchHypothesis],
    alternatives: Sequence[Tuple[Point, Point, float]],
    *,
    best_score: float,
    score_ratio: float,
    min_angle: float,
    merge_radius: float,
    projection_distance: float,
    bounds: WorldBounds,
) -> List[BranchHypothesis]:
    """Merge high-quality directionally distinct alternatives as virtual leaves."""
    branches = list(existing)
    next_id = 0 if not branches else max(branch.branch_id for branch in branches) + 1
    for frontier_point, normal, score in sorted(alternatives, key=lambda item: -item[2]):
        if score + 1e-9 < best_score * score_ratio:
            continue
        magnitude = math.hypot(normal[0], normal[1])
        if magnitude <= 1e-9:
            continue
        unit = normal[0] / magnitude, normal[1] / magnitude
        projected = (
            frontier_point[0] + unit[0] * projection_distance,
            frontier_point[1] + unit[1] * projection_distance,
        )
        if not bounds.contains(projected):
            continue
        if any(
            math.dist(projected, branch.point) < merge_radius
            or _angle_between(unit, branch.normal) < min_angle
            for branch in branches
            if not branch.blocked and not branch.explored
        ):
            continue
        branches.append(BranchHypothesis(next_id, frontier_point, projected, unit, score))
        next_id += 1
    return branches


def update_branch_failure(
    branches: Sequence[BranchHypothesis],
    branch_id: int,
    failure_limit: int,
) -> List[BranchHypothesis]:
    """Increment a branch failure count and block it after bounded retries."""
    updated = []
    for branch in branches:
        if branch.branch_id != branch_id:
            updated.append(branch)
            continue
        failures = branch.failures + 1
        updated.append(replace(branch, failures=failures, blocked=failures >= failure_limit))
    return updated


def mark_branch_explored(
    branches: Sequence[BranchHypothesis],
    branch_id: int,
) -> List[BranchHypothesis]:
    return [
        replace(branch, explored=True) if branch.branch_id == branch_id else branch
        for branch in branches
    ]


def branch_path_has_no_known_obstacle(
    grid: np.ndarray,
    geometry: GridGeometry,
    branch: BranchHypothesis,
) -> bool:
    """Allow unknown cells but reject a virtual branch ray crossing a known obstacle."""
    distance = math.dist(branch.frontier_point, branch.point)
    count = max(1, int(math.ceil(distance / max(geometry.resolution * 0.5, 1e-6))))
    for index in range(count + 1):
        ratio = index / count
        point = (
            branch.frontier_point[0] + (branch.point[0] - branch.frontier_point[0]) * ratio,
            branch.frontier_point[1] + (branch.point[1] - branch.frontier_point[1]) * ratio,
        )
        cell = _world_to_grid(point, geometry)
        if cell is not None and grid[cell] > 50:
            return False
    return True


def build_online_topology(
    grid: np.ndarray,
    geometry: GridGeometry,
    robot_xy: Point,
    branches: Sequence[BranchHypothesis],
    old_graph: Optional[nx.Graph] = None,
    old_explored: Optional[Iterable[int]] = None,
    *,
    clearance: float = 0.20,
    spur_prune_length: float = 0.50,
    support_vertex_spacing: float = 2.0,
    migration_radius: float = 1.0,
    migration_overlap: float = 0.50,
) -> OnlineTopology:
    """Build a sparse skeleton graph from the currently known free component."""
    valid_free = _clear_known_free(grid, geometry, clearance)
    skeleton = _thin_free_bbox(valid_free)
    skeleton = prune_short_spurs(skeleton, geometry.resolution, spur_prune_length)
    graph = skeleton_to_graph(skeleton, geometry, support_vertex_spacing)
    graph = _retain_robot_component(graph, robot_xy)
    inherited = migrate_explored_vertices(
        graph,
        old_graph,
        set() if old_explored is None else set(old_explored),
        migration_radius,
        migration_overlap,
    )
    _attach_branches(graph, branches)
    targets = {
        node_id
        for node_id, attributes in graph.nodes(data=True)
        if node_id not in inherited
        and not attributes.get('blocked', False)
        and not attributes.get('explored', False)
    }
    return OnlineTopology(graph, targets, inherited)


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


def _thin_free_bbox(mask: np.ndarray) -> np.ndarray:
    """Thin only the known-free bounding box, then restore full-map indices."""
    cells = np.argwhere(mask)
    if cells.size == 0:
        return np.zeros_like(mask, dtype=bool)
    min_row, min_col = np.min(cells, axis=0)
    max_row, max_col = np.max(cells, axis=0) + 1
    skeleton = np.zeros_like(mask, dtype=bool)
    skeleton[min_row:max_row, min_col:max_col] = zhang_suen_thinning(
        mask[min_row:max_row, min_col:max_col]
    )
    return skeleton


def prune_short_spurs(skeleton: np.ndarray, resolution: float, min_length: float) -> np.ndarray:
    """Delete short endpoint chains while preserving junctions and longer corridors."""
    result = np.asarray(skeleton, dtype=bool).copy()
    if min_length <= 0.0 or resolution <= 0.0:
        return result
    while True:
        adjacency = _skeleton_adjacency(result)
        endpoints = sorted(cell for cell, neighbors in adjacency.items() if len(neighbors) == 1)
        removed = False
        for endpoint in endpoints:
            if not result[endpoint]:
                continue
            chain = [endpoint]
            previous = None
            current = endpoint
            while True:
                next_cells = [cell for cell in adjacency.get(current, []) if cell != previous]
                if len(next_cells) != 1:
                    break
                previous, current = current, next_cells[0]
                if len(adjacency.get(current, [])) != 2:
                    break
                chain.append(current)
            length = max(0, len(chain)) * resolution
            if len(adjacency.get(current, [])) >= 3 and length < min_length:
                for cell in chain:
                    result[cell] = False
                removed = True
                break
        if not removed:
            return result


def skeleton_to_graph(
    skeleton: np.ndarray,
    geometry: GridGeometry,
    support_vertex_spacing: float,
) -> nx.Graph:
    """Compress skeleton chains into metric graph edges with sparse support vertices."""
    adjacency = _skeleton_adjacency(skeleton)
    graph = nx.Graph()
    if not adjacency:
        return graph
    key_cells = {cell for cell, neighbors in adjacency.items() if len(neighbors) != 2}
    if not key_cells:
        key_cells.add(min(adjacency))
    node_for_cell = {}

    def ensure_node(cell: GridCell) -> int:
        if cell not in node_for_cell:
            node_id = len(node_for_cell)
            node_for_cell[cell] = node_id
            x, y = _grid_to_world(cell, geometry)
            graph.add_node(node_id, x=x, y=y, kind='skeleton')
        return node_for_cell[cell]

    for cell in sorted(key_cells):
        ensure_node(cell)
    visited_edges = set()
    max_spacing = max(geometry.resolution, support_vertex_spacing)
    for start in sorted(key_cells):
        for neighbor in adjacency[start]:
            edge = _cell_edge(start, neighbor)
            if edge in visited_edges:
                continue
            chain = [start]
            previous, current = start, neighbor
            visited_edges.add(edge)
            while current not in key_cells:
                chain.append(current)
                next_cells = [cell for cell in adjacency[current] if cell != previous]
                if not next_cells:
                    break
                next_cell = next_cells[0]
                visited_edges.add(_cell_edge(current, next_cell))
                previous, current = current, next_cell
            chain.append(current)
            split_indices = [0]
            accumulated = 0.0
            for index in range(1, len(chain)):
                step = (
                    math.hypot(
                        chain[index][0] - chain[index - 1][0],
                        chain[index][1] - chain[index - 1][1],
                    )
                    * geometry.resolution
                )
                if accumulated + step > max_spacing and index - 1 > split_indices[-1]:
                    split_indices.append(index - 1)
                    accumulated = 0.0
                accumulated += step
            if split_indices[-1] != len(chain) - 1:
                split_indices.append(len(chain) - 1)
            for source_index, target_index in zip(split_indices, split_indices[1:]):
                source_cell = chain[source_index]
                target_cell = chain[target_index]
                source = ensure_node(source_cell)
                target = ensure_node(target_cell)
                if source == target:
                    continue
                distance = sum(
                    math.hypot(
                        chain[index + 1][0] - chain[index][0],
                        chain[index + 1][1] - chain[index][1],
                    )
                    * geometry.resolution
                    for index in range(source_index, target_index)
                )
                graph.add_edge(
                    source,
                    target,
                    weight=distance,
                    information_weight=1.0 / max(distance, 1e-6),
                    virtual=False,
                )
    return graph


def migrate_explored_vertices(
    new_graph: nx.Graph,
    old_graph: Optional[nx.Graph],
    old_explored: Set[int],
    radius: float,
    overlap_threshold: float,
) -> Set[int]:
    """Carry explored state across rebuilds using sampled coverage-disk overlap."""
    if old_graph is None or not old_explored or radius <= 0.0:
        return set()
    old_points = np.asarray(
        [vertex_point(old_graph, node_id) for node_id in old_explored if node_id in old_graph],
        dtype=np.float64,
    )
    if old_points.size == 0:
        return set()
    offsets = _disk_offsets(radius, samples=9)
    inherited = set()
    for node_id in new_graph.nodes:
        center = np.asarray(vertex_point(new_graph, node_id))
        samples = center[None, :] + offsets
        distances = np.hypot(
            samples[:, None, 0] - old_points[None, :, 0],
            samples[:, None, 1] - old_points[None, :, 1],
        )
        overlap = float(np.count_nonzero(np.any(distances <= radius, axis=1))) / len(samples)
        if overlap >= overlap_threshold:
            inherited.add(node_id)
    return inherited


def vertex_point(graph: nx.Graph, node_id: int) -> Point:
    return float(graph.nodes[node_id]['x']), float(graph.nodes[node_id]['y'])


def _clear_known_free(grid: np.ndarray, geometry: GridGeometry, clearance: float) -> np.ndarray:
    free = grid == 0
    cells = max(0, int(math.ceil(clearance / geometry.resolution)))
    if cells == 0:
        return free
    blocked = grid > 50
    padded = np.pad(blocked.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    rows = np.arange(geometry.height)
    cols = np.arange(geometry.width)
    row0 = np.maximum(0, rows - cells)
    row1 = np.minimum(geometry.height, rows + cells + 1)
    col0 = np.maximum(0, cols - cells)
    col1 = np.minimum(geometry.width, cols + cells + 1)
    counts = (
        padded[row1[:, None], col1[None, :]]
        - padded[row0[:, None], col1[None, :]]
        - padded[row1[:, None], col0[None, :]]
        + padded[row0[:, None], col0[None, :]]
    )
    return free & (counts == 0)


def _retain_robot_component(graph: nx.Graph, robot_xy: Point) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph
    nearest = min(
        graph.nodes,
        key=lambda node_id: math.dist(vertex_point(graph, node_id), robot_xy),
    )
    return graph.subgraph(nx.node_connected_component(graph, nearest)).copy()


def _attach_branches(graph: nx.Graph, branches: Sequence[BranchHypothesis]):
    if graph.number_of_nodes() == 0:
        return
    for branch in branches:
        if branch.explored:
            continue
        anchor = min(
            graph.nodes,
            key=lambda node_id: math.dist(vertex_point(graph, node_id), branch.frontier_point),
        )
        node_id = max(graph.nodes, default=-1) + 1
        graph.add_node(
            node_id,
            x=branch.point[0],
            y=branch.point[1],
            kind='branch',
            branch_id=branch.branch_id,
            blocked=branch.blocked,
            failures=branch.failures,
        )
        distance = math.dist(vertex_point(graph, anchor), branch.point)
        graph.add_edge(
            anchor,
            node_id,
            weight=distance,
            information_weight=1.0 / max(distance, 1e-6),
            virtual=True,
        )


def _skeleton_adjacency(skeleton: np.ndarray):
    cells = {tuple(cell) for cell in np.argwhere(skeleton)}
    adjacency = {}
    for i, j in cells:
        adjacency[(i, j)] = sorted(
            (i + di, j + dj)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
            if (di or dj) and (i + di, j + dj) in cells
        )
    return adjacency


def _cells_inside_bounds(geometry: GridGeometry, bounds: WorldBounds):
    rows = np.arange(geometry.height)
    cols = np.arange(geometry.width)
    xs = geometry.origin_x + (cols + 0.5) * geometry.resolution
    ys = geometry.origin_y + (rows + 0.5) * geometry.resolution
    return rows[(ys >= bounds.min_y) & (ys <= bounds.max_y)], cols[
        (xs >= bounds.min_x) & (xs <= bounds.max_x)
    ]


def _world_to_grid(point: Point, geometry: GridGeometry) -> Optional[GridCell]:
    row = int(math.floor((point[1] - geometry.origin_y) / geometry.resolution))
    col = int(math.floor((point[0] - geometry.origin_x) / geometry.resolution))
    if row < 0 or col < 0 or row >= geometry.height or col >= geometry.width:
        return None
    return row, col


def _grid_to_world(cell: GridCell, geometry: GridGeometry) -> Point:
    return (
        geometry.origin_x + (cell[1] + 0.5) * geometry.resolution,
        geometry.origin_y + (cell[0] + 0.5) * geometry.resolution,
    )


def _cell_edge(source: GridCell, target: GridCell):
    return tuple(sorted((source, target)))


def _angle_between(source: Point, target: Point) -> float:
    dot = source[0] * target[0] + source[1] * target[1]
    cross = source[0] * target[1] - source[1] * target[0]
    return abs(math.atan2(cross, dot))


def _disk_offsets(radius: float, samples: int):
    values = np.linspace(-radius, radius, samples)
    x, y = np.meshgrid(values, values)
    points = np.column_stack((x.ravel(), y.ravel()))
    return points[np.hypot(points[:, 0], points[:, 1]) <= radius + 1e-9]
