import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np


UNKNOWN_FRONTIER = 'unknown'
OPEN_EDGE_FRONTIER = 'open_edge'
GVD_SLAM_MODES = frozenset(('gvd_gbsae', 'gvd_hierarchical', 'gvd_guide'))


@dataclass(frozen=True)
class FrontierCandidate:
    cluster: Any
    safe_goal: Any
    information_gain: float
    utility: float
    range_gain: float = 0.0


def make_frontier_candidate(
    cluster,
    safe_goal,
    information_gain: float,
    robot_x: float,
    robot_y: float,
    on_cooldown: bool = False,
    range_gain: float = 0.0,
) -> Optional[FrontierCandidate]:
    """Create a shared-pool candidate after cheap local filtering."""
    if safe_goal is None or on_cooldown:
        return None
    distance = math.hypot(safe_goal.point[0] - robot_x, safe_goal.point[1] - robot_y)
    return FrontierCandidate(
        cluster=cluster,
        safe_goal=safe_goal,
        information_gain=information_gain,
        utility=information_gain / (distance + 0.1),
        range_gain=max(0.0, float(range_gain)),
    )


def ranked_frontier_candidates(
    candidates: Iterable[FrontierCandidate],
    limit: int,
) -> List[FrontierCandidate]:
    """Return the best candidates regardless of frontier source."""
    return sorted(candidates, key=lambda candidate: candidate.utility, reverse=True)[
        :max(0, limit)
    ]


def ranked_range_gain_candidates(
    candidates: Iterable[FrontierCandidate],
    robot_x: float,
    robot_y: float,
    limit: int,
) -> List[FrontierCandidate]:
    """Rank warmup frontiers by open-edge priority, then exploration range."""
    return sorted(
        candidates,
        key=lambda candidate: (
            0 if candidate.cluster.source == OPEN_EDGE_FRONTIER else 1,
            -candidate.range_gain,
            math.hypot(
                candidate.safe_goal.point[0] - robot_x,
                candidate.safe_goal.point[1] - robot_y,
            ),
            candidate.safe_goal.point,
        ),
    )[:max(0, limit)]


def frontier_range_gains(
    grid: np.ndarray,
    resolution: float,
    clusters: Sequence[Any],
) -> List[float]:
    """Return the unknown connected-component area adjacent to each frontier."""
    if grid.size == 0 or resolution <= 0.0:
        return [0.0 for _ in clusters]
    labels = np.full(grid.shape, -1, dtype=np.int32)
    areas: List[float] = []
    unknown = grid == -1
    height, width = grid.shape
    for row, col in np.argwhere(unknown):
        row = int(row)
        col = int(col)
        if labels[row, col] >= 0:
            continue
        component_id = len(areas)
        count = 0
        queue = deque([(row, col)])
        labels[row, col] = component_id
        while queue:
            cell = queue.popleft()
            count += 1
            for neighbor in _neighbors4(cell, grid.shape):
                if not unknown[neighbor] or labels[neighbor] >= 0:
                    continue
                labels[neighbor] = component_id
                queue.append(neighbor)
        areas.append(count * resolution * resolution)

    gains: List[float] = []
    for cluster in clusters:
        component_ids = set()
        for cell in getattr(cluster, 'cells', ()):
            for neighbor in _neighbors4(cell, grid.shape):
                component_id = labels[neighbor]
                if component_id >= 0:
                    component_ids.add(int(component_id))
        gain = sum(areas[component_id] for component_id in component_ids)
        if (
            gain <= 0.0
            and getattr(cluster, 'source', UNKNOWN_FRONTIER) == OPEN_EDGE_FRONTIER
        ):
            gain = len(getattr(cluster, 'cells', ())) * resolution * resolution
        gains.append(gain)
    return gains


def _neighbors4(cell, shape):
    row, col = cell
    height, width = shape
    for delta_row, delta_col in ((-1, 0), (0, -1), (0, 1), (1, 0)):
        neighbor = row + delta_row, col + delta_col
        if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width:
            yield neighbor


def ranked_local_cleanup_candidates(
    candidates: Iterable[FrontierCandidate],
    robot_x: float,
    robot_y: float,
    limit: int,
) -> List[FrontierCandidate]:
    """Rank micro-cleanup frontiers with the deliberately simple size/distance rule."""
    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.cluster.size
            / (
                math.hypot(
                    candidate.safe_goal.point[0] - robot_x,
                    candidate.safe_goal.point[1] - robot_y,
                )
                + 0.1
            ),
            -candidate.cluster.size,
            candidate.safe_goal.point,
        ),
    )[:max(0, limit)]


def frontier_probes_enabled_for_mode(
    slam_mode: str,
    *,
    frontier_modes_enabled: bool,
    gvd_modes_enabled: bool,
    hierarchical_local_cleanup: bool = False,
    hierarchical_local_cleanup_enabled: bool = True,
    gvd_guide_warmup: bool = False,
) -> bool:
    """Apply distinct probe defaults to frontier-driven and GVD-driven modes."""
    if slam_mode == 'gvd_guide' and gvd_guide_warmup:
        return frontier_modes_enabled
    if (
        slam_mode == 'gvd_hierarchical'
        and hierarchical_local_cleanup
        and hierarchical_local_cleanup_enabled
    ):
        return True
    return gvd_modes_enabled if slam_mode in GVD_SLAM_MODES else frontier_modes_enabled
