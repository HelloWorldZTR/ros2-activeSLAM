import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional


UNKNOWN_FRONTIER = 'unknown'
OPEN_EDGE_FRONTIER = 'open_edge'
GVD_SLAM_MODES = frozenset(('gvd_gbsae', 'gvd_hierarchical'))


@dataclass(frozen=True)
class FrontierCandidate:
    cluster: Any
    safe_goal: Any
    information_gain: float
    utility: float


def make_frontier_candidate(
    cluster,
    safe_goal,
    information_gain: float,
    robot_x: float,
    robot_y: float,
    on_cooldown: bool = False,
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
    )


def ranked_frontier_candidates(
    candidates: Iterable[FrontierCandidate],
    limit: int,
) -> List[FrontierCandidate]:
    """Return the best candidates regardless of frontier source."""
    return sorted(candidates, key=lambda candidate: candidate.utility, reverse=True)[
        :max(0, limit)
    ]


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
) -> bool:
    """Apply distinct probe defaults to frontier-driven and GVD-driven modes."""
    return gvd_modes_enabled if slam_mode in GVD_SLAM_MODES else frontier_modes_enabled
