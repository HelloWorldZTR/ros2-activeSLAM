import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional


UNKNOWN_FRONTIER = 'unknown'
OPEN_EDGE_FRONTIER = 'open_edge'


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
    min_information_gain: float,
    on_cooldown: bool = False,
) -> Optional[FrontierCandidate]:
    """Create a shared-pool candidate after cheap local filtering."""
    if safe_goal is None or on_cooldown or information_gain < min_information_gain:
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
