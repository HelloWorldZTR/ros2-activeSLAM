import numpy as np
import pytest

from activeslam.frontier_goal_utils import (
    FailedGoalCooldown,
    GridGeometry,
    SafeGoalSearchConfig,
    is_goal_outside_reach_radius,
    navigation_timed_out,
    select_safe_frontier_goal,
)


def _geometry(size=20):
    return GridGeometry(
        origin_x=0.0,
        origin_y=0.0,
        resolution=0.1,
        width=size,
        height=size,
    )


def _config(**overrides):
    values = {
        'search_radius': 0.3,
        'clearance': 0.1,
        'standoff': 0.4,
        'map_edge_clearance': 0.0,
        'min_advance': 0.35,
        'reach_radius': 0.25,
        'point_sample_limit': 40,
    }
    values.update(overrides)
    return SafeGoalSearchConfig(**values)


def test_safe_goal_search_selects_standoff_cell():
    grid = np.zeros((20, 20), dtype=np.int8)

    goal = select_safe_frontier_goal(
        grid,
        _geometry(),
        frontier_cells=[(10, 15)],
        robot_xy=(0.55, 1.05),
        config=_config(),
    )

    assert goal == pytest.approx((1.15, 1.05))


def test_safe_goal_search_rejects_obstacles_clearance_and_map_edge():
    grid = np.zeros((20, 20), dtype=np.int8)
    grid[9:12, 10:13] = 100

    assert select_safe_frontier_goal(
        grid,
        _geometry(),
        frontier_cells=[(10, 15)],
        robot_xy=(0.55, 1.05),
        config=_config(search_radius=0.0, clearance=0.2),
    ) is None
    assert select_safe_frontier_goal(
        np.zeros((20, 20), dtype=np.int8),
        _geometry(),
        frontier_cells=[(1, 8)],
        robot_xy=(0.15, 0.15),
        config=_config(search_radius=0.0, map_edge_clearance=0.3),
    ) is None


def test_safe_goal_search_rejects_short_advance_and_reach_radius():
    grid = np.zeros((20, 20), dtype=np.int8)

    assert select_safe_frontier_goal(
        grid,
        _geometry(),
        frontier_cells=[(10, 8)],
        robot_xy=(0.55, 1.05),
        config=_config(search_radius=0.0, min_advance=0.35),
    ) is None
    assert not is_goal_outside_reach_radius((0.0, 0.0), (0.15, 0.20), 0.25)
    assert is_goal_outside_reach_radius((0.0, 0.0), (0.26, 0.0), 0.25)


def test_failed_goal_cooldown_expires_and_filters_nearby_goals():
    cooldown = FailedGoalCooldown(duration=20.0, radius=0.6)
    cooldown.mark((1.0, 2.0), now=10.0)

    assert cooldown.contains((1.5, 2.0), now=20.0)
    assert not cooldown.contains((1.6, 2.0), now=20.0)
    assert not cooldown.contains((1.0, 2.0), now=30.0)
    assert cooldown.goals == ()


def test_navigation_timeout_is_disabled_or_triggered_by_deadline():
    assert not navigation_timed_out(None, timeout=30.0, now=100.0)
    assert not navigation_timed_out(10.0, timeout=0.0, now=100.0)
    assert not navigation_timed_out(10.0, timeout=30.0, now=39.9)
    assert navigation_timed_out(10.0, timeout=30.0, now=40.0)
