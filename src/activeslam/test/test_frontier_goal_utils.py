import numpy as np
import pytest

from activeslam.frontier_goal_utils import (
    FailedGoalCooldown,
    GridGeometry,
    SafeGoalSearchConfig,
    is_goal_outside_reach_radius,
    navigation_timed_out,
    open_edge_outward_normal,
    potential_unknown_area,
    prepare_safe_goal_grid,
    segment_is_obstacle_free,
    select_safe_frontier_goal,
    unknown_frontier_outward_normal,
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

    assert goal.point == pytest.approx((1.15, 1.05))
    assert goal.seed == (10, 15)


def test_safe_goal_search_reuses_prepared_grid_without_changing_goal():
    grid = np.zeros((20, 20), dtype=np.int8)
    geometry = _geometry()
    config = _config()

    implicit_goal = select_safe_frontier_goal(
        grid,
        geometry,
        frontier_cells=[(10, 15)],
        robot_xy=(0.55, 1.05),
        config=config,
    )
    prepared_goal = select_safe_frontier_goal(
        grid,
        geometry,
        frontier_cells=[(10, 15)],
        robot_xy=(0.55, 1.05),
        config=config,
        prepared_grid=prepare_safe_goal_grid(grid, geometry, config),
    )

    assert prepared_goal == implicit_goal


def test_safe_goal_search_uses_region_mask_to_keep_goal_inside_local_cleanup():
    grid = np.zeros((20, 20), dtype=np.int8)
    allowed_goal_mask = np.zeros_like(grid, dtype=bool)
    allowed_goal_mask[10, 10] = True

    goal = select_safe_frontier_goal(
        grid,
        _geometry(),
        frontier_cells=[(10, 15)],
        robot_xy=(0.55, 1.05),
        config=_config(clearance=0.0),
        allowed_goal_mask=allowed_goal_mask,
    )

    assert goal.point == pytest.approx((1.05, 1.05))


def test_safe_goal_search_rejects_region_mask_shape_mismatch():
    with pytest.raises(ValueError, match='Allowed goal mask shape'):
        select_safe_frontier_goal(
            np.zeros((20, 20), dtype=np.int8),
            _geometry(),
            frontier_cells=[(10, 15)],
            robot_xy=(0.55, 1.05),
            config=_config(),
            allowed_goal_mask=np.ones((19, 20), dtype=bool),
        )


def test_prepared_grid_applies_obstacle_clearance_and_map_margin():
    grid = np.zeros((20, 20), dtype=np.int8)
    grid[10, 10] = 100

    prepared = prepare_safe_goal_grid(
        grid,
        _geometry(),
        _config(clearance=0.2, map_edge_clearance=0.3),
    )

    assert not prepared.valid_goal_mask[10, 12]
    assert prepared.valid_goal_mask[10, 13]
    assert not prepared.valid_goal_mask[2, 10]
    assert prepared.valid_goal_mask[3, 10]


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


def test_safe_goal_search_allows_short_advance_but_rejects_reach_radius():
    grid = np.zeros((20, 20), dtype=np.int8)

    assert select_safe_frontier_goal(
        grid,
        _geometry(),
        frontier_cells=[(10, 12)],
        robot_xy=(0.55, 1.05),
        config=_config(search_radius=0.0),
    ) is not None
    assert select_safe_frontier_goal(
        grid,
        _geometry(),
        frontier_cells=[(10, 8)],
        robot_xy=(0.55, 1.05),
        config=_config(search_radius=0.0),
    ) is None
    assert not is_goal_outside_reach_radius((0.0, 0.0), (0.15, 0.20), 0.25)
    assert is_goal_outside_reach_radius((0.0, 0.0), (0.26, 0.0), 0.25)


def test_safe_goal_search_rejects_standoff_segment_through_obstacle():
    grid = np.zeros((20, 20), dtype=np.int8)
    grid[:, 13] = 100

    assert select_safe_frontier_goal(
        grid,
        _geometry(),
        frontier_cells=[(10, 15)],
        robot_xy=(0.55, 1.05),
        config=_config(search_radius=0.0, clearance=0.0),
    ) is None


def test_standoff_segment_allows_unknown_but_rejects_outside_map():
    grid = np.zeros((20, 20), dtype=np.int8)
    grid[:, 5] = -1

    assert segment_is_obstacle_free(grid, _geometry(), (0.15, 1.05), (0.85, 1.05))
    assert not segment_is_obstacle_free(grid, _geometry(), (-0.05, 1.05), (0.85, 1.05))


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


def test_open_edge_normal_uses_local_outside_neighbor_votes():
    normal = open_edge_outward_normal(
        frontier_cells=[(row, 19) for row in range(20)],
        seed=(10, 19),
        geometry=_geometry(),
        radius=0.35,
    )

    assert normal == pytest.approx((1.0, 0.0))


def test_open_edge_normal_handles_corners_and_ignores_distant_edge_votes():
    normal = open_edge_outward_normal(
        frontier_cells=[(0, 0), (0, 1), (19, 19)],
        seed=(0, 0),
        geometry=_geometry(),
        radius=0.15,
    )

    assert normal == pytest.approx((-2 ** -0.5, -2 ** -0.5))


def test_open_edge_normal_returns_none_without_local_edge_cells():
    assert open_edge_outward_normal(
        frontier_cells=[(10, 10)],
        seed=(10, 10),
        geometry=_geometry(),
        radius=0.35,
    ) is None


def test_unknown_frontier_normal_uses_local_unknown_neighbor_votes():
    grid = np.zeros((20, 20), dtype=np.int8)
    grid[:, 11:] = -1

    normal = unknown_frontier_outward_normal(
        grid,
        frontier_cells=[(row, 10) for row in range(20)],
        seed=(10, 10),
        geometry=_geometry(),
        radius=0.35,
    )

    assert normal == pytest.approx((1.0, 0.0))


def test_unknown_frontier_normal_returns_none_without_unknown_neighbors():
    assert unknown_frontier_outward_normal(
        np.zeros((20, 20), dtype=np.int8),
        frontier_cells=[(10, 10)],
        seed=(10, 10),
        geometry=_geometry(),
        radius=0.35,
    ) is None


def test_potential_unknown_area_counts_internal_unknown_cells_in_square_meters():
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[1, 2] = -1

    area = potential_unknown_area(grid, _geometry(size=3), (1, 1), radius=0.1)

    assert area == pytest.approx(0.01)


def test_potential_unknown_area_only_counts_outside_map_when_requested():
    grid = np.zeros((3, 3), dtype=np.int8)
    geometry = _geometry(size=3)

    ordinary_area = potential_unknown_area(grid, geometry, (0, 0), radius=0.1)
    open_edge_area = potential_unknown_area(
        grid,
        geometry,
        (0, 0),
        radius=0.1,
        include_outside_map=True,
    )

    assert ordinary_area == 0.0
    assert open_edge_area == pytest.approx(0.02)


def test_potential_unknown_area_honors_radius():
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[1, 2] = -1

    area = potential_unknown_area(grid, _geometry(size=3), (1, 1), radius=0.09)

    assert area == 0.0


def test_potential_unknown_area_clips_circle_at_map_edge():
    grid = np.full((6, 6), -1, dtype=np.int8)

    area = potential_unknown_area(grid, _geometry(size=6), (3, 0), radius=0.35)

    assert area == pytest.approx(0.20)
