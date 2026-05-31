import math
import random

import networkx as nx
import numpy as np

from activeslam.frontier_goal_utils import GridGeometry
from activeslam.gvd_exploration import (
    GVDTopology,
    GVDWeights,
    HierarchicalGVDTracker,
    TopologyConnectionCache,
    TrajectorySweepTracker,
    WorldBounds,
    astar_path,
    bidirectional_astar_path,
    boundary_unknown_score,
    bounds_geometry,
    build_obstacle_gvd_topology,
    build_obstacle_traversability,
    distance_to_mask,
    grid_to_world,
    cluster_touches_mask,
    local_free_flood_mask,
    local_unknown_area,
    path_crosses_new_obstacle,
    path_crosses_obstacle,
    path_suffix_from_nearest,
    path_overlap_ratio,
    progress_watchdog_expired,
    rank_gvd_goals,
    rectangle_mask_outline,
    repair_topology_connectivity,
    robot_component_graph,
    sample_random_recovery_motion,
    update_translation_progress,
)


def _geometry(width=10, height=10, resolution=1.0):
    return GridGeometry(0.0, 0.0, resolution, width, height)


def test_obstacle_only_topology_treats_unknown_as_flat_ground_and_inflates_walls():
    grid = np.full((10, 10), -1, dtype=np.int8)
    grid[5, 5] = 100

    topology = build_obstacle_gvd_topology(
        grid,
        _geometry(),
        WorldBounds(0.0, 10.0, 0.0, 10.0),
        resolution=1.0,
        clearance=1.0,
        boundary_margin=0.0,
        support_vertex_spacing=2.0,
    )

    assert topology.traversable[0, 0]
    assert not topology.traversable[5, 5]
    assert not topology.traversable[4, 5]
    assert not topology.traversable[6, 6]


def test_robot_radius_clearance_rejects_pixel_gap_but_keeps_wide_doorway():
    geometry = _geometry(width=11, height=11, resolution=0.1)
    bounds = WorldBounds(0.0, 1.1, 0.0, 1.1)
    narrow_grid = np.zeros((11, 11), dtype=np.int8)
    narrow_grid[:, 5] = 100
    narrow_grid[5, 5] = 0

    _, pixel_traversable = build_obstacle_traversability(
        narrow_grid,
        geometry,
        bounds,
        resolution=0.1,
        clearance=0.0,
        boundary_margin=0.0,
    )
    _, radius_traversable = build_obstacle_traversability(
        narrow_grid,
        geometry,
        bounds,
        resolution=0.1,
        clearance=0.18,
        boundary_margin=0.0,
    )

    assert bidirectional_astar_path(pixel_traversable, (5, 1), (5, 9))
    assert not bidirectional_astar_path(radius_traversable, (5, 1), (5, 9))

    wide_grid = np.zeros((11, 11), dtype=np.int8)
    wide_grid[:, 5] = 100
    wide_grid[3:8, 5] = 0
    _, wide_traversable = build_obstacle_traversability(
        wide_grid,
        geometry,
        bounds,
        resolution=0.1,
        clearance=0.18,
        boundary_margin=0.0,
    )

    assert bidirectional_astar_path(wide_traversable, (5, 1), (5, 9))


def test_astar_routes_around_known_obstacle_and_prefers_reachable_goal():
    traversable = np.ones((7, 7), dtype=bool)
    traversable[1:6, 3] = False
    skeleton = np.zeros_like(traversable)
    skeleton[3, :] = True

    path = astar_path(traversable, skeleton, (3, 1), (3, 5), 1.0, 2.5)

    assert path[0] == (3, 1)
    assert path[-1] == (3, 5)
    assert all(traversable[cell] for cell in path)
    assert any(row in (0, 6) for row, _ in path)


def test_bidirectional_astar_routes_around_wall_without_cutting_corners():
    traversable = np.ones((7, 7), dtype=bool)
    traversable[1:6, 3] = False

    path = bidirectional_astar_path(traversable, (3, 1), (3, 5))

    assert path[0] == (3, 1)
    assert path[-1] == (3, 5)
    assert all(traversable[cell] for cell in path)
    assert any(row in (0, 6) for row, _ in path)


def test_topology_connection_cache_prefers_gvd_and_reuses_reverse_query():
    traversable = np.ones((3, 5), dtype=bool)
    skeleton = np.zeros_like(traversable)
    skeleton[1, :] = True
    cache = TopologyConnectionCache()

    first = cache.connection(
        traversable,
        skeleton,
        (1, 0),
        (1, 4),
        resolution=1.0,
        revision=1,
    )
    reverse = cache.connection(
        traversable,
        skeleton,
        (1, 4),
        (1, 0),
        resolution=1.0,
        revision=1,
    )

    assert first is not None and first.mode == 'gvd'
    assert reverse is not None and reverse.path == tuple(reversed(first.path))
    assert cache.misses == 1
    assert cache.hits == 1


def test_topology_connection_cache_invalidates_blocked_astar_bridge():
    traversable = np.ones((3, 5), dtype=bool)
    skeleton = np.zeros_like(traversable)
    skeleton[1, (0, 4)] = True
    cache = TopologyConnectionCache()

    first = cache.connection(
        traversable,
        skeleton,
        (1, 0),
        (1, 4),
        resolution=1.0,
        revision=1,
    )
    blocked = traversable.copy()
    blocked[:, 2] = False
    second = cache.connection(
        blocked,
        skeleton,
        (1, 0),
        (1, 4),
        resolution=1.0,
        revision=2,
    )

    assert first is not None and first.mode == 'astar'
    assert second is None
    assert cache.invalidations == 1


def test_topology_connection_cache_rechecks_astar_diagonal_corners():
    traversable = np.ones((2, 2), dtype=bool)
    skeleton = np.zeros_like(traversable)
    cache = TopologyConnectionCache()

    first = cache.connection(
        traversable,
        skeleton,
        (0, 0),
        (1, 1),
        resolution=1.0,
        revision=1,
    )
    blocked_corner = traversable.copy()
    blocked_corner[0, 1] = False
    blocked_corner[1, 0] = False
    second = cache.connection(
        blocked_corner,
        skeleton,
        (0, 0),
        (1, 1),
        resolution=1.0,
        revision=2,
    )

    assert first is not None
    assert second is None
    assert cache.invalidations == 1


def test_topology_repair_switches_to_astar_for_reachable_skeleton_break():
    geometry = _geometry(width=7, height=3)
    traversable = np.ones((3, 7), dtype=bool)
    skeleton = np.zeros_like(traversable)
    skeleton[1, :3] = True
    skeleton[1, 4:] = True
    graph = nx.Graph()
    graph.add_node(0, x=0.5, y=1.5)
    graph.add_node(1, x=6.5, y=1.5)

    repaired, stats = repair_topology_connectivity(
        graph,
        skeleton,
        traversable,
        geometry,
        TopologyConnectionCache(),
        neighbor_limit=10,
        map_revision=1,
    )

    assert nx.is_connected(repaired)
    assert repaired.edges[0, 1]['connection_mode'] == 'astar'
    assert stats.astar_edges == 1
    assert stats.unresolved_components == 1


def test_topology_repair_keeps_truly_isolated_components_separate():
    geometry = _geometry(width=7, height=3)
    traversable = np.ones((3, 7), dtype=bool)
    traversable[:, 3] = False
    skeleton = np.zeros_like(traversable)
    skeleton[1, :3] = True
    skeleton[1, 4:] = True
    graph = nx.Graph()
    graph.add_node(0, x=0.5, y=1.5)
    graph.add_node(1, x=6.5, y=1.5)

    repaired, stats = repair_topology_connectivity(
        graph,
        skeleton,
        traversable,
        geometry,
        TopologyConnectionCache(),
        neighbor_limit=10,
        map_revision=1,
    )

    assert not nx.is_connected(repaired)
    assert repaired.number_of_edges() == 0
    assert stats.unresolved_components == 2


def test_hierarchical_tracker_prefers_unknown_area_over_similarly_near_vertex():
    graph = nx.Graph()
    graph.add_node(0, x=0.5, y=1.5)
    graph.add_node(1, x=2.5, y=1.5)
    graph.add_node(2, x=0.5, y=3.5)
    graph.add_edge(0, 1, weight=2.0)
    graph.add_edge(0, 2, weight=2.0)
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2:5, :2] = -1
    tracker = HierarchicalGVDTracker(migration_radius=0.75)
    tracker.update_graph(graph)
    tracker.mark_reached(0)

    target = tracker.select_macro_target(
        (0.5, 1.5),
        grid,
        _geometry(width=5, height=5),
        unknown_radius=1.5,
    )

    assert target is not None
    assert target.vertex_id == 2
    assert target.unknown_area > 0.0


def test_hierarchical_tracker_migrates_explored_and_cleared_vertices():
    first = nx.Graph()
    first.add_node(0, x=1.0, y=1.0)
    first.add_node(1, x=2.0, y=1.0)
    first.add_edge(0, 1, weight=1.0)
    second = nx.Graph()
    second.add_node(10, x=1.1, y=1.0)
    second.add_node(11, x=2.1, y=1.0)
    second.add_edge(10, 11, weight=1.0)
    tracker = HierarchicalGVDTracker(migration_radius=0.25)
    tracker.update_graph(first)
    tracker.mark_reached(0)
    tracker.mark_local_cleared(0)

    tracker.update_graph(second)

    assert tracker.explored_vertices == {10}
    assert tracker.cleared_vertices == {10}
    assert tracker.active_vertex == 10


def test_hierarchical_tracker_requests_cleanup_for_leaf_or_nearly_exhausted_branch():
    graph = nx.Graph()
    graph.add_edges_from(((0, 1), (1, 2), (1, 3)))
    for node_id in graph.nodes:
        graph.nodes[node_id].update(x=float(node_id), y=0.0)
    tracker = HierarchicalGVDTracker(migration_radius=0.5)
    tracker.update_graph(graph)

    assert tracker.should_clear_local(0)
    assert not tracker.should_clear_local(1)
    tracker.explored_vertices.update({0, 2})
    assert tracker.should_clear_local(1)


def test_local_free_flood_stays_inside_room_and_filters_frontier_clusters():
    geometry = _geometry(width=7, height=5)
    grid = np.zeros((5, 7), dtype=np.int8)
    grid[:, 3] = 100

    mask = local_free_flood_mask(grid, geometry, (1.5, 2.5), half_extent=5.0)

    assert mask[2, 1]
    assert not mask[2, 5]
    assert cluster_touches_mask([(2, 1)], mask)
    assert not cluster_touches_mask([(2, 5)], mask)


def test_local_free_flood_expands_through_unknown_until_size_limit():
    geometry = _geometry(width=7, height=7)
    grid = np.full((7, 7), -1, dtype=np.int8)
    grid[3, 3] = 0

    mask = local_free_flood_mask(grid, geometry, (3.5, 3.5), half_extent=2.0)

    assert np.all(mask[1:6, 1:6])
    assert not np.any(mask[0, :])
    assert not np.any(mask[6, :])
    assert not np.any(mask[:, 0])
    assert not np.any(mask[:, 6])


def test_local_free_flood_stays_inside_coarse_prior_bounds():
    geometry = _geometry(width=7, height=7)
    grid = np.zeros((7, 7), dtype=np.int8)

    mask = local_free_flood_mask(
        grid,
        geometry,
        (3.5, 3.5),
        half_extent=10.0,
        bounds=WorldBounds(1.0, 6.0, 1.0, 6.0),
    )

    assert np.all(mask[1:6, 1:6])
    assert not np.any(mask[0, :])
    assert not np.any(mask[6, :])
    assert not np.any(mask[:, 0])
    assert not np.any(mask[:, 6])


def test_local_free_flood_selects_most_square_greedy_candidate():
    geometry = _geometry(width=5, height=5)
    free = np.asarray(
        (
            (1, 1, 1, 1, 1),
            (1, 0, 1, 0, 1),
            (1, 1, 1, 1, 1),
            (1, 1, 1, 0, 1),
            (1, 1, 0, 1, 1),
        ),
        dtype=bool,
    )
    grid = np.where(free, 0, 100).astype(np.int8)

    mask = local_free_flood_mask(grid, geometry, (2.5, 2.5), half_extent=5.0)
    rows, columns = np.flatnonzero(np.any(mask, axis=1)), np.flatnonzero(np.any(mask, axis=0))

    # Greedy expansion can produce a 4x1 vertical strip, a 1x5 horizontal
    # strip, or this 2x3 rectangle.  The local Region should prefer 2x3.
    assert (rows[0], rows[-1], columns[0], columns[-1]) == (2, 3, 0, 2)


def test_local_free_flood_stops_before_other_gvd_vertex():
    geometry = _geometry(width=9, height=9)
    grid = np.zeros((9, 9), dtype=np.int8)

    mask = local_free_flood_mask(
        grid,
        geometry,
        (4.5, 4.5),
        half_extent=10.0,
        bounds=WorldBounds(0.0, 9.0, 0.0, 9.0),
        excluded_points=((6.5, 4.5),),
    )

    assert mask[4, 4]
    assert not mask[4, 6]
    assert not np.any(mask[:, 6:])


def test_rectangle_mask_outline_uses_snapshot_geometry_origin():
    geometry = GridGeometry(-2.0, 3.0, 0.5, 5, 4)
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:5] = True

    outline = rectangle_mask_outline(mask, geometry)

    assert outline == (
        (-1.0, 3.5),
        (0.5, 3.5),
        (0.5, 4.5),
        (-1.0, 4.5),
        (-1.0, 3.5),
    )


def test_local_unknown_area_counts_only_disk_cells():
    geometry = _geometry(width=5, height=5)
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = -1
    grid[0, 0] = -1

    area = local_unknown_area(grid, geometry, (2.5, 2.5), radius=1.0)

    assert area == 1.0


def test_local_unknown_area_can_count_slam_map_exterior_as_unknown():
    geometry = _geometry(width=3, height=3)
    grid = np.zeros((3, 3), dtype=np.int8)

    inside_only = local_unknown_area(grid, geometry, (0.5, 0.5), radius=1.0)
    with_exterior = local_unknown_area(
        grid,
        geometry,
        (0.5, 0.5),
        radius=1.0,
        include_outside_map=True,
    )

    assert inside_only == 0.0
    assert with_exterior > 0.0


def test_astar_centerline_distance_penalty_prefers_medial_detour():
    traversable = np.ones((7, 7), dtype=bool)
    skeleton = np.zeros_like(traversable)
    skeleton[1, 1:6] = True
    skeleton[1:4, 1] = True
    skeleton[1:4, 5] = True
    distance = distance_to_mask(skeleton, resolution=1.0)

    path = astar_path(
        traversable,
        skeleton,
        (3, 1),
        (3, 5),
        skeleton_cost=1.0,
        off_skeleton_cost=1.0,
        centerline_distance=distance,
        centerline_distance_weight=5.0,
    )

    assert (1, 3) in path
    assert len(path) > 5


def test_boundary_unknown_score_rewards_unswept_border_direction():
    bounds = WorldBounds(0.0, 10.0, 0.0, 10.0)
    sweep = TrajectorySweepTracker(bounds, resolution=1.0, sweep_radius=1.5, overlap_radius=0.5)
    sweep.mark_pose((1.0, 5.0))
    geometry = bounds_geometry(bounds, 1.0)

    explored_direction = boundary_unknown_score((0.0, 5.0), (5.0, 5.0), geometry, sweep)
    unknown_direction = boundary_unknown_score((9.0, 5.0), (5.0, 5.0), geometry, sweep)

    assert unknown_direction > explored_direction


def test_path_overlap_ratio_counts_crossing_area():
    bounds = WorldBounds(0.0, 10.0, 0.0, 10.0)
    sweep = TrajectorySweepTracker(bounds, resolution=0.5, sweep_radius=1.0, overlap_radius=0.6)
    for y in np.linspace(2.0, 8.0, 13):
        sweep.mark_pose((5.0, float(y)))

    crossing = [(x, 5.0) for x in np.linspace(2.0, 8.0, 13)]
    disjoint = [(x, 1.0) for x in np.linspace(2.0, 8.0, 13)]

    assert path_overlap_ratio(crossing, sweep.geometry, sweep) > 0.0
    assert path_overlap_ratio(crossing, sweep.geometry, sweep) > path_overlap_ratio(
        disjoint,
        sweep.geometry,
        sweep,
    )


def test_ranked_goals_apply_straightness_and_overlap_terms():
    geometry = _geometry(width=9, height=9)
    traversable = np.ones((9, 9), dtype=bool)
    skeleton = np.zeros_like(traversable)
    skeleton[4, 1:8] = True
    skeleton[1:8, 4] = True
    topology = GVDTopology(nx.Graph(), geometry, skeleton, traversable)
    sweep = TrajectorySweepTracker(WorldBounds(0.0, 9.0, 0.0, 9.0), 1.0, 1.0, 0.4)
    sweep.mark_pose((4.5, 4.5))

    goals = rank_gvd_goals(
        topology,
        (4.5, 4.5),
        0.0,
        sweep,
        GVDWeights(
            boundary_unknown=0.0,
            goal_distance=0.0,
            path_overlap_penalty=0.0,
            straightness=1.0,
        ),
        min_goal_distance=1.0,
        max_goal_distance=4.0,
        candidate_limit=20,
        skeleton_cost=1.0,
        off_skeleton_cost=2.0,
    )

    assert goals
    assert math.isclose(goals[0].straightness, 1.0)
    assert goals[0].point[0] > 4.5


def test_new_obstacle_invalidates_active_path():
    geometry = _geometry(width=5, height=5)
    traversable = np.ones((5, 5), dtype=bool)
    topology = GVDTopology(nx.Graph(), geometry, traversable.copy(), traversable)
    path = [grid_to_world((2, column), geometry) for column in range(5)]
    assert not path_crosses_obstacle(path, topology)

    blocked = traversable.copy()
    blocked[2, 2] = False
    topology = GVDTopology(nx.Graph(), geometry, traversable.copy(), blocked)
    assert path_crosses_obstacle(path, topology)


def test_path_suffix_ignores_obstacles_behind_robot():
    path = [(float(column), 2.0) for column in range(6)]

    suffix = path_suffix_from_nearest(path, (4.1, 2.0), lookbehind_points=1)

    assert suffix == ((3.0, 2.0), (4.0, 2.0), (5.0, 2.0))


def test_path_obstruction_only_considers_new_cells_on_forward_suffix():
    geometry = _geometry(width=6, height=5)
    path = [(float(column) + 0.5, 2.5) for column in range(6)]
    suffix = path_suffix_from_nearest(path, (3.6, 2.5), lookbehind_points=1)
    previous = np.ones((5, 6), dtype=bool)
    previous[2, 4] = False

    existing_wall = previous.copy()
    assert not path_crosses_new_obstacle(suffix, geometry, previous, existing_wall)

    wall_behind_robot = previous.copy()
    wall_behind_robot[2, 0] = False
    assert not path_crosses_new_obstacle(suffix, geometry, previous, wall_behind_robot)

    wall_ahead = previous.copy()
    wall_ahead[2, 5] = False
    assert path_crosses_new_obstacle(suffix, geometry, previous, wall_ahead)


def test_robot_component_graph_drops_disconnected_skeleton_regions():
    graph = nx.Graph()
    graph.add_node(0, x=0.0, y=0.0)
    graph.add_node(1, x=1.0, y=0.0)
    graph.add_node(2, x=9.0, y=9.0)
    graph.add_edge(0, 1, weight=1.0, information_weight=1.0)

    component = robot_component_graph(graph, (0.1, 0.1))

    assert set(component.nodes) == {0, 1}


def test_translation_progress_watchdog_ignores_small_position_jitter():
    anchor, timestamp, refreshed = update_translation_progress(
        (1.0, 1.0),
        5.0,
        (1.05, 1.03),
        min_distance=0.15,
        now=9.0,
    )

    assert anchor == (1.0, 1.0)
    assert timestamp == 5.0
    assert not refreshed
    assert progress_watchdog_expired(timestamp, timeout=5.0, now=10.1)


def test_translation_progress_watchdog_refreshes_after_effective_displacement():
    anchor, timestamp, refreshed = update_translation_progress(
        (1.0, 1.0),
        5.0,
        (1.16, 1.0),
        min_distance=0.15,
        now=9.0,
    )

    assert anchor == (1.16, 1.0)
    assert timestamp == 9.0
    assert refreshed
    assert not progress_watchdog_expired(timestamp, timeout=5.0, now=10.1)


def test_random_recovery_motion_is_bounded():

    motion = sample_random_recovery_motion(
        random.Random(7),
        min_abs_yaw=0.6,
        max_abs_yaw=2.4,
        distance=0.45,
        speed=0.10,
    )

    assert 0.6 <= abs(motion.yaw_delta) <= 2.4
    assert motion.distance == 0.45
    assert motion.speed == 0.10
