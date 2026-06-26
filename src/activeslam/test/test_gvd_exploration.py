import math
import random
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pytest

from activeslam.frontier_goal_utils import GridGeometry
from activeslam.gvd_exploration import (
    GVDGuidePlanner,
    GVDTopology,
    GVDWeights,
    HierarchicalGVDTracker,
    TopologyConnectionCache,
    TrajectorySweepTracker,
    WorldBounds,
    _cluster_close_vertices,
    _split_chain_vertices,
    astar_reconnection_segments,
    astar_path,
    bidirectional_astar_path,
    boundary_unknown_score,
    bounds_geometry,
    build_sparse_gvd_graph,
    build_sparse_gvd_graph_from_topology,
    build_obstacle_gvd_topology,
    build_obstacle_traversability,
    distance_to_mask,
    gvd_guide_edge_blocked_run,
    gvd_guide_plan_steps,
    grid_to_world,
    cluster_touches_mask,
    insert_gvd_guide_loop_revisits,
    local_free_flood_mask,
    local_region_known_ratio,
    local_unknown_area,
    local_unknown_ratio,
    normalize_topology_vertex_kinds,
    off_graph_new_free_area,
    offset_repeated_route_segments,
    path_crosses_new_obstacle,
    path_crosses_obstacle,
    path_suffix_from_nearest,
    path_overlap_ratio,
    progress_watchdog_expired,
    rank_gvd_goals,
    rectangle_mask_outline,
    repair_topology_connectivity,
    robot_component_graph,
    route_replan_due,
    sample_random_recovery_motion,
    sparse_open_tsp_route,
    shortcut_gvd_guide_route,
    suppress_unconfident_cycles,
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


def test_sparse_gvd_graph_keeps_only_chain_endpoints_and_edge_polyline():
    skeleton = np.zeros((5, 7), dtype=bool)
    skeleton[2, 1:6] = True

    graph = build_sparse_gvd_graph(skeleton, _geometry(width=7, height=5))

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert all(graph.degree[node_id] == 1 for node_id in graph.nodes)
    edge = next(iter(graph.edges(data=True)))
    assert len(edge[2]['polyline']) == 5
    assert edge[2]['weight'] == pytest.approx(4.0)


def test_sparse_gvd_graph_anchors_cycle_without_structural_vertices():
    skeleton = np.zeros((5, 5), dtype=bool)
    for cell in ((1, 2), (2, 3), (3, 2), (2, 1)):
        skeleton[cell] = True

    graph = build_sparse_gvd_graph(skeleton, _geometry(width=5, height=5))

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() >= 1
    assert nx.is_connected(graph)
    assert {data['kind'] for _, data in graph.nodes(data=True)} == {'cycle_anchor'}


def test_gvd_guide_sparse_graph_uses_repaired_processed_topology():
    topology_graph = nx.Graph()
    topology_graph.add_node(10, x=0.0, y=0.0, kind='endpoint')
    topology_graph.add_node(11, x=1.0, y=0.0, kind='support')
    topology_graph.add_node(12, x=2.0, y=0.0, kind='support')
    topology_graph.add_node(13, x=3.0, y=0.0, kind='endpoint')
    topology_graph.add_edge(
        10,
        11,
        weight=1.0,
        connection_mode='gvd',
        polyline=((0.0, 0.0), (1.0, 0.0)),
    )
    topology_graph.add_edge(
        11,
        12,
        weight=1.0,
        connection_mode='gvd',
        polyline=((1.0, 0.0), (2.0, 0.0)),
    )
    topology_graph.add_edge(
        12,
        13,
        weight=1.4,
        connection_mode='astar',
        path=((2.0, 0.0), (2.5, 0.5), (3.0, 0.0)),
    )

    graph = build_sparse_gvd_graph_from_topology(topology_graph, _geometry())

    assert set(graph.nodes) == {10, 13}
    assert graph.number_of_edges() == 1
    edge = graph.edges[10, 13]
    assert edge['source_graph_nodes'] == (10, 11, 12, 13)
    assert edge['connection_mode'] == 'mixed'
    assert edge['connection_modes'] == ('gvd', 'gvd', 'astar')
    assert edge['weight'] == pytest.approx(3.4)
    assert (2.5, 0.5) in edge['polyline']


def test_gvd_guide_plan_migrates_explored_vertices_across_rebuilds():
    graph = nx.Graph()
    graph.add_node(0, x=0.0, y=0.0, kind='branch')
    graph.add_node(1, x=1.0, y=0.0, kind='endpoint')
    graph.add_node(2, x=0.0, y=1.0, kind='endpoint')
    graph.add_edge(0, 1, weight=1.0, polyline=((0.0, 0.0), (1.0, 0.0)))
    graph.add_edge(0, 2, weight=1.0, polyline=((0.0, 0.0), (0.0, 1.0)))
    topology = GVDTopology(
        graph=graph,
        geometry=_geometry(),
        skeleton=np.zeros((1, 1), dtype=bool),
        traversable=np.ones((1, 1), dtype=bool),
    )

    planner = GVDGuidePlanner.build(
        topology,
        np.zeros((1, 1), dtype=np.int8),
        (0.0, 0.0),
        (),
        loop_path_cost_weight=10.0,
        frontier_detour_weight=1.0,
        frontier_detour_max_extra_distance=1.0,
        frontier_detour_min_gain=1.0,
        explored_points=((1.05, 0.0),),
        migration_radius=0.2,
    )

    assert planner.explored_vertices == (1,)
    assert all(step.vertex_id != 1 for step in planner.plan_queue)
    assert any(step.vertex_id == 2 for step in planner.plan_queue)


def test_gvd_guide_rebuild_start_prefers_target_hint_over_robot_nearest():
    graph = nx.Graph()
    graph.add_node(0, x=0.0, y=0.0, kind='endpoint')
    graph.add_node(1, x=1.0, y=0.0, kind='support')
    graph.add_node(2, x=2.0, y=0.0, kind='endpoint')
    graph.add_edge(0, 1, weight=1.0, polyline=((0.0, 0.0), (1.0, 0.0)))
    graph.add_edge(1, 2, weight=1.0, polyline=((1.0, 0.0), (2.0, 0.0)))
    topology = GVDTopology(
        graph=graph,
        geometry=_geometry(width=3, height=1),
        skeleton=np.ones((1, 3), dtype=bool),
        traversable=np.ones((1, 3), dtype=bool),
    )

    planner = GVDGuidePlanner.build(
        topology,
        np.zeros((1, 3), dtype=np.int8),
        robot_xy=(0.05, 0.0),
        frontiers=[],
        loop_path_cost_weight=10.0,
        frontier_detour_weight=1.0,
        frontier_detour_max_extra_distance=1.0,
        frontier_detour_min_gain=1.0,
        start_hint=(2.0, 0.0),
    )

    assert planner.route_vertices[0] == 2
    assert planner.start_hint == (2.0, 0.0)


def test_gvd_guide_rebuild_start_hint_uses_obstacle_aware_astar_estimate():
    graph = nx.Graph()
    graph.add_node(1, x=2.5, y=0.5, kind='endpoint')
    graph.add_node(2, x=0.5, y=2.5, kind='endpoint')
    graph.add_edge(
        1,
        2,
        weight=4.0,
        polyline=((2.5, 0.5), (0.5, 2.5)),
    )
    traversable = np.ones((4, 4), dtype=bool)
    traversable[0, 2] = False
    topology = GVDTopology(
        graph=graph,
        geometry=_geometry(width=4, height=4),
        skeleton=np.zeros((4, 4), dtype=bool),
        traversable=traversable,
    )

    planner = GVDGuidePlanner.build(
        topology,
        np.zeros((4, 4), dtype=np.int8),
        robot_xy=(0.5, 2.5),
        frontiers=[],
        loop_path_cost_weight=10.0,
        frontier_detour_weight=1.0,
        frontier_detour_max_extra_distance=1.0,
        frontier_detour_min_gain=1.0,
        start_hint=(2.5, 1.5),
    )

    assert planner.route_vertices[0] == 2


def test_gvd_guide_edge_blocked_run_rewards_contiguous_obstruction():
    geometry = _geometry(width=6, height=1)
    polyline = tuple((col + 0.5, 0.5) for col in range(6))
    isolated = np.ones((1, 6), dtype=bool)
    isolated[0, 2] = False
    contiguous = np.ones((1, 6), dtype=bool)
    contiguous[0, 2:5] = False

    assert gvd_guide_edge_blocked_run(polyline, geometry, contiguous) > (
        gvd_guide_edge_blocked_run(polyline, geometry, isolated)
    )
    assert gvd_guide_edge_blocked_run(polyline, geometry, contiguous) > 0.28


def test_gvd_guide_edge_blocked_run_uses_inflated_obstacle_mask():
    geometry = _geometry(width=10, height=3, resolution=0.1)
    bounds = WorldBounds(0.0, 1.0, 0.0, 0.3)
    grid = np.zeros((3, 10), dtype=np.int8)
    grid[2, 5] = 100
    polyline = ((0.05, 0.05), (0.55, 0.05), (0.95, 0.05))
    _, raw_traversable = build_obstacle_traversability(
        grid,
        geometry,
        bounds,
        resolution=0.1,
        clearance=0.0,
        boundary_margin=0.0,
    )
    _, inflated_traversable = build_obstacle_traversability(
        grid,
        geometry,
        bounds,
        resolution=0.1,
        clearance=0.14,
        boundary_margin=0.0,
    )

    assert gvd_guide_edge_blocked_run(polyline, geometry, raw_traversable) == 0.0
    assert gvd_guide_edge_blocked_run(polyline, geometry, inflated_traversable) > 0.0


def test_off_graph_new_free_area_counts_only_far_unknown_to_free_cells():
    geometry = _geometry(width=5, height=5)
    graph = nx.Graph()
    graph.add_node(0, x=0.5, y=0.5)
    graph.add_node(1, x=4.5, y=0.5)
    graph.add_edge(
        0,
        1,
        polyline=tuple((col + 0.5, 0.5) for col in range(5)),
        weight=4.0,
    )
    rebuild = np.full((5, 5), -1, dtype=np.int8)
    current = rebuild.copy()
    current[0, 2] = 0
    current[4, 3] = 0
    current[4, 4] = 0

    area = off_graph_new_free_area(
        rebuild,
        current,
        geometry,
        graph,
        distance_threshold=1.0,
    )

    assert area == pytest.approx(2.0)


def test_sparse_open_tsp_route_connects_current_start_to_required_targets():
    graph = nx.path_graph(3)
    for node_id in graph.nodes:
        graph.nodes[node_id].update(x=float(node_id), y=0.0)
    nx.set_edge_attributes(graph, 1.0, 'weight')

    route = sparse_open_tsp_route(graph, start=1, targets=(0, 2))

    assert route[0] == 1
    assert set(route) == {0, 1, 2}


def test_gvd_guide_route_shortcut_removes_clear_repeated_interior_vertices():
    graph = nx.Graph()
    graph.add_node(0, x=1.5, y=1.5)
    graph.add_node(1, x=1.5, y=3.5)
    graph.add_node(2, x=3.5, y=1.5)
    graph.add_node(3, x=3.5, y=3.5)
    for source, target in ((0, 1), (0, 2), (0, 3)):
        graph.add_edge(source, target, weight=2.0, information_weight=0.5)
    traversable = np.ones((5, 5), dtype=bool)

    route, shortcuts = shortcut_gvd_guide_route(
        graph,
        (0, 1, 0, 2, 0, 3, 0),
        _geometry(width=5, height=5),
        traversable,
    )

    assert route == (0, 1, 2, 3, 0)
    assert shortcuts == 2
    assert graph.edges[1, 2]['connection_mode'] == 'shortcut'
    assert graph.edges[2, 3]['connection_mode'] == 'shortcut'


def test_gvd_guide_route_shortcut_keeps_repeated_vertex_when_blocked():
    graph = nx.Graph()
    graph.add_node(0, x=1.5, y=1.5)
    graph.add_node(1, x=1.5, y=3.5)
    graph.add_node(2, x=3.5, y=1.5)
    graph.add_edge(0, 1, weight=2.0, information_weight=0.5)
    graph.add_edge(0, 2, weight=2.0, information_weight=0.5)
    traversable = np.ones((5, 5), dtype=bool)
    traversable[2, 2] = False

    route, shortcuts = shortcut_gvd_guide_route(
        graph,
        (0, 1, 0, 2, 0),
        _geometry(width=5, height=5),
        traversable,
    )

    assert route == (0, 1, 0, 2, 0)
    assert shortcuts == 0
    assert not graph.has_edge(1, 2)


def test_gvd_guide_spectral_loop_insertion_uses_positive_dopt_gain():
    graph = nx.Graph()
    for node_id, point in ((0, (0.0, 0.0)), (1, (1.0, 0.0)), (2, (0.5, 1.0))):
        graph.add_node(node_id, x=point[0], y=point[1])
    for source, target in ((0, 1), (1, 2), (0, 2)):
        graph.add_edge(source, target, weight=1.0, information_weight=1.0)

    steps, loop_edges = insert_gvd_guide_loop_revisits(graph, (0, 1, 2), 0.0)

    assert loop_edges == ((0, 2),)
    assert [(step.vertex_id, step.loop_revisit) for step in steps] == [
        (0, False),
        (1, False),
        (2, False),
        (0, True),
        (2, True),
    ]


@dataclass
class _GuideGoal:
    point: tuple


@dataclass
class _GuideFrontier:
    safe_goal: _GuideGoal
    information_gain: float


def test_gvd_guide_frontier_detour_inserted_only_when_gain_beats_extra_distance():
    graph = nx.path_graph(2)
    graph.nodes[0].update(x=0.0, y=0.0)
    graph.nodes[1].update(x=2.0, y=0.0)
    graph.edges[0, 1].update(weight=2.0, information_weight=0.5)
    route_steps, _ = insert_gvd_guide_loop_revisits(graph, (0, 1), 10.0)
    good = _GuideFrontier(_GuideGoal((1.0, 0.2)), information_gain=3.0)
    weak = _GuideFrontier(_GuideGoal((1.0, 0.3)), information_gain=0.2)

    queue = gvd_guide_plan_steps(
        graph,
        route_steps,
        {1: [weak, good]},
        frontier_detour_weight=1.0,
        frontier_detour_max_extra_distance=1.0,
        frontier_detour_min_gain=1.0,
    )

    assert [step.kind for step in queue] == ['frontier_detour', 'gvd_vertex']
    assert queue[0].frontier is good


def test_gvd_guide_frontier_detour_skips_obstacle_blocked_candidate():
    graph = nx.path_graph(2)
    graph.nodes[0].update(x=0.5, y=2.5)
    graph.nodes[1].update(x=4.5, y=2.5)
    graph.edges[0, 1].update(weight=4.0, information_weight=0.25)
    route_steps, _ = insert_gvd_guide_loop_revisits(graph, (0, 1), 10.0)
    candidate = _GuideFrontier(_GuideGoal((2.5, 2.5)), information_gain=10.0)
    traversable = np.ones((5, 5), dtype=bool)
    traversable[:, 1] = False

    queue = gvd_guide_plan_steps(
        graph,
        route_steps,
        {1: [candidate]},
        frontier_detour_weight=0.0,
        frontier_detour_max_extra_distance=10.0,
        frontier_detour_min_gain=0.0,
        geometry=_geometry(width=5, height=5),
        traversable=traversable,
    )

    assert [step.kind for step in queue] == ['gvd_vertex']


def test_gvd_guide_online_detour_refresh_updates_next_local_segment():
    graph = nx.path_graph(3)
    graph.nodes[0].update(x=0.0, y=0.0)
    graph.nodes[1].update(x=2.0, y=0.0)
    graph.nodes[2].update(x=4.0, y=0.0)
    graph.edges[0, 1].update(weight=2.0, information_weight=0.5)
    graph.edges[1, 2].update(weight=2.0, information_weight=0.5)
    route_steps, _ = insert_gvd_guide_loop_revisits(graph, (0, 1, 2), 10.0)
    queue = gvd_guide_plan_steps(
        graph,
        route_steps,
        {},
        frontier_detour_weight=1.0,
        frontier_detour_max_extra_distance=1.0,
        frontier_detour_min_gain=1.0,
    )
    planner = GVDGuidePlanner(graph, np.zeros((2, 2), dtype=np.int8), (0, 1, 2), queue)
    reached = planner.advance_active_step()
    assert reached.vertex_id == 1

    good = _GuideFrontier(_GuideGoal((3.0, 0.2)), information_gain=3.0)
    update = planner.refresh_online_frontier_detour(
        1,
        [good],
        frontier_detour_weight=1.0,
        frontier_detour_max_extra_distance=1.0,
        frontier_detour_min_gain=1.0,
    )

    assert update.inserted
    assert update.target_vertex == 2
    assert planner.online_detour_updates == 1
    assert planner.online_detour_insertions == 1
    assert planner.active_step.kind == 'frontier_detour'
    assert planner.active_step.frontier is good

    update = planner.refresh_online_frontier_detour(
        1,
        [],
        frontier_detour_weight=1.0,
        frontier_detour_max_extra_distance=1.0,
        frontier_detour_min_gain=1.0,
    )

    assert not update.inserted
    assert update.removed_detours == 1
    assert planner.active_step.kind == 'gvd_vertex'
    assert planner.active_step.vertex_id == 2


def test_gvd_guide_macro_edge_is_split_into_bounded_waypoints():
    graph = nx.path_graph(2)
    graph.nodes[0].update(x=0.0, y=0.0)
    graph.nodes[1].update(x=10.0, y=0.0)
    graph.edges[0, 1].update(
        weight=10.0,
        information_weight=0.1,
        polyline=((0.0, 0.0), (10.0, 0.0)),
    )
    route_steps, _ = insert_gvd_guide_loop_revisits(graph, (0, 1), 10.0)

    queue = gvd_guide_plan_steps(
        graph,
        route_steps,
        {},
        frontier_detour_weight=1.0,
        frontier_detour_max_extra_distance=1.0,
        frontier_detour_min_gain=1.0,
        max_waypoint_distance=4.0,
    )

    assert [step.goal_xy for step in queue] == [(4.0, 0.0), (8.0, 0.0), (10.0, 0.0)]
    assert [step.vertex_id for step in queue] == [None, None, 1]
    assert [step.expected_cost for step in queue] == pytest.approx([4.0, 4.0, 2.0])


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


def test_topology_repair_uses_relaxed_astar_mask_for_narrow_bridge():
    geometry = _geometry(width=7, height=3)
    traversable = np.ones((3, 7), dtype=bool)
    traversable[:, 3] = False
    astar_traversable = np.ones_like(traversable)
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
        astar_traversable=astar_traversable,
        neighbor_limit=10,
        map_revision=1,
    )

    assert nx.is_connected(repaired)
    assert repaired.edges[0, 1]['connection_mode'] == 'astar'
    assert stats.astar_edges == 1


def test_astar_reconnection_segments_render_only_fallback_bridge_paths():
    graph = nx.Graph()
    graph.add_edge(
        0,
        1,
        connection_mode='astar',
        path=((0.0, 0.0), (1.0, 0.5), (2.0, 0.0)),
    )
    graph.add_edge(
        1,
        2,
        connection_mode='gvd',
        path=((2.0, 0.0), (3.0, 0.0)),
    )

    assert astar_reconnection_segments(graph) == (
        ((0.0, 0.0), (1.0, 0.5)),
        ((1.0, 0.5), (2.0, 0.0)),
    )


def test_repeated_tsp_route_segments_are_offset_and_keep_direction():
    segments = offset_repeated_route_segments(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 0.0)),
        spacing=0.10,
    )

    assert segments == (
        ((0.0, -0.05), (1.0, -0.05)),
        ((1.0, 0.05), (0.0, 0.05)),
    )


def test_hierarchical_tracker_expands_tsp_walk_through_transit_vertices():
    graph = nx.Graph()
    graph.add_node(0, x=0.5, y=1.5)
    graph.add_node(1, x=2.5, y=1.5)
    graph.add_node(2, x=4.5, y=1.5)
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 2, weight=1.0)
    tracker = HierarchicalGVDTracker(migration_radius=0.75)
    tracker.update_graph(graph)
    tracker.mark_local_cleared(1)
    tracker.rebuild_route((0.5, 1.5))

    target = tracker.select_macro_target(
        (0.5, 1.5),
    )

    assert target is not None
    assert target.vertex_id == 1
    assert target.travel_cost == pytest.approx(1.0)
    assert tracker.route_targets == (0, 2)
    assert 1 in tracker.transit_route


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


def test_hierarchical_tracker_remaps_inflight_target_after_graph_rebuild():
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
    tracker.rebuild_route((1.0, 1.0))
    target = tracker.select_macro_target((1.0, 1.0))
    assert target is not None
    assert target.vertex_id == 1

    tracker.update_graph(second)
    remapped = tracker.remap_target(target)

    assert remapped is not None
    assert remapped.vertex_id == 11
    assert remapped.point == (2.1, 1.0)


def test_hierarchical_tracker_marks_rebuilt_vertex_reached_by_goal_point():
    graph = nx.Graph()
    graph.add_node(10, x=1.1, y=1.0)
    graph.add_node(11, x=2.1, y=1.0)
    graph.add_edge(10, 11, weight=1.0)
    tracker = HierarchicalGVDTracker(migration_radius=0.25)
    tracker.update_graph(graph)

    reached = tracker.mark_reached_point((1.0, 1.0))

    assert reached == 10
    assert tracker.active_vertex == 10
    assert tracker.should_clear_local(10)


def test_hierarchical_tracker_does_not_reuse_stale_vertex_id_for_completed_goal():
    graph = nx.Graph()
    graph.add_node(0, x=10.0, y=10.0)
    tracker = HierarchicalGVDTracker(migration_radius=0.25)
    tracker.update_graph(graph)

    reached = tracker.mark_reached_point((1.0, 1.0))

    assert reached is None
    assert tracker.explored_vertices == set()


def test_hierarchical_tracker_allows_uncleared_leaf_cleanup_without_explored_state():
    graph = nx.Graph()
    graph.add_edges_from(((0, 1), (1, 2), (1, 3)))
    for node_id in graph.nodes:
        graph.nodes[node_id].update(x=float(node_id), y=0.0)
    tracker = HierarchicalGVDTracker(migration_radius=0.5)
    tracker.update_graph(graph)

    assert tracker.should_clear_local(0)
    tracker.mark_local_cleared(0)
    assert not tracker.should_clear_local(0)
    tracker.mark_reached(1)
    assert not tracker.should_clear_local(1)


def test_hierarchical_tracker_counts_astar_bridge_as_an_ordinary_leaf_edge():
    graph = nx.Graph()
    graph.add_node(0, x=0.0, y=0.0)
    graph.add_node(1, x=1.0, y=0.0)
    graph.add_edge(0, 1, weight=1.0, connection_mode='astar')
    tracker = HierarchicalGVDTracker(migration_radius=0.5)
    tracker.update_graph(graph)

    tracker.mark_reached(0)

    assert tracker.should_clear_local(0)


def test_hierarchical_tracker_keeps_explored_but_uncleared_vertex_as_transit_only():
    graph = nx.path_graph(3)
    for node_id in graph.nodes:
        graph.nodes[node_id].update(x=float(node_id), y=0.0)
    nx.set_edge_attributes(graph, 1.0, 'weight')
    tracker = HierarchicalGVDTracker(migration_radius=0.5)
    tracker.update_graph(graph)
    tracker.mark_reached(0)
    tracker.mark_reached(1)

    tracker.rebuild_route((0.0, 0.0))

    assert tracker.route_targets == (2,)
    assert 1 in tracker.transit_route


def test_hierarchical_tracker_keeps_support_and_corner_vertices_as_transit_only():
    graph = nx.path_graph(4)
    kinds = ('endpoint', 'support', 'corner', 'endpoint')
    for node_id, kind in enumerate(kinds):
        graph.nodes[node_id].update(x=float(node_id), y=0.0, kind=kind)
    nx.set_edge_attributes(graph, 1.0, 'weight')
    tracker = HierarchicalGVDTracker(migration_radius=0.5)
    tracker.update_graph(graph)

    tracker.rebuild_route((0.0, 0.0))

    assert tracker.route_targets == (0, 3)
    assert 1 in tracker.transit_route
    assert 2 in tracker.transit_route


def test_hierarchical_tracker_keeps_branch_when_component_has_no_endpoint_target():
    graph = nx.cycle_graph(4)
    for node_id in graph.nodes:
        graph.nodes[node_id].update(x=float(node_id), y=0.0, kind='support')
    graph.nodes[0]['kind'] = 'branch'
    nx.set_edge_attributes(graph, 1.0, 'weight')
    tracker = HierarchicalGVDTracker(migration_radius=0.5)
    tracker.update_graph(graph)

    tracker.rebuild_route((1.0, 0.0))

    assert tracker.route_targets == (0,)


def test_hierarchical_tracker_runs_expansion_before_deferred_leaf_cleanup():
    graph = nx.path_graph(3)
    for node_id in graph.nodes:
        graph.nodes[node_id].update(x=float(node_id), y=0.0, kind='endpoint')
    graph.nodes[1]['kind'] = 'support'
    nx.set_edge_attributes(graph, 1.0, 'weight')
    tracker = HierarchicalGVDTracker(migration_radius=0.5)
    tracker.update_graph(graph)
    tracker.mark_reached(0)
    tracker.mark_reached(1)

    tracker.rebuild_route((1.0, 0.0))

    assert tracker.route_phase == 'expansion'
    assert tracker.expansion_targets == (2,)
    assert tracker.cleanup_targets == (0,)
    assert tracker.route_targets == (2,)


def test_hierarchical_tracker_switches_to_cleanup_route_after_expansion():
    graph = nx.path_graph(3)
    for node_id in graph.nodes:
        graph.nodes[node_id].update(x=float(node_id), y=0.0, kind='endpoint')
    graph.nodes[1]['kind'] = 'support'
    nx.set_edge_attributes(graph, 1.0, 'weight')
    tracker = HierarchicalGVDTracker(migration_radius=0.5)
    tracker.update_graph(graph)
    tracker.mark_reached(0)
    tracker.mark_reached(1)
    tracker.mark_reached(2)

    tracker.rebuild_route((1.0, 0.0))

    assert tracker.route_phase == 'cleanup'
    assert tracker.expansion_targets == ()
    assert tracker.cleanup_targets == (0, 2)
    assert tracker.route_targets == (0, 2)


def test_hierarchical_tracker_treats_completed_branch_as_degenerate_leaf():
    graph = nx.Graph()
    for node_id, kind in ((0, 'branch'), (1, 'endpoint'), (2, 'endpoint')):
        graph.add_node(node_id, x=float(node_id), y=0.0, kind=kind)
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(0, 2, weight=1.0, connection_mode='astar')
    tracker = HierarchicalGVDTracker(migration_radius=0.5)
    tracker.update_graph(graph)
    tracker.mark_reached(1)
    tracker.mark_reached(2)

    assert tracker.should_clear_local(0)


def test_hierarchical_tracker_keeps_branch_open_while_one_subtree_needs_expansion():
    graph = nx.Graph()
    for node_id, kind in ((0, 'branch'), (1, 'endpoint'), (2, 'endpoint')):
        graph.add_node(node_id, x=float(node_id), y=0.0, kind=kind)
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(0, 2, weight=1.0, connection_mode='astar')
    tracker = HierarchicalGVDTracker(migration_radius=0.5)
    tracker.update_graph(graph)
    tracker.mark_reached(1)

    assert not tracker.should_clear_local(0)


def test_hierarchical_tracker_joins_open_tsp_from_nearer_endpoint():
    graph = nx.path_graph(4)
    for node_id in graph.nodes:
        graph.nodes[node_id].update(x=float(node_id), y=0.0)
    nx.set_edge_attributes(graph, 1.0, 'weight')
    tracker = HierarchicalGVDTracker(migration_radius=0.5)
    tracker.update_graph(graph)

    route = tracker.rebuild_route((3.0, 0.0))

    assert route[0] == 3
    assert route[-1] == 0


def test_hierarchical_tracker_preserves_rebuild_direction_with_one_forced_successor():
    first = nx.Graph()
    for node_id, point, kind in (
        (0, (0.0, 0.0), 'branch'),
        (1, (1.0, 0.0), 'endpoint'),
        (2, (0.0, 1.0), 'endpoint'),
    ):
        first.add_node(node_id, x=point[0], y=point[1], kind=kind)
    first.add_edge(0, 1, weight=1.0)
    first.add_edge(0, 2, weight=1.0)
    second = nx.Graph()
    for node_id, point, kind in (
        (10, (0.0, 0.0), 'branch'),
        (11, (1.0, 0.1), 'endpoint'),
        (12, (-0.1, 1.0), 'endpoint'),
    ):
        second.add_node(node_id, x=point[0], y=point[1], kind=kind)
    second.add_edge(10, 11, weight=1.0)
    second.add_edge(10, 12, weight=1.0)
    tracker = HierarchicalGVDTracker(migration_radius=0.5)
    tracker.update_graph(first)
    tracker.mark_reached(0)
    tracker.transit_route = (1,)

    tracker.update_graph(second)
    route = tracker.rebuild_route((0.0, 0.0))

    assert tracker.continuation_successor == 11
    assert route[0] == 11
    assert 10 in tracker.explored_vertices


def test_hierarchical_tracker_rebuild_stops_at_active_leaf_for_immediate_cleanup():
    graph = nx.path_graph(2)
    graph.nodes[0].update(x=0.0, y=0.0, kind='endpoint')
    graph.nodes[1].update(x=1.0, y=0.0, kind='endpoint')
    nx.set_edge_attributes(graph, 1.0, 'weight')
    tracker = HierarchicalGVDTracker(migration_radius=0.5)
    tracker.update_graph(graph)
    tracker.mark_reached(0)
    tracker.transit_route = (1,)

    tracker.update_graph(graph)
    route = tracker.rebuild_route((0.0, 0.0))

    assert route == ()
    assert tracker.route_phase == 'cleanup_ready'
    assert tracker.continuation_successor is None


def test_hierarchical_tracker_runs_networkx_tsp_once_after_forced_successor(monkeypatch):
    graph = nx.Graph()
    for node_id, point, kind in (
        (0, (0.0, 0.0), 'branch'),
        (1, (1.0, 0.0), 'endpoint'),
        (2, (0.0, 1.0), 'endpoint'),
        (3, (-1.0, 0.0), 'endpoint'),
    ):
        graph.add_node(node_id, x=point[0], y=point[1], kind=kind)
    graph.add_edges_from(((0, 1), (0, 2), (0, 3)), weight=1.0)
    tracker = HierarchicalGVDTracker(migration_radius=0.5)
    tracker.update_graph(graph)
    tracker.mark_reached(0)
    tracker.transit_route = (1,)
    calls = []

    def fake_tsp(*args, **kwargs):
        calls.append((args, kwargs))
        return [1, 0, 2, 0, 3]

    monkeypatch.setattr(nx.approximation, 'traveling_salesman_problem', fake_tsp)
    tracker.update_graph(graph)
    route = tracker.rebuild_route((0.0, 0.0))

    assert len(calls) == 1
    assert tracker.continuation_successor == 1
    assert route == (1, 0, 2, 0, 3)


def test_split_chain_vertices_inserts_visible_corner():
    chain = ((1, 1), (1, 2), (1, 3), (2, 3), (3, 3))

    vertices = _split_chain_vertices(
        chain,
        resolution=1.0,
        spacing=10.0,
        corner_turn_threshold=math.pi / 4.0,
    )

    assert vertices == [(0, 'support'), (2, 'corner'), (4, 'support')]


def test_stable_vertex_clustering_preserves_bridge_metadata_and_unrelated_neighbor():
    graph = nx.Graph()
    graph.add_node(0, x=0.0, y=0.0, kind='branch')
    graph.add_node(1, x=0.2, y=0.0, kind='support')
    graph.add_node(2, x=2.0, y=0.0, kind='endpoint')
    graph.add_node(3, x=0.1, y=0.1, kind='endpoint')
    graph.add_edge(0, 1, weight=0.2, connection_mode='gvd')
    graph.add_edge(1, 2, weight=1.8, connection_mode='astar', path=((0.0, 0.0), (2.0, 0.0)))

    clustered = _cluster_close_vertices(graph, min_spacing=1.0)

    assert set(clustered.nodes) == {0, 2, 3}
    assert clustered.edges[0, 2]['connection_mode'] == 'astar'
    assert clustered.edges[0, 2]['path'] == ((0.0, 0.0), (2.0, 0.0))
    assert clustered.edges[0, 2]['weight'] == pytest.approx(2.0)


def test_route_replan_due_limits_dirty_route_refresh_rate():
    assert not route_replan_due(10.0, interval=0.5, now=10.49)
    assert route_replan_due(10.0, interval=0.5, now=10.5)


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


def test_local_free_flood_keeps_leaf_at_region_center():
    geometry = _geometry(width=7, height=7)
    grid = np.zeros((7, 7), dtype=np.int8)

    mask = local_free_flood_mask(grid, geometry, (3.5, 3.5), half_extent=2.0)
    rows, columns = np.flatnonzero(np.any(mask, axis=1)), np.flatnonzero(np.any(mask, axis=0))

    assert (rows[0], rows[-1], columns[0], columns[-1]) == (1, 5, 1, 5)


def test_local_free_flood_weights_trade_area_for_squareness():
    geometry = _geometry(width=7, height=7)
    grid = np.zeros((7, 7), dtype=np.int8)
    grid[1, :] = 100
    grid[5, :] = 100

    area_mask = local_free_flood_mask(
        grid,
        geometry,
        (3.5, 3.5),
        half_extent=3.0,
        area_weight=3.0,
        squareness_weight=1.0,
    )
    square_mask = local_free_flood_mask(
        grid,
        geometry,
        (3.5, 3.5),
        half_extent=3.0,
        area_weight=0.0,
        squareness_weight=1.0,
    )

    assert np.all(area_mask[2:5, :])
    assert not np.any(area_mask[:2, :])
    assert not np.any(area_mask[5:, :])
    assert np.all(square_mask[2:5, 2:5])
    assert np.count_nonzero(square_mask) == 9


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


def test_local_region_known_ratio_counts_only_masked_observed_cells():
    grid = np.array(
        (
            (0, -1, 100),
            (25, 75, -1),
        ),
        dtype=np.int8,
    )
    mask = np.array(
        (
            (True, True, False),
            (True, False, True),
        ),
        dtype=bool,
    )

    assert local_region_known_ratio(grid, mask) == pytest.approx(0.5)


def test_local_region_known_ratio_rejects_empty_or_mismatched_region():
    grid = np.zeros((2, 2), dtype=np.int8)

    assert local_region_known_ratio(grid, np.zeros((2, 2), dtype=bool)) == 0.0
    assert local_region_known_ratio(grid, np.zeros((1, 1), dtype=bool)) == 0.0


def test_local_region_known_ratio_projects_snapshot_after_map_expansion():
    grid = np.zeros((4, 4), dtype=np.int8)
    grid[1, 2] = -1
    region_mask = np.ones((2, 2), dtype=bool)

    ratio = local_region_known_ratio(
        grid,
        region_mask,
        grid_geometry=GridGeometry(-1.0, -1.0, 1.0, 4, 4),
        region_geometry=GridGeometry(0.0, 0.0, 1.0, 2, 2),
    )

    assert ratio == pytest.approx(0.75)


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


def test_local_unknown_ratio_counts_unknown_and_map_exterior():
    geometry = _geometry(width=3, height=3)
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[0, 0] = -1

    ratio = local_unknown_ratio(grid, geometry, (0.5, 0.5), radius=1.0)

    assert ratio > 0.0
    assert ratio < 1.0


def test_unknown_heavy_cycle_suppression_keeps_only_local_mst_edges():
    graph = nx.cycle_graph(4)
    for node_id, point in enumerate(((0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5))):
        graph.nodes[node_id].update(x=point[0], y=point[1])
    nx.set_edge_attributes(graph, 1.0, 'weight')
    grid = np.full((3, 3), -1, dtype=np.int8)

    pruned, stats = suppress_unconfident_cycles(
        graph,
        grid,
        _geometry(width=3, height=3),
        radius=0.1,
        ratio_threshold=0.5,
    )

    assert stats.unconfident_vertices == 4
    assert stats.removed_edges == 1
    assert nx.is_tree(pruned)


def test_topology_kind_normalization_promotes_pruned_chain_ends_to_endpoints():
    graph = nx.path_graph(3)
    for node_id in graph.nodes:
        graph.nodes[node_id].update(x=float(node_id), y=0.0, kind='support')

    normalized = normalize_topology_vertex_kinds(graph)

    assert normalized.nodes[0]['kind'] == 'endpoint'
    assert normalized.nodes[1]['kind'] == 'support'
    assert normalized.nodes[2]['kind'] == 'endpoint'


def test_confident_cycle_is_not_suppressed():
    graph = nx.cycle_graph(4)
    for node_id, point in enumerate(((0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5))):
        graph.nodes[node_id].update(x=point[0], y=point[1])
    nx.set_edge_attributes(graph, 1.0, 'weight')
    grid = np.zeros((3, 3), dtype=np.int8)

    pruned, stats = suppress_unconfident_cycles(
        graph,
        grid,
        _geometry(width=3, height=3),
        radius=0.1,
        ratio_threshold=0.5,
    )

    assert stats.unconfident_vertices == 0
    assert stats.removed_edges == 0
    assert pruned.number_of_edges() == 4


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
