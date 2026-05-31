import math

import networkx as nx
import numpy as np

from activeslam.frontier_goal_utils import GridGeometry
from activeslam.gvd_exploration import (
    GVDTopology,
    GVDWeights,
    TrajectorySweepTracker,
    WorldBounds,
    astar_path,
    boundary_unknown_score,
    bounds_geometry,
    build_obstacle_gvd_topology,
    grid_to_world,
    path_crosses_obstacle,
    path_overlap_ratio,
    rank_gvd_goals,
    robot_component_graph,
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


def test_robot_component_graph_drops_disconnected_skeleton_regions():
    graph = nx.Graph()
    graph.add_node(0, x=0.0, y=0.0)
    graph.add_node(1, x=1.0, y=0.0)
    graph.add_node(2, x=9.0, y=9.0)
    graph.add_edge(0, 1, weight=1.0, information_weight=1.0)

    component = robot_component_graph(graph, (0.1, 0.1))

    assert set(component.nodes) == {0, 1}
