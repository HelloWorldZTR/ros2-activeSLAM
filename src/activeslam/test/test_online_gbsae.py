import math

import networkx as nx
import numpy as np
import pytest

from activeslam.frontier_goal_utils import GridGeometry
from activeslam.online_gbsae import (
    BootstrapWeights,
    BranchHypothesis,
    WorldBounds,
    bootstrap_score,
    branch_path_has_no_known_obstacle,
    build_online_topology,
    directional_remaining_unknown,
    known_area_ratio,
    mark_branch_explored,
    migrate_explored_vertices,
    normal_unknown_depth,
    path_known_ratio,
    record_branch_hypotheses,
    skeleton_to_graph,
    update_branch_failure,
    zhang_suen_thinning,
)


def _geometry(width, height, resolution=1.0):
    return GridGeometry(0.0, 0.0, resolution, width, height)


def _node(graph, node_id, x, y):
    graph.add_node(node_id, x=float(x), y=float(y))


def test_known_ratio_uses_prior_area_not_dynamic_map_extent():
    grid = np.asarray([[0, -1], [0, 0]], dtype=np.int8)

    ratio = known_area_ratio(grid, _geometry(2, 2), WorldBounds(0.0, 4.0, 0.0, 2.0))

    assert ratio == pytest.approx(3.0 / 8.0)


def test_directional_unknown_counts_prior_cells_outside_current_map():
    grid = np.zeros((2, 2), dtype=np.int8)
    bounds = WorldBounds(0.0, 4.0, 0.0, 2.0)

    east = directional_remaining_unknown(
        grid,
        _geometry(2, 2),
        bounds,
        origin=(1.0, 1.0),
        target=(4.0, 1.0),
        half_angle=math.pi / 4.0,
    )

    assert east > 0.0


def test_normal_unknown_depth_and_path_known_ratio_are_normalized():
    grid = np.asarray([[0, -1, -1, 100]], dtype=np.int8)

    depth = normal_unknown_depth(
        grid,
        _geometry(4, 1),
        start=(0.5, 0.5),
        normal=(1.0, 0.0),
        max_depth=3.0,
    )
    known_ratio = path_known_ratio(
        [(0.0, 0.5), (1.5, 0.5), (2.5, 0.5), (3.5, 0.5)],
        grid,
        _geometry(4, 1),
    )

    assert depth == pytest.approx(2.0 / 3.0)
    assert known_ratio == pytest.approx(1.5 / 3.5)


def test_bootstrap_score_prefers_expansion_weights():
    score = bootstrap_score(
        unknown_depth=0.5,
        directional_unknown=0.75,
        path_known_ratio=0.25,
        weights=BootstrapWeights(),
    )

    assert score.path_known_ratio == pytest.approx(0.25)
    assert score.total == pytest.approx(2.5 + 1.125 - 25.0)


def test_bootstrap_score_has_no_distance_reward():
    common = {
        'unknown_depth': 0.5,
        'directional_unknown': 0.75,
        'path_known_ratio': 0.25,
        'weights': BootstrapWeights(),
    }

    score = bootstrap_score(**common)

    assert not hasattr(score, 'capped_goal_distance')
    with pytest.raises(TypeError, match='robot_goal_distance'):
        bootstrap_score(robot_goal_distance=6.0, **common)


def test_branch_hypotheses_merge_retry_block_and_explore():
    bounds = WorldBounds(-5.0, 5.0, -5.0, 5.0)
    branches = record_branch_hypotheses(
        [],
        [
            ((0.0, 0.0), (1.0, 0.0), 1.0),
            ((0.1, 0.0), (1.0, 0.0), 0.9),
            ((3.0, 0.0), (1.0, 0.0), 0.85),
            ((0.0, 0.0), (0.0, 1.0), 0.8),
        ],
        best_score=1.0,
        score_ratio=0.7,
        min_angle=math.pi / 4.0,
        merge_radius=1.0,
        projection_distance=1.0,
        bounds=bounds,
    )

    assert len(branches) == 2
    branches = update_branch_failure(branches, branches[0].branch_id, 2)
    branches = update_branch_failure(branches, branches[0].branch_id, 2)
    assert branches[0].blocked
    assert mark_branch_explored(branches, branches[1].branch_id)[1].explored


def test_branch_ray_allows_unknown_but_rejects_known_obstacle():
    branch = BranchHypothesis(0, (0.5, 0.5), (2.5, 0.5), (1.0, 0.0), 1.0)
    unknown = np.asarray([[0, -1, -1]], dtype=np.int8)
    obstacle = np.asarray([[0, -1, 100]], dtype=np.int8)

    assert branch_path_has_no_known_obstacle(unknown, _geometry(3, 1), branch)
    assert not branch_path_has_no_known_obstacle(obstacle, _geometry(3, 1), branch)


def test_thinning_and_support_vertices_compress_a_corridor():
    corridor = np.zeros((3, 7), dtype=bool)
    corridor[1, :] = True

    skeleton = zhang_suen_thinning(corridor)
    graph = skeleton_to_graph(skeleton, _geometry(7, 3), support_vertex_spacing=2.0)

    assert np.array_equal(skeleton, corridor)
    assert graph.number_of_nodes() == 4
    assert max(attributes['weight'] for _, _, attributes in graph.edges(data=True)) <= 2.0


def test_explored_vertices_migrate_by_coverage_disk_overlap():
    old = nx.Graph()
    new = nx.Graph()
    _node(old, 0, 0.0, 0.0)
    _node(new, 0, 0.2, 0.0)
    _node(new, 1, 3.0, 0.0)

    inherited = migrate_explored_vertices(new, old, {0}, radius=1.0, overlap_threshold=0.5)

    assert inherited == {0}


def test_online_topology_attaches_pending_branch_as_virtual_leaf():
    grid = np.full((3, 7), -1, dtype=np.int8)
    grid[1, :] = 0
    branch = BranchHypothesis(3, (6.5, 1.5), (7.5, 1.5), (1.0, 0.0), 1.0)

    topology = build_online_topology(
        grid,
        _geometry(7, 3),
        robot_xy=(0.5, 1.5),
        branches=[branch],
        clearance=0.0,
        spur_prune_length=0.0,
        support_vertex_spacing=2.0,
    )

    branch_nodes = [
        node_id
        for node_id, attributes in topology.graph.nodes(data=True)
        if attributes.get('kind') == 'branch'
    ]
    assert len(branch_nodes) == 1
    branch_node = branch_nodes[0]
    assert branch_node in topology.target_vertices
    assert topology.graph.nodes[branch_node]['branch_id'] == 3
    assert topology.graph.edges[next(iter(topology.graph.neighbors(branch_node))), branch_node][
        'virtual'
    ]
