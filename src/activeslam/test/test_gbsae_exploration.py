import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from activeslam.gbsae_exploration import (
    GBSAEPlanner,
    greedy_visit_route,
    insert_spectral_loop_revisits,
    load_prior_graph,
    point_is_known_free,
    shortest_path_expansion,
    weighted_spanning_tree_d_opt,
)


def _payload():
    return {
        'world': 'test_world',
        'nodes': [
            {'id': 0, 'x': 0.0, 'y': 0.0},
            {'id': 1, 'x': 1.0, 'y': 0.0},
            {'id': 2, 'x': 2.0, 'y': 0.0},
            {'id': 3, 'x': 1.0, 'y': 1.0},
        ],
        'edges': [[0, 1], [1, 2], [1, 3]],
    }


def _write_graph(tmp_path, payload=None):
    path = tmp_path / 'test_world.gbsae.json'
    path.write_text(json.dumps(_payload() if payload is None else payload))
    return path


def _graph(edges):
    graph = nx.Graph()
    for source, target, distance in edges:
        graph.add_edge(
            source,
            target,
            weight=distance,
            information_weight=1.0 / distance,
        )
    for node_id in graph.nodes:
        graph.nodes[node_id]['x'] = float(node_id)
        graph.nodes[node_id]['y'] = 0.0
    return graph


def _frontier(x, y=0.0, utility=1.0):
    return SimpleNamespace(
        cluster=SimpleNamespace(centroid_x=x, centroid_y=y),
        utility=utility,
    )


def test_load_prior_graph_adds_metric_weights(tmp_path):
    graph = load_prior_graph(_write_graph(tmp_path), expected_world='test_world')

    assert sorted(graph.nodes) == [0, 1, 2, 3]
    assert graph.edges[0, 1]['weight'] == pytest.approx(1.0)
    assert graph.edges[0, 1]['information_weight'] == pytest.approx(1.0)


def test_slam_office_prior_graph_asset_loads():
    repo_src = Path(__file__).resolve().parents[2]
    path = repo_src / 'activeslam_resource' / 'maps' / 'slam_office.gbsae.json'

    graph = load_prior_graph(path, expected_world='slam_office')

    assert graph.number_of_nodes() == 89
    assert graph.number_of_edges() == 143
    assert graph.nodes[0] == {'x': 12.1, 'y': 1.5}


@pytest.mark.parametrize(
    ('mutate', 'message'),
    [
        (lambda payload: payload['nodes'].append(payload['nodes'][0]), 'Duplicate'),
        (lambda payload: payload['edges'].append([2, 99]), 'unknown node'),
        (lambda payload: payload['edges'].append([0, 0]), 'self-edge'),
        (lambda payload: payload['edges'].pop(), 'connected'),
    ],
)
def test_load_prior_graph_rejects_invalid_shape(tmp_path, mutate, message):
    payload = _payload()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        load_prior_graph(_write_graph(tmp_path, payload))


def test_load_prior_graph_rejects_missing_asset(tmp_path):
    with pytest.raises(FileNotFoundError, match='GBSAE prior graph is missing'):
        load_prior_graph(tmp_path / 'missing.gbsae.json')


def test_greedy_route_is_deterministic_and_shortest_paths_expand():
    graph = _graph([
        (0, 1, 1.0),
        (1, 2, 1.0),
        (1, 3, 1.0),
    ])

    assert greedy_visit_route(graph, 0) == [0, 1, 2, 1, 3]
    assert shortest_path_expansion(graph, [0, 2, 3]) == [0, 1, 2, 1, 3]


def test_spectral_loop_revisit_is_inserted_only_when_objective_improves():
    graph = _graph([
        (0, 1, 1.0),
        (1, 2, 1.0),
        (0, 2, 1.0),
    ])

    inserted, loop_edges = insert_spectral_loop_revisits(graph, [0, 1, 2], 0.0)
    rejected, no_loop_edges = insert_spectral_loop_revisits(graph, [0, 1, 2], 10.0)

    assert loop_edges == [(0, 2)]
    assert [(step.vertex_id, step.loop_revisit) for step in inserted] == [
        (0, False),
        (1, False),
        (2, False),
        (0, True),
        (2, True),
    ]
    assert no_loop_edges == []
    assert [step.vertex_id for step in rejected] == [0, 1, 2]


def test_weighted_spanning_tree_score_increases_with_loop_edge():
    path = _graph([(0, 1, 1.0), (1, 2, 1.0)])
    loop = path.copy()
    loop.add_edge(0, 2, weight=1.0, information_weight=1.0)

    assert weighted_spanning_tree_d_opt(loop) > weighted_spanning_tree_d_opt(path)


def test_frontier_allocation_excludes_completed_regions_and_tracks_active_step():
    graph = _graph([(0, 1, 1.0), (1, 2, 1.0)])
    planner = GBSAEPlanner(graph, (0.0, 0.0), loop_path_cost_weight=10.0)
    planner.advance_active_step()
    active_frontier = _frontier(1.1, utility=0.5)
    future_frontier = _frontier(2.0, utility=2.0)
    completed_frontier = _frontier(0.1, utility=10.0)

    assigned = planner.allocate_frontiers([
        completed_frontier,
        future_frontier,
        active_frontier,
    ])

    assert 0 not in assigned
    assert assigned[1] == [completed_frontier, active_frontier]
    assert assigned[2] == [future_frontier]
    assert planner.frontiers_for_active([future_frontier, active_frontier]) == [
        active_frontier
    ]


def test_route_progression_advances_reached_prefix():
    graph = _graph([(0, 1, 1.0), (1, 2, 1.0)])
    planner = GBSAEPlanner(graph, (0.0, 0.0), loop_path_cost_weight=10.0)

    reached = planner.advance_reached_steps((0.1, 0.0), reach_radius=0.25)

    assert [step.vertex_id for step in reached] == [0]
    assert planner.visited_prefix == [0]
    assert planner.active_step.vertex_id == 1


def test_online_target_subset_uses_explored_nodes_only_as_transit():
    graph = _graph([(0, 1, 1.0), (1, 2, 1.0), (1, 3, 1.0)])

    planner = GBSAEPlanner(
        graph,
        (0.0, 0.0),
        loop_path_cost_weight=10.0,
        target_vertices={2, 3},
        explored_vertices={0, 1},
    )

    assert planner.completed_vertices == {0, 1}
    assert [step.vertex_id for step in planner.route] == [0, 1, 2, 1, 3]


def test_known_free_grid_check_rejects_unknown_and_outside_points():
    grid = np.asarray([[0, -1], [100, 0]], dtype=np.int8)
    msg = SimpleNamespace(
        info=SimpleNamespace(
            width=2,
            height=2,
            resolution=1.0,
            origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0)),
        )
    )

    assert point_is_known_free(msg, grid, (0.5, 0.5))
    assert not point_is_known_free(msg, grid, (1.5, 0.5))
    assert not point_is_known_free(msg, grid, (-0.1, 0.5))
