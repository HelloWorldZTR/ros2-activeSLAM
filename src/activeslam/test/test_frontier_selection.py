from types import SimpleNamespace

import pytest

from activeslam.frontier_selection import (
    make_frontier_candidate,
    ranked_frontier_candidates,
)


def _cluster(source, size=1):
    return SimpleNamespace(source=source, size=size)


def _goal(x, y=0.0):
    return SimpleNamespace(point=(x, y))


def test_frontier_candidates_share_one_information_gain_ranking():
    ordinary = make_frontier_candidate(_cluster('unknown'), _goal(2.0), 1.0, 0.0, 0.0, 0.45)
    open_edge = make_frontier_candidate(_cluster('open_edge'), _goal(1.0), 1.0, 0.0, 0.0, 0.45)

    ranked = ranked_frontier_candidates([ordinary, open_edge], limit=8)

    assert ranked == [open_edge, ordinary]


def test_long_open_edge_cluster_does_not_win_from_size_alone():
    ordinary = make_frontier_candidate(
        _cluster('unknown', size=3),
        _goal(1.0),
        1.0,
        0.0,
        0.0,
        0.45,
    )
    open_edge = make_frontier_candidate(
        _cluster('open_edge', size=100),
        _goal(1.0),
        0.5,
        0.0,
        0.0,
        0.45,
    )

    ranked = ranked_frontier_candidates([open_edge, ordinary], limit=1)

    assert ranked == [ordinary]


def test_candidate_filter_rejects_missing_goal_low_gain_and_cooldown():
    cluster = _cluster('unknown')

    assert make_frontier_candidate(cluster, None, 1.0, 0.0, 0.0, 0.45) is None
    assert make_frontier_candidate(cluster, _goal(1.0), 0.44, 0.0, 0.0, 0.45) is None
    assert make_frontier_candidate(
        cluster,
        _goal(1.0),
        1.0,
        0.0,
        0.0,
        0.45,
        on_cooldown=True,
    ) is None


def test_candidate_utility_is_gain_over_safe_goal_distance():
    candidate = make_frontier_candidate(_cluster('unknown'), _goal(0.9), 1.0, 0.0, 0.0, 0.45)

    assert candidate.utility == pytest.approx(1.0)
