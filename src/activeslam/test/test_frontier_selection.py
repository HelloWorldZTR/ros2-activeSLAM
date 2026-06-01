from types import SimpleNamespace

import pytest

from activeslam.frontier_selection import (
    frontier_probes_enabled_for_mode,
    make_frontier_candidate,
    ranked_frontier_candidates,
    ranked_local_cleanup_candidates,
)


def _cluster(source, size=1):
    return SimpleNamespace(source=source, size=size)


def _goal(x, y=0.0):
    return SimpleNamespace(point=(x, y))


def test_frontier_candidates_share_one_information_gain_ranking():
    ordinary = make_frontier_candidate(_cluster('unknown'), _goal(2.0), 1.0, 0.0, 0.0)
    open_edge = make_frontier_candidate(_cluster('open_edge'), _goal(1.0), 1.0, 0.0, 0.0)

    ranked = ranked_frontier_candidates([ordinary, open_edge], limit=8)

    assert ranked == [open_edge, ordinary]


def test_long_open_edge_cluster_does_not_win_from_size_alone():
    ordinary = make_frontier_candidate(
        _cluster('unknown', size=3),
        _goal(1.0),
        1.0,
        0.0,
        0.0,
    )
    open_edge = make_frontier_candidate(
        _cluster('open_edge', size=100),
        _goal(1.0),
        0.5,
        0.0,
        0.0,
    )

    ranked = ranked_frontier_candidates([open_edge, ordinary], limit=1)

    assert ranked == [ordinary]


def test_candidate_filter_rejects_missing_goal_and_cooldown_but_allows_low_gain():
    cluster = _cluster('unknown')

    assert make_frontier_candidate(cluster, None, 1.0, 0.0, 0.0) is None
    assert make_frontier_candidate(cluster, _goal(1.0), 0.0, 0.0, 0.0) is not None
    assert make_frontier_candidate(
        cluster,
        _goal(1.0),
        1.0,
        0.0,
        0.0,
        on_cooldown=True,
    ) is None


def test_candidate_utility_is_gain_over_safe_goal_distance():
    candidate = make_frontier_candidate(_cluster('unknown'), _goal(0.9), 1.0, 0.0, 0.0)

    assert candidate.utility == pytest.approx(1.0)


def test_local_cleanup_ranking_uses_cluster_size_over_distance():
    large = make_frontier_candidate(_cluster('unknown', size=10), _goal(2.0), 0.1, 0.0, 0.0)
    small = make_frontier_candidate(_cluster('unknown', size=2), _goal(0.5), 10.0, 0.0, 0.0)

    ranked = ranked_local_cleanup_candidates([small, large], 0.0, 0.0, limit=2)

    assert ranked == [large, small]


@pytest.mark.parametrize('slam_mode', ('frontier', 'approx_graph', 'gbsae'))
def test_frontier_driven_modes_enable_probes_by_default(slam_mode):
    assert frontier_probes_enabled_for_mode(
        slam_mode,
        frontier_modes_enabled=True,
        gvd_modes_enabled=False,
    )


@pytest.mark.parametrize('slam_mode', ('gvd_gbsae', 'gvd_hierarchical'))
def test_gvd_driven_modes_disable_probes_by_default(slam_mode):
    assert not frontier_probes_enabled_for_mode(
        slam_mode,
        frontier_modes_enabled=True,
        gvd_modes_enabled=False,
    )


def test_gvd_probe_default_can_be_overridden_for_ablation():
    assert frontier_probes_enabled_for_mode(
        'gvd_hierarchical',
        frontier_modes_enabled=True,
        gvd_modes_enabled=True,
    )


def test_hierarchical_local_cleanup_enables_probes_without_enabling_macro_gvd_probes():
    assert frontier_probes_enabled_for_mode(
        'gvd_hierarchical',
        frontier_modes_enabled=True,
        gvd_modes_enabled=False,
        hierarchical_local_cleanup=True,
    )
    assert not frontier_probes_enabled_for_mode(
        'gvd_hierarchical',
        frontier_modes_enabled=True,
        gvd_modes_enabled=False,
        hierarchical_local_cleanup=False,
    )


def test_hierarchical_local_cleanup_probe_can_be_disabled_for_ablation():
    assert not frontier_probes_enabled_for_mode(
        'gvd_hierarchical',
        frontier_modes_enabled=True,
        gvd_modes_enabled=False,
        hierarchical_local_cleanup=True,
        hierarchical_local_cleanup_enabled=False,
    )
