import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np


try:
    importlib.import_module('nav_msgs.msg')
except ModuleNotFoundError:
    nav_msgs = ModuleType('nav_msgs')
    nav_msgs_msg = ModuleType('nav_msgs.msg')
    nav_msgs_msg.OccupancyGrid = object
    nav_msgs.msg = nav_msgs_msg
    sys.modules['nav_msgs'] = nav_msgs
    sys.modules['nav_msgs.msg'] = nav_msgs_msg

FrontierDetector = importlib.import_module('activeslam.frontier_detector').FrontierDetector


def _grid(data):
    array = np.array(data, dtype=np.int8)
    return SimpleNamespace(
        info=SimpleNamespace(
            width=array.shape[1],
            height=array.shape[0],
            resolution=1.0,
            origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0)),
        ),
        data=array.reshape(-1).tolist(),
    )


def test_default_minimum_frontier_size_is_ten_pixels():
    assert FrontierDetector().min_frontier_size == 10


def test_cluster_connects_diagonal_frontier_cells():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    mask[2, 2] = True

    clusters = FrontierDetector(min_frontier_size=1)._cluster(mask)

    assert clusters == [[(0, 0), (1, 1), (2, 2)]]


def test_detects_unknown_frontier_cluster():
    clusters, _ = FrontierDetector(min_frontier_size=3).detect(_grid([
        [100, 100, 100, 100, 100],
        [100, -1, -1, -1, 100],
        [100, 0, 0, 0, 100],
        [100, 100, 100, 100, 100],
    ]))

    assert len(clusters) == 1
    assert clusters[0].source == 'unknown'
    assert clusters[0].size == 3


def test_detects_open_map_edge_frontier_cluster():
    clusters, _ = FrontierDetector(min_frontier_size=3).detect(_grid([
        [100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100],
        [100, 0, 0, 0, 100],
    ]))

    assert len(clusters) == 1
    assert clusters[0].source == 'open_edge'
    assert clusters[0].size == 3


def test_open_map_edge_requires_free_cells():
    clusters, _ = FrontierDetector(min_frontier_size=1).detect(_grid([
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
    ]))

    assert clusters == []


def test_unknown_frontier_takes_priority_over_open_edge_for_same_cell():
    clusters, _ = FrontierDetector(min_frontier_size=1).detect(_grid([
        [100, 0, 100],
        [100, -1, 100],
        [100, 100, 100],
    ]))

    assert len(clusters) == 1
    assert clusters[0].source == 'unknown'
    assert clusters[0].cells == ((0, 1),)


def test_open_map_edge_reuses_minimum_frontier_size():
    clusters, _ = FrontierDetector(min_frontier_size=3).detect(_grid([
        [100, 100, 100],
        [100, 100, 100],
        [0, 0, 100],
    ]))

    assert clusters == []


def test_open_map_edge_detection_can_be_disabled():
    clusters, _ = FrontierDetector(
        min_frontier_size=1,
        include_open_map_edges=False,
    ).detect(_grid([
        [100, 100, 100],
        [100, 100, 100],
        [0, 0, 100],
    ]))

    assert clusters == []


def test_detect_reuses_cached_grid_without_changing_clusters():
    msg = _grid([
        [100, 100, 100, 100, 100],
        [100, -1, -1, -1, 100],
        [100, 0, 0, 0, 100],
        [100, 100, 100, 100, 100],
    ])
    data = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
    detector = FrontierDetector(min_frontier_size=3)

    uncached_clusters, uncached_mask = detector.detect(msg)
    cached_clusters, cached_mask = detector.detect(msg, data)

    assert cached_clusters == uncached_clusters
    assert np.array_equal(cached_mask, uncached_mask)


def test_small_unknown_pocket_uses_low_confidence_free_fill_when_free_dominates():
    data = np.zeros((5, 5), dtype=np.int8)
    data[2, 2] = -1

    filled = FrontierDetector().fill_small_unknown_regions(data)

    assert filled[2, 2] == 25
    assert data[2, 2] == -1


def test_small_unknown_pocket_uses_low_confidence_occupied_fill_when_walls_dominate():
    data = np.full((5, 5), 100, dtype=np.int8)
    data[2, 2] = -1
    data[2, 3] = 0

    detector = FrontierDetector()
    filled = detector.fill_small_unknown_regions(data)
    clusters, _ = detector.detect(_grid(data.tolist()))

    assert filled[2, 2] == 75
    assert clusters == []


def test_unknown_component_larger_than_fill_limit_remains_unknown():
    data = np.zeros((5, 7), dtype=np.int8)
    data[2, 2:5] = -1

    filled = FrontierDetector(
        low_confidence_fill_max_unknown_cells=2,
    ).fill_small_unknown_regions(data)

    assert np.all(filled[2, 2:5] == -1)


def test_unknown_component_touching_map_edge_remains_unknown():
    data = np.zeros((5, 5), dtype=np.int8)
    data[0, 2] = -1

    filled = FrontierDetector().fill_small_unknown_regions(data)

    assert filled[0, 2] == -1


def test_low_confidence_fill_can_be_disabled():
    data = np.zeros((5, 5), dtype=np.int8)
    data[2, 2] = -1

    filled = FrontierDetector(
        low_confidence_fill_enabled=False,
    ).fill_small_unknown_regions(data)

    assert filled[2, 2] == -1
