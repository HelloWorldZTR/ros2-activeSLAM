import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _stub_ros_messages():
    try:
        importlib.import_module('nav_msgs.msg')
    except ModuleNotFoundError:
        nav_msgs = ModuleType('nav_msgs')
        nav_msgs_msg = ModuleType('nav_msgs.msg')
        nav_msgs_msg.OccupancyGrid = object
        nav_msgs.msg = nav_msgs_msg
        sys.modules['nav_msgs'] = nav_msgs
        sys.modules['nav_msgs.msg'] = nav_msgs_msg

    try:
        importlib.import_module('geometry_msgs.msg')
    except ModuleNotFoundError:
        geometry_msgs = ModuleType('geometry_msgs')
        geometry_msgs_msg = ModuleType('geometry_msgs.msg')
        geometry_msgs_msg.Point = object
        geometry_msgs.msg = geometry_msgs_msg
        sys.modules['geometry_msgs'] = geometry_msgs
        sys.modules['geometry_msgs.msg'] = geometry_msgs_msg

    try:
        importlib.import_module('visualization_msgs.msg')
    except ModuleNotFoundError:
        visualization_msgs = ModuleType('visualization_msgs')
        visualization_msgs_msg = ModuleType('visualization_msgs.msg')
        visualization_msgs_msg.Marker = object
        visualization_msgs_msg.MarkerArray = object
        visualization_msgs.msg = visualization_msgs_msg
        sys.modules['visualization_msgs'] = visualization_msgs
        sys.modules['visualization_msgs.msg'] = visualization_msgs_msg


_stub_ros_messages()
graph_exploration = importlib.import_module('activeslam.graph_exploration')


def _grid(data):
    array = np.asarray(data, dtype=np.int8)
    return SimpleNamespace(
        info=SimpleNamespace(
            width=array.shape[1],
            height=array.shape[0],
            resolution=1.0,
            origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0)),
        ),
        data=array.reshape(-1).tolist(),
    )


def test_cell_information_reuses_cached_grid_without_changing_ratios():
    msg = _grid([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0],
        [0, 0, 100, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    data = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)

    uncached = graph_exploration.cell_information(msg, 2.5, 2.5, 1.1)
    cached = graph_exploration.cell_information(msg, 2.5, 2.5, 1.1, data)

    assert cached == uncached
    assert cached == pytest.approx((0.2, 0.2))


def test_graph_score_reuses_cached_grid_without_changing_score():
    msg = _grid([
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 100, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    data = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
    scorer = graph_exploration.GraphBasedFrontierScorer(
        info_radius=1.1,
        hallucinated_node_spacing=0.5,
        loop_closure_radius=2.0,
        loop_closure_min_separation=20,
        loop_closure_occupied_threshold=0.03,
        loop_closure_weight=1.5,
        max_loop_closures_per_node=3,
        path_cost_weight=0.05,
        odom_information=graph_exploration.make_information_matrix(0.04, 0.04, 0.008),
    )
    graph = graph_exploration.WeightedPoseGraph()
    path = [(0.5, 0.5), (1.5, 1.5), (2.5, 2.5)]

    uncached = scorer.score(graph, msg, path)
    cached = scorer.score(graph, msg, path, data)

    assert cached == pytest.approx(uncached)
