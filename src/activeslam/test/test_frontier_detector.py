import importlib
import sys
from types import ModuleType

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


def test_cluster_connects_diagonal_frontier_cells():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    mask[2, 2] = True

    clusters = FrontierDetector(min_frontier_size=1)._cluster(mask)

    assert clusters == [[(0, 0), (1, 1), (2, 2)]]
