from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from nav_msgs.msg import OccupancyGrid

from .frontier_selection import OPEN_EDGE_FRONTIER, UNKNOWN_FRONTIER


@dataclass
class FrontierCluster:
    centroid_x: float
    centroid_y: float
    size: int
    cells: Tuple[Tuple[int, int], ...]
    source: str = UNKNOWN_FRONTIER


class FrontierDetector:
    def __init__(
        self,
        min_frontier_size: int = 5,
        include_open_map_edges: bool = True,
    ):
        self.min_frontier_size = min_frontier_size
        self.include_open_map_edges = include_open_map_edges

    def detect(self, grid_msg: OccupancyGrid) -> Tuple[List[FrontierCluster], np.ndarray]:
        width = grid_msg.info.width
        height = grid_msg.info.height
        resolution = grid_msg.info.resolution
        origin_x = grid_msg.info.origin.position.x
        origin_y = grid_msg.info.origin.position.y

        data = np.array(grid_msg.data, dtype=np.int8).reshape(height, width)

        unknown_mask = np.zeros((height, width), dtype=bool)
        open_edge_mask = np.zeros((height, width), dtype=bool)
        for i in range(height):
            for j in range(width):
                if data[i, j] != 0:
                    continue
                has_unknown_neighbor = False
                has_outside_neighbor = False
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < height and 0 <= nj < width):
                        has_outside_neighbor = True
                    elif data[ni, nj] == -1:
                        has_unknown_neighbor = True
                if has_unknown_neighbor:
                    unknown_mask[i, j] = True
                elif self.include_open_map_edges and has_outside_neighbor:
                    open_edge_mask[i, j] = True

        result = self._make_clusters(
            unknown_mask,
            UNKNOWN_FRONTIER,
            origin_x,
            origin_y,
            resolution,
        )
        result.extend(
            self._make_clusters(
                open_edge_mask,
                OPEN_EDGE_FRONTIER,
                origin_x,
                origin_y,
                resolution,
            )
        )
        return result, np.logical_or(unknown_mask, open_edge_mask)

    def _make_clusters(
        self,
        mask: np.ndarray,
        source: str,
        origin_x: float,
        origin_y: float,
        resolution: float,
    ) -> List[FrontierCluster]:
        result = []
        for cells in self._cluster(mask):
            if len(cells) < self.min_frontier_size:
                continue
            mean_i = sum(c[0] for c in cells) / len(cells)
            mean_j = sum(c[1] for c in cells) / len(cells)
            cx = origin_x + (mean_j + 0.5) * resolution
            cy = origin_y + (mean_i + 0.5) * resolution
            result.append(
                FrontierCluster(
                    centroid_x=cx,
                    centroid_y=cy,
                    size=len(cells),
                    cells=tuple(cells),
                    source=source,
                )
            )
        return result

    def _cluster(self, mask: np.ndarray) -> List[List[Tuple[int, int]]]:
        height, width = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        clusters = []

        for i in range(height):
            for j in range(width):
                if mask[i, j] and not visited[i, j]:
                    cluster = []
                    queue = deque([(i, j)])
                    visited[i, j] = True
                    while queue:
                        ci, cj = queue.popleft()
                        cluster.append((ci, cj))
                        for di, dj in (
                            (-1, -1),
                            (-1, 0),
                            (-1, 1),
                            (0, -1),
                            (0, 1),
                            (1, -1),
                            (1, 0),
                            (1, 1),
                        ):
                            ni, nj = ci + di, cj + dj
                            if (
                                0 <= ni < height
                                and 0 <= nj < width
                                and mask[ni, nj]
                                and not visited[ni, nj]
                            ):
                                visited[ni, nj] = True
                                queue.append((ni, nj))
                    clusters.append(cluster)

        return clusters
