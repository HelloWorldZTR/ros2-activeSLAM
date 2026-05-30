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

    def detect(
        self,
        grid_msg: OccupancyGrid,
        data: np.ndarray = None,
    ) -> Tuple[List[FrontierCluster], np.ndarray]:
        width = grid_msg.info.width
        height = grid_msg.info.height
        resolution = grid_msg.info.resolution
        origin_x = grid_msg.info.origin.position.x
        origin_y = grid_msg.info.origin.position.y

        if data is None:
            data = np.asarray(grid_msg.data, dtype=np.int8).reshape(height, width)
        elif data.shape != (height, width):
            raise ValueError('Cached occupancy grid shape does not match OccupancyGrid info.')

        free_mask = data == 0
        has_unknown_neighbor = np.zeros((height, width), dtype=bool)
        has_unknown_neighbor[1:, :] |= data[:-1, :] == -1
        has_unknown_neighbor[:-1, :] |= data[1:, :] == -1
        has_unknown_neighbor[:, 1:] |= data[:, :-1] == -1
        has_unknown_neighbor[:, :-1] |= data[:, 1:] == -1
        unknown_mask = np.logical_and(free_mask, has_unknown_neighbor)

        open_edge_mask = np.zeros((height, width), dtype=bool)
        if self.include_open_map_edges and height > 0 and width > 0:
            has_outside_neighbor = np.zeros((height, width), dtype=bool)
            has_outside_neighbor[0, :] = True
            has_outside_neighbor[-1, :] = True
            has_outside_neighbor[:, 0] = True
            has_outside_neighbor[:, -1] = True
            open_edge_mask = np.logical_and(
                free_mask,
                np.logical_and(has_outside_neighbor, np.logical_not(unknown_mask)),
            )

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
