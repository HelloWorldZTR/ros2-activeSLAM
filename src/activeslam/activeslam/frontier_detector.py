from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from nav_msgs.msg import OccupancyGrid


@dataclass
class FrontierCluster:
    centroid_x: float
    centroid_y: float
    size: int


class FrontierDetector:
    def __init__(self, min_frontier_size: int = 5):
        self.min_frontier_size = min_frontier_size

    def detect(self, grid_msg: OccupancyGrid) -> Tuple[List[FrontierCluster], np.ndarray]:
        width = grid_msg.info.width
        height = grid_msg.info.height
        resolution = grid_msg.info.resolution
        origin_x = grid_msg.info.origin.position.x
        origin_y = grid_msg.info.origin.position.y

        data = np.array(grid_msg.data, dtype=np.int8).reshape(height, width)

        frontier_mask = np.zeros((height, width), dtype=bool)
        for i in range(height):
            for j in range(width):
                if data[i, j] == 0:
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width and data[ni, nj] == -1:
                            frontier_mask[i, j] = True
                            break

        clusters = self._cluster(frontier_mask)

        result = []
        for cells in clusters:
            if len(cells) < self.min_frontier_size:
                continue
            mean_i = sum(c[0] for c in cells) / len(cells)
            mean_j = sum(c[1] for c in cells) / len(cells)
            cx = origin_x + (mean_j + 0.5) * resolution
            cy = origin_y + (mean_i + 0.5) * resolution
            result.append(FrontierCluster(centroid_x=cx, centroid_y=cy, size=len(cells)))

        return result, frontier_mask

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
                        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            ni, nj = ci + di, cj + dj
                            if 0 <= ni < height and 0 <= nj < width and mask[ni, nj] and not visited[ni, nj]:
                                visited[ni, nj] = True
                                queue.append((ni, nj))
                    clusters.append(cluster)

        return clusters
