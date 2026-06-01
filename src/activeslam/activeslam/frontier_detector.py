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
        min_frontier_size: int = 10,
        include_open_map_edges: bool = True,
        low_confidence_fill_enabled: bool = True,
        low_confidence_fill_max_unknown_cells: int = 64,
        low_confidence_free_value: int = 25,
        low_confidence_occupied_value: int = 75,
    ):
        self.min_frontier_size = min_frontier_size
        self.include_open_map_edges = include_open_map_edges
        self.low_confidence_fill_enabled = low_confidence_fill_enabled
        self.low_confidence_fill_max_unknown_cells = max(
            0,
            int(low_confidence_fill_max_unknown_cells),
        )
        self.low_confidence_free_value = int(np.clip(low_confidence_free_value, 0, 50))
        self.low_confidence_occupied_value = int(
            np.clip(low_confidence_occupied_value, 51, 100)
        )

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

        data = self.fill_small_unknown_regions(data)
        unknown_mask = self._unknown_frontier_mask(data)
        free_mask = data == 0

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

    def fill_small_unknown_regions(self, data: np.ndarray) -> np.ndarray:
        """Infer bounded unknown pockets exposed only by tiny frontier clusters.

        Low-confidence values suppress small SLAM speckles without turning
        them into safe navigation goals.  Components touching the map edge or
        exceeding the configured size remain unknown so narrow entrances into
        large unexplored areas are preserved.
        """
        result = np.asarray(data, dtype=np.int8).copy()
        if (
            not self.low_confidence_fill_enabled
            or self.min_frontier_size <= 1
            or self.low_confidence_fill_max_unknown_cells <= 0
            or result.size == 0
        ):
            return result

        frontier_mask = self._unknown_frontier_mask(result)
        small_frontiers = [
            cells
            for cells in self._cluster(frontier_mask)
            if len(cells) < self.min_frontier_size
        ]
        if not small_frontiers:
            return result

        unknown_components = self._cluster(result == -1)
        component_by_cell = {
            cell: index
            for index, cells in enumerate(unknown_components)
            for cell in cells
        }
        frontier_sizes_by_component = {}
        for frontier_cells in self._cluster(frontier_mask):
            size = len(frontier_cells)
            for cell in frontier_cells:
                for neighbor in self._neighbors4(cell, result.shape):
                    component = component_by_cell.get(neighbor)
                    if component is not None:
                        frontier_sizes_by_component.setdefault(component, []).append(size)

        candidate_components = set()
        for frontier_cells in small_frontiers:
            for cell in frontier_cells:
                for neighbor in self._neighbors4(cell, result.shape):
                    component = component_by_cell.get(neighbor)
                    if component is not None:
                        candidate_components.add(component)

        height, width = result.shape
        for component in sorted(candidate_components):
            cells = unknown_components[component]
            if (
                len(cells) > self.low_confidence_fill_max_unknown_cells
                or any(
                    row in (0, height - 1) or column in (0, width - 1)
                    for row, column in cells
                )
                or any(
                    size >= self.min_frontier_size
                    for size in frontier_sizes_by_component.get(component, ())
                )
            ):
                continue
            occupied_votes, free_votes = self._geometry_votes(result, cells)
            if occupied_votes == 0 and free_votes == 0:
                continue
            value = (
                self.low_confidence_occupied_value
                if occupied_votes >= free_votes
                else self.low_confidence_free_value
            )
            rows, columns = zip(*cells)
            result[rows, columns] = value
        return result

    @staticmethod
    def _unknown_frontier_mask(data: np.ndarray) -> np.ndarray:
        free_mask = data == 0
        has_unknown_neighbor = np.zeros(data.shape, dtype=bool)
        has_unknown_neighbor[1:, :] |= data[:-1, :] == -1
        has_unknown_neighbor[:-1, :] |= data[1:, :] == -1
        has_unknown_neighbor[:, 1:] |= data[:, :-1] == -1
        has_unknown_neighbor[:, :-1] |= data[:, 1:] == -1
        return np.logical_and(free_mask, has_unknown_neighbor)

    @classmethod
    def _geometry_votes(
        cls,
        data: np.ndarray,
        cells: List[Tuple[int, int]],
    ) -> Tuple[int, int]:
        occupied_votes = 0
        free_votes = 0
        component = set(cells)
        for cell in cells:
            for neighbor in cls._neighbors8(cell, data.shape):
                if neighbor in component:
                    continue
                value = data[neighbor]
                occupied_votes += value > 50
                free_votes += value == 0
        return occupied_votes, free_votes

    @staticmethod
    def _neighbors4(cell: Tuple[int, int], shape):
        row, column = cell
        height, width = shape
        for delta_row, delta_column in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            neighbor = row + delta_row, column + delta_column
            if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width:
                yield neighbor

    @staticmethod
    def _neighbors8(cell: Tuple[int, int], shape):
        row, column = cell
        height, width = shape
        for delta_row in (-1, 0, 1):
            for delta_column in (-1, 0, 1):
                if delta_row == 0 and delta_column == 0:
                    continue
                neighbor = row + delta_row, column + delta_column
                if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width:
                    yield neighbor

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
