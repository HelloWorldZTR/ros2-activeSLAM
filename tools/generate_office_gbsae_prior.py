#!/usr/bin/env python3
"""Generate a reviewable GBSAE prior graph for the IoU-friendly office map.

The upstream occupancy image distinguishes obstacles from free space, but not
office floor from exterior free space. Keep the sampling bounds conservative,
validate waypoint clearance and line-of-sight edges, then retain only the
component connected to the PublicBathroomB seed.
"""

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from generate_slam_office_world import decode_rgb_png


Point = Tuple[float, float]
Edge = Tuple[int, int]


class OccupancyImage:
    """Query the office occupancy PNG in ROS map coordinates."""

    def __init__(
        self,
        path: Path,
        resolution: float,
        origin_x: float,
        origin_y: float,
        occupied_threshold: int,
    ):
        self.width, self.height, self.pixels = decode_rgb_png(path)
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.occupied_threshold = occupied_threshold

    def occupied(self, x: float, y: float) -> bool:
        column = int((x - self.origin_x) / self.resolution)
        map_row = int((y - self.origin_y) / self.resolution)
        image_row = self.height - 1 - map_row
        if not (0 <= column < self.width and 0 <= image_row < self.height):
            return True
        offset = (image_row * self.width + column) * 3
        return max(self.pixels[offset:offset + 3]) <= self.occupied_threshold

    def disk_is_clear(self, point: Point, radius: float) -> bool:
        x, y = point
        steps = math.ceil(radius / self.resolution)
        for row_offset in range(-steps, steps + 1):
            for column_offset in range(-steps, steps + 1):
                if math.hypot(row_offset, column_offset) * self.resolution > radius:
                    continue
                if self.occupied(
                    x + column_offset * self.resolution,
                    y + row_offset * self.resolution,
                ):
                    return False
        return True

    def segment_is_clear(self, source: Point, target: Point, radius: float) -> bool:
        distance = math.dist(source, target)
        count = max(1, math.ceil(distance / (self.resolution / 2.0)))
        for index in range(count + 1):
            alpha = index / count
            point = (
                source[0] + (target[0] - source[0]) * alpha,
                source[1] + (target[1] - source[1]) * alpha,
            )
            if not self.disk_is_clear(point, radius):
                return False
        return True


def sample_waypoints(
    grid: OccupancyImage,
    seed: Point,
    bounds: Tuple[float, float, float, float],
    spacing: float,
    clearance: float,
) -> List[Point]:
    """Sample conservative office-floor waypoints and preserve the start seed."""
    points = [seed]
    min_x, max_x, min_y, max_y = bounds
    y = min_y
    while y <= max_y + 1e-9:
        x = min_x
        while x <= max_x + 1e-9:
            point = (round(x, 4), round(y, 4))
            if point != seed and grid.disk_is_clear(point, clearance):
                points.append(point)
            x += spacing
        y += spacing
    if not grid.disk_is_clear(seed, clearance):
        raise ValueError(f'Office GBSAE seed is not clear: {seed}.')
    return points


def visibility_edges(
    grid: OccupancyImage,
    points: Sequence[Point],
    max_distance: float,
    clearance: float,
) -> List[Edge]:
    """Connect nearby waypoints only when a clearance disk can traverse the edge."""
    edges = []
    for source, source_point in enumerate(points):
        for target in range(source + 1, len(points)):
            target_point = points[target]
            if math.dist(source_point, target_point) > max_distance:
                continue
            if grid.segment_is_clear(source_point, target_point, clearance):
                edges.append((source, target))
    return edges


def connected_seed_component(
    points: Sequence[Point],
    edges: Iterable[Edge],
) -> Tuple[List[Point], List[Edge]]:
    """Retain and stably reindex only the component containing seed index zero."""
    edges = list(edges)
    adjacency: Dict[int, List[int]] = {index: [] for index in range(len(points))}
    for source, target in edges:
        adjacency[source].append(target)
        adjacency[target].append(source)

    included = {0}
    queue = deque([0])
    while queue:
        for target in adjacency[queue.popleft()]:
            if target not in included:
                included.add(target)
                queue.append(target)

    old_ids = sorted(included)
    new_id = {old_id: index for index, old_id in enumerate(old_ids)}
    component_points = [points[old_id] for old_id in old_ids]
    component_edges = [
        (new_id[source], new_id[target])
        for source, target in edges
        if source in included and target in included
    ]
    if len(component_points) < 2:
        raise ValueError('Office GBSAE seed component needs at least two nodes.')
    return component_points, component_edges


def payload(world: str, points: Sequence[Point], edges: Sequence[Edge]) -> dict:
    return {
        'world': world,
        'nodes': [
            {'id': index, 'x': point[0], 'y': point[1]}
            for index, point in enumerate(points)
        ],
        'edges': [list(edge) for edge in edges],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('input_png', type=Path)
    parser.add_argument('output_json', type=Path)
    parser.add_argument('--world', default='slam_office')
    parser.add_argument('--resolution', type=float, default=0.05)
    parser.add_argument('--origin-x', type=float, default=-30.0)
    parser.add_argument('--origin-y', type=float, default=-30.0)
    parser.add_argument('--occupied-threshold', type=int, default=76)
    parser.add_argument('--seed-x', type=float, default=12.1)
    parser.add_argument('--seed-y', type=float, default=1.5)
    parser.add_argument('--min-x', type=float, default=-26.0)
    parser.add_argument('--max-x', type=float, default=20.0)
    parser.add_argument('--min-y', type=float, default=1.5)
    parser.add_argument('--max-y', type=float, default=21.5)
    parser.add_argument('--spacing', type=float, default=2.5)
    parser.add_argument('--node-clearance', type=float, default=0.32)
    parser.add_argument('--edge-clearance', type=float, default=0.24)
    parser.add_argument('--edge-max-distance', type=float, default=3.8)
    args = parser.parse_args()

    grid = OccupancyImage(
        args.input_png,
        args.resolution,
        args.origin_x,
        args.origin_y,
        args.occupied_threshold,
    )
    points = sample_waypoints(
        grid,
        (args.seed_x, args.seed_y),
        (args.min_x, args.max_x, args.min_y, args.max_y),
        args.spacing,
        args.node_clearance,
    )
    edges = visibility_edges(grid, points, args.edge_max_distance, args.edge_clearance)
    points, edges = connected_seed_component(points, edges)
    args.output_json.write_text(json.dumps(payload(args.world, points, edges), indent=2) + '\n')
    print(
        f'Generated {args.output_json}: {len(points)} nodes, {len(edges)} edges, '
        f'seed=({args.seed_x:.2f}, {args.seed_y:.2f})'
    )


if __name__ == '__main__':
    main()
