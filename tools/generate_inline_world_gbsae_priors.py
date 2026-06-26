#!/usr/bin/env python3
"""Generate GBSAE prior graphs from inline Gazebo box-collision worlds.

The small `slam_*` benchmark worlds describe navigable structure as static SDF
box collisions. This tool samples conservative waypoints inside configured
free-space envelopes, removes points and edges that collide with those boxes,
and writes connected topo-metric GBSAE prior assets.
"""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Point = Tuple[float, float]
Edge = Tuple[int, int]
Rect = Tuple[float, float, float, float]


@dataclass(frozen=True)
class OrientedBox:
    """2D footprint of an SDF collision box in world coordinates."""

    x: float
    y: float
    yaw: float
    size_x: float
    size_y: float

    def contains(self, point: Point, clearance: float) -> bool:
        dx = point[0] - self.x
        dy = point[1] - self.y
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        return (
            abs(local_x) <= self.size_x / 2.0 + clearance
            and abs(local_y) <= self.size_y / 2.0 + clearance
        )


@dataclass(frozen=True)
class WorldConfig:
    """Sampling envelope and fixed anchors for one world-specific prior."""

    domains: Tuple[Rect, ...]
    anchors: Tuple[Point, ...] = ()
    spacing: float = 1.8
    edge_max_distance: float = 3.0
    node_clearance: float = 0.32
    edge_clearance: float = 0.24
    sample_margin: float = 0.35


WORLD_CONFIGS: Dict[str, WorldConfig] = {
    'slam_landmarks': WorldConfig(
        domains=((-5.4, 5.4, -4.4, 4.4),),
        anchors=((-2.0, -0.5), (-4.4, 3.8), (4.4, 3.8), (-4.4, -3.8), (4.4, -3.8)),
        spacing=1.8,
        edge_max_distance=3.0,
    ),
    'slam_loop': WorldConfig(
        domains=((-5.1, 5.1, -3.6, 3.6),),
        anchors=(
            (-2.0, -0.5), (-4.3, 0.0), (-4.3, 2.6), (-1.8, 2.6),
            (0.2, 2.6), (2.5, 2.6), (4.7, 2.6), (4.7, 0.0),
            (4.7, -2.6), (2.5, -2.6), (0.2, -2.6), (-1.8, -2.6),
            (-4.3, -2.6),
        ),
        spacing=1.7,
        edge_max_distance=3.0,
    ),
    'slam_rooms': WorldConfig(
        domains=((-5.4, 5.4, -4.4, 4.4),),
        anchors=(
            (-2.0, -0.5), (-4.0, 0.0), (-4.0, 3.0), (-4.0, -3.0),
            (0.0, 0.0), (0.0, 3.0), (0.0, -3.0),
            (4.0, 0.0), (4.0, 3.0), (4.0, -3.0),
        ),
        spacing=1.8,
        edge_max_distance=3.0,
    ),
    'slam_rooms_corridor': WorldConfig(
        domains=(
            (-5.4, 6.0, -4.4, 4.4),
            (5.0, 14.4, -0.6, 0.6),
            (13.6, 17.5, -1.8, 1.8),
        ),
        anchors=(
            (-2.0, -0.5), (-4.0, 0.0), (-4.0, 3.0), (-4.0, -3.0),
            (0.0, 0.0), (0.0, 3.0), (0.0, -3.0),
            (4.0, 0.0), (4.0, 3.0), (4.0, -3.0),
            (6.4, 0.0), (8.4, 0.0), (10.4, 0.0), (12.4, 0.0),
            (14.4, 0.0), (16.0, 0.0), (16.0, 1.5), (16.0, -1.5),
        ),
        spacing=1.8,
        edge_max_distance=3.0,
    ),
}


def _float_tuple(text: str) -> Tuple[float, ...]:
    return tuple(float(item) for item in text.split())


def load_collision_boxes(path: Path) -> List[OrientedBox]:
    """Parse inline SDF collision boxes into 2D obstacle footprints."""
    root = ET.parse(path).getroot()
    boxes: List[OrientedBox] = []
    for link in root.findall('.//link'):
        link_pose = _float_tuple(link.findtext('pose', default='0 0 0 0 0 0'))
        link_x, link_y, link_yaw = link_pose[0], link_pose[1], link_pose[5]
        for collision in link.findall('collision'):
            size_node = collision.find('./geometry/box/size')
            if size_node is None or not size_node.text:
                continue
            collision_pose = _float_tuple(collision.findtext('pose', default='0 0 0 0 0 0'))
            local_x, local_y, local_yaw = collision_pose[0], collision_pose[1], collision_pose[5]
            cos_yaw = math.cos(link_yaw)
            sin_yaw = math.sin(link_yaw)
            size = _float_tuple(size_node.text)
            boxes.append(
                OrientedBox(
                    x=link_x + cos_yaw * local_x - sin_yaw * local_y,
                    y=link_y + sin_yaw * local_x + cos_yaw * local_y,
                    yaw=link_yaw + local_yaw,
                    size_x=size[0],
                    size_y=size[1],
                )
            )
    if not boxes:
        raise ValueError(f'No inline collision boxes found in {path}.')
    return boxes


def point_in_domains(point: Point, domains: Sequence[Rect], margin: float) -> bool:
    x, y = point
    return any(
        min_x + margin <= x <= max_x - margin
        and min_y + margin <= y <= max_y - margin
        for min_x, max_x, min_y, max_y in domains
    )


def point_is_clear(point: Point, boxes: Sequence[OrientedBox], clearance: float) -> bool:
    return not any(box.contains(point, clearance) for box in boxes)


def segment_is_clear(
    source: Point,
    target: Point,
    boxes: Sequence[OrientedBox],
    clearance: float,
    domains: Sequence[Rect],
    sample_margin: float,
) -> bool:
    distance = math.dist(source, target)
    count = max(1, math.ceil(distance / 0.05))
    for index in range(count + 1):
        alpha = index / count
        point = (
            source[0] + (target[0] - source[0]) * alpha,
            source[1] + (target[1] - source[1]) * alpha,
        )
        if not point_in_domains(point, domains, sample_margin):
            return False
        if not point_is_clear(point, boxes, clearance):
            return False
    return True


def sample_points(config: WorldConfig, boxes: Sequence[OrientedBox]) -> List[Point]:
    """Create deterministic clear waypoints from anchors plus a regular grid."""
    points: List[Point] = []
    seen = set()

    def add(point: Point) -> None:
        rounded = (round(point[0], 3), round(point[1], 3))
        if rounded in seen:
            return
        if not point_in_domains(rounded, config.domains, config.sample_margin):
            return
        if not point_is_clear(rounded, boxes, config.node_clearance):
            return
        seen.add(rounded)
        points.append(rounded)

    for anchor in config.anchors:
        add(anchor)

    for min_x, max_x, min_y, max_y in config.domains:
        y = min_y + config.sample_margin
        while y <= max_y - config.sample_margin + 1e-9:
            x = min_x + config.sample_margin
            while x <= max_x - config.sample_margin + 1e-9:
                add((x, y))
                x += config.spacing
            y += config.spacing

    if len(points) < 2:
        raise ValueError('GBSAE prior needs at least two clear sampled points.')
    return points


def visibility_edges(
    points: Sequence[Point],
    boxes: Sequence[OrientedBox],
    config: WorldConfig,
) -> List[Edge]:
    """Connect nearby waypoints only when the robot footprint can traverse them."""
    edges: List[Edge] = []
    for source, source_point in enumerate(points):
        for target in range(source + 1, len(points)):
            target_point = points[target]
            if math.dist(source_point, target_point) > config.edge_max_distance:
                continue
            if segment_is_clear(
                source_point,
                target_point,
                boxes,
                config.edge_clearance,
                config.domains,
                config.sample_margin,
            ):
                edges.append((source, target))
    return edges


def connected_seed_component(
    points: Sequence[Point],
    edges: Iterable[Edge],
) -> Tuple[List[Point], List[Edge]]:
    """Keep and reindex only the component connected to the first waypoint."""
    edges = list(edges)
    adjacency: Dict[int, List[int]] = {index: [] for index in range(len(points))}
    for source, target in edges:
        adjacency[source].append(target)
        adjacency[target].append(source)

    included = {0}
    queue = deque([0])
    while queue:
        source = queue.popleft()
        for target in adjacency[source]:
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
        raise ValueError('GBSAE seed component needs at least two nodes.')
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


def generate_prior(world: str, maps_dir: Path) -> dict:
    if world not in WORLD_CONFIGS:
        raise ValueError(f'No GBSAE generator config for world={world!r}.')
    config = WORLD_CONFIGS[world]
    boxes = load_collision_boxes(maps_dir / f'{world}.world')
    points = sample_points(config, boxes)
    edges = visibility_edges(points, boxes, config)
    points, edges = connected_seed_component(points, edges)
    return payload(world, points, edges)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--maps-dir',
        type=Path,
        default=Path('src/activeslam_resource/maps'),
        help='Directory containing slam_*.world files and receiving .gbsae.json outputs.',
    )
    parser.add_argument(
        '--world',
        action='append',
        choices=sorted(WORLD_CONFIGS),
        help='World basename to generate. Repeatable. Defaults to all configured worlds.',
    )
    args = parser.parse_args()

    maps_dir = args.maps_dir
    worlds = args.world or sorted(WORLD_CONFIGS)
    for world in worlds:
        graph = generate_prior(world, maps_dir)
        output_path = maps_dir / f'{world}.gbsae.json'
        output_path.write_text(json.dumps(graph, indent=2) + '\n')
        print(
            f'Generated {output_path}: '
            f'{len(graph["nodes"])} nodes, {len(graph["edges"])} edges'
        )


if __name__ == '__main__':
    main()
