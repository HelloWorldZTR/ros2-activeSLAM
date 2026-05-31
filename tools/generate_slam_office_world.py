#!/usr/bin/env python3
"""Generate an IoU-friendly Gazebo office world from the upstream map PNG.

The source image is distributed by:
https://github.com/mlherd/Dataset-of-Gazebo-Worlds-Models-and-Maps

The original ServiceSim world contains mesh collisions and model includes that
the lightweight evaluator intentionally does not expand. This generator turns
the supplied occupancy map into inline box collisions so Gazebo and evaluator
use the same two-dimensional obstacle geometry.
"""

import argparse
import struct
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Rectangle = Tuple[int, int, int, int]


def decode_rgb_png(path: Path) -> Tuple[int, int, bytes]:
    """Decode a non-interlaced 8-bit RGB PNG using only the standard library."""
    data = path.read_bytes()
    signature = b'\x89PNG\r\n\x1a\n'
    if not data.startswith(signature):
        raise ValueError(f'Not a PNG file: {path}')

    offset = len(signature)
    width = height = None
    compressed = bytearray()
    while offset < len(data):
        length = struct.unpack('>I', data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        offset += 12 + length
        if kind == b'IHDR':
            width, height, bit_depth, color_type, compression, filtering, interlace = (
                struct.unpack('>IIBBBBB', payload)
            )
            if (bit_depth, color_type, compression, filtering, interlace) != (
                8, 2, 0, 0, 0
            ):
                raise ValueError('Expected a non-interlaced 8-bit RGB PNG')
        elif kind == b'IDAT':
            compressed.extend(payload)
        elif kind == b'IEND':
            break

    if width is None or height is None:
        raise ValueError('PNG is missing an IHDR chunk')

    bytes_per_pixel = 3
    stride = width * bytes_per_pixel
    raw = zlib.decompress(bytes(compressed))
    rows = []
    previous = bytearray(stride)
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        encoded = raw[cursor:cursor + stride]
        cursor += stride
        decoded = bytearray(stride)
        for index, value in enumerate(encoded):
            left = decoded[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            above = previous[index]
            upper_left = (
                previous[index - bytes_per_pixel]
                if index >= bytes_per_pixel else 0
            )
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                predictor = paeth(left, above, upper_left)
            else:
                raise ValueError(f'Unsupported PNG filter type: {filter_type}')
            decoded[index] = (value + predictor) & 0xff
        rows.append(decoded)
        previous = decoded
    return width, height, b''.join(rows)


def paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    distances = (
        abs(estimate - left),
        abs(estimate - above),
        abs(estimate - upper_left),
    )
    return (left, above, upper_left)[distances.index(min(distances))]


def occupied_rows(
    width: int,
    height: int,
    pixels: bytes,
    occupied_threshold: int,
) -> Iterable[List[Tuple[int, int]]]:
    """Yield inclusive horizontal occupied runs for each image row."""
    for row in range(height):
        runs = []
        start = None
        for col in range(width):
            offset = (row * width + col) * 3
            red, green, blue = pixels[offset:offset + 3]
            occupied = max(red, green, blue) <= occupied_threshold
            if occupied and start is None:
                start = col
            elif not occupied and start is not None:
                runs.append((start, col - 1))
                start = None
        if start is not None:
            runs.append((start, width - 1))
        yield runs


def merge_rectangles(rows: Iterable[Sequence[Tuple[int, int]]]) -> List[Rectangle]:
    """Merge identical horizontal runs on adjacent image rows."""
    active: Dict[Tuple[int, int], int] = {}
    rectangles: List[Rectangle] = []
    row_index = 0
    for row_index, runs in enumerate(rows):
        current = set(runs)
        for run, start_row in list(active.items()):
            if run not in current:
                rectangles.append((run[0], run[1], start_row, row_index - 1))
                del active[run]
        for run in runs:
            active.setdefault(run, row_index)
    for run, start_row in active.items():
        rectangles.append((run[0], run[1], start_row, row_index))
    return rectangles


def world_text(
    rectangles: Sequence[Rectangle],
    image_height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> str:
    """Render a static Gazebo Classic world with inline box collisions."""
    links = []
    for index, (min_col, max_col, min_row, max_row) in enumerate(rectangles):
        size_x = (max_col - min_col + 1) * resolution
        size_y = (max_row - min_row + 1) * resolution
        x = origin_x + (min_col + max_col + 1) * resolution / 2.0
        # PNG rows grow downward while the ROS map y axis grows upward.
        y = origin_y + (
            2 * image_height - min_row - max_row - 1
        ) * resolution / 2.0
        links.append(f'''      <link name="wall_{index}">
        <pose>{x:.4f} {y:.4f} 0.4 0 0 0</pose>
        <collision name="collision">
          <geometry><box><size>{size_x:.4f} {size_y:.4f} 0.8</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{size_x:.4f} {size_y:.4f} 0.8</size></box></geometry>
          <material>
            <ambient>0.62 0.66 0.72 1</ambient>
            <diffuse>0.62 0.66 0.72 1</diffuse>
          </material>
        </visual>
      </link>''')

    return f'''<?xml version="1.0" ?>
<!--
  Generated by tools/generate_slam_office_world.py from the office map in:
  https://github.com/mlherd/Dataset-of-Gazebo-Worlds-Models-and-Maps

  The occupancy image is converted to inline box collisions so slam_evaluator
  can compute coverage and IoU without recursively parsing model:// includes.
-->
<sdf version="1.6">
  <world name="default">
    <gui>
      <camera name="user_camera">
        <pose>0 -18 30 0 0.9 1.5708</pose>
      </camera>
    </gui>
    <scene>
      <ambient>0.65 0.65 0.65 1</ambient>
      <background>0.55 0.58 0.62 1</background>
      <shadows>1</shadows>
      <grid>0</grid>
      <origin_visual>0</origin_visual>
    </scene>
    <include><uri>model://ground_plane</uri></include>
    <include><uri>model://sun</uri></include>
    <model name="slam_office_obstacles">
      <static>true</static>
{chr(10).join(links)}
    </model>
  </world>
</sdf>
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('input_png', type=Path)
    parser.add_argument('output_world', type=Path)
    parser.add_argument('--resolution', type=float, default=0.05)
    parser.add_argument('--origin-x', type=float, default=-30.0)
    parser.add_argument('--origin-y', type=float, default=-30.0)
    parser.add_argument('--occupied-threshold', type=int, default=76)
    args = parser.parse_args()

    width, height, pixels = decode_rgb_png(args.input_png)
    rectangles = merge_rectangles(
        occupied_rows(width, height, pixels, args.occupied_threshold)
    )
    args.output_world.write_text(
        world_text(rectangles, height, args.resolution, args.origin_x, args.origin_y)
    )
    print(
        f'Generated {args.output_world}: {width}x{height} pixels, '
        f'{len(rectangles)} obstacle boxes'
    )


if __name__ == '__main__':
    main()
