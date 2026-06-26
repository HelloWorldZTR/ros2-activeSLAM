#!/usr/bin/env python3
"""Generate and visualize an inferred occupancy PGM from an inline Gazebo world.

This helper is intended for the project benchmark worlds whose static geometry
is encoded as SDF box collisions. It writes a ROS-map-style ``.pgm`` and
``.yaml`` pair, and can overlay the matching GBSAE prior graph on a paper-style
PNG/PDF visualization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVESLAM_SRC = REPO_ROOT / "src" / "activeslam"
if str(ACTIVESLAM_SRC) not in sys.path:
    sys.path.insert(0, str(ACTIVESLAM_SRC))

from activeslam.slam_evaluator_utils import (  # noqa: E402
    derive_bounds_from_obstacles,
    extract_box_obstacles,
    rasterize_obstacles,
)


TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 17
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 12.5

PGM_FREE = 254
PGM_OCCUPIED = 0


@dataclass(frozen=True)
class InferredMap:
    world: str
    occupied: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int


def require_matplotlib():
    try:
        cache_dir = Path(tempfile.gettempdir()) / "activeslam_matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - host package dependent.
        raise RuntimeError("matplotlib is required to visualize the inferred map") from exc

    return plt


def world_title(world: str) -> str:
    return world.replace("_", " ").title()


def configure_axis(axis, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=12)
    axis.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    axis.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    axis.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    axis.grid(True, alpha=0.25, linewidth=0.8)


def save_figure(fig, output_base: Path, dpi: int) -> list[Path]:
    written = []
    for suffix in (".png", ".pdf"):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    return written


def resolve_world_path(world_arg: str, maps_dir: Path) -> tuple[str, Path]:
    candidate = Path(world_arg).expanduser()
    if candidate.suffix == ".world" or candidate.exists():
        world_path = candidate.resolve()
        world = world_path.stem
    else:
        world = world_arg.removesuffix(".world")
        world_path = (maps_dir / f"{world}.world").expanduser().resolve()

    if not world_path.exists():
        raise FileNotFoundError(f"world file not found: {world_path}")
    return world, world_path


def aligned_bounds(
    raw_bounds: tuple[float, float, float, float],
    resolution: float,
) -> tuple[float, float, float, float]:
    min_x, max_x, min_y, max_y = raw_bounds
    return (
        math.floor(min_x / resolution) * resolution,
        math.ceil(max_x / resolution) * resolution,
        math.floor(min_y / resolution) * resolution,
        math.ceil(max_y / resolution) * resolution,
    )


def infer_map(world: str, world_path: Path, resolution: float, margin: float) -> InferredMap:
    """Rasterize inline SDF collision boxes into a free/occupied occupancy image."""
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")

    obstacles = extract_box_obstacles(str(world_path))
    bounds = derive_bounds_from_obstacles(obstacles, margin=margin)
    if bounds is None:
        raise ValueError(f"no inline box collision geometry found in {world_path}")

    min_x, max_x, min_y, max_y = aligned_bounds(bounds, resolution)
    width = int(math.ceil((max_x - min_x) / resolution))
    height = int(math.ceil((max_y - min_y) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid inferred map bounds for {world_path}")

    occupied = rasterize_obstacles(
        obstacles,
        width=width,
        height=height,
        resolution=resolution,
        origin_x=min_x,
        origin_y=min_y,
    )
    return InferredMap(
        world=world,
        occupied=occupied,
        resolution=resolution,
        origin_x=min_x,
        origin_y=min_y,
        width=width,
        height=height,
    )


def pgm_image(map_data: InferredMap) -> np.ndarray:
    values = np.where(map_data.occupied, PGM_OCCUPIED, PGM_FREE).astype(np.uint8)
    return np.flipud(values)


def write_pgm(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(image.tobytes())


def write_yaml(path: Path, pgm_path: Path, map_data: InferredMap) -> None:
    path.write_text(
        "\n".join(
            [
                f"image: {pgm_path.name}",
                f"resolution: {map_data.resolution:.6f}",
                f"origin: [{map_data.origin_x:.6f}, {map_data.origin_y:.6f}, 0.000000]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.196",
                "",
            ]
        )
    )


def resolve_prior_path(world: str, maps_dir: Path, prior_arg: str | None, no_prior: bool) -> Path | None:
    if no_prior:
        return None
    if prior_arg:
        return Path(prior_arg).expanduser().resolve()
    candidate = (maps_dir / f"{world}.gbsae.json").expanduser().resolve()
    return candidate if candidate.exists() else None


def load_prior(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"GBSAE prior not found: {path}")
    with path.open() as handle:
        return json.load(handle)


def prior_nodes(prior: dict) -> dict[int, tuple[float, float]]:
    nodes = {}
    for index, node in enumerate(prior.get("nodes", [])):
        node_id = int(node.get("id", index))
        nodes[node_id] = (float(node["x"]), float(node["y"]))
    return nodes


def draw_prior(axis, prior: dict, label_nodes: bool) -> None:
    nodes = prior_nodes(prior)
    edge_plotted = False
    for edge in prior.get("edges", []):
        if len(edge) != 2:
            continue
        source, target = int(edge[0]), int(edge[1])
        if source not in nodes or target not in nodes:
            continue
        source_xy = nodes[source]
        target_xy = nodes[target]
        axis.plot(
            [source_xy[0], target_xy[0]],
            [source_xy[1], target_xy[1]],
            color="#54A24B",
            linewidth=2.1,
            alpha=0.88,
            zorder=3,
            label="GBSAE prior edges" if not edge_plotted else None,
        )
        edge_plotted = True

    if nodes:
        xs = [point[0] for point in nodes.values()]
        ys = [point[1] for point in nodes.values()]
        axis.scatter(
            xs,
            ys,
            marker="o",
            s=34,
            color="#54A24B",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
            label="GBSAE prior nodes",
        )

    if label_nodes:
        for node_id, (x, y) in nodes.items():
            axis.text(
                x,
                y,
                str(node_id),
                fontsize=8,
                color="#1B5E20",
                ha="center",
                va="bottom",
                zorder=5,
            )


def plot_map(
    plt,
    map_data: InferredMap,
    image: np.ndarray,
    prior: dict | None,
    output_base: Path,
    dpi: int,
    label_prior_nodes: bool,
) -> list[Path]:
    fig, axis = plt.subplots(figsize=(9.2, 8.0))
    extent = [
        map_data.origin_x,
        map_data.origin_x + map_data.width * map_data.resolution,
        map_data.origin_y,
        map_data.origin_y + map_data.height * map_data.resolution,
    ]
    axis.imshow(
        image / 255.0,
        cmap="gray",
        extent=extent,
        origin="upper",
        interpolation="nearest",
        zorder=0,
    )

    if prior is not None:
        draw_prior(axis, prior, label_prior_nodes)

    configure_axis(
        axis,
        f"{world_title(map_data.world)}: Inferred World Map",
        "x [m]",
        "y [m]",
    )
    axis.axis("equal")
    if prior is not None:
        axis.legend(fontsize=LEGEND_FONTSIZE, loc="best", frameon=True, framealpha=0.95)
    fig.tight_layout()

    written = save_figure(fig, output_base, dpi)
    plt.close(fig)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer a ROS-style occupancy PGM from inline Gazebo world boxes and "
            "visualize it, optionally with a GBSAE prior overlay."
        )
    )
    parser.add_argument(
        "world",
        help="World basename, e.g. slam_rooms, or path to a .world file.",
    )
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=REPO_ROOT / "src" / "activeslam_resource" / "maps",
        help="Directory containing <world>.world and optional <world>.gbsae.json files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "products" / "world_pgm",
        help="Output directory for the inferred PGM/YAML and visualization.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        help="Output map resolution [m/cell].",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Free-space margin around world collision bounds [m].",
    )
    parser.add_argument(
        "--prior",
        help="Explicit GBSAE prior JSON path. Defaults to <maps-dir>/<world>.gbsae.json.",
    )
    parser.add_argument(
        "--no-prior",
        action="store_true",
        help="Disable automatic GBSAE prior overlay.",
    )
    parser.add_argument(
        "--label-prior-nodes",
        action="store_true",
        help="Annotate GBSAE prior node ids on the visualization.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output DPI for PNG figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    maps_dir = args.maps_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        world, world_path = resolve_world_path(args.world, maps_dir)
        map_data = infer_map(world, world_path, args.resolution, args.margin)
        image = pgm_image(map_data)
        prior_path = resolve_prior_path(world, maps_dir, args.prior, args.no_prior)
        prior = load_prior(prior_path)
        plt = require_matplotlib()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if prior is not None and prior.get("world") not in (None, world):
        print(
            f"warning: prior world={prior.get('world')!r} does not match map world={world!r}",
            file=sys.stderr,
        )

    pgm_path = output_dir / f"{world}_inferred.pgm"
    yaml_path = output_dir / f"{world}_inferred.yaml"
    figure_base = output_dir / f"{world}_inferred"

    write_pgm(pgm_path, image)
    write_yaml(yaml_path, pgm_path, map_data)
    written = [pgm_path, yaml_path]
    written.extend(
        plot_map(
            plt,
            map_data,
            image,
            prior,
            figure_base,
            args.dpi,
            args.label_prior_nodes,
        )
    )

    print(f"World: {world_path}")
    print(
        f"Map: {map_data.width}x{map_data.height} cells, "
        f"{map_data.resolution:.3f} m/cell, origin=({map_data.origin_x:.3f}, {map_data.origin_y:.3f})"
    )
    if prior_path is not None:
        print(f"GBSAE prior: {prior_path}")
    else:
        print("GBSAE prior: none")
    print(f"Output directory: {output_dir}")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
