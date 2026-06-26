#!/usr/bin/env python3
"""Generate paper-ready comparison figures from active SLAM experiment results.

The script scans a result root containing directories named like
``run_<world>_<method>_<timestamp>`` and writes one coverage comparison and one
multi-method trajectory comparison for every detected world.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


METHOD_ORDER = [
    "frontier",
    "approx_graph",
    "gbsae",
    "gvd_gbsae",
    "gvd_hierarchical",
]
METHOD_LABELS = {
    "frontier": "Frontier",
    "approx_graph": "Approx. Graph",
    "gbsae": "GBSAE",
    "gvd_gbsae": "GVD-GBSAE",
    "gvd_hierarchical": "GVD-Hierarchical (Ours)",
}
METHOD_COLORS = {
    "frontier": "#4C78A8",
    "approx_graph": "#F58518",
    "gbsae": "#54A24B",
    "gvd_gbsae": "#B279A2",
    "gvd_hierarchical": "#D62728",
}
METHOD_LINESTYLES = {
    "frontier": "-",
    "approx_graph": "--",
    "gbsae": "-.",
    "gvd_gbsae": ":",
    "gvd_hierarchical": "-",
}
RUN_RE = re.compile(
    r"^run_(?P<world>.+)_(?P<method>"
    + "|".join(re.escape(method) for method in sorted(METHOD_ORDER, key=len, reverse=True))
    + r")_(?P<timestamp>\d{8}_\d{6})$"
)


@dataclass(frozen=True)
class RunData:
    world: str
    method: str
    timestamp: str
    root_dir: Path
    run_dir: Path
    metrics: dict


def require_matplotlib():
    try:
        cache_dir = Path(tempfile.gettempdir()) / "activeslam_matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - host package dependent.
        raise RuntimeError("matplotlib is required to generate paper figures") from exc

    return plt


def read_csv_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []

    rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = math.nan
            rows.append(parsed)
    return rows


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def finite_xy(
    rows: Iterable[dict[str, float]], x_key: str, y_key: str
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = row.get(x_key, math.nan)
        y = row.get(y_key, math.nan)
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return xs, ys


def resolve_latest_run_dir(root_dir: Path) -> Path | None:
    if (root_dir / "coverage_time.csv").exists():
        return root_dir
    candidates = sorted(
        (path for path in root_dir.glob("run_*") if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def discover_runs(results_root: Path) -> list[RunData]:
    runs: list[RunData] = []
    for root_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        match = RUN_RE.match(root_dir.name)
        if not match:
            continue

        run_dir = resolve_latest_run_dir(root_dir)
        if run_dir is None:
            continue
        if not (run_dir / "coverage_time.csv").exists() and not (
            run_dir / "trajectory_gt.csv"
        ).exists():
            continue

        runs.append(
            RunData(
                world=match.group("world"),
                method=match.group("method"),
                timestamp=match.group("timestamp"),
                root_dir=root_dir,
                run_dir=run_dir,
                metrics=read_json(run_dir / "metrics.json"),
            )
        )
    return runs


def latest_per_world_method(runs: Iterable[RunData]) -> dict[str, dict[str, RunData]]:
    selected: dict[str, dict[str, RunData]] = {}
    for run in runs:
        world_runs = selected.setdefault(run.world, {})
        previous = world_runs.get(run.method)
        if previous is None or run.timestamp > previous.timestamp:
            world_runs[run.method] = run
    return selected


def method_sort_key(method: str) -> tuple[int, str]:
    try:
        return METHOD_ORDER.index(method), method
    except ValueError:
        return len(METHOD_ORDER), method


def world_title(world: str) -> str:
    return world.replace("_", " ").title()


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def configure_axis(axis, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, fontsize=20, fontweight="bold", pad=12)
    axis.set_xlabel(xlabel, fontsize=17)
    axis.set_ylabel(ylabel, fontsize=17)
    axis.tick_params(axis="both", labelsize=14)
    axis.grid(True, alpha=0.25, linewidth=0.8)


def save_figure(fig, output_base: Path, dpi: int) -> list[Path]:
    written = []
    for suffix in (".png", ".pdf"):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    return written


def plot_coverage_comparison(
    plt,
    world: str,
    runs_by_method: dict[str, RunData],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    fig, axis = plt.subplots(figsize=(10.5, 6.4))
    plotted = False

    for method in sorted(runs_by_method, key=method_sort_key):
        run = runs_by_method[method]
        rows = read_csv_rows(run.run_dir / "coverage_time.csv")
        xs, ys = finite_xy(rows, "time_sec", "coverage")
        if not xs:
            continue
        label = method_label(method)
        if method == "gvd_hierarchical":
            label += f"  final={ys[-1] * 100:.1f}%"
        axis.plot(
            xs,
            [value * 100.0 for value in ys],
            color=METHOD_COLORS.get(method),
            linestyle=METHOD_LINESTYLES.get(method, "-"),
            linewidth=3.2 if method == "gvd_hierarchical" else 2.3,
            label=label,
            alpha=1.0 if method == "gvd_hierarchical" else 0.92,
        )
        plotted = True

    configure_axis(
        axis,
        f"{world_title(world)}: Exploration Coverage",
        "Time [s]",
        "Explored area [%]",
    )
    axis.set_ylim(bottom=0)
    axis.legend(fontsize=13, loc="lower right", frameon=True, framealpha=0.95)
    fig.tight_layout()

    written = save_figure(fig, output_dir / f"{world}_coverage_comparison", dpi)
    plt.close(fig)
    return written if plotted else []


def parse_simple_yaml(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            values[key] = [
                float(item.strip())
                for item in value[1:-1].split(",")
                if item.strip()
            ]
            continue
        try:
            values[key] = float(value)
        except ValueError:
            values[key] = value.strip("'\"")
    return values


def _read_pgm_token(handle) -> bytes:
    token = bytearray()
    while True:
        char = handle.read(1)
        if not char:
            return bytes(token)
        if char == b"#":
            handle.readline()
            continue
        if char.isspace():
            if token:
                return bytes(token)
            continue
        token.extend(char)


def read_pgm(path: Path) -> tuple[list[list[int]], int, int, int] | None:
    if not path.exists():
        return None

    with path.open("rb") as handle:
        magic = _read_pgm_token(handle)
        if magic not in (b"P2", b"P5"):
            return None
        width = int(_read_pgm_token(handle))
        height = int(_read_pgm_token(handle))
        max_value = int(_read_pgm_token(handle))
        if width <= 0 or height <= 0 or max_value <= 0:
            return None
        count = width * height
        if magic == b"P5":
            if max_value < 256:
                raw = handle.read(count)
                values = list(raw)
            else:
                raw = handle.read(count * 2)
                values = list(struct.unpack(f">{count}H", raw))
        else:
            values = [int(_read_pgm_token(handle)) for _ in range(count)]
    rows = [values[index : index + width] for index in range(0, count, width)]
    return rows, width, height, max_value


def choose_background_run(runs_by_method: dict[str, RunData]) -> RunData | None:
    if "gvd_hierarchical" in runs_by_method:
        return runs_by_method["gvd_hierarchical"]
    if not runs_by_method:
        return None
    return max(
        runs_by_method.values(),
        key=lambda run: float(run.metrics.get("final_coverage", -1.0)),
    )


def add_map_background(axis, run: RunData | None) -> None:
    if run is None:
        return
    yaml_data = parse_simple_yaml(run.run_dir / "final_map.yaml")
    image_name = yaml_data.get("image")
    resolution = yaml_data.get("resolution")
    origin = yaml_data.get("origin")
    if not isinstance(image_name, str) or not isinstance(resolution, float):
        return
    if not isinstance(origin, list) or len(origin) < 2:
        return

    pgm_path = (run.run_dir / image_name).resolve()
    pgm = read_pgm(pgm_path)
    if pgm is None:
        return
    rows, width, height, max_value = pgm
    image = [[value / max_value for value in row] for row in rows]
    x0 = float(origin[0])
    y0 = float(origin[1])
    extent = [x0, x0 + width * resolution, y0, y0 + height * resolution]
    axis.imshow(
        image,
        cmap="gray",
        extent=extent,
        origin="upper",
        alpha=0.28,
        interpolation="nearest",
        zorder=0,
    )


def plot_trajectory_comparison(
    plt,
    world: str,
    runs_by_method: dict[str, RunData],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    fig, axis = plt.subplots(figsize=(9.2, 8.0))
    add_map_background(axis, choose_background_run(runs_by_method))
    plotted = False

    for method in sorted(runs_by_method, key=method_sort_key):
        run = runs_by_method[method]
        rows = read_csv_rows(run.run_dir / "trajectory_gt.csv")
        xs, ys = finite_xy(rows, "x", "y")
        if not xs:
            rows = read_csv_rows(run.run_dir / "trajectory_est.csv")
            xs, ys = finite_xy(rows, "x", "y")
        if not xs:
            continue

        linewidth = 3.0 if method == "gvd_hierarchical" else 2.0
        axis.plot(
            xs,
            ys,
            color=METHOD_COLORS.get(method),
            linestyle=METHOD_LINESTYLES.get(method, "-"),
            linewidth=linewidth,
            label=method_label(method),
            alpha=1.0 if method == "gvd_hierarchical" else 0.85,
            zorder=3 if method == "gvd_hierarchical" else 2,
        )
        axis.scatter(
            [xs[0]],
            [ys[0]],
            marker="o",
            s=42,
            color=METHOD_COLORS.get(method),
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        axis.scatter(
            [xs[-1]],
            [ys[-1]],
            marker="X",
            s=76,
            color=METHOD_COLORS.get(method),
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        plotted = True

    configure_axis(
        axis,
        f"{world_title(world)}: Trajectory Comparison",
        "x [m]",
        "y [m]",
    )
    axis.axis("equal")
    axis.legend(fontsize=12.5, loc="best", frameon=True, framealpha=0.95)
    fig.tight_layout()

    written = save_figure(fig, output_dir / f"{world}_trajectory_comparison", dpi)
    plt.close(fig)
    return written if plotted else []


def write_summary_csv(
    worlds: dict[str, dict[str, RunData]], output_dir: Path
) -> Path:
    path = output_dir / "paper_figure_runs.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "world",
                "method",
                "label",
                "timestamp",
                "run_dir",
                "final_coverage",
                "total_path_length",
                "total_time",
                "ate_rmse",
                "occupied_iou",
                "free_iou",
            ],
        )
        writer.writeheader()
        for world in sorted(worlds):
            for method in sorted(worlds[world], key=method_sort_key):
                run = worlds[world][method]
                writer.writerow(
                    {
                        "world": world,
                        "method": method,
                        "label": method_label(method),
                        "timestamp": run.timestamp,
                        "run_dir": run.run_dir,
                        "final_coverage": run.metrics.get("final_coverage", ""),
                        "total_path_length": run.metrics.get("total_path_length", ""),
                        "total_time": run.metrics.get("total_time", ""),
                        "ate_rmse": run.metrics.get("ate_rmse", ""),
                        "occupied_iou": run.metrics.get("occupied_iou", ""),
                        "free_iou": run.metrics.get("free_iou", ""),
                    }
                )
    return path


def parse_args() -> argparse.Namespace:
    default_results = Path(__file__).resolve().parent / " results"
    parser = argparse.ArgumentParser(
        description="Generate per-world paper figures from active SLAM results."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_results,
        help=f"Directory containing run_<world>_<method>_<timestamp> folders. "
        f"Default: {default_results}",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Figure output directory. Default: <results-root>/paper_figures",
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
    results_root = args.results_root.expanduser().resolve()
    output_dir = (args.output_dir or (results_root / "paper_figures")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plt = require_matplotlib()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    runs = discover_runs(results_root)
    worlds = latest_per_world_method(runs)
    if not worlds:
        print(f"error: no experiment runs found under {results_root}", file=sys.stderr)
        return 2

    written: list[Path] = []
    for world in sorted(worlds):
        written.extend(plot_coverage_comparison(plt, world, worlds[world], output_dir, args.dpi))
        written.extend(plot_trajectory_comparison(plt, world, worlds[world], output_dir, args.dpi))
    written.append(write_summary_csv(worlds, output_dir))

    print(f"Found {sum(len(methods) for methods in worlds.values())} runs in {len(worlds)} worlds.")
    print(f"Output directory: {output_dir}")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
