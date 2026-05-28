#!/usr/bin/env python3
"""Plot active SLAM experiment results from a slam_evaluator run directory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Iterable


def read_csv_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = math.nan
            rows.append(parsed)
        return rows


def read_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def resolve_run_dir(dirname: Path) -> Path:
    dirname = dirname.expanduser().resolve()
    if (dirname / "coverage_time.csv").exists():
        return dirname

    candidates = sorted(
        (path for path in dirname.glob("run_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"{dirname} is not a run directory and contains no run_* directories"
    )


def finite_xy(rows: Iterable[dict[str, float]], x_key: str, y_key: str):
    xs = []
    ys = []
    for row in rows:
        x = row.get(x_key, math.nan)
        y = row.get(y_key, math.nan)
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return xs, ys


def compute_ate_series(
    estimated: list[dict[str, float]],
    ground_truth: list[dict[str, float]],
    max_dt: float,
) -> tuple[list[float], list[float], float | None]:
    """Match estimated and ground-truth poses by nearest timestamp."""
    gt = [
        row
        for row in ground_truth
        if all(math.isfinite(row.get(key, math.nan)) for key in ("time_sec", "x", "y"))
    ]
    est = [
        row
        for row in estimated
        if all(math.isfinite(row.get(key, math.nan)) for key in ("time_sec", "x", "y"))
    ]
    if not gt or not est:
        return [], [], None

    gt.sort(key=lambda row: row["time_sec"])
    gt_times = [row["time_sec"] for row in gt]
    times = []
    errors = []

    for row in sorted(est, key=lambda item: item["time_sec"]):
        index = bisect_left(gt_times, row["time_sec"])
        best = None
        best_dt = None
        for candidate_index in (index - 1, index):
            if candidate_index < 0 or candidate_index >= len(gt):
                continue
            candidate = gt[candidate_index]
            dt = abs(candidate["time_sec"] - row["time_sec"])
            if best_dt is None or dt < best_dt:
                best = candidate
                best_dt = dt

        if best is None or best_dt is None or best_dt > max_dt:
            continue

        times.append(row["time_sec"])
        errors.append(math.hypot(row["x"] - best["x"], row["y"] - best["y"]))

    if not errors:
        return [], [], None

    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return times, errors, rmse


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on host packages.
        raise RuntimeError("matplotlib is required to plot experiment results") from exc

    return plt


def plot_results(run_dir: Path, output_dir: Path, ate_max_dt: float) -> list[Path]:
    plt = require_matplotlib()

    coverage_time = read_csv_rows(run_dir / "coverage_time.csv")
    coverage_path = read_csv_rows(run_dir / "coverage_path.csv")
    trajectory_est = read_csv_rows(run_dir / "trajectory_est.csv")
    trajectory_gt = read_csv_rows(run_dir / "trajectory_gt.csv")
    metrics = read_metrics(run_dir / "metrics.json")
    ate_times, ate_values, ate_rmse = compute_ate_series(
        trajectory_est,
        trajectory_gt,
        ate_max_dt,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Active SLAM Experiment: {run_dir.name}")

    xs, ys = finite_xy(coverage_time, "time_sec", "coverage")
    axes[0][0].plot(xs, ys, color="tab:blue")
    axes[0][0].set_title("Coverage vs. Time")
    axes[0][0].set_xlabel("Time [s]")
    axes[0][0].set_ylabel("Coverage")
    axes[0][0].grid(True, alpha=0.3)

    xs, ys = finite_xy(coverage_path, "path_length", "coverage")
    axes[0][1].plot(xs, ys, color="tab:green")
    axes[0][1].set_title("Coverage vs. Path Length")
    axes[0][1].set_xlabel("Path Length [m]")
    axes[0][1].set_ylabel("Coverage")
    axes[0][1].grid(True, alpha=0.3)

    est_x, est_y = finite_xy(trajectory_est, "x", "y")
    gt_x, gt_y = finite_xy(trajectory_gt, "x", "y")
    if gt_x and gt_y:
        axes[1][0].plot(gt_x, gt_y, color="black", linewidth=1.5, label="Ground truth")
    if est_x and est_y:
        axes[1][0].plot(est_x, est_y, color="tab:orange", linewidth=1.2, label="Estimated")
    axes[1][0].set_title("Trajectory")
    axes[1][0].set_xlabel("x [m]")
    axes[1][0].set_ylabel("y [m]")
    axes[1][0].axis("equal")
    axes[1][0].grid(True, alpha=0.3)
    axes[1][0].legend()

    axes[1][1].plot(ate_times, ate_values, color="tab:red")
    axes[1][1].set_title("ATE Over Time")
    axes[1][1].set_xlabel("Time [s]")
    axes[1][1].set_ylabel("ATE [m]")
    axes[1][1].grid(True, alpha=0.3)

    final_coverage = metrics.get("final_coverage")
    total_path = metrics.get("total_path_length")
    total_time = metrics.get("total_time")
    summary_lines = []
    if final_coverage is not None:
        summary_lines.append(f"coverage={final_coverage:.3f}")
    if total_path is not None:
        summary_lines.append(f"path={total_path:.2f} m")
    if total_time is not None:
        summary_lines.append(f"time={total_time:.1f} s")
    if ate_rmse is not None:
        summary_lines.append(f"ATE RMSE={ate_rmse:.3f} m")
    if summary_lines:
        fig.text(0.5, 0.01, " | ".join(summary_lines), ha="center")

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    summary_path = output_dir / "experiment_summary.png"
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)
    written.append(summary_path)

    plot_single_metric(
        plt,
        output_dir / "coverage_time.png",
        coverage_time,
        "time_sec",
        "coverage",
        "Coverage vs. Time",
        "Time [s]",
        "Coverage",
        "tab:blue",
    )
    written.append(output_dir / "coverage_time.png")

    plot_single_metric(
        plt,
        output_dir / "coverage_path.png",
        coverage_path,
        "path_length",
        "coverage",
        "Coverage vs. Path Length",
        "Path Length [m]",
        "Coverage",
        "tab:green",
    )
    written.append(output_dir / "coverage_path.png")

    if ate_times and ate_values:
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.plot(ate_times, ate_values, color="tab:red")
        axis.set_title("ATE Over Time")
        axis.set_xlabel("Time [s]")
        axis.set_ylabel("ATE [m]")
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        ate_path = output_dir / "ate_time.png"
        fig.savefig(ate_path, dpi=150)
        plt.close(fig)
        written.append(ate_path)

    metrics_out = dict(metrics)
    if ate_rmse is not None:
        metrics_out["recomputed_ate_rmse"] = ate_rmse
        metrics_out["recomputed_ate_samples"] = len(ate_values)
    summary_json = output_dir / "plot_metrics.json"
    summary_json.write_text(json.dumps(metrics_out, indent=2) + "\n")
    written.append(summary_json)

    return written


def plot_single_metric(
    plt,
    path: Path,
    rows: list[dict[str, float]],
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
) -> None:
    xs, ys = finite_xy(rows, x_key, y_key)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(xs, ys, color=color)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot slam_evaluator CSV results. Pass either one run_* directory "
            "or a log root containing run_* directories."
        )
    )
    parser.add_argument("dirname", type=Path, help="Experiment run directory or log root")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to <run_dir>/plots.",
    )
    parser.add_argument(
        "--ate-max-dt",
        type=float,
        default=0.5,
        help="Maximum timestamp difference for ATE matching in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = resolve_run_dir(args.dirname)
    output_dir = args.output_dir or (run_dir / "plots")
    try:
        written = plot_results(run_dir, output_dir, args.ate_max_dt)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "Install matplotlib in the active Python environment, for example: "
            "python3 -m pip install matplotlib",
            file=sys.stderr,
        )
        return 1
    print(f"Plotted {run_dir}")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
