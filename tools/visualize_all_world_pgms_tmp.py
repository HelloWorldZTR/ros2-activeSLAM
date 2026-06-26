#!/usr/bin/env python3
"""Temporarily batch-render inferred world maps and GBSAE prior overlays."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPS_DIR = REPO_ROOT / "src" / "activeslam_resource" / "maps"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "products" / "world_pgm_all_tmp"
VISUALIZER = REPO_ROOT / "tools" / "visualize_world_pgm.py"


def discover_worlds(maps_dir: Path, gbsae_only: bool) -> list[Path]:
    worlds = sorted(maps_dir.glob("*.world"))
    if not gbsae_only:
        return worlds
    return [path for path in worlds if path.with_suffix(".gbsae.json").exists()]


def render_world(
    world_path: Path,
    maps_dir: Path,
    output_dir: Path,
    resolution: float,
    margin: float,
    dpi: int,
    label_prior_nodes: bool,
    no_prior: bool,
) -> dict[str, object]:
    world = world_path.stem
    command = [
        sys.executable,
        str(VISUALIZER),
        world,
        "--maps-dir",
        str(maps_dir),
        "--output-dir",
        str(output_dir),
        "--resolution",
        str(resolution),
        "--margin",
        str(margin),
        "--dpi",
        str(dpi),
    ]
    if label_prior_nodes:
        command.append("--label-prior-nodes")
    if no_prior:
        command.append("--no-prior")

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "world": world,
        "world_path": str(world_path),
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "outputs": [
            str(output_dir / f"{world}_inferred.pgm"),
            str(output_dir / f"{world}_inferred.yaml"),
            str(output_dir / f"{world}_inferred.png"),
            str(output_dir / f"{world}_inferred.pdf"),
        ]
        if result.returncode == 0
        else [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-render inferred PGM visualizations for all map worlds."
    )
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=DEFAULT_MAPS_DIR,
        help="Directory containing .world files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory receiving all generated PGM/YAML/PNG/PDF files.",
    )
    parser.add_argument(
        "--gbsae-only",
        action="store_true",
        help="Only render worlds that have a matching .gbsae.json prior.",
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
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--label-prior-nodes",
        action="store_true",
        help="Annotate GBSAE node ids in every generated overlay.",
    )
    parser.add_argument(
        "--no-prior",
        action="store_true",
        help="Render maps without GBSAE prior overlays.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first world that cannot be rendered.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a failure exit code when any world is skipped.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    maps_dir = args.maps_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    worlds = discover_worlds(maps_dir, args.gbsae_only)
    if not worlds:
        print(f"error: no .world files found under {maps_dir}", file=sys.stderr)
        return 2

    results = []
    for index, world_path in enumerate(worlds, start=1):
        print(f"[{index}/{len(worlds)}] {world_path.stem}")
        result = render_world(
            world_path,
            maps_dir,
            output_dir,
            args.resolution,
            args.margin,
            args.dpi,
            args.label_prior_nodes,
            args.no_prior,
        )
        results.append(result)
        if result["ok"]:
            print("  ok")
        else:
            print(f"  skipped: {result['stderr'] or result['stdout']}")
            if args.fail_fast:
                break

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")

    ok_count = sum(1 for result in results if result["ok"])
    skipped_count = len(results) - ok_count
    print(f"Rendered {ok_count}/{len(results)} worlds; skipped {skipped_count}.")
    print(f"Summary: {summary_path}")
    if args.strict and skipped_count:
        return 1
    return 0 if ok_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
