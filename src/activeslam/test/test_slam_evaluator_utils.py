from pathlib import Path

import numpy as np
import pytest

from activeslam.slam_evaluator_utils import (
    Pose2D,
    accumulate_path_length,
    compute_ate,
    compute_coverage,
    compute_map_iou,
    derive_bounds_from_obstacles,
    extract_box_obstacles,
    rasterize_obstacles,
)


def test_compute_coverage_counts_known_cells_inside_bounds():
    data = [-1, 0, 100, 0, -1, 100]

    coverage, known, total = compute_coverage(
        data,
        width=3,
        height=2,
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
        bounds=(0.0, 3.0, 0.0, 2.0),
    )

    assert coverage == 4 / 6
    assert known == 4
    assert total == 6


def test_compute_coverage_uses_fixed_bounds_denominator():
    data = [0, 100]

    coverage, known, total = compute_coverage(
        data,
        width=2,
        height=1,
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
        bounds=(0.0, 4.0, 0.0, 1.0),
    )

    assert coverage == 2 / 4
    assert known == 2
    assert total == 4


def test_accumulate_path_length_uses_consecutive_positions():
    total, previous = accumulate_path_length(None, (0.0, 0.0), 0.0)
    total, previous = accumulate_path_length(previous, (3.0, 4.0), total)
    total, previous = accumulate_path_length(previous, (6.0, 8.0), total)

    assert total == 10.0
    assert previous == (6.0, 8.0)


def test_extract_and_rasterize_inline_sdf_box_obstacles():
    repo_src = Path(__file__).resolve().parents[2]
    world_path = repo_src / 'activeslam_resource' / 'maps' / 'slam_loop.world'

    obstacles = extract_box_obstacles(str(world_path))
    bounds = derive_bounds_from_obstacles(obstacles, margin=0.5)
    raster = rasterize_obstacles(
        obstacles,
        width=240,
        height=200,
        resolution=0.05,
        origin_x=-6.0,
        origin_y=-5.0,
    )

    assert len(obstacles) > 0
    assert bounds is not None
    assert np.count_nonzero(raster) > 0


def test_slam_office_uses_inline_box_obstacles_for_iou():
    repo_src = Path(__file__).resolve().parents[2]
    world_path = repo_src / 'activeslam_resource' / 'maps' / 'slam_office.world'

    obstacles = extract_box_obstacles(str(world_path))
    bounds = derive_bounds_from_obstacles(obstacles, margin=0.5)

    assert len(obstacles) == 374
    assert bounds == pytest.approx((-28.15, 21.75, -0.6, 23.1))


def test_compute_map_iou_excludes_unknown_cells():
    pred = [100, 0, -1, 100]
    gt_occupied = np.array([[True, False], [True, True]])

    occupied_iou, free_iou = compute_map_iou(
        pred,
        gt_occupied,
        width=2,
        height=2,
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
        bounds=None,
    )

    assert occupied_iou == 1.0
    assert free_iou == 1.0


def test_compute_ate_matches_by_timestamp_and_aligns_first_pose():
    estimated = [
        Pose2D(0.0, 0.0, 0.0, 0.0),
        Pose2D(1.0, 1.0, 0.0, 0.0),
    ]
    ground_truth = [
        Pose2D(0.05, 10.0, 10.0, 0.0),
        Pose2D(1.05, 11.0, 10.0, 0.0),
    ]

    rmse, errors = compute_ate(estimated, ground_truth, max_dt=0.2)

    assert rmse == 0.0
    assert errors == [(0.0, 0.0), (1.0, 0.0)]
