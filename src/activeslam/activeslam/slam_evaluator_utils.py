import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


Bounds = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Pose2D:
    timestamp: float
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class BoxObstacle:
    x: float
    y: float
    z: float
    yaw: float
    size_x: float
    size_y: float
    size_z: float


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def parse_pose_text(text: Optional[str]) -> Tuple[float, float, float, float]:
    if not text:
        return 0.0, 0.0, 0.0, 0.0
    values = [float(v) for v in text.split()]
    values.extend([0.0] * (6 - len(values)))
    return values[0], values[1], values[2], values[5]


def compose_pose_2d(
    parent: Tuple[float, float, float, float],
    child: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    px, py, pz, pyaw = parent
    cx, cy, cz, cyaw = child
    cos_yaw = math.cos(pyaw)
    sin_yaw = math.sin(pyaw)
    return (
        px + cos_yaw * cx - sin_yaw * cy,
        py + sin_yaw * cx + cos_yaw * cy,
        pz + cz,
        pyaw + cyaw,
    )


def extract_box_obstacles(world_path: str) -> List[BoxObstacle]:
    tree = ET.parse(world_path)
    root = tree.getroot()
    obstacles: List[BoxObstacle] = []

    for model in root.findall('.//model'):
        model_pose = parse_pose_text(_child_text(model, 'pose'))
        for link in model.findall('link'):
            link_pose = compose_pose_2d(
                model_pose,
                parse_pose_text(_child_text(link, 'pose')),
            )
            for collision in link.findall('collision'):
                size_text = _child_text(collision, 'geometry/box/size')
                if not size_text:
                    continue
                size = [float(v) for v in size_text.split()]
                if len(size) != 3:
                    continue
                pose = compose_pose_2d(
                    link_pose,
                    parse_pose_text(_child_text(collision, 'pose')),
                )
                obstacles.append(
                    BoxObstacle(
                        x=pose[0],
                        y=pose[1],
                        z=pose[2],
                        yaw=pose[3],
                        size_x=size[0],
                        size_y=size[1],
                        size_z=size[2],
                    )
                )
    return obstacles


def derive_bounds_from_obstacles(
    obstacles: Sequence[BoxObstacle],
    margin: float,
) -> Optional[Bounds]:
    if not obstacles:
        return None

    xs = []
    ys = []
    for obstacle in obstacles:
        for x, y in box_corners(obstacle):
            xs.append(x)
            ys.append(y)

    return (
        min(xs) - margin,
        max(xs) + margin,
        min(ys) - margin,
        max(ys) + margin,
    )


def box_corners(obstacle: BoxObstacle) -> List[Tuple[float, float]]:
    half_x = obstacle.size_x / 2.0
    half_y = obstacle.size_y / 2.0
    cos_yaw = math.cos(obstacle.yaw)
    sin_yaw = math.sin(obstacle.yaw)
    corners = []
    for lx, ly in (
        (-half_x, -half_y),
        (-half_x, half_y),
        (half_x, -half_y),
        (half_x, half_y),
    ):
        corners.append((
            obstacle.x + cos_yaw * lx - sin_yaw * ly,
            obstacle.y + sin_yaw * lx + cos_yaw * ly,
        ))
    return corners


def compute_coverage(
    grid_data: Iterable[int],
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    bounds: Optional[Bounds],
) -> Tuple[float, int, int]:
    data = np.array(list(grid_data), dtype=np.int16).reshape(height, width)
    mask = evaluation_mask(width, height, resolution, origin_x, origin_y, bounds)
    total = (
        _fixed_bounds_cell_count(bounds, resolution)
        if bounds is not None
        else int(np.count_nonzero(mask))
    )
    if total == 0:
        return 0.0, 0, 0
    known = int(np.count_nonzero((data != -1) & mask))
    return known / total, known, total


def _fixed_bounds_cell_count(bounds: Bounds, resolution: float) -> int:
    """Return the full evaluation-region cell count independent of map extent."""
    if resolution <= 0.0:
        return 0
    min_x, max_x, min_y, max_y = bounds
    width = max(0, int(math.ceil((max_x - min_x) / resolution)))
    height = max(0, int(math.ceil((max_y - min_y) / resolution)))
    return width * height


def accumulate_path_length(
    previous: Optional[Tuple[float, float]],
    current: Tuple[float, float],
    total: float,
) -> Tuple[float, Tuple[float, float]]:
    if previous is None:
        return total, current
    segment = math.hypot(current[0] - previous[0], current[1] - previous[1])
    return total + segment, current


def rasterize_obstacles(
    obstacles: Sequence[BoxObstacle],
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> np.ndarray:
    occupied = np.zeros((height, width), dtype=bool)

    for obstacle in obstacles:
        corners = box_corners(obstacle)
        min_x = min(x for x, _ in corners)
        max_x = max(x for x, _ in corners)
        min_y = min(y for _, y in corners)
        max_y = max(y for _, y in corners)

        min_col = max(0, int(math.floor((min_x - origin_x) / resolution)) - 1)
        max_col = min(width - 1, int(math.ceil((max_x - origin_x) / resolution)) + 1)
        min_row = max(0, int(math.floor((min_y - origin_y) / resolution)) - 1)
        max_row = min(height - 1, int(math.ceil((max_y - origin_y) / resolution)) + 1)
        if min_col > max_col or min_row > max_row:
            continue

        cols = np.arange(min_col, max_col + 1)
        rows = np.arange(min_row, max_row + 1)
        xs = origin_x + (cols + 0.5) * resolution
        ys = origin_y + (rows + 0.5) * resolution
        grid_x, grid_y = np.meshgrid(xs, ys)

        dx = grid_x - obstacle.x
        dy = grid_y - obstacle.y
        cos_yaw = math.cos(obstacle.yaw)
        sin_yaw = math.sin(obstacle.yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy

        half_x = obstacle.size_x / 2.0 + 1e-9
        half_y = obstacle.size_y / 2.0 + 1e-9
        mask = (np.abs(local_x) <= half_x) & (np.abs(local_y) <= half_y)
        occupied[min_row:max_row + 1, min_col:max_col + 1] |= mask

    return occupied


def evaluation_mask(
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    bounds: Optional[Bounds],
) -> np.ndarray:
    if bounds is None:
        return np.ones((height, width), dtype=bool)

    min_x, max_x, min_y, max_y = bounds
    cols = np.arange(width)
    rows = np.arange(height)
    xs = origin_x + (cols + 0.5) * resolution
    ys = origin_y + (rows + 0.5) * resolution
    x_mask = (xs >= min_x) & (xs <= max_x)
    y_mask = (ys >= min_y) & (ys <= max_y)
    return np.outer(y_mask, x_mask)


def compute_map_iou(
    pred_data: Iterable[int],
    gt_occupied: np.ndarray,
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    bounds: Optional[Bounds],
) -> Tuple[Optional[float], Optional[float]]:
    data = np.array(list(pred_data), dtype=np.int16).reshape(height, width)
    eval_area = evaluation_mask(width, height, resolution, origin_x, origin_y, bounds)
    known = (data != -1) & eval_area
    pred_occupied = (data >= 50) & known
    pred_free = (data >= 0) & (data < 50) & known
    gt_occupied = gt_occupied & eval_area
    gt_free = (~gt_occupied) & eval_area

    occupied_iou = _masked_iou(pred_occupied, gt_occupied, known)
    free_iou = _masked_iou(pred_free, gt_free, known)
    return occupied_iou, free_iou


def compute_ate(
    estimated: Sequence[Pose2D],
    ground_truth: Sequence[Pose2D],
    max_dt: float = 0.2,
) -> Tuple[Optional[float], List[Tuple[float, float]]]:
    matches = match_trajectories(estimated, ground_truth, max_dt)
    if not matches:
        return None, []

    first_est, first_gt = matches[0]
    offset_x = first_gt.x - first_est.x
    offset_y = first_gt.y - first_est.y

    errors = []
    for est, gt in matches:
        error = math.hypot(est.x + offset_x - gt.x, est.y + offset_y - gt.y)
        errors.append((est.timestamp, error))

    rmse = math.sqrt(sum(error * error for _, error in errors) / len(errors))
    return rmse, errors


def match_trajectories(
    estimated: Sequence[Pose2D],
    ground_truth: Sequence[Pose2D],
    max_dt: float,
) -> List[Tuple[Pose2D, Pose2D]]:
    if not estimated or not ground_truth:
        return []

    gt_sorted = sorted(ground_truth, key=lambda p: p.timestamp)
    matches = []
    start_index = 0

    for est in sorted(estimated, key=lambda p: p.timestamp):
        best_index = None
        best_dt = None
        while (
            start_index + 1 < len(gt_sorted)
            and gt_sorted[start_index + 1].timestamp <= est.timestamp
        ):
            start_index += 1

        for candidate_index in (start_index, start_index + 1):
            if candidate_index >= len(gt_sorted):
                continue
            dt = abs(gt_sorted[candidate_index].timestamp - est.timestamp)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_index = candidate_index

        if best_index is not None and best_dt is not None and best_dt <= max_dt:
            matches.append((est, gt_sorted[best_index]))

    return matches


def _child_text(element: ET.Element, path: str) -> Optional[str]:
    child = element.find(path)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def _masked_iou(
    predicted: np.ndarray,
    truth: np.ndarray,
    valid_mask: np.ndarray,
) -> Optional[float]:
    intersection = int(np.count_nonzero(predicted & truth & valid_mask))
    union = int(np.count_nonzero((predicted | truth) & valid_mask))
    if union == 0:
        return None
    return intersection / union
