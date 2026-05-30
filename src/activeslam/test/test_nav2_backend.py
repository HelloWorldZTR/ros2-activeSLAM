from types import SimpleNamespace

import pytest

from activeslam.nav2_backend import (
    GenerationGuard,
    configure_drive_on_heading_goal,
    heading_to_target,
    path_length,
    path_to_xy,
)


def _pose(x, y):
    return SimpleNamespace(
        pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y)),
    )


def test_path_to_xy_and_length_use_nav_path_positions():
    path = SimpleNamespace(poses=[_pose(0.0, 0.0), _pose(3.0, 4.0), _pose(6.0, 8.0)])

    points = path_to_xy(path)

    assert points == [(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)]
    assert path_length(points) == 10.0


def test_heading_to_target_faces_frontier_centroid():
    assert heading_to_target((1.0, 1.0), (1.0, 2.0)) == pytest.approx(1.5707963)


def test_generation_guard_rejects_late_callbacks():
    guard = GenerationGuard()
    old_generation = guard.advance()
    current_generation = guard.advance()

    assert not guard.is_current(old_generation)
    assert guard.is_current(current_generation)


def test_drive_on_heading_goal_uses_forward_distance_speed_and_timeout():
    duration = object()
    goal = SimpleNamespace(target=SimpleNamespace(x=0.0), speed=0.0, time_allowance=None)

    configure_drive_on_heading_goal(goal, distance=0.4, speed=0.08, time_allowance=duration)

    assert goal.target.x == pytest.approx(0.4)
    assert goal.speed == pytest.approx(0.08)
    assert goal.time_allowance is duration
