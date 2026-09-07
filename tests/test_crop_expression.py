"""Unit tests for crop dimensions and FFmpeg expression generation."""

from freecher_worker.crop.expression import build_ffmpeg_crop_x_expression, simplify_trajectory_points
from freecher_worker.crop.models import CropPoint, CropTrajectory
from freecher_worker.crop.tracker import calculate_crop_dimensions


def test_calculate_crop_dimensions():
    """Verify even-integer 9:16 crop dimensions for landscape inputs."""
    # 1920x1080 -> 1080 * 9 / 16 = 607.5 -> 608
    w, h = calculate_crop_dimensions(1920, 1080)
    assert h == 1080
    assert w == 608
    assert w % 2 == 0
    assert h % 2 == 0

    # 1280x720 -> 720 * 9 / 16 = 405 -> 404 or 406
    w720, h720 = calculate_crop_dimensions(1280, 720)
    assert h720 == 720
    assert w720 % 2 == 0
    assert 400 <= w720 <= 410


def test_static_crop_expression():
    """If all trajectory points have the same crop_x, return a simple static number."""
    points = [
        CropPoint(time=0.0, center_x=960.0, center_y=540.0, crop_x=656, crop_y=0, crop_w=608, crop_h=1080, subject_type="center"),
        CropPoint(time=2.0, center_x=960.0, center_y=540.0, crop_x=656, crop_y=0, crop_w=608, crop_h=1080, subject_type="center"),
    ]
    traj = CropTrajectory(source_width=1920, source_height=1080, crop_w=608, crop_h=1080, points=points)

    expr = build_ffmpeg_crop_x_expression(traj)
    assert expr == "656"


def test_moving_crop_expression_syntax():
    """Verify piecewise-linear x(t) expression is generated for moving points."""
    points = [
        CropPoint(time=0.0, center_x=400.0, center_y=540.0, crop_x=96, crop_y=0, crop_w=608, crop_h=1080, subject_type="face"),
        CropPoint(time=2.0, center_x=800.0, center_y=540.0, crop_x=496, crop_y=0, crop_w=608, crop_h=1080, subject_type="face"),
        CropPoint(time=4.0, center_x=1200.0, center_y=540.0, crop_x=896, crop_y=0, crop_w=608, crop_h=1080, subject_type="face"),
    ]
    traj = CropTrajectory(source_width=1920, source_height=1080, crop_w=608, crop_h=1080, points=points)

    expr = build_ffmpeg_crop_x_expression(traj)
    assert "if(lte(t" in expr
    assert "in_w-out_w" in expr
    assert "trunc" in expr
