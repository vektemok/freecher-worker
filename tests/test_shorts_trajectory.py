"""Crop trajectory validation and FFmpeg delivery.

Regression coverage for the dynamic-crop failure: FFmpeg's expression evaluator caps a single
expression at roughly one hundred parsed nodes, so a 30-second short sampled at 5 fps (~155
keyframes) made `crop` fail with "Failed to configure input pad ... Invalid argument".
"""

import math
import shutil
import subprocess

import pytest

from freecher_worker.crop.expression import (
    build_ffmpeg_crop_expression,
    fit_points_to_expression_budget,
)
from freecher_worker.crop.models import CropPoint, CropTrajectory
from freecher_worker.shorts.trajectory import (
    CROP_DRIVER_EXPRESSION,
    CROP_DRIVER_SENDCMD,
    CROP_DRIVER_STATIC,
    EXPRESSION_KEYFRAME_BUDGET,
    TrajectoryValidationError,
    build_sendcmd_script,
    plan_crop_driver,
    static_center_driver,
    validate_and_sanitize_trajectory,
)

FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg is not installed")

SOURCE_W, SOURCE_H, CROP_W, CROP_H = 1920, 1080, 608, 1080
MAX_X = SOURCE_W - CROP_W


def wobbling_trajectory(n_points: int, fps: float = 5.0) -> CropTrajectory:
    """A realistic, non-linear camera move that trajectory simplification cannot collapse."""
    points = []
    for i in range(n_points):
        x = 600 + 350 * math.sin(i * 0.31) + 90 * math.sin(i * 1.7)
        x = (min(max(int(round(x)), 0), MAX_X) // 2) * 2
        points.append(
            CropPoint(
                time=round(i / fps, 3), center_x=x + CROP_W / 2, center_y=SOURCE_H / 2,
                crop_x=x, crop_y=0, crop_w=CROP_W, crop_h=CROP_H, subject_type="face",
            )
        )
    return CropTrajectory(
        source_width=SOURCE_W, source_height=SOURCE_H, crop_w=CROP_W, crop_h=CROP_H, points=points
    )


def run_crop_filter(chain: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
         "-i", f"color=c=black:s={SOURCE_W}x{SOURCE_H}:d=0.2", "-vf", chain, "-f", "null", "-"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_clean_trajectory_passes_validation_unchanged():
    sanitized, report = validate_and_sanitize_trajectory(wobbling_trajectory(50))

    assert report.valid
    assert report.point_count_in == report.point_count_out == 50
    assert report.non_finite_dropped == 0
    assert report.out_of_range_clamped == 0
    assert report.odd_coordinates_fixed == 0
    assert 0 <= report.min_x <= report.max_x <= report.max_x_allowed
    assert report.max_x_allowed == MAX_X
    assert "source=1920x1080" in report.summary()


def test_out_of_range_coordinates_are_clamped_into_the_crop_window():
    traj = wobbling_trajectory(10)
    traj.points[3].crop_x = 99999
    traj.points[5].crop_x = -500
    traj.points[7].crop_y = 4000

    sanitized, report = validate_and_sanitize_trajectory(traj)

    assert report.out_of_range_clamped == 3
    for point in sanitized.points:
        assert 0 <= point.crop_x <= MAX_X
        assert 0 <= point.crop_y <= SOURCE_H - CROP_H


def test_odd_coordinates_are_snapped_to_even():
    """H.264 / yuv420p requires even crop offsets."""
    traj = wobbling_trajectory(10)
    traj.points[2].crop_x = 301
    traj.points[4].crop_x = 405

    sanitized, report = validate_and_sanitize_trajectory(traj)

    assert report.odd_coordinates_fixed == 2
    assert all(p.crop_x % 2 == 0 and p.crop_y % 2 == 0 for p in sanitized.points)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_points_are_dropped(bad):
    traj = wobbling_trajectory(10)
    traj.points[4].center_x = bad

    sanitized, report = validate_and_sanitize_trajectory(traj)

    assert report.non_finite_dropped == 1
    assert report.point_count_out == 9
    assert all(math.isfinite(p.center_x) and math.isfinite(p.center_y) for p in sanitized.points)


def test_non_monotonic_timestamps_are_repaired():
    traj = wobbling_trajectory(10)
    traj.points[5].time = traj.points[4].time
    traj.points[6].time = 0.0

    sanitized, report = validate_and_sanitize_trajectory(traj)

    assert report.non_monotonic_times_fixed == 2
    times = [p.time for p in sanitized.points]
    assert times == sorted(times)
    assert len(set(times)) == len(times)


def test_crop_larger_than_source_is_rejected():
    traj = wobbling_trajectory(5)
    traj.crop_w = SOURCE_W + 100
    with pytest.raises(TrajectoryValidationError, match="exceeds source width"):
        validate_and_sanitize_trajectory(traj)


def test_odd_crop_size_is_rejected():
    traj = wobbling_trajectory(5)
    traj.crop_w = 607
    with pytest.raises(TrajectoryValidationError, match="not even"):
        validate_and_sanitize_trajectory(traj)


def test_trajectory_with_no_usable_points_is_rejected():
    traj = wobbling_trajectory(3)
    for point in traj.points:
        point.center_x = float("nan")
    with pytest.raises(TrajectoryValidationError, match="no finite"):
        validate_and_sanitize_trajectory(traj)


# ---------------------------------------------------------------------------
# Driver selection
# ---------------------------------------------------------------------------


def test_constant_trajectory_uses_a_static_crop():
    traj = wobbling_trajectory(40)
    for point in traj.points:
        point.crop_x = 400
    driver = plan_crop_driver(traj, sendcmd_available=True)

    assert driver.driver == CROP_DRIVER_STATIC
    assert driver.x_expr == "400"
    assert driver.script is None


def test_sendcmd_carries_every_keyframe():
    """The whole point of the fix: no tracking resolution is traded away."""
    traj = wobbling_trajectory(225)
    driver = plan_crop_driver(traj, sendcmd_available=True)

    assert driver.driver == CROP_DRIVER_SENDCMD
    assert driver.keyframes == 225
    assert driver.resolution_reduced is False
    assert driver.script and driver.script.count("crop x") >= 225


def test_expression_fallback_stays_within_the_ffmpeg_node_budget():
    traj = wobbling_trajectory(225)
    driver = plan_crop_driver(traj, sendcmd_available=False)

    assert driver.driver == CROP_DRIVER_EXPRESSION
    assert driver.keyframes <= EXPRESSION_KEYFRAME_BUDGET
    assert driver.keyframes_available == 225
    assert driver.resolution_reduced is True


def test_expression_budget_fitting_preserves_the_endpoints():
    points = wobbling_trajectory(300).points
    fitted, reduced = fit_points_to_expression_budget(points, 45)

    assert reduced is True
    assert len(fitted) <= 45
    assert fitted[0].time == points[0].time
    assert fitted[-1].time == points[-1].time
    assert [p.time for p in fitted] == sorted(p.time for p in fitted)


def test_small_trajectories_are_left_alone_by_budget_fitting():
    points = wobbling_trajectory(20).points
    fitted, reduced = fit_points_to_expression_budget(points, 45)
    assert reduced is False
    assert len(fitted) == 20


def test_static_center_driver_is_even_and_in_range():
    driver = static_center_driver(SOURCE_W, SOURCE_H, CROP_W, CROP_H)
    assert driver.driver == CROP_DRIVER_STATIC
    assert int(driver.x_expr) % 2 == 0
    assert 0 <= int(driver.x_expr) <= MAX_X


def test_sendcmd_script_is_clamped_and_even_on_every_interval():
    script = build_sendcmd_script(wobbling_trajectory(60))
    lines = [line for line in script.splitlines() if line.strip()]
    intervals = [line for line in lines if "-" in line.split(" ", 1)[0]]
    holds = [line for line in lines if line not in intervals]

    assert len(intervals) == 59  # one per keyframe gap
    for line in intervals:
        assert "2*trunc(" in line and "min(max(0," in line
        assert "in_w-out_w" in line or "in_h-out_h" in line

    # The trailing command holds the final value for any frame past the last keyframe.
    assert len(holds) == 1
    final_value = int(holds[0].split("'")[1])
    assert final_value % 2 == 0
    assert 0 <= final_value <= MAX_X


def test_sendcmd_script_omits_a_constant_axis():
    """Vertical framing is constant for a full-height 9:16 crop, so no y commands are emitted."""
    script = build_sendcmd_script(wobbling_trajectory(40))
    assert "crop x" in script
    assert "crop y" not in script


# ---------------------------------------------------------------------------
# Real FFmpeg regression
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_nested_if_expression_still_breaks_above_the_node_budget():
    """Documents the original defect so a regression to that form is caught immediately."""
    traj = wobbling_trajectory(155)
    expr = build_ffmpeg_crop_expression(traj, axis="x", escape_for_filter=True)
    result = run_crop_filter(f"crop={CROP_W}:{CROP_H}:{expr}:0")

    assert result.returncode != 0
    assert "crop" in result.stderr.lower()


@needs_ffmpeg
@pytest.mark.parametrize("n_points", [150, 190, 225])
def test_sendcmd_renders_a_full_resolution_trajectory(tmp_path, n_points):
    """150-225 keyframes, the range a 30-45 s short at 5 fps produces."""
    traj = wobbling_trajectory(n_points)
    driver = plan_crop_driver(traj, sendcmd_available=True)
    script = tmp_path / "cmds.txt"
    script.write_text(driver.script, encoding="utf-8")

    chain = (
        f"sendcmd=f={script},crop={CROP_W}:{CROP_H}:{driver.x_expr}:{driver.y_expr},"
        f"scale=1080:1920,setsar=1"
    )
    result = run_crop_filter(chain)
    assert result.returncode == 0, result.stderr


@needs_ffmpeg
def test_expression_fallback_actually_parses():
    traj = wobbling_trajectory(225)
    driver = plan_crop_driver(traj, sendcmd_available=False)
    result = run_crop_filter(f"crop={CROP_W}:{CROP_H}:{driver.x_expr}:{driver.y_expr}")
    assert result.returncode == 0, result.stderr


@needs_ffmpeg
def test_sendcmd_trajectory_produces_a_valid_1080x1920_mp4(tmp_path):
    """Full regression: 1920x1080 input, 40 s, 5 fps tracking, 200 keyframes -> vertical MP4."""
    source = tmp_path / "source.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc=size={SOURCE_W}x{SOURCE_H}:rate=25:duration=40",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=40",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(source)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )

    traj = wobbling_trajectory(200)
    assert 150 <= len(traj.points) <= 225
    driver = plan_crop_driver(traj, sendcmd_available=True)
    assert driver.driver == CROP_DRIVER_SENDCMD

    script = tmp_path / "cmds.txt"
    script.write_text(driver.script, encoding="utf-8")
    out = tmp_path / "vertical.mp4"

    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-t", "40", "-i", str(source),
         "-filter_complex",
         f"[0:v]sendcmd=f={script},crop={CROP_W}:{CROP_H}:{driver.x_expr}:{driver.y_expr},"
         f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[v];"
         f"[0:a]anull[a]",
         "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(out)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file() and out.stat().st_size > 0

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,codec_name", "-of", "csv=p=0", str(out)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True, check=True,
    )
    assert probe.stdout.strip() == "h264,1080,1920"
