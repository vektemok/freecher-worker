"""Smart 9:16 reframing: tracking, subject arbitration, smoothing and fallbacks."""

from typing import List

import cv2
import numpy as np
import pytest

from freecher_worker.crop.expression import build_ffmpeg_crop_expression
from freecher_worker.crop.models import CropPoint, CropTrajectory, DetectedSubject
from freecher_worker.shorts.reframe import (
    FALLBACK_CENTER,
    REFRAME_MODE_CENTER,
    REFRAME_MODE_SMART,
    ReframeConfig,
    SubjectDetector,
    build_reframe_plan,
    calculate_vertical_crop,
    estimate_dominant_center,
    render_debug_overlay,
)

SOURCE_W, SOURCE_H = 1280, 720
FPS = 10


class ScriptedDetector(SubjectDetector):
    """Detector driven by a fixed timeline, so tracking behaviour is exactly reproducible.

    ``script`` maps a sample index to a list of (center_x, center_y, size) boxes in the
    coordinates of the frame it is handed (detection runs on a downscaled frame).
    """

    def __init__(self, script, frame_width: int = SOURCE_W, fps: float = 5.0):
        self.script = script
        self.frame_width = frame_width
        self.fps = fps
        self.calls = 0

    def detect(self, frame_bgr, timestamp: float) -> List[DetectedSubject]:
        index = self.calls
        self.calls += 1
        entries = self.script(index) if callable(self.script) else self.script.get(index, [])
        scale = frame_bgr.shape[1] / float(self.frame_width)
        subjects = []
        for cx, cy, size in entries:
            w = h = int(size * scale)
            x = int(cx * scale - w / 2)
            y = int(cy * scale - h / 2)
            subjects.append(
                DetectedSubject(
                    box=(x, y, w, h),
                    confidence=0.9,
                    subject_type="face",
                    area=float(w * h),
                    center_x=x + w / 2.0,
                    center_y=y + h / 2.0,
                )
            )
        return subjects


def write_video(path, seconds: float = 6.0, scene_cut_at: float = None) -> str:
    """Write a plain synthetic clip; detections come from ScriptedDetector, not from pixels."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SOURCE_W, SOURCE_H))
    total = int(seconds * FPS)
    for i in range(total):
        t = i / FPS
        dark = scene_cut_at is not None and t >= scene_cut_at
        frame = np.full((SOURCE_H, SOURCE_W, 3), 20 if dark else 200, dtype=np.uint8)
        cv2.rectangle(frame, (100 + i, 100), (200 + i, 200), (0, 0, 255) if dark else (255, 0, 0), -1)
        writer.write(frame)
    writer.release()
    return str(path)


@pytest.fixture
def clip(tmp_path):
    return write_video(tmp_path / "clip.mp4")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_vertical_crop_is_916_and_even():
    crop_w, crop_h = calculate_vertical_crop(1920, 1080)
    assert crop_h == 1080
    assert crop_w == 608  # 1080 * 9 / 16, rounded to an even integer
    assert crop_w % 2 == 0 and crop_h % 2 == 0
    assert abs((crop_w / crop_h) - (9 / 16)) < 0.005


def test_vertical_crop_fits_inside_narrow_sources():
    crop_w, crop_h = calculate_vertical_crop(480, 1080)
    assert crop_w <= 480 and crop_h <= 1080
    assert crop_w % 2 == 0 and crop_h % 2 == 0


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


def test_single_face_is_tracked_across_the_frame(clip):
    """One subject walking left to right: the crop must follow without ever losing them."""
    def script(i):
        return [(200 + i * 30, 300, 120)]

    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=ScriptedDetector(script),
        config=ReframeConfig(analysis_fps=5.0),
        source_width=SOURCE_W, source_height=SOURCE_H,
    )

    assert plan.mode == REFRAME_MODE_SMART
    # A single dropped sample is tolerated: timestamp seeking is only frame-accurate.
    assert plan.diagnostics.frames_with_detection >= plan.diagnostics.sampled_frames - 1
    assert plan.diagnostics.dominant_subject_switches == 0
    xs = [p.crop_x for p in plan.trajectory.points]
    assert xs[-1] > xs[0]
    assert plan.diagnostics.trajectory.crop_x_range > 50


def test_static_subject_produces_a_static_crop(clip):
    """A motionless subject must not make the camera drift."""
    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=ScriptedDetector(lambda i: [(400, 300, 140)]),
        config=ReframeConfig(analysis_fps=5.0),
        source_width=SOURCE_W, source_height=SOURCE_H,
    )
    xs = {p.crop_x for p in plan.trajectory.points}
    assert len(xs) == 1
    assert plan.diagnostics.trajectory.stationary_ratio == pytest.approx(1.0)


def test_two_faces_switch_slowly_and_only_once(clip):
    """The active subject changes when the other becomes dominant, without ping-ponging."""
    def script(i):
        # Left face shrinks away, right face grows: one clean handover, not a flicker.
        left = (250, 300, max(40, 170 - i * 10))
        right = (1050, 300, min(200, 50 + i * 10))
        return [left, right]

    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=ScriptedDetector(script),
        config=ReframeConfig(analysis_fps=5.0, switch_hold_sec=0.6, min_switch_interval_sec=1.0),
        source_width=SOURCE_W, source_height=SOURCE_H,
    )

    assert plan.diagnostics.max_simultaneous_subjects == 2
    assert plan.diagnostics.dominant_subject_switches <= 2
    xs = [p.crop_x for p in plan.trajectory.points]
    assert xs[-1] > xs[0]


def test_two_close_faces_are_framed_together(clip):
    """When both fit inside 9:16 the crop holds both instead of picking a winner."""
    def script(i):
        return [(600, 300, 110), (720, 300, 110)]

    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=ScriptedDetector(script),
        config=ReframeConfig(analysis_fps=5.0),
        source_width=SOURCE_W, source_height=SOURCE_H,
    )

    assert plan.diagnostics.dual_subject_frames > 0
    crop_w = plan.trajectory.crop_w
    for point in plan.trajectory.points[2:]:
        assert point.crop_x <= 600 - 55 + 5
        assert point.crop_x + crop_w >= 720 + 55 - 5


def test_alternating_faces_do_not_cause_constant_switching(clip):
    """Two comparable faces must not make the crop oscillate every sample."""
    def script(i):
        big, small = (170, 150) if i % 2 == 0 else (150, 170)
        return [(300, 300, big), (980, 300, small)]

    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=ScriptedDetector(script),
        config=ReframeConfig(analysis_fps=5.0),
        source_width=SOURCE_W, source_height=SOURCE_H,
    )
    assert plan.diagnostics.dominant_subject_switches <= 1


def test_temporary_detection_loss_holds_the_previous_crop(clip):
    """Losing the subject for a moment must not snap the camera back to center."""
    def script(i):
        return [] if 10 <= i <= 14 else [(950, 300, 140)]

    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=ScriptedDetector(script),
        config=ReframeConfig(analysis_fps=5.0, track_max_misses=2),
        source_width=SOURCE_W, source_height=SOURCE_H,
    )

    center_crop_x = (SOURCE_W - plan.trajectory.crop_w) // 2
    gap_points = [p for p in plan.trajectory.points if 2.0 <= p.time <= 2.8]
    assert gap_points
    for point in gap_points:
        assert abs(point.crop_x - center_crop_x) > 50
    assert plan.diagnostics.fallback_previous_frames > 0
    assert plan.diagnostics.fallback_used is True


def test_scene_cut_allows_an_immediate_re_anchor(tmp_path):
    """Across a hard cut the crop is allowed to jump instead of slowly panning."""
    path = write_video(tmp_path / "cut.mp4", seconds=6.0, scene_cut_at=3.0)

    def script(i):
        return [(250, 300, 140)] if i < 15 else [(1030, 300, 140)]

    plan = build_reframe_plan(
        path, 0.0, 6.0,
        detector=ScriptedDetector(script),
        config=ReframeConfig(analysis_fps=5.0),
        source_width=SOURCE_W, source_height=SOURCE_H,
    )

    assert plan.diagnostics.scene_cuts >= 1
    before = [p.crop_x for p in plan.trajectory.points if p.time < 2.9]
    after = [p.crop_x for p in plan.trajectory.points if p.time > 3.2]
    assert after and before
    assert max(after) - max(before) > 100


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def test_crop_never_exceeds_the_configured_pan_velocity(clip):
    """A subject teleporting across the frame must still produce operator-like motion."""
    def script(i):
        return [(150, 300, 140)] if i % 4 < 2 else [(1130, 300, 140)]

    config = ReframeConfig(analysis_fps=5.0, max_velocity_px_per_sec=120.0)
    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=ScriptedDetector(script),
        config=config,
        source_width=SOURCE_W, source_height=SOURCE_H,
    )

    points = plan.trajectory.points
    for prev, curr in zip(points, points[1:]):
        dt = max(1e-3, curr.time - prev.time)
        assert abs(curr.crop_x - prev.crop_x) / dt <= config.max_velocity_px_per_sec * 1.35


def test_micro_jitter_is_suppressed(clip):
    """Sub-pixel-scale detector noise must not reach the rendered crop."""
    def script(i):
        return [(640 + (2 if i % 2 else -2), 300, 140)]

    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=ScriptedDetector(script),
        config=ReframeConfig(analysis_fps=5.0),
        source_width=SOURCE_W, source_height=SOURCE_H,
    )
    assert plan.diagnostics.trajectory.crop_x_range == 0


def test_crop_stays_inside_the_source_frame(clip):
    """Even with subjects hard against the bezel the crop window must remain valid."""
    def script(i):
        return [(10, 300, 160)] if i < 15 else [(SOURCE_W - 10, 300, 160)]

    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=ScriptedDetector(script),
        config=ReframeConfig(analysis_fps=5.0),
        source_width=SOURCE_W, source_height=SOURCE_H,
    )
    traj = plan.trajectory
    for point in traj.points:
        assert 0 <= point.crop_x <= traj.source_width - traj.crop_w
        assert 0 <= point.crop_y <= traj.source_height - traj.crop_h
        assert point.crop_x % 2 == 0 and point.crop_y % 2 == 0


def test_reframing_is_deterministic(clip):
    def script(i):
        return [(200 + i * 25, 300, 130)]

    kwargs = dict(
        config=ReframeConfig(analysis_fps=5.0), source_width=SOURCE_W, source_height=SOURCE_H
    )
    first = build_reframe_plan(clip, 0.0, 6.0, detector=ScriptedDetector(script), **kwargs)
    second = build_reframe_plan(clip, 0.0, 6.0, detector=ScriptedDetector(script), **kwargs)
    assert [p.crop_x for p in first.trajectory.points] == [p.crop_x for p in second.trajectory.points]


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


def test_no_detections_at_all_still_yields_a_valid_trajectory(clip):
    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=ScriptedDetector(lambda i: []),
        config=ReframeConfig(analysis_fps=5.0),
        source_width=SOURCE_W, source_height=SOURCE_H,
    )
    traj = plan.trajectory
    assert traj.points
    assert traj.crop_w > 0 and traj.crop_h > 0
    assert plan.diagnostics.fallback_used is True
    assert (
        plan.diagnostics.fallback_dominant_frames + plan.diagnostics.fallback_center_frames
    ) > 0
    for point in traj.points:
        assert 0 <= point.crop_x <= traj.source_width - traj.crop_w


def test_detector_exceptions_do_not_break_the_render(clip):
    class ExplodingDetector(SubjectDetector):
        def detect(self, frame_bgr, timestamp):
            raise RuntimeError("model weights are on fire")

    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=ExplodingDetector(),
        config=ReframeConfig(analysis_fps=5.0),
        source_width=SOURCE_W, source_height=SOURCE_H,
    )
    assert plan.trajectory.points
    assert plan.diagnostics.fallback_used is True


def test_unreadable_video_falls_back_to_static_center_crop(tmp_path):
    missing = tmp_path / "does_not_exist.mp4"
    plan = build_reframe_plan(
        missing, 0.0, 6.0,
        config=ReframeConfig(analysis_fps=5.0),
        source_width=SOURCE_W, source_height=SOURCE_H,
    )
    assert plan.mode == REFRAME_MODE_CENTER
    assert plan.diagnostics.fallback_used is True
    expected_x = ((SOURCE_W - plan.trajectory.crop_w) // 4) * 2
    assert all(p.crop_x == expected_x for p in plan.trajectory.points)
    assert all(p.subject_type == FALLBACK_CENTER for p in plan.trajectory.points)


def test_dominant_region_estimation_finds_the_detailed_side():
    frame = np.zeros((360, 640), dtype=np.uint8)
    for x in range(430, 530, 6):  # dense vertical texture on the right
        frame[:, x : x + 3] = 255
    center = estimate_dominant_center(frame, crop_w=203, source_width=1280)
    assert center is not None
    assert center > 640  # right half of the 1280-wide source


def test_dominant_region_estimation_returns_none_on_a_blank_frame():
    assert estimate_dominant_center(np.zeros((360, 640), dtype=np.uint8), 203, 1280) is None


# ---------------------------------------------------------------------------
# FFmpeg expression and debug overlay
# ---------------------------------------------------------------------------


def test_crop_expressions_are_clamped_and_even():
    traj = CropTrajectory(
        source_width=1280, source_height=720, crop_w=404, crop_h=720,
        points=[
            CropPoint(time=0.0, center_x=300, center_y=360, crop_x=100, crop_y=0,
                      crop_w=404, crop_h=720, subject_type="face"),
            CropPoint(time=3.0, center_x=900, center_y=360, crop_x=700, crop_y=0,
                      crop_w=404, crop_h=720, subject_type="face"),
        ],
    )
    x_expr = build_ffmpeg_crop_expression(traj, axis="x")
    assert "in_w-out_w" in x_expr and "2*trunc" in x_expr
    assert r"\," in x_expr  # commas escaped for the filtergraph
    # A constant axis collapses to a plain integer rather than a nested if-chain.
    assert build_ffmpeg_crop_expression(traj, axis="y") == "0"


def test_crop_expression_rejects_unknown_axis():
    traj = CropTrajectory(source_width=1280, source_height=720, crop_w=404, crop_h=720, points=[])
    with pytest.raises(ValueError):
        build_ffmpeg_crop_expression(traj, axis="z")


def test_debug_overlay_renders_a_diagnostic_video(clip, tmp_path):
    plan = build_reframe_plan(
        clip, 0.0, 4.0,
        detector=ScriptedDetector(lambda i: [(400 + i * 20, 300, 130)]),
        config=ReframeConfig(analysis_fps=5.0),
        source_width=SOURCE_W, source_height=SOURCE_H,
        collect_debug=True,
    )
    assert plan.debug_samples

    out = render_debug_overlay(clip, plan, 0.0, tmp_path / "debug.mp4")
    assert out is not None and out.is_file() and out.stat().st_size > 0
