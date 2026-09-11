"""Crop trajectory diagnostics and the detector-default regression.

`generate_crop_trajectory` used to hardcode the legacy Haar detector, which on
OpenCV 5 (no CascadeClassifier/HOGDescriptor) detects nothing at all. Every frame
therefore fell through to `center_fallback` and "smart" crop was silently a static
centre crop, with no way to tell from the artifact why.
"""
from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest

import freecher_worker.rendering  # noqa: F401  (import-order; subtitles<->rendering cycle)
from freecher_worker.crop.models import DetectedSubject
from freecher_worker.crop.tracker import generate_crop_trajectory


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("crop") / "tiny.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         "testsrc=size=1280x720:rate=10", "-t", "2", "-pix_fmt", "yuv420p",
         "-c:v", "libx264", "-y", str(out)],
        check=True, timeout=120,
    )
    return out


class _StubDetector:
    """Detector that always finds one face at a moving x position."""

    is_operational = True

    def __init__(self):
        self.calls = 0

    def detect(self, frame, t_abs):
        self.calls += 1
        cx = 200.0 + 40.0 * self.calls
        return [DetectedSubject(box=(int(cx) - 40, 300, 80, 80), confidence=0.9,
                                subject_type="face", area=6400.0,
                                center_x=cx, center_y=340.0)]


class _DeadDetector:
    """Stands in for Haar on OpenCV 5: present, but detects nothing, ever."""

    is_operational = False

    def detect(self, frame, t_abs):
        return []


# ------------------------------------------------------------------ regression
def test_default_detector_is_not_the_dead_legacy_one():
    sig = inspect.signature(generate_crop_trajectory)
    assert sig.parameters["detector_name"].default == "auto", (
        "the default must not go back to 'haar'; it detects nothing on OpenCV 5"
    )


# ----------------------------------------------------------------- diagnostics
def test_tracked_clip_reports_movement_and_no_fallback(tiny_video):
    traj = generate_crop_trajectory(tiny_video, 0.0, 2.0, detector=_StubDetector(),
                                    analysis_fps=2.0)
    d = traj.diagnostics
    assert d is not None
    assert d.detector_operational is True
    assert d.frames_with_detections > 0
    assert d.total_detections > 0
    assert d.subject_type_counts.get("face", 0) > 0
    # The final sample lands on/after the last frame of a short clip, so a
    # decode-failure keyframe is expected; what matters is that the clip is
    # mostly tracked and that any fallback carries a reason.
    assert d.tracked_fraction > 0.5
    assert set(d.fallback_reasons) <= {"frame_decode_failed"}
    assert sum(d.fallback_reasons.values()) == round(d.fallback_fraction * len(traj.points))
    assert d.crop_center_x_range and d.crop_center_x_range > 0


def test_dead_detector_records_an_explicit_reason(tiny_video):
    traj = generate_crop_trajectory(tiny_video, 0.0, 2.0, detector=_DeadDetector(),
                                    analysis_fps=2.0)
    d = traj.diagnostics
    assert d.detector_operational is False
    assert d.frames_with_detections == 0
    assert d.tracked_fraction == 0.0
    assert d.fallback_fraction == pytest.approx(1.0)
    # THE acceptance criterion: center_fallback is never unexplained.
    assert d.fallback_reasons.get("detector_not_operational", 0) > 0
    assert all(p.subject_type == "center_fallback" for p in traj.points)


def test_operational_detector_with_no_subjects_says_so(tiny_video):
    class Empty(_DeadDetector):
        is_operational = True

    d = generate_crop_trajectory(tiny_video, 0.0, 2.0, detector=Empty(),
                                 analysis_fps=2.0).diagnostics
    assert d.fallback_reasons.get("no_subjects_detected", 0) > 0
    assert "detector_not_operational" not in d.fallback_reasons


def test_every_fallback_keyframe_is_accounted_for(tiny_video):
    traj = generate_crop_trajectory(tiny_video, 0.0, 2.0, detector=_DeadDetector(),
                                    analysis_fps=2.0)
    d = traj.diagnostics
    fallback_points = sum(1 for p in traj.points if p.subject_type == "center_fallback")
    assert sum(d.fallback_reasons.values()) >= fallback_points > 0


def test_summary_is_human_readable_and_names_the_detector(tiny_video):
    d = generate_crop_trajectory(tiny_video, 0.0, 2.0, detector=_StubDetector(),
                                 analysis_fps=2.0).diagnostics
    assert "detector=" in d.summary
    assert "tracked" in d.summary
    assert "fallback" in d.summary


def test_diagnostics_survive_json_round_trip(tiny_video):
    traj = generate_crop_trajectory(tiny_video, 0.0, 2.0, detector=_StubDetector(),
                                    analysis_fps=2.0)
    from freecher_worker.crop.models import CropTrajectory
    again = CropTrajectory.model_validate_json(traj.model_dump_json())
    assert again.diagnostics is not None
    assert again.diagnostics.summary == traj.diagnostics.summary
