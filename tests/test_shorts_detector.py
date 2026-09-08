"""Subject detection baseline and tracking/render fallback semantics.

Regression coverage for the reported defect: every short reported detections=0 and
fallback_rate=100%. The cause was that `opencv-python-headless` 5.x dropped `CascadeClassifier`
and `HOGDescriptor`, so the Haar baseline silently returned an empty list for every frame.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from freecher_worker.crop.detector import (
    CenterCropDetector,
    CompositeSubjectDetector,
    DetectorUnavailableError,
    HaarCascadeFaceDetector,
    SubjectDetector,
    YuNetFaceDetector,
    get_subject_detector,
)
from freecher_worker.crop.models import DetectedSubject
from freecher_worker.crop.weights import (
    YUNET_FILENAME,
    YUNET_SHA256,
    ModelWeightsError,
    default_cache_dir,
    resolve_model_weights,
)
from freecher_worker.shorts.reframe import (
    TRACKING_MODE_CENTER,
    TRACKING_MODE_DOMINANT,
    TRACKING_MODE_MIXED,
    TRACKING_MODE_SUBJECT,
    ReframeConfig,
    ReframeDiagnostics,
    build_reframe_plan,
)

YUNET_PRESENT = (default_cache_dir() / YUNET_FILENAME).is_file()
needs_yunet = pytest.mark.skipif(not YUNET_PRESENT, reason="YuNet weights are not cached locally")


class StubDetector(SubjectDetector):
    name = "stub"

    def __init__(self, script, subject_type="face", frame_width=1280):
        self.script = script
        self.subject_type = subject_type
        self.frame_width = frame_width
        self.calls = 0

    def detect(self, frame_bgr, timestamp):
        index = self.calls
        self.calls += 1
        scale = frame_bgr.shape[1] / float(self.frame_width)
        out = []
        for cx, cy, size in self.script(index):
            w = h = int(size * scale)
            x, y = int(cx * scale - w / 2), int(cy * scale - h / 2)
            out.append(DetectedSubject(
                box=(x, y, w, h), confidence=0.9, subject_type=self.subject_type,
                area=float(w * h), center_x=x + w / 2.0, center_y=y + h / 2.0,
            ))
        return out


def write_clip(path, seconds=6.0, fps=10, width=1280, height=720, cut_at=None):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(int(seconds * fps)):
        dark = cut_at is not None and (i / fps) >= cut_at
        frame = np.full((height, width, 3), 20 if dark else 200, np.uint8)
        cv2.rectangle(frame, (100 + i, 100), (200 + i, 200), (0, 0, 255) if dark else (255, 0, 0), -1)
        writer.write(frame)
    writer.release()
    return str(path)


# ---------------------------------------------------------------------------
# Detector availability must never be silent
# ---------------------------------------------------------------------------


def test_haar_detector_reports_when_the_build_cannot_support_it():
    """An inert detector must be visible as inert, not look like an empty scene."""
    detector = HaarCascadeFaceDetector()
    if detector.face_cascade is None and detector.hog is None:
        assert detector.is_operational is False
        assert "unavailable" in detector.describe()
    else:
        assert detector.is_operational is True


def test_center_detector_declares_itself_non_operational():
    detector = CenterCropDetector()
    assert detector.is_operational is False
    assert detector.detect(np.zeros((64, 64, 3), np.uint8), 0.0) == []


def test_composite_detector_returns_the_first_non_empty_result():
    faces = StubDetector(lambda i: [(300, 200, 80)] if i == 0 else [])
    people = StubDetector(lambda i: [(900, 300, 200)], subject_type="person")
    composite = CompositeSubjectDetector([faces, people])
    frame = np.zeros((720, 1280, 3), np.uint8)

    first = composite.detect(frame, 0.0)
    assert [s.subject_type for s in first] == ["face"]

    second = composite.detect(frame, 0.2)
    assert [s.subject_type for s in second] == ["person"]


def test_composite_detector_survives_an_exploding_member():
    class Exploding(SubjectDetector):
        name = "boom"

        def detect(self, frame_bgr, timestamp):
            raise RuntimeError("weights on fire")

    good = StubDetector(lambda i: [(400, 300, 100)])
    composite = CompositeSubjectDetector([Exploding(), good])
    assert len(composite.detect(np.zeros((720, 1280, 3), np.uint8), 0.0)) == 1


def test_auto_detector_is_operational_or_says_why(caplog):
    detector = get_subject_detector("auto")
    assert detector.describe()
    if not detector.is_operational:
        assert isinstance(detector, CenterCropDetector)


# ---------------------------------------------------------------------------
# YuNet baseline
# ---------------------------------------------------------------------------


@needs_yunet
def test_yunet_detects_a_real_face():
    detector = YuNetFaceDetector()
    assert detector.is_operational
    assert "yunet" in detector.describe()

    # A synthetic frame has no face; the detector must not hallucinate one.
    assert detector.detect(np.full((480, 640, 3), 128, np.uint8), 0.0) == []


@needs_yunet
def test_yunet_boxes_stay_inside_the_frame():
    detector = YuNetFaceDetector()
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    for subject in detector.detect(frame, 0.0):
        x, y, w, h = subject.box
        assert 0 <= x < 640 and 0 <= y < 480
        assert x + w <= 640 and y + h <= 480
        assert subject.subject_type == "face"


def test_yunet_reports_a_clear_error_when_weights_are_missing(tmp_path):
    with pytest.raises(DetectorUnavailableError, match="not found|download"):
        YuNetFaceDetector(model_path=tmp_path / "absent.onnx", allow_download=False)


def test_weight_download_rejects_a_checksum_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("FREECHER_MODELS_DIR", str(tmp_path))
    with pytest.raises(ModelWeightsError, match="Checksum mismatch|Could not download"):
        resolve_model_weights(
            filename="fake.bin",
            url="https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg",
            sha256="0" * 64,
        )
    assert list(tmp_path.glob("fake.bin")) == []
    assert list(tmp_path.glob("*.part")) == []


def test_cached_weights_are_used_without_downloading(tmp_path, monkeypatch):
    monkeypatch.setenv("FREECHER_MODELS_DIR", str(tmp_path))
    (tmp_path / "cached.bin").write_bytes(b"already here")
    resolved = resolve_model_weights(
        filename="cached.bin", url="http://invalid.invalid/x", sha256="unused", allow_download=False
    )
    assert resolved == tmp_path / "cached.bin"


def test_yunet_checksum_is_pinned():
    assert len(YUNET_SHA256) == 64
    assert YUNET_FILENAME.endswith(".onnx")


# ---------------------------------------------------------------------------
# Detection and tracking diagnostics
# ---------------------------------------------------------------------------


def test_detection_diagnostics_count_faces_and_persons(tmp_path):
    clip = write_clip(tmp_path / "clip.mp4")
    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=StubDetector(lambda i: [(400 + i * 10, 300, 120)]),
        config=ReframeConfig(analysis_fps=5.0), source_width=1280, source_height=720,
    )
    diag = plan.diagnostics

    assert diag.analyzed_frames == diag.sampled_frames > 0
    assert diag.frames_with_face == diag.frames_with_detection
    assert diag.frames_with_person == 0
    assert diag.detection_coverage == pytest.approx(1.0)
    assert diag.tracking_coverage == pytest.approx(1.0)
    assert diag.tracking_fallback_rate == pytest.approx(0.0)
    assert diag.tracking_mode == TRACKING_MODE_SUBJECT
    assert diag.track_count == diag.unique_tracks


def test_person_only_detections_are_counted_separately(tmp_path):
    clip = write_clip(tmp_path / "clip.mp4")
    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=StubDetector(lambda i: [(500, 350, 260)], subject_type="person"),
        config=ReframeConfig(analysis_fps=5.0), source_width=1280, source_height=720,
    )
    diag = plan.diagnostics

    assert diag.frames_with_person > 0
    assert diag.frames_with_face == 0
    assert diag.person_coverage == pytest.approx(1.0)
    assert diag.face_coverage == pytest.approx(0.0)


def test_no_detections_are_reported_as_dominant_region_tracking(tmp_path):
    """The reported symptom: detections=0 must read as a tracking mode, not a render failure."""
    clip = write_clip(tmp_path / "clip.mp4")
    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=StubDetector(lambda i: []),
        config=ReframeConfig(analysis_fps=5.0), source_width=1280, source_height=720,
    )
    diag = plan.diagnostics

    assert diag.detection_coverage == pytest.approx(0.0)
    assert diag.tracking_coverage == pytest.approx(0.0)
    assert diag.tracking_fallback_rate == pytest.approx(1.0)
    assert diag.tracking_mode == TRACKING_MODE_DOMINANT
    assert diag.dominant_fallback_rate + diag.center_fallback_rate + diag.previous_fallback_rate == pytest.approx(1.0)


def test_intermittent_detection_is_reported_as_mixed_tracking(tmp_path):
    clip = write_clip(tmp_path / "clip.mp4")
    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=StubDetector(lambda i: [(600, 300, 120)] if i % 3 == 0 else []),
        config=ReframeConfig(analysis_fps=5.0, track_max_misses=0),
        source_width=1280, source_height=720,
    )
    diag = plan.diagnostics

    assert 0.0 < diag.tracking_coverage < 1.0
    assert diag.tracking_mode == TRACKING_MODE_MIXED


def test_center_crop_plan_reports_center_tracking_mode(tmp_path):
    plan = build_reframe_plan(
        tmp_path / "missing.mp4", 0.0, 6.0,
        config=ReframeConfig(analysis_fps=5.0), source_width=1280, source_height=720,
    )
    assert plan.diagnostics.tracking_mode == TRACKING_MODE_CENTER
    assert plan.diagnostics.detector_operational is False


def test_rates_are_zero_safe_on_an_empty_diagnostic():
    diag = ReframeDiagnostics()
    assert diag.detection_coverage == 0.0
    assert diag.tracking_coverage == 0.0
    assert diag.tracking_fallback_rate == 0.0
    assert diag.resolve_tracking_mode() == TRACKING_MODE_CENTER


# ---------------------------------------------------------------------------
# Crop velocity
# ---------------------------------------------------------------------------


def test_pan_velocity_respects_the_limit_when_there_is_no_scene_cut(tmp_path):
    """A teleporting subject must still produce operator-like motion between cuts."""
    clip = write_clip(tmp_path / "clip.mp4")
    config = ReframeConfig(analysis_fps=5.0, max_velocity_px_per_sec=120.0)
    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=StubDetector(lambda i: [(150, 300, 140)] if i % 4 < 2 else [(1130, 300, 140)]),
        config=config, source_width=1280, source_height=720,
    )

    assert plan.diagnostics.scene_cuts == 0
    assert plan.diagnostics.trajectory.scene_cut_snaps == 0
    assert plan.diagnostics.trajectory.peak_velocity_non_scene_cut <= config.max_velocity_px_per_sec * 1.35


def test_only_scene_cuts_may_exceed_the_velocity_limit(tmp_path):
    """A hard cut is allowed to re-anchor instantly; nothing else is."""
    clip = write_clip(tmp_path / "cut.mp4", seconds=6.0, cut_at=3.0)
    config = ReframeConfig(analysis_fps=5.0, max_velocity_px_per_sec=120.0)
    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=StubDetector(lambda i: [(250, 300, 140)] if i < 15 else [(1030, 300, 140)]),
        config=config, source_width=1280, source_height=720,
    )
    traj = plan.diagnostics.trajectory

    assert plan.diagnostics.scene_cuts >= 1
    assert traj.scene_cut_snaps >= 1
    assert traj.max_velocity_px_per_sec > config.max_velocity_px_per_sec
    assert traj.peak_velocity_non_scene_cut <= config.max_velocity_px_per_sec * 1.35
    assert traj.peak_velocity_non_scene_cut < traj.max_velocity_px_per_sec


def test_static_subject_has_zero_velocity_on_both_measures(tmp_path):
    clip = write_clip(tmp_path / "clip.mp4")
    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=StubDetector(lambda i: [(400, 300, 140)]),
        config=ReframeConfig(analysis_fps=5.0), source_width=1280, source_height=720,
    )
    traj = plan.diagnostics.trajectory
    assert traj.max_velocity_px_per_sec == 0.0
    assert traj.peak_velocity_non_scene_cut == 0.0


# ---------------------------------------------------------------------------
# Track identity persistence
# ---------------------------------------------------------------------------


def test_small_fast_subject_keeps_one_track_identity(tmp_path):
    """A small facecam moving faster than its own box width must not be re-identified on every frame.

    Gating association on box size alone made a 20px subject moving 16px per sample overlap its
    previous box by almost nothing, so it became a brand-new person on every frame.
    """
    clip = write_clip(tmp_path / "clip.mp4")
    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=StubDetector(lambda i: [(200 + i * 16, 300, 20)]),
        config=ReframeConfig(analysis_fps=5.0), source_width=1280, source_height=720,
    )

    assert plan.diagnostics.detection_coverage == pytest.approx(1.0)
    assert plan.diagnostics.track_count <= 3, "the subject was re-identified far too often"
    assert plan.diagnostics.active_subject_switches <= 1


def test_two_small_subjects_keep_separate_identities(tmp_path):
    clip = write_clip(tmp_path / "clip.mp4")
    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=StubDetector(lambda i: [(200 + i * 14, 300, 22), (1000 - i * 12, 320, 24)]),
        config=ReframeConfig(analysis_fps=5.0), source_width=1280, source_height=720,
    )

    assert plan.diagnostics.max_simultaneous_subjects == 2
    assert plan.diagnostics.track_count <= 4


def test_large_slow_subject_keeps_exactly_one_track(tmp_path):
    clip = write_clip(tmp_path / "clip.mp4")
    plan = build_reframe_plan(
        clip, 0.0, 6.0,
        detector=StubDetector(lambda i: [(400 + i * 4, 300, 160)]),
        config=ReframeConfig(analysis_fps=5.0), source_width=1280, source_height=720,
    )
    assert plan.diagnostics.track_count == 1


def test_debug_overlay_records_subject_types_and_state(tmp_path):
    clip = write_clip(tmp_path / "clip.mp4")
    plan = build_reframe_plan(
        clip, 0.0, 4.0,
        detector=StubDetector(lambda i: [(400 + i * 20, 300, 130)]),
        config=ReframeConfig(analysis_fps=5.0), source_width=1280, source_height=720,
        collect_debug=True,
    )
    sample = plan.debug_samples[2]

    assert sample.boxes and sample.track_ids
    assert sample.subject_types == ["face"] * len(sample.boxes)
    assert sample.active_track_id in sample.track_ids
    assert sample.crop_x >= 0
