"""Smart 9:16 reframing — keep the important subject inside the vertical frame.

Architecture (all local, no per-frame LLM calls):

    sampled frames -> face/person detection -> track association -> active subject selection
    -> safe framing target -> spring smoothing + velocity/acceleration clamps -> crop trajectory

Priority of what is kept in frame, highest first:

1. the actively speaking subject (mouth-region motion proxy),
2. any detected face,
3. two interacting people, framed together when they fit,
4. the dominant visual region of the frame,
5. static center crop.

Steps 4 and 5 are also the fallback ladder: the stage always yields a usable trajectory.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field

from freecher_worker.crop.detector import SubjectDetector, get_subject_detector
from freecher_worker.crop.models import CropPoint, CropTrajectory, DetectedSubject

logger = logging.getLogger("freecher_worker")

REFRAME_VERSION = "smart_reframe_v1"

REFRAME_MODE_SMART = "smart"
REFRAME_MODE_CENTER = "center"

FALLBACK_NONE = "subject"
FALLBACK_DUAL = "dual_subject"
FALLBACK_PREVIOUS = "previous_stable"
FALLBACK_DOMINANT = "dominant_region"
FALLBACK_CENTER = "center_crop"


class ReframeConfig(BaseModel):
    """Tunable parameters for detection, subject arbitration, framing and smoothing."""

    analysis_fps: float = Field(default=5.0, gt=0.0)
    detect_max_width: int = Field(default=640, gt=0)
    subject_padding_ratio: float = Field(default=0.55, ge=0.0)
    head_position_ratio: float = Field(default=0.38, ge=0.0, le=1.0)
    headroom_ratio: float = Field(default=0.45, ge=0.0)
    edge_margin_ratio: float = Field(default=0.06, ge=0.0, le=0.4)
    deadzone_ratio: float = Field(default=0.02, ge=0.0)
    smoothing_alpha: float = Field(default=0.22, gt=0.0, le=1.0)
    max_velocity_px_per_sec: float = Field(default=160.0, gt=0.0)
    max_acceleration_px_per_sec2: float = Field(default=420.0, gt=0.0)
    switch_hold_sec: float = Field(default=0.8, ge=0.0)
    switch_margin: float = Field(default=0.25, ge=0.0)
    min_switch_interval_sec: float = Field(default=1.5, ge=0.0)
    track_max_misses: int = Field(default=6, ge=0)
    scene_cut_threshold: float = Field(default=0.35, gt=0.0)
    dual_subject_balance: float = Field(default=0.35, ge=0.0, le=1.0)
    jitter_epsilon_px: float = Field(default=2.0, ge=0.0)


class TrajectoryStats(BaseModel):
    """Aggregate motion statistics of the emitted crop trajectory."""

    mean_velocity_px_per_sec: float = 0.0
    max_velocity_px_per_sec: float = 0.0
    total_travel_px: float = 0.0
    stationary_ratio: float = Field(default=1.0, description="Fraction of samples with no crop movement")
    crop_x_min: int = 0
    crop_x_max: int = 0
    crop_x_range: int = 0
    crop_y_min: int = 0
    crop_y_max: int = 0


class ReframeDiagnostics(BaseModel):
    """Per-short quality diagnostics for the smart reframing stage."""

    version: str = REFRAME_VERSION
    detector: str = ""
    analysis_fps: float = 0.0
    sampled_frames: int = 0
    frames_with_detection: int = 0
    frames_with_active_subject: int = 0
    detected_subjects_total: int = 0
    max_simultaneous_subjects: int = 0
    unique_tracks: int = 0
    dominant_subject_switches: int = 0
    dual_subject_frames: int = 0
    scene_cuts: int = 0
    fallback_previous_frames: int = 0
    fallback_dominant_frames: int = 0
    fallback_center_frames: int = 0
    fallback_used: bool = False
    edge_clamped_frames: int = 0
    analysis_seconds: float = 0.0
    trajectory: TrajectoryStats = Field(default_factory=TrajectoryStats)

    @property
    def detection_coverage(self) -> float:
        """Fraction of sampled frames in which the detector found at least one subject."""
        return round(self.frames_with_detection / self.sampled_frames, 4) if self.sampled_frames else 0.0

    @property
    def tracking_coverage(self) -> float:
        """Fraction of sampled frames framed from a tracked subject rather than a fallback."""
        return (
            round(self.frames_with_active_subject / self.sampled_frames, 4) if self.sampled_frames else 0.0
        )

    @property
    def fallback_rate(self) -> float:
        """Fraction of sampled frames that used any rung of the fallback ladder."""
        used = (
            self.fallback_previous_frames
            + self.fallback_dominant_frames
            + self.fallback_center_frames
        )
        return round(used / self.sampled_frames, 4) if self.sampled_frames else 0.0


class DebugSample(BaseModel):
    """Per-sample record retained for the optional debug overlay render."""

    time: float
    boxes: List[Tuple[int, int, int, int]] = Field(default_factory=list)
    track_ids: List[int] = Field(default_factory=list)
    active_track_id: Optional[int] = None
    target_x: float = 0.0
    target_y: float = 0.0
    crop_x: int = 0
    crop_y: int = 0
    fallback: str = FALLBACK_NONE
    scene_cut: bool = False


class ReframePlan(BaseModel):
    """Crop trajectory plus diagnostics for one short."""

    mode: str = REFRAME_MODE_SMART
    trajectory: CropTrajectory
    diagnostics: ReframeDiagnostics = Field(default_factory=ReframeDiagnostics)
    debug_samples: List[DebugSample] = Field(default_factory=list)


@dataclass
class SubjectTrack:
    """A subject followed across sampled frames."""

    track_id: int
    box: Tuple[int, int, int, int]
    center_x: float
    center_y: float
    area: float
    subject_type: str
    confidence: float
    first_seen: float
    last_seen: float
    hits: int = 1
    misses: int = 0
    speaking_score: float = 0.0
    mouth_patch: Optional[np.ndarray] = field(default=None, repr=False)

    def update(self, subject: DetectedSubject, timestamp: float) -> None:
        self.box = subject.box
        self.center_x = subject.center_x
        self.center_y = subject.center_y
        self.area = subject.area
        self.subject_type = subject.subject_type
        self.confidence = subject.confidence
        self.last_seen = timestamp
        self.hits += 1
        self.misses = 0


def calculate_vertical_crop(source_width: int, source_height: int) -> Tuple[int, int]:
    """Largest even-sized 9:16 window that fits inside the source frame."""
    crop_w = int(round(((source_height * 9.0) / 16.0) / 2.0)) * 2
    crop_h = source_height - (source_height % 2)
    if crop_w > source_width:
        crop_w = source_width - (source_width % 2)
        crop_h = int(round(((crop_w * 16.0) / 9.0) / 2.0)) * 2
        crop_h = min(crop_h, source_height - (source_height % 2))
    return max(2, crop_w), max(2, crop_h)


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _associate(
    tracks: List[SubjectTrack],
    subjects: List[DetectedSubject],
    timestamp: float,
    next_track_id: int,
) -> Tuple[List[SubjectTrack], List[Tuple[SubjectTrack, DetectedSubject]], int]:
    """Greedy IoU/distance association of detections to existing tracks."""
    pairs: List[Tuple[float, int, int]] = []
    for t_idx, track in enumerate(tracks):
        for s_idx, subject in enumerate(subjects):
            iou = _iou(track.box, subject.box)
            if iou >= 0.20:
                pairs.append((-iou, t_idx, s_idx))
                continue
            reach = 0.6 * max(track.box[2], subject.box[2], 1)
            dist = abs(track.center_x - subject.center_x) + abs(track.center_y - subject.center_y)
            if dist <= reach:
                pairs.append((dist / max(reach, 1.0), t_idx, s_idx))

    pairs.sort()
    existing_count = len(tracks)
    used_tracks: set[int] = set()
    used_subjects: set[int] = set()
    matched: List[Tuple[SubjectTrack, DetectedSubject]] = []

    for _, t_idx, s_idx in pairs:
        if t_idx in used_tracks or s_idx in used_subjects:
            continue
        used_tracks.add(t_idx)
        used_subjects.add(s_idx)
        tracks[t_idx].update(subjects[s_idx], timestamp)
        matched.append((tracks[t_idx], subjects[s_idx]))

    for s_idx, subject in enumerate(subjects):
        if s_idx in used_subjects:
            continue
        track = SubjectTrack(
            track_id=next_track_id,
            box=subject.box,
            center_x=subject.center_x,
            center_y=subject.center_y,
            area=subject.area,
            subject_type=subject.subject_type,
            confidence=subject.confidence,
            first_seen=timestamp,
            last_seen=timestamp,
        )
        next_track_id += 1
        tracks.append(track)
        matched.append((track, subject))

    # Only tracks that already existed can be "missed"; tracks created from this frame's
    # detections are live by definition.
    for t_idx in range(existing_count):
        if t_idx not in used_tracks:
            tracks[t_idx].misses += 1

    return tracks, matched, next_track_id


def _mouth_patch(frame_gray: np.ndarray, box: Sequence[int]) -> Optional[np.ndarray]:
    """Crop the lower-middle of a face box, used as the active-speaker proxy."""
    x, y, w, h = box
    if w < 8 or h < 8:
        return None
    my = int(y + 0.58 * h)
    mh = max(2, int(0.34 * h))
    mx = int(x + 0.20 * w)
    mw = max(2, int(0.60 * w))
    patch = frame_gray[max(0, my) : my + mh, max(0, mx) : mx + mw]
    if patch.size == 0:
        return None
    import cv2

    return cv2.resize(patch, (24, 12), interpolation=cv2.INTER_AREA).astype(np.float32)


def _update_speaking(track: SubjectTrack, patch: Optional[np.ndarray]) -> None:
    """Update the speaking score from mouth-region temporal change."""
    if patch is None:
        track.speaking_score *= 0.7
        return
    if track.mouth_patch is not None and track.mouth_patch.shape == patch.shape:
        motion = float(np.mean(np.abs(patch - track.mouth_patch))) / 255.0
        observed = min(1.0, motion * 12.0)
        track.speaking_score = 0.6 * track.speaking_score + 0.4 * observed
    track.mouth_patch = patch


def _track_weight(track: SubjectTrack, max_area: float, source_width: int) -> float:
    """Importance of a track for the active-subject decision."""
    area_norm = track.area / max_area if max_area > 0 else 0.0
    half_w = source_width / 2.0
    centrality = 1.0 - min(1.0, abs(track.center_x - half_w) / max(half_w, 1.0))
    persistence = min(1.0, track.hits / 5.0)
    face_bonus = 1.0 if track.subject_type == "face" else 0.75
    return face_bonus * (
        0.42 * track.speaking_score + 0.28 * area_norm + 0.15 * centrality + 0.15 * persistence
    )


def estimate_dominant_center(frame_gray: np.ndarray, crop_w: int, source_width: int) -> Optional[float]:
    """Locate the horizontal center of the highest-detail region of a frame.

    Used when no subject is detected: column-wise gradient energy is a cheap, deterministic
    stand-in for "where the interesting content is".
    """
    if frame_gray.size == 0:
        return None
    import cv2

    grad = np.abs(cv2.Sobel(frame_gray, cv2.CV_32F, 1, 0, ksize=3))
    columns = grad.sum(axis=0)
    if columns.size < 4 or float(columns.sum()) <= 0.0:
        return None

    scale = source_width / float(frame_gray.shape[1])
    window = max(2, int(round(crop_w / scale)))
    if window >= columns.size:
        return float(source_width / 2.0)

    cumulative = np.concatenate(([0.0], np.cumsum(columns)))
    sums = cumulative[window:] - cumulative[:-window]
    best = int(np.argmax(sums))
    return float((best + window / 2.0) * scale)


def _framing_target(
    track: SubjectTrack,
    crop_w: int,
    crop_h: int,
    source_width: int,
    source_height: int,
    cfg: ReframeConfig,
) -> Tuple[float, float, bool]:
    """Safe-framing target center for one subject; returns (x, y, edge_clamped)."""
    box_x, box_y, box_w, box_h = track.box
    half_w = crop_w / 2.0

    target_x = track.center_x
    pad = box_w * cfg.subject_padding_ratio
    left_need = box_x - pad
    right_need = box_x + box_w + pad
    edge_clamped = False
    if (right_need - left_need) <= crop_w:
        lo = right_need - half_w
        hi = left_need + half_w
        if lo <= hi:
            clamped = min(max(target_x, lo), hi)
            edge_clamped = abs(clamped - target_x) > 1e-6
            target_x = clamped

    # Keep the face near the upper third with headroom, then never cut the top of the head.
    target_y = track.center_y + crop_h * (0.5 - cfg.head_position_ratio)
    headroom_limit = box_y - cfg.headroom_ratio * box_h + crop_h / 2.0
    target_y = min(target_y, headroom_limit)
    target_y = max(target_y, box_y + box_h - crop_h / 2.0)
    target_y = min(max(target_y, crop_h / 2.0), source_height - crop_h / 2.0)

    return target_x, target_y, edge_clamped


@dataclass
class _Observation:
    time: float
    target_x: float
    target_y: float
    subject_type: str
    fallback: str
    scene_cut: bool
    active_track_id: Optional[int]
    boxes: List[Tuple[int, int, int, int]]
    track_ids: List[int]
    edge_clamped: bool


#: Seeking to exactly the clip end lands past the final frame, so the last sample sits just inside.
_END_MARGIN_SEC = 0.05


def _sample_times(
    duration_seconds: float,
    analysis_fps: float,
    end_margin_sec: float = _END_MARGIN_SEC,
) -> List[float]:
    """Sample offsets covering the clip, with the final sample kept inside the last frame.

    The FFmpeg crop expression holds the last keyframe's value for any later ``t``, so stopping
    just short of the end loses nothing while avoiding a spurious decode failure on every clip.
    """
    interval = 1.0 / analysis_fps
    last = max(0.0, duration_seconds - end_margin_sec)
    steps = max(2, int(last / interval) + 1)
    times = [min(last, i * interval) for i in range(steps)]
    if times[-1] < last:
        times.append(round(last, 3))
    return times


def build_center_crop_plan(
    source_width: int,
    source_height: int,
    duration_seconds: float,
    analysis_fps: float,
    reason: str,
    detector_name: str = "",
) -> ReframePlan:
    """Static center-crop trajectory: the last line of defense, always renderable."""
    crop_w, crop_h = calculate_vertical_crop(source_width, source_height)
    crop_x = ((source_width - crop_w) // 4) * 2
    crop_y = ((source_height - crop_h) // 4) * 2
    times = _sample_times(duration_seconds, analysis_fps)
    points = [
        CropPoint(
            time=round(t, 3),
            center_x=source_width / 2.0,
            center_y=source_height / 2.0,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_w=crop_w,
            crop_h=crop_h,
            subject_type=FALLBACK_CENTER,
        )
        for t in (times[0], times[-1])
    ]
    logger.info(f"[reframe] Using static center crop: {reason}")
    return ReframePlan(
        mode=REFRAME_MODE_CENTER,
        trajectory=CropTrajectory(
            source_width=source_width,
            source_height=source_height,
            crop_w=crop_w,
            crop_h=crop_h,
            points=points,
        ),
        diagnostics=ReframeDiagnostics(
            detector=detector_name,
            analysis_fps=analysis_fps,
            sampled_frames=len(points),
            fallback_center_frames=len(points),
            fallback_used=True,
            trajectory=TrajectoryStats(
                crop_x_min=crop_x, crop_x_max=crop_x, crop_y_min=crop_y, crop_y_max=crop_y
            ),
        ),
    )


def _collect_observations(
    video_path: Path,
    start_seconds: float,
    duration_seconds: float,
    detector: SubjectDetector,
    cfg: ReframeConfig,
    crop_w: int,
    crop_h: int,
    source_width: int,
    source_height: int,
    diagnostics: ReframeDiagnostics,
) -> List[_Observation]:
    """Sample the clip, detect and track subjects, and emit one framing target per sample."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video for reframing analysis: {video_path}")

    # Timestamp seeking is only accurate to a frame, so stay a full source frame clear of the end.
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    end_margin = max(_END_MARGIN_SEC, (1.0 / source_fps) if source_fps > 0 else 0.0)

    tracks: List[SubjectTrack] = []
    next_track_id = 1
    active_id: Optional[int] = None
    pending_id: Optional[int] = None
    pending_since = 0.0
    last_switch = -1e9
    prev_small: Optional[np.ndarray] = None
    last_target: Optional[Tuple[float, float]] = None
    last_subject_target: Optional[Tuple[float, float]] = None
    last_subject_time = -1e9
    # How long a lost subject's framing is held before the crop starts following content again.
    hold_seconds = max(0.5, cfg.track_max_misses / cfg.analysis_fps)

    observations: List[_Observation] = []

    for t_rel in _sample_times(duration_seconds, cfg.analysis_fps, end_margin):
        capture.set(cv2.CAP_PROP_POS_MSEC, (start_seconds + t_rel) * 1000.0)
        ok, frame = capture.read()
        diagnostics.sampled_frames += 1

        if not ok or frame is None:
            # A dropped frame says nothing about the scene, so simply hold the current framing.
            target = last_target or (source_width / 2.0, source_height / 2.0)
            fallback = FALLBACK_PREVIOUS if last_target else FALLBACK_CENTER
            if last_target:
                diagnostics.fallback_previous_frames += 1
            else:
                diagnostics.fallback_center_frames += 1
            observations.append(
                _Observation(t_rel, target[0], target[1], fallback, fallback, False, None, [], [], False)
            )
            continue

        scale = min(1.0, cfg.detect_max_width / max(1, frame.shape[1]))
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        scene_cut = False
        if prev_small is not None and prev_small.shape == small_gray.shape:
            diff = float(np.mean(np.abs(small_gray.astype(np.float32) - prev_small.astype(np.float32)))) / 255.0
            scene_cut = diff > cfg.scene_cut_threshold
        prev_small = small_gray
        if scene_cut:
            diagnostics.scene_cuts += 1

        try:
            raw_subjects = detector.detect(small, start_seconds + t_rel)
        except Exception as exc:
            logger.warning(f"[reframe] Detector failed at t={t_rel:.2f}s: {exc}")
            raw_subjects = []

        subjects: List[DetectedSubject] = []
        inv = 1.0 / scale if scale > 0 else 1.0
        for subject in raw_subjects:
            x, y, w, h = subject.box
            box = (int(x * inv), int(y * inv), int(w * inv), int(h * inv))
            subjects.append(
                DetectedSubject(
                    box=box,
                    confidence=subject.confidence,
                    subject_type=subject.subject_type,
                    area=float(box[2] * box[3]),
                    center_x=box[0] + box[2] / 2.0,
                    center_y=box[1] + box[3] / 2.0,
                )
            )

        diagnostics.detected_subjects_total += len(subjects)
        diagnostics.max_simultaneous_subjects = max(diagnostics.max_simultaneous_subjects, len(subjects))
        if subjects:
            diagnostics.frames_with_detection += 1

        tracks, matched, next_track_id = _associate(tracks, subjects, t_rel, next_track_id)
        for track, subject in matched:
            _update_speaking(track, _mouth_patch(small_gray, [int(v * scale) for v in subject.box]))
        tracks = [t for t in tracks if t.misses <= cfg.track_max_misses]
        diagnostics.unique_tracks = max(diagnostics.unique_tracks, next_track_id - 1)

        live = [t for t in tracks if t.misses == 0]
        if not live:
            # Fallback ladder. A briefly lost subject is held, but once the hold window expires
            # the crop follows the dominant visual region rather than freezing on a stale face.
            recently_had_subject = (
                last_subject_target is not None and (t_rel - last_subject_time) <= hold_seconds
            )
            if recently_had_subject:
                target_x, target_y = last_subject_target
                fallback = FALLBACK_PREVIOUS
                diagnostics.fallback_previous_frames += 1
            else:
                dominant = estimate_dominant_center(small_gray, crop_w, source_width)
                if dominant is not None:
                    target_x, target_y = dominant, source_height / 2.0
                    fallback = FALLBACK_DOMINANT
                    diagnostics.fallback_dominant_frames += 1
                elif last_target is not None:
                    target_x, target_y = last_target
                    fallback = FALLBACK_PREVIOUS
                    diagnostics.fallback_previous_frames += 1
                else:
                    target_x, target_y = source_width / 2.0, source_height / 2.0
                    fallback = FALLBACK_CENTER
                    diagnostics.fallback_center_frames += 1
            observations.append(
                _Observation(t_rel, target_x, target_y, fallback, fallback, scene_cut, None, [], [], False)
            )
            last_target = (target_x, target_y)
            continue

        max_area = max(t.area for t in live)
        weights = {t.track_id: _track_weight(t, max_area, source_width) for t in live}
        by_id = {t.track_id: t for t in live}
        best_id = max(weights, key=lambda tid: (weights[tid], -tid))

        if active_id not in by_id:
            if active_id is not None:
                diagnostics.dominant_subject_switches += 1
                last_switch = t_rel
            active_id = best_id
            pending_id = None
        elif scene_cut and best_id != active_id:
            diagnostics.dominant_subject_switches += 1
            active_id = best_id
            pending_id = None
            last_switch = t_rel
        elif best_id != active_id:
            if weights[best_id] > weights[active_id] * (1.0 + cfg.switch_margin):
                if pending_id != best_id:
                    pending_id = best_id
                    pending_since = t_rel
                elif (t_rel - pending_since) >= cfg.switch_hold_sec and (
                    t_rel - last_switch
                ) >= cfg.min_switch_interval_sec:
                    diagnostics.dominant_subject_switches += 1
                    active_id = best_id
                    pending_id = None
                    last_switch = t_rel
            else:
                pending_id = None
        else:
            pending_id = None

        active = by_id[active_id]
        faces = [t for t in live if t.subject_type == "face"]
        edge_clamped = False
        subject_type = active.subject_type

        # Two interacting people: frame both when 9:16 can actually hold them.
        if len(faces) == 2:
            left, right = sorted(faces, key=lambda t: t.center_x)
            span = (right.center_x + right.box[2] / 2.0) - (left.center_x - left.box[2] / 2.0)
            balanced = abs(weights[left.track_id] - weights[right.track_id]) <= cfg.dual_subject_balance * max(
                weights[left.track_id], weights[right.track_id], 1e-6
            )
            if span <= crop_w * (1.0 - 2.0 * cfg.edge_margin_ratio) and balanced:
                target_x = (left.center_x + right.center_x) / 2.0
                target_y = (left.center_y + right.center_y) / 2.0 + crop_h * (0.5 - cfg.head_position_ratio)
                target_y = min(max(target_y, crop_h / 2.0), source_height - crop_h / 2.0)
                diagnostics.dual_subject_frames += 1
                diagnostics.frames_with_active_subject += 1
                observations.append(
                    _Observation(
                        t_rel,
                        target_x,
                        target_y,
                        FALLBACK_DUAL,
                        FALLBACK_DUAL,
                        scene_cut,
                        active_id,
                        [t.box for t in live],
                        [t.track_id for t in live],
                        False,
                    )
                )
                last_target = last_subject_target = (target_x, target_y)
                last_subject_time = t_rel
                continue

        target_x, target_y, edge_clamped = _framing_target(
            active, crop_w, crop_h, source_width, source_height, cfg
        )
        if edge_clamped:
            diagnostics.edge_clamped_frames += 1
        diagnostics.frames_with_active_subject += 1

        observations.append(
            _Observation(
                t_rel,
                target_x,
                target_y,
                subject_type,
                FALLBACK_NONE,
                scene_cut,
                active_id,
                [t.box for t in live],
                [t.track_id for t in live],
                edge_clamped,
            )
        )
        last_target = last_subject_target = (target_x, target_y)
        last_subject_time = t_rel

    capture.release()
    return observations


def _smooth(
    observations: List[_Observation],
    crop_w: int,
    crop_h: int,
    source_width: int,
    source_height: int,
    cfg: ReframeConfig,
) -> List[CropPoint]:
    """Convert raw framing targets into an operator-like crop trajectory.

    Dead-zone, proportional tracking, velocity and acceleration clamps remove shake and
    micro-jitter; a scene cut is allowed to re-anchor instantly.
    """
    points: List[CropPoint] = []
    deadzone = cfg.deadzone_ratio * source_width

    cur_x = cur_y = 0.0
    vel_x = vel_y = 0.0
    prev_t = 0.0
    last_emitted_x: Optional[int] = None
    last_emitted_y: Optional[int] = None

    for idx, obs in enumerate(observations):
        if idx == 0:
            cur_x, cur_y = obs.target_x, obs.target_y
            vel_x = vel_y = 0.0
        elif obs.scene_cut:
            cur_x, cur_y = obs.target_x, obs.target_y
            vel_x = vel_y = 0.0
        else:
            dt = max(1e-3, obs.time - prev_t)
            for axis in ("x", "y"):
                cur = cur_x if axis == "x" else cur_y
                vel = vel_x if axis == "x" else vel_y
                target = obs.target_x if axis == "x" else obs.target_y

                error = target - cur
                if abs(error) <= deadzone:
                    effective = 0.0
                else:
                    effective = error - (deadzone if error > 0 else -deadzone)

                desired_vel = (effective * cfg.smoothing_alpha) / dt
                desired_vel = min(max(desired_vel, -cfg.max_velocity_px_per_sec), cfg.max_velocity_px_per_sec)
                max_dv = cfg.max_acceleration_px_per_sec2 * dt
                vel += min(max(desired_vel - vel, -max_dv), max_dv)
                cur += vel * dt

                if axis == "x":
                    cur_x, vel_x = cur, vel
                else:
                    cur_y, vel_y = cur, vel
        prev_t = obs.time

        if not (math.isfinite(cur_x) and math.isfinite(cur_y)):
            # A non-finite target would poison every later sample; re-anchor at centre instead.
            logger.warning(f"[reframe] Non-finite crop centre at t={obs.time:.2f}s; re-anchoring")
            cur_x, cur_y = source_width / 2.0, source_height / 2.0
            vel_x = vel_y = 0.0

        raw_x = int(round(cur_x - crop_w / 2.0))
        raw_y = int(round(cur_y - crop_h / 2.0))
        crop_x = ((min(max(raw_x, 0), source_width - crop_w)) // 2) * 2
        crop_y = ((min(max(raw_y, 0), source_height - crop_h)) // 2) * 2

        # Suppress residual few-pixel hops that read as jitter rather than camera movement.
        if last_emitted_x is not None and abs(crop_x - last_emitted_x) < cfg.jitter_epsilon_px:
            crop_x = last_emitted_x
        if last_emitted_y is not None and abs(crop_y - last_emitted_y) < cfg.jitter_epsilon_px:
            crop_y = last_emitted_y
        last_emitted_x, last_emitted_y = crop_x, crop_y

        points.append(
            CropPoint(
                time=round(obs.time, 3),
                center_x=round(cur_x, 2),
                center_y=round(cur_y, 2),
                crop_x=crop_x,
                crop_y=crop_y,
                crop_w=crop_w,
                crop_h=crop_h,
                subject_type=obs.subject_type,
            )
        )

    return points


def _trajectory_stats(points: List[CropPoint]) -> TrajectoryStats:
    if not points:
        return TrajectoryStats()

    velocities: List[float] = []
    travel = 0.0
    stationary = 0
    for i in range(1, len(points)):
        dt = max(1e-3, points[i].time - points[i - 1].time)
        dx = abs(points[i].crop_x - points[i - 1].crop_x)
        travel += dx
        velocities.append(dx / dt)
        if dx == 0:
            stationary += 1

    xs = [p.crop_x for p in points]
    ys = [p.crop_y for p in points]
    denominator = max(1, len(points) - 1)
    return TrajectoryStats(
        mean_velocity_px_per_sec=round(sum(velocities) / len(velocities), 3) if velocities else 0.0,
        max_velocity_px_per_sec=round(max(velocities), 3) if velocities else 0.0,
        total_travel_px=round(travel, 2),
        stationary_ratio=round(stationary / denominator, 4),
        crop_x_min=min(xs),
        crop_x_max=max(xs),
        crop_x_range=max(xs) - min(xs),
        crop_y_min=min(ys),
        crop_y_max=max(ys),
    )


def build_reframe_plan(
    video_path: Path | str,
    source_start_sec: float,
    duration_sec: float,
    detector: Optional[SubjectDetector] = None,
    config: Optional[ReframeConfig] = None,
    detector_name: str = "haar",
    source_width: Optional[int] = None,
    source_height: Optional[int] = None,
    collect_debug: bool = False,
) -> ReframePlan:
    """Analyze a clip and produce a smoothed 9:16 crop trajectory with diagnostics.

    Never raises for detector or decode problems: it degrades to previous stable crop,
    dominant region, and finally static center crop so a valid 9:16 render is always possible.
    """
    cfg = config or ReframeConfig()
    started = time.perf_counter()
    path = Path(video_path)

    width, height = source_width or 0, source_height or 0
    if width <= 0 or height <= 0:
        try:
            import cv2

            probe = cv2.VideoCapture(str(path))
            if probe.isOpened():
                width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            probe.release()
        except Exception as exc:
            logger.warning(f"[reframe] Unable to probe {path}: {exc}")

    if width <= 0 or height <= 0:
        from freecher_worker.media.probe import probe_media

        try:
            info = probe_media(path)
            width, height = info.width, info.height
        except Exception as exc:
            logger.warning(f"[reframe] ffprobe fallback failed for {path}: {exc}")
            width, height = 1920, 1080

    crop_w, crop_h = calculate_vertical_crop(width, height)
    diagnostics = ReframeDiagnostics(detector=detector_name, analysis_fps=cfg.analysis_fps)

    try:
        active_detector = detector or get_subject_detector(detector_name)
        observations = _collect_observations(
            video_path=path,
            start_seconds=source_start_sec,
            duration_seconds=duration_sec,
            detector=active_detector,
            cfg=cfg,
            crop_w=crop_w,
            crop_h=crop_h,
            source_width=width,
            source_height=height,
            diagnostics=diagnostics,
        )
    except Exception as exc:
        logger.warning(f"[reframe] Analysis failed ({exc}); falling back to static center crop")
        plan = build_center_crop_plan(width, height, duration_sec, cfg.analysis_fps, str(exc), detector_name)
        plan.diagnostics.analysis_seconds = round(time.perf_counter() - started, 3)
        return plan

    if not observations:
        plan = build_center_crop_plan(
            width, height, duration_sec, cfg.analysis_fps, "no frames could be sampled", detector_name
        )
        plan.diagnostics.analysis_seconds = round(time.perf_counter() - started, 3)
        return plan

    points = _smooth(observations, crop_w, crop_h, width, height, cfg)
    diagnostics.fallback_used = (
        diagnostics.fallback_previous_frames
        + diagnostics.fallback_dominant_frames
        + diagnostics.fallback_center_frames
    ) > 0
    diagnostics.trajectory = _trajectory_stats(points)
    diagnostics.analysis_seconds = round(time.perf_counter() - started, 3)

    debug_samples: List[DebugSample] = []
    if collect_debug:
        for obs, point in zip(observations, points):
            debug_samples.append(
                DebugSample(
                    time=round(obs.time, 3),
                    boxes=obs.boxes,
                    track_ids=obs.track_ids,
                    active_track_id=obs.active_track_id,
                    target_x=round(obs.target_x, 2),
                    target_y=round(obs.target_y, 2),
                    crop_x=point.crop_x,
                    crop_y=point.crop_y,
                    fallback=obs.fallback,
                    scene_cut=obs.scene_cut,
                )
            )

    return ReframePlan(
        mode=REFRAME_MODE_SMART,
        trajectory=CropTrajectory(
            source_width=width,
            source_height=height,
            crop_w=crop_w,
            crop_h=crop_h,
            points=points,
        ),
        diagnostics=diagnostics,
        debug_samples=debug_samples,
    )


def render_debug_overlay(
    video_path: Path | str,
    plan: ReframePlan,
    source_start_sec: float,
    output_path: Path | str,
) -> Optional[Path]:
    """Render a diagnostic video with detection boxes, crop rectangle and active subject.

    Silent, source-resolution, sampled at the analysis rate. Diagnostics only — never published.
    """
    if not plan.debug_samples:
        logger.warning("[reframe] Debug overlay requested but no debug samples were collected")
        return None

    import cv2

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        logger.warning(f"[reframe] Debug overlay could not open {video_path}")
        return None

    traj = plan.trajectory
    fps = max(1.0, plan.diagnostics.analysis_fps)
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (traj.source_width, traj.source_height)
    )

    for sample in plan.debug_samples:
        capture.set(cv2.CAP_PROP_POS_MSEC, (source_start_sec + sample.time) * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue

        for box, track_id in zip(sample.boxes, sample.track_ids):
            x, y, w, h = box
            is_active = track_id == sample.active_track_id
            color = (0, 255, 0) if is_active else (180, 180, 180)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3 if is_active else 1)
            cv2.putText(
                frame, f"#{track_id}", (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        cv2.rectangle(
            frame,
            (sample.crop_x, sample.crop_y),
            (sample.crop_x + traj.crop_w, sample.crop_y + traj.crop_h),
            (0, 128, 255),
            3,
        )
        cv2.drawMarker(
            frame,
            (int(sample.crop_x + traj.crop_w / 2), int(traj.source_height / 2)),
            (0, 128, 255),
            cv2.MARKER_CROSS,
            28,
            2,
        )
        cv2.drawMarker(
            frame, (int(sample.target_x), int(sample.target_y)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2
        )
        label = f"t={sample.time:.2f}s  mode={sample.fallback}" + ("  SCENE CUT" if sample.scene_cut else "")
        cv2.putText(frame, label, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    capture.release()
    logger.info(f"[reframe] Debug overlay written to {out}")
    return out
