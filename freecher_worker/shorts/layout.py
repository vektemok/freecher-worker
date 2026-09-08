"""Adaptive vertical layout planning — decide *what shape* a moment needs, not just where to point.

A single 9:16 crop is the right answer for one dominant speaker and the wrong answer for a
two-person exchange or a group/context shot: it silently deletes people from the scene. The
planner therefore sits between visual analysis and rendering, and emits a declarative plan:

    final subclip -> visual analysis -> subject tracks -> layout planner -> layout plan -> renderer

It renders nothing itself. Three layouts exist in v1:

``single_subject``
    one dominant persistent subject, framed by the existing smart crop.
``dual_stack``
    two persistent subjects that cannot share one safe vertical crop, stacked in two viewports.
``full_frame_context``
    the complete source frame, fitted inside 9:16 over a blurred background — used for groups,
    ambiguous composition, and whenever the evidence is too thin to crop safely.

The product rule behind every tie-break:

    LOSS OF CONTEXT IS WORSE THAN A SMALLER SUBJECT.

A slightly smaller person is recoverable; an important participant cropped out of the frame is not.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from freecher_worker.crop.models import CropPoint, CropTrajectory

from .reframe import (
    FramingTarget,
    FrameObservation,
    ReframeConfig,
    TrackObservation,
    framing_target_for_box,
    smooth_crop_points,
)

logger = logging.getLogger("freecher_worker")

LAYOUT_VERSION = "adaptive_layout_v1"

#: Layouts a segment can request.
LAYOUT_SINGLE = "single_subject"
LAYOUT_DUAL = "dual_stack"
LAYOUT_FULL = "full_frame_context"
AVAILABLE_LAYOUTS = (LAYOUT_SINGLE, LAYOUT_DUAL, LAYOUT_FULL)

#: What the operator asks for on the command line.
LAYOUT_MODE_SINGLE = "single"
LAYOUT_MODE_ADAPTIVE = "adaptive"
LAYOUT_MODE_FULL_FRAME = "full-frame"
AVAILABLE_LAYOUT_MODES = (LAYOUT_MODE_SINGLE, LAYOUT_MODE_ADAPTIVE, LAYOUT_MODE_FULL_FRAME)

#: Viewport names. They double as FFmpeg filter instance ids (``crop@single`` and friends).
VIEWPORT_SINGLE = "single"
VIEWPORT_DUAL_TOP = "dual_top"
VIEWPORT_DUAL_BOTTOM = "dual_bottom"


class LayoutConfig(BaseModel):
    """Thresholds for layout planning. Every number here is configurable; none is inlined."""

    # --- analysis windows --------------------------------------------------
    window_sec: float = Field(default=0.75, gt=0.0, description="Length of one decision window")

    # --- what counts as a real subject ------------------------------------
    persistent_min_visible_sec: float = Field(
        default=1.0, ge=0.0, description="A track seen for less than this is noise, not a subject"
    )
    persistent_min_visibility_ratio: float = Field(
        default=0.20, ge=0.0, le=1.0, description="Minimum share of the clip a persistent track is visible for"
    )
    persistent_min_hits: int = Field(default=4, ge=1, description="Minimum detections behind a persistent track")
    significant_min_area_ratio: float = Field(
        default=0.0012, ge=0.0, description="Minimum mean box area as a fraction of the source frame"
    )

    # --- per-window subject thresholds ------------------------------------
    dominant_min_visibility: float = Field(default=0.65, ge=0.0, le=1.0)
    secondary_min_visibility: float = Field(default=0.45, ge=0.0, le=1.0)
    group_min_subjects: int = Field(default=3, ge=2, description="Significant subjects that make it a group scene")
    min_detection_coverage: float = Field(
        default=0.35, ge=0.0, le=1.0, description="Below this, framing evidence is too thin to crop"
    )
    min_tracking_confidence: float = Field(default=0.45, ge=0.0, le=1.0)

    # --- single-crop feasibility ------------------------------------------
    min_face_height_ratio: float = Field(
        default=0.055,
        ge=0.0,
        description="A subject smaller than this share of the output height is not readable",
    )
    safe_top_ratio: float = Field(
        default=0.06, ge=0.0, le=0.5, description="Top band reserved for platform UI"
    )
    safe_bottom_ratio: float = Field(
        default=0.24, ge=0.0, le=0.6, description="Bottom band reserved for captions and platform UI"
    )

    # --- temporal stability -----------------------------------------------
    min_layout_duration_sec: float = Field(default=2.5, gt=0.0)
    switch_confirmation_sec: float = Field(
        default=1.5, ge=0.0, description="A new layout must hold for this long before it is accepted"
    )
    subject_missing_grace_sec: float = Field(
        default=1.2, ge=0.0, description="A briefly lost subject still counts as present"
    )
    layout_switch_penalty: float = Field(
        default=0.08,
        ge=0.0,
        description="Confidence the challenger must beat the incumbent by; keeps near-ties stable",
    )
    scene_cut_allows_immediate_switch: bool = Field(default=True)

    # --- presentation ------------------------------------------------------
    full_frame_blur_background: bool = Field(
        default=True, description="Fill the unused vertical space with a blurred copy of the frame"
    )
    full_frame_blur_sigma: float = Field(default=12.0, ge=0.0)
    full_frame_background_color: str = Field(default="black", description="Used when blur is disabled")


class SemanticContext(BaseModel):
    """Hints from a future semantic-framing stage.

    Deliberately not wired to ``multimodal_v1_1``: highlight intelligence and framing
    intelligence stay separate bounded components. v1 only honours what is filled in here.
    """

    important_subject_ids: List[str] = Field(default_factory=list)
    important_object: Optional[Tuple[int, int, int, int]] = None
    speaker_hint: Optional[str] = None
    multimodal_layout_hint: Optional[str] = None
    prefer_full_frame: bool = False


class TrackSummary(BaseModel):
    """Temporal summary of one subject across the whole clip.

    Layout decisions are made from these, never from a single frame's bounding boxes: a face that
    flickers for two frames must not be able to create a second viewport.
    """

    track_id: int
    label: str
    subject_type: str = "face"
    first_seen: float = 0.0
    last_seen: float = 0.0
    frames_seen: int = 0
    visible_duration: float = 0.0
    visibility_ratio: float = 0.0
    screen_time: float = 0.0
    mean_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    mean_bbox_area: float = 0.0
    mean_center_x: float = 0.0
    mean_center_y: float = 0.0
    position_variance: float = 0.0
    detection_confidence: float = 0.0
    mean_speaking_score: float = 0.0
    continuity_score: float = 0.0
    scene_ids: List[int] = Field(default_factory=list)
    persistent: bool = False
    significant: bool = False

    @property
    def importance(self) -> float:
        """Deterministic ranking weight for dominant/secondary selection."""
        return (
            0.45 * self.visibility_ratio
            + 0.25 * min(1.0, self.mean_bbox_area / 40000.0)
            + 0.20 * self.mean_speaking_score
            + 0.10 * self.continuity_score
        )


class WindowFeatures(BaseModel):
    """Per-window evidence and the layout it argues for, before temporal smoothing."""

    index: int
    start: float
    end: float
    frame_count: int = 0
    face_count: float = 0.0
    persistent_face_count: int = 0
    dominant_track: Optional[int] = None
    secondary_track: Optional[int] = None
    dominant_visibility: float = 0.0
    secondary_visibility: float = 0.0
    horizontal_distance: float = 0.0
    union_box: Optional[Tuple[int, int, int, int]] = None
    single_crop_feasible: bool = False
    single_crop_reason: str = ""
    detection_coverage: float = 0.0
    tracking_confidence: float = 0.0
    frame_activity: float = 0.0
    scene_cut: bool = False
    layout: str = LAYOUT_FULL
    confidence: float = 0.0
    reason: str = ""


class LayoutSegment(BaseModel):
    """One contiguous stretch of the short rendered with a single layout."""

    start: float
    end: float
    layout: str
    subjects: List[str] = Field(default_factory=list)
    subject_ids: List[int] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class LayoutPlan(BaseModel):
    """Declarative layout plan for one short. The renderer consumes this and nothing else."""

    version: str = LAYOUT_VERSION
    mode_requested: str = LAYOUT_MODE_SINGLE
    duration: float = 0.0
    segments: List[LayoutSegment] = Field(default_factory=list)
    semantic_context: SemanticContext = Field(default_factory=SemanticContext)

    # --- observability -----------------------------------------------------
    switch_count: int = 0
    duration_by_mode: Dict[str, float] = Field(default_factory=dict)
    track_count: int = 0
    persistent_track_count: int = 0
    significant_track_count: int = 0
    dominant_track_id: Optional[int] = None
    mean_tracking_confidence: float = 0.0
    analysis_windows: int = 0
    fallback_reason: Optional[str] = None
    tracks: List[TrackSummary] = Field(default_factory=list)

    def layout_at(self, time: float) -> str:
        for segment in self.segments:
            if segment.start <= time < segment.end:
                return segment.layout
        return self.segments[-1].layout if self.segments else LAYOUT_SINGLE

    def segments_for(self, layout: str) -> List[LayoutSegment]:
        return [s for s in self.segments if s.layout == layout]

    def uses(self, layout: str) -> bool:
        return any(s.layout == layout for s in self.segments)


def calculate_aspect_crop(
    source_width: int, source_height: int, ratio_w: int, ratio_h: int
) -> Tuple[int, int]:
    """Largest even-sized ``ratio_w:ratio_h`` window that fits inside the source frame."""
    crop_w = int(round(((source_height * ratio_w) / float(ratio_h)) / 2.0)) * 2
    crop_h = source_height - (source_height % 2)
    if crop_w > source_width:
        crop_w = source_width - (source_width % 2)
        crop_h = int(round(((crop_w * ratio_h) / float(ratio_w)) / 2.0)) * 2
        crop_h = min(crop_h, source_height - (source_height % 2))
    return max(2, crop_w), max(2, crop_h)


def can_fit_subjects_in_single_vertical_crop(
    boxes: Sequence[Sequence[int]],
    crop_w: int,
    crop_h: int,
    source_width: int,
    source_height: int,
    output_height: int = 1920,
    config: Optional[LayoutConfig] = None,
    reframe: Optional[ReframeConfig] = None,
) -> Tuple[bool, str]:
    """Can one 9:16 crop hold every one of these subjects, legibly and safely?

    Three things have to be true at once, and a "no" to any of them is what makes a stacked or
    full-frame layout the honest answer:

    1. the padded union of the subjects fits horizontally inside the crop window;
    2. no subject shrinks below the readable size once the crop is scaled to the output;
    3. every face lands inside the safe band, clear of the caption and platform-UI zones.
    """
    cfg = config or LayoutConfig()
    rcfg = reframe or ReframeConfig()
    valid = [tuple(int(v) for v in box) for box in boxes if box is not None and box[2] > 0 and box[3] > 0]
    if not valid:
        return False, "no subject boxes"

    margin = rcfg.edge_margin_ratio * crop_w
    left = min(b[0] for b in valid)
    right = max(b[0] + b[2] for b in valid)
    if (right - left) + 2.0 * margin > crop_w:
        return False, "subjects wider than a safe vertical crop"

    # Everything is scaled by output_height / crop_h on the way to the final frame.
    scale = output_height / float(max(1, crop_h))
    smallest = min(b[3] for b in valid)
    if (smallest * scale) / float(output_height) < cfg.min_face_height_ratio:
        return False, "a subject would be too small to read"

    # Place the crop window vertically the way the framing stage would, then check the safe band.
    top = min(b[1] for b in valid)
    bottom = max(b[1] + b[3] for b in valid)
    centre_y = (top + bottom) / 2.0
    crop_y = centre_y + crop_h * (0.5 - rcfg.head_position_ratio) - crop_h / 2.0
    crop_y = min(max(crop_y, 0.0), max(0.0, source_height - crop_h))
    safe_top = cfg.safe_top_ratio * crop_h
    safe_bottom = (1.0 - cfg.safe_bottom_ratio) * crop_h
    for box in valid:
        face_centre = (box[1] + box[3] / 2.0) - crop_y
        if face_centre < safe_top or face_centre > safe_bottom:
            return False, "a subject would fall into the caption or platform-UI band"

    return True, "subjects fit one safe vertical crop"


# ---------------------------------------------------------------------------
# Track summaries
# ---------------------------------------------------------------------------


def summarize_tracks(
    frames: Sequence[FrameObservation],
    source_width: int,
    source_height: int,
    config: Optional[LayoutConfig] = None,
) -> List[TrackSummary]:
    """Aggregate per-frame detections into one temporal record per subject."""
    cfg = config or LayoutConfig()
    if not frames:
        return []

    total_frames = len(frames)
    duration = max(f.time for f in frames) - min(f.time for f in frames)
    frame_interval = duration / max(1, total_frames - 1) if total_frames > 1 else 0.0

    scene_index = 0
    per_track: Dict[int, Dict[str, list]] = {}
    for frame in frames:
        if frame.scene_cut:
            scene_index += 1
        for track in frame.tracks:
            bucket = per_track.setdefault(
                track.track_id,
                {"times": [], "boxes": [], "conf": [], "speaking": [], "scenes": [], "type": [], "hits": []},
            )
            bucket["times"].append(frame.time)
            bucket["boxes"].append(tuple(track.box))
            bucket["conf"].append(track.confidence)
            bucket["speaking"].append(track.speaking_score)
            bucket["scenes"].append(scene_index)
            bucket["type"].append(track.subject_type)
            bucket["hits"].append(track.hits)

    frame_area = float(max(1, source_width * source_height))
    summaries: List[TrackSummary] = []
    for track_id in sorted(per_track):
        bucket = per_track[track_id]
        times = bucket["times"]
        boxes = bucket["boxes"]
        seen = len(times)
        span = max(times) - min(times)
        centres_x = [b[0] + b[2] / 2.0 for b in boxes]
        centres_y = [b[1] + b[3] / 2.0 for b in boxes]
        mean_cx = sum(centres_x) / seen
        mean_cy = sum(centres_y) / seen
        variance = sum((c - mean_cx) ** 2 for c in centres_x) / seen
        mean_box = tuple(int(round(sum(b[i] for b in boxes) / seen)) for i in range(4))
        area = float(mean_box[2] * mean_box[3])
        # Visible duration counts each observed sample, so a track seen on one frame is not
        # credited with the whole gap it happens to span.
        visible = seen * frame_interval if frame_interval > 0 else float(seen)
        types = bucket["type"]
        summary = TrackSummary(
            track_id=track_id,
            label=f"track_{track_id}",
            subject_type=max(set(types), key=types.count),
            first_seen=round(min(times), 3),
            last_seen=round(max(times), 3),
            frames_seen=seen,
            visible_duration=round(visible, 3),
            visibility_ratio=round(seen / total_frames, 4),
            screen_time=round(visible, 3),
            mean_box=mean_box,  # type: ignore[arg-type]
            mean_bbox_area=round(area, 2),
            mean_center_x=round(mean_cx, 2),
            mean_center_y=round(mean_cy, 2),
            position_variance=round(variance, 2),
            detection_confidence=round(sum(bucket["conf"]) / seen, 4),
            mean_speaking_score=round(sum(bucket["speaking"]) / seen, 4),
            continuity_score=round(seen / max(1.0, (span / frame_interval) + 1.0), 4)
            if frame_interval > 0
            else 1.0,
            scene_ids=sorted(set(bucket["scenes"])),
        )
        summary.persistent = (
            summary.visible_duration >= cfg.persistent_min_visible_sec
            and summary.visibility_ratio >= cfg.persistent_min_visibility_ratio
            and max(bucket["hits"]) >= cfg.persistent_min_hits
        )
        summary.significant = summary.persistent and (area / frame_area) >= cfg.significant_min_area_ratio
        summaries.append(summary)

    return summaries


# ---------------------------------------------------------------------------
# Window features and per-window decisions
# ---------------------------------------------------------------------------


def _window_bounds(duration: float, window_sec: float) -> List[Tuple[float, float]]:
    count = max(1, int(math.ceil(duration / window_sec - 1e-9)))
    bounds = []
    for i in range(count):
        start = i * window_sec
        end = min(duration, start + window_sec)
        if end - start <= 1e-6 and bounds:
            break
        bounds.append((round(start, 4), round(end, 4)))
    if bounds:
        bounds[-1] = (bounds[-1][0], round(duration, 4))
    return bounds


def _visible_boxes(
    frames: Sequence[FrameObservation],
) -> Dict[int, List[TrackObservation]]:
    grouped: Dict[int, List[TrackObservation]] = {}
    for frame in frames:
        for track in frame.tracks:
            grouped.setdefault(track.track_id, []).append(track)
    return grouped


def _decide_window(
    features: WindowFeatures,
    fitting: Tuple[bool, str],
    significant_now: List[int],
    config: LayoutConfig,
    semantic: SemanticContext,
) -> Tuple[str, float, str]:
    """Rules for one window. Order matters: safety rungs are checked before cropping rungs."""
    if semantic.prefer_full_frame:
        return LAYOUT_FULL, 0.9, "semantic context asked for the full frame"

    if features.frame_count == 0:
        return LAYOUT_FULL, 0.5, "no frames were analyzed in this window"

    if features.detection_coverage < config.min_detection_coverage:
        return LAYOUT_FULL, 0.6, "detection coverage too low to crop safely"

    if features.tracking_confidence < config.min_tracking_confidence:
        return LAYOUT_FULL, 0.6, "tracking confidence too low to crop safely"

    if len(significant_now) >= config.group_min_subjects:
        return LAYOUT_FULL, 0.75, f"group scene: {len(significant_now)} significant subjects"

    if (
        features.secondary_track is not None
        and features.dominant_visibility >= config.secondary_min_visibility
        and features.secondary_visibility >= config.secondary_min_visibility
    ):
        if fitting[0]:
            return LAYOUT_SINGLE, 0.8, f"two subjects share one safe crop ({fitting[1]})"
        return LAYOUT_DUAL, 0.82, f"two persistent subjects cannot fit one safe crop ({fitting[1]})"

    if features.dominant_track is not None and features.dominant_visibility >= config.dominant_min_visibility:
        return LAYOUT_SINGLE, 0.85, "one dominant persistent subject"

    return LAYOUT_FULL, 0.55, "no dominant subject with enough evidence to crop"


def analyze_windows(
    frames: Sequence[FrameObservation],
    tracks: Sequence[TrackSummary],
    duration: float,
    crop_w: int,
    crop_h: int,
    source_width: int,
    source_height: int,
    output_height: int = 1920,
    config: Optional[LayoutConfig] = None,
    reframe: Optional[ReframeConfig] = None,
    semantic: Optional[SemanticContext] = None,
) -> List[WindowFeatures]:
    """Turn the analysis frames into one evidence record and one proposed layout per window."""
    cfg = config or LayoutConfig()
    rcfg = reframe or ReframeConfig()
    ctx = semantic or SemanticContext()
    by_id = {t.track_id: t for t in tracks}
    significant_ids = {t.track_id for t in tracks if t.significant}
    preferred = {
        t.track_id for t in tracks if t.label in set(ctx.important_subject_ids)
    }

    seen_times: Dict[int, List[float]] = {}
    for frame in frames:
        for track in frame.tracks:
            seen_times.setdefault(track.track_id, []).append(frame.time)

    windows: List[WindowFeatures] = []
    for index, (start, end) in enumerate(_window_bounds(duration, cfg.window_sec)):
        in_window = [f for f in frames if start <= f.time < end] or [
            f for f in frames if abs(f.time - start) < cfg.window_sec
        ]
        features = WindowFeatures(index=index, start=start, end=end, frame_count=len(in_window))
        if not in_window:
            features.layout, features.confidence, features.reason = _decide_window(
                features, (False, "no frames"), [], cfg, ctx
            )
            windows.append(features)
            continue

        detected_frames = sum(1 for f in in_window if f.detected)
        features.detection_coverage = round(detected_frames / len(in_window), 4)
        confidences = [f.mean_confidence for f in in_window if f.detected]
        features.tracking_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        features.scene_cut = any(f.scene_cut for f in in_window)
        features.face_count = round(
            sum(len(f.tracks) for f in in_window) / len(in_window), 3
        )

        grouped = _visible_boxes(in_window)
        # A subject the detector drops for a moment must not tear the layout apart, so a track
        # lost less recently than the grace period still counts as present in this window.
        # The sighting that matters is the last one *before* this window - a track's overall
        # last_seen may lie in the future, which would silently disable the grace entirely.
        held: Dict[int, float] = {}
        for track_id in significant_ids:
            if track_id in grouped:
                continue
            previous = [t for t in seen_times.get(track_id, ()) if t < start]
            if not previous:
                continue
            if (start - previous[-1]) <= cfg.subject_missing_grace_sec:
                held[track_id] = cfg.secondary_min_visibility

        visibility: Dict[int, float] = {
            tid: round(len(obs) / len(in_window), 4) for tid, obs in grouped.items()
        }
        visibility.update(held)

        significant_now = sorted(
            tid for tid in visibility if tid in significant_ids and visibility[tid] > 0.0
        )
        features.persistent_face_count = len(significant_now)

        def rank(track_id: int) -> Tuple[float, int]:
            summary = by_id.get(track_id)
            base = summary.importance if summary else 0.0
            bonus = 0.5 if track_id in preferred else 0.0
            return (visibility.get(track_id, 0.0) * 0.5 + base + bonus, -track_id)

        ordered = sorted(significant_now, key=rank, reverse=True)
        if ordered:
            features.dominant_track = ordered[0]
            features.dominant_visibility = visibility.get(ordered[0], 0.0)
        if len(ordered) > 1:
            features.secondary_track = ordered[1]
            features.secondary_visibility = visibility.get(ordered[1], 0.0)

        boxes: List[Tuple[int, int, int, int]] = []
        for track_id in (features.dominant_track, features.secondary_track):
            if track_id is None:
                continue
            observations = grouped.get(track_id)
            if observations:
                latest = observations[-1]
                boxes.append(tuple(int(v) for v in latest.box))  # type: ignore[arg-type]
            elif by_id.get(track_id) is not None:
                boxes.append(by_id[track_id].mean_box)

        if len(boxes) >= 2:
            left = min(b[0] for b in boxes)
            right = max(b[0] + b[2] for b in boxes)
            top = min(b[1] for b in boxes)
            bottom = max(b[1] + b[3] for b in boxes)
            features.union_box = (left, top, right - left, bottom - top)
            centres = sorted(b[0] + b[2] / 2.0 for b in boxes)
            features.horizontal_distance = round(
                (centres[-1] - centres[0]) / float(max(1, source_width)), 4
            )

        fitting = (
            can_fit_subjects_in_single_vertical_crop(
                boxes, crop_w, crop_h, source_width, source_height, output_height, cfg, rcfg
            )
            if boxes
            else (False, "no subject boxes")
        )
        features.single_crop_feasible, features.single_crop_reason = fitting

        motion: List[float] = []
        previous: Dict[int, Tuple[float, float]] = {}
        for frame in in_window:
            for track in frame.tracks:
                if track.track_id in previous:
                    px, py = previous[track.track_id]
                    motion.append(math.hypot(track.center_x - px, track.center_y - py))
                previous[track.track_id] = (track.center_x, track.center_y)
        features.frame_activity = round(
            (sum(motion) / len(motion)) / float(max(1, source_width)), 5
        ) if motion else 0.0

        features.layout, features.confidence, features.reason = _decide_window(
            features, fitting, significant_now, cfg, ctx
        )
        windows.append(features)

    return windows


# ---------------------------------------------------------------------------
# Temporal smoothing into segments
# ---------------------------------------------------------------------------


def _segment_subjects(
    layout: str, window: WindowFeatures, by_id: Dict[int, TrackSummary]
) -> Tuple[List[int], List[str]]:
    if layout == LAYOUT_DUAL:
        ids = [window.dominant_track, window.secondary_track]
        ids = [i for i in ids if i is not None]
        # A stacked layout showing the same person twice is a bug, never a style choice.
        deduped: List[int] = []
        for track_id in ids:
            if track_id not in deduped:
                deduped.append(track_id)
        if len(deduped) < 2:
            return [], []
        # Left subject on top keeps the stack spatially consistent with the source frame.
        deduped.sort(key=lambda tid: (by_id[tid].mean_center_x if tid in by_id else 0.0, tid))
        return deduped, [f"track_{i}" for i in deduped]
    if layout == LAYOUT_SINGLE and window.dominant_track is not None:
        return [window.dominant_track], [f"track_{window.dominant_track}"]
    return [], []


def _switch_is_representable(
    layout: str, window: WindowFeatures, by_id: Dict[int, TrackSummary]
) -> bool:
    """Whether the renderer could actually honour this layout for this window."""
    if layout != LAYOUT_DUAL:
        return True
    return len(_segment_subjects(LAYOUT_DUAL, window, by_id)[0]) >= 2


def smooth_layout_decisions(
    windows: Sequence[WindowFeatures],
    duration: float,
    tracks: Sequence[TrackSummary],
    config: Optional[LayoutConfig] = None,
) -> List[LayoutSegment]:
    """Apply hysteresis so the layout describes the scene rather than chasing it.

    A viewer reads a layout change as an edit. Three of them in thirty seconds is direction;
    one every half-second is a fault. A challenger therefore has to win a *share* of the
    evidence over the last ``switch_confirmation_sec`` - beating the incumbent's share by
    ``layout_switch_penalty`` - and the incumbent has to have held for
    ``min_layout_duration_sec`` first. Comparing shares rather than single windows is what
    makes one dissenting window harmless in both directions: it neither flips the layout nor
    resets a challenger that is otherwise winning.

    A real scene cut makes the change free: the viewer is already being shown something new.
    """
    cfg = config or LayoutConfig()
    by_id = {t.track_id: t for t in tracks}
    if not windows:
        return [
            LayoutSegment(
                start=0.0,
                end=round(duration, 3),
                layout=LAYOUT_FULL,
                confidence=0.5,
                reason="no analysis windows; context-safe fallback",
            )
        ]

    accepted: List[str] = []
    current = windows[0].layout
    current_since = windows[0].start

    for index, window in enumerate(windows):
        proposal = window.layout
        if proposal != current and _switch_is_representable(proposal, window, by_id):
            # Weigh the last `switch_confirmation_sec` of evidence rather than trusting the
            # window in front of us: one dissenting window should not flip the layout, and one
            # dissenting window should not reset a challenger that is otherwise winning.
            lookback = window.start - cfg.switch_confirmation_sec
            support: Dict[str, float] = {}
            for previous in windows[: index + 1]:
                if previous.start < lookback - 1e-9:
                    continue
                support[previous.layout] = support.get(previous.layout, 0.0) + max(
                    previous.confidence, 1e-6
                )
            total = sum(support.values()) or 1.0
            challenger = support.get(proposal, 0.0) / total
            incumbent = support.get(current, 0.0) / total
            mature = (window.start - current_since) >= cfg.min_layout_duration_sec
            immediate = cfg.scene_cut_allows_immediate_switch and window.scene_cut
            if immediate or (mature and challenger >= incumbent + cfg.layout_switch_penalty):
                current = proposal
                current_since = window.start
        accepted.append(current)

    segments: List[LayoutSegment] = []
    for window, layout in zip(windows, accepted):
        if segments and segments[-1].layout == layout:
            segments[-1].end = window.end
            if window.confidence > segments[-1].confidence:
                segments[-1].confidence = window.confidence
            if layout in (LAYOUT_DUAL, LAYOUT_SINGLE) and not segments[-1].subject_ids:
                ids, labels = _segment_subjects(layout, window, by_id)
                segments[-1].subject_ids, segments[-1].subjects = ids, labels
            continue
        ids, labels = _segment_subjects(layout, window, by_id)
        reason = window.reason
        if layout == LAYOUT_DUAL and len(ids) < 2:
            # Two viewports need two different people. Without them, context beats a bad crop.
            layout, ids, labels = LAYOUT_FULL, [], []
            reason = "dual stack needs two distinct subjects; fell back to full frame"
        segments.append(
            LayoutSegment(
                start=window.start,
                end=window.end,
                layout=layout,
                subject_ids=ids,
                subjects=labels,
                confidence=window.confidence,
                reason=reason,
            )
        )

    return _enforce_minimum_durations(segments, duration, cfg)


def _enforce_minimum_durations(
    segments: List[LayoutSegment], duration: float, cfg: LayoutConfig
) -> List[LayoutSegment]:
    """Absorb segments too short to read, then make the plan exactly cover the clip."""
    changed = True
    while changed and len(segments) > 1:
        changed = False
        for index, segment in enumerate(segments):
            if segment.duration >= cfg.min_layout_duration_sec:
                continue
            if index == 0:
                segments[1].start = segment.start
            elif index == len(segments) - 1:
                segments[index - 1].end = segment.end
            else:
                previous, following = segments[index - 1], segments[index + 1]
                # Give the orphan to whichever neighbour is already carrying more of the clip.
                if previous.duration >= following.duration:
                    previous.end = segment.end
                else:
                    following.start = segment.start
            segments.pop(index)
            changed = True
            break

    merged: List[LayoutSegment] = []
    for segment in segments:
        if merged and merged[-1].layout == segment.layout and merged[-1].subject_ids == segment.subject_ids:
            merged[-1].end = segment.end
            continue
        merged.append(segment)

    if merged:
        merged[0].start = 0.0
        merged[-1].end = round(duration, 3)
        for previous, following in zip(merged, merged[1:]):
            following.start = previous.end
        for segment in merged:
            segment.start = round(max(0.0, segment.start), 3)
            segment.end = round(min(duration, segment.end), 3)
    return merged


# ---------------------------------------------------------------------------
# Planner entry point
# ---------------------------------------------------------------------------


def build_layout_plan(
    frames: Sequence[FrameObservation],
    duration: float,
    source_width: int,
    source_height: int,
    crop_w: int,
    crop_h: int,
    output_height: int = 1920,
    mode: str = LAYOUT_MODE_ADAPTIVE,
    config: Optional[LayoutConfig] = None,
    reframe: Optional[ReframeConfig] = None,
    semantic: Optional[SemanticContext] = None,
) -> LayoutPlan:
    """Plan the layout of one short. Deterministic: identical inputs give an identical plan."""
    cfg = config or LayoutConfig()
    ctx = semantic or SemanticContext()
    duration = float(max(0.0, duration))

    if mode == LAYOUT_MODE_SINGLE:
        return _uniform_plan(LAYOUT_SINGLE, duration, mode, ctx, "layout mode 'single' requested")
    if mode == LAYOUT_MODE_FULL_FRAME:
        return _uniform_plan(LAYOUT_FULL, duration, mode, ctx, "layout mode 'full-frame' requested")

    if not frames:
        plan = _uniform_plan(
            LAYOUT_FULL, duration, mode, ctx, "no visual analysis frames; context-safe fallback"
        )
        plan.fallback_reason = "no analysis frames"
        return plan

    tracks = summarize_tracks(frames, source_width, source_height, cfg)
    windows = analyze_windows(
        frames=frames,
        tracks=tracks,
        duration=duration,
        crop_w=crop_w,
        crop_h=crop_h,
        source_width=source_width,
        source_height=source_height,
        output_height=output_height,
        config=cfg,
        reframe=reframe,
        semantic=ctx,
    )
    segments = smooth_layout_decisions(windows, duration, tracks, cfg)

    durations: Dict[str, float] = {layout: 0.0 for layout in AVAILABLE_LAYOUTS}
    for segment in segments:
        durations[segment.layout] = round(durations.get(segment.layout, 0.0) + segment.duration, 3)

    detected = [f.mean_confidence for f in frames if f.detected]
    dominant = max(tracks, key=lambda t: (t.importance, -t.track_id), default=None)

    plan = LayoutPlan(
        mode_requested=mode,
        duration=round(duration, 3),
        segments=segments,
        semantic_context=ctx,
        switch_count=max(0, len(segments) - 1),
        duration_by_mode=durations,
        track_count=len(tracks),
        persistent_track_count=sum(1 for t in tracks if t.persistent),
        significant_track_count=sum(1 for t in tracks if t.significant),
        dominant_track_id=dominant.track_id if dominant else None,
        mean_tracking_confidence=round(sum(detected) / len(detected), 4) if detected else 0.0,
        analysis_windows=len(windows),
        tracks=tracks,
    )
    logger.info(
        f"[layout] {LAYOUT_VERSION}: {len(segments)} segment(s), {plan.switch_count} switch(es); "
        + ", ".join(f"{k}={v:.1f}s" for k, v in durations.items() if v > 0)
        + f"; tracks={plan.track_count} persistent={plan.persistent_track_count}"
    )
    return plan


def _uniform_plan(
    layout: str, duration: float, mode: str, semantic: SemanticContext, reason: str
) -> LayoutPlan:
    return LayoutPlan(
        mode_requested=mode,
        duration=round(duration, 3),
        segments=[
            LayoutSegment(
                start=0.0, end=round(duration, 3), layout=layout, confidence=1.0, reason=reason
            )
        ],
        semantic_context=semantic,
        switch_count=0,
        duration_by_mode={layout: round(duration, 3)},
    )


# ---------------------------------------------------------------------------
# Viewport trajectories
# ---------------------------------------------------------------------------


class ViewportPlan(BaseModel):
    """One rectangular window into the source, tracking one subject over the whole clip."""

    name: str
    trajectory: CropTrajectory
    track_ids: List[int] = Field(default_factory=list)


def build_viewport_trajectory(
    frames: Sequence[FrameObservation],
    assignment: Dict[int, Optional[int]],
    viewport_w: int,
    viewport_h: int,
    source_width: int,
    source_height: int,
    name: str,
    reframe: Optional[ReframeConfig] = None,
) -> ViewportPlan:
    """Follow one subject per frame through its own viewport.

    ``assignment`` maps a frame index to the track that viewport should hold. Frames with no
    assignment (or whose subject is momentarily undetected) hold the last good framing, so a
    viewport never lurches when the detector blinks.
    """
    rcfg = reframe or ReframeConfig()
    targets: List[FramingTarget] = []
    last: Optional[Tuple[float, float]] = None
    used: List[int] = []

    for index, frame in enumerate(frames):
        track_id = assignment.get(index)
        box = None
        if track_id is not None:
            box = next((t.box for t in frame.tracks if t.track_id == track_id), None)
            if track_id not in used:
                used.append(track_id)
        if box is not None:
            x, y, _ = framing_target_for_box(
                box, viewport_w, viewport_h, source_width, source_height, rcfg
            )
            last = (x, y)
        elif last is None:
            last = (source_width / 2.0, source_height / 2.0)
        targets.append(
            FramingTarget(
                time=frame.time,
                x=last[0],
                y=last[1],
                scene_cut=frame.scene_cut,
                subject_type=name,
            )
        )

    points: List[CropPoint] = smooth_crop_points(
        targets, viewport_w, viewport_h, source_width, source_height, rcfg
    )
    return ViewportPlan(
        name=name,
        trajectory=CropTrajectory(
            source_width=source_width,
            source_height=source_height,
            crop_w=viewport_w,
            crop_h=viewport_h,
            points=points,
        ),
        track_ids=used,
    )


def build_dual_viewports(
    plan: LayoutPlan,
    frames: Sequence[FrameObservation],
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    reframe: Optional[ReframeConfig] = None,
) -> Optional[Tuple[ViewportPlan, ViewportPlan]]:
    """Build the two stacked viewports, or ``None`` when the plan never uses ``dual_stack``."""
    dual_segments = plan.segments_for(LAYOUT_DUAL)
    if not dual_segments or not frames:
        return None

    viewport_h = output_height // 2
    if viewport_h * 2 != output_height:
        logger.warning("[layout] Output height is not divisible by two; dual_stack unavailable")
        return None

    crop_w, crop_h = calculate_aspect_crop(source_width, source_height, output_width, viewport_h)
    top: Dict[int, Optional[int]] = {}
    bottom: Dict[int, Optional[int]] = {}
    for index, frame in enumerate(frames):
        segment = next(
            (s for s in dual_segments if s.start <= frame.time < s.end), None
        )
        if segment is None or len(segment.subject_ids) < 2:
            continue
        top[index] = segment.subject_ids[0]
        bottom[index] = segment.subject_ids[1]

    if not top:
        return None

    return (
        build_viewport_trajectory(
            frames, top, crop_w, crop_h, source_width, source_height, VIEWPORT_DUAL_TOP, reframe
        ),
        build_viewport_trajectory(
            frames, bottom, crop_w, crop_h, source_width, source_height, VIEWPORT_DUAL_BOTTOM, reframe
        ),
    )
