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

TRACKING_MODE_SUBJECT = "subject"
TRACKING_MODE_MIXED = "mixed"
TRACKING_MODE_DOMINANT = "dominant_region"
TRACKING_MODE_CENTER = "center"

FALLBACK_NONE = "subject"
FALLBACK_DUAL = "dual_subject"
FALLBACK_PREVIOUS = "previous_stable"
FALLBACK_DOMINANT = "dominant_region"
FALLBACK_CENTER = "center_crop"


class ReframeConfig(BaseModel):
    """Tunable parameters for detection, subject arbitration, framing and smoothing."""

    analysis_fps: float = Field(default=5.0, gt=0.0)
    detect_max_width: int = Field(default=960, gt=0)
    face_score_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    face_model_path: Optional[str] = Field(default=None)
    allow_model_download: bool = Field(default=True)
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
    track_max_misses: int = Field(
        default=6,
        ge=0,
        description="Missed samples before the crop stops holding a lost subject's framing. "
                    "Identity lifetime is governed by track_max_gap_sec, not by this",
    )
    track_match_iou: float = Field(default=0.15, ge=0.0, le=1.0)
    track_max_motion_px_per_sec: float = Field(
        default=600.0, gt=0.0, description="How far a subject may plausibly move between samples"
    )
    track_min_reach_px: float = Field(
        default=28.0, ge=0.0, description="Association floor, so small subjects stay trackable"
    )
    track_max_size_ratio: float = Field(
        default=3.0,
        gt=1.0,
        description="Reject an association whose box is this many times larger/smaller than the track",
    )
    track_predict_motion: bool = Field(
        default=True,
        description="Associate against the track's predicted position rather than its last one",
    )
    track_max_gap_sec: float = Field(
        default=1.6,
        ge=0.0,
        description="How long a subject may stay undetected and still recover its identity",
    )
    track_confirm_hits: int = Field(
        default=3,
        ge=1,
        description="Detections before a candidate becomes a real identity rather than a blip",
    )
    track_max_reassociation_px: float = Field(
        default=520.0,
        gt=0.0,
        description="Hard ceiling on how far a recovered identity may jump; stops people merging",
    )
    track_appearance_enabled: bool = Field(
        default=True,
        description="Compare a tiny normalized patch of the subject to confirm re-attachment",
    )
    track_appearance_size: int = Field(default=16, gt=3, description="Appearance patch edge in pixels")
    track_appearance_min_similarity: float = Field(
        default=0.15,
        ge=-1.0,
        le=1.0,
        description="Below this correlation, a re-attachment after a gap is refused",
    )
    track_appearance_weight: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Share of the association cost driven by appearance"
    )
    track_scene_cut_reset: bool = Field(
        default=True,
        description="A scene cut resets motion prediction and drops tracks that are already lost",
    )
    scene_cut_threshold: float = Field(default=0.35, gt=0.0)
    dual_subject_balance: float = Field(default=0.35, ge=0.0, le=1.0)
    jitter_epsilon_px: float = Field(default=2.0, ge=0.0)


class TrajectoryStats(BaseModel):
    """Aggregate motion statistics of the emitted crop trajectory."""

    mean_velocity_px_per_sec: float = 0.0
    max_velocity_px_per_sec: float = 0.0
    peak_velocity_non_scene_cut: float = Field(
        default=0.0,
        description="Peak pan velocity excluding scene-cut re-anchors, which are allowed to jump",
    )
    scene_cut_snaps: int = Field(default=0, description="Transitions that re-anchored on a scene cut")
    total_travel_px: float = 0.0
    stationary_ratio: float = Field(default=1.0, description="Fraction of samples with no crop movement")
    crop_x_min: int = 0
    crop_x_max: int = 0
    crop_x_range: int = 0
    crop_y_min: int = 0
    crop_y_max: int = 0


class TrackFragmentationReport(BaseModel):
    """How badly identity broke up across the clip.

    A clip where the detector fires constantly but no identity survives is not a clip with no
    people in it - it is a tracking failure, and it looks identical to an empty room unless it
    is measured. These numbers are what tell those two cases apart.
    """

    identities: int = Field(default=0, description="Tracks that were confirmed as real subjects")
    raw_candidates: int = Field(default=0, description="Every track id minted, ghosts included")
    detections_total: int = 0
    detections_per_identity: float = Field(
        default=0.0, description="Detections divided by confirmed identities; ~frames per person"
    )
    mean_lifetime_sec: float = 0.0
    median_lifetime_sec: float = 0.0
    longest_lifetime_sec: float = 0.0
    tracks_under_half_second: int = 0
    tracks_under_one_second: int = 0
    fragmentation_ratio: float = Field(
        default=0.0, description="Raw candidates per confirmed identity; 1.0 is perfect continuity"
    )
    reattachments: int = Field(
        default=0, description="Times an identity was recovered after a detection gap"
    )
    discarded_tentative: int = Field(
        default=0, description="Candidate tracks that never earned an identity"
    )
    warning: Optional[str] = Field(
        default=None, description="Set when plentiful detections still yielded no stable identity"
    )

    def summary(self) -> str:
        return (
            f"identities={self.identities} raw={self.raw_candidates} "
            f"frag_ratio={self.fragmentation_ratio:.2f} "
            f"lifetime(mean={self.mean_lifetime_sec:.2f}s median={self.median_lifetime_sec:.2f}s "
            f"longest={self.longest_lifetime_sec:.2f}s) "
            f"short(<0.5s={self.tracks_under_half_second}, <1s={self.tracks_under_one_second}) "
            f"detections={self.detections_total} per_identity={self.detections_per_identity:.1f} "
            f"reattached={self.reattachments}"
        )


class ReframeDiagnostics(BaseModel):
    """Per-short quality diagnostics for the smart reframing stage."""

    version: str = REFRAME_VERSION
    detector: str = ""
    detector_description: str = ""
    detector_operational: bool = True
    analysis_fps: float = 0.0
    sampled_frames: int = 0
    frames_with_detection: int = 0
    frames_with_face: int = 0
    frames_with_person: int = 0
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
    fallback_used: bool = Field(
        default=False, description="Any tracking fallback rung was used (not a render failure)"
    )
    tracking_mode: str = Field(
        default=TRACKING_MODE_SUBJECT,
        description="subject | mixed | dominant_region | center - how framing was actually driven",
    )
    edge_clamped_frames: int = 0
    analysis_seconds: float = 0.0
    trajectory: TrajectoryStats = Field(default_factory=TrajectoryStats)
    fragmentation: TrackFragmentationReport = Field(
        default_factory=TrackFragmentationReport,
        description="Identity continuity across the clip; distinguishes an empty room from a "
                    "tracking failure",
    )

    def _rate(self, count: int) -> float:
        return round(count / self.sampled_frames, 4) if self.sampled_frames else 0.0

    @property
    def analyzed_frames(self) -> int:
        """Frames actually sampled and analyzed."""
        return self.sampled_frames

    @property
    def track_count(self) -> int:
        """Distinct subjects tracked across the clip."""
        return self.unique_tracks

    @property
    def active_subject_switches(self) -> int:
        """Times the active subject changed."""
        return self.dominant_subject_switches

    @property
    def detection_coverage(self) -> float:
        """Fraction of analyzed frames in which the detector found at least one subject."""
        return self._rate(self.frames_with_detection)

    @property
    def face_coverage(self) -> float:
        """Fraction of analyzed frames containing at least one detected face."""
        return self._rate(self.frames_with_face)

    @property
    def person_coverage(self) -> float:
        """Fraction of analyzed frames containing at least one detected person."""
        return self._rate(self.frames_with_person)

    @property
    def tracking_coverage(self) -> float:
        """Fraction of analyzed frames framed from a tracked subject rather than a fallback."""
        return self._rate(self.frames_with_active_subject)

    @property
    def tracking_fallback_rate(self) -> float:
        """Fraction of analyzed frames framed by any rung of the tracking fallback ladder.

        This says nothing about whether the render succeeded: a clip can be reframed entirely
        from the dominant-region fallback and still be rendered by the dynamic crop driver.
        """
        return self._rate(
            self.fallback_previous_frames
            + self.fallback_dominant_frames
            + self.fallback_center_frames
        )

    @property
    def previous_fallback_rate(self) -> float:
        """Fraction of analyzed frames that held the last known subject framing."""
        return self._rate(self.fallback_previous_frames)

    @property
    def dominant_fallback_rate(self) -> float:
        """Fraction of analyzed frames framed on the dominant visual region."""
        return self._rate(self.fallback_dominant_frames)

    @property
    def center_fallback_rate(self) -> float:
        """Fraction of analyzed frames that fell all the way back to a center crop."""
        return self._rate(self.fallback_center_frames)

    def resolve_tracking_mode(self) -> str:
        """Classify how the framing was actually driven across the clip."""
        if not self.sampled_frames:
            return TRACKING_MODE_CENTER
        if self.frames_with_active_subject == self.sampled_frames:
            return TRACKING_MODE_SUBJECT
        if self.frames_with_active_subject > 0:
            return TRACKING_MODE_MIXED
        if self.fallback_dominant_frames > 0:
            return TRACKING_MODE_DOMINANT
        return TRACKING_MODE_CENTER


class DebugSample(BaseModel):
    """Per-sample record retained for the optional debug overlay render."""

    time: float
    boxes: List[Tuple[int, int, int, int]] = Field(default_factory=list)
    track_ids: List[int] = Field(default_factory=list)
    subject_types: List[str] = Field(default_factory=list)
    active_track_id: Optional[int] = None
    target_x: float = 0.0
    target_y: float = 0.0
    crop_x: int = 0
    crop_y: int = 0
    fallback: str = FALLBACK_NONE
    scene_cut: bool = False


class TrackObservation(BaseModel):
    """One tracked subject as seen in one analysis frame.

    This is the raw material the adaptive layout planner reasons over: it deliberately keeps
    per-track identity and geometry instead of collapsing to a single framing target, because
    "which people are on screen right now" is a different question from "where do I point the
    single crop".
    """

    track_id: int
    box: Tuple[int, int, int, int]
    subject_type: str
    confidence: float = 1.0
    speaking_score: float = 0.0
    hits: int = 1

    @property
    def center_x(self) -> float:
        return self.box[0] + self.box[2] / 2.0

    @property
    def center_y(self) -> float:
        return self.box[1] + self.box[3] / 2.0

    @property
    def area(self) -> float:
        return float(self.box[2] * self.box[3])


class FrameObservation(BaseModel):
    """Everything the analysis pass saw in one sampled frame."""

    time: float
    scene_cut: bool = False
    active_track_id: Optional[int] = None
    tracks: List[TrackObservation] = Field(default_factory=list)

    @property
    def detected(self) -> bool:
        return bool(self.tracks)

    @property
    def mean_confidence(self) -> float:
        return sum(t.confidence for t in self.tracks) / len(self.tracks) if self.tracks else 0.0


class ReframePlan(BaseModel):
    """Crop trajectory plus diagnostics for one short."""

    mode: str = REFRAME_MODE_SMART
    trajectory: CropTrajectory
    diagnostics: ReframeDiagnostics = Field(default_factory=ReframeDiagnostics)
    debug_samples: List[DebugSample] = Field(default_factory=list)
    frames: List[FrameObservation] = Field(
        default_factory=list,
        description="Per-sample subject tracks, consumed by the adaptive layout planner",
    )


#: A track that has not yet earned an identity. Real detectors emit single-frame ghosts, and
#: promoting every one of them to a person is what turns a two-person scene into thirty tracks.
TRACK_TENTATIVE = "tentative"
TRACK_CONFIRMED = "confirmed"


@dataclass
class SubjectTrack:
    """A subject followed across sampled frames.

    A track starts *tentative*: it is followed and framed exactly like any other, but it is not
    an identity until it has been seen ``track_confirm_hits`` times. Only confirmed tracks are
    counted, reported, or handed to the layout planner, so a one-frame false positive can no
    longer masquerade as a person.
    """

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
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    state: str = TRACK_TENTATIVE
    reattachments: int = 0
    mouth_patch: Optional[np.ndarray] = field(default=None, repr=False)
    appearance: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def confirmed(self) -> bool:
        return self.state == TRACK_CONFIRMED

    def update(self, subject: DetectedSubject, timestamp: float, confirm_hits: int = 3) -> None:
        dt = timestamp - self.last_seen
        if dt > 1e-3:
            # Exponentially smoothed so one noisy detection cannot fling the prediction away.
            self.velocity_x = 0.5 * self.velocity_x + 0.5 * (subject.center_x - self.center_x) / dt
            self.velocity_y = 0.5 * self.velocity_y + 0.5 * (subject.center_y - self.center_y) / dt
        if self.misses > 0:
            self.reattachments += 1
        self.box = subject.box
        self.center_x = subject.center_x
        self.center_y = subject.center_y
        self.area = subject.area
        self.subject_type = subject.subject_type
        self.confidence = subject.confidence
        self.last_seen = timestamp
        self.hits += 1
        self.misses = 0
        if self.hits >= confirm_hits:
            self.state = TRACK_CONFIRMED

    def gap(self, timestamp: float) -> float:
        """Seconds since this track was last actually detected."""
        return max(0.0, timestamp - self.last_seen)

    def predict(self, timestamp: float, cfg: "ReframeConfig") -> Tuple[float, float]:
        """Where this track is expected to be at ``timestamp``.

        A subject that keeps moving while the detector loses it for a few frames would otherwise
        be re-identified as a brand-new person the moment it is seen again, which is exactly the
        track-id churn that makes a two-person layout decision meaningless.
        """
        if not cfg.track_predict_motion:
            return self.center_x, self.center_y
        gap = self.gap(timestamp)
        if gap <= 0.0 or self.hits < 2:
            return self.center_x, self.center_y
        limit = cfg.track_max_motion_px_per_sec
        vx = min(max(self.velocity_x, -limit), limit)
        vy = min(max(self.velocity_y, -limit), limit)
        return self.center_x + vx * gap, self.center_y + vy * gap

    def blend_appearance(self, patch: Optional[np.ndarray]) -> None:
        """Fold a new appearance patch into the running descriptor."""
        if patch is None:
            return
        if self.appearance is None or self.appearance.shape != patch.shape:
            self.appearance = patch
        else:
            self.appearance = 0.7 * self.appearance + 0.3 * patch


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


def _association_reach(
    track: SubjectTrack,
    subject: DetectedSubject,
    timestamp: float,
    sample_dt: float,
    cfg: ReframeConfig,
    scene_cut: bool = False,
) -> float:
    """How far apart a track and a detection may be and still be the same subject.

    Gating on box size alone breaks down for small, fast subjects: a 20 px facecam moving 16 px
    between samples overlaps its previous box by almost nothing, so it would be re-identified as
    a brand-new person on every single frame.

    The elapsed time that matters is the gap since *this track* was last seen, not the interval
    between the last two samples. Using the sample interval was the bug that made re-attachment
    after a dropout impossible: a subject lost for a second is allowed to have moved for a
    second, and gating it as though only 200 ms had passed guarantees a fresh identity every
    time the detector blinks.

    The reach is capped regardless, so "recover the identity after a gap" never becomes "merge
    the two people standing at opposite ends of the frame".
    """
    size_reach = 0.6 * max(track.box[2], subject.box[2], 1)
    elapsed = max(sample_dt, track.gap(timestamp))
    # Across a scene cut, motion prediction means nothing: the camera, not the subject, moved.
    motion_reach = 0.0 if (scene_cut and cfg.track_scene_cut_reset) else (
        cfg.track_max_motion_px_per_sec * max(elapsed, 1e-3)
    )
    reach = max(size_reach, motion_reach, cfg.track_min_reach_px)
    return min(reach, cfg.track_max_reassociation_px)


def _appearance_patch(
    frame_gray: np.ndarray, box: Sequence[int], cfg: ReframeConfig
) -> Optional[np.ndarray]:
    """A tiny contrast-normalized thumbnail of the subject, used as a cheap identity cue.

    Deliberately not a re-identification network: this must stay CPU-only and free, so it is a
    16x16 patch with its mean and standard deviation divided out. That is enough to answer "is
    this plausibly the same face I lost half a second ago", which is the only question asked
    of it, and never enough to be trusted on its own.
    """
    if not cfg.track_appearance_enabled:
        return None
    x, y, w, h = (int(v) for v in box)
    if w < 6 or h < 6:
        return None
    patch = frame_gray[max(0, y) : y + h, max(0, x) : x + w]
    if patch.size == 0:
        return None
    import cv2

    edge = cfg.track_appearance_size
    small = cv2.resize(patch, (edge, edge), interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= float(small.mean())
    deviation = float(small.std())
    if deviation < 1e-3:
        return None
    return small / deviation


def _appearance_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """Normalized correlation of two appearance patches, or ``None`` if either is missing."""
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(np.clip(np.mean(a * b), -1.0, 1.0))


def _associate(
    tracks: List[SubjectTrack],
    subjects: List[DetectedSubject],
    timestamp: float,
    next_track_id: int,
    dt: float = 0.2,
    cfg: Optional[ReframeConfig] = None,
    appearances: Optional[List[Optional[np.ndarray]]] = None,
    scene_cut: bool = False,
) -> Tuple[List[SubjectTrack], List[Tuple[SubjectTrack, DetectedSubject]], int]:
    """Associate detections to tracks by prediction, overlap, proximity, size and appearance.

    Confirmed identities are matched first, on their own, before any tentative track is allowed
    to compete. Without that ordering a one-frame ghost that happens to appear near a real
    subject can capture the detection, drag the identity away from the person, and leave the
    real face to start a new track on the next sample - which is how a handful of ghosts turns
    into dozens of tracks.
    """
    config = cfg or ReframeConfig()
    patches = appearances or [None] * len(subjects)

    def candidate_pairs(pool: List[int], available: List[int]) -> List[Tuple[float, int, int]]:
        pairs: List[Tuple[float, int, int]] = []
        for t_idx in pool:
            track = tracks[t_idx]
            predicted_x, predicted_y = track.predict(timestamp, config)
            gap = track.gap(timestamp)
            for s_idx in available:
                subject = subjects[s_idx]
                # A face cannot plausibly become three times bigger between two samples;
                # allowing it lets a near subject swallow a distant one's identity.
                larger = max(track.box[2], subject.box[2])
                smaller = max(1, min(track.box[2], subject.box[2]))
                if larger / smaller > config.track_max_size_ratio:
                    continue

                iou = _iou(track.box, subject.box)
                reach = _association_reach(track, subject, timestamp, dt, config, scene_cut)
                distance = math.hypot(predicted_x - subject.center_x, predicted_y - subject.center_y)
                if iou < config.track_match_iou and distance > reach:
                    continue

                similarity = _appearance_similarity(track.appearance, patches[s_idx])
                # Geometry alone is weak evidence once the subject has been missing: two people
                # a second apart can easily be within reach of each other. Appearance is asked
                # to agree before an identity is allowed to survive a gap.
                if (
                    similarity is not None
                    and gap > dt * 1.5
                    and iou < config.track_match_iou
                    and similarity < config.track_appearance_min_similarity
                ):
                    continue

                # Single blended cost so overlap, proximity and appearance stay comparable.
                geometric = 0.5 * (1.0 - iou) + 0.5 * min(1.0, distance / max(reach, 1.0))
                if similarity is None:
                    cost = geometric
                else:
                    weight = config.track_appearance_weight
                    cost = (1.0 - weight) * geometric + weight * (1.0 - (similarity + 1.0) / 2.0)
                if track.subject_type != subject.subject_type:
                    cost += 0.25
                # Prefer the identity that has been away for less time when costs tie.
                cost += 0.02 * min(1.0, gap / max(config.track_max_gap_sec, 1e-3))
                pairs.append((cost, t_idx, s_idx))
        pairs.sort()
        return pairs

    existing_count = len(tracks)
    used_tracks: set[int] = set()
    used_subjects: set[int] = set()
    matched: List[Tuple[SubjectTrack, DetectedSubject]] = []

    confirmed_pool = [i for i in range(existing_count) if tracks[i].confirmed]
    tentative_pool = [i for i in range(existing_count) if not tracks[i].confirmed]

    for pool in (confirmed_pool, tentative_pool):
        if not pool:
            continue
        available = [i for i in range(len(subjects)) if i not in used_subjects]
        if not available:
            break
        for _, t_idx, s_idx in candidate_pairs(pool, available):
            if t_idx in used_tracks or s_idx in used_subjects:
                continue
            used_tracks.add(t_idx)
            used_subjects.add(s_idx)
            tracks[t_idx].update(subjects[s_idx], timestamp, config.track_confirm_hits)
            tracks[t_idx].blend_appearance(patches[s_idx])
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
            state=TRACK_CONFIRMED if config.track_confirm_hits <= 1 else TRACK_TENTATIVE,
        )
        track.blend_appearance(patches[s_idx])
        next_track_id += 1
        tracks.append(track)
        matched.append((track, subject))

    # Only tracks that already existed can be "missed"; tracks created from this frame's
    # detections are live by definition.
    for t_idx in range(existing_count):
        if t_idx not in used_tracks:
            tracks[t_idx].misses += 1

    return tracks, matched, next_track_id


def _expire_tracks(
    tracks: List[SubjectTrack],
    timestamp: float,
    cfg: ReframeConfig,
    scene_cut: bool = False,
) -> Tuple[List[SubjectTrack], int]:
    """Drop tracks that can no longer plausibly come back; report how many were mere blips.

    The grace is expressed in seconds rather than in samples so it means the same thing at any
    analysis frame rate, and a scene cut ends the grace outright: a subject that was already
    missing when the shot changed is not going to walk back into this one.
    """
    kept: List[SubjectTrack] = []
    discarded_tentative = 0
    for track in tracks:
        gap = track.gap(timestamp)
        expired = gap > cfg.track_max_gap_sec
        if scene_cut and cfg.track_scene_cut_reset and track.misses > 0:
            expired = True
        if expired:
            if not track.confirmed:
                discarded_tentative += 1
            continue
        kept.append(track)
    return kept, discarded_tentative


def _record(track: SubjectTrack) -> TrackObservation:
    """Snapshot a live track for the layout planner."""
    return TrackObservation(
        track_id=track.track_id,
        box=track.box,
        subject_type=track.subject_type,
        confidence=round(float(track.confidence), 4),
        speaking_score=round(float(track.speaking_score), 4),
        hits=track.hits,
    )


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


def framing_target_for_box(
    box: Sequence[int],
    crop_w: int,
    crop_h: int,
    source_width: int,
    source_height: int,
    cfg: ReframeConfig,
) -> Tuple[float, float, bool]:
    """Safe-framing target center for one subject box; returns (x, y, edge_clamped).

    Shared by the single-viewport crop and by each half of a stacked dual layout, so a subject
    is framed by exactly the same rules whichever viewport it ends up in.
    """
    box_x, box_y, box_w, box_h = box
    half_w = crop_w / 2.0

    target_x = box_x + box_w / 2.0
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
    target_y = (box_y + box_h / 2.0) + crop_h * (0.5 - cfg.head_position_ratio)
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
    subject_types: List[str]
    edge_clamped: bool
    records: List[TrackObservation] = field(default_factory=list)


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
            detector_description="disabled (static center crop)",
            detector_operational=False,
            analysis_fps=analysis_fps,
            sampled_frames=len(points),
            fallback_center_frames=len(points),
            fallback_used=True,
            tracking_mode=TRACKING_MODE_CENTER,
            trajectory=TrajectoryStats(
                crop_x_min=crop_x, crop_x_max=crop_x, crop_y_min=crop_y, crop_y_max=crop_y
            ),
        ),
    )


def _fragmentation_report(
    identity_seen: dict,
    reattachments: dict,
    raw_candidates: int,
    discarded_tentative: int,
    detections_total: int,
    analysis_fps: float,
) -> TrackFragmentationReport:
    """Summarize how well physical people kept one identity across the clip."""
    sample = 1.0 / max(analysis_fps, 1e-3)
    lifetimes = sorted(max(sample, (last - first) + sample) for first, last in identity_seen.values())
    identities = len(lifetimes)

    report = TrackFragmentationReport(
        identities=identities,
        raw_candidates=raw_candidates,
        detections_total=detections_total,
        detections_per_identity=round(detections_total / identities, 2) if identities else 0.0,
        mean_lifetime_sec=round(sum(lifetimes) / identities, 3) if identities else 0.0,
        median_lifetime_sec=round(lifetimes[identities // 2], 3) if identities else 0.0,
        longest_lifetime_sec=round(lifetimes[-1], 3) if identities else 0.0,
        tracks_under_half_second=sum(1 for life in lifetimes if life < 0.5),
        tracks_under_one_second=sum(1 for life in lifetimes if life < 1.0),
        fragmentation_ratio=round(raw_candidates / identities, 3) if identities else float(raw_candidates),
        reattachments=sum(reattachments.values()),
        discarded_tentative=discarded_tentative,
    )
    # The case this whole report exists for: the detector was firing, and nothing survived.
    if detections_total >= 10 and identities == 0:
        report.warning = (
            f"{detections_total} detections produced no stable identity at all "
            f"({raw_candidates} candidate tracks were minted and discarded); "
            f"subject-driven framing and layout cannot work on this clip"
        )
    elif identities and report.median_lifetime_sec < 1.0 and raw_candidates >= 4 * identities:
        report.warning = (
            f"identity is fragmenting: {raw_candidates} candidate tracks for {identities} "
            f"identities, median lifetime {report.median_lifetime_sec:.2f}s"
        )
    return report


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
    raw_track_ids = 0
    discarded_tentative = 0
    #: Confirmed identities only, keyed by track id -> [first_seen, last_seen].
    identity_seen: dict[int, List[float]] = {}
    identity_reattachments: dict[int, int] = {}
    active_id: Optional[int] = None
    pending_id: Optional[int] = None
    pending_since = 0.0
    last_switch = -1e9
    prev_small: Optional[np.ndarray] = None
    previous_sample_time = -1.0 / max(cfg.analysis_fps, 1e-3)
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
                _Observation(t_rel, target[0], target[1], fallback, fallback, False, None, [], [], [], False)
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
        if any(s.subject_type == "face" for s in subjects):
            diagnostics.frames_with_face += 1
        if any(s.subject_type == "person" for s in subjects):
            diagnostics.frames_with_person += 1

        sample_dt = max(1e-3, t_rel - previous_sample_time)
        previous_sample_time = t_rel
        appearances = [
            _appearance_patch(small_gray, [int(v * scale) for v in s.box], cfg) for s in subjects
        ]
        tracks, matched, next_track_id = _associate(
            tracks,
            subjects,
            t_rel,
            next_track_id,
            dt=sample_dt,
            cfg=cfg,
            appearances=appearances,
            scene_cut=scene_cut,
        )
        for track, subject in matched:
            _update_speaking(track, _mouth_patch(small_gray, [int(v * scale) for v in subject.box]))
        tracks, discarded = _expire_tracks(tracks, t_rel, cfg, scene_cut)
        discarded_tentative += discarded
        if scene_cut and cfg.track_scene_cut_reset:
            # The camera moved, not the subject: whatever velocity was learned is now noise.
            for track in tracks:
                track.velocity_x = track.velocity_y = 0.0
        raw_track_ids = next_track_id - 1
        for track in tracks:
            if track.confirmed:
                identity_seen.setdefault(track.track_id, [track.first_seen, track.last_seen])[1] = (
                    track.last_seen
                )
                identity_reattachments[track.track_id] = track.reattachments
        diagnostics.unique_tracks = len(identity_seen)

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
                _Observation(t_rel, target_x, target_y, fallback, fallback, scene_cut, None, [], [], [], False)
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
                        [t.subject_type for t in live],
                        False,
                        [_record(t) for t in live if t.confirmed],
                    )
                )
                last_target = last_subject_target = (target_x, target_y)
                last_subject_time = t_rel
                continue

        target_x, target_y, edge_clamped = framing_target_for_box(
            active.box, crop_w, crop_h, source_width, source_height, cfg
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
                [t.subject_type for t in live],
                edge_clamped,
                [_record(t) for t in live if t.confirmed],
            )
        )
        last_target = last_subject_target = (target_x, target_y)
        last_subject_time = t_rel

    capture.release()
    diagnostics.fragmentation = _fragmentation_report(
        identity_seen=identity_seen,
        reattachments=identity_reattachments,
        raw_candidates=max(raw_track_ids, next_track_id - 1),
        discarded_tentative=discarded_tentative,
        detections_total=diagnostics.detected_subjects_total,
        analysis_fps=cfg.analysis_fps,
    )
    logger.info(f"[reframe] track continuity: {diagnostics.fragmentation.summary()}")
    if diagnostics.fragmentation.warning:
        logger.warning(f"[reframe] {diagnostics.fragmentation.warning}")
    return observations


@dataclass
class FramingTarget:
    """One raw framing intent, before smoothing turns it into a crop keyframe."""

    time: float
    x: float
    y: float
    scene_cut: bool = False
    subject_type: str = FALLBACK_NONE


def smooth_crop_points(
    targets: Sequence[FramingTarget],
    crop_w: int,
    crop_h: int,
    source_width: int,
    source_height: int,
    cfg: ReframeConfig,
) -> List[CropPoint]:
    """Convert raw framing targets into an operator-like crop trajectory.

    Dead-zone, proportional tracking, velocity and acceleration clamps remove shake and
    micro-jitter; a scene cut is allowed to re-anchor instantly.

    Every viewport goes through this function - the single crop and both halves of a stacked
    dual layout - so all of them move with the same camera operator.
    """
    points: List[CropPoint] = []
    deadzone = cfg.deadzone_ratio * source_width

    cur_x = cur_y = 0.0
    vel_x = vel_y = 0.0
    prev_t = 0.0
    last_emitted_x: Optional[int] = None
    last_emitted_y: Optional[int] = None

    for idx, obs in enumerate(targets):
        if idx == 0:
            cur_x, cur_y = obs.x, obs.y
            vel_x = vel_y = 0.0
        elif obs.scene_cut:
            cur_x, cur_y = obs.x, obs.y
            vel_x = vel_y = 0.0
        else:
            dt = max(1e-3, obs.time - prev_t)
            for axis in ("x", "y"):
                cur = cur_x if axis == "x" else cur_y
                vel = vel_x if axis == "x" else vel_y
                target = obs.x if axis == "x" else obs.y

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


def _smooth(
    observations: List[_Observation],
    crop_w: int,
    crop_h: int,
    source_width: int,
    source_height: int,
    cfg: ReframeConfig,
) -> List[CropPoint]:
    """Smooth the analysis pass's own framing targets into the single-viewport trajectory."""
    return smooth_crop_points(
        [
            FramingTarget(o.time, o.target_x, o.target_y, o.scene_cut, o.subject_type)
            for o in observations
        ],
        crop_w,
        crop_h,
        source_width,
        source_height,
        cfg,
    )


def _trajectory_stats(
    points: List[CropPoint],
    scene_cuts: Optional[List[bool]] = None,
) -> TrajectoryStats:
    """Summarize crop motion, separating deliberate scene-cut re-anchors from panning.

    A scene cut is allowed to move the crop instantly, so its apparent velocity is unbounded and
    would otherwise dominate the peak. ``peak_velocity_non_scene_cut`` is the number that must
    respect the configured velocity limit.
    """
    if not points:
        return TrajectoryStats()

    cuts = scene_cuts or [False] * len(points)
    velocities: List[float] = []
    smooth_velocities: List[float] = []
    travel = 0.0
    stationary = 0
    snaps = 0
    for i in range(1, len(points)):
        dt = max(1e-3, points[i].time - points[i - 1].time)
        dx = abs(points[i].crop_x - points[i - 1].crop_x)
        travel += dx
        velocity = dx / dt
        velocities.append(velocity)
        if i < len(cuts) and cuts[i]:
            snaps += 1
        else:
            smooth_velocities.append(velocity)
        if dx == 0:
            stationary += 1

    xs = [p.crop_x for p in points]
    ys = [p.crop_y for p in points]
    denominator = max(1, len(points) - 1)
    return TrajectoryStats(
        mean_velocity_px_per_sec=round(sum(velocities) / len(velocities), 3) if velocities else 0.0,
        max_velocity_px_per_sec=round(max(velocities), 3) if velocities else 0.0,
        peak_velocity_non_scene_cut=round(max(smooth_velocities), 3) if smooth_velocities else 0.0,
        scene_cut_snaps=snaps,
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
        active_detector = detector or get_subject_detector(
            detector_name,
            model_path=Path(cfg.face_model_path) if cfg.face_model_path else None,
            allow_download=cfg.allow_model_download,
            score_threshold=cfg.face_score_threshold,
        )
        diagnostics.detector = getattr(active_detector, "name", detector_name)
        diagnostics.detector_description = active_detector.describe()
        diagnostics.detector_operational = active_detector.is_operational
        if not active_detector.is_operational:
            logger.error(
                f"[reframe] Detector '{diagnostics.detector_description}' is not operational on this "
                f"build; framing will fall back to the dominant visual region"
            )
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
    diagnostics.fallback_used = diagnostics.tracking_fallback_rate > 0.0
    diagnostics.tracking_mode = diagnostics.resolve_tracking_mode()
    diagnostics.trajectory = _trajectory_stats(points, [o.scene_cut for o in observations])
    diagnostics.analysis_seconds = round(time.perf_counter() - started, 3)

    debug_samples: List[DebugSample] = []
    if collect_debug:
        for obs, point in zip(observations, points):
            debug_samples.append(
                DebugSample(
                    time=round(obs.time, 3),
                    boxes=obs.boxes,
                    track_ids=obs.track_ids,
                    subject_types=obs.subject_types,
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
        frames=[
            FrameObservation(
                time=round(o.time, 3),
                scene_cut=o.scene_cut,
                active_track_id=o.active_track_id,
                tracks=o.records,
            )
            for o in observations
        ],
    )


def render_debug_overlay(
    video_path: Path | str,
    plan: ReframePlan,
    source_start_sec: float,
    output_path: Path | str,
    layout_labels: Optional[Sequence[Tuple[float, float, str]]] = None,
    safe_band: Optional[Tuple[float, float]] = None,
) -> Optional[Path]:
    """Render a diagnostic video with detection boxes, crop rectangle and active subject.

    ``layout_labels`` are ``(start, end, text)`` triples describing the layout in force at each
    moment; ``safe_band`` is the ``(top, bottom)`` caption/UI reservation as a fraction of the
    crop height. Both are supplied by the caller rather than imported, so the analysis layer
    never has to know about the layout layer.

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

        types = sample.subject_types or ["?"] * len(sample.boxes)
        for box, track_id, subject_type in zip(sample.boxes, sample.track_ids, types):
            x, y, w, h = box
            is_active = track_id == sample.active_track_id
            color = (0, 255, 0) if is_active else (180, 180, 180)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3 if is_active else 1)
            cv2.putText(
                frame, f"#{track_id} {subject_type}", (x, max(14, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
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
        tracked = sample.fallback in (FALLBACK_NONE, FALLBACK_DUAL, "face", "person")
        state = "TRACKED" if tracked else f"FALLBACK:{sample.fallback}"
        colour = (0, 255, 0) if tracked else (0, 165, 255)
        cv2.putText(
            frame,
            f"t={sample.time:.2f}s  {state}  subjects={len(sample.boxes)}"
            + (f"  active=#{sample.active_track_id}" if sample.active_track_id is not None else "")
            + ("  SCENE CUT" if sample.scene_cut else ""),
            (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2,
        )
        cv2.putText(
            frame, f"crop x={sample.crop_x} y={sample.crop_y} {traj.crop_w}x{traj.crop_h}",
            (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2,
        )

        if safe_band is not None:
            top_ratio, bottom_ratio = safe_band
            safe_top = int(sample.crop_y + top_ratio * traj.crop_h)
            safe_bottom = int(sample.crop_y + (1.0 - bottom_ratio) * traj.crop_h)
            cv2.rectangle(
                frame,
                (sample.crop_x, safe_top),
                (sample.crop_x + traj.crop_w, safe_bottom),
                (255, 255, 0),
                2,
            )
            cv2.putText(
                frame, "safe area", (sample.crop_x + 8, max(0, safe_top) + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
            )

        if layout_labels:
            text = next(
                (label for start, end, label in layout_labels if start <= sample.time < end),
                layout_labels[-1][2],
            )
            for offset, line in enumerate(text.split("\n")):
                cv2.putText(
                    frame, line, (16, 124 + offset * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2,
                )

        writer.write(frame)

    writer.release()
    capture.release()
    logger.info(f"[reframe] Debug overlay written to {out}")
    return out
