"""Deterministic transcript chaptering for contextual_reranker_v1.

Chapters target ~3-5 minutes but are cut on real transcript boundaries (the largest
speech gap near the target) and, when a cached source activity profile exists, on
scene changes. No new media analysis is performed here.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from freecher_worker.transcription.models import Transcript, TranscriptSegment
from freecher_worker.utils.json_io import load_json

logger = logging.getLogger("freecher_worker")

DEFAULT_CHAPTER_TARGET_SECONDS = 240.0
DEFAULT_CHAPTER_MIN_SECONDS = 150.0
DEFAULT_CHAPTER_MAX_SECONDS = 420.0
DEFAULT_BOUNDARY_MIN_GAP_SECONDS = 0.6
#: A scene change this close to a segment boundary makes that boundary preferable.
SCENE_SNAP_SECONDS = 2.0


@dataclass
class ChapterPlan:
    """A planned chapter: boundaries plus the transcript segments inside it."""

    index: int
    start: float
    end: float
    segments: List[TranscriptSegment] = field(default_factory=list)
    boundary_source: str = "transcript_gap"

    @property
    def chapter_id(self) -> str:
        return f"chapter_{self.index:03d}"

    @property
    def segment_ids(self) -> List[int]:
        return [s.id for s in self.segments]

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip()).strip()

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def transcript_hash(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(f"{self.start:.3f}|{self.end:.3f}|{len(self.segments)}".encode("utf-8"))
        for seg in self.segments:
            hasher.update(f"|{seg.id}:{seg.start:.3f}-{seg.end:.3f}:{seg.text}".encode("utf-8"))
        return hasher.hexdigest()[:16]

    def timestamped_text(self) -> str:
        """Transcript lines with absolute timestamps, as sent to the model."""
        lines = []
        for seg in self.segments:
            txt = seg.text.strip()
            if txt:
                lines.append(f"[{seg.start:.1f}s] {txt}")
        return "\n".join(lines)


def load_scene_change_times(run_dir: Path | str) -> List[float]:
    """Read scene-change timestamps from the cached multimodal activity profile.

    Returns an empty list when the cache does not exist. Never computes anything.
    """
    cache_file = (
        Path(run_dir).resolve()
        / "multimodal"
        / "cache"
        / "source_temporal_activity_profile_v1_1.json"
    )
    if not cache_file.is_file():
        return []
    try:
        data = load_json(cache_file)
    except Exception as exc:  # noqa: BLE001 - an unreadable optional cache is not fatal
        logger.warning(f"[contextual-chapters] Could not read activity profile: {exc}")
        return []
    timeline = data.get("timeline") if isinstance(data, dict) else None
    if not isinstance(timeline, list):
        return []
    return [
        float(pt["absolute_timestamp"])
        for pt in timeline
        if isinstance(pt, dict) and pt.get("scene_change") and "absolute_timestamp" in pt
    ]


def build_chapter_plan(
    transcript: Optional[Transcript],
    target_seconds: float = DEFAULT_CHAPTER_TARGET_SECONDS,
    min_seconds: float = DEFAULT_CHAPTER_MIN_SECONDS,
    max_seconds: float = DEFAULT_CHAPTER_MAX_SECONDS,
    boundary_min_gap: float = DEFAULT_BOUNDARY_MIN_GAP_SECONDS,
    scene_change_times: Optional[Sequence[float]] = None,
) -> List[ChapterPlan]:
    """Split a transcript into contiguous, non-overlapping chapters.

    A chapter closes at the first segment boundary after ``target_seconds`` that is a
    real pause (or a scene change), and is force-closed at ``max_seconds``. A trailing
    chapter shorter than ``min_seconds`` is merged into its predecessor.
    """
    if transcript is None or not transcript.segments:
        return []

    segments = sorted(transcript.segments, key=lambda s: (s.start, s.id))
    scenes = sorted(float(t) for t in (scene_change_times or []))

    def is_near_scene_change(t: float) -> bool:
        if not scenes:
            return False
        # scenes is sorted; a linear scan is fine at 1 Hz binning.
        return any(abs(t - s) <= SCENE_SNAP_SECONDS for s in scenes)

    plans: List[ChapterPlan] = []
    current: List[TranscriptSegment] = []
    chapter_start = segments[0].start
    boundary_source = "transcript_gap"

    for idx, seg in enumerate(segments):
        if not current:
            chapter_start = seg.start
            boundary_source = "transcript_gap"
        current.append(seg)

        is_last = idx == len(segments) - 1
        if is_last:
            break

        elapsed = seg.end - chapter_start
        next_seg = segments[idx + 1]
        gap = max(0.0, next_seg.start - seg.end)

        cut = False
        if elapsed >= max_seconds:
            cut = True
            boundary_source = "max_duration"
        elif elapsed >= target_seconds and (
            gap >= boundary_min_gap or is_near_scene_change(seg.end)
        ):
            cut = True
            boundary_source = "scene_change" if is_near_scene_change(seg.end) else "transcript_gap"

        if cut:
            plans.append(
                ChapterPlan(
                    index=len(plans) + 1,
                    start=round(chapter_start, 3),
                    end=round(seg.end, 3),
                    segments=list(current),
                    boundary_source=boundary_source,
                )
            )
            current = []

    if current:
        plans.append(
            ChapterPlan(
                index=len(plans) + 1,
                start=round(chapter_start, 3),
                end=round(current[-1].end, 3),
                segments=list(current),
                boundary_source=boundary_source,
            )
        )

    # Merge a too-short tail chapter into its predecessor.
    if len(plans) >= 2 and plans[-1].duration < min_seconds:
        tail = plans.pop()
        prev = plans[-1]
        prev.segments.extend(tail.segments)
        prev.end = tail.end
        prev.boundary_source = "tail_merge"

    for position, plan in enumerate(plans, start=1):
        plan.index = position

    return plans


def chapter_plan_hash(plans: Sequence[ChapterPlan]) -> str:
    """Deterministic identity of a whole chapter plan."""
    hasher = hashlib.sha256()
    for plan in plans:
        hasher.update(f"{plan.chapter_id}:{plan.transcript_hash()}|".encode("utf-8"))
    return hasher.hexdigest()[:16]


def find_chapter_for_range(
    plans: Sequence[ChapterPlan],
    start: float,
    end: float,
) -> Optional[ChapterPlan]:
    """Return the chapter with the largest temporal overlap with [start, end]."""
    if not plans:
        return None
    best: Optional[ChapterPlan] = None
    best_overlap = -1.0
    for plan in plans:
        overlap = min(end, plan.end) - max(start, plan.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = plan
    if best_overlap <= 0.0:
        # No overlap at all: fall back to the nearest chapter by midpoint distance.
        midpoint = (start + end) / 2.0
        best = min(plans, key=lambda p: abs(((p.start + p.end) / 2.0) - midpoint))
    return best
