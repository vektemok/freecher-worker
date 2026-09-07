"""Highlight boundary refinement using speech transcript segment boundaries."""

from __future__ import annotations

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from arny_worker.highlights.models import Highlight, HighlightScore
from arny_worker.transcription.models import Transcript, TranscriptSegment


class RefinedHighlight(BaseModel):
    """Highlight with speech-boundary refined start and end timestamps."""

    candidate_id: str = Field(description="Identifier of candidate highlight window")
    rank: int = Field(description="Highlight rank (1-based)")
    original_start: float = Field(description="Original start time in seconds")
    original_end: float = Field(description="Original end time in seconds")
    refined_start: float = Field(description="Refined start time in seconds")
    refined_end: float = Field(description="Refined end time in seconds")
    duration: float = Field(description="Refined duration in seconds")
    refinement_reason: str = Field(description="Explanation of boundary adjustments made")
    score: float = Field(default=0.0, description="Overall highlight score")
    text: str = Field(default="", description="Aggregated transcript text")
    score_breakdown: Optional[HighlightScore] = Field(default=None, description="Detailed score dimensions")


def refine_boundaries(
    start: float,
    end: float,
    transcript: Transcript,
    video_duration: float,
    max_shift_seconds: float = 5.0,
    context_before: float = 0.5,
    context_after: float = 0.5,
) -> Tuple[float, float, str]:
    """Refine temporal boundaries to align with natural speech phrases and trim silence.

    Args:
        start: Candidate start time in seconds.
        end: Candidate end time in seconds.
        transcript: Source transcript containing segment boundaries.
        video_duration: Total duration of the source video in seconds.
        max_shift_seconds: Maximum allowed shift in seconds for either boundary.
        context_before: Context padding in seconds before phrase onset.
        context_after: Context padding in seconds after phrase completion.

    Returns:
        Tuple of (refined_start, refined_end, refinement_reason).
    """
    segments = transcript.segments
    if not segments:
        refined_start = max(0.0, min(start, video_duration))
        refined_end = max(refined_start + 1.0, min(end, video_duration))
        return round(refined_start, 3), round(refined_end, 3), "No transcript segments available"

    reasons: List[str] = []
    refined_start = start
    refined_end = end

    # -------------------------------------------------------------
    # 1. Start boundary refinement
    # -------------------------------------------------------------
    # Find segment that overlaps start or the earliest segment starting after start
    start_seg = None
    for seg in segments:
        if seg.start <= start <= seg.end:
            start_seg = seg
            break

    if start_seg is not None:
        # Start cuts mid-phrase
        cut_in_distance = start - start_seg.start
        if cut_in_distance <= max_shift_seconds:
            refined_start = max(0.0, start_seg.start - context_before)
            reasons.append(f"snapped start earlier by {cut_in_distance:.2f}s to phrase start ({start_seg.start:.2f}s)")
        else:
            refined_start = max(0.0, start - max_shift_seconds)
            reasons.append(f"shifted start earlier by max {max_shift_seconds:.1f}s towards phrase start")
    else:
        # Start is in a silence gap between segments or before the first segment
        subsequent_segs = [s for s in segments if s.start >= start]
        if subsequent_segs:
            next_seg = subsequent_segs[0]
            silence_lead = next_seg.start - start
            if silence_lead > 1.0:
                # Trim excessive silence
                refined_start = max(start, next_seg.start - context_before)
                reasons.append(f"trimmed {silence_lead:.2f}s leading silence to {refined_start:.2f}s")
            elif silence_lead > 0:
                refined_start = max(0.0, next_seg.start - context_before)
                reasons.append(f"aligned start to phrase onset ({next_seg.start:.2f}s)")

    # -------------------------------------------------------------
    # 2. End boundary refinement
    # -------------------------------------------------------------
    end_seg = None
    for seg in segments:
        if seg.start <= end <= seg.end:
            end_seg = seg
            break

    if end_seg is not None:
        # End cuts mid-phrase
        cut_off_distance = end_seg.end - end
        if cut_off_distance <= max_shift_seconds:
            refined_end = min(video_duration, end_seg.end + context_after)
            reasons.append(f"extended end by {cut_off_distance:.2f}s to complete phrase ({end_seg.end:.2f}s)")
        else:
            # Shift back to previous complete segment if duration permits (>= 15.0s)
            prior_segs = [s for s in segments if s.end < end]
            if prior_segs and (prior_segs[-1].end - refined_start) >= 15.0:
                refined_end = min(video_duration, prior_segs[-1].end + context_after)
                reasons.append(f"trimmed cutoff phrase at end to previous phrase completion ({prior_segs[-1].end:.2f}s)")
            else:
                refined_end = min(video_duration, end + max_shift_seconds)
                reasons.append(f"extended end by max {max_shift_seconds:.1f}s towards phrase end")
    else:
        # End is in silence gap
        preceding_segs = [s for s in segments if s.end <= end]
        if preceding_segs:
            prev_seg = preceding_segs[-1]
            silence_trail = end - prev_seg.end
            if silence_trail > 1.0:
                # Trim excessive trailing silence
                refined_end = min(end, prev_seg.end + context_after)
                reasons.append(f"trimmed {silence_trail:.2f}s trailing silence to {refined_end:.2f}s")

    # -------------------------------------------------------------
    # 3. Final duration and boundary guards
    # -------------------------------------------------------------
    refined_start = max(0.0, min(refined_start, video_duration - 1.0))
    refined_end = max(refined_start + 1.0, min(refined_end, video_duration))

    final_reason = "; ".join(reasons) if reasons else "boundaries preserved"
    return round(refined_start, 3), round(refined_end, 3), final_reason


def refine_highlight(
    highlight: Highlight,
    transcript: Transcript,
    video_duration: float,
    max_shift_seconds: float = 5.0,
    context_before: float = 0.5,
    context_after: float = 0.5,
) -> RefinedHighlight:
    """Refine Highlight object into RefinedHighlight with speech-aligned boundaries."""
    r_start, r_end, reason = refine_boundaries(
        start=highlight.start,
        end=highlight.end,
        transcript=transcript,
        video_duration=video_duration,
        max_shift_seconds=max_shift_seconds,
        context_before=context_before,
        context_after=context_after,
    )

    return RefinedHighlight(
        candidate_id=highlight.candidate_id,
        rank=highlight.rank,
        original_start=highlight.start,
        original_end=highlight.end,
        refined_start=r_start,
        refined_end=r_end,
        duration=round(r_end - r_start, 3),
        refinement_reason=reason,
        score=highlight.score,
        text=highlight.text,
        score_breakdown=highlight.score_breakdown,
    )
