"""Candidate window generation from transcript segments."""

from __future__ import annotations

from typing import Optional
from arny_worker.transcription.models import Transcript, TranscriptSegment
from .models import CandidateWindow


def generate_candidate_windows(
    transcript: Transcript,
    min_seconds: float = 30.0,
    target_seconds: float = 60.0,
    max_seconds: float = 90.0,
    overlap_seconds: float = 15.0,
) -> list[CandidateWindow]:
    """Segment a transcript into overlapping candidate highlight windows.

    Windows are bounded by Whisper segment boundaries (never cutting mid-sentence).

    Args:
        transcript: The normalized transcript.
        min_seconds: Minimum duration for a valid candidate window.
        target_seconds: Target duration for windows.
        max_seconds: Maximum allowed duration for a candidate.
        overlap_seconds: Desired overlap between adjacent windows.

    Returns:
        List of CandidateWindow objects.
    """
    segments: list[TranscriptSegment] = [s for s in transcript.segments if s.text.strip()]
    if not segments:
        return []

    total_duration = segments[-1].end - segments[0].start
    # Edge case: video is shorter than min_seconds
    if total_duration < min_seconds:
        if total_duration >= 5.0 or len(segments) > 0:
            full_text = " ".join(s.text.strip() for s in segments)
            return [
                CandidateWindow(
                    id="cand_001",
                    start=round(segments[0].start, 2),
                    end=round(segments[-1].end, 2),
                    duration=round(segments[-1].end - segments[0].start, 2),
                    text=full_text,
                    segment_ids=[s.id for s in segments],
                )
            ]
        return []

    candidates: list[CandidateWindow] = []
    cand_counter = 0
    start_idx = 0
    n = len(segments)

    while start_idx < n:
        cand_start = segments[start_idx].start
        cand_end = segments[start_idx].end
        end_idx = start_idx

        # Accumulate segments until target duration or max duration is approached
        while end_idx + 1 < n:
            next_seg = segments[end_idx + 1]
            proposed_duration = next_seg.end - cand_start

            if proposed_duration > max_seconds:
                # Adding next segment would exceed max allowed length
                break

            end_idx += 1
            cand_end = next_seg.end

            if proposed_duration >= target_seconds:
                # Reached or exceeded target duration with natural sentence boundary
                break

        duration = cand_end - cand_start

        # Include candidate if it satisfies min_seconds,
        # or if we are at the end of video and candidate is at least 60% of min_seconds
        if duration >= min_seconds or (end_idx == n - 1 and duration >= (min_seconds * 0.6)):
            cand_counter += 1
            cand_text = " ".join(segments[i].text.strip() for i in range(start_idx, end_idx + 1))
            candidates.append(
                CandidateWindow(
                    id=f"cand_{cand_counter:03d}",
                    start=round(cand_start, 2),
                    end=round(cand_end, 2),
                    duration=round(duration, 2),
                    text=cand_text,
                    segment_ids=[segments[i].id for i in range(start_idx, end_idx + 1)],
                )
            )

        if end_idx >= n - 1:
            # Reached end of transcript
            break

        # Calculate advance for the next candidate window:
        # Desired next window start: cand_start + target_seconds - overlap_seconds
        target_next_start = cand_start + max(10.0, target_seconds - overlap_seconds)

        next_start_idx = start_idx + 1
        while next_start_idx < n and segments[next_start_idx].start < target_next_start:
            next_start_idx += 1

        # Fallback: if stepping overshot or didn't advance, advance at least by 1 segment
        if next_start_idx <= start_idx:
            next_start_idx = start_idx + 1

        start_idx = next_start_idx

    # Fallback if no candidate met the strict min_seconds criteria
    if not candidates and segments:
        full_text = " ".join(s.text.strip() for s in segments)
        candidates.append(
            CandidateWindow(
                id="cand_001",
                start=round(segments[0].start, 2),
                end=round(segments[-1].end, 2),
                duration=round(segments[-1].end - segments[0].start, 2),
                text=full_text,
                segment_ids=[s.id for s in segments],
            )
        )

    return candidates
