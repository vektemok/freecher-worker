"""Temporal overlap calculation and deduplication (Non-Maximum Suppression)."""

from __future__ import annotations

from typing import Sequence
from .models import CandidateWindow


def calculate_overlap_ratio(cand_a: CandidateWindow, cand_b: CandidateWindow) -> float:
    """Calculate temporal overlap ratio between two candidate windows.

    Returns the ratio of the intersection duration to the smaller of the two candidate durations.
    This safely catches both partial overlaps and sub-interval containment.
    """
    intersection = max(0.0, min(cand_a.end, cand_b.end) - max(cand_a.start, cand_b.start))
    if intersection <= 0.0:
        return 0.0

    min_dur = min(cand_a.duration, cand_b.duration)
    if min_dur <= 0.0:
        return 0.0

    return intersection / min_dur


def calculate_iou(cand_a: CandidateWindow, cand_b: CandidateWindow) -> float:
    """Calculate Intersection over Union (IoU) of two candidate windows."""
    intersection = max(0.0, min(cand_a.end, cand_b.end) - max(cand_a.start, cand_b.start))
    if intersection <= 0.0:
        return 0.0

    union = (cand_a.end - cand_a.start) + (cand_b.end - cand_b.start) - intersection
    return intersection / union if union > 0 else 0.0


def is_temporally_overlapping(
    cand_a: CandidateWindow,
    cand_b: CandidateWindow,
    threshold: float = 0.60,
) -> bool:
    """Return True if overlap between cand_a and cand_b exceeds threshold."""
    return calculate_overlap_ratio(cand_a, cand_b) >= threshold
