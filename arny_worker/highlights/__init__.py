"""Highlights segmentation, scoring, deduplication, and ranking module."""

from .models import (
    CandidateWindow,
    CandidateDocument,
    Highlight,
    HighlightScore,
    compute_candidate_set_id,
    SEGMENTATION_VERSION,
)
from .segmenter import generate_candidate_windows
from .dedup import calculate_overlap_ratio, calculate_iou, is_temporally_overlapping
from .ranker import rank_and_deduplicate

__all__ = [
    "CandidateWindow",
    "CandidateDocument",
    "Highlight",
    "HighlightScore",
    "compute_candidate_set_id",
    "SEGMENTATION_VERSION",
    "generate_candidate_windows",
    "calculate_overlap_ratio",
    "calculate_iou",
    "is_temporally_overlapping",
    "rank_and_deduplicate",
]
