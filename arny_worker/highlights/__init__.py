"""Highlights segmentation, scoring, deduplication, and ranking module."""

from .models import CandidateWindow, Highlight, HighlightScore
from .segmenter import generate_candidate_windows
from .dedup import calculate_overlap_ratio, calculate_iou, is_temporally_overlapping
from .ranker import rank_and_deduplicate

__all__ = [
    "CandidateWindow",
    "Highlight",
    "HighlightScore",
    "generate_candidate_windows",
    "calculate_overlap_ratio",
    "calculate_iou",
    "is_temporally_overlapping",
    "rank_and_deduplicate",
]
