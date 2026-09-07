"""Ranking and top-K highlight selection with temporal deduplication."""

from __future__ import annotations

from typing import Optional
from .models import CandidateWindow, Highlight, HighlightScore
from .dedup import is_temporally_overlapping


def rank_and_deduplicate(
    candidates: list[CandidateWindow],
    scores: list[HighlightScore],
    top_k: int = 5,
    overlap_threshold: float = 0.60,
) -> list[Highlight]:
    """Rank scored candidate windows, apply temporal NMS deduplication, and return top-K highlights.

    Args:
        candidates: List of CandidateWindow objects.
        scores: Corresponding list of HighlightScore objects.
        top_k: Maximum number of top highlights to retain.
        overlap_threshold: Maximum overlap ratio allowed before suppressing a lower-scored candidate.

    Returns:
        List of Highlight objects ordered by rank 1..K.
    """
    if len(candidates) != len(scores):
        raise ValueError(
            f"Mismatch between number of candidates ({len(candidates)}) and scores ({len(scores)})"
        )

    if not candidates:
        return []

    # Pair candidates with their scores and sort descending by overall score
    paired = list(zip(candidates, scores))
    paired.sort(key=lambda item: item[1].score, reverse=True)

    selected: list[tuple[CandidateWindow, HighlightScore]] = []

    for cand, score in paired:
        # Check overlap against all already selected candidates
        has_overlap = False
        for chosen_cand, _ in selected:
            if is_temporally_overlapping(cand, chosen_cand, threshold=overlap_threshold):
                has_overlap = True
                break

        if not has_overlap:
            selected.append((cand, score))
            if len(selected) >= top_k:
                break

    highlights: list[Highlight] = []
    for rank, (cand, score) in enumerate(selected, start=1):
        highlights.append(
            Highlight(
                rank=rank,
                start=cand.start,
                end=cand.end,
                duration=cand.duration,
                score=score.score,
                reason=score.reason,
                candidate_id=cand.id,
                text=cand.text,
                score_breakdown=score,
            )
        )

    return highlights
