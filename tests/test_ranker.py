"""Tests for ranking and top-K highlight selection."""

import pytest
from freecher_worker.highlights.models import CandidateWindow, HighlightScore
from freecher_worker.highlights.ranker import rank_and_deduplicate


def test_ranker_sorting_and_top_k():
    candidates = [
        CandidateWindow(id="c1", start=0.0, end=60.0, duration=60.0, text="One", segment_ids=[0]),
        CandidateWindow(id="c2", start=100.0, end=160.0, duration=60.0, text="Two", segment_ids=[1]),
        CandidateWindow(id="c3", start=200.0, end=260.0, duration=60.0, text="Three", segment_ids=[2]),
        CandidateWindow(id="c4", start=300.0, end=360.0, duration=60.0, text="Four", segment_ids=[3]),
    ]
    scores = [
        HighlightScore(score=70.0, hook_score=70, standalone_score=70, emotion_score=70, information_score=70, shareability_score=70, reason="ok"),
        HighlightScore(score=95.0, hook_score=95, standalone_score=95, emotion_score=95, information_score=95, shareability_score=95, reason="best"),
        HighlightScore(score=85.0, hook_score=85, standalone_score=85, emotion_score=85, information_score=85, shareability_score=85, reason="great"),
        HighlightScore(score=60.0, hook_score=60, standalone_score=60, emotion_score=60, information_score=60, shareability_score=60, reason="fair"),
    ]

    highlights = rank_and_deduplicate(candidates, scores, top_k=2, overlap_threshold=0.60)
    assert len(highlights) == 2
    assert highlights[0].rank == 1
    assert highlights[0].candidate_id == "c2"  # score 95.0
    assert highlights[0].score == 95.0

    assert highlights[1].rank == 2
    assert highlights[1].candidate_id == "c3"  # score 85.0
    assert highlights[1].score == 85.0


def test_ranker_deduplication():
    # c1 and c2 heavily overlap, c1 has higher score -> c2 suppressed
    c1 = CandidateWindow(id="c1", start=0.0, end=60.0, duration=60.0, text="A", segment_ids=[0])
    c2 = CandidateWindow(id="c2", start=5.0, end=65.0, duration=60.0, text="B", segment_ids=[1])
    c3 = CandidateWindow(id="c3", start=120.0, end=180.0, duration=60.0, text="C", segment_ids=[2])

    candidates = [c1, c2, c3]
    scores = [
        HighlightScore(score=90.0, hook_score=90, standalone_score=90, emotion_score=90, information_score=90, shareability_score=90, reason="top"),
        HighlightScore(score=80.0, hook_score=80, standalone_score=80, emotion_score=80, information_score=80, shareability_score=80, reason="suppressed"),
        HighlightScore(score=75.0, hook_score=75, standalone_score=75, emotion_score=75, information_score=75, shareability_score=75, reason="distinct"),
    ]

    highlights = rank_and_deduplicate(candidates, scores, top_k=5, overlap_threshold=0.60)
    assert len(highlights) == 2
    assert highlights[0].candidate_id == "c1"
    assert highlights[1].candidate_id == "c3"


def test_ranker_mismatched_lengths():
    with pytest.raises(ValueError):
        rank_and_deduplicate([], [HighlightScore(score=50, hook_score=50, standalone_score=50, emotion_score=50, information_score=50, shareability_score=50, reason="")])
