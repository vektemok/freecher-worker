"""Tests for temporal overlap and deduplication."""

from arny_worker.highlights.dedup import (
    calculate_iou,
    calculate_overlap_ratio,
    is_temporally_overlapping,
)
from arny_worker.highlights.models import CandidateWindow


def test_disjoint_windows():
    c1 = CandidateWindow(id="1", start=0.0, end=60.0, duration=60.0, text="A", segment_ids=[0])
    c2 = CandidateWindow(id="2", start=70.0, end=130.0, duration=60.0, text="B", segment_ids=[1])

    assert calculate_overlap_ratio(c1, c2) == 0.0
    assert calculate_iou(c1, c2) == 0.0
    assert not is_temporally_overlapping(c1, c2, threshold=0.60)


def test_identical_windows():
    c1 = CandidateWindow(id="1", start=10.0, end=70.0, duration=60.0, text="A", segment_ids=[0])
    c2 = CandidateWindow(id="2", start=10.0, end=70.0, duration=60.0, text="B", segment_ids=[0])

    assert calculate_overlap_ratio(c1, c2) == 1.0
    assert calculate_iou(c1, c2) == 1.0
    assert is_temporally_overlapping(c1, c2, threshold=0.60)


def test_partial_overlap():
    # c1: [0, 60], c2: [40, 100] -> intersection [40, 60] = 20s. min_dur = 60s. ratio = 20/60 = 0.333
    c1 = CandidateWindow(id="1", start=0.0, end=60.0, duration=60.0, text="A", segment_ids=[0])
    c2 = CandidateWindow(id="2", start=40.0, end=100.0, duration=60.0, text="B", segment_ids=[1])

    ratio = calculate_overlap_ratio(c1, c2)
    assert abs(ratio - (20.0 / 60.0)) < 0.001
    assert not is_temporally_overlapping(c1, c2, threshold=0.60)


def test_high_overlap_dedup():
    # c1: [0, 60], c2: [10, 65] -> intersection [10, 60] = 50s. c2 duration = 55s. ratio = 50/55 = 0.909
    c1 = CandidateWindow(id="1", start=0.0, end=60.0, duration=60.0, text="A", segment_ids=[0])
    c2 = CandidateWindow(id="2", start=10.0, end=65.0, duration=55.0, text="B", segment_ids=[1])

    assert is_temporally_overlapping(c1, c2, threshold=0.60)
