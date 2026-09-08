"""Timestamp semantics: candidate-relative offsets must never mix with absolute source times."""

import pytest

from freecher_worker.shorts.timeframe import (
    CandidateTimeframe,
    OffsetRegion,
    TimestampSemanticsError,
    interpret_observed_region,
)


def make_timeframe(start: float = 1234.2, end: float = 1294.2) -> CandidateTimeframe:
    return CandidateTimeframe(candidate_id="cand_037", source_start_sec=start, source_end_sec=end)


def test_timeframe_reports_duration_and_converts_both_directions():
    tf = make_timeframe()
    assert tf.duration_sec == pytest.approx(60.0)
    assert tf.to_source(8.4) == pytest.approx(1242.6)
    assert tf.to_offset(1242.6) == pytest.approx(8.4)


def test_timeframe_rejects_inverted_window():
    # Raised inside a pydantic validator, so it surfaces wrapped in ValidationError (a ValueError).
    with pytest.raises(ValueError):
        CandidateTimeframe(candidate_id="c", source_start_sec=100.0, source_end_sec=100.0)


@pytest.mark.parametrize("offset", [-0.5, 60.5, 1242.6])
def test_offset_outside_candidate_is_rejected(offset):
    tf = make_timeframe()
    with pytest.raises(TimestampSemanticsError):
        tf.validate_offset(offset)


def test_absolute_timestamp_outside_candidate_is_rejected():
    tf = make_timeframe()
    with pytest.raises(TimestampSemanticsError):
        tf.to_offset(30.0)


def test_offset_region_requires_positive_span():
    with pytest.raises(ValueError):
        OffsetRegion(start_offset_sec=10.0, end_offset_sec=10.0)


def test_offset_region_bound_to_wrong_candidate_is_rejected():
    region = OffsetRegion(start_offset_sec=1.0, end_offset_sec=40.0)
    short_tf = CandidateTimeframe(candidate_id="c", source_start_sec=500.0, source_end_sec=520.0)
    with pytest.raises(TimestampSemanticsError):
        region.bind(short_tf)


def test_relative_offsets_are_accepted_as_offsets():
    tf = make_timeframe()
    result = interpret_observed_region({"start_offset": 8.4, "end_offset": 31.8}, tf)
    assert result.accepted
    assert result.interpretation == "offset"
    assert result.region.start_offset_sec == pytest.approx(8.4)
    assert result.region.end_offset_sec == pytest.approx(31.8)


def test_absolute_source_timestamps_are_detected_and_converted():
    """The historic bug: a payload carrying absolute source times in offset fields."""
    tf = make_timeframe()
    result = interpret_observed_region({"start_offset": 1242.6, "end_offset": 1266.0}, tf)
    assert result.accepted
    assert result.interpretation == "absolute_converted"
    assert result.region.start_offset_sec == pytest.approx(8.4)
    assert result.region.end_offset_sec == pytest.approx(31.8)


def test_ambiguous_region_is_rejected_in_strict_mode():
    """A candidate starting before its own duration makes both readings valid; refuse to guess."""
    tf = CandidateTimeframe(candidate_id="cand_001", source_start_sec=10.0, source_end_sec=70.0)
    result = interpret_observed_region({"start_offset": 20.0, "end_offset": 40.0}, tf, strict=True)
    assert not result.accepted
    assert result.interpretation == "ambiguous"
    assert result.region is None


def test_ambiguous_region_falls_back_to_offset_when_not_strict():
    tf = CandidateTimeframe(candidate_id="cand_001", source_start_sec=10.0, source_end_sec=70.0)
    result = interpret_observed_region({"start_offset": 20.0, "end_offset": 40.0}, tf, strict=False)
    assert result.accepted
    assert result.interpretation == "offset"


def test_candidate_starting_at_zero_is_not_ambiguous():
    """With source_start == 0 the two coordinate systems coincide, so there is nothing to resolve."""
    tf = CandidateTimeframe(candidate_id="cand_000", source_start_sec=0.0, source_end_sec=60.0)
    result = interpret_observed_region({"start_offset": 5.0, "end_offset": 25.0}, tf, strict=True)
    assert result.accepted
    assert result.interpretation == "offset"


def test_out_of_range_region_is_rejected():
    tf = make_timeframe()
    result = interpret_observed_region({"start_offset": 5000.0, "end_offset": 5010.0}, tf)
    assert not result.accepted
    assert result.interpretation == "out_of_range"


@pytest.mark.parametrize(
    "payload",
    [None, {}, {"start_offset": 5.0}, {"start_offset": "a", "end_offset": "b"}, {"start_offset": 20.0, "end_offset": 10.0}],
)
def test_unusable_payloads_never_produce_a_region(payload):
    tf = make_timeframe()
    result = interpret_observed_region(payload, tf)
    assert not result.accepted
    assert result.region is None
