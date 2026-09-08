"""Dynamic Subclip Refinement: pick the strongest finished fragment inside a candidate."""

import pytest

from freecher_worker.shorts.refinement import (
    DURATION_MODE_FULL,
    SubclipConfig,
    build_speech_units,
    find_payoff_block,
    refine_subclip,
)
from freecher_worker.shorts.signals import (
    build_curve_from_activity_profile,
    build_curve_from_transcript,
    build_flat_curve,
)
from freecher_worker.shorts.timeframe import CandidateTimeframe, OffsetRegion
from freecher_worker.transcription.models import Transcript, TranscriptSegment

CANDIDATE_START = 1234.2
CANDIDATE_END = 1294.2


def make_transcript(spans):
    """Build a transcript from (offset_start, offset_end, text) triples."""
    segments = [
        TranscriptSegment(id=i, start=CANDIDATE_START + a, end=CANDIDATE_START + b, text=text)
        for i, (a, b, text) in enumerate(spans)
    ]
    return Transcript(
        language="ru",
        duration=3600.0,
        model="small",
        compute_type="int8",
        device="cpu",
        segments=segments,
    )


DULL_INTRO_STRONG_MIDDLE = [
    (0.0, 1.2, "Ну."),
    (3.0, 7.0, "Так вот, представь себе ситуацию, мы стоим прямо посреди дороги и ничего не понимаем."),
    (7.0, 12.0, "И тут этот парень поворачивается, смотрит на нас спокойно и говорит невероятную вещь."),
    (12.0, 18.0, "Он говорит что всю неделю вообще не спал и делал это исключительно ради спора."),
    (18.0, 25.0, "И в этот момент мы все просто взорвались от смеха, потому что это оказалось правдой!"),
] + [(float(i), float(i) + 1.0, "Ага.") for i in range(26, 60, 5)]


def make_timeframe(start: float = CANDIDATE_START, end: float = CANDIDATE_END) -> CandidateTimeframe:
    return CandidateTimeframe(candidate_id="cand_037", source_start_sec=start, source_end_sec=end)


def refine(transcript, timeframe=None, config=None, **kwargs):
    tf = timeframe or make_timeframe()
    curve = build_curve_from_transcript(transcript, tf)
    return refine_subclip(tf, curve, transcript, config or SubclipConfig(), **kwargs)


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_sixty_second_candidate_yields_a_short_subclip():
    """A 60 s analysis window must not become a 60 s short."""
    transcript = make_transcript(DULL_INTRO_STRONG_MIDDLE)
    selection = refine(transcript)

    assert 15.0 <= selection.duration_sec <= 30.0
    assert selection.duration_sec < 30.0
    assert selection.candidate_duration_sec == pytest.approx(60.0)


def test_dull_intro_is_trimmed_and_payoff_is_kept():
    """Matches the worked example: boring 0-3 s dropped, strong 3-25 s kept."""
    transcript = make_transcript(DULL_INTRO_STRONG_MIDDLE)
    selection = refine(transcript)

    assert 2.0 <= selection.start_offset_sec <= 4.0
    assert 24.0 <= selection.end_offset_sec <= 27.0
    assert selection.components["payoff_coverage"] >= 0.9
    assert selection.components["setup_preserved"] == pytest.approx(1.0)


def test_durations_are_not_snapped_to_round_numbers():
    transcript = make_transcript(DULL_INTRO_STRONG_MIDDLE)
    selection = refine(transcript)
    assert selection.duration_sec not in (15.0, 30.0, 45.0, 60.0)


def test_offsets_and_absolute_timestamps_stay_consistent():
    transcript = make_transcript(DULL_INTRO_STRONG_MIDDLE)
    selection = refine(transcript)

    assert selection.short_source_start_sec == pytest.approx(
        CANDIDATE_START + selection.start_offset_sec, abs=1e-3
    )
    assert selection.short_source_end_sec == pytest.approx(
        CANDIDATE_START + selection.end_offset_sec, abs=1e-3
    )
    assert selection.duration_sec == pytest.approx(
        selection.end_offset_sec - selection.start_offset_sec, abs=1e-3
    )
    assert 0.0 <= selection.start_offset_sec < selection.end_offset_sec <= selection.candidate_duration_sec


def test_selection_is_deterministic():
    transcript = make_transcript(DULL_INTRO_STRONG_MIDDLE)
    first = refine(transcript)
    second = refine(transcript)
    assert first.model_dump() == second.model_dump()


# ---------------------------------------------------------------------------
# Duration bounds
# ---------------------------------------------------------------------------


def test_minimum_duration_is_respected():
    transcript = make_transcript(DULL_INTRO_STRONG_MIDDLE)
    config = SubclipConfig(min_duration_sec=20.0, target_min_duration_sec=22.0,
                           target_max_duration_sec=28.0, max_duration_sec=45.0)
    selection = refine(transcript, config=config)
    assert selection.duration_sec >= 20.0


def test_maximum_duration_is_respected():
    transcript = make_transcript(DULL_INTRO_STRONG_MIDDLE)
    config = SubclipConfig(min_duration_sec=8.0, target_min_duration_sec=10.0,
                           target_max_duration_sec=12.0, max_duration_sec=14.0)
    selection = refine(transcript, config=config)
    assert selection.duration_sec <= 14.0


def test_candidate_shorter_than_the_minimum_is_kept_whole():
    tf = make_timeframe(100.0, 106.0)
    transcript = make_transcript([(0.0, 6.0, "Короткий кандидат целиком.")])
    curve = build_curve_from_transcript(transcript, tf)
    selection = refine_subclip(tf, curve, transcript, SubclipConfig(min_duration_sec=8.0))

    assert selection.boundary_source == "full_candidate"
    assert selection.start_offset_sec == 0.0
    assert selection.duration_sec == pytest.approx(6.0)


def test_full_duration_mode_keeps_the_candidate_clamped_to_the_maximum():
    transcript = make_transcript(DULL_INTRO_STRONG_MIDDLE)
    selection = refine(transcript, duration_mode=DURATION_MODE_FULL)

    assert selection.duration_mode == DURATION_MODE_FULL
    assert selection.duration_sec == pytest.approx(45.0)


def test_unknown_duration_mode_is_rejected():
    transcript = make_transcript(DULL_INTRO_STRONG_MIDDLE)
    with pytest.raises(ValueError, match="duration_mode"):
        refine(transcript, duration_mode="square")


def test_configured_bounds_are_normalized_into_a_consistent_order():
    config = SubclipConfig(min_duration_sec=30.0, target_min_duration_sec=5.0,
                           target_max_duration_sec=8.0, max_duration_sec=12.0).clamped()
    assert config.min_duration_sec <= config.target_min_duration_sec
    assert config.target_min_duration_sec <= config.target_max_duration_sec
    assert config.target_max_duration_sec <= config.max_duration_sec


# ---------------------------------------------------------------------------
# Boundary quality
# ---------------------------------------------------------------------------


def test_boundaries_snap_to_speech_units_not_mid_phrase():
    transcript = make_transcript(DULL_INTRO_STRONG_MIDDLE)
    tf = make_timeframe()
    units = build_speech_units(transcript, tf)
    selection = refine(transcript)

    unit_starts = {round(max(0.0, u.start_offset_sec - 0.15), 2) for u in units} | {0.0}
    unit_ends = {round(min(60.0, u.end_offset_sec + 0.30), 2) for u in units} | {60.0}
    assert round(selection.start_offset_sec, 2) in unit_starts
    assert round(selection.end_offset_sec, 2) in unit_ends
    assert selection.boundary_source == "speech_units"


def test_speech_units_are_clipped_to_the_candidate_window():
    tf = make_timeframe()
    transcript = Transcript(
        language="ru", duration=3600.0, model="small", compute_type="int8", device="cpu",
        segments=[
            TranscriptSegment(id=0, start=CANDIDATE_START - 20.0, end=CANDIDATE_START + 5.0, text="Через край."),
            TranscriptSegment(id=1, start=CANDIDATE_END - 5.0, end=CANDIDATE_END + 40.0, text="И снова."),
        ],
    )
    units = build_speech_units(transcript, tf)
    assert all(0.0 <= u.start_offset_sec <= 60.0 for u in units)
    assert all(0.0 <= u.end_offset_sec <= 60.0 for u in units)


def test_grid_fallback_when_no_transcript_is_available():
    tf = make_timeframe()
    curve = build_flat_curve(tf)
    selection = refine_subclip(tf, curve, transcript=None, config=SubclipConfig())

    assert selection.boundary_source == "grid"
    assert 8.0 <= selection.duration_sec <= 45.0


def test_payoff_block_covers_the_contiguous_strong_region():
    transcript = make_transcript(DULL_INTRO_STRONG_MIDDLE)
    tf = make_timeframe()
    curve = build_curve_from_transcript(transcript, tf)
    block_start, block_end = find_payoff_block(curve, curve.peak_offset(), 0.25)

    assert block_start <= 4.0
    assert 24.0 <= block_end <= 28.0


# ---------------------------------------------------------------------------
# Advisory region wiring
# ---------------------------------------------------------------------------


def _overlap(selection, region: OffsetRegion) -> float:
    return max(
        0.0,
        min(selection.end_offset_sec, region.end_offset_sec)
        - max(selection.start_offset_sec, region.start_offset_sec),
    )


def test_advisory_region_pulls_the_selection_toward_itself():
    """On a featureless curve the local signal cannot decide, so the model's read should move it."""
    transcript = make_transcript(
        [(float(i), float(i) + 5.0, "Ровный и одинаково плотный текст без пауз для этого сегмента.")
         for i in range(0, 60, 5)]
    )
    tf = make_timeframe()
    curve = build_curve_from_transcript(transcript, tf)
    advisory = OffsetRegion(start_offset_sec=35.0, end_offset_sec=55.0)

    without = refine_subclip(tf, curve, transcript, SubclipConfig())
    with_advisory = refine_subclip(
        tf,
        curve,
        transcript,
        SubclipConfig(),
        advisory=advisory,
        advisory_interpretation="offset",
    )

    assert with_advisory.advisory_used is True
    assert with_advisory.advisory_interpretation == "offset"
    assert "advisory_overlap" in with_advisory.components
    assert _overlap(with_advisory, advisory) > _overlap(without, advisory)


def test_scores_stay_comparable_with_and_without_an_advisory_region():
    """Inactive components are renormalized away rather than scoring as zero."""
    transcript = make_transcript(DULL_INTRO_STRONG_MIDDLE)
    tf = make_timeframe()
    curve = build_curve_from_transcript(transcript, tf)
    without = refine_subclip(tf, curve, transcript, SubclipConfig())
    assert "advisory_overlap" not in without.components
    assert without.score > 50.0


# ---------------------------------------------------------------------------
# Signal sources
# ---------------------------------------------------------------------------


def test_cached_activity_profile_is_sliced_into_candidate_offsets():
    tf = make_timeframe()
    profile = {
        "bin_size_seconds": 1.0,
        "timeline": [
            {
                "absolute_timestamp": CANDIDATE_START + i,
                "audio_energy": 0.2,
                "audio_delta": 0.1,
                "speech_activity": 0.5,
                "visual_motion": 0.1,
                "scene_change": i == 30,
                "combined_activity": 0.9 if 10 <= i < 35 else 0.1,
            }
            for i in range(60)
        ]
        # Bins outside the candidate must be ignored entirely.
        + [{"absolute_timestamp": CANDIDATE_END + 5, "combined_activity": 1.0}],
    }
    curve = build_curve_from_activity_profile(profile, tf)

    assert curve.source == "source_activity_profile"
    assert len(curve.offsets) == 60
    assert curve.offsets[0] == pytest.approx(0.0)
    assert max(curve.offsets) < 60.0
    assert 10.0 <= curve.peak_offset() < 35.0
    assert curve.scene_cut_offsets() == [30.0]


def test_activity_profile_drives_selection_toward_its_peak():
    tf = make_timeframe()
    transcript = make_transcript(
        [(float(i), float(i) + 4.5, "Одинаково плотный текст без явных различий между сегментами.")
         for i in range(0, 60, 5)]
    )
    profile = {
        "bin_size_seconds": 1.0,
        "timeline": [
            {
                "absolute_timestamp": CANDIDATE_START + i,
                "audio_energy": 0.5,
                "audio_delta": 0.2,
                "speech_activity": 0.8,
                "visual_motion": 0.2,
                "scene_change": False,
                "combined_activity": 0.95 if 30 <= i < 50 else 0.05,
            }
            for i in range(60)
        ],
    }
    curve = build_curve_from_activity_profile(profile, tf)
    selection = refine_subclip(tf, curve, transcript, SubclipConfig())

    assert selection.signal_source == "source_activity_profile"
    assert selection.start_offset_sec >= 25.0
    assert selection.end_offset_sec <= 55.0
