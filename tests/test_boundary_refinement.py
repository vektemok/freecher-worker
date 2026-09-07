"""Unit tests for highlight boundary refinement."""

import pytest
from freecher_worker.highlights.models import Highlight
from freecher_worker.rendering.boundaries import refine_boundaries, refine_highlight
from freecher_worker.transcription.models import Transcript, TranscriptSegment


def _create_mock_transcript() -> Transcript:
    segments = [
        TranscriptSegment(id=0, start=5.0, end=15.0, text="Первое предложение начинается здесь.", words=[]),
        TranscriptSegment(id=1, start=20.0, end=40.0, text="Второе очень важное предложение о масштабировании.", words=[]),
        TranscriptSegment(id=2, start=45.0, end=70.0, text="Третье предложение завершает мысль.", words=[]),
    ]
    return Transcript(
        language="ru",
        language_probability=0.99,
        duration=100.0,
        model="small",
        compute_type="int8",
        device="cpu",
        segments=segments,
    )


def test_refine_start_snaps_to_phrase_beginning():
    """If highlight starts mid-phrase within 5s of phrase start, snap to phrase start - context_before."""
    transcript = _create_mock_transcript()
    # Candidate starts at 22.0 (cuts 2.0s into segment 1 which starts at 20.0)
    r_start, r_end, reason = refine_boundaries(
        start=22.0,
        end=35.0,
        transcript=transcript,
        video_duration=100.0,
        max_shift_seconds=5.0,
        context_before=0.5,
    )

    # Snaps to 20.0 - 0.5 = 19.5
    assert r_start == 19.5
    assert "snapped start earlier" in reason


def test_refine_end_extends_to_phrase_completion():
    """If highlight ends mid-phrase within 5s of phrase end, extend to phrase end + context_after."""
    transcript = _create_mock_transcript()
    # Candidate ends at 38.0 (cuts off segment 1 which ends at 40.0, diff = 2.0s <= 5.0s)
    r_start, r_end, reason = refine_boundaries(
        start=20.0,
        end=38.0,
        transcript=transcript,
        video_duration=100.0,
        max_shift_seconds=5.0,
        context_after=0.5,
    )

    # Extends to 40.0 + 0.5 = 40.5
    assert r_end == 40.5
    assert "extended end by 2.00s" in reason


def test_refine_trims_leading_and_trailing_silence():
    """If candidate has excessive silence before first segment or after last, trim it."""
    transcript = _create_mock_transcript()
    # Candidate starts at 0.0, but first speech is at 5.0 (5.0s silence > 1.0s)
    # Candidate ends at 85.0, but last speech is at 70.0 (15.0s silence > 1.0s)
    r_start, r_end, reason = refine_boundaries(
        start=0.0,
        end=85.0,
        transcript=transcript,
        video_duration=100.0,
        context_before=0.5,
        context_after=0.5,
    )

    # Trimmed start: 5.0 - 0.5 = 4.5
    assert r_start == 4.5
    # Trimmed end: 70.0 + 0.5 = 70.5
    assert r_end == 70.5
    assert "trimmed" in reason


def test_refine_clamp_to_video_duration():
    """Ensure refined boundaries never exceed 0.0 or video_duration."""
    transcript = _create_mock_transcript()
    r_start, r_end, _ = refine_boundaries(
        start=-2.0,
        end=105.0,
        transcript=transcript,
        video_duration=100.0,
    )
    assert r_start >= 0.0
    assert r_end <= 100.0


def test_refine_highlight_model():
    """Test refine_highlight creates a valid RefinedHighlight object."""
    transcript = _create_mock_transcript()
    hl = Highlight(
        rank=1,
        start=22.0,
        end=38.0,
        duration=16.0,
        score=90.0,
        reason="Good hook",
        candidate_id="c01",
        text="Sample text",
        file="clips/clip_01.mp4",
    )

    refined = refine_highlight(hl, transcript, video_duration=100.0)
    assert refined.rank == 1
    assert refined.original_start == 22.0
    assert refined.refined_start == 19.5
    assert refined.refined_end == 40.5
    assert refined.duration == round(40.5 - 19.5, 3)
    assert len(refined.refinement_reason) > 0
