"""Tests for candidate window generation / segmentation."""

from arny_worker.highlights.segmenter import generate_candidate_windows
from arny_worker.transcription.models import Transcript, TranscriptSegment


def test_empty_transcript():
    transcript = Transcript(
        language="ru",
        language_probability=1.0,
        duration=0.0,
        model="small",
        compute_type="int8",
        device="cuda",
        segments=[],
    )
    candidates = generate_candidate_windows(transcript)
    assert candidates == []


def test_short_video_single_candidate():
    # Video total duration is 20s (< min 30s)
    segments = [
        TranscriptSegment(id=0, start=1.0, end=8.0, text="Первая короткая фраза."),
        TranscriptSegment(id=1, start=10.0, end=18.0, text="Вторая фраза видео."),
    ]
    transcript = Transcript(
        language="ru",
        language_probability=0.99,
        duration=20.0,
        model="small",
        compute_type="int8_float16",
        device="cuda",
        segments=segments,
    )
    candidates = generate_candidate_windows(transcript, min_seconds=30.0, target_seconds=60.0)
    assert len(candidates) == 1
    assert candidates[0].start == 1.0
    assert candidates[0].end == 18.0
    assert candidates[0].duration == 17.0
    assert candidates[0].segment_ids == [0, 1]
    assert "Первая короткая фраза." in candidates[0].text
    assert "Вторая фраза видео." in candidates[0].text


def test_normal_sliding_window():
    # Create 20 segments of ~10s each = ~200s total
    segments = []
    for i in range(20):
        start = float(i * 10)
        end = float(start + 8.5)
        segments.append(
            TranscriptSegment(
                id=i,
                start=start,
                end=end,
                text=f"Фрагмент речи номер {i}.",
            )
        )

    transcript = Transcript(
        language="ru",
        language_probability=0.99,
        duration=200.0,
        model="small",
        compute_type="int8_float16",
        device="cuda",
        segments=segments,
    )

    candidates = generate_candidate_windows(
        transcript,
        min_seconds=30.0,
        target_seconds=60.0,
        max_seconds=90.0,
        overlap_seconds=15.0,
    )

    assert len(candidates) >= 3
    for cand in candidates:
        assert cand.duration >= 30.0 * 0.6  # allowing for edge cases
        assert cand.duration <= 90.0
        assert len(cand.segment_ids) > 0
        assert cand.id.startswith("cand_")
        # Check text corresponds to segments
        for seg_id in cand.segment_ids:
            assert f"номер {seg_id}" in cand.text
