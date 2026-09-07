"""Tests for transcription models and JSON serialization."""

from freecher_worker.transcription.models import Transcript, TranscriptSegment
from freecher_worker.utils.json_io import load_json, save_json


def test_transcript_model_serialization(tmp_path):
    seg1 = TranscriptSegment(id=0, start=0.0, end=4.5, text="Hello world", avg_logprob=-0.15, no_speech_prob=0.01)
    seg2 = TranscriptSegment(id=1, start=4.8, end=9.2, text="This is a test transcript.", avg_logprob=-0.22)
    transcript = Transcript(
        language="en",
        language_probability=0.98,
        duration=10.0,
        model="small",
        compute_type="int8_float16",
        device="cuda",
        segments=[seg1, seg2],
    )

    out_file = tmp_path / "transcript.json"
    save_json(transcript, out_file)
    assert out_file.is_file()

    loaded_raw = load_json(out_file)
    loaded = Transcript.model_validate(loaded_raw)
    assert loaded.language == "en"
    assert loaded.model == "small"
    assert len(loaded.segments) == 2
    assert loaded.segments[0].text == "Hello world"
    assert loaded.segments[0].avg_logprob == -0.15
