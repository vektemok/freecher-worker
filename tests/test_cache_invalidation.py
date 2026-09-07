"""Tests for configuration-aware cache invalidation and fingerprinting."""

from pathlib import Path
from unittest.mock import patch

import pytest
from freecher_worker.config import Settings
from freecher_worker.media.fingerprint import compute_source_fingerprint
from freecher_worker.media.probe import MediaInfo
from freecher_worker.pipeline.processor import run_pipeline
from freecher_worker.transcription.models import Transcript, TranscriptSegment
from freecher_worker.transcription.whisper import BaseTranscriber


class CountingTranscriber(BaseTranscriber):
    def __init__(self):
        self.call_count = 0
        self.last_model = None
        self.last_language = None
        self.last_beam_size = None

    def transcribe(self, audio_path, language=None, source_fingerprint_id=None):
        self.call_count += 1
        self.last_language = language
        segments = [
            TranscriptSegment(id=0, start=0.0, end=30.0, text="Первая фраза для проверки кеша."),
            TranscriptSegment(id=1, start=31.0, end=65.0, text="Вторая фраза с вопросом: почему это работает?"),
            TranscriptSegment(id=2, start=66.0, end=95.0, text="Третья фраза с невероятным результатом!"),
        ]
        return Transcript(
            source_fingerprint_id=source_fingerprint_id,
            language=language or "ru",
            language_probability=0.99,
            duration=100.0,
            model=getattr(self, "model_name", "small"),
            compute_type="int8",
            device="cpu",
            beam_size=getattr(self, "beam_size", 5),
            vad_filter=getattr(self, "vad_filter", True),
            segments=segments,
        )


def fake_clip(source_video, output_clip, *args, **kwargs):
    p = Path(output_clip)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"dummy_clip_data")
    return p


def test_source_fingerprint_changes_on_file_modification(tmp_path):
    vid = tmp_path / "video.mp4"
    vid.write_bytes(b"initial_bytes_12345" * 100)
    fp1 = compute_source_fingerprint(vid, duration_seconds=60.0)

    # Modify file content
    vid.write_bytes(b"modified_bytes_67890" * 200)
    fp2 = compute_source_fingerprint(vid, duration_seconds=60.0)

    assert fp1.fingerprint_id != fp2.fingerprint_id
    assert fp1.file_size != fp2.file_size
    assert fp1.content_hash != fp2.content_hash


@patch("freecher_worker.pipeline.processor.clip_video")
@patch("freecher_worker.pipeline.processor.extract_audio")
@patch("freecher_worker.pipeline.processor.probe_media")
def test_cache_invalidation_on_asr_param_change(mock_probe, mock_extract, mock_clip, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"sample_content_for_cache_test" * 50)

    mock_probe.return_value = MediaInfo(
        path=str(video),
        duration_seconds=100.0,
        width=1280,
        height=720,
        fps=25.0,
        video_codec="h264",
        audio_codec="aac",
        file_size=len(b"sample_content_for_cache_test" * 50),
        has_audio=True,
    )
    mock_extract.side_effect = lambda src, dst: Path(dst).touch() or Path(dst)
    mock_clip.side_effect = fake_clip

    transcriber = CountingTranscriber()
    run_dir = tmp_path / "runs" / "fixed_run"

    # 1. First run with model="small", language="ru", beam_size=5
    cfg1 = Settings(
        asr_model="small",
        asr_language="ru",
        asr_beam_size=5,
        asr_device="cpu",
        asr_compute_type="int8",
        output_dir=tmp_path / "runs",
    )
    transcriber.model_name = "small"
    transcriber.beam_size = 5

    run_pipeline(video, config=cfg1, run_id="fixed_run", transcriber=transcriber)
    assert transcriber.call_count == 1

    # 2. Run with EXACT SAME params -> cache HIT, call_count should stay 1
    run_pipeline(video, config=cfg1, run_id="fixed_run", transcriber=transcriber)
    assert transcriber.call_count == 1

    # 3. Invalidate: change model to "medium"
    cfg2 = Settings(
        asr_model="medium",
        asr_language="ru",
        asr_beam_size=5,
        asr_device="cpu",
        asr_compute_type="int8",
        output_dir=tmp_path / "runs",
    )
    transcriber.model_name = "medium"
    run_pipeline(video, config=cfg2, run_id="fixed_run", transcriber=transcriber)
    assert transcriber.call_count == 2

    # 4. Invalidate: change language to "en"
    cfg3 = Settings(
        asr_model="medium",
        asr_language="en",
        asr_beam_size=5,
        asr_device="cpu",
        asr_compute_type="int8",
        output_dir=tmp_path / "runs",
    )
    run_pipeline(video, config=cfg3, run_id="fixed_run", transcriber=transcriber)
    assert transcriber.call_count == 3

    # 5. Invalidate: change beam_size to 8
    cfg4 = Settings(
        asr_model="medium",
        asr_language="en",
        asr_beam_size=8,
        asr_device="cpu",
        asr_compute_type="int8",
        output_dir=tmp_path / "runs",
    )
    transcriber.beam_size = 8
    run_pipeline(video, config=cfg4, run_id="fixed_run", transcriber=transcriber)
    assert transcriber.call_count == 4


@patch("freecher_worker.pipeline.processor.clip_video")
@patch("freecher_worker.pipeline.processor.extract_audio")
@patch("freecher_worker.pipeline.processor.probe_media")
def test_candidate_cache_invalidation_on_window_param_change(mock_probe, mock_extract, mock_clip, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"content_for_window_test" * 50)

    mock_probe.return_value = MediaInfo(
        path=str(video),
        duration_seconds=100.0,
        width=1280,
        height=720,
        fps=25.0,
        video_codec="h264",
        audio_codec="aac",
        file_size=len(b"content_for_window_test" * 50),
        has_audio=True,
    )
    mock_extract.side_effect = lambda src, dst: Path(dst).touch() or Path(dst)
    mock_clip.side_effect = fake_clip

    transcriber = CountingTranscriber()
    transcriber.model_name = "small"

    cfg_a = Settings(
        asr_model="small",
        asr_device="cpu",
        asr_compute_type="int8",
        highlight_target_seconds=60.0,
        output_dir=tmp_path / "runs",
    )
    man1 = run_pipeline(video, config=cfg_a, run_id="cand_test_run", transcriber=transcriber)
    assert transcriber.call_count == 1

    # Invalidate candidate cache by changing target_seconds to 35.0
    cfg_b = Settings(
        asr_model="small",
        asr_device="cpu",
        asr_compute_type="int8",
        highlight_target_seconds=35.0,
        output_dir=tmp_path / "runs",
    )
    man2 = run_pipeline(video, config=cfg_b, run_id="cand_test_run", transcriber=transcriber)
    # Transcription was cached (call_count still 1), but candidates were regenerated with new target
    assert transcriber.call_count == 1
    assert man2.candidate_config.target_seconds == 35.0
