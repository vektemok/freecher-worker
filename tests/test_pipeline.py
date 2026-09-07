"""Tests for end-to-end pipeline orchestration and caching."""

from pathlib import Path
from unittest.mock import patch

from arny_worker.config import Settings
from arny_worker.media.probe import MediaInfo
from arny_worker.pipeline.processor import PIPELINE_VERSION, run_pipeline
from arny_worker.transcription.models import Transcript, TranscriptSegment
from arny_worker.transcription.whisper import BaseTranscriber


class FakeTranscriber(BaseTranscriber):
    def __init__(self):
        self.call_count = 0

    def transcribe(self, audio_path, language=None, source_fingerprint_id=None):
        self.call_count += 1
        segments = [
            TranscriptSegment(id=0, start=0.0, end=15.0, text="Привет всем, это первое введение в тему!"),
            TranscriptSegment(id=1, start=16.0, end=35.0, text="Почему это так важно? Давайте разберем главную ошибку."),
            TranscriptSegment(id=2, start=36.0, end=70.0, text="Вау, невероятно, 100 тысяч рублей было сэкономлено благодаря этому правилу!"),
            TranscriptSegment(id=3, start=71.0, end=110.0, text="В итоге результат превзошел все ожидания, обязательно попробуйте сами."),
        ]
        return Transcript(
            source_fingerprint_id=source_fingerprint_id,
            language="ru",
            language_probability=0.99,
            duration=115.0,
            model="small",
            compute_type="int8_float16",
            device="cuda",
            beam_size=5,
            vad_filter=True,
            segments=segments,
        )


@patch("arny_worker.pipeline.processor.clip_video")
@patch("arny_worker.pipeline.processor.extract_audio")
@patch("arny_worker.pipeline.processor.probe_media")
def test_pipeline_run_and_cache(mock_probe, mock_extract, mock_clip, tmp_path):
    video_file = tmp_path / "sample_video.mp4"
    video_file.write_bytes(b"dummy_video_content_for_test")

    media_info = MediaInfo(
        path=str(video_file),
        duration_seconds=115.0,
        width=1920,
        height=1080,
        fps=30.0,
        video_codec="h264",
        audio_codec="aac",
        file_size=len(b"dummy_video_content_for_test"),
        has_audio=True,
    )
    mock_probe.return_value = media_info

    # Make extract_audio touch the target wav
    def fake_extract(src, dst):
        p = Path(dst)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"dummy_wav_data")
        return p
    mock_extract.side_effect = fake_extract

    # Make clip_video touch the clip file
    def fake_clip(source_video, output_clip, start_seconds, end_seconds, **kwargs):
        p = Path(output_clip)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"dummy_clip_data")
        return p
    mock_clip.side_effect = fake_clip

    config = Settings(
        asr_model="small",
        asr_device="cuda",
        asr_compute_type="int8_float16",
        highlight_top_k=2,
        output_dir=tmp_path / "runs",
    )

    fake_transcriber = FakeTranscriber()

    # FIRST RUN: Full execution
    manifest1 = run_pipeline(
        video_path=video_file,
        output_dir=tmp_path / "runs",
        config=config,
        force=False,
        transcriber=fake_transcriber,
    )

    assert mock_probe.call_count == 1
    assert mock_extract.call_count == 1
    assert fake_transcriber.call_count == 1
    assert len(manifest1.highlights) > 0
    assert manifest1.highlights[0].rank == 1
    assert manifest1.timings.total_seconds >= 0.0
    assert manifest1.pipeline_version == PIPELINE_VERSION

    # Check files created
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    active_run = run_dirs[0]

    assert (active_run / "media.json").is_file()
    assert (active_run / "audio.wav").is_file()
    assert (active_run / "transcript.json").is_file()
    assert (active_run / "candidates.json").is_file()
    assert (active_run / "highlights.json").is_file()
    assert (active_run / "manifest.json").is_file()
    assert (active_run / "logs" / "worker.log").is_file()

    # SECOND RUN: Test caching - probe, audio extraction, and transcription should be reused
    manifest2 = run_pipeline(
        video_path=video_file,
        output_dir=tmp_path / "runs",
        config=config,
        force=False,
        transcriber=fake_transcriber,
    )

    # Transcriber call count must NOT increase because transcript.json was reused
    assert fake_transcriber.call_count == 1
    assert len(manifest2.highlights) == len(manifest1.highlights)
