"""Tests for --analysis-only mode."""

from pathlib import Path
from unittest.mock import patch

from freecher_worker.config import Settings
from freecher_worker.media.probe import MediaInfo
from freecher_worker.pipeline.processor import run_pipeline
from freecher_worker.transcription.models import Transcript, TranscriptSegment
from freecher_worker.transcription.whisper import BaseTranscriber


class SimpleTranscriber(BaseTranscriber):
    def transcribe(self, audio_path, language=None, source_fingerprint_id=None):
        return Transcript(
            source_fingerprint_id=source_fingerprint_id,
            language="ru",
            language_probability=0.99,
            duration=60.0,
            model="small",
            compute_type="int8",
            device="cpu",
            segments=[
                TranscriptSegment(id=0, start=0.0, end=25.0, text="Первый блок для анализа."),
                TranscriptSegment(id=1, start=26.0, end=55.0, text="Второй важный фрагмент разговора!"),
            ],
        )


@patch("freecher_worker.pipeline.processor.clip_video")
@patch("freecher_worker.pipeline.processor.extract_audio")
@patch("freecher_worker.pipeline.processor.probe_media")
def test_analysis_only_mode(mock_probe, mock_extract, mock_clip, tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"analysis_only_dummy_bytes")

    mock_probe.return_value = MediaInfo(
        path=str(video),
        duration_seconds=60.0,
        width=1280,
        height=720,
        fps=25.0,
        video_codec="h264",
        audio_codec="aac",
        file_size=len(b"analysis_only_dummy_bytes"),
        has_audio=True,
    )
    mock_extract.side_effect = lambda src, dst: Path(dst).touch() or Path(dst)

    cfg = Settings(output_dir=tmp_path / "runs")
    manifest = run_pipeline(
        video,
        config=cfg,
        run_id="analysis_run",
        analysis_only=True,
        transcriber=SimpleTranscriber(),
    )

    # clip_video should NOT have been called
    assert mock_clip.call_count == 0
    assert manifest.timings.clipping_seconds == 0.0

    run_dir = tmp_path / "runs" / "analysis_run"
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "transcript.json").is_file()
    assert (run_dir / "candidates.json").is_file()
    assert (run_dir / "highlights.json").is_file()

    # Verify no clips in clips directory
    clips_dir = run_dir / "clips"
    clip_files = list(clips_dir.glob("*.mp4"))
    assert len(clip_files) == 0

    # Highlights file property should be None
    for hl in manifest.highlights:
        assert hl.file is None
