"""Tests for media probing and audio extraction."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from freecher_worker.media.audio import AudioExtractionError, extract_audio
from freecher_worker.media.probe import (
    MediaInfo,
    MediaProbeError,
    NoAudioStreamError,
    _parse_fps,
    probe_media,
)


def test_parse_fps():
    assert _parse_fps("30/1") == 30.0
    assert abs(_parse_fps("24000/1001") - 23.976) < 0.01
    assert _parse_fps("60") == 60.0
    assert _parse_fps("invalid") == 0.0
    assert _parse_fps("30/0") == 0.0


def test_media_info_model():
    info = MediaInfo(
        path="/path/to/video.mp4",
        duration_seconds=120.5,
        width=1920,
        height=1080,
        fps=30.0,
        video_codec="h264",
        audio_codec="aac",
        file_size=1048576,
        has_audio=True,
    )
    assert info.has_audio is True
    assert info.duration_seconds == 120.5
    assert info.width == 1920


def test_probe_nonexistent_file(tmp_path):
    missing_file = tmp_path / "nonexistent.mp4"
    with pytest.raises(FileNotFoundError):
        probe_media(missing_file)


@patch("subprocess.run")
def test_probe_media_valid(mock_run, tmp_path):
    fake_video = tmp_path / "sample.mp4"
    fake_video.touch()

    ffprobe_output = {
        "format": {
            "duration": "125.400",
            "size": "5242880",
        },
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
            },
        ],
    }

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps(ffprobe_output),
        stderr="",
    )

    info = probe_media(fake_video)
    assert info.duration_seconds == 125.4
    assert info.width == 1920
    assert info.height == 1080
    assert info.fps == 30.0
    assert info.video_codec == "h264"
    assert info.audio_codec == "aac"
    assert info.has_audio is True


@patch("subprocess.run")
def test_probe_media_missing_audio(mock_run, tmp_path):
    fake_video = tmp_path / "no_audio.mp4"
    fake_video.touch()

    ffprobe_output = {
        "format": {"duration": "60.0", "size": "1000000"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "r_frame_rate": "24/1",
            }
        ],
    }

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps(ffprobe_output),
        stderr="",
    )

    with pytest.raises(NoAudioStreamError):
        probe_media(fake_video)


@patch("subprocess.run")
def test_probe_media_ffprobe_error(mock_run, tmp_path):
    fake_video = tmp_path / "corrupt.mp4"
    fake_video.touch()

    mock_run.return_value = MagicMock(
        returncode=1,
        stdout="",
        stderr="Invalid data found when processing input",
    )

    with pytest.raises(MediaProbeError) as exc_info:
        probe_media(fake_video)
    assert "Invalid data found" in str(exc_info.value)


@patch("subprocess.run")
def test_extract_audio_failure(mock_run, tmp_path):
    src = tmp_path / "input.mp4"
    src.touch()
    dst = tmp_path / "audio.wav"

    mock_run.return_value = MagicMock(
        returncode=1,
        stdout="",
        stderr="Codec not supported",
    )

    with pytest.raises(AudioExtractionError) as exc_info:
        extract_audio(src, dst)
    assert "Codec not supported" in str(exc_info.value)
