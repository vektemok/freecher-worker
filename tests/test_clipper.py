"""Tests for video clipper."""

from pathlib import Path
import pytest
from freecher_worker.media.clipper import VideoClippingError, clip_video, is_nvenc_available


def test_is_nvenc_available_returns_bool():
    res = is_nvenc_available()
    assert isinstance(res, bool)


def test_clip_video_missing_source(tmp_path):
    missing_src = tmp_path / "missing.mp4"
    dst = tmp_path / "out.mp4"
    with pytest.raises(FileNotFoundError):
        clip_video(missing_src, dst, start_seconds=10.0, end_seconds=20.0)


def test_clip_video_invalid_duration(tmp_path):
    src = tmp_path / "dummy.mp4"
    src.touch()
    dst = tmp_path / "out.mp4"
    with pytest.raises(VideoClippingError) as exc_info:
        clip_video(src, dst, start_seconds=50.0, end_seconds=40.0)
    assert "duration <= 0" in str(exc_info.value)
