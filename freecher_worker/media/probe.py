"""ffprobe wrapper for extracting media information."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field


class MediaProbeError(Exception):
    """Raised when ffprobe execution fails or output is invalid."""
    pass


class NoAudioStreamError(MediaProbeError):
    """Raised when a media file contains no audio stream."""
    pass


class MediaInfo(BaseModel):
    """Metadata extracted from media container and streams."""

    path: str
    duration_seconds: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: Optional[str] = None
    file_size: int
    has_audio: bool = Field(default=False)


def _parse_fps(rate_str: str) -> float:
    """Parse frame rate string like '30/1' or '24000/1001' into float."""
    try:
        if "/" in rate_str:
            num, den = rate_str.split("/", 1)
            denominator = float(den)
            return float(num) / denominator if denominator != 0 else 0.0
        return float(rate_str)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0


def probe_media(video_path: Path | str) -> MediaInfo:
    """Probe video file using ffprobe and return validated MediaInfo.

    Raises:
        FileNotFoundError: If the input video does not exist.
        NoAudioStreamError: If the video has no audio track.
        MediaProbeError: If ffprobe fails or output cannot be parsed.
    """
    path = Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30.0,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaProbeError(f"ffprobe timed out: {exc}") from exc
    except FileNotFoundError as exc:
        raise MediaProbeError(
            "ffprobe executable not found. Please make sure ffmpeg/ffprobe is installed and available in PATH."
        ) from exc
    except Exception as exc:
        raise MediaProbeError(f"Unexpected error running ffprobe: {exc}") from exc

    if result.returncode != 0:
        stderr_msg = result.stderr.strip() or "Unknown error"
        raise MediaProbeError(
            f"ffprobe failed for {path} with exit code {result.returncode}. Error details: {stderr_msg}"
        )

    try:
        data: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError(f"Failed to decode ffprobe JSON output: {exc}\nRaw output: {result.stdout}") from exc

    format_info = data.get("format", {})
    streams = data.get("streams", [])

    video_stream: Optional[dict[str, Any]] = None
    audio_stream: Optional[dict[str, Any]] = None

    for s in streams:
        codec_type = s.get("codec_type")
        if codec_type == "video" and video_stream is None:
            video_stream = s
        elif codec_type == "audio" and audio_stream is None:
            audio_stream = s

    if not video_stream:
        raise MediaProbeError(f"No video stream found in {path}")

    # Check for audio track
    has_audio = audio_stream is not None
    if not has_audio:
        raise NoAudioStreamError(
            f"Input media {path} does not contain any audio stream. An audio track is required for ASR transcription."
        )

    # Duration: prefer format.duration, fallback to video/audio stream duration
    duration = 0.0
    if "duration" in format_info:
        try:
            duration = float(format_info["duration"])
        except (ValueError, TypeError):
            pass

    if duration <= 0.0 and video_stream and "duration" in video_stream:
        try:
            duration = float(video_stream["duration"])
        except (ValueError, TypeError):
            pass

    if duration <= 0.0 and audio_stream and "duration" in audio_stream:
        try:
            duration = float(audio_stream["duration"])
        except (ValueError, TypeError):
            pass

    if duration <= 0.0:
        raise MediaProbeError(f"Could not determine valid duration for {path}")

    # Resolution
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    video_codec = str(video_stream.get("codec_name", "unknown"))

    # FPS
    fps_str = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0"
    fps = _parse_fps(fps_str)
    if fps <= 0.0 and "r_frame_rate" in video_stream:
        fps = _parse_fps(video_stream["r_frame_rate"])

    # Audio codec
    audio_codec = str(audio_stream.get("codec_name", "unknown")) if audio_stream else None

    # File size
    file_size = 0
    if "size" in format_info:
        try:
            file_size = int(format_info["size"])
        except (ValueError, TypeError):
            pass
    if file_size <= 0:
        file_size = path.stat().st_size

    return MediaInfo(
        path=str(path),
        duration_seconds=round(duration, 3),
        width=width,
        height=height,
        fps=round(fps, 3),
        video_codec=video_codec,
        audio_codec=audio_codec,
        file_size=file_size,
        has_audio=has_audio,
    )
