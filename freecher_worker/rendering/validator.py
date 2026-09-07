"""Quality validation for rendered short-form vertical videos."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("freecher_worker")


class RenderValidationError(Exception):
    """Raised when a rendered short-form video fails quality validation."""
    pass


class VideoValidationResult(BaseModel):
    """Validation report for a rendered video file."""

    valid: bool = Field(description="True if video passes all quality checks")
    path: str = Field(description="Path to validated video file")
    file_size_bytes: int = Field(default=0, description="Size in bytes")
    width: int = Field(default=0, description="Video width in pixels")
    height: int = Field(default=0, description="Video height in pixels")
    duration: float = Field(default=0.0, description="Duration in seconds")
    video_codec: str = Field(default="", description="Video codec name")
    has_audio: bool = Field(default=False, description="Audio stream presence")
    audio_codec: str = Field(default="", description="Audio codec name")
    error_message: Optional[str] = Field(default=None, description="Explanation if invalid")

    @property
    def passed(self) -> bool:
        """Alias for valid."""
        return self.valid


def validate_rendered_video(
    video_path: Path,
    expected_width: int = 1080,
    expected_height: int = 1920,
    expected_duration: Optional[float] = None,
    duration_tolerance: float = 0.60,
    strict: bool = True,
) -> VideoValidationResult:
    """Validate that rendered MP4 satisfies resolution, audio, duration, and codec constraints."""
    p = Path(video_path)
    if not p.is_file():
        msg = f"Rendered video file not found: {p}"
        if strict:
            raise RenderValidationError(msg)
        return VideoValidationResult(valid=False, path=str(p), error_message=msg)

    size = p.stat().st_size
    if size == 0:
        msg = f"Rendered video file is empty (0 bytes): {p}"
        if strict:
            raise RenderValidationError(msg)
        return VideoValidationResult(valid=False, path=str(p), file_size_bytes=0, error_message=msg)

    # Inspect streams via ffprobe
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(p),
    ]

    try:
        res = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10.0,
            check=True,
        )
        data = json.loads(res.stdout)
    except subprocess.CalledProcessError as exc:
        msg = f"ffprobe failed to inspect {p}: {exc.stderr.strip() or exc}"
        if strict:
            raise RenderValidationError(msg)
        return VideoValidationResult(valid=False, path=str(p), file_size_bytes=size, error_message=msg)
    except Exception as exc:
        msg = f"ffprobe failed to inspect {p}: {exc}"
        if strict:
            raise RenderValidationError(msg)
        return VideoValidationResult(valid=False, path=str(p), file_size_bytes=size, error_message=msg)

    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if not video_stream:
        msg = f"No video stream found in {p}"
        if strict:
            raise RenderValidationError(msg)
        return VideoValidationResult(valid=False, path=str(p), file_size_bytes=size, error_message=msg)

    w = int(video_stream.get("width", 0))
    h = int(video_stream.get("height", 0))
    vcodec = video_stream.get("codec_name", "")

    format_info = data.get("format", {})
    actual_duration = float(format_info.get("duration", 0.0))

    if w != expected_width or h != expected_height:
        msg = f"Resolution mismatch: expected {expected_width}x{expected_height}, got {w}x{h}"
        if strict:
            raise RenderValidationError(msg)
        return VideoValidationResult(
            valid=False,
            path=str(p),
            file_size_bytes=size,
            width=w,
            height=h,
            duration=actual_duration,
            video_codec=vcodec,
            has_audio=audio_stream is not None,
            error_message=msg,
        )

    if not audio_stream:
        msg = f"No audio stream found in {p}"
        if strict:
            raise RenderValidationError(msg)
        return VideoValidationResult(
            valid=False,
            path=str(p),
            file_size_bytes=size,
            width=w,
            height=h,
            duration=actual_duration,
            video_codec=vcodec,
            has_audio=False,
            error_message=msg,
        )

    acodec = audio_stream.get("codec_name", "")

    if expected_duration is not None:
        diff = abs(actual_duration - expected_duration)
        if diff > duration_tolerance:
            msg = f"Duration mismatch: expected {expected_duration:.2f}s, got {actual_duration:.2f}s (diff {diff:.2f}s > {duration_tolerance}s)"
            if strict:
                raise RenderValidationError(msg)
            return VideoValidationResult(
                valid=False,
                path=str(p),
                file_size_bytes=size,
                width=w,
                height=h,
                duration=actual_duration,
                video_codec=vcodec,
                has_audio=True,
                audio_codec=acodec,
                error_message=msg,
            )

    return VideoValidationResult(
        valid=True,
        path=str(p),
        file_size_bytes=size,
        width=w,
        height=h,
        duration=actual_duration,
        video_codec=vcodec,
        has_audio=True,
        audio_codec=acodec,
    )
