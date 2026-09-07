"""FFmpeg video clipping module with hardware acceleration detection and fallback."""

from __future__ import annotations

import functools
import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger("arny_worker")


class VideoClippingError(Exception):
    """Raised when video clipping fails."""
    pass


@functools.lru_cache(maxsize=1)
def is_nvenc_available() -> bool:
    """Check if ffmpeg supports h264_nvenc hardware acceleration and an NVIDIA GPU is functional."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v", "error",
        "-f", "lavfi",
        "-i", "color=c=black:s=64x64:d=0.1",
        "-c:v", "h264_nvenc",
        "-f", "null",
        "-",
    ]
    try:
        res = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


def clip_video(
    source_video: Path | str,
    output_clip: Path | str,
    start_seconds: float,
    end_seconds: float,
    use_nvenc: Optional[bool] = None,
) -> Path:
    """Clip video segment with accurate timestamps and re-encode to H.264 / AAC.

    Args:
        source_video: Source video file path.
        output_clip: Destination path for clipped mp4.
        start_seconds: Start timestamp in seconds.
        end_seconds: End timestamp in seconds.
        use_nvenc: Explicitly enable/disable h264_nvenc. If None, auto-detected.

    Returns:
        Path to the written clip file.

    Raises:
        FileNotFoundError: If source video does not exist.
        VideoClippingError: If FFmpeg clipping fails.
    """
    src = Path(source_video).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Source video file not found: {src}")

    dst = Path(output_clip).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if start_seconds < 0:
        start_seconds = 0.0
    duration = end_seconds - start_seconds
    if duration <= 0:
        raise VideoClippingError(
            f"Invalid clipping range: start={start_seconds:.2f}s, end={end_seconds:.2f}s (duration <= 0)"
        )

    nvenc_candidate = use_nvenc if use_nvenc is not None else is_nvenc_available()

    # Try encoding with nvenc if available, fallback to libx264
    encoders = []
    if nvenc_candidate:
        encoders.append(("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "24"]))
    encoders.append(("libx264", ["-c:v", "libx264", "-preset", "fast", "-crf", "22"]))

    last_error = ""
    for encoder_name, video_args in encoders:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-ss", f"{start_seconds:.3f}",
            "-i", str(src),
            "-t", f"{duration:.3f}",
            *video_args,
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(dst),
        ]

        try:
            res = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=600.0,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise VideoClippingError(f"ffmpeg clipping timed out: {exc}") from exc
        except FileNotFoundError as exc:
            raise VideoClippingError(
                "ffmpeg executable not found. Ensure ffmpeg is installed and available in PATH."
            ) from exc
        except Exception as exc:
            raise VideoClippingError(f"Unexpected error running ffmpeg clipping: {exc}") from exc

        if res.returncode == 0 and dst.is_file() and dst.stat().st_size > 0:
            return dst

        last_error = res.stderr.strip() or f"Exit code {res.returncode}"
        logger.warning(f"[clipping] Encoder {encoder_name} failed: {last_error}. Attempting fallback...")

    raise VideoClippingError(
        f"Failed to clip video from {start_seconds:.2f}s to {end_seconds:.2f}s into {dst}. Error: {last_error}"
    )
