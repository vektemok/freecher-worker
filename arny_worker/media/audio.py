"""Audio extraction module using ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path


class AudioExtractionError(Exception):
    """Raised when audio extraction from media fails."""
    pass


def extract_audio(video_path: Path | str, output_audio_path: Path | str) -> Path:
    """Extract 16000 Hz mono PCM WAV audio from video using FFmpeg.

    Args:
        video_path: Path to the input video file.
        output_audio_path: Destination path for the WAV audio file.

    Returns:
        Path to the extracted audio file.

    Raises:
        FileNotFoundError: If input video doesn't exist.
        AudioExtractionError: If ffmpeg fails.
    """
    src = Path(video_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Input video file not found: {src}")

    dst = Path(output_audio_path).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",               # overwrite without asking
        "-i", str(src),     # input
        "-vn",              # disable video recording
        "-ac", "1",         # mono channel
        "-ar", "16000",     # 16 kHz sample rate
        "-c:a", "pcm_s16le",# 16-bit PCM WAV
        str(dst),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600.0,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioExtractionError(f"ffmpeg audio extraction timed out: {exc}") from exc
    except FileNotFoundError as exc:
        raise AudioExtractionError(
            "ffmpeg executable not found. Ensure ffmpeg is installed and available in PATH."
        ) from exc
    except Exception as exc:
        raise AudioExtractionError(f"Unexpected error running ffmpeg audio extraction: {exc}") from exc

    if result.returncode != 0:
        stderr_msg = result.stderr.strip() or "Unknown ffmpeg error"
        raise AudioExtractionError(
            f"ffmpeg audio extraction failed with exit code {result.returncode}. Details: {stderr_msg}"
        )

    if not dst.is_file() or dst.stat().st_size == 0:
        raise AudioExtractionError(
            f"ffmpeg reported success, but extracted audio file {dst} is missing or empty."
        )

    return dst
