"""Candidate video frame extraction using reliable FFmpeg software decoding."""

from __future__ import annotations

import functools
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from freecher_worker.highlights.models import CandidateWindow
from freecher_worker.media.probe import probe_media
from .models import ExtractedFrame

logger = logging.getLogger("freecher_worker")

UNIFORM_FRACTIONS = (0.05, 0.18, 0.31, 0.44, 0.56, 0.69, 0.82, 0.95)
MAX_TOTAL_FRAMES = 12
DEFAULT_MAX_LONG_EDGE = 640


@functools.lru_cache(maxsize=1)
def get_available_ffmpeg_decoders() -> set[str]:
    """Inspect and cache available FFmpeg video decoders."""
    try:
        res = subprocess.run(
            ["ffmpeg", "-decoders"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10.0,
            check=False,
        )
        decoders = set()
        for line in res.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and ("V" in parts[0]):
                decoders.add(parts[1])
        return decoders
    except Exception as exc:
        logger.warning(f"[multimodal-frames] Could not inspect ffmpeg decoders: {exc}")
        return set()


def probe_software_decoder(video_codec: str) -> Tuple[Optional[str], str]:
    """Probe for the safest software decoder without hardware acceleration.

    Returns:
        (decoder_name_or_none, description)
    """
    codec_lower = video_codec.lower()
    avail = get_available_ffmpeg_decoders()

    if "av1" in codec_lower:
        if "libdav1d" in avail:
            return "libdav1d", "libdav1d (software AV1)"
        if "av1" in avail:
            return "av1", "av1 (reference software AV1)"
        return None, "ffmpeg_default_software"

    if "h264" in codec_lower or "avc" in codec_lower:
        if "h264" in avail:
            return "h264", "h264 (software)"
        return None, "ffmpeg_default_software"

    if "hevc" in codec_lower or "h265" in codec_lower:
        if "hevc" in avail:
            return "hevc", "hevc (software)"
        return None, "ffmpeg_default_software"

    return None, "ffmpeg_default_software"


def compute_frame_sample_timestamps(
    candidate_start: float,
    candidate_duration: float,
    uniform_fractions: Tuple[float, ...] = UNIFORM_FRACTIONS,
    scene_change_offsets: Optional[List[float]] = None,
    max_total_frames: int = MAX_TOTAL_FRAMES,
) -> List[Tuple[float, float, str]]:
    """Compute relative and absolute timestamps for frame extraction.

    Returns:
        List of (timestamp_offset, absolute_timestamp, source_type)
    """
    dur = max(0.5, candidate_duration)
    samples: List[Tuple[float, float, str]] = []

    # 1. Uniform samples
    for frac in uniform_fractions:
        offset = round(dur * frac, 3)
        abs_ts = round(candidate_start + offset, 3)
        samples.append((offset, abs_ts, "uniform"))

    # 2. Scene change samples (if provided, up to max_total_frames)
    if scene_change_offsets:
        seen_offsets = {s[0] for s in samples}
        for sc_off in scene_change_offsets:
            if len(samples) >= max_total_frames:
                break
            sc_off_rounded = round(sc_off, 3)
            # Avoid placing samples too close (< 0.5s) to existing uniform frames
            if not any(abs(sc_off_rounded - existing) < 0.5 for existing in seen_offsets):
                abs_ts = round(candidate_start + sc_off_rounded, 3)
                samples.append((sc_off_rounded, abs_ts, "scene_change"))
                seen_offsets.add(sc_off_rounded)

    # Sort deterministically by offset
    samples.sort(key=lambda s: s[0])
    return samples[:max_total_frames]


def extract_candidate_frames(
    source_video_path: Path | str,
    candidate: CandidateWindow,
    destination_dir: Path | str,
    uniform_fractions: Tuple[float, ...] = UNIFORM_FRACTIONS,
    scene_change_offsets: Optional[List[float]] = None,
    max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
    max_total_frames: int = MAX_TOTAL_FRAMES,
) -> Tuple[List[ExtractedFrame], str, int, int]:
    """Extract downscaled JPEG frames for a candidate using software FFmpeg decoding.

    Returns:
        (extracted_frames, decoder_used, requested_count, failed_count)
    """
    src = Path(source_video_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Source video not found: {src}")

    dst_dir = Path(destination_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Probe source codec
    media_info = probe_media(src)
    decoder_name, decoder_desc = probe_software_decoder(media_info.video_codec)

    sample_plan = compute_frame_sample_timestamps(
        candidate_start=candidate.start,
        candidate_duration=candidate.duration,
        uniform_fractions=uniform_fractions,
        scene_change_offsets=scene_change_offsets,
        max_total_frames=max_total_frames,
    )

    requested_count = len(sample_plan)
    extracted: List[ExtractedFrame] = []
    failed_count = 0

    scale_filter = (
        f"scale='if(gt(iw,ih),min({max_long_edge},iw),-2)':"
        f"'if(gt(ih,iw),min({max_long_edge},ih),-2)'"
    )

    for idx, (offset, abs_ts, stype) in enumerate(sample_plan, start=1):
        frame_filename = f"frame_{idx:02d}.jpg"
        frame_path = dst_dir / frame_filename

        # If frame already extracted and non-empty, reuse it
        if frame_path.is_file() and frame_path.stat().st_size > 0:
            extracted.append(
                ExtractedFrame(
                    timestamp_offset=offset,
                    absolute_timestamp=abs_ts,
                    image_path=str(frame_path),
                    width=max_long_edge,
                    height=max_long_edge,
                    source_type=stype,
                )
            )
            continue

        cmd = ["ffmpeg", "-nostdin", "-y"]
        if decoder_name:
            cmd.extend(["-c:v", decoder_name])

        # Accurate fast-seek
        cmd.extend([
            "-ss", f"{abs_ts:.3f}",
            "-i", str(src),
            "-frames:v", "1",
            "-vf", scale_filter,
            "-q:v", "3",
            str(frame_path),
        ])

        try:
            res = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20.0,
                check=False,
            )
            if res.returncode == 0 and frame_path.is_file() and frame_path.stat().st_size > 0:
                extracted.append(
                    ExtractedFrame(
                        timestamp_offset=offset,
                        absolute_timestamp=abs_ts,
                        image_path=str(frame_path),
                        width=max_long_edge,
                        height=max_long_edge,
                        source_type=stype,
                    )
                )
            else:
                failed_count += 1
                logger.warning(
                    f"[multimodal-frames] Failed extracting frame @ {abs_ts:.2f}s for {candidate.id}: {res.stderr.strip()[:100]}"
                )
        except Exception as exc:
            failed_count += 1
            logger.warning(
                f"[multimodal-frames] Error extracting frame @ {abs_ts:.2f}s for {candidate.id}: {exc}"
            )

    return extracted, decoder_desc, requested_count, failed_count
