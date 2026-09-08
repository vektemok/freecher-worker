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


from pydantic import BaseModel, Field


class SafeDecoderResolution(BaseModel):
    """Detailed resolution of video decoding strategy."""

    source_codec: str = Field(description="Raw source video codec reported by ffprobe")
    decoder_mode: str = Field(description="Resolved decode mode: 'libdav1d' or 'ffmpeg_auto'")
    requested_decoder: Optional[str] = Field(default=None, description="Explicit decoder requested if available")
    ffmpeg_decoder_arg: Optional[str] = Field(default=None, description="Exact argument to pass to '-c:v', or None for auto")
    hardware_acceleration: bool = Field(default=False, description="Hardware acceleration flag (strictly False)")
    smoke_test_passed: bool = Field(default=False, description="Whether runtime smoke test succeeded")

    def to_dict(self) -> dict:
        return {
            "source_codec": self.source_codec,
            "decoder_mode": self.decoder_mode,
            "requested_decoder": self.requested_decoder,
            "hardware_acceleration": self.hardware_acceleration,
            "smoke_test_passed": self.smoke_test_passed,
        }


DISALLOWED_HARDWARE_DECODERS = {
    "av1_cuvid", "av1_qsv", "av1_nvdec",
    "h264_cuvid", "h264_qsv", "h264_nvdec",
    "hevc_cuvid", "hevc_qsv", "hevc_nvdec",
}


@functools.lru_cache(maxsize=128)
def run_decoder_smoke_test(
    video_path_str: str,
    decoder_name: str,
    timeout_seconds: float = 6.0,
) -> bool:
    """Run a fast runtime smoke decode on a small sample of the source video.

    Attempts to decode 1 single frame to null muxer using the specified decoder.
    Returns True if decoding succeeds, False if it fails, hangs, or errors.
    """
    vpath = Path(video_path_str).resolve()
    if not vpath.is_file():
        return False

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel", "error",
        "-y",
        "-c:v", decoder_name,
        "-ss", "0",
        "-t", "2",
        "-i", str(vpath),
        "-frames:v", "1",
        "-f", "null",
        "-",
    ]
    try:
        res = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
        if res.returncode == 0:
            return True
        logger.warning(
            f"[multimodal-decoder] Smoke decode with '-c:v {decoder_name}' returned code {res.returncode}: "
            f"{res.stderr.decode('utf-8', errors='ignore') if isinstance(res.stderr, bytes) else res.stderr}"
        )
        return False
    except (subprocess.TimeoutExpired, Exception) as exc:
        logger.warning(f"[multimodal-decoder] Smoke decode with '-c:v {decoder_name}' failed/timed out: {exc}")
        return False


def resolve_safe_video_decoder(
    video_codec: str,
    source_video_path: Optional[Path | str] = None,
) -> SafeDecoderResolution:
    """Resolve the safest software decoder without hardware acceleration.

    Rules:
    1. NEVER select native 'av1' as explicit decoder (-c:v av1 is broken on many platforms).
    2. NEVER select hardware decoders (cuvid, qsv, nvdec).
    3. For AV1: prefer 'libdav1d' ONLY if installed and (when source path is given) passes smoke test.
    4. Otherwise use 'ffmpeg_auto' (omit -c:v completely), which allows FFmpeg's native
       automatic software decoder selection to succeed.
    """
    codec_raw = video_codec or "unknown"
    codec_lower = codec_raw.lower()
    avail = get_available_ffmpeg_decoders()

    if "av1" in codec_lower:
        # Check if libdav1d is installed
        if "libdav1d" in avail:
            if source_video_path is not None and Path(source_video_path).is_file():
                smoke_ok = run_decoder_smoke_test(str(source_video_path), "libdav1d")
                if smoke_ok:
                    return SafeDecoderResolution(
                        source_codec=codec_raw,
                        decoder_mode="libdav1d",
                        requested_decoder="libdav1d",
                        ffmpeg_decoder_arg="libdav1d",
                        hardware_acceleration=False,
                        smoke_test_passed=True,
                    )
                else:
                    logger.warning(
                        "[multimodal-decoder] 'libdav1d' smoke test failed on source video. "
                        "Falling back to 'ffmpeg_auto' (omitting -c:v)."
                    )
                    return SafeDecoderResolution(
                        source_codec=codec_raw,
                        decoder_mode="ffmpeg_auto",
                        requested_decoder="libdav1d",
                        ffmpeg_decoder_arg=None,
                        hardware_acceleration=False,
                        smoke_test_passed=False,
                    )
            else:
                # No video file provided (e.g. static inspection)
                return SafeDecoderResolution(
                    source_codec=codec_raw,
                    decoder_mode="libdav1d",
                    requested_decoder="libdav1d",
                    ffmpeg_decoder_arg="libdav1d",
                    hardware_acceleration=False,
                    smoke_test_passed=False,
                )

        # libdav1d not installed: NEVER fall back to "-c:v av1"!
        logger.info("[multimodal-decoder] 'libdav1d' not found. Using 'ffmpeg_auto' for AV1.")
        return SafeDecoderResolution(
            source_codec=codec_raw,
            decoder_mode="ffmpeg_auto",
            requested_decoder=None,
            ffmpeg_decoder_arg=None,
            hardware_acceleration=False,
            smoke_test_passed=False,
        )

    # For other standard codecs (e.g. h264, hevc)
    if "h264" in codec_lower or "avc" in codec_lower:
        if "h264" in avail:
            return SafeDecoderResolution(
                source_codec=codec_raw,
                decoder_mode="h264",
                requested_decoder="h264",
                ffmpeg_decoder_arg="h264",
                hardware_acceleration=False,
                smoke_test_passed=True,
            )

    if "hevc" in codec_lower or "h265" in codec_lower:
        if "hevc" in avail:
            return SafeDecoderResolution(
                source_codec=codec_raw,
                decoder_mode="hevc",
                requested_decoder="hevc",
                ffmpeg_decoder_arg="hevc",
                hardware_acceleration=False,
                smoke_test_passed=True,
            )

    return SafeDecoderResolution(
        source_codec=codec_raw,
        decoder_mode="ffmpeg_auto",
        requested_decoder=None,
        ffmpeg_decoder_arg=None,
        hardware_acceleration=False,
        smoke_test_passed=False,
    )


def probe_software_decoder(
    video_codec: str,
    source_video_path: Optional[Path | str] = None,
) -> Tuple[Optional[str], str]:
    """Probe for the safest software decoder without hardware acceleration.

    Returns:
        (ffmpeg_decoder_arg_or_none, description)
    """
    res = resolve_safe_video_decoder(video_codec, source_video_path=source_video_path)
    if res.decoder_mode == "libdav1d":
        desc = "libdav1d (software AV1)"
    elif res.decoder_mode == "ffmpeg_auto":
        desc = "ffmpeg_auto (automatic software decode)"
    else:
        desc = f"{res.decoder_mode} (software)"
    return res.ffmpeg_decoder_arg, desc


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


UNIFORM_FRACTIONS_V1_1 = (0.10, 0.35, 0.65, 0.90)


def compute_v1_1_sample_timestamps(
    candidate_start: float,
    candidate_duration: float,
    burst_1_center: float,
    burst_2_center: float,
    max_total_frames: int = MAX_TOTAL_FRAMES,
) -> List[Tuple[float, float, str]]:
    """Compute relative and absolute timestamps for v1.1 hybrid sampling.

    Structure:
    - 4 global uniform frames (10%, 35%, 65%, 90%)
    - 4 frames for burst 1: [t1 - 0.75, t1 - 0.25, t1 + 0.25, t1 + 0.75]
    - 4 frames for burst 2: [t2 - 0.75, t2 - 0.25, t2 + 0.25, t2 + 0.75]
    Total <= 12 frames.
    """
    import numpy as np

    dur = max(0.5, candidate_duration)
    samples: List[Tuple[float, float, str]] = []
    seen_offsets: List[float] = []

    def _add_sample(off: float, stype: str):
        clamped_off = round(float(np.clip(off, 0.05, max(0.05, dur - 0.05))), 3)
        if any(abs(clamped_off - existing) < 0.1 for existing in seen_offsets):
            return
        if len(samples) >= max_total_frames:
            return
        abs_ts = round(candidate_start + clamped_off, 3)
        samples.append((clamped_off, abs_ts, stype))
        seen_offsets.append(clamped_off)

    # 1. 4 Global frames
    for frac in UNIFORM_FRACTIONS_V1_1:
        _add_sample(round(dur * frac, 3), "global")

    # 2. Burst 1 frames (4 frames around t1)
    burst_offsets = (-0.75, -0.25, 0.25, 0.75)
    for b_off in burst_offsets:
        _add_sample(burst_1_center + b_off, "burst_1")

    # 3. Burst 2 frames (4 frames around t2)
    for b_off in burst_offsets:
        _add_sample(burst_2_center + b_off, "burst_2")

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
    sample_plan: Optional[List[Tuple[float, float, str]]] = None,
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

    if sample_plan is None:
        sample_plan = compute_frame_sample_timestamps(
            candidate_start=candidate.start,
            candidate_duration=candidate.duration,
            uniform_fractions=uniform_fractions,
            scene_change_offsets=scene_change_offsets,
            max_total_frames=max_total_frames,
        )

    requested_count = len(sample_plan)

    # Fast-path: if all requested frames already exist on disk, return without invoking FFmpeg
    all_cached = True
    cached_frames: List[ExtractedFrame] = []
    for idx, (offset, abs_ts, stype) in enumerate(sample_plan, start=1):
        frame_filename = f"frame_{idx:02d}.jpg"
        frame_path = dst_dir / frame_filename
        if frame_path.is_file() and frame_path.stat().st_size > 0:
            cached_frames.append(
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
            all_cached = False
            break

    if all_cached and len(cached_frames) == requested_count:
        media_info = probe_media(src)
        dec_res = resolve_safe_video_decoder(media_info.video_codec, source_video_path=None)
        return cached_frames, dec_res.ffmpeg_decoder_arg or "ffmpeg_auto", requested_count, 0

    # Probe source codec and resolve safe decoder
    media_info = probe_media(src)
    dec_res = resolve_safe_video_decoder(media_info.video_codec, source_video_path=src)
    decoder_name = dec_res.ffmpeg_decoder_arg
    decoder_mode = dec_res.decoder_mode
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
        if decoder_name and decoder_name not in ("ffmpeg_auto", "auto", "av1", "none"):
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

    return extracted, decoder_mode, requested_count, failed_count
