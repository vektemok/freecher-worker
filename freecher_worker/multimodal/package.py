"""Multimodal candidate package builder and disk cache."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

from freecher_worker.highlights.models import CandidateWindow
from freecher_worker.transcription.models import Transcript

TranscriptDocument = Transcript
from freecher_worker.utils.json_io import load_json, save_json

from .activity import (
    select_temporal_burst_peaks,
    slice_candidate_activity_curve,
)
from .audio_features import extract_candidate_audio_features
from .frames import (
    DEFAULT_MAX_LONG_EDGE,
    compute_v1_1_sample_timestamps,
    extract_candidate_frames,
)
from .models import (
    ActivityCurveSummary,
    MultimodalCandidatePackage,
    SourceAudioProfile,
    SourceTemporalActivityProfile,
    TemporalBurst,
)
from .visual_features import extract_candidate_visual_features

logger = logging.getLogger("freecher_worker")

PACKAGE_VERSION_V1 = "multimodal_package_v1"
PACKAGE_VERSION_V1_1 = "multimodal_package_v1_1"


def compute_package_hash(
    candidate_set_id: str,
    source_fingerprint: str,
    candidate_id: str,
    start: float,
    end: float,
    max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
    package_version: str = PACKAGE_VERSION_V1,
) -> str:
    """Deterministic hash of candidate extraction parameters, isolated by package version."""
    key = f"{package_version}:{candidate_set_id}:{source_fingerprint}:{candidate_id}:{start:.3f}:{end:.3f}:{max_long_edge}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def extract_candidate_transcript_context(
    candidate: CandidateWindow,
    transcript_doc: Optional[TranscriptDocument],
    context_window_seconds: float = 45.0,
) -> tuple[str, str, str]:
    """Extract candidate speech and surrounding context (preceding and following 45s).

    Returns:
        (candidate_transcript, previous_context, next_context)
    """
    cand_text = candidate.text.strip()
    if transcript_doc is None or not transcript_doc.segments:
        return cand_text, "", ""

    cand_start = candidate.start
    cand_end = candidate.end

    prev_texts: list[str] = []
    next_texts: list[str] = []

    for seg in transcript_doc.segments:
        if seg.end <= cand_start:
            if cand_start - seg.start <= context_window_seconds:
                prev_texts.append(seg.text.strip())
        elif seg.start >= cand_end:
            if seg.end - cand_end <= context_window_seconds:
                next_texts.append(seg.text.strip())

    prev_ctx = " ".join(prev_texts).strip()
    next_ctx = " ".join(next_texts).strip()
    return cand_text, prev_ctx, next_ctx


def extract_temporal_burst_transcript(
    candidate_start: float,
    burst: TemporalBurst,
    transcript_doc: Optional[TranscriptDocument],
    padding: float = 0.75,
) -> str:
    """Extract speech segments overlapping [burst_start - 0.75s, burst_end + 0.75s]."""
    if transcript_doc is None or not transcript_doc.segments:
        return ""

    abs_burst_start = max(0.0, candidate_start + burst.start_offset - padding)
    abs_burst_end = candidate_start + burst.end_offset + padding

    burst_texts: list[str] = []
    for seg in transcript_doc.segments:
        # Check segment overlaps with [abs_burst_start, abs_burst_end]
        if seg.end >= abs_burst_start and seg.start <= abs_burst_end:
            txt = seg.text.strip()
            if txt and txt not in burst_texts:
                burst_texts.append(txt)

    return " ".join(burst_texts).strip()


def build_multimodal_package(
    candidate: CandidateWindow,
    transcript_doc: Optional[TranscriptDocument],
    source_video_path: Path | str,
    source_wav_path: Path | str,
    source_fingerprint: str,
    candidate_set_id: str,
    run_dir: Path | str,
    source_audio_profile: Optional[SourceAudioProfile] = None,
    source_activity_profile: Optional[SourceTemporalActivityProfile] = None,
    package_version: str = PACKAGE_VERSION_V1,
    force_rebuild: bool = False,
    max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
) -> MultimodalCandidatePackage:
    """Construct or retrieve a cached multimodal evidence package for a candidate.

    Guarantees strict isolation from human evaluation labels.
    """
    is_v1_1 = package_version == PACKAGE_VERSION_V1_1

    pkg_hash = compute_package_hash(
        candidate_set_id=candidate_set_id,
        source_fingerprint=source_fingerprint,
        candidate_id=candidate.id,
        start=candidate.start,
        end=candidate.end,
        max_long_edge=max_long_edge,
        package_version=package_version,
    )

    cache_dir = Path(run_dir) / "multimodal" / "cache" / candidate.id / pkg_hash
    package_json_path = cache_dir / "package.json"

    if package_json_path.is_file() and not force_rebuild:
        try:
            cached_pkg = MultimodalCandidatePackage.model_validate(load_json(package_json_path))
            if all(Path(f.image_path).is_file() for f in cached_pkg.frames):
                logger.debug(f"[multimodal-package] Loaded cached package for {candidate.id} ({pkg_hash})")
                return cached_pkg
        except Exception as exc:
            logger.warning(f"[multimodal-package] Cached package invalid at {package_json_path}: {exc}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = cache_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 1. Temporal bursts and activity curve for v1.1
    bursts: Optional[list[TemporalBurst]] = None
    activity_summary: Optional[ActivityCurveSummary] = None

    if is_v1_1:
        if source_activity_profile is not None:
            activity_summary = slice_candidate_activity_curve(
                source_profile=source_activity_profile,
                candidate_start=candidate.start,
                candidate_duration=candidate.duration,
            )
            bursts = select_temporal_burst_peaks(
                activity_summary=activity_summary,
                candidate_duration=candidate.duration,
            )
        else:
            burst_1_c = round(candidate.duration * 0.35, 2)
            burst_2_c = round(candidate.duration * 0.70, 2)
            bursts = [
                TemporalBurst(
                    burst_index=1,
                    center_offset=burst_1_c,
                    start_offset=max(0.0, burst_1_c - 1.0),
                    end_offset=min(candidate.duration, burst_1_c + 1.0),
                    selection_reason="uniform_fallback",
                    combined_activity=0.5,
                    activity_rank=1,
                ),
                TemporalBurst(
                    burst_index=2,
                    center_offset=burst_2_c,
                    start_offset=max(0.0, burst_2_c - 1.0),
                    end_offset=min(candidate.duration, burst_2_c + 1.0),
                    selection_reason="uniform_fallback",
                    combined_activity=0.5,
                    activity_rank=2,
                ),
            ]

        # Slices speech text overlapping [burst_start - 0.75s, burst_end + 0.75s]
        for b in bursts:
            b.transcript = extract_temporal_burst_transcript(
                candidate_start=candidate.start,
                burst=b,
                transcript_doc=transcript_doc,
                padding=0.75,
            )

        # Build v1.1 hybrid sample plan (4 global + 2 bursts x 4 frames)
        sample_plan = compute_v1_1_sample_timestamps(
            candidate_start=candidate.start,
            candidate_duration=candidate.duration,
            burst_1_center=bursts[0].center_offset,
            burst_2_center=bursts[1].center_offset,
        )
        frames, decoder_used, req_count, fail_count = extract_candidate_frames(
            source_video_path=source_video_path,
            candidate=candidate,
            destination_dir=frames_dir,
            max_long_edge=max_long_edge,
            sample_plan=sample_plan,
        )

        # Distribute extracted frames into respective bursts
        bursts[0].frames = [f for f in frames if f.source_type == "burst_1"]
        bursts[1].frames = [f for f in frames if f.source_type == "burst_2"]

    else:
        # Standard v1 sparse extraction
        frames, decoder_used, req_count, fail_count = extract_candidate_frames(
            source_video_path=source_video_path,
            candidate=candidate,
            destination_dir=frames_dir,
            max_long_edge=max_long_edge,
        )

    # 2. Extract visual features
    frame_paths = [f.image_path for f in frames]
    from freecher_worker.media.probe import probe_media
    try:
        media_info = probe_media(source_video_path)
        src_codec = getattr(media_info, "video_codec", "unknown")
    except Exception:
        src_codec = "unknown"

    dec_mode = decoder_used if decoder_used in ("libdav1d", "ffmpeg_auto") else "ffmpeg_auto"
    req_dec = "libdav1d" if "av1" in src_codec.lower() else (decoder_used if decoder_used != "ffmpeg_auto" else None)

    visual_features = extract_candidate_visual_features(
        frame_paths=frame_paths,
        decoder_used=decoder_used,
        requested_count=req_count,
        failed_count=fail_count,
        decoder_mode=dec_mode,
        requested_decoder=req_dec,
        hardware_acceleration=False,
    )

    decoder_info = {
        "source_codec": src_codec,
        "decoder_mode": dec_mode,
        "requested_decoder": req_dec,
        "hardware_acceleration": False,
    }

    # 3. Extract audio features
    audio_features = extract_candidate_audio_features(
        wav_path=source_wav_path,
        start=candidate.start,
        end=candidate.end,
        source_profile=source_audio_profile,
    )

    # 4. Extract transcript & context
    cand_transcript, prev_context, next_context = extract_candidate_transcript_context(
        candidate=candidate,
        transcript_doc=transcript_doc,
    )

    # 5. Visual sufficiency flag (less than 4 frames decoded)
    insufficient_visual = len(frames) < 4

    package = MultimodalCandidatePackage(
        candidate_id=candidate.id,
        start=candidate.start,
        end=candidate.end,
        duration=candidate.duration,
        candidate_transcript=cand_transcript,
        previous_context=prev_context,
        next_context=next_context,
        frames=frames,
        audio_features=audio_features,
        visual_features=visual_features,
        source_fingerprint=source_fingerprint,
        candidate_set_id=candidate_set_id,
        package_hash=pkg_hash,
        insufficient_visual_evidence=insufficient_visual,
        package_version=package_version,
        temporal_bursts=bursts,
        activity_curve=activity_summary,
        decoder_info=decoder_info,
    )

    save_json(package, package_json_path)
    logger.debug(f"[multimodal-package] Built and cached package ({package_version}) for {candidate.id} at {package_json_path}")
    return package
