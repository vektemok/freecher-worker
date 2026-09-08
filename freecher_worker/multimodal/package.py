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

from .audio_features import extract_candidate_audio_features
from .frames import DEFAULT_MAX_LONG_EDGE, extract_candidate_frames
from .models import MultimodalCandidatePackage, SourceAudioProfile
from .visual_features import extract_candidate_visual_features

logger = logging.getLogger("freecher_worker")


def compute_package_hash(
    candidate_set_id: str,
    source_fingerprint: str,
    candidate_id: str,
    start: float,
    end: float,
    max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
) -> str:
    """Deterministic hash of candidate extraction parameters."""
    key = f"{candidate_set_id}:{source_fingerprint}:{candidate_id}:{start:.3f}:{end:.3f}:{max_long_edge}"
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


def build_multimodal_package(
    candidate: CandidateWindow,
    transcript_doc: Optional[TranscriptDocument],
    source_video_path: Path | str,
    source_wav_path: Path | str,
    source_fingerprint: str,
    candidate_set_id: str,
    run_dir: Path | str,
    source_audio_profile: Optional[SourceAudioProfile] = None,
    force_rebuild: bool = False,
    max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
) -> MultimodalCandidatePackage:
    """Construct or retrieve a cached multimodal evidence package for a candidate.

    Guarantees strict isolation from human evaluation labels.
    """
    pkg_hash = compute_package_hash(
        candidate_set_id=candidate_set_id,
        source_fingerprint=source_fingerprint,
        candidate_id=candidate.id,
        start=candidate.start,
        end=candidate.end,
        max_long_edge=max_long_edge,
    )

    cache_dir = Path(run_dir) / "multimodal" / "cache" / candidate.id / pkg_hash
    package_json_path = cache_dir / "package.json"

    if package_json_path.is_file() and not force_rebuild:
        try:
            cached_pkg = MultimodalCandidatePackage.model_validate(load_json(package_json_path))
            # Verify frame files still exist on disk
            if all(Path(f.image_path).is_file() for f in cached_pkg.frames):
                logger.debug(f"[multimodal-package] Loaded cached package for {candidate.id} ({pkg_hash})")
                return cached_pkg
        except Exception as exc:
            logger.warning(f"[multimodal-package] Cached package invalid at {package_json_path}: {exc}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = cache_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract frames
    frames, decoder_used, req_count, fail_count = extract_candidate_frames(
        source_video_path=source_video_path,
        candidate=candidate,
        destination_dir=frames_dir,
        max_long_edge=max_long_edge,
    )

    # 2. Extract visual features
    frame_paths = [f.image_path for f in frames]
    visual_features = extract_candidate_visual_features(
        frame_paths=frame_paths,
        decoder_used=decoder_used,
        requested_count=req_count,
        failed_count=fail_count,
    )

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
    )

    save_json(package, package_json_path)
    logger.debug(f"[multimodal-package] Built and cached package for {candidate.id} at {package_json_path}")
    return package
