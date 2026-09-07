"""End-to-end processing pipeline orchestrator with configuration-aware caching and observability."""

from __future__ import annotations

import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from freecher_worker.config import Settings, get_settings
from freecher_worker.evaluation.models import ScorerPredictionDocument, ScorerPredictionItem
from freecher_worker.highlights.dedup import calculate_iou, calculate_overlap_ratio
from freecher_worker.highlights.models import (
    CandidateDocument,
    CandidateWindow,
    Highlight,
    HighlightScore,
    compute_candidate_set_id,
)
from freecher_worker.highlights.ranker import rank_and_deduplicate
from freecher_worker.highlights.segmenter import generate_candidate_windows
from freecher_worker.media.audio import extract_audio
from freecher_worker.media.clipper import clip_video
from freecher_worker.media.fingerprint import SourceFingerprint, compute_source_fingerprint
from freecher_worker.media.probe import MediaInfo, probe_media
from freecher_worker.scoring.base import HighlightScorer
from freecher_worker.scoring.heuristic import HeuristicScorer
from freecher_worker.scoring.llm import OpenAILLMScorer
from freecher_worker.transcription.models import Transcript
from freecher_worker.transcription.whisper import BaseTranscriber, WhisperTranscriber
from freecher_worker.utils.json_io import load_json, save_json
from freecher_worker.utils.logging import WorkerLogger

PIPELINE_VERSION = "0.2.0"


class EnvironmentInfo(BaseModel):
    python_version: str
    platform: str


class AsrManifestInfo(BaseModel):
    model: str
    device: str
    compute_type: str
    language: Optional[str] = None
    beam_size: int = 5
    vad_filter: bool = True


class CandidateConfigInfo(BaseModel):
    candidate_set_id: Optional[str] = None
    min_seconds: float
    target_seconds: float
    max_seconds: float
    overlap: float


class ScoringManifestInfo(BaseModel):
    scorer: str
    scorer_version: str
    llm_model: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None


class RankingManifestInfo(BaseModel):
    top_k: int
    dedup_threshold: float


class PipelineTimings(BaseModel):
    probe_seconds: float = 0.0
    audio_seconds: float = 0.0
    transcription_seconds: float = 0.0
    candidate_seconds: float = 0.0
    scoring_seconds: float = 0.0
    ranking_seconds: float = 0.0
    clipping_seconds: float = 0.0
    total_seconds: float = 0.0


class PipelineStatistics(BaseModel):
    transcript_segment_count: int = 0
    candidate_count: int = 0
    selected_highlight_count: int = 0


class HighlightManifestItem(BaseModel):
    rank: int
    start: float
    end: float
    duration: float
    score: float
    reason: str
    file: Optional[str] = None
    candidate_id: str
    padded_start: Optional[float] = None
    padded_end: Optional[float] = None
    score_breakdown: Optional[HighlightScore] = None


class Manifest(BaseModel):
    pipeline_version: str = PIPELINE_VERSION
    created_at: str
    source: str
    source_fingerprint: SourceFingerprint
    environment: EnvironmentInfo
    asr: AsrManifestInfo
    candidate_config: CandidateConfigInfo
    scoring: ScoringManifestInfo
    ranking: RankingManifestInfo
    timings: PipelineTimings
    statistics: PipelineStatistics
    highlights: list[HighlightManifestItem] = Field(default_factory=list)


def _resolve_run_dir(
    video_path: Path,
    output_base_dir: Path,
    run_id: Optional[str] = None,
    force: bool = False,
) -> tuple[Path, str]:
    """Determine run directory.

    If run_id is given, uses that.
    If force is False and an existing run directory for this video exists, resumes the latest run.
    Otherwise, creates a new timestamped directory.
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)
    clean_stem = re.sub(r"[^\w\-]", "_", video_path.stem)

    if run_id:
        run_dir = output_base_dir / run_id
        return run_dir, run_id

    if not force:
        # Check for existing matching runs to resume
        candidates = sorted(
            [d for d in output_base_dir.iterdir() if d.is_dir() and d.name.endswith(f"_{clean_stem}")],
            key=lambda d: d.name,
            reverse=True,
        )
        if candidates:
            latest = candidates[0]
            return latest, latest.name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_run_id = f"{timestamp}_{clean_stem}"
    run_dir = output_base_dir / new_run_id
    return run_dir, new_run_id


def run_pipeline(
    video_path: Path | str,
    output_dir: Optional[Path | str] = None,
    config: Optional[Settings] = None,
    force: bool = False,
    run_id: Optional[str] = None,
    analysis_only: bool = False,
    transcriber: Optional[BaseTranscriber] = None,
    scorer: Optional[HighlightScorer] = None,
) -> Manifest:
    """Execute the freecher-worker Phase 1.1 processing pipeline on a video file.

    Stages:
        1. [probe]: ffprobe video verification & source fingerprint computation
        2. [audio]: FFmpeg extraction to 16kHz mono WAV
        3. [transcription]: faster-whisper transcription with configuration-aware caching
        4. [candidates]: temporal transcript-window segmentation
        5. [scoring]: heuristic or LLM highlight scoring with fallback tracking
        6. [ranking]: deduplication (NMS) and top-K highlight ranking
        7. [clipping]: FFmpeg cutting with contextual padding (skipped in analysis-only mode)
        8. [done]: generate expanded manifest.json
    """
    start_total_time = time.perf_counter()
    src_video = Path(video_path).resolve()
    if not src_video.is_file():
        raise FileNotFoundError(f"Input video file not found: {src_video}")

    cfg = config or get_settings()
    base_out = Path(output_dir) if output_dir else cfg.output_dir

    run_dir, resolved_run_id = _resolve_run_dir(src_video, base_out, run_id=run_id, force=force)
    run_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "worker.log"
    logger = WorkerLogger(log_file=log_file)
    logger.info("pipeline", f"Starting freecher-worker run: {resolved_run_id} (version {PIPELINE_VERSION})")
    logger.info("pipeline", f"Source: {src_video}")
    if analysis_only:
        logger.info("pipeline", "Mode: --analysis-only (clipping stage will be skipped)")

    # Artifact paths
    media_json_path = run_dir / "media.json"
    audio_wav_path = run_dir / "audio.wav"
    transcript_json_path = run_dir / "transcript.json"
    candidates_json_path = run_dir / "candidates.json"
    highlights_json_path = run_dir / "highlights.json"
    manifest_json_path = run_dir / "manifest.json"

    timings = PipelineTimings()

    # ==========================================
    # 1. STAGE: PROBE & SOURCE FINGERPRINT
    # ==========================================
    media_info: MediaInfo
    source_fp: SourceFingerprint
    with logger.stage("probe") as s:
        probe_start = time.perf_counter()
        reused_media = False
        if not force and media_json_path.is_file():
            try:
                cached_data = load_json(media_json_path)
                cand_info = MediaInfo.model_validate(cached_data)
                # Verify source file has not changed
                cand_fp = compute_source_fingerprint(src_video, cand_info.duration_seconds)
                if cand_info.file_size == cand_fp.file_size:
                    media_info = cand_info
                    source_fp = cand_fp
                    reused_media = True
                    logger.info("probe", f"cache HIT: reused media.json (duration={media_info.duration_seconds:.1f}s, fp={source_fp.fingerprint_id})")
            except Exception as exc:
                logger.warning("probe", f"Cached media.json invalid ({exc}), reprobing...")

        if not reused_media:
            logger.info("probe", "cache MISS: probing media with ffprobe...")
            media_info = probe_media(src_video)
            source_fp = compute_source_fingerprint(src_video, media_info.duration_seconds)
            save_json(media_info, media_json_path)
            logger.info(
                "probe",
                f"CREATED media.json: {media_info.width}x{media_info.height} @ {media_info.fps:.1f}fps, duration {media_info.duration_seconds:.1f}s (fp={source_fp.fingerprint_id})",
            )
        timings.probe_seconds = round(time.perf_counter() - probe_start, 3)

    # ==========================================
    # 2. STAGE: AUDIO EXTRACTION
    # ==========================================
    with logger.stage("audio") as s:
        audio_start = time.perf_counter()
        if not force and audio_wav_path.is_file() and audio_wav_path.stat().st_size > 0:
            logger.info("audio", "cache HIT: reused 16kHz mono audio.wav")
        else:
            logger.info("audio", "cache MISS: extracting 16kHz mono WAV via ffmpeg...")
            extract_audio(src_video, audio_wav_path)
            logger.info("audio", f"CREATED audio.wav ({audio_wav_path.stat().st_size / 1024:.1f} KB)")
        timings.audio_seconds = round(time.perf_counter() - audio_start, 3)

    # ==========================================
    # 3. STAGE: TRANSCRIPTION (Configuration-Aware Cache)
    # ==========================================
    transcript: Transcript
    with logger.stage("transcription") as s:
        trans_start = time.perf_counter()
        reused_transcript = False

        if not force and transcript_json_path.is_file():
            try:
                cached_data = load_json(transcript_json_path)
                cand_transcript = Transcript.model_validate(cached_data)

                # Check cache invalidation conditions
                mismatch_reason = None
                if cand_transcript.source_fingerprint_id and cand_transcript.source_fingerprint_id != source_fp.fingerprint_id:
                    mismatch_reason = f"source fingerprint mismatch (cached={cand_transcript.source_fingerprint_id}, current={source_fp.fingerprint_id})"
                elif cand_transcript.model != cfg.asr_model:
                    mismatch_reason = f"ASR model changed (cached='{cand_transcript.model}', requested='{cfg.asr_model}')"
                elif cand_transcript.device != cfg.asr_device:
                    mismatch_reason = f"ASR device changed (cached='{cand_transcript.device}', requested='{cfg.asr_device}')"
                elif cand_transcript.compute_type != cfg.asr_compute_type:
                    mismatch_reason = f"ASR compute_type changed (cached='{cand_transcript.compute_type}', requested='{cfg.asr_compute_type}')"
                elif cfg.asr_language is not None and cand_transcript.language != cfg.asr_language:
                    mismatch_reason = f"ASR language changed (cached='{cand_transcript.language}', requested='{cfg.asr_language}')"
                elif getattr(cand_transcript, "beam_size", 5) != cfg.asr_beam_size:
                    mismatch_reason = f"ASR beam_size changed (cached={getattr(cand_transcript, 'beam_size', 5)}, requested={cfg.asr_beam_size})"
                elif getattr(cand_transcript, "vad_filter", True) != cfg.asr_vad_filter:
                    mismatch_reason = f"ASR vad_filter changed (cached={getattr(cand_transcript, 'vad_filter', True)}, requested={cfg.asr_vad_filter})"

                if mismatch_reason is None:
                    transcript = cand_transcript
                    reused_transcript = True
                    logger.info(
                        "transcription",
                        f"cache HIT: reused transcript.json ({len(transcript.segments)} segments, lang={transcript.language})",
                    )
                else:
                    logger.info("transcription", f"cache MISS: {mismatch_reason}")

            except Exception as exc:
                logger.warning("transcription", f"cache INVALIDATED: cached transcript unreadable ({exc})")

        if not reused_transcript:
            active_transcriber = transcriber or WhisperTranscriber(
                model_name=cfg.asr_model,
                device=cfg.asr_device,
                compute_type=cfg.asr_compute_type,
                beam_size=cfg.asr_beam_size,
                vad_filter=cfg.asr_vad_filter,
            )
            transcript = active_transcriber.transcribe(
                audio_wav_path,
                language=cfg.asr_language,
                source_fingerprint_id=source_fp.fingerprint_id,
            )
            save_json(transcript, transcript_json_path)
            logger.info(
                "transcription",
                f"CREATED transcript.json: {len(transcript.segments)} segments (lang={transcript.language}, p={transcript.language_probability:.2f})",
            )
        timings.transcription_seconds = round(time.perf_counter() - trans_start, 3)

    # ==========================================
    # 4. STAGE: CANDIDATE WINDOW GENERATION
    # ==========================================
    candidates: list[CandidateWindow]
    candidate_set_id: str = ""
    transcript_hash = transcript.compute_transcript_hash()
    with logger.stage("candidates") as s:
        cand_start = time.perf_counter()
        reused_candidates = False

        if not force and candidates_json_path.is_file():
            try:
                cached_data = load_json(candidates_json_path)
                # Check whether candidates.json is wrapped CandidateDocument or legacy list
                if isinstance(cached_data, dict) and "candidates" in cached_data:
                    doc = CandidateDocument.model_validate(cached_data)
                    mismatch_reason = None
                    if doc.transcript_hash != transcript_hash:
                        mismatch_reason = f"transcript content changed (hash mismatch)"
                    elif doc.min_seconds != cfg.highlight_min_seconds:
                        mismatch_reason = f"min_seconds changed ({doc.min_seconds} -> {cfg.highlight_min_seconds})"
                    elif doc.target_seconds != cfg.highlight_target_seconds:
                        mismatch_reason = f"target_seconds changed ({doc.target_seconds} -> {cfg.highlight_target_seconds})"
                    elif doc.max_seconds != cfg.highlight_max_seconds:
                        mismatch_reason = f"max_seconds changed ({doc.max_seconds} -> {cfg.highlight_max_seconds})"
                    elif doc.overlap_seconds != cfg.highlight_overlap_seconds:
                        mismatch_reason = f"overlap changed ({doc.overlap_seconds} -> {cfg.highlight_overlap_seconds})"

                    if mismatch_reason is None:
                        candidates = doc.candidates
                        candidate_set_id = doc.candidate_set_id
                        reused_candidates = True
                        logger.info("candidates", f"cache HIT: reused candidates.json ({len(candidates)} windows, id={candidate_set_id})")
                    else:
                        logger.info("candidates", f"cache MISS: {mismatch_reason}")
                elif isinstance(cached_data, list) and not reused_transcript:
                    logger.info("candidates", "cache MISS: legacy format or transcript updated")
                elif isinstance(cached_data, list) and reused_transcript:
                    # Legacy list matching current transcript
                    candidates = [CandidateWindow.model_validate(item) for item in cached_data]
                    candidate_set_id = compute_candidate_set_id(
                        transcript_hash=transcript_hash,
                        min_seconds=cfg.highlight_min_seconds,
                        target_seconds=cfg.highlight_target_seconds,
                        max_seconds=cfg.highlight_max_seconds,
                        overlap_seconds=cfg.highlight_overlap_seconds,
                    )
                    reused_candidates = True
                    logger.info("candidates", f"cache HIT: reused candidates.json ({len(candidates)} windows, id={candidate_set_id})")
            except Exception as exc:
                logger.warning("candidates", f"cache INVALIDATED: cached candidates invalid ({exc})")

        if not reused_candidates:
            candidates = generate_candidate_windows(
                transcript,
                min_seconds=cfg.highlight_min_seconds,
                target_seconds=cfg.highlight_target_seconds,
                max_seconds=cfg.highlight_max_seconds,
                overlap_seconds=cfg.highlight_overlap_seconds,
            )
            cand_doc = CandidateDocument(
                transcript_hash=transcript_hash,
                min_seconds=cfg.highlight_min_seconds,
                target_seconds=cfg.highlight_target_seconds,
                max_seconds=cfg.highlight_max_seconds,
                overlap_seconds=cfg.highlight_overlap_seconds,
                candidates=candidates,
            )
            candidate_set_id = cand_doc.candidate_set_id
            save_json(cand_doc, candidates_json_path)
            logger.info("candidates", f"CREATED candidates.json ({len(candidates)} candidate windows generated, id={candidate_set_id})")
        timings.candidate_seconds = round(time.perf_counter() - cand_start, 3)

    if not candidates:
        logger.warning("candidates", "No speech candidate windows found. Pipeline terminating with empty highlights.")
        manifest = Manifest(
            pipeline_version=PIPELINE_VERSION,
            created_at=datetime.now().isoformat(),
            source=str(src_video),
            source_fingerprint=source_fp,
            environment=EnvironmentInfo(
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                platform=platform.platform(),
            ),
            asr=AsrManifestInfo(
                model=cfg.asr_model,
                device=cfg.asr_device,
                compute_type=cfg.asr_compute_type,
                language=cfg.asr_language or transcript.language,
                beam_size=cfg.asr_beam_size,
                vad_filter=cfg.asr_vad_filter,
            ),
            candidate_config=CandidateConfigInfo(
                candidate_set_id=candidate_set_id,
                min_seconds=cfg.highlight_min_seconds,
                target_seconds=cfg.highlight_target_seconds,
                max_seconds=cfg.highlight_max_seconds,
                overlap=cfg.highlight_overlap_seconds,
            ),
            scoring=ScoringManifestInfo(
                scorer=cfg.scorer,
                scorer_version="heuristic_v1" if cfg.scorer == "heuristic" else "1.1.0",
                llm_model=cfg.llm_model if cfg.scorer == "llm" else None,
                fallback_used=False,
                fallback_reason=None,
            ),
            ranking=RankingManifestInfo(
                top_k=cfg.highlight_top_k,
                dedup_threshold=cfg.dedup_overlap_threshold,
            ),
            timings=timings,
            statistics=PipelineStatistics(
                transcript_segment_count=len(transcript.segments),
                candidate_count=0,
                selected_highlight_count=0,
            ),
            highlights=[],
        )
        save_json(manifest, manifest_json_path)
        return manifest

    # ==========================================
    # 5. STAGE: SCORING (With Fallback Tracking)
    # ==========================================
    scores: list[HighlightScore]
    scoring_fallback_used = False
    scoring_fallback_reason: Optional[str] = None
    with logger.stage("scoring") as s:
        scoring_start = time.perf_counter()
        active_scorer = scorer
        if active_scorer is None:
            if cfg.scorer == "llm":
                active_scorer = OpenAILLMScorer(
                    base_url=cfg.llm_base_url,
                    api_key=cfg.llm_api_key,
                    model=cfg.llm_model or "gpt-4o-mini",
                )
            else:
                active_scorer = HeuristicScorer()

        scorer_name = getattr(active_scorer, "name", type(active_scorer).__name__)
        scorer_ver = getattr(active_scorer, "version", "heuristic_v1")

        scores = active_scorer.score_batch(candidates)

        for sc in scores:
            if getattr(sc, "fallback_used", False):
                scoring_fallback_used = True
                if not scoring_fallback_reason:
                    scoring_fallback_reason = getattr(sc, "fallback_reason", "LLM scoring failed")

        if scoring_fallback_used:
            logger.warning(
                "scoring",
                f"LLM fallback was engaged during scoring: {scoring_fallback_reason}",
            )

        logger.info(
            "scoring",
            f"scored {len(scores)} candidates using {scorer_name} (v{scorer_ver})",
        )

        # Persist model predictions to runs/<run_id>/scores/<scorer>_<version>.json
        try:
            scores_dir = run_dir / "scores"
            scores_dir.mkdir(parents=True, exist_ok=True)
            if scorer_ver.startswith(f"{scorer_name.lower()}_"):
                scorer_file_name = f"{scorer_ver}.json"
            else:
                scorer_file_name = f"{scorer_name.lower()}_{scorer_ver}.json"

            cand_score_pairs = list(zip(candidates, scores))
            cand_score_pairs.sort(key=lambda cs: cs[1].score, reverse=True)

            pred_items = [
                ScorerPredictionItem(
                    candidate_id=c.id,
                    rank=r_idx,
                    score=sc_item.score,
                    reason=sc_item.reason,
                    subscores={
                        "hook_score": sc_item.hook_score,
                        "standalone_score": sc_item.standalone_score,
                        "emotion_score": sc_item.emotion_score,
                        "information_score": sc_item.information_score,
                        "shareability_score": getattr(sc_item, "shareability_score", 0.0),
                    },
                )
                for r_idx, (c, sc_item) in enumerate(cand_score_pairs, start=1)
            ]

            pred_doc = ScorerPredictionDocument(
                candidate_set_id=candidate_set_id,
                scorer=scorer_name.lower(),
                scorer_version=scorer_ver,
                model=cfg.llm_model if cfg.scorer == "llm" else None,
                predictions=pred_items,
            )
            save_json(pred_doc, scores_dir / scorer_file_name)
            logger.info("scoring", f"saved predictions to scores/{scorer_file_name}")
        except Exception as score_save_err:
            logger.warning("scoring", f"failed to write prediction file: {score_save_err}")

        timings.scoring_seconds = round(time.perf_counter() - scoring_start, 3)

    # ==========================================
    # 6. STAGE: RANKING & DEDUPLICATION
    # ==========================================
    highlights: list[Highlight]
    with logger.stage("ranking") as s:
        ranking_start = time.perf_counter()
        highlights = rank_and_deduplicate(
            candidates,
            scores,
            top_k=cfg.highlight_top_k,
            overlap_threshold=cfg.dedup_overlap_threshold,
        )
        save_json(highlights, highlights_json_path)
        logger.info("ranking", f"selected {len(highlights)} top highlights (top_k={cfg.highlight_top_k}, dedup_threshold={cfg.dedup_overlap_threshold})")
        timings.ranking_seconds = round(time.perf_counter() - ranking_start, 3)

    # ==========================================
    # 7. STAGE: CLIPPING (Optional in --analysis-only)
    # ==========================================
    manifest_highlights: list[HighlightManifestItem] = []
    with logger.stage("clipping") as s:
        clipping_start = time.perf_counter()

        if analysis_only:
            logger.info("clipping", "analysis-only mode: skipping MP4 clip rendering")
            for hl in highlights:
                manifest_highlights.append(
                    HighlightManifestItem(
                        rank=hl.rank,
                        start=hl.start,
                        end=hl.end,
                        duration=hl.duration,
                        score=hl.score,
                        reason=hl.reason,
                        file=None,
                        candidate_id=hl.candidate_id,
                        padded_start=None,
                        padded_end=None,
                        score_breakdown=hl.score_breakdown,
                    )
                )
            timings.clipping_seconds = 0.0
        else:
            reused_clip_count = 0
            created_clip_count = 0

            for idx, hl in enumerate(highlights, start=1):
                pad = cfg.clip_padding_seconds
                padded_start = max(0.0, hl.start - pad)
                padded_end = min(media_info.duration_seconds, hl.end + pad)

                clip_filename = f"clip_{hl.rank:02d}.mp4"
                clip_path = clips_dir / clip_filename
                rel_file_path = f"clips/{clip_filename}"

                # Check if clip exists and can be reused
                if not force and clip_path.is_file() and clip_path.stat().st_size > 0:
                    reused_clip_count += 1
                    logger.info("clipping", f"clip {idx}/{len(highlights)}: REUSED cached clip ({rel_file_path})")
                else:
                    clip_video(
                        source_video=src_video,
                        output_clip=clip_path,
                        start_seconds=padded_start,
                        end_seconds=padded_end,
                    )
                    created_clip_count += 1
                    logger.info("clipping", f"clip {idx}/{len(highlights)}: CREATED clip ({rel_file_path})")

                hl.file = rel_file_path
                hl.padded_start = round(padded_start, 2)
                hl.padded_end = round(padded_end, 2)

                manifest_highlights.append(
                    HighlightManifestItem(
                        rank=hl.rank,
                        start=hl.start,
                        end=hl.end,
                        duration=hl.duration,
                        score=hl.score,
                        reason=hl.reason,
                        file=rel_file_path,
                        candidate_id=hl.candidate_id,
                        padded_start=round(padded_start, 2),
                        padded_end=round(padded_end, 2),
                        score_breakdown=hl.score_breakdown,
                    )
                )

            # Update highlights.json with clip paths and padding
            save_json(highlights, highlights_json_path)
            logger.info(
                "clipping",
                f"finished clipping ({created_clip_count} created, {reused_clip_count} reused)",
            )
            timings.clipping_seconds = round(time.perf_counter() - clipping_start, 3)

    # ==========================================
    # 8. STAGE: MANIFEST & DONE
    # ==========================================
    total_time = round(time.perf_counter() - start_total_time, 3)
    timings.total_seconds = total_time

    manifest = Manifest(
        pipeline_version=PIPELINE_VERSION,
        created_at=datetime.now().isoformat(),
        source=str(src_video),
        source_fingerprint=source_fp,
        environment=EnvironmentInfo(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform=platform.platform(),
        ),
        asr=AsrManifestInfo(
            model=cfg.asr_model,
            device=cfg.asr_device,
            compute_type=cfg.asr_compute_type,
            language=cfg.asr_language or transcript.language,
            beam_size=cfg.asr_beam_size,
            vad_filter=cfg.asr_vad_filter,
        ),
        candidate_config=CandidateConfigInfo(
            candidate_set_id=candidate_set_id,
            min_seconds=cfg.highlight_min_seconds,
            target_seconds=cfg.highlight_target_seconds,
            max_seconds=cfg.highlight_max_seconds,
            overlap=cfg.highlight_overlap_seconds,
        ),
        scoring=ScoringManifestInfo(
            scorer=cfg.scorer,
            scorer_version=scorer_ver,
            llm_model=cfg.llm_model if cfg.scorer == "llm" else None,
            fallback_used=scoring_fallback_used,
            fallback_reason=scoring_fallback_reason,
        ),
        ranking=RankingManifestInfo(
            top_k=cfg.highlight_top_k,
            dedup_threshold=cfg.dedup_overlap_threshold,
        ),
        timings=timings,
        statistics=PipelineStatistics(
            transcript_segment_count=len(transcript.segments),
            candidate_count=len(candidates),
            selected_highlight_count=len(manifest_highlights),
        ),
        highlights=manifest_highlights,
    )
    save_json(manifest, manifest_json_path)

    logger.info("done", f"total {total_time:.2f} sec, manifest saved to {manifest_json_path}")
    return manifest
