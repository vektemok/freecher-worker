"""End-to-end processing pipeline orchestrator."""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from arny_worker.config import Settings, get_settings
from arny_worker.highlights.models import CandidateWindow, Highlight, HighlightScore
from arny_worker.highlights.ranker import rank_and_deduplicate
from arny_worker.highlights.segmenter import generate_candidate_windows
from arny_worker.media.audio import extract_audio
from arny_worker.media.clipper import clip_video
from arny_worker.media.probe import MediaInfo, probe_media
from arny_worker.scoring.base import HighlightScorer
from arny_worker.scoring.heuristic import HeuristicScorer
from arny_worker.scoring.llm import OpenAILLMScorer
from arny_worker.transcription.models import Transcript
from arny_worker.transcription.whisper import BaseTranscriber, WhisperTranscriber
from arny_worker.utils.json_io import load_json, save_json
from arny_worker.utils.logging import WorkerLogger


class AsrManifestInfo(BaseModel):
    model: str
    compute_type: str
    device: str


class HighlightManifestItem(BaseModel):
    rank: int
    start: float
    end: float
    duration: float
    score: float
    reason: str
    file: str
    candidate_id: str
    padded_start: float
    padded_end: float


class Manifest(BaseModel):
    source: str
    duration: float
    processing_time: float
    created_at: str
    asr: AsrManifestInfo
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
    transcriber: Optional[BaseTranscriber] = None,
    scorer: Optional[HighlightScorer] = None,
) -> Manifest:
    """Execute the full Phase 1 processing pipeline on a video file.

    Stages:
        1. [probe]: ffprobe video verification & MediaInfo extraction
        2. [audio]: FFmpeg extraction to 16kHz mono WAV
        3. [transcription]: faster-whisper transcription to normalized Transcript
        4. [candidates]: sliding window candidate generation
        5. [scoring]: heuristic or LLM highlight scoring
        6. [ranking]: deduplication and top-K highlight ranking
        7. [clipping]: FFmpeg cutting with contextual padding
        8. [done]: generate manifest.json
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
    logger.info("pipeline", f"Starting arny-worker run: {resolved_run_id}")
    logger.info("pipeline", f"Source: {src_video}")

    # Paths to stage artifacts
    media_json_path = run_dir / "media.json"
    audio_wav_path = run_dir / "audio.wav"
    transcript_json_path = run_dir / "transcript.json"
    candidates_json_path = run_dir / "candidates.json"
    highlights_json_path = run_dir / "highlights.json"
    manifest_json_path = run_dir / "manifest.json"

    # ==========================================
    # 1. STAGE: PROBE
    # ==========================================
    media_info: MediaInfo
    with logger.stage("probe") as s:
        if not force and media_json_path.is_file():
            try:
                cached_data = load_json(media_json_path)
                media_info = MediaInfo.model_validate(cached_data)
                s.complete(f"loaded cached media.json (duration {media_info.duration_seconds:.1f}s)")
            except Exception as exc:
                logger.warning("probe", f"Failed loading cached media.json ({exc}), reprobing...")
                media_info = probe_media(src_video)
                save_json(media_info, media_json_path)
                s.complete(f"probed video: {media_info.width}x{media_info.height} @ {media_info.fps:.1f}fps, duration {media_info.duration_seconds:.1f}s")
        else:
            media_info = probe_media(src_video)
            save_json(media_info, media_json_path)
            s.complete(f"probed video: {media_info.width}x{media_info.height} @ {media_info.fps:.1f}fps, duration {media_info.duration_seconds:.1f}s")

    # ==========================================
    # 2. STAGE: AUDIO EXTRACTION
    # ==========================================
    with logger.stage("audio") as s:
        if not force and audio_wav_path.is_file() and audio_wav_path.stat().st_size > 0:
            s.complete("reusing cached 16kHz mono audio.wav")
        else:
            extract_audio(src_video, audio_wav_path)
            s.complete(f"extracted 16kHz mono audio ({audio_wav_path.stat().st_size / 1024:.1f} KB)")

    # ==========================================
    # 3. STAGE: TRANSCRIPTION
    # ==========================================
    transcript: Transcript
    with logger.stage("transcription") as s:
        reused_cache = False
        if not force and transcript_json_path.is_file():
            try:
                cached_data = load_json(transcript_json_path)
                cand_transcript = Transcript.model_validate(cached_data)
                # Verify that ASR parameters match
                if (
                    cand_transcript.model == cfg.asr_model
                    and cand_transcript.compute_type == cfg.asr_compute_type
                    and cand_transcript.device == cfg.asr_device
                ):
                    transcript = cand_transcript
                    reused_cache = True
                    s.complete(
                        f"reusing cached transcript.json ({len(transcript.segments)} segments, lang={transcript.language})"
                    )
            except Exception as exc:
                logger.warning("transcription", f"Cached transcript invalid ({exc}), retranscribing...")

        if not reused_cache:
            active_transcriber = transcriber or WhisperTranscriber(
                model_name=cfg.asr_model,
                device=cfg.asr_device,
                compute_type=cfg.asr_compute_type,
                beam_size=cfg.asr_beam_size,
                vad_filter=cfg.asr_vad_filter,
            )
            transcript = active_transcriber.transcribe(audio_wav_path, language=cfg.asr_language)
            save_json(transcript, transcript_json_path)
            s.complete(
                f"transcribed {len(transcript.segments)} segments (lang={transcript.language}, p={transcript.language_probability:.2f})"
            )

    # ==========================================
    # 4. STAGE: CANDIDATE WINDOW GENERATION
    # ==========================================
    candidates: list[CandidateWindow]
    with logger.stage("candidates") as s:
        if not force and candidates_json_path.is_file():
            try:
                cached_data = load_json(candidates_json_path)
                candidates = [CandidateWindow.model_validate(item) for item in cached_data]
                s.complete(f"reusing cached candidates.json ({len(candidates)} candidates)")
            except Exception as exc:
                logger.warning("candidates", f"Cached candidates invalid ({exc}), regenerating...")
                candidates = generate_candidate_windows(
                    transcript,
                    min_seconds=cfg.highlight_min_seconds,
                    target_seconds=cfg.highlight_target_seconds,
                    max_seconds=cfg.highlight_max_seconds,
                    overlap_seconds=cfg.highlight_overlap_seconds,
                )
                save_json(candidates, candidates_json_path)
                s.complete(f"generated {len(candidates)} candidates")
        else:
            candidates = generate_candidate_windows(
                transcript,
                min_seconds=cfg.highlight_min_seconds,
                target_seconds=cfg.highlight_target_seconds,
                max_seconds=cfg.highlight_max_seconds,
                overlap_seconds=cfg.highlight_overlap_seconds,
            )
            save_json(candidates, candidates_json_path)
            s.complete(f"generated {len(candidates)} candidates")

    if not candidates:
        logger.warning("candidates", "No speech candidates found in video. Pipeline terminating with empty highlights.")
        manifest = Manifest(
            source=str(src_video),
            duration=media_info.duration_seconds,
            processing_time=round(time.perf_counter() - start_total_time, 2),
            created_at=datetime.now().isoformat(),
            asr=AsrManifestInfo(
                model=cfg.asr_model,
                compute_type=cfg.asr_compute_type,
                device=cfg.asr_device,
            ),
            highlights=[],
        )
        save_json(manifest, manifest_json_path)
        return manifest

    # ==========================================
    # 5. STAGE: SCORING
    # ==========================================
    scores: list[HighlightScore]
    with logger.stage("scoring") as s:
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

        scores = active_scorer.score_batch(candidates)
        s.complete(f"scored {len(scores)} candidates using {type(active_scorer).__name__}")

    # ==========================================
    # 6. STAGE: RANKING & DEDUPLICATION
    # ==========================================
    highlights: list[Highlight]
    with logger.stage("ranking") as s:
        highlights = rank_and_deduplicate(
            candidates,
            scores,
            top_k=cfg.highlight_top_k,
            overlap_threshold=cfg.dedup_overlap_threshold,
        )
        s.complete(f"selected {len(highlights)} top highlights (target top_k={cfg.highlight_top_k})")

    # ==========================================
    # 7. STAGE: CLIPPING
    # ==========================================
    with logger.stage("clipping") as s:
        manifest_highlights: list[HighlightManifestItem] = []

        for idx, hl in enumerate(highlights, start=1):
            logger.info("clipping", f"clip {idx}/{len(highlights)} (rank {hl.rank}, score {hl.score:.1f})...")

            # Apply contextual padding, respecting boundaries
            pad = cfg.clip_padding_seconds
            padded_start = max(0.0, hl.start - pad)
            padded_end = min(media_info.duration_seconds, hl.end + pad)

            clip_filename = f"clip_{hl.rank:02d}.mp4"
            clip_path = clips_dir / clip_filename

            # Re-clip or check if existing clip is already valid
            if force or not clip_path.is_file() or clip_path.stat().st_size == 0:
                clip_video(
                    source_video=src_video,
                    output_clip=clip_path,
                    start_seconds=padded_start,
                    end_seconds=padded_end,
                )

            rel_file_path = f"clips/{clip_filename}"
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
                )
            )

        # Save highlights.json
        save_json(highlights, highlights_json_path)
        s.complete(f"created {len(highlights)} mp4 clips in {clips_dir}")

    # ==========================================
    # 8. STAGE: MANIFEST & DONE
    # ==========================================
    total_time = round(time.perf_counter() - start_total_time, 2)
    manifest = Manifest(
        source=str(src_video),
        duration=media_info.duration_seconds,
        processing_time=total_time,
        created_at=datetime.now().isoformat(),
        asr=AsrManifestInfo(
            model=cfg.asr_model,
            compute_type=cfg.asr_compute_type,
            device=cfg.asr_device,
        ),
        highlights=manifest_highlights,
    )
    save_json(manifest, manifest_json_path)

    logger.info("done", f"total {total_time:.2f} sec, manifest saved to {manifest_json_path}")
    return manifest
