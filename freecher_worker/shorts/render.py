"""Production stage: ranked candidate -> dynamic subclip -> smart 9:16 reframe -> 1080x1920 MP4.

Freecher only produces vertical short-form video. There is deliberately no aspect-ratio option:
the output is always 9:16, 1080x1920, H.264 + AAC.
"""

from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from freecher_worker.config import Settings, get_settings
from freecher_worker.crop.expression import build_ffmpeg_crop_expression
from freecher_worker.highlights.models import CandidateDocument, CandidateWindow, Highlight
from freecher_worker.media.clipper import is_nvenc_available
from freecher_worker.media.probe import probe_media
from freecher_worker.rendering.audio import build_loudnorm_filter, measure_loudness
from freecher_worker.rendering.validator import VideoValidationResult, validate_rendered_video
from freecher_worker.transcription.models import Transcript
from freecher_worker.utils.json_io import load_json, save_json

from .reframe import (
    REFRAME_MODE_CENTER,
    REFRAME_MODE_SMART,
    ReframeConfig,
    ReframeDiagnostics,
    ReframePlan,
    build_center_crop_plan,
    build_reframe_plan,
    render_debug_overlay,
)
from .refinement import (
    AVAILABLE_DURATION_MODES,
    DURATION_MODE_AUTO,
    SubclipConfig,
    SubclipSelection,
    build_speech_units,
    refine_subclip,
)
from .signals import load_signal_curve
from .timeframe import CandidateTimeframe, interpret_observed_region

logger = logging.getLogger("freecher_worker")

SHORTS_PIPELINE_VERSION = "shorts_v1"

#: The only output format Freecher produces.
ASPECT_RATIO = "9:16"

ENCODER_AUTO = "auto"
ENCODER_X264 = "libx264"
ENCODER_NVENC = "h264_nvenc"


class ShortMetadata(BaseModel):
    """Metadata describing one produced vertical short."""

    candidate_id: str
    source_start_sec: float = Field(description="Absolute candidate window start")
    source_end_sec: float = Field(description="Absolute candidate window end")

    short_start_offset_sec: float = Field(description="Short start relative to candidate start")
    short_end_offset_sec: float = Field(description="Short end relative to candidate start")

    short_source_start_sec: float = Field(description="Absolute short start in the source video")
    short_source_end_sec: float = Field(description="Absolute short end in the source video")

    duration_sec: float

    aspect_ratio: str = ASPECT_RATIO
    width: int = 1080
    height: int = 1920

    reframing_mode: str = REFRAME_MODE_SMART
    duration_mode: str = DURATION_MODE_AUTO

    # --- diagnostics -------------------------------------------------------
    pipeline_version: str = SHORTS_PIPELINE_VERSION
    rank: Optional[int] = None
    index: int = 1
    file: str = ""
    debug_file: Optional[str] = None
    encoder: str = ENCODER_X264
    encoder_fallback_used: bool = False
    audio_normalized: bool = False
    candidate_duration_sec: float = 0.0
    subclip: Optional[SubclipSelection] = None
    reframe: Optional[ReframeDiagnostics] = None
    advisory_interpretation: Optional[str] = None
    advisory_reason: Optional[str] = None
    timings: Dict[str, float] = Field(default_factory=dict)
    validation: Optional[VideoValidationResult] = None


class ShortsManifest(BaseModel):
    """Manifest for a batch of produced shorts."""

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    pipeline_version: str = SHORTS_PIPELINE_VERSION
    source_video: str
    aspect_ratio: str = ASPECT_RATIO
    width: int = 1080
    height: int = 1920
    duration_mode: str = DURATION_MODE_AUTO
    requested: int = 0
    shorts: List[ShortMetadata] = Field(default_factory=list)


def subclip_config_from_settings(settings: Settings) -> SubclipConfig:
    """Build the refinement configuration from application settings."""
    return SubclipConfig(
        min_duration_sec=settings.subclip_min_duration_sec,
        target_min_duration_sec=settings.subclip_target_min_duration_sec,
        target_max_duration_sec=settings.subclip_target_max_duration_sec,
        max_duration_sec=settings.subclip_max_duration_sec,
        hook_window_sec=settings.subclip_hook_window_sec,
        tail_window_sec=settings.subclip_tail_window_sec,
        pre_roll_sec=settings.subclip_pre_roll_sec,
        post_roll_sec=settings.subclip_post_roll_sec,
        boring_threshold=settings.subclip_boring_threshold,
    )


def reframe_config_from_settings(settings: Settings) -> ReframeConfig:
    """Build the reframing configuration from application settings."""
    return ReframeConfig(
        analysis_fps=settings.reframe_analysis_fps,
        detect_max_width=settings.reframe_detect_max_width,
        subject_padding_ratio=settings.reframe_subject_padding_ratio,
        head_position_ratio=settings.reframe_head_position_ratio,
        headroom_ratio=settings.reframe_headroom_ratio,
        edge_margin_ratio=settings.reframe_edge_margin_ratio,
        deadzone_ratio=settings.reframe_deadzone_ratio,
        smoothing_alpha=settings.reframe_smoothing_alpha,
        max_velocity_px_per_sec=settings.reframe_max_velocity_px_per_sec,
        max_acceleration_px_per_sec2=settings.reframe_max_acceleration_px_per_sec2,
        switch_hold_sec=settings.reframe_switch_hold_sec,
        switch_margin=settings.reframe_switch_margin,
        min_switch_interval_sec=settings.reframe_min_switch_interval_sec,
        track_max_misses=settings.reframe_track_max_misses,
        scene_cut_threshold=settings.reframe_scene_cut_threshold,
        dual_subject_balance=settings.reframe_dual_subject_balance,
        jitter_epsilon_px=settings.reframe_jitter_epsilon_px,
    )


def resolve_encoder(requested: str = ENCODER_AUTO) -> str:
    """Resolve the video encoder, defaulting to libx264 whenever NVENC is unavailable.

    NVENC is never required: on the GTX 1650 / WSL2 target it is typically absent, and the
    render must still succeed.
    """
    choice = (requested or ENCODER_AUTO).lower()
    if choice == ENCODER_X264:
        return ENCODER_X264
    if choice == ENCODER_NVENC:
        if is_nvenc_available():
            return ENCODER_NVENC
        logger.warning("[shorts] h264_nvenc requested but unavailable; using libx264")
        return ENCODER_X264
    return ENCODER_NVENC if is_nvenc_available() else ENCODER_X264


def _encoder_args(encoder: str, settings: Settings) -> List[str]:
    if encoder == ENCODER_NVENC:
        return ["-c:v", ENCODER_NVENC, "-preset", "p4", "-cq", "24"]
    return [
        "-c:v",
        ENCODER_X264,
        "-preset",
        settings.shorts_x264_preset,
        "-crf",
        str(settings.shorts_x264_crf),
        "-pix_fmt",
        "yuv420p",
    ]


def build_vertical_filter(plan: ReframePlan, width: int, height: int) -> str:
    """Build the video filter chain that turns the source into an exact 1080x1920 frame."""
    traj = plan.trajectory
    x_expr = build_ffmpeg_crop_expression(traj, axis="x", escape_for_filter=True)
    y_expr = build_ffmpeg_crop_expression(traj, axis="y", escape_for_filter=True)
    return (
        f"crop={traj.crop_w}:{traj.crop_h}:{x_expr}:{y_expr},"
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1"
    )


def _run_ffmpeg(cmd: List[str], timeout: float = 1800.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def render_short(
    source_video: Path,
    timeframe: CandidateTimeframe,
    output_path: Path,
    transcript: Optional[Transcript] = None,
    run_dir: Optional[Path] = None,
    advisory_region: Optional[Dict[str, Any]] = None,
    duration_mode: str = DURATION_MODE_AUTO,
    settings: Optional[Settings] = None,
    enable_smart_reframe: bool = True,
    enable_audio_normalization: bool = True,
    encoder: str = ENCODER_AUTO,
    rank: Optional[int] = None,
    index: int = 1,
    debug_overlay: bool = False,
    trajectory_output: Optional[Path] = None,
) -> ShortMetadata:
    """Produce one publication-ready 9:16 short from a ranked candidate window."""
    cfg = settings or get_settings()
    timings: Dict[str, float] = {}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- stage 1
    t0 = time.perf_counter()
    interpretation = interpret_observed_region(
        advisory_region,
        timeframe,
        strict=cfg.subclip_strict_timestamps,
    )
    advisory = interpretation.region if (interpretation.accepted and cfg.subclip_use_advisory_region) else None
    if advisory_region and not interpretation.accepted:
        logger.warning(
            f"[shorts] Advisory region for {timeframe.candidate_id} not usable "
            f"({interpretation.interpretation}): {interpretation.reason}"
        )

    curve = load_signal_curve(timeframe, transcript=transcript, run_dir=run_dir)
    selection = refine_subclip(
        timeframe=timeframe,
        curve=curve,
        transcript=transcript,
        config=subclip_config_from_settings(cfg),
        advisory=advisory,
        advisory_interpretation=interpretation.interpretation,
        duration_mode=duration_mode,
        units=build_speech_units(transcript, timeframe),
    )
    timings["subclip_refinement_seconds"] = round(time.perf_counter() - t0, 3)
    logger.info(
        f"[shorts] {timeframe.candidate_id}: candidate {timeframe.duration_sec:.2f}s -> short "
        f"{selection.duration_sec:.2f}s at offsets "
        f"[{selection.start_offset_sec:.2f}, {selection.end_offset_sec:.2f}] "
        f"(source {selection.short_source_start_sec:.2f}-{selection.short_source_end_sec:.2f}s, "
        f"signal={curve.source}, score={selection.score:.1f})"
    )

    # ---------------------------------------------------------------- stage 2
    t_reframe = time.perf_counter()
    info = probe_media(source_video)
    if enable_smart_reframe:
        plan = build_reframe_plan(
            video_path=source_video,
            source_start_sec=selection.short_source_start_sec,
            duration_sec=selection.duration_sec,
            config=reframe_config_from_settings(cfg),
            detector_name=cfg.reframe_detector,
            source_width=info.width,
            source_height=info.height,
            collect_debug=debug_overlay,
        )
    else:
        plan = build_center_crop_plan(
            info.width,
            info.height,
            selection.duration_sec,
            cfg.reframe_analysis_fps,
            "smart reframing disabled by caller",
            cfg.reframe_detector,
        )
    timings["reframe_analysis_seconds"] = round(time.perf_counter() - t_reframe, 3)

    if trajectory_output is not None:
        save_json(plan.trajectory, trajectory_output)

    debug_file: Optional[str] = None
    if debug_overlay:
        overlay_path = output_path.with_name(f"{output_path.stem}_debug.mp4")
        written = render_debug_overlay(
            source_video, plan, selection.short_source_start_sec, overlay_path
        )
        if written is not None:
            debug_file = written.name

    # ---------------------------------------------------------------- stage 3
    audio_filter = "anull"
    audio_normalized = False
    if enable_audio_normalization and cfg.audio_normalize_loudness:
        t_audio = time.perf_counter()
        measured = measure_loudness(
            source_media=source_video,
            start=selection.short_source_start_sec,
            duration=selection.duration_sec,
            target_i=cfg.audio_target_i,
            target_lra=cfg.audio_target_lra,
            target_tp=cfg.audio_target_tp,
        )
        audio_filter = build_loudnorm_filter(
            measured=measured,
            target_i=cfg.audio_target_i,
            target_lra=cfg.audio_target_lra,
            target_tp=cfg.audio_target_tp,
        )
        audio_normalized = True
        timings["audio_analysis_seconds"] = round(time.perf_counter() - t_audio, 3)

    width, height = cfg.shorts_output_width, cfg.shorts_output_height
    filter_complex = (
        f"[0:v]{build_vertical_filter(plan, width, height)}[v];"
        f"[0:a]{audio_filter}[a]"
    )

    chosen_encoder = resolve_encoder(encoder if encoder != ENCODER_AUTO else cfg.shorts_encoder)
    fallback_used = False

    def _command(enc: str) -> List[str]:
        return [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-ss", f"{selection.short_source_start_sec:.3f}",
            "-t", f"{selection.duration_sec:.3f}",
            "-i", str(source_video),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "[a]",
            *_encoder_args(enc, cfg),
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path),
        ]

    t_render = time.perf_counter()
    result = _run_ffmpeg(_command(chosen_encoder))
    if result.returncode != 0 and chosen_encoder != ENCODER_X264:
        logger.warning(
            f"[shorts] {chosen_encoder} encode failed; retrying with {ENCODER_X264}. "
            f"FFmpeg said: {result.stderr.strip()[-400:]}"
        )
        chosen_encoder = ENCODER_X264
        fallback_used = True
        result = _run_ffmpeg(_command(chosen_encoder))
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed to render short for {timeframe.candidate_id}: {result.stderr.strip()[-1500:]}"
        )
    timings["render_seconds"] = round(time.perf_counter() - t_render, 3)

    validation = validate_rendered_video(
        video_path=output_path,
        expected_width=width,
        expected_height=height,
        expected_duration=selection.duration_sec,
        strict=False,
    )

    metadata = ShortMetadata(
        candidate_id=timeframe.candidate_id,
        source_start_sec=round(timeframe.source_start_sec, 3),
        source_end_sec=round(timeframe.source_end_sec, 3),
        short_start_offset_sec=selection.start_offset_sec,
        short_end_offset_sec=selection.end_offset_sec,
        short_source_start_sec=selection.short_source_start_sec,
        short_source_end_sec=selection.short_source_end_sec,
        duration_sec=selection.duration_sec,
        width=width,
        height=height,
        reframing_mode=plan.mode if enable_smart_reframe else REFRAME_MODE_CENTER,
        duration_mode=duration_mode,
        rank=rank,
        index=index,
        file=output_path.name,
        debug_file=debug_file,
        encoder=chosen_encoder,
        encoder_fallback_used=fallback_used,
        audio_normalized=audio_normalized,
        candidate_duration_sec=round(timeframe.duration_sec, 3),
        subclip=selection,
        reframe=plan.diagnostics,
        advisory_interpretation=interpretation.interpretation,
        advisory_reason=interpretation.reason,
        timings=timings,
        validation=validation,
    )

    logger.info(
        f"[shorts] Rendered {output_path.name}: {metadata.duration_sec:.2f}s "
        f"{width}x{height} via {chosen_encoder}; subjects={plan.diagnostics.max_simultaneous_subjects}, "
        f"switches={plan.diagnostics.dominant_subject_switches}, "
        f"fallback={'yes' if plan.diagnostics.fallback_used else 'no'}, "
        f"render={timings.get('render_seconds', 0.0):.1f}s"
    )
    return metadata


# ---------------------------------------------------------------------------
# Run-level orchestration
# ---------------------------------------------------------------------------


class RunContext(BaseModel):
    """Everything the shorts stage needs from an existing run directory."""

    run_dir: Path
    source_video: Path
    video_duration_sec: float
    transcript: Optional[Transcript] = None
    candidates: Dict[str, CandidateWindow] = Field(default_factory=dict)
    highlights: List[Highlight] = Field(default_factory=list)
    advisory_regions: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    ranked_order: List[str] = Field(default_factory=list)


def load_run_context(run_dir: Path) -> RunContext:
    """Load source media, transcript, candidates, ranking and advisory regions from a run."""
    run_dir = Path(run_dir)
    manifest_file = run_dir / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"manifest.json not found in {run_dir}")

    manifest = load_json(manifest_file)
    source_video = Path(manifest["source"])
    if not source_video.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source_video}")

    duration = float(manifest.get("source_fingerprint", {}).get("duration_seconds", 0.0))
    if duration <= 0.0:
        duration = probe_media(source_video).duration_seconds

    transcript: Optional[Transcript] = None
    transcript_file = run_dir / "transcript.json"
    if transcript_file.is_file():
        transcript = Transcript.model_validate(load_json(transcript_file))

    candidates: Dict[str, CandidateWindow] = {}
    candidates_file = run_dir / "candidates.json"
    if candidates_file.is_file():
        raw = load_json(candidates_file)
        doc = CandidateDocument.model_validate(raw) if isinstance(raw, dict) else None
        items = doc.candidates if doc else [CandidateWindow.model_validate(c) for c in raw]
        candidates = {c.id: c for c in items}

    highlights: List[Highlight] = []
    highlights_file = run_dir / "highlights.json"
    if highlights_file.is_file():
        highlights = sorted(
            (Highlight.model_validate(h) for h in load_json(highlights_file)), key=lambda h: h.rank
        )

    advisory: Dict[str, Dict[str, Any]] = {}
    ranked_order: List[str] = []
    for name in ("multimodal_v1_1.json", "multimodal_v1.json"):
        scores_file = run_dir / "scores" / name
        if not scores_file.is_file():
            continue
        try:
            doc = load_json(scores_file)
            for prediction in sorted(doc.get("predictions", []), key=lambda p: p.get("rank", 10**6)):
                cid = prediction.get("candidate_id")
                if not cid:
                    continue
                ranked_order.append(cid)
                region = prediction.get("best_observed_region")
                if region and cid not in advisory:
                    advisory[cid] = region
        except Exception as exc:
            logger.warning(f"[shorts] Unable to read {scores_file}: {exc}")
        break

    return RunContext(
        run_dir=run_dir,
        source_video=source_video,
        video_duration_sec=duration,
        transcript=transcript,
        candidates=candidates,
        highlights=highlights,
        advisory_regions=advisory,
        ranked_order=ranked_order,
    )


def resolve_timeframe(context: RunContext, candidate_id: str) -> Tuple[CandidateTimeframe, Optional[int]]:
    """Locate a candidate's absolute window, preferring the ranked highlight record."""
    for highlight in context.highlights:
        if highlight.candidate_id == candidate_id:
            return (
                CandidateTimeframe(
                    candidate_id=candidate_id,
                    source_start_sec=highlight.start,
                    source_end_sec=highlight.end,
                ),
                highlight.rank,
            )

    window = context.candidates.get(candidate_id)
    if window is None:
        raise KeyError(f"Candidate '{candidate_id}' not found in this run")
    return (
        CandidateTimeframe(
            candidate_id=candidate_id,
            source_start_sec=window.start,
            source_end_sec=window.end,
        ),
        None,
    )


def select_ranked_candidates(context: RunContext, top: int) -> List[str]:
    """Return the top-N candidate ids in ranking order, without re-ranking anything."""
    if context.highlights:
        return [h.candidate_id for h in context.highlights[:top]]
    if context.ranked_order:
        return context.ranked_order[:top]
    raise FileNotFoundError(
        "No ranked highlights found in this run (expected highlights.json or scores/multimodal_v1_1.json)"
    )


def render_shorts_for_run(
    run_dir: Path,
    candidate_ids: Optional[List[str]] = None,
    top: int = 5,
    duration_mode: str = DURATION_MODE_AUTO,
    settings: Optional[Settings] = None,
    enable_smart_reframe: bool = True,
    enable_audio_normalization: bool = True,
    encoder: str = ENCODER_AUTO,
    debug_overlay: bool = False,
    output_subdir: str = "shorts",
) -> ShortsManifest:
    """Render one or more 9:16 shorts from an existing run's ranked candidates."""
    if duration_mode not in AVAILABLE_DURATION_MODES:
        raise ValueError(f"Unknown duration_mode '{duration_mode}'. Available: {AVAILABLE_DURATION_MODES}")

    cfg = settings or get_settings()
    context = load_run_context(Path(run_dir))
    targets = candidate_ids or select_ranked_candidates(context, top)

    out_dir = Path(run_dir) / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    produced: List[ShortMetadata] = []
    for index, candidate_id in enumerate(targets, start=1):
        timeframe, rank = resolve_timeframe(context, candidate_id)
        output_path = out_dir / f"short_{index:02d}.mp4"
        metadata = render_short(
            source_video=context.source_video,
            timeframe=timeframe,
            output_path=output_path,
            transcript=context.transcript,
            run_dir=context.run_dir,
            advisory_region=context.advisory_regions.get(candidate_id),
            duration_mode=duration_mode,
            settings=cfg,
            enable_smart_reframe=enable_smart_reframe,
            enable_audio_normalization=enable_audio_normalization,
            encoder=encoder,
            rank=rank,
            index=index,
            debug_overlay=debug_overlay,
            trajectory_output=out_dir / f"short_{index:02d}_crop_trajectory.json",
        )
        save_json(metadata, out_dir / f"short_{index:02d}.json")
        produced.append(metadata)

    manifest = ShortsManifest(
        source_video=str(context.source_video),
        width=cfg.shorts_output_width,
        height=cfg.shorts_output_height,
        duration_mode=duration_mode,
        requested=len(targets),
        shorts=produced,
    )
    save_json(manifest, out_dir / "shorts_manifest.json")
    logger.info(f"[shorts] Wrote {len(produced)} short(s) and manifest to {out_dir}")
    return manifest
