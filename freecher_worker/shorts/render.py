"""Production stage: ranked candidate -> dynamic subclip -> smart 9:16 reframe -> 1080x1920 MP4.

Freecher only produces vertical short-form video. There is deliberately no aspect-ratio option:
the output is always 9:16, 1080x1920, H.264 + AAC.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from freecher_worker.config import Settings, get_settings
from freecher_worker.highlights.models import CandidateDocument, CandidateWindow, Highlight
from freecher_worker.media.clipper import is_nvenc_available
from freecher_worker.media.probe import probe_media
from freecher_worker.rendering.audio import build_loudnorm_filter, measure_loudness
from freecher_worker.rendering.renderer import is_ffmpeg_filter_supported
from freecher_worker.rendering.validator import VideoValidationResult, validate_rendered_video
from freecher_worker.transcription.models import Transcript
from freecher_worker.utils.json_io import load_json, save_json

from .reframe import (
    REFRAME_MODE_CENTER,
    calculate_vertical_crop,
    REFRAME_MODE_SMART,
    ReframeConfig,
    ReframeDiagnostics,
    ReframePlan,
    build_center_crop_plan,
    build_reframe_plan,
    render_debug_overlay,
)
from .layout import (
    AVAILABLE_LAYOUT_MODES,
    LAYOUT_MODE_ADAPTIVE,
    LAYOUT_MODE_FULL_FRAME,
    LAYOUT_MODE_SINGLE,
    LAYOUT_VERSION,
    LayoutConfig,
    LayoutPlan,
    SemanticContext,
    build_dual_viewports,
    build_layout_plan,
)
from .layout_render import (
    AdaptiveRenderPlan,
    LayoutRenderError,
    build_adaptive_render_plan,
    build_adaptive_video_filter,
    crop_driver_of,
    describe_plan,
    supports_named_filter_instances,
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
from .trajectory import (
    CROP_DRIVER_SENDCMD,
    CROP_DRIVER_STATIC,
    CropDriverPlan,
    TrajectoryReport,
    TrajectoryValidationError,
    escape_filtergraph_value,
    plan_crop_driver,
    static_center_driver,
    validate_and_sanitize_trajectory,
)

logger = logging.getLogger("freecher_worker")

SHORTS_PIPELINE_VERSION = "shorts_v1"

#: The only output format Freecher produces.
ASPECT_RATIO = "9:16"

ENCODER_AUTO = "auto"
ENCODER_X264 = "libx264"
ENCODER_NVENC = "h264_nvenc"

STATUS_SUCCESS = "success"
STATUS_FALLBACK = "fallback"
STATUS_FAILED = "failed"


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
    layout_mode_requested: str = Field(
        default=LAYOUT_MODE_SINGLE, description="Layout strategy asked for: single | adaptive | full-frame"
    )
    layout_version: str = LAYOUT_VERSION
    layout: Optional[LayoutPlan] = Field(
        default=None, description="The layout plan this short was actually rendered from"
    )
    layout_fallback_reason: Optional[str] = Field(
        default=None, description="Why the adaptive layout was abandoned, if it was"
    )

    # --- diagnostics -------------------------------------------------------
    pipeline_version: str = SHORTS_PIPELINE_VERSION
    status: str = STATUS_SUCCESS
    rank: Optional[int] = None
    model_rank: Optional[int] = Field(default=None, description="Rank assigned by the ranking pipeline")
    model_score: Optional[float] = Field(default=None, description="Score assigned by the ranking pipeline")
    index: Optional[int] = 1
    file: str = ""
    debug_file: Optional[str] = None
    crop_driver: str = Field(default=CROP_DRIVER_STATIC, description="sendcmd | expression | static")
    crop_keyframes: int = 0
    crop_keyframes_available: int = 0
    crop_resolution_reduced: bool = False
    trajectory_report: Optional[TrajectoryReport] = None
    render_fallback_used: bool = Field(
        default=False,
        description="The dynamic crop render failed or was unusable, so a static crop was rendered",
    )
    render_failure_reason: Optional[str] = Field(
        default=None, description="Why the dynamic crop render was abandoned, if it was"
    )
    tracking_mode: str = Field(
        default="subject",
        description="How framing was driven: subject | mixed | dominant_region | center",
    )
    tracking_fallback_rate: float = Field(
        default=0.0,
        description="Fraction of analyzed frames framed by a tracking fallback; independent of render success",
    )
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


class ShortResult(BaseModel):
    """Outcome for a single requested candidate, successful or not."""

    index: Optional[int] = None
    candidate_id: str
    rank: Optional[int] = None
    status: str = Field(description="success | fallback | failed")
    file: Optional[str] = None
    reason: Optional[str] = None


class ShortsManifest(BaseModel):
    """Manifest for a batch of produced shorts."""

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    pipeline_version: str = SHORTS_PIPELINE_VERSION
    source_video: str
    aspect_ratio: str = ASPECT_RATIO
    width: int = 1080
    height: int = 1920
    duration_mode: str = DURATION_MODE_AUTO
    layout_mode: str = Field(default=LAYOUT_MODE_SINGLE, description="Layout strategy requested for the batch")
    layout_version: str = LAYOUT_VERSION
    ranking_source: str = Field(default="", description="Scorer whose ranking selected these candidates")
    ranking_origin: str = Field(default="", description="File the ranking was read from")
    ranking_model: Optional[str] = Field(default=None, description="Model behind the ranking scorer")
    requested: int = 0
    success_count: int = 0
    fallback_count: int = 0
    failure_count: int = 0
    results: List[ShortResult] = Field(default_factory=list)
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
        face_score_threshold=settings.reframe_face_score_threshold,
        face_model_path=str(settings.reframe_face_model_path) if settings.reframe_face_model_path else None,
        allow_model_download=settings.reframe_allow_model_download,
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
        track_max_gap_sec=settings.reframe_track_max_gap_sec,
        track_confirm_hits=settings.reframe_track_confirm_hits,
        track_max_reassociation_px=settings.reframe_track_max_reassociation_px,
        track_appearance_enabled=settings.reframe_track_appearance,
        track_appearance_min_similarity=settings.reframe_track_appearance_min_similarity,
        track_appearance_weight=settings.reframe_track_appearance_weight,
        track_scene_cut_reset=settings.reframe_track_scene_cut_reset,
        scene_cut_threshold=settings.reframe_scene_cut_threshold,
        dual_subject_balance=settings.reframe_dual_subject_balance,
        jitter_epsilon_px=settings.reframe_jitter_epsilon_px,
    )


def layout_config_from_settings(settings: Settings) -> LayoutConfig:
    """Build the adaptive layout configuration from application settings."""
    return LayoutConfig(
        window_sec=settings.layout_window_sec,
        persistent_min_visible_sec=settings.layout_persistent_min_visible_sec,
        persistent_min_visibility_ratio=settings.layout_persistent_min_visibility_ratio,
        persistent_min_hits=settings.layout_persistent_min_hits,
        significant_min_area_ratio=settings.layout_significant_min_area_ratio,
        dominant_min_visibility=settings.layout_dominant_min_visibility,
        secondary_min_visibility=settings.layout_secondary_min_visibility,
        group_min_subjects=settings.layout_group_min_subjects,
        min_detection_coverage=settings.layout_min_detection_coverage,
        min_tracking_confidence=settings.layout_min_tracking_confidence,
        min_face_height_ratio=settings.layout_min_face_height_ratio,
        safe_top_ratio=settings.layout_safe_top_ratio,
        safe_bottom_ratio=settings.layout_safe_bottom_ratio,
        min_layout_duration_sec=settings.layout_min_duration_sec,
        switch_confirmation_sec=settings.layout_switch_confirmation_sec,
        subject_missing_grace_sec=settings.layout_subject_missing_grace_sec,
        layout_switch_penalty=settings.layout_switch_penalty,
        scene_cut_allows_immediate_switch=settings.layout_scene_cut_immediate_switch,
        full_frame_blur_background=settings.layout_full_frame_blur,
        full_frame_blur_sigma=settings.layout_full_frame_blur_sigma,
        full_frame_background_color=settings.layout_full_frame_background_color,
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


def build_vertical_filter(
    driver: CropDriverPlan,
    width: int,
    height: int,
    script_path: Optional[Path] = None,
) -> str:
    """Build the video filter chain that turns the source into an exact WIDTHxHEIGHT frame.

    The final `scale`+`crop` pair guarantees the declared output size for any source aspect
    ratio, so the produced file is always exactly 9:16 with no letterboxing.
    """
    parts: List[str] = []
    if driver.driver == CROP_DRIVER_SENDCMD:
        if script_path is None:
            raise ValueError("sendcmd driver requires a command script path")
        parts.append(f"sendcmd=f={escape_filtergraph_value(str(script_path))}")

    parts.append(f"crop={driver.crop_w}:{driver.crop_h}:{driver.x_expr}:{driver.y_expr}")
    parts.append(f"scale={width}:{height}:force_original_aspect_ratio=increase")
    parts.append(f"crop={width}:{height}")
    parts.append("setsar=1")
    return ",".join(parts)


def resolve_crop_driver(
    plan: ReframePlan,
    sendcmd_available: bool,
) -> Tuple[CropDriverPlan, TrajectoryReport]:
    """Validate the trajectory, log its geometry, and choose how to deliver it to FFmpeg."""
    sanitized, report = validate_and_sanitize_trajectory(plan.trajectory)
    logger.info(f"[crop-driver] trajectory {report.summary()}")
    if report.non_finite_dropped or report.out_of_range_clamped or report.odd_coordinates_fixed:
        logger.warning(
            f"[crop-driver] trajectory needed repairs before rendering: "
            f"{report.non_finite_dropped} non-finite dropped, "
            f"{report.out_of_range_clamped} clamped, "
            f"{report.odd_coordinates_fixed} snapped to even"
        )
    driver = plan_crop_driver(sanitized, sendcmd_available=sendcmd_available)
    logger.info(
        f"[crop-driver] using '{driver.driver}' with {driver.keyframes}/{driver.keyframes_available} "
        f"keyframes: {driver.reason}"
    )
    return driver, report


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


def short_stem(candidate_id: str, index: Optional[int] = None) -> str:
    """Collision-safe output stem.

    The candidate id is always part of the filename, so a single-candidate render can never
    silently land on top of a batch result (and vice versa).
    """
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in candidate_id)
    return f"short_{index:02d}_{safe}" if index is not None else f"short_{safe}"


def _guard_existing_output(output_path: Path, candidate_id: str) -> None:
    """Refuse to overwrite an output that belongs to a different candidate."""
    sidecar = output_path.with_suffix(".json")
    if sidecar.is_file():
        try:
            owner = load_json(sidecar).get("candidate_id")
        except Exception:
            owner = None
        if owner and owner != candidate_id:
            raise FileExistsError(
                f"{output_path.name} already belongs to candidate '{owner}'; refusing to overwrite "
                f"it with '{candidate_id}'"
            )
    if output_path.exists():
        logger.info(f"[shorts] Replacing existing {output_path.name} for {candidate_id}")


def _cleanup(*paths: Optional[Path]) -> None:
    """Remove partial artifacts so a failed render never leaves a zero-byte file behind."""
    for path in paths:
        if path is None:
            continue
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:  # pragma: no cover - unlink failures are not worth aborting for
            logger.warning(f"[shorts] Could not remove {path}: {exc}")


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
    model_score: Optional[float] = None,
    index: Optional[int] = 1,
    debug_overlay: bool = False,
    trajectory_output: Optional[Path] = None,
    layout_mode: str = LAYOUT_MODE_SINGLE,
    semantic_context: Optional[SemanticContext] = None,
) -> ShortMetadata:
    """Produce one publication-ready 9:16 short from a ranked candidate.

    The file is rendered to a temporary path and only moved into place after FFmpeg succeeds and
    ffprobe confirms the result, so a failed render never leaves a zero-byte MP4 behind. If the
    dynamic crop fails at the FFmpeg stage, the render is retried once with a static center crop
    rather than losing the short entirely.
    """
    cfg = settings or get_settings()
    timings: Dict[str, float] = {}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _guard_existing_output(output_path, timeframe.candidate_id)

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

    render_fallback_used = not enable_smart_reframe
    render_failure_reason: Optional[str] = None
    trajectory_report: Optional[TrajectoryReport] = None

    try:
        driver, trajectory_report = resolve_crop_driver(
            plan, sendcmd_available=is_ffmpeg_filter_supported("sendcmd")
        )
    except TrajectoryValidationError as exc:
        logger.warning(
            f"[shorts] Crop trajectory for {timeframe.candidate_id} is unusable ({exc}); "
            f"falling back to a static center crop"
        )
        crop_w, crop_h = calculate_vertical_crop(info.width, info.height)
        driver = static_center_driver(info.width, info.height, crop_w, crop_h)
        render_fallback_used = True
        render_failure_reason = f"trajectory validation failed: {exc}"

    if trajectory_output is not None:
        save_json(plan.trajectory, trajectory_output)

    # ------------------------------------------------------- stage 2b: layout
    # Presentation only. Nothing below can move the short's source range - the layout stage
    # decides the *shape* of the frame, never which seconds are published.
    if layout_mode not in AVAILABLE_LAYOUT_MODES:
        raise ValueError(
            f"Unknown layout_mode '{layout_mode}'. Available: {', '.join(AVAILABLE_LAYOUT_MODES)}"
        )
    layout_cfg = layout_config_from_settings(cfg)
    t_layout = time.perf_counter()
    layout_plan = build_layout_plan(
        frames=plan.frames,
        duration=selection.duration_sec,
        source_width=plan.trajectory.source_width,
        source_height=plan.trajectory.source_height,
        crop_w=plan.trajectory.crop_w,
        crop_h=plan.trajectory.crop_h,
        output_height=cfg.shorts_output_height,
        mode=layout_mode,
        config=layout_cfg,
        reframe=reframe_config_from_settings(cfg),
        semantic=semantic_context,
    )
    layout_fallback_reason: Optional[str] = layout_plan.fallback_reason
    adaptive_plan: Optional[AdaptiveRenderPlan] = None
    layout_scripts: Dict[str, Path] = {}

    if layout_mode != LAYOUT_MODE_SINGLE:
        if not supports_named_filter_instances():
            layout_fallback_reason = (
                "this FFmpeg build does not support named filter instances (crop@id)"
            )
            logger.warning(f"[layout] {layout_fallback_reason}; rendering the single-subject layout")
        else:
            try:
                dual = build_dual_viewports(
                    plan=layout_plan,
                    frames=plan.frames,
                    source_width=plan.trajectory.source_width,
                    source_height=plan.trajectory.source_height,
                    output_width=cfg.shorts_output_width,
                    output_height=cfg.shorts_output_height,
                    reframe=reframe_config_from_settings(cfg),
                )
                adaptive_plan = build_adaptive_render_plan(
                    plan=layout_plan,
                    single_trajectory=plan.trajectory,
                    width=cfg.shorts_output_width,
                    height=cfg.shorts_output_height,
                    dual_viewports=dual,
                    config=layout_cfg,
                )
                logger.info(f"[layout] {describe_plan(adaptive_plan)}")
            except (LayoutRenderError, TrajectoryValidationError) as exc:
                layout_fallback_reason = f"adaptive layout could not be prepared: {exc}"
                logger.warning(f"[layout] {layout_fallback_reason}; rendering the single-subject layout")
                adaptive_plan = None
    timings["layout_planning_seconds"] = round(time.perf_counter() - t_layout, 3)

    debug_file: Optional[str] = None
    if debug_overlay and plan.debug_samples:
        overlay_tmp = output_path.with_name(f"{output_path.stem}_debug.tmp.mp4")
        overlay_final = output_path.with_name(f"{output_path.stem}_debug.mp4")
        written = render_debug_overlay(
            source_video,
            plan,
            selection.short_source_start_sec,
            overlay_tmp,
            layout_labels=[
                (
                    segment.start,
                    segment.end,
                    f"layout={segment.layout.upper()}\n"
                    + (
                        f"subjects={'+'.join('#' + str(i) for i in segment.subject_ids)}\n"
                        if segment.subject_ids
                        else "subjects=none\n"
                    )
                    + f"confidence={segment.confidence:.2f}  reason={segment.reason}",
                )
                for segment in layout_plan.segments
            ],
            safe_band=(layout_cfg.safe_top_ratio, layout_cfg.safe_bottom_ratio),
        )
        if written is not None and written.is_file() and written.stat().st_size > 0:
            os.replace(written, overlay_final)
            debug_file = overlay_final.name
        else:
            _cleanup(overlay_tmp)

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
    tmp_output = output_path.with_name(f"{output_path.stem}.tmp.mp4")
    script_path: Optional[Path] = None
    if driver.script:
        script_path = output_path.with_name(f"{output_path.stem}_crop_commands.txt")
        script_path.write_text(driver.script, encoding="utf-8")

    if adaptive_plan is not None:
        # Each viewport pans on its own, so each one needs its own command stream.
        for name in adaptive_plan.script_names():
            path = output_path.with_name(f"{output_path.stem}_crop_commands_{name}.txt")
            path.write_text(adaptive_plan.viewports[name].script or "", encoding="utf-8")
            layout_scripts[name] = path

    chosen_encoder = resolve_encoder(encoder if encoder != ENCODER_AUTO else cfg.shorts_encoder)
    encoder_fallback = False

    def _classic_filter(active_driver: CropDriverPlan, script: Optional[Path]) -> str:
        return (
            f"[0:v]{build_vertical_filter(active_driver, width, height, script)}[v];"
            f"[0:a]{audio_filter}[a]"
        )

    def _adaptive_filter(active: AdaptiveRenderPlan) -> str:
        return (
            f"{build_adaptive_video_filter(active, layout_scripts)};"
            f"[0:a]{audio_filter}[a]"
        )

    def _command(filter_complex: str, enc: str) -> List[str]:
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
            str(tmp_output),
        ]

    def _attempt(filter_complex: str, enc: str):
        _cleanup(tmp_output)
        result = _run_ffmpeg(_command(filter_complex, enc))
        if result.returncode != 0:
            _cleanup(tmp_output)
            return None, result.stderr.strip()[-1500:]
        validation = validate_rendered_video(
            video_path=tmp_output,
            expected_width=width,
            expected_height=height,
            expected_duration=selection.duration_sec,
            strict=False,
        )
        if not validation.valid:
            _cleanup(tmp_output)
            return None, f"output failed validation: {validation.error_message}"
        return validation, None

    t_render = time.perf_counter()
    try:
        validation = None
        error: Optional[str] = None

        if adaptive_plan is not None:
            validation, error = _attempt(_adaptive_filter(adaptive_plan), chosen_encoder)
            if error and chosen_encoder != ENCODER_X264:
                logger.warning(
                    f"[shorts] {chosen_encoder} encode failed on the adaptive layout; "
                    f"retrying with libx264: {error}"
                )
                chosen_encoder = ENCODER_X264
                encoder_fallback = True
                validation, error = _attempt(_adaptive_filter(adaptive_plan), chosen_encoder)
            if error:
                # An adaptive layout that will not render is not worth losing the short over:
                # drop back to the single-subject crop, which is the pre-adaptive behaviour.
                logger.warning(
                    f"[shorts] Adaptive layout render failed for {timeframe.candidate_id}; "
                    f"falling back to the single-subject layout. FFmpeg said: {error}"
                )
                layout_fallback_reason = f"adaptive layout render failed: {error}"
                _cleanup(*layout_scripts.values())
                layout_scripts = {}
                adaptive_plan = None
                layout_plan = build_layout_plan(
                    frames=[],
                    duration=selection.duration_sec,
                    source_width=plan.trajectory.source_width,
                    source_height=plan.trajectory.source_height,
                    crop_w=plan.trajectory.crop_w,
                    crop_h=plan.trajectory.crop_h,
                    output_height=height,
                    mode=LAYOUT_MODE_SINGLE,
                    config=layout_cfg,
                )
                layout_plan.fallback_reason = layout_fallback_reason
                error = None

        if adaptive_plan is None:
            validation, error = _attempt(_classic_filter(driver, script_path), chosen_encoder)

            if error and chosen_encoder != ENCODER_X264:
                logger.warning(f"[shorts] {chosen_encoder} encode failed; retrying with libx264: {error}")
                chosen_encoder = ENCODER_X264
                encoder_fallback = True
                validation, error = _attempt(_classic_filter(driver, script_path), chosen_encoder)

            if error and driver.driver != CROP_DRIVER_STATIC:
                logger.warning(
                    f"[shorts] Dynamic crop render failed for {timeframe.candidate_id} "
                    f"(driver={driver.driver}); retrying with a static center crop. FFmpeg said: {error}"
                )
                render_fallback_used = True
                render_failure_reason = f"dynamic crop render failed ({driver.driver}): {error}"
                driver = static_center_driver(info.width, info.height, driver.crop_w, driver.crop_h)
                _cleanup(script_path)
                script_path = None
                validation, error = _attempt(_classic_filter(driver, script_path), chosen_encoder)

        if error:
            raise RuntimeError(
                f"FFmpeg failed to render short for {timeframe.candidate_id}: {error}"
            )

        os.replace(tmp_output, output_path)
    except Exception:
        _cleanup(tmp_output)
        raise
    timings["render_seconds"] = round(time.perf_counter() - t_render, 3)

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
        reframing_mode=REFRAME_MODE_CENTER if render_fallback_used else plan.mode,
        duration_mode=duration_mode,
        layout_mode_requested=layout_mode,
        layout=layout_plan,
        layout_fallback_reason=layout_fallback_reason,
        status=STATUS_FALLBACK if (render_fallback_used and enable_smart_reframe) else STATUS_SUCCESS,
        rank=rank,
        model_rank=rank,
        model_score=model_score,
        index=index,
        file=output_path.name,
        debug_file=debug_file,
        crop_driver=crop_driver_of(adaptive_plan) if adaptive_plan is not None else driver.driver,
        crop_keyframes=driver.keyframes,
        crop_keyframes_available=driver.keyframes_available,
        crop_resolution_reduced=driver.resolution_reduced,
        trajectory_report=trajectory_report,
        render_fallback_used=render_fallback_used,
        render_failure_reason=render_failure_reason,
        tracking_mode=plan.diagnostics.tracking_mode,
        tracking_fallback_rate=plan.diagnostics.tracking_fallback_rate,
        encoder=chosen_encoder,
        encoder_fallback_used=encoder_fallback,
        audio_normalized=audio_normalized,
        candidate_duration_sec=round(timeframe.duration_sec, 3),
        subclip=selection,
        reframe=plan.diagnostics,
        advisory_interpretation=interpretation.interpretation,
        advisory_reason=interpretation.reason,
        timings=timings,
        validation=validation,
    )

    layout_summary = ", ".join(
        f"{name}={seconds:.1f}s" for name, seconds in layout_plan.duration_by_mode.items() if seconds > 0
    )
    logger.info(
        f"[shorts] Rendered {output_path.name}: {metadata.duration_sec:.2f}s "
        f"{width}x{height} via {chosen_encoder}/{metadata.crop_driver}; "
        f"layout[{layout_summary}] switches={layout_plan.switch_count}, "
        f"subjects={plan.diagnostics.max_simultaneous_subjects}, "
        f"subject_switches={plan.diagnostics.dominant_subject_switches}, "
        f"tracking={plan.diagnostics.tracking_mode}, "
        f"render={timings.get('render_seconds', 0.0):.1f}s"
    )
    return metadata


# ---------------------------------------------------------------------------
# Run-level orchestration
# ---------------------------------------------------------------------------


#: Ranking sources in production preference order. The multimodal reranker is the final stage
#: of the ranking pipeline, so its verdict outranks the earlier heuristic and LLM passes.
SCORER_PREFERENCE = (
    "multimodal_v1_1",
    "multimodal_v1",
    "highlight_v2_1",
    "highlight_v2",
    "heuristic_v1",
)
RANKING_SOURCE_AUTO = "auto"
RANKING_SOURCE_HIGHLIGHTS = "highlights"


class RankedCandidate(BaseModel):
    """One candidate as ranked by a specific scorer."""

    candidate_id: str
    rank: int
    score: Optional[float] = None


class RankingSource(BaseModel):
    """Where a batch's candidate ordering came from."""

    name: str = Field(description="Scorer version, or 'highlights' for the pipeline's own output")
    origin: str = Field(description="File the ranking was read from, relative to the run directory")
    model: Optional[str] = Field(default=None, description="Underlying model, when the scorer used one")
    candidates: List[RankedCandidate] = Field(default_factory=list)

    def rank_of(self, candidate_id: str) -> Optional[int]:
        return next((c.rank for c in self.candidates if c.candidate_id == candidate_id), None)

    def score_of(self, candidate_id: str) -> Optional[float]:
        return next((c.score for c in self.candidates if c.candidate_id == candidate_id), None)


class RunContext(BaseModel):
    """Everything the shorts stage needs from an existing run directory."""

    run_dir: Path
    source_video: Path
    video_duration_sec: float
    transcript: Optional[Transcript] = None
    candidates: Dict[str, CandidateWindow] = Field(default_factory=dict)
    highlights: List[Highlight] = Field(default_factory=list)
    advisory_regions: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    rankings: Dict[str, RankingSource] = Field(
        default_factory=dict, description="Every ranking found in the run, keyed by scorer name"
    )

    def available_rankings(self) -> List[str]:
        """Scorer names present in this run, in production preference order."""
        ordered = [name for name in SCORER_PREFERENCE if name in self.rankings]
        ordered += sorted(n for n in self.rankings if n not in SCORER_PREFERENCE and n != RANKING_SOURCE_HIGHLIGHTS)
        if RANKING_SOURCE_HIGHLIGHTS in self.rankings:
            ordered.append(RANKING_SOURCE_HIGHLIGHTS)
        return ordered


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

    # Every scorer that ran on this candidate set leaves scores/<scorer_version>.json behind.
    advisory: Dict[str, Dict[str, Any]] = {}
    rankings: Dict[str, RankingSource] = {}
    scores_dir = run_dir / "scores"
    if scores_dir.is_dir():
        for scores_file in sorted(scores_dir.glob("*.json")):
            try:
                doc = load_json(scores_file)
            except Exception as exc:
                logger.warning(f"[shorts] Unable to read {scores_file}: {exc}")
                continue
            if not isinstance(doc, dict) or "predictions" not in doc:
                continue

            name = doc.get("scorer_version") or doc.get("scorer") or scores_file.stem
            ranked: List[RankedCandidate] = []
            for prediction in sorted(doc.get("predictions", []), key=lambda p: p.get("rank", 10**6)):
                cid = prediction.get("candidate_id")
                if not cid:
                    continue
                ranked.append(
                    RankedCandidate(
                        candidate_id=cid,
                        rank=int(prediction.get("rank", len(ranked) + 1)),
                        score=prediction.get("score"),
                    )
                )
                region = prediction.get("best_observed_region")
                if region and cid not in advisory:
                    advisory[cid] = region
            if ranked:
                rankings[name] = RankingSource(
                    name=name,
                    origin=str(scores_file.relative_to(run_dir)),
                    model=doc.get("model"),
                    candidates=ranked,
                )

    if highlights:
        rankings[RANKING_SOURCE_HIGHLIGHTS] = RankingSource(
            name=RANKING_SOURCE_HIGHLIGHTS,
            origin="highlights.json",
            candidates=[
                RankedCandidate(candidate_id=h.candidate_id, rank=h.rank, score=h.score)
                for h in highlights
            ],
        )

    return RunContext(
        run_dir=run_dir,
        source_video=source_video,
        video_duration_sec=duration,
        transcript=transcript,
        candidates=candidates,
        highlights=highlights,
        advisory_regions=advisory,
        rankings=rankings,
    )


def resolve_ranking_source(context: RunContext, scorer: str = RANKING_SOURCE_AUTO) -> RankingSource:
    """Pick which ranking drives the batch.

    ``auto`` walks :data:`SCORER_PREFERENCE` and therefore uses the multimodal reranker whenever
    the run has one. This matters: ``highlights.json`` holds the *heuristic* pipeline's top-K, so
    preferring it silently rendered the wrong candidates once a multimodal pass had reordered them.
    """
    available = context.available_rankings()
    if not available:
        raise FileNotFoundError(
            f"No ranking found in {context.run_dir}. Expected scores/<scorer>.json or highlights.json"
        )

    if scorer and scorer != RANKING_SOURCE_AUTO:
        key = scorer.removesuffix(".json")
        if key not in context.rankings:
            raise KeyError(
                f"Ranking '{scorer}' not found in this run. Available: {', '.join(available)}"
            )
        return context.rankings[key]

    chosen = context.rankings[available[0]]
    if len(available) > 1:
        logger.info(
            f"[shorts] Ranking source '{chosen.name}' selected automatically "
            f"(also available: {', '.join(available[1:])})"
        )
    return chosen


def resolve_timeframe(context: RunContext, candidate_id: str) -> CandidateTimeframe:
    """Locate a candidate's absolute window from the frozen candidate set."""
    window = context.candidates.get(candidate_id)
    if window is not None:
        return CandidateTimeframe(
            candidate_id=candidate_id,
            source_start_sec=window.start,
            source_end_sec=window.end,
        )

    for highlight in context.highlights:
        if highlight.candidate_id == candidate_id:
            return CandidateTimeframe(
                candidate_id=candidate_id,
                source_start_sec=highlight.start,
                source_end_sec=highlight.end,
            )

    raise KeyError(f"Candidate '{candidate_id}' not found in this run")


def select_ranked_candidates(
    context: RunContext,
    top: int,
    scorer: str = RANKING_SOURCE_AUTO,
) -> Tuple[List[str], RankingSource]:
    """Return the top-N candidate ids in ranking order, without re-ranking anything."""
    source = resolve_ranking_source(context, scorer)
    ordered = sorted(source.candidates, key=lambda c: c.rank)
    return [c.candidate_id for c in ordered[:top]], source


def render_shorts_for_run(
    run_dir: Path,
    candidate_ids: Optional[List[str]] = None,
    top: int = 5,
    scorer: str = RANKING_SOURCE_AUTO,
    duration_mode: str = DURATION_MODE_AUTO,
    settings: Optional[Settings] = None,
    enable_smart_reframe: bool = True,
    enable_audio_normalization: bool = True,
    encoder: str = ENCODER_AUTO,
    debug_overlay: bool = False,
    output_subdir: str = "shorts",
    layout_mode: str = LAYOUT_MODE_SINGLE,
) -> ShortsManifest:
    """Render one or more 9:16 shorts from an existing run's ranked candidates.

    A candidate that cannot be rendered is retried with smart reframing disabled and, failing
    that, recorded as failed. One bad candidate never aborts the rest of the batch.
    """
    if duration_mode not in AVAILABLE_DURATION_MODES:
        raise ValueError(f"Unknown duration_mode '{duration_mode}'. Available: {AVAILABLE_DURATION_MODES}")
    if layout_mode not in AVAILABLE_LAYOUT_MODES:
        raise ValueError(f"Unknown layout_mode '{layout_mode}'. Available: {AVAILABLE_LAYOUT_MODES}")

    cfg = settings or get_settings()
    context = load_run_context(Path(run_dir))
    explicit = candidate_ids is not None
    if explicit:
        ranking = resolve_ranking_source(context, scorer)
        targets = candidate_ids
    else:
        targets, ranking = select_ranked_candidates(context, top, scorer)
    logger.info(
        f"[shorts] Ranking source: {ranking.name} ({ranking.origin}); "
        f"targets: {', '.join(targets)}"
    )

    out_dir = Path(run_dir) / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    produced: List[ShortMetadata] = []
    results: List[ShortResult] = []

    for position, candidate_id in enumerate(targets, start=1):
        # A single-candidate render is not part of a numbered batch, so it gets its own stem
        # and can never land on top of a batch result.
        index = None if (explicit and len(targets) == 1) else position
        stem = short_stem(candidate_id, index)
        output_path = out_dir / f"{stem}.mp4"

        try:
            timeframe = resolve_timeframe(context, candidate_id)
            rank = ranking.rank_of(candidate_id)
        except Exception as exc:
            logger.error(f"[shorts] Skipping {candidate_id}: {exc}")
            results.append(
                ShortResult(index=index, candidate_id=candidate_id, status=STATUS_FAILED, reason=str(exc))
            )
            continue

        score = ranking.score_of(candidate_id)

        common = dict(
            source_video=context.source_video,
            timeframe=timeframe,
            output_path=output_path,
            transcript=context.transcript,
            run_dir=context.run_dir,
            advisory_region=context.advisory_regions.get(candidate_id),
            duration_mode=duration_mode,
            settings=cfg,
            enable_audio_normalization=enable_audio_normalization,
            encoder=encoder,
            rank=rank,
            model_score=score,
            index=index,
            trajectory_output=out_dir / f"{stem}_crop_trajectory.json",
            layout_mode=layout_mode,
        )

        metadata: Optional[ShortMetadata] = None
        failure_reason: Optional[str] = None
        try:
            metadata = render_short(
                enable_smart_reframe=enable_smart_reframe,
                debug_overlay=debug_overlay,
                **common,
            )
        except Exception as exc:
            failure_reason = str(exc)
            logger.error(f"[shorts] Smart render failed for {candidate_id}: {exc}")
            if enable_smart_reframe:
                logger.info(f"[shorts] Retrying {candidate_id} with a static center crop")
                try:
                    # The salvage render is deliberately the simplest thing that can work:
                    # one static crop, no layout planning on top of an already failing clip.
                    metadata = render_short(
                        enable_smart_reframe=False,
                        debug_overlay=False,
                        **{**common, "layout_mode": LAYOUT_MODE_SINGLE},
                    )
                    metadata.status = STATUS_FALLBACK
                    metadata.render_fallback_used = True
                    metadata.render_failure_reason = failure_reason
                except Exception as fallback_exc:
                    failure_reason = f"{failure_reason} | static fallback also failed: {fallback_exc}"
                    logger.error(f"[shorts] Static fallback failed for {candidate_id}: {fallback_exc}")

        if metadata is None:
            _cleanup(output_path.with_name(f"{output_path.stem}.tmp.mp4"))
            results.append(
                ShortResult(
                    index=index, candidate_id=candidate_id, rank=rank,
                    status=STATUS_FAILED, reason=failure_reason,
                )
            )
            continue

        save_json(metadata, out_dir / f"{stem}.json")
        produced.append(metadata)
        results.append(
            ShortResult(
                index=index,
                candidate_id=candidate_id,
                rank=rank,
                status=metadata.status,
                file=metadata.file,
                reason=metadata.render_failure_reason,
            )
        )

    manifest = ShortsManifest(
        source_video=str(context.source_video),
        ranking_source=ranking.name,
        ranking_origin=ranking.origin,
        ranking_model=ranking.model,
        width=cfg.shorts_output_width,
        height=cfg.shorts_output_height,
        duration_mode=duration_mode,
        layout_mode=layout_mode,
        requested=len(targets),
        success_count=sum(1 for r in results if r.status == STATUS_SUCCESS),
        fallback_count=sum(1 for r in results if r.status == STATUS_FALLBACK),
        failure_count=sum(1 for r in results if r.status == STATUS_FAILED),
        results=results,
        shorts=produced,
    )
    save_json(manifest, out_dir / "shorts_manifest.json")
    logger.info(
        f"[shorts] {manifest.success_count} succeeded, {manifest.fallback_count} used a fallback, "
        f"{manifest.failure_count} failed. Manifest written to {out_dir / 'shorts_manifest.json'}"
    )
    return manifest
