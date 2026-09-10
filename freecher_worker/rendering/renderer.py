"""Single-pass rendering pipeline for vertical short-form videos."""

from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from freecher_worker.config import Settings, get_settings
from freecher_worker.crop.expression import build_ffmpeg_crop_x_expression
from freecher_worker.crop.tracker import calculate_crop_dimensions, generate_crop_trajectory
from freecher_worker.highlights.models import Highlight
from freecher_worker.media.clipper import is_nvenc_available
from freecher_worker.media.probe import probe_media
from freecher_worker.transcription.models import Transcript
from freecher_worker.utils.json_io import load_json, save_json

from .asr_refinement import HighlightWordTranscriber
from .audio import build_loudnorm_filter, measure_loudness
from .boundaries import RefinedHighlight, refine_highlight
from .presets import RenderPreset, get_preset
from .validator import VideoValidationResult, validate_rendered_video
from freecher_worker.subtitles.ass import save_ass_file
from freecher_worker.subtitles.segmenter import segment_words_to_events

logger = logging.getLogger("freecher_worker")


def is_ffmpeg_filter_supported(filter_name: str) -> bool:
    """Check if the installed ffmpeg binary supports a specific filter."""
    try:
        res = subprocess.run(
            ["ffmpeg", "-nostdin", "-h", f"filter={filter_name}"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=5.0,
        )
        return res.returncode == 0 and f"Filter {filter_name}" in res.stdout
    except Exception:
        return False


class RenderItemManifest(BaseModel):
    """Manifest metadata for a single rendered short-form video."""

    rank: int
    candidate_id: str
    source_start: float
    source_end: float
    refined_start: float
    refined_end: float
    duration: float
    refinement_reason: str

    resolution: str = "1080x1920"
    crop_mode: str = "smart"
    subtitle_style: str = "ass_karaoke"
    audio_normalized: bool = True

    encoder: str
    file: str

    timings: Dict[str, float] = Field(default_factory=dict)
    validation: Optional[VideoValidationResult] = None


class RenderManifest(BaseModel):
    """Manifest of all rendered short-form videos for a run."""

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    source_video: str
    preset: str
    top_k: int
    shorts: List[RenderItemManifest] = Field(default_factory=list)


def render_single_short(
    source_video: Path,
    highlight: Highlight,
    transcript: Transcript,
    source_fingerprint_id: str,
    video_duration: float,
    run_dir: Path,
    preset: RenderPreset,
    config: Optional[Settings] = None,
    transcriber: Optional[HighlightWordTranscriber] = None,
    enable_smart_crop: bool = True,
    enable_subtitles: bool = True,
    enable_audio_normalization: bool = True,
    force: bool = False,
    refine_boundaries: bool = True,
) -> RenderItemManifest:
    """Execute complete end-to-end rendering for one highlight into a publication-ready vertical video."""
    cfg = config or get_settings()
    timings: Dict[str, float] = {}

    # 1. Boundary Refinement
    t0 = time.perf_counter()
    if refine_boundaries:
        refined = refine_highlight(
            highlight=highlight,
            transcript=transcript,
            video_duration=video_duration,
            max_shift_seconds=cfg.boundary_max_shift_seconds,
            context_before=cfg.boundary_context_before,
            context_after=cfg.boundary_context_after,
        )
    else:
        # Cut exactly what the candidate set froze. Refinement both snaps to
        # phrase boundaries and pads by context_before/context_after, so it
        # moves the window even at max_shift_seconds=0 -- there is no way to
        # get the frozen boundaries by tuning it.
        refined = RefinedHighlight(
            rank=highlight.rank,
            candidate_id=highlight.candidate_id,
            original_start=highlight.start,
            original_end=highlight.end,
            refined_start=highlight.start,
            refined_end=highlight.end,
            duration=round(highlight.end - highlight.start, 3),
            refinement_reason="boundary refinement disabled; frozen candidate boundaries used verbatim",
        )
    timings["boundary_refinement_seconds"] = round(time.perf_counter() - t0, 3)

    r_start = refined.refined_start
    r_end = refined.refined_end
    r_dur = refined.duration

    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    out_mp4_path = final_dir / f"short_{highlight.rank:02d}.mp4"

    # 2. Refined ASR with word timestamps (if subtitles enabled)
    ass_path = None
    if enable_subtitles:
        t_asr = time.perf_counter()
        words_dir = run_dir / "words"
        words_dir.mkdir(parents=True, exist_ok=True)
        words_cache = words_dir / f"highlight_{highlight.rank:02d}.json"

        active_transcriber = transcriber or HighlightWordTranscriber(
            model_name=cfg.refinement_asr_model,
            device=cfg.asr_device,
            compute_type=cfg.refinement_asr_compute_type,
        )
        word_doc = active_transcriber.transcribe_highlight(
            source_media=source_video,
            refined_start=r_start,
            refined_end=r_end,
            source_fingerprint_id=source_fingerprint_id,
            language=transcript.language,
            cache_path=words_cache,
            force=force,
        )
        timings["word_asr_seconds"] = round(time.perf_counter() - t_asr, 3)

        # 3. Subtitle Segmentation and ASS file generation
        t_sub = time.perf_counter()
        events = segment_words_to_events(
            word_doc.words,
            max_words=preset.max_words_per_subtitle,
        )
        sub_dir = run_dir / "subtitles"
        sub_dir.mkdir(parents=True, exist_ok=True)
        ass_path = sub_dir / f"highlight_{highlight.rank:02d}.ass"
        save_ass_file(
            events=events,
            output_path=ass_path,
            font_family=cfg.subtitle_font,
            font_size=preset.font_size,
            active_word_highlight=preset.active_word_highlight,
        )
        timings["subtitle_generation_seconds"] = round(time.perf_counter() - t_sub, 3)

    # 4. Smart Crop 9:16 Analysis & Trajectory
    crop_mode = "smart"
    if enable_smart_crop and preset.smart_crop:
        t_crop = time.perf_counter()
        crop_dir = run_dir / "crop_paths"
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_cache = crop_dir / f"highlight_{highlight.rank:02d}.json"

        trajectory = generate_crop_trajectory(
            video_path=source_video,
            start_seconds=r_start,
            duration_seconds=r_dur,
            analysis_fps=cfg.crop_analysis_fps,
            deadzone_ratio=cfg.crop_deadzone_ratio,
            max_velocity_px_per_sec=cfg.crop_max_velocity_pixels_per_sec,
        )
        save_json(trajectory, crop_cache)

        crop_x_expr = build_ffmpeg_crop_x_expression(trajectory, escape_for_filter=True)
        crop_scale_filter = f"crop={trajectory.crop_w}:{trajectory.crop_h}:{crop_x_expr}:0,scale={preset.width}:{preset.height}"
        timings["crop_tracking_seconds"] = round(time.perf_counter() - t_crop, 3)
    else:
        crop_mode = "center"
        # Fallback center crop
        info = probe_media(source_video)
        src_w = info.width or 1920
        src_h = info.height or 1080
        cw, ch = calculate_crop_dimensions(src_w, src_h)
        crop_scale_filter = f"crop={cw}:{ch}:(in_w-out_w)/2:0,scale={preset.width}:{preset.height}"

    # 5. Audio Loudness Analysis Pass
    audio_normalized = False
    loudnorm_filter_str = "anull"
    if enable_audio_normalization and cfg.audio_normalize_loudness:
        t_audio = time.perf_counter()
        measured = measure_loudness(
            source_media=source_video,
            start=r_start,
            duration=r_dur,
            target_i=preset.target_lufs,
            target_lra=preset.target_lra,
            target_tp=preset.target_tp,
        )
        loudnorm_filter_str = build_loudnorm_filter(
            measured=measured,
            target_i=preset.target_lufs,
            target_lra=preset.target_lra,
            target_tp=preset.target_tp,
        )
        audio_normalized = True
        timings["audio_analysis_seconds"] = round(time.perf_counter() - t_audio, 3)

    # 6. Build filtergraph for ONE final video encode
    video_filter_parts = [crop_scale_filter]
    if ass_path and ass_path.is_file():
        # Escape path for FFmpeg filtergraph
        escaped_ass = str(ass_path.resolve()).replace("\\", "/").replace(":", r"\:")
        if is_ffmpeg_filter_supported("ass"):
            video_filter_parts.append(f"ass='{escaped_ass}'")
        elif is_ffmpeg_filter_supported("subtitles"):
            video_filter_parts.append(f"subtitles='{escaped_ass}'")
        else:
            logger.warning(
                f"[render] Current FFmpeg build does not have 'ass' or 'subtitles' filter enabled (libass missing). "
                f"Subtitles file saved to {ass_path} but will not be burned into video stream."
            )

    vf_chain = ",".join(video_filter_parts)
    af_chain = loudnorm_filter_str

    filter_complex = f"[0:v]{vf_chain}[v];[0:a]{af_chain}[a]"

    # Choose encoder: prefer h264_nvenc, fallback libx264
    encoder = "h264_nvenc" if is_nvenc_available() else "libx264"
    if encoder == "h264_nvenc":
        enc_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "24"]
    else:
        enc_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "22"]

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-ss", f"{r_start:.3f}",
        "-t", f"{r_dur:.3f}",
        "-i", str(source_video),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "[a]",
        *enc_args,
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_mp4_path),
    ]

    t_render = time.perf_counter()
    res = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=600.0,
        check=False,
    )
    if res.returncode != 0:
        logger.error(f"[render] FFmpeg encoding failed: {res.stderr}")
        raise RuntimeError(f"FFmpeg encoding failed for highlight {highlight.rank}: {res.stderr}")

    timings["final_render_seconds"] = round(time.perf_counter() - t_render, 3)

    # 7. Quality Validation
    val_result = validate_rendered_video(
        video_path=out_mp4_path,
        expected_width=preset.width,
        expected_height=preset.height,
        expected_duration=r_dur,
        strict=True,
    )

    return RenderItemManifest(
        rank=highlight.rank,
        candidate_id=highlight.candidate_id,
        source_start=highlight.start,
        source_end=highlight.end,
        refined_start=r_start,
        refined_end=r_end,
        duration=r_dur,
        refinement_reason=refined.refinement_reason,
        resolution=f"{preset.width}x{preset.height}",
        crop_mode=crop_mode,
        subtitle_style="ass_karaoke" if enable_subtitles else "none",
        audio_normalized=audio_normalized,
        encoder=encoder,
        file=f"final/short_{highlight.rank:02d}.mp4",
        timings=timings,
        validation=val_result,
    )


def render_highlights_for_run(
    run_dir: Path,
    preset_name: str = "shorts",
    top_k: int = 3,
    enable_smart_crop: bool = True,
    enable_subtitles: bool = True,
    enable_audio_normalization: bool = True,
    force: bool = False,
    source_video_override: Optional[Path] = None,
) -> RenderManifest:
    """Render top highlights from an existing run directory into vertical short-form videos.

    `source_video_override` names the video explicitly, for runs whose manifest
    does not point at a local file -- an R2-backed run records the s3:// URI of
    the transcript it was built from, because no local video existed when it
    was made. Nothing is guessed: without the override the manifest is used
    exactly as before.
    """
    manifest_file = run_dir / "manifest.json"
    highlights_file = run_dir / "highlights.json"
    transcript_file = run_dir / "transcript.json"

    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest not found in {run_dir}")
    if not highlights_file.is_file():
        raise FileNotFoundError(f"Highlights not found in {run_dir}")
    if not transcript_file.is_file():
        raise FileNotFoundError(f"Transcript not found in {run_dir}")

    man = load_json(manifest_file)
    if source_video_override is not None:
        source_video = Path(source_video_override).expanduser().resolve()
        if not source_video.is_file():
            raise FileNotFoundError(
                f"--source-video does not exist or is not a file: {source_video}"
            )
    else:
        source_video = Path(man["source"])
        if not source_video.is_file():
            raise FileNotFoundError(f"Source video file does not exist: {source_video}")

    source_fp_id = man.get("source_fingerprint", {}).get("fingerprint_id", "unknown_fp")
    video_duration = float(man.get("source_fingerprint", {}).get("duration_seconds", 0.0))
    if video_duration <= 0.0:
        info = probe_media(source_video)
        video_duration = info.duration_seconds or 300.0

    transcript = Transcript.model_validate(load_json(transcript_file))
    raw_highlights = load_json(highlights_file)
    highlights = [Highlight.model_validate(h) for h in raw_highlights]

    # Filter to top_k
    target_highlights = sorted(highlights, key=lambda h: h.rank)[:top_k]
    preset = get_preset(preset_name)

    rendered_items: List[RenderItemManifest] = []
    logger.info(f"Rendering top {len(target_highlights)} highlights using preset '{preset.name}'...")

    for hl in target_highlights:
        logger.info(f"Rendering highlight #{hl.rank} (candidate: {hl.candidate_id})...")
        item_manifest = render_single_short(
            source_video=source_video,
            highlight=hl,
            transcript=transcript,
            source_fingerprint_id=source_fp_id,
            video_duration=video_duration,
            run_dir=run_dir,
            preset=preset,
            enable_smart_crop=enable_smart_crop,
            enable_subtitles=enable_subtitles,
            enable_audio_normalization=enable_audio_normalization,
            force=force,
        )
        rendered_items.append(item_manifest)

    render_manifest = RenderManifest(
        source_video=str(source_video),
        preset=preset.name,
        top_k=len(rendered_items),
        shorts=rendered_items,
    )

    save_json(render_manifest, run_dir / "render_manifest.json")
    logger.info(f"Rendering complete. Render manifest saved to {run_dir / 'render_manifest.json'}")
    return render_manifest
