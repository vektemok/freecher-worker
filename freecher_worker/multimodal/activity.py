"""Source-wide and candidate-local temporal activity profiling for multimodal highlight reranking (v1.1)."""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
from typing import List, Optional, Tuple
import wave

import numpy as np

from freecher_worker.highlights.models import CandidateWindow
from freecher_worker.media.probe import probe_media
from freecher_worker.utils.json_io import load_json, save_json
from .frames import probe_software_decoder
from .models import (
    ActivityCurveSummary,
    ActivityPoint,
    SourceTemporalActivityPoint,
    SourceTemporalActivityProfile,
    TemporalBurst,
)

logger = logging.getLogger("freecher_worker")

ACTIVITY_V1_1_FORMULA_VERSION = "activity_v1_1_formula_v1"
ACTIVITY_BIN_SECONDS = 1.0
BURST_WINDOW_SECONDS = 2.0
BURST_MIN_SEPARATION_SECONDS = 3.0
AUDIO_WINDOW_MS = 50
DEFAULT_SILENCE_THRESHOLD = 0.01  # -40 dB relative to full scale
MOTION_DIMS = (160, 90)
SCENE_CHANGE_THRESHOLD = 0.35


def compute_combined_activity(
    audio_delta: float,
    audio_energy: float,
    visual_motion: float,
    scene_change: bool,
) -> float:
    """Deterministic formula for combined multimodal activity.

    Formula: activity_v1_1_formula_v1
        combined_activity = audio_delta * 0.35
                          + audio_energy * 0.25
                          + visual_motion * 0.30
                          + scene_change_signal * 0.10
    """
    sc_signal = 1.0 if scene_change else 0.0
    val = (
        audio_delta * 0.35
        + audio_energy * 0.25
        + visual_motion * 0.30
        + sc_signal * 0.10
    )
    return round(float(np.clip(val, 0.0, 1.0)), 4)


def _compute_source_audio_timeline(
    wav_path: Path | str,
    bin_seconds: float = ACTIVITY_BIN_SECONDS,
) -> Tuple[List[float], List[float], List[float], float]:
    """Compute 1-second binned audio energy, audio delta, and speech activity across source WAV.

    Returns:
        (energies_norm, deltas_norm, speech_activities, duration_seconds)
    """
    path = Path(wav_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        total_frames = wf.getnframes()
        duration_seconds = total_frames / float(sample_rate)

        raw_bytes = wf.readframes(total_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if n_channels > 1 and len(samples) > 0:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    bin_samples = max(1, int(sample_rate * bin_seconds))
    subwin_samples = max(1, int(sample_rate * (AUDIO_WINDOW_MS / 1000.0)))
    total_bins = max(1, int(np.ceil(len(samples) / bin_samples)))

    raw_energies: List[float] = []
    speech_acts: List[float] = []

    for b in range(total_bins):
        start_idx = b * bin_samples
        end_idx = min(len(samples), (b + 1) * bin_samples)
        chunk = samples[start_idx:end_idx]

        if len(chunk) == 0:
            raw_energies.append(0.0)
            speech_acts.append(0.0)
            continue

        rms = float(np.sqrt(np.mean(chunk**2)))
        raw_energies.append(rms)

        # Speech activity: fraction of 50ms subwindows above silence threshold
        n_subwins = len(chunk) // subwin_samples
        if n_subwins > 0:
            sub_chunks = chunk[: n_subwins * subwin_samples].reshape(n_subwins, subwin_samples)
            sub_rms = np.sqrt(np.mean(sub_chunks**2, axis=1))
            speech_ratio = float(np.mean(sub_rms >= DEFAULT_SILENCE_THRESHOLD))
        else:
            speech_ratio = 1.0 if rms >= DEFAULT_SILENCE_THRESHOLD else 0.0
        speech_acts.append(round(speech_ratio, 4))

    # Normalize audio energy
    max_energy = max(raw_energies) if raw_energies else 1.0
    norm_factor = max_energy if max_energy > 1e-5 else 1.0
    energies_norm = [round(float(np.clip(e / norm_factor, 0.0, 1.0)), 4) for e in raw_energies]

    # Compute deltas
    deltas: List[float] = []
    for i in range(len(energies_norm)):
        if i == 0:
            deltas.append(0.0)
        else:
            deltas.append(abs(energies_norm[i] - energies_norm[i - 1]))

    max_delta = max(deltas) if deltas else 1.0
    delta_factor = max_delta if max_delta > 1e-5 else 1.0
    deltas_norm = [round(float(np.clip(d / delta_factor, 0.0, 1.0)), 4) for d in deltas]

    return energies_norm, deltas_norm, speech_acts, duration_seconds


def _compute_source_visual_timeline(
    video_path: Path | str,
    duration_seconds: float,
    decoder_name: Optional[str] = None,
    bin_seconds: float = ACTIVITY_BIN_SECONDS,
) -> Tuple[List[float], List[bool]]:
    """Compute 1-second binned visual motion and scene cut flags using software FFmpeg decode.

    Returns:
        (motions_norm, scene_changes)
    """
    total_bins = max(1, int(np.ceil(duration_seconds / bin_seconds)))
    default_motions = [0.0] * total_bins
    default_scenes = [False] * total_bins

    src = Path(video_path).resolve()
    if not src.is_file():
        return default_motions, default_scenes

    w, h = MOTION_DIMS
    frame_bytes_len = w * h

    cmd = ["ffmpeg", "-loglevel", "error", "-y"]
    if decoder_name:
        cmd.extend(["-c:v", decoder_name])
    cmd.extend([
        "-i", str(src),
        "-vf", f"fps=1/{bin_seconds},scale={w}:{h}",
        "-f", "image2pipe",
        "-vcodec", "rawvideo",
        "-pix_fmt", "gray",
        "-",
    ])

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        raw_frames: List[np.ndarray] = []
        while len(raw_frames) < total_bins:
            chunk = proc.stdout.read(frame_bytes_len)
            if not chunk or len(chunk) < frame_bytes_len:
                break
            raw_frames.append(np.frombuffer(chunk, dtype=np.uint8).astype(np.float32) / 255.0)

        proc.stdout.close()
        proc.wait(timeout=10.0)

        if not raw_frames:
            return default_motions, default_scenes

        motions: List[float] = [0.0]
        scenes: List[bool] = [False]

        for i in range(1, len(raw_frames)):
            mad = float(np.mean(np.abs(raw_frames[i] - raw_frames[i - 1])))
            motions.append(round(min(1.0, mad), 4))
            scenes.append(bool(mad >= SCENE_CHANGE_THRESHOLD))

        while len(motions) < total_bins:
            motions.append(0.0)
            scenes.append(False)

        return motions[:total_bins], scenes[:total_bins]

    except Exception as exc:
        logger.warning(f"[multimodal-activity] Visual timeline extraction error: {exc}")
        return default_motions, default_scenes


def compute_source_temporal_activity_profile(
    wav_path: Path | str,
    video_path: Path | str,
    source_fingerprint: str,
    cache_file: Optional[Path | str] = None,
    decoder_name: Optional[str] = None,
    force_recompute: bool = False,
) -> SourceTemporalActivityProfile:
    """Build and cache whole-source temporal activity profile once.

    Scans the source file only once; candidates subsequently slice this profile.
    """
    if cache_file and Path(cache_file).is_file() and not force_recompute:
        try:
            cached_data = load_json(cache_file)
            profile = SourceTemporalActivityProfile.model_validate(cached_data)
            if (
                profile.source_fingerprint == source_fingerprint
                and profile.formula_version == ACTIVITY_V1_1_FORMULA_VERSION
            ):
                logger.debug(
                    f"[multimodal-activity] Loaded cached source activity profile ({len(profile.timeline)} bins)"
                )
                return profile
        except Exception as exc:
            logger.warning(f"[multimodal-activity] Failed to load cached activity profile: {exc}")

    logger.info("[multimodal-activity] Computing single-pass source temporal activity profile...")

    energies, deltas, speeches, duration = _compute_source_audio_timeline(
        wav_path=wav_path,
        bin_seconds=ACTIVITY_BIN_SECONDS,
    )

    motions, scenes = _compute_source_visual_timeline(
        video_path=video_path,
        duration_seconds=duration,
        decoder_name=decoder_name,
        bin_seconds=ACTIVITY_BIN_SECONDS,
    )

    total_bins = max(len(energies), len(motions))
    timeline: List[SourceTemporalActivityPoint] = []

    for b in range(total_bins):
        ts = round(b * ACTIVITY_BIN_SECONDS, 2)
        e = energies[b] if b < len(energies) else 0.0
        d = deltas[b] if b < len(deltas) else 0.0
        sp = speeches[b] if b < len(speeches) else 0.0
        m = motions[b] if b < len(motions) else 0.0
        sc = scenes[b] if b < len(scenes) else False

        comb = compute_combined_activity(
            audio_delta=d,
            audio_energy=e,
            visual_motion=m,
            scene_change=sc,
        )

        pt = SourceTemporalActivityPoint(
            absolute_timestamp=ts,
            audio_energy=e,
            audio_delta=d,
            speech_activity=sp,
            visual_motion=m,
            scene_change=sc,
            combined_activity=comb,
        )
        timeline.append(pt)

    profile = SourceTemporalActivityProfile(
        source_fingerprint=source_fingerprint,
        formula_version=ACTIVITY_V1_1_FORMULA_VERSION,
        bin_size_seconds=ACTIVITY_BIN_SECONDS,
        duration_seconds=duration,
        timeline=timeline,
    )

    if cache_file:
        c_path = Path(cache_file)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(profile, c_path)
        logger.info(
            f"[multimodal-activity] Cached source activity profile ({len(timeline)} bins) to {c_path}"
        )

    return profile


def slice_candidate_activity_curve(
    source_profile: SourceTemporalActivityProfile,
    candidate_start: float,
    candidate_duration: float,
) -> ActivityCurveSummary:
    """Slice cached source activity profile into candidate-relative 1-second curve and find peaks."""
    candidate_end = candidate_start + candidate_duration
    sliced_points: List[ActivityPoint] = []

    for pt in source_profile.timeline:
        if candidate_start <= pt.absolute_timestamp < candidate_end:
            rel_off = round(pt.absolute_timestamp - candidate_start, 2)
            sliced_points.append(
                ActivityPoint(
                    offset=rel_off,
                    absolute_timestamp=pt.absolute_timestamp,
                    audio_energy=pt.audio_energy,
                    audio_delta=pt.audio_delta,
                    speech_activity=pt.speech_activity,
                    visual_motion=pt.visual_motion,
                    scene_change=pt.scene_change,
                    combined_activity=pt.combined_activity,
                )
            )

    if not sliced_points:
        sliced_points.append(
            ActivityPoint(
                offset=round(candidate_duration / 2.0, 2),
                absolute_timestamp=round(candidate_start + candidate_duration / 2.0, 2),
                audio_energy=0.5,
                audio_delta=0.5,
                speech_activity=0.5,
                visual_motion=0.0,
                scene_change=False,
                combined_activity=0.5,
            )
        )

    sorted_by_audio = sorted(
        sliced_points,
        key=lambda p: (p.audio_delta + p.audio_energy),
        reverse=True,
    )
    top_audio_peaks = [p.offset for p in sorted_by_audio[:3]]

    sorted_by_motion = sorted(
        sliced_points,
        key=lambda p: p.visual_motion,
        reverse=True,
    )
    top_motion_peaks = [p.offset for p in sorted_by_motion[:3]]

    sorted_by_comb = sorted(
        sliced_points,
        key=lambda p: p.combined_activity,
        reverse=True,
    )
    top_comb_peaks = [p.offset for p in sorted_by_comb[:3]]

    return ActivityCurveSummary(
        curve=sliced_points,
        top_audio_peaks=top_audio_peaks,
        top_motion_peaks=top_motion_peaks,
        top_combined_activity_peaks=top_comb_peaks,
    )


def select_temporal_burst_peaks(
    activity_summary: ActivityCurveSummary,
    candidate_duration: float,
) -> List[TemporalBurst]:
    """Select 2 separated candidate-local activity burst regions and provenance metadata."""
    curve = activity_summary.curve
    if not curve:
        center_1 = round(candidate_duration * 0.35, 2)
        center_2 = round(candidate_duration * 0.70, 2)
        return [
            _create_burst(1, center_1, "uniform_fallback", 0.5, 1, candidate_duration, 0.5, 0.0, False),
            _create_burst(2, center_2, "uniform_fallback", 0.5, 2, candidate_duration, 0.5, 0.0, False),
        ]

    ranked_pts = sorted(curve, key=lambda p: (p.combined_activity, p.audio_delta), reverse=True)

    pt_1 = ranked_pts[0]
    reason_1 = _determine_burst_reason(pt_1)

    pt_2: Optional[ActivityPoint] = None
    for pt in ranked_pts[1:]:
        if abs(pt.offset - pt_1.offset) >= BURST_MIN_SEPARATION_SECONDS:
            pt_2 = pt
            break

    if pt_2 is None:
        for pt in ranked_pts[1:]:
            if abs(pt.offset - pt_1.offset) >= 2.0:
                pt_2 = pt
                break

    if pt_2 is None:
        target_off = pt_1.offset + 4.0 if (pt_1.offset + 4.0 < candidate_duration) else max(0.5, pt_1.offset - 4.0)
        pt_2 = min(curve, key=lambda p: abs(p.offset - target_off))

    reason_2 = _determine_burst_reason(pt_2)

    selected_pairs = [(pt_1, reason_1, 1), (pt_2, reason_2, 2)]
    selected_pairs.sort(key=lambda item: item[0].offset)

    bursts: List[TemporalBurst] = []
    for idx, (pt, reason, orig_rank) in enumerate(selected_pairs, start=1):
        burst = _create_burst(
            burst_index=idx,
            center_offset=pt.offset,
            selection_reason=reason,
            combined_activity=pt.combined_activity,
            activity_rank=orig_rank,
            candidate_duration=candidate_duration,
            audio_energy=pt.audio_energy,
            motion=pt.visual_motion,
            has_scene_change=pt.scene_change,
        )
        bursts.append(burst)

    return bursts


def _determine_burst_reason(pt: ActivityPoint) -> str:
    """Determine provenance tag for a selected peak."""
    if pt.scene_change:
        return "scene_change_proximity"
    if pt.audio_delta >= 0.6 and pt.audio_delta >= pt.visual_motion:
        return "audio_delta_peak"
    if pt.visual_motion >= 0.5 and pt.visual_motion >= pt.audio_delta:
        return "visual_motion_peak"
    if pt.audio_energy >= 0.7:
        return "audio_energy_peak"
    return "combined_activity_peak"


def _create_burst(
    burst_index: int,
    center_offset: float,
    selection_reason: str,
    combined_activity: float,
    activity_rank: int,
    candidate_duration: float,
    audio_energy: float,
    motion: float,
    has_scene_change: bool,
) -> TemporalBurst:
    """Construct TemporalBurst object bounded by candidate duration."""
    half_window = BURST_WINDOW_SECONDS / 2.0
    start_off = max(0.0, round(center_offset - half_window, 2))
    end_off = min(round(candidate_duration, 2), round(center_offset + half_window, 2))

    return TemporalBurst(
        burst_index=burst_index,
        center_offset=round(center_offset, 2),
        start_offset=start_off,
        end_offset=end_off,
        selection_reason=selection_reason,
        combined_activity=round(combined_activity, 4),
        activity_rank=activity_rank,
        transcript="",
        frames=[],
        audio_energy_mean=round(audio_energy, 4),
        motion_mean=round(motion, 4),
        has_scene_change=has_scene_change,
    )
