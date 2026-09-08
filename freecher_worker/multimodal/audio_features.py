"""Audio energy and activity feature extraction for multimodal highlight reranking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import wave

import numpy as np

from freecher_worker.utils.json_io import load_json, save_json
from .models import AudioFeatures, SourceAudioProfile

logger = logging.getLogger("freecher_worker")

WINDOW_MS = 50  # 50ms analysis window
DEFAULT_SILENCE_THRESHOLD = 0.01  # -40 dB relative to full scale


def _read_wav_slice(
    wav_path: Path | str,
    start_sec: float,
    duration_sec: float,
) -> Tuple[np.ndarray, int]:
    """Read a segment of mono/stereo 16-bit PCM WAV as float32 array in [-1, 1].

    Returns:
        (samples_float32, sample_rate)
    """
    path = Path(wav_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        total_frames = wf.getnframes()

        start_frame = max(0, int(start_sec * sample_rate))
        if start_frame >= total_frames:
            return np.zeros(0, dtype=np.float32), sample_rate

        frames_to_read = max(0, int(duration_sec * sample_rate))
        if start_frame + frames_to_read > total_frames:
            frames_to_read = total_frames - start_frame

        wf.setpos(start_frame)
        raw_bytes = wf.readframes(frames_to_read)

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

    return samples, sample_rate


def _compute_windowed_rms(
    samples: np.ndarray,
    sample_rate: int,
    window_ms: int = WINDOW_MS,
) -> np.ndarray:
    """Compute RMS energy across non-overlapping window_ms windows."""
    if len(samples) == 0:
        return np.zeros(0, dtype=np.float32)

    win_len = max(1, int(sample_rate * (window_ms / 1000.0)))
    n_wins = len(samples) // win_len
    if n_wins == 0:
        # Less than one window, compute RMS of available samples
        return np.array([np.sqrt(np.mean(samples**2))], dtype=np.float32)

    windows = samples[: n_wins * win_len].reshape(n_wins, win_len)
    rms = np.sqrt(np.mean(windows**2, axis=1))
    return rms.astype(np.float32)


def estimate_rms_percentile(mean_rms: float, percentiles: Dict[str, float]) -> float:
    """Estimate the percentile [0, 1] of candidate mean RMS relative to source distribution."""
    if not percentiles:
        return 0.5

    percentile_keys = ["p10", "p25", "p50", "p75", "p90", "p95", "p99"]
    xs = [0.0]
    ys = [0.0]

    for k in percentile_keys:
        if k in percentiles:
            val = float(percentiles[k])
            pct = float(k[1:]) / 100.0
            xs.append(val)
            ys.append(pct)

    # Append upper ceiling
    max_val = max(xs[-1] * 1.5, 1.0)
    xs.append(max_val)
    ys.append(1.0)

    # Sort strictly ascending
    sorted_pairs = sorted(zip(xs, ys), key=lambda p: p[0])
    xs_sorted = [p[0] for p in sorted_pairs]
    ys_sorted = [p[1] for p in sorted_pairs]

    interpolated = float(np.interp(mean_rms, xs_sorted, ys_sorted))
    return round(float(np.clip(interpolated, 0.0, 1.0)), 4)


def compute_source_audio_profile(
    wav_path: Path | str,
    source_fingerprint: str,
    cache_file: Optional[Path | str] = None,
) -> SourceAudioProfile:
    """Compute single-pass source-wide audio energy profile and cache it.

    If cache_file exists and is valid, returns the cached profile without rescanning.

    Args:
        wav_path: Path to source audio WAV.
        source_fingerprint: Fingerprint identifier of the media.
        cache_file: Optional path to persist/load the profile JSON.

    Returns:
        SourceAudioProfile containing whole-source RMS percentiles.
    """
    if cache_file:
        c_path = Path(cache_file)
        if c_path.is_file():
            try:
                data = load_json(c_path)
                cached = SourceAudioProfile.model_validate(data)
                if cached.source_fingerprint == source_fingerprint:
                    logger.debug(f"[multimodal-audio] Loaded cached source audio profile from {c_path}")
                    return cached
            except Exception as exc:
                logger.warning(f"[multimodal-audio] Failed reading cached profile at {c_path}: {exc}")

    w_path = Path(wav_path).resolve()
    if not w_path.is_file():
        raise FileNotFoundError(f"Source WAV file not found: {w_path}")

    with wave.open(str(w_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        total_frames = wf.getnframes()
        duration_seconds = total_frames / float(sample_rate) if sample_rate > 0 else 0.0

        # Read in chunks of 60 seconds to conserve memory
        chunk_frames = sample_rate * 60
        all_rms_list = []

        while True:
            raw_bytes = wf.readframes(chunk_frames)
            if not raw_bytes:
                break

            if sampwidth == 2:
                chunk_samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                chunk_samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                chunk_samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if n_channels > 1 and len(chunk_samples) > 0:
                chunk_samples = chunk_samples.reshape(-1, n_channels).mean(axis=1)

            rms_chunk = _compute_windowed_rms(chunk_samples, sample_rate, WINDOW_MS)
            if len(rms_chunk) > 0:
                all_rms_list.append(rms_chunk)

    if all_rms_list:
        all_rms = np.concatenate(all_rms_list)
        percentile_values = np.percentile(all_rms, [10, 25, 50, 75, 90, 95, 99])
        rms_percentiles = {
            "p10": round(float(percentile_values[0]), 6),
            "p25": round(float(percentile_values[1]), 6),
            "p50": round(float(percentile_values[2]), 6),
            "p75": round(float(percentile_values[3]), 6),
            "p90": round(float(percentile_values[4]), 6),
            "p95": round(float(percentile_values[5]), 6),
            "p99": round(float(percentile_values[6]), 6),
        }
    else:
        rms_percentiles = {
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    profile = SourceAudioProfile(
        source_fingerprint=source_fingerprint,
        sample_rate=sample_rate,
        duration_seconds=round(duration_seconds, 2),
        rms_percentiles=rms_percentiles,
    )

    if cache_file:
        c_path = Path(cache_file)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(profile, c_path)
        logger.info(f"[multimodal-audio] Cached source audio profile saved to {c_path}")

    return profile


def extract_candidate_audio_features(
    wav_path: Path | str,
    start: float,
    end: float,
    source_profile: Optional[SourceAudioProfile] = None,
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
) -> AudioFeatures:
    """Extract locally computed audio energy signals for a candidate highlight window."""
    duration = max(0.1, end - start)
    samples, sample_rate = _read_wav_slice(wav_path, start, duration)

    if len(samples) == 0:
        return AudioFeatures(
            rms_mean=0.0,
            rms_std=0.0,
            peak=0.0,
            silence_ratio=1.0,
            speech_coverage=0.0,
            energy_change_rate=0.0,
            energy_percentile=0.0 if source_profile else None,
            beginning_rms=0.0,
            middle_rms=0.0,
            ending_rms=0.0,
        )

    peak = round(float(np.max(np.abs(samples))), 4)
    rms_windows = _compute_windowed_rms(samples, sample_rate, WINDOW_MS)

    if len(rms_windows) == 0:
        rms_mean = 0.0
        rms_std = 0.0
        silence_ratio = 1.0
        speech_coverage = 0.0
        energy_change_rate = 0.0
        beg_rms = 0.0
        mid_rms = 0.0
        end_rms = 0.0
    else:
        rms_mean = round(float(np.mean(rms_windows)), 6)
        rms_std = round(float(np.std(rms_windows)), 6)
        silent_count = int(np.sum(rms_windows < silence_threshold))
        silence_ratio = round(float(silent_count / len(rms_windows)), 4)
        speech_coverage = round(float(np.clip(1.0 - silence_ratio, 0.0, 1.0)), 4)

        if len(rms_windows) > 1:
            energy_change_rate = round(float(np.mean(np.abs(np.diff(rms_windows)))), 6)
        else:
            energy_change_rate = 0.0

        # Three temporal thirds
        n = len(rms_windows)
        third = max(1, n // 3)
        beg_rms = round(float(np.mean(rms_windows[:third])), 6)
        mid_rms = round(float(np.mean(rms_windows[third : 2 * third])), 6)
        end_rms = round(float(np.mean(rms_windows[2 * third :])), 6)

    energy_pct = None
    if source_profile and source_profile.rms_percentiles:
        energy_pct = estimate_rms_percentile(rms_mean, source_profile.rms_percentiles)

    return AudioFeatures(
        rms_mean=rms_mean,
        rms_std=rms_std,
        peak=peak,
        silence_ratio=silence_ratio,
        speech_coverage=speech_coverage,
        energy_change_rate=energy_change_rate,
        energy_percentile=energy_pct,
        beginning_rms=beg_rms,
        middle_rms=mid_rms,
        ending_rms=end_rms,
    )
