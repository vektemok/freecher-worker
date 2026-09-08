"""Candidate-local temporal signal curve used by Dynamic Subclip Refinement.

The curve is intentionally cheap and offline. Three sources are supported, in order of
preference:

1. the cached whole-source activity profile written by the multimodal reranker
   (``multimodal/cache/source_temporal_activity_profile_v1_1.json``),
2. the run's extracted ``audio.wav`` (speech energy only),
3. the transcript alone (speech coverage and speaking rate).

No source is required: the transcript-only path keeps refinement fully deterministic and
runnable without FFmpeg, which is what the unit tests exercise.
"""

from __future__ import annotations

import logging
import math
import wave
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field

from freecher_worker.transcription.models import Transcript
from freecher_worker.utils.json_io import load_json

from .timeframe import CandidateTimeframe

logger = logging.getLogger("freecher_worker")

SIGNAL_CURVE_VERSION = "subclip_signal_v1"
DEFAULT_BIN_SECONDS = 1.0

SOURCE_ACTIVITY_PROFILE = "source_activity_profile"
SOURCE_AUDIO_WAV = "audio_wav"
SOURCE_TRANSCRIPT = "transcript_only"
SOURCE_FLAT = "flat"


class SignalCurve(BaseModel):
    """Binned candidate-relative activity signals, normalized within the candidate."""

    version: str = Field(default=SIGNAL_CURVE_VERSION)
    source: str = Field(description="Which signal source produced this curve")
    bin_seconds: float = Field(default=DEFAULT_BIN_SECONDS, description="Bin width in seconds")
    offsets: List[float] = Field(default_factory=list, description="Bin start offsets relative to candidate start")
    speech: List[float] = Field(default_factory=list, description="Speech coverage per bin [0, 1]")
    energy: List[float] = Field(default_factory=list, description="Audio energy per bin [0, 1]")
    motion: List[float] = Field(default_factory=list, description="Visual motion per bin [0, 1]")
    scene_change: List[bool] = Field(default_factory=list, description="Scene cut flag per bin")
    combined: List[float] = Field(default_factory=list, description="Raw combined activity per bin [0, 1]")
    combined_norm: List[float] = Field(default_factory=list, description="Combined activity min-max normalized per candidate")

    def _bin_index(self, offset_sec: float) -> int:
        if not self.offsets:
            return 0
        idx = int(math.floor(offset_sec / self.bin_seconds))
        return min(max(idx, 0), len(self.offsets) - 1)

    def _slice(self, start_offset_sec: float, end_offset_sec: float) -> List[float]:
        if not self.combined_norm:
            return []
        lo = self._bin_index(start_offset_sec)
        hi = self._bin_index(max(start_offset_sec, end_offset_sec - 1e-6))
        return self.combined_norm[lo : hi + 1]

    def mean_in(self, start_offset_sec: float, end_offset_sec: float) -> float:
        """Mean normalized activity over a span, 0.5 when no signal is available."""
        values = self._slice(start_offset_sec, end_offset_sec)
        return float(sum(values) / len(values)) if values else 0.5

    def max_in(self, start_offset_sec: float, end_offset_sec: float) -> float:
        """Peak normalized activity over a span, 0.5 when no signal is available."""
        values = self._slice(start_offset_sec, end_offset_sec)
        return float(max(values)) if values else 0.5

    def low_activity_ratio(self, start_offset_sec: float, end_offset_sec: float, threshold: float) -> float:
        """Fraction of bins in a span whose normalized activity is below ``threshold``."""
        values = self._slice(start_offset_sec, end_offset_sec)
        if not values:
            return 0.0
        return float(sum(1 for v in values if v < threshold) / len(values))

    def peak_offset(self) -> float:
        """Offset of the strongest bin in the candidate."""
        if not self.combined_norm:
            return 0.0
        best_idx = max(range(len(self.combined_norm)), key=lambda i: (self.combined_norm[i], -i))
        return self.offsets[best_idx]

    def scene_cut_offsets(self) -> List[float]:
        """Offsets of bins flagged as scene cuts."""
        return [off for off, cut in zip(self.offsets, self.scene_change) if cut]


def _normalize(values: List[float]) -> List[float]:
    """Min-max normalize a series into [0, 1]; a flat series maps to 0.5."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [round((v - lo) / (hi - lo), 4) for v in values]


def _finalize(
    source: str,
    bin_seconds: float,
    offsets: List[float],
    speech: List[float],
    energy: List[float],
    motion: List[float],
    scene_change: List[bool],
    combined: List[float],
) -> SignalCurve:
    return SignalCurve(
        source=source,
        bin_seconds=bin_seconds,
        offsets=[round(o, 3) for o in offsets],
        speech=[round(v, 4) for v in speech],
        energy=[round(v, 4) for v in energy],
        motion=[round(v, 4) for v in motion],
        scene_change=scene_change,
        combined=[round(v, 4) for v in combined],
        combined_norm=_normalize(combined),
    )


def build_curve_from_activity_profile(
    profile_data: dict,
    timeframe: CandidateTimeframe,
    bin_seconds: float = DEFAULT_BIN_SECONDS,
) -> Optional[SignalCurve]:
    """Slice the cached whole-source activity profile into a candidate-relative curve."""
    timeline = profile_data.get("timeline") or []
    if not timeline:
        return None

    offsets: List[float] = []
    speech: List[float] = []
    energy: List[float] = []
    motion: List[float] = []
    cuts: List[bool] = []
    combined: List[float] = []

    for point in timeline:
        abs_ts = float(point.get("absolute_timestamp", 0.0))
        if not (timeframe.source_start_sec <= abs_ts < timeframe.source_end_sec):
            continue
        offsets.append(abs_ts - timeframe.source_start_sec)
        speech.append(float(point.get("speech_activity", 0.0)))
        energy.append(float(point.get("audio_energy", 0.0)))
        motion.append(float(point.get("visual_motion", 0.0)))
        cuts.append(bool(point.get("scene_change", False)))
        combined.append(float(point.get("combined_activity", 0.0)))

    if len(offsets) < 2:
        return None

    return _finalize(
        SOURCE_ACTIVITY_PROFILE,
        float(profile_data.get("bin_size_seconds", bin_seconds)),
        offsets,
        speech,
        energy,
        motion,
        cuts,
        combined,
    )


def build_curve_from_wav(
    wav_path: Path | str,
    timeframe: CandidateTimeframe,
    bin_seconds: float = DEFAULT_BIN_SECONDS,
) -> Optional[SignalCurve]:
    """Build a speech-energy curve from a 16 kHz mono WAV covering the whole source."""
    path = Path(wav_path)
    if not path.is_file():
        return None

    try:
        with wave.open(str(path), "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            total_frames = wf.getnframes()

            start_frame = int(timeframe.source_start_sec * sample_rate)
            end_frame = min(total_frames, int(timeframe.source_end_sec * sample_rate))
            if start_frame >= end_frame:
                return None

            wf.setpos(start_frame)
            raw = wf.readframes(end_frame - start_frame)
    except Exception as exc:  # pragma: no cover - corrupt WAV is not worth failing a render over
        logger.warning(f"[subclip-signals] Unable to read {path}: {exc}")
        return None

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sample_width, np.int16)
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if dtype is np.uint8:
        samples = (samples - 128.0) / 128.0
    elif dtype is np.int16:
        samples = samples / 32768.0
    else:
        samples = samples / 2147483648.0

    if n_channels > 1 and samples.size:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    if samples.size == 0:
        return None

    bin_samples = max(1, int(sample_rate * bin_seconds))
    n_bins = max(1, int(math.ceil(samples.size / bin_samples)))
    if n_bins < 2:
        return None

    offsets: List[float] = []
    energy: List[float] = []
    speech: List[float] = []
    for b in range(n_bins):
        chunk = samples[b * bin_samples : (b + 1) * bin_samples]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(chunk**2)))
        offsets.append(b * bin_seconds)
        energy.append(rms)
        # 50 ms sub-window voicing ratio as a cheap speech-presence proxy.
        sub = max(1, int(sample_rate * 0.05))
        n_sub = chunk.size // sub
        if n_sub > 0:
            sub_rms = np.sqrt(np.mean(chunk[: n_sub * sub].reshape(n_sub, sub) ** 2, axis=1))
            speech.append(float(np.mean(sub_rms >= 0.01)))
        else:
            speech.append(1.0 if rms >= 0.01 else 0.0)

    peak = max(energy) or 1.0
    energy_norm = [e / peak for e in energy]
    deltas = [0.0] + [abs(energy_norm[i] - energy_norm[i - 1]) for i in range(1, len(energy_norm))]
    combined = [
        min(1.0, 0.55 * energy_norm[i] + 0.30 * speech[i] + 0.15 * min(1.0, deltas[i] * 4.0))
        for i in range(len(energy_norm))
    ]

    return _finalize(
        SOURCE_AUDIO_WAV,
        bin_seconds,
        offsets,
        speech,
        energy_norm,
        [0.0] * len(offsets),
        [False] * len(offsets),
        combined,
    )


def build_curve_from_transcript(
    transcript: Transcript,
    timeframe: CandidateTimeframe,
    bin_seconds: float = DEFAULT_BIN_SECONDS,
) -> SignalCurve:
    """Build a speech-coverage and speaking-rate curve from transcript segments alone."""
    duration = timeframe.duration_sec
    n_bins = max(1, int(math.ceil(duration / bin_seconds)))
    offsets = [b * bin_seconds for b in range(n_bins)]

    coverage = [0.0] * n_bins
    chars = [0.0] * n_bins

    for seg in transcript.segments:
        seg_start = max(seg.start, timeframe.source_start_sec)
        seg_end = min(seg.end, timeframe.source_end_sec)
        if seg_end <= seg_start:
            continue
        seg_len = seg.end - seg.start
        char_rate = (len(seg.text) / seg_len) if seg_len > 1e-6 else 0.0

        rel_start = seg_start - timeframe.source_start_sec
        rel_end = seg_end - timeframe.source_start_sec
        first = min(int(rel_start / bin_seconds), n_bins - 1)
        last = min(int(math.ceil(rel_end / bin_seconds)) - 1, n_bins - 1)
        for b in range(first, last + 1):
            bin_lo = b * bin_seconds
            bin_hi = bin_lo + bin_seconds
            overlap = max(0.0, min(rel_end, bin_hi) - max(rel_start, bin_lo))
            if overlap <= 0.0:
                continue
            coverage[b] = min(1.0, coverage[b] + overlap / bin_seconds)
            chars[b] += char_rate * overlap

    max_chars = max(chars) if chars else 0.0
    rate_norm = [(c / max_chars) if max_chars > 1e-9 else 0.0 for c in chars]
    combined = [min(1.0, 0.65 * coverage[b] + 0.35 * rate_norm[b]) for b in range(n_bins)]

    return _finalize(
        SOURCE_TRANSCRIPT,
        bin_seconds,
        offsets,
        coverage,
        rate_norm,
        [0.0] * n_bins,
        [False] * n_bins,
        combined,
    )


def build_flat_curve(timeframe: CandidateTimeframe, bin_seconds: float = DEFAULT_BIN_SECONDS) -> SignalCurve:
    """Featureless curve used when no signal source is available at all."""
    n_bins = max(1, int(math.ceil(timeframe.duration_sec / bin_seconds)))
    offsets = [b * bin_seconds for b in range(n_bins)]
    return _finalize(
        SOURCE_FLAT,
        bin_seconds,
        offsets,
        [0.5] * n_bins,
        [0.5] * n_bins,
        [0.0] * n_bins,
        [False] * n_bins,
        [0.5] * n_bins,
    )


def load_signal_curve(
    timeframe: CandidateTimeframe,
    transcript: Optional[Transcript] = None,
    run_dir: Optional[Path] = None,
    bin_seconds: float = DEFAULT_BIN_SECONDS,
) -> SignalCurve:
    """Resolve the best available candidate-local signal curve."""
    if run_dir is not None:
        profile_path = (
            Path(run_dir) / "multimodal" / "cache" / "source_temporal_activity_profile_v1_1.json"
        )
        if profile_path.is_file():
            try:
                curve = build_curve_from_activity_profile(load_json(profile_path), timeframe, bin_seconds)
                if curve is not None:
                    return curve
            except Exception as exc:
                logger.warning(f"[subclip-signals] Failed to use cached activity profile: {exc}")

        wav_path = Path(run_dir) / "audio.wav"
        if wav_path.is_file():
            try:
                curve = build_curve_from_wav(wav_path, timeframe, bin_seconds)
                if curve is not None:
                    return curve
            except Exception as exc:
                logger.warning(f"[subclip-signals] Failed to use run audio.wav: {exc}")

    if transcript is not None and transcript.segments:
        return build_curve_from_transcript(transcript, timeframe, bin_seconds)

    return build_flat_curve(timeframe, bin_seconds)
