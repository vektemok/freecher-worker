"""Dynamic Subclip Refinement — pick the strongest self-contained fragment inside a candidate.

A ranked candidate window is an *analysis* window (typically ~60 s). The published short is a
sub-span of it, chosen so that it starts on the hook, keeps the setup only when the payoff needs
it, contains the payoff, and ends on a completed thought.

Durations are never quantized: the selected length is whatever the boundary search produced
(11.4 s, 18.7 s, 24.2 s, 31.6 s, 42.0 s are all normal outcomes).

This stage runs strictly *after* ranking and does not read or modify any scorer.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from freecher_worker.transcription.models import Transcript

from .signals import SignalCurve
from .timeframe import CandidateTimeframe, OffsetRegion

logger = logging.getLogger("freecher_worker")

SUBCLIP_FORMULA_VERSION = "subclip_v1_formula_v1"

DURATION_MODE_AUTO = "auto"
DURATION_MODE_FULL = "full"
AVAILABLE_DURATION_MODES = (DURATION_MODE_AUTO, DURATION_MODE_FULL)

#: Speech separated by more than this is treated as a phrase break.
PHRASE_GAP_SEC = 0.35
#: Speech separated by more than this starts a new narrative run (setup boundary).
RUN_GAP_SEC = 1.2

_TERMINAL_RE = re.compile(r"[.!?…]+[\"'»)\]]*\s*$")

#: Component weights of subclip_v1_formula_v1. Inactive components are renormalized away so
#: candidates with and without an advisory region stay comparable.
_WEIGHTS: Dict[str, float] = {
    "hook": 0.22,
    "payoff": 0.20,
    "payoff_coverage": 0.10,
    "mean_activity": 0.10,
    "completeness": 0.12,
    "setup_preserved": 0.08,
    "duration_prior": 0.06,
    "advisory_overlap": 0.12,
}
_PENALTY_WEIGHTS: Dict[str, float] = {
    "boring_lead": 12.0,
    "boring_tail": 8.0,
}


class SubclipConfig(BaseModel):
    """Configurable bounds and shaping parameters for subclip selection."""

    min_duration_sec: float = Field(default=8.0, gt=0.0)
    target_min_duration_sec: float = Field(default=15.0, gt=0.0)
    target_max_duration_sec: float = Field(default=30.0, gt=0.0)
    max_duration_sec: float = Field(default=45.0, gt=0.0)
    hook_window_sec: float = Field(default=3.0, gt=0.0)
    tail_window_sec: float = Field(default=3.0, gt=0.0)
    pre_roll_sec: float = Field(default=0.15, ge=0.0)
    post_roll_sec: float = Field(default=0.30, ge=0.0)
    boring_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    grid_step_sec: float = Field(default=0.5, gt=0.0)

    def clamped(self) -> "SubclipConfig":
        """Return a copy with the duration bounds forced into a consistent order."""
        min_d = max(0.5, self.min_duration_sec)
        max_d = max(min_d, self.max_duration_sec)
        t_min = min(max(self.target_min_duration_sec, min_d), max_d)
        t_max = min(max(self.target_max_duration_sec, t_min), max_d)
        return self.model_copy(
            update={
                "min_duration_sec": min_d,
                "max_duration_sec": max_d,
                "target_min_duration_sec": t_min,
                "target_max_duration_sec": t_max,
            }
        )


class SpeechUnit(BaseModel):
    """A transcript phrase clipped to the candidate window, in candidate-relative offsets."""

    index: int
    start_offset_sec: float
    end_offset_sec: float
    text: str = ""
    terminal: bool = Field(default=False, description="Ends on sentence-final punctuation")
    gap_after_sec: float = Field(default=0.0, description="Silence until the next unit")
    run_index: int = Field(default=0, description="Index of the uninterrupted narrative run")


class SubclipSelection(BaseModel):
    """The fragment chosen inside a candidate, in both coordinate systems."""

    candidate_id: str
    duration_mode: str
    formula_version: str = SUBCLIP_FORMULA_VERSION
    boundary_source: str = Field(description="speech_units | grid | full_candidate")

    source_start_sec: float = Field(description="Absolute candidate start")
    source_end_sec: float = Field(description="Absolute candidate end")
    candidate_duration_sec: float

    start_offset_sec: float = Field(description="Short start relative to candidate start")
    end_offset_sec: float = Field(description="Short end relative to candidate start")
    short_source_start_sec: float = Field(description="Absolute short start")
    short_source_end_sec: float = Field(description="Absolute short end")
    duration_sec: float

    score: float = Field(description="subclip_v1_formula_v1 score, 0-100")
    components: Dict[str, float] = Field(default_factory=dict)
    penalties: Dict[str, float] = Field(default_factory=dict)
    reason: str = ""
    signal_source: str = ""
    advisory_used: bool = False
    advisory_interpretation: Optional[str] = None
    evaluated_windows: int = 0


def build_speech_units(transcript: Optional[Transcript], timeframe: CandidateTimeframe) -> List[SpeechUnit]:
    """Clip transcript segments to the candidate window and express them as offsets."""
    if transcript is None or not transcript.segments:
        return []

    raw: List[Tuple[float, float, str]] = []
    for seg in transcript.segments:
        start = max(seg.start, timeframe.source_start_sec)
        end = min(seg.end, timeframe.source_end_sec)
        if end - start < 0.15:
            continue
        raw.append(
            (
                start - timeframe.source_start_sec,
                end - timeframe.source_start_sec,
                seg.text.strip(),
            )
        )

    raw.sort(key=lambda item: item[0])

    units: List[SpeechUnit] = []
    run_index = 0
    for idx, (start, end, text) in enumerate(raw):
        gap_after = (raw[idx + 1][0] - end) if idx + 1 < len(raw) else 0.0
        if idx > 0 and (start - raw[idx - 1][1]) > RUN_GAP_SEC:
            run_index += 1
        units.append(
            SpeechUnit(
                index=idx,
                start_offset_sec=round(timeframe.validate_offset(start, "unit start"), 3),
                end_offset_sec=round(timeframe.validate_offset(end, "unit end"), 3),
                text=text,
                terminal=bool(_TERMINAL_RE.search(text)) or gap_after >= RUN_GAP_SEC,
                gap_after_sec=round(max(0.0, gap_after), 3),
                run_index=run_index,
            )
        )
    return units


def _duration_prior(duration: float, cfg: SubclipConfig) -> float:
    """Trapezoid preference for the target band that never snaps a duration to a round number."""
    if cfg.target_min_duration_sec <= duration <= cfg.target_max_duration_sec:
        return 1.0
    if duration < cfg.target_min_duration_sec:
        span = max(1e-6, cfg.target_min_duration_sec - cfg.min_duration_sec)
        frac = (duration - cfg.min_duration_sec) / span
    else:
        span = max(1e-6, cfg.max_duration_sec - cfg.target_max_duration_sec)
        frac = (cfg.max_duration_sec - duration) / span
    return round(0.35 + 0.65 * min(max(frac, 0.0), 1.0), 4)


def _overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Intersection over union of two spans."""
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return round(inter / union, 4) if union > 1e-9 else 0.0


def find_payoff_block(curve: SignalCurve, peak_offset: float, boring_threshold: float) -> Tuple[float, float]:
    """Contiguous run of above-threshold bins containing the peak.

    This is the span the short must actually contain: cutting before its end is what leaves the
    payoff out of the clip, which is exactly the failure Dynamic Subclip Refinement exists to fix.
    """
    if not curve.combined_norm:
        return 0.0, 0.0

    threshold = max(boring_threshold, 0.5 * max(curve.combined_norm))
    peak_idx = min(
        range(len(curve.offsets)),
        key=lambda i: (abs(curve.offsets[i] - peak_offset), i),
    )

    lo = peak_idx
    while lo - 1 >= 0 and curve.combined_norm[lo - 1] >= threshold:
        lo -= 1
    hi = peak_idx
    while hi + 1 < len(curve.combined_norm) and curve.combined_norm[hi + 1] >= threshold:
        hi += 1

    return curve.offsets[lo], curve.offsets[hi] + curve.bin_seconds


def resolve_payoff_block(
    signal_block: Tuple[float, float],
    advisory: Optional[OffsetRegion],
    candidate_duration: float,
) -> Tuple[float, float]:
    """Reconcile the signal-derived payoff span with the multimodal model's advisory span.

    When both point at the same stretch their union is taken, so neither the model's semantic
    read nor the local activity evidence is thrown away. When they disagree outright, the
    advisory wins: locating the payoff is precisely what the multimodal pass is good at, and it
    has already survived strict timestamp validation by this point.
    """
    if advisory is None:
        return signal_block

    a_start, a_end = advisory.start_offset_sec, advisory.end_offset_sec
    s_start, s_end = signal_block
    overlaps = min(s_end, a_end) > max(s_start, a_start)
    if overlaps and s_end > s_start:
        start, end = min(s_start, a_start), max(s_end, a_end)
    else:
        start, end = a_start, a_end
    return max(0.0, start), min(candidate_duration, end)


def _boundary_sets(
    units: List[SpeechUnit], duration: float, cfg: SubclipConfig
) -> Tuple[List[float], List[float], str]:
    """Derive legal start/end offsets, preferring phrase boundaries over a blind grid."""
    if units:
        starts = {0.0}
        ends = {round(duration, 3)}
        for unit in units:
            starts.add(round(max(0.0, unit.start_offset_sec - cfg.pre_roll_sec), 3))
            ends.add(round(min(duration, unit.end_offset_sec + cfg.post_roll_sec), 3))
        return sorted(starts), sorted(ends), "speech_units"

    steps = int(duration / cfg.grid_step_sec) + 1
    grid = sorted({round(min(i * cfg.grid_step_sec, duration), 3) for i in range(steps + 1)})
    return grid, grid, "grid"


def _describe(components: Dict[str, float], penalties: Dict[str, float], units_hit: Tuple[bool, bool]) -> str:
    parts = [
        f"hook={components.get('hook', 0.0):.2f}",
        f"payoff={components.get('payoff', 0.0):.2f}",
        f"payoff_coverage={components.get('payoff_coverage', 0.0):.2f}",
        f"activity={components.get('mean_activity', 0.0):.2f}",
        f"completeness={components.get('completeness', 0.0):.2f}",
    ]
    if components.get("setup_preserved", 0.0) >= 0.99:
        parts.append("setup preserved")
    if components.get("advisory_overlap") is not None and components.get("advisory_overlap", 0.0) > 0.0:
        parts.append(f"advisory_iou={components['advisory_overlap']:.2f}")
    if penalties.get("boring_lead", 0.0) > 0.0:
        parts.append(f"trimmed dull lead-in (penalty {penalties['boring_lead']:.1f})")
    if penalties.get("boring_tail", 0.0) > 0.0:
        parts.append(f"dead air at tail (penalty {penalties['boring_tail']:.1f})")
    start_snap, end_snap = units_hit
    parts.append("starts on phrase onset" if start_snap else "starts mid-phrase")
    parts.append("ends on completed phrase" if end_snap else "ends mid-phrase")
    return "; ".join(parts)


def _score_window(
    start: float,
    end: float,
    curve: SignalCurve,
    units: List[SpeechUnit],
    cfg: SubclipConfig,
    advisory: Optional[OffsetRegion],
    peak_offset: float,
    setup_run_start: Optional[float],
    candidate_duration: float,
    payoff_block: Tuple[float, float],
) -> Tuple[float, Dict[str, float], Dict[str, float], Tuple[bool, bool]]:
    """Evaluate one candidate window with ``subclip_v1_formula_v1``."""
    duration = end - start

    hook_end = min(end, start + cfg.hook_window_sec)
    hook = curve.mean_in(start, hook_end)

    late_start = start + 0.4 * duration
    payoff_peak = curve.max_in(late_start, end)
    peak_inside = 1.0 if (start - 1e-6) <= peak_offset <= (end + 1e-6) else 0.0
    payoff = 0.7 * payoff_peak + 0.3 * peak_inside

    mean_activity = curve.mean_in(start, end)

    block_start, block_end = payoff_block
    block_len = block_end - block_start
    if block_len <= 1e-6:
        payoff_coverage = 0.5
    else:
        covered = max(0.0, min(end, block_end) - max(start, block_start))
        payoff_coverage = min(1.0, covered / block_len)

    start_snapped = any(abs(start - max(0.0, u.start_offset_sec - cfg.pre_roll_sec)) < 0.06 for u in units)
    end_unit = next(
        (
            u
            for u in units
            if abs(end - min(candidate_duration, u.end_offset_sec + cfg.post_roll_sec)) < 0.06
        ),
        None,
    )
    if end_unit is not None and end_unit.terminal:
        end_quality = 1.0
    elif end_unit is not None and end_unit.gap_after_sec >= PHRASE_GAP_SEC:
        end_quality = 0.65
    elif end_unit is not None:
        end_quality = 0.4
    else:
        end_quality = 0.0 if units else 0.5
    start_quality = 1.0 if start_snapped else (0.3 if units else 0.5)
    completeness = 0.4 * start_quality + 0.6 * end_quality

    if setup_run_start is None:
        setup_preserved = 0.5
    elif start <= setup_run_start + 0.25:
        setup_preserved = 1.0
    elif start < peak_offset:
        setup_preserved = 0.5
    else:
        setup_preserved = 0.0

    components: Dict[str, float] = {
        "hook": round(hook, 4),
        "payoff": round(payoff, 4),
        "payoff_coverage": round(payoff_coverage, 4),
        "mean_activity": round(mean_activity, 4),
        "completeness": round(completeness, 4),
        "setup_preserved": round(setup_preserved, 4),
        "duration_prior": _duration_prior(duration, cfg),
    }
    if advisory is not None:
        components["advisory_overlap"] = _overlap_ratio(
            start, end, advisory.start_offset_sec, advisory.end_offset_sec
        )

    active_weight = sum(_WEIGHTS[name] for name in components)
    positive = sum(_WEIGHTS[name] * value for name, value in components.items()) / active_weight

    boring_lead = curve.low_activity_ratio(start, hook_end, cfg.boring_threshold)
    boring_tail = curve.low_activity_ratio(max(start, end - cfg.tail_window_sec), end, cfg.boring_threshold)
    penalties = {
        "boring_lead": round(_PENALTY_WEIGHTS["boring_lead"] * boring_lead, 4),
        "boring_tail": round(_PENALTY_WEIGHTS["boring_tail"] * boring_tail, 4),
    }

    score = 100.0 * positive - penalties["boring_lead"] - penalties["boring_tail"]
    return round(min(max(score, 0.0), 100.0), 4), components, penalties, (start_snapped, end_unit is not None)


def _full_candidate_selection(
    timeframe: CandidateTimeframe,
    cfg: SubclipConfig,
    curve: SignalCurve,
    duration_mode: str,
    reason: str,
    boundary_source: str = "full_candidate",
) -> SubclipSelection:
    end = min(timeframe.duration_sec, cfg.max_duration_sec)
    return SubclipSelection(
        candidate_id=timeframe.candidate_id,
        duration_mode=duration_mode,
        boundary_source=boundary_source,
        source_start_sec=round(timeframe.source_start_sec, 3),
        source_end_sec=round(timeframe.source_end_sec, 3),
        candidate_duration_sec=round(timeframe.duration_sec, 3),
        start_offset_sec=0.0,
        end_offset_sec=round(end, 3),
        short_source_start_sec=timeframe.to_source(0.0),
        short_source_end_sec=timeframe.to_source(end),
        duration_sec=round(end, 3),
        score=round(100.0 * curve.mean_in(0.0, end), 4),
        components={},
        penalties={},
        reason=reason,
        signal_source=curve.source,
    )


def refine_subclip(
    timeframe: CandidateTimeframe,
    curve: SignalCurve,
    transcript: Optional[Transcript] = None,
    config: Optional[SubclipConfig] = None,
    advisory: Optional[OffsetRegion] = None,
    advisory_interpretation: Optional[str] = None,
    duration_mode: str = DURATION_MODE_AUTO,
    units: Optional[List[SpeechUnit]] = None,
) -> SubclipSelection:
    """Select the best sub-span of a ranked candidate.

    Args:
        timeframe: Absolute placement of the candidate; the only source of absolute timestamps.
        curve: Candidate-local activity curve.
        transcript: Full transcript, used to derive phrase boundaries.
        config: Duration bounds and shaping parameters.
        advisory: Validated candidate-relative advisory region, or None.
        advisory_interpretation: How the advisory region was classified, for diagnostics.
        duration_mode: ``auto`` searches for the best fragment, ``full`` keeps the whole candidate.
        units: Precomputed speech units (derived from ``transcript`` when omitted).

    Returns:
        A :class:`SubclipSelection` carrying both offsets and absolute timestamps.
    """
    if duration_mode not in AVAILABLE_DURATION_MODES:
        raise ValueError(f"Unknown duration_mode '{duration_mode}'. Available: {AVAILABLE_DURATION_MODES}")

    cfg = (config or SubclipConfig()).clamped()
    duration = timeframe.duration_sec

    if duration_mode == DURATION_MODE_FULL:
        return _full_candidate_selection(
            timeframe, cfg, curve, duration_mode, "duration_mode=full: whole candidate kept"
        )

    if duration <= cfg.min_duration_sec:
        return _full_candidate_selection(
            timeframe,
            cfg,
            curve,
            duration_mode,
            f"candidate is {duration:.2f}s, at or below the {cfg.min_duration_sec:.1f}s minimum; kept whole",
        )

    speech_units = units if units is not None else build_speech_units(transcript, timeframe)
    starts, ends, boundary_source = _boundary_sets(speech_units, duration, cfg)

    peak_offset = curve.peak_offset()
    setup_run_start: Optional[float] = None
    if speech_units:
        peak_unit = next(
            (u for u in speech_units if u.start_offset_sec <= peak_offset <= u.end_offset_sec),
            None,
        )
        if peak_unit is None:
            peak_unit = min(speech_units, key=lambda u: abs(u.start_offset_sec - peak_offset))
        run_units = [u for u in speech_units if u.run_index == peak_unit.run_index]
        if run_units:
            setup_run_start = max(0.0, run_units[0].start_offset_sec - cfg.pre_roll_sec)

    payoff_block = resolve_payoff_block(
        find_payoff_block(curve, peak_offset, cfg.boring_threshold), advisory, duration
    )

    best: Optional[Tuple[Tuple[float, float, float], SubclipSelection]] = None
    evaluated = 0

    for start in starts:
        for end in ends:
            window = end - start
            if window < cfg.min_duration_sec - 1e-6 or window > cfg.max_duration_sec + 1e-6:
                continue
            evaluated += 1
            score, components, penalties, snaps = _score_window(
                start,
                end,
                curve,
                speech_units,
                cfg,
                advisory,
                peak_offset,
                setup_run_start,
                duration,
                payoff_block,
            )
            # Deterministic ordering: highest score, then shortest, then earliest.
            key = (-score, round(window, 3), round(start, 3))
            if best is None or key < best[0]:
                selection = SubclipSelection(
                    candidate_id=timeframe.candidate_id,
                    duration_mode=duration_mode,
                    boundary_source=boundary_source,
                    source_start_sec=round(timeframe.source_start_sec, 3),
                    source_end_sec=round(timeframe.source_end_sec, 3),
                    candidate_duration_sec=round(duration, 3),
                    start_offset_sec=round(start, 3),
                    end_offset_sec=round(end, 3),
                    short_source_start_sec=timeframe.to_source(start, "short start offset"),
                    short_source_end_sec=timeframe.to_source(end, "short end offset"),
                    duration_sec=round(window, 3),
                    score=score,
                    components=components,
                    penalties=penalties,
                    reason=_describe(components, penalties, snaps),
                    signal_source=curve.source,
                    advisory_used=advisory is not None,
                    advisory_interpretation=advisory_interpretation,
                )
                best = (key, selection)

    if best is None:
        # No phrase-aligned window fits the duration bounds; fall back to a blind grid search.
        if boundary_source != "grid":
            logger.info(
                f"[subclip] No phrase-aligned window fits [{cfg.min_duration_sec}, {cfg.max_duration_sec}]s "
                f"for {timeframe.candidate_id}; falling back to grid search"
            )
            return refine_subclip(
                timeframe=timeframe,
                curve=curve,
                transcript=transcript,
                config=cfg,
                advisory=advisory,
                advisory_interpretation=advisory_interpretation,
                duration_mode=duration_mode,
                units=[],
            )
        return _full_candidate_selection(
            timeframe,
            cfg,
            curve,
            duration_mode,
            "no window satisfied the duration bounds; candidate kept and clamped to the maximum",
        )

    selection = best[1]
    selection.evaluated_windows = evaluated
    return selection
