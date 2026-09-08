"""Cheap reaction/energy evidence derived from already-computed pipeline data.

Nothing here runs a new model or decodes media. The only input is the source temporal
activity profile the multimodal stage already cached. Signals that genuinely require
diarization (interruption, overlapping speech) are left as explicit ``None`` extension
points rather than guessed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from freecher_worker.multimodal.models import SourceTemporalActivityProfile
from freecher_worker.utils.json_io import load_json

from .models import ReactionSignals

logger = logging.getLogger("freecher_worker")

ACTIVITY_PROFILE_RELPATH = (
    "multimodal/cache/source_temporal_activity_profile_v1_1.json"
)

#: A bin is "loud" when its energy exceeds the candidate mean by this margin.
EXCITED_MARGIN = 0.12
#: A bin counts as silent below this energy with no speech activity.
SILENCE_ENERGY = 0.10
#: An energy delta at or above this is a sudden change.
SUDDEN_DELTA = 0.25


def load_source_activity_profile(
    run_dir: Path | str,
) -> Optional[SourceTemporalActivityProfile]:
    """Load the cached whole-source activity profile, or None when it does not exist."""
    path = Path(run_dir).resolve() / ACTIVITY_PROFILE_RELPATH
    if not path.is_file():
        return None
    try:
        return SourceTemporalActivityProfile.model_validate(load_json(path))
    except Exception as exc:  # noqa: BLE001 - an optional cache never breaks scoring
        logger.warning(f"[contextual-signals] Could not load activity profile: {exc}")
        return None


def extract_reaction_signals(
    profile: Optional[SourceTemporalActivityProfile],
    start: float,
    end: float,
) -> ReactionSignals:
    """Derive candidate-local reaction signals from the cached activity timeline."""
    if profile is None or not profile.timeline:
        return ReactionSignals(
            available=False,
            source="none",
            notes=[
                "No cached source activity profile; run multimodal-score first to enable "
                "reaction signals."
            ],
        )

    points = [pt for pt in profile.timeline if start <= pt.absolute_timestamp < end]
    if len(points) < 2:
        return ReactionSignals(
            available=False,
            source="source_temporal_activity_profile_v1_1",
            notes=["Fewer than two activity bins overlap this candidate."],
        )

    points.sort(key=lambda p: p.absolute_timestamp)
    n = len(points)
    energies = [p.audio_energy for p in points]
    deltas = [p.audio_delta for p in points]
    speech = [p.speech_activity for p in points]
    mean_energy = sum(energies) / n

    excited_bins = [
        i for i, e in enumerate(energies) if e >= mean_energy + EXCITED_MARGIN
    ]
    excited_ratio = len(excited_bins) / n

    # Laughter-like: a short high-energy burst with speech present and a sharp onset.
    laughter_bins = [
        i
        for i in range(n)
        if deltas[i] >= SUDDEN_DELTA and energies[i] >= mean_energy and speech[i] > 0.0
    ]
    laughter_activity = min(1.0, len(laughter_bins) / max(1.0, n / 4.0))

    sudden_increase = 0.0
    for i in range(1, n):
        sudden_increase = max(sudden_increase, energies[i] - energies[i - 1])
    sudden_increase = max(0.0, min(1.0, sudden_increase))

    silent_bins = [
        i for i in range(n) if energies[i] <= SILENCE_ENERGY and speech[i] <= 0.05
    ]
    dead_air_ratio = len(silent_bins) / n

    pause_then_reaction = any(
        i + 1 < n
        and energies[i] <= SILENCE_ENERGY
        and energies[i + 1] >= mean_energy + EXCITED_MARGIN
        for i in silent_bins
    )

    chain = 0
    longest_chain = 0
    for d in deltas:
        if d >= SUDDEN_DELTA:
            chain += 1
            longest_chain = max(longest_chain, chain)
        else:
            chain = 0
    rapid_chain = min(1.0, longest_chain / 4.0)

    ranked = sorted(points, key=lambda p: p.combined_activity, reverse=True)[:3]
    peak_offsets: List[float] = [round(p.absolute_timestamp - start, 2) for p in ranked]

    return ReactionSignals(
        available=True,
        source="source_temporal_activity_profile_v1_1",
        laughter_like_activity=round(laughter_activity, 3),
        sudden_energy_increase=round(sudden_increase, 3),
        excited_speech_ratio=round(excited_ratio, 3),
        pause_then_reaction=pause_then_reaction,
        rapid_reaction_chain=round(rapid_chain, 3),
        dead_air_ratio=round(dead_air_ratio, 3),
        peak_offsets=peak_offsets,
        interruption=None,
        overlapping_speech=None,
        notes=[
            "Derived from cached 1s activity bins; treat as evidence, not as a verdict.",
            "interruption and overlapping_speech require speaker diarization (extension point).",
        ],
    )
