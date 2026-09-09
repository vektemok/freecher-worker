"""Blends a text score with prosody, because the two see different things.

Measured on the frozen gold set (cset_72b311f2cb41727f, 146/146 labeled):

    text alone   (logprob binary, 146 LLM calls)   rho +0.201
    audio alone  (rms_std, no API calls at all)    rho +0.232
    0.4 audio + 0.6 text                           rho +0.307

The blend is worth more than either part because the parts barely overlap:
text and audio rank the same candidates with a mutual correlation of +0.072.
`rms_std` is the spread of loudness inside a window rather than its level --
laughter, a shout, a sudden drop before a punchline -- which is exactly what a
transcript cannot carry. Notably `silence_ratio` and `speech_coverage` were
near zero (+/-0.043): what matters is the dynamics of speech, not its presence.
This mirrors the audio contribution reported for engagement prediction in
arXiv:2508.02516.

Two honest limits on that +0.307. The weight was chosen on the same 146
candidates it is measured on, so it is an optimistic estimate until a second
source is scored. And audio alone is *worse* than text on top-K (5 vs 8 strong
in the top 30) despite the higher correlation -- it belongs here as a second
opinion, not as a replacement.

Blending happens over ranks, not raw values, so a bounded 0-100 text score and
an unbounded RMS spread can be combined without one dominating by scale. That
makes the scorer inherently batch-wise: a single candidate has no rank.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence

from freecher_worker.highlights.models import CandidateWindow, HighlightScore
from freecher_worker.multimodal.audio_features import (
    compute_source_audio_profile,
    extract_candidate_audio_features,
)
from freecher_worker.scoring.base import HighlightScorer

logger = logging.getLogger("freecher_worker")

AUDIO_TEXT_SCORER_VERSION = "audio_text_v1"

#: The blend measured above. 0.4-0.5 was a plateau rather than a spike, so the
#: exact value is not load-bearing; the midpoint of the plateau is taken.
DEFAULT_AUDIO_WEIGHT = 0.4

#: The single audio feature that carried the most signal. Kept configurable
#: because it was selected on one source and may not be the best on another.
DEFAULT_AUDIO_FEATURE = "rms_std"

WAV_SAMPLE_RATE = 16_000


class AudioTextScoringError(Exception):
    """Raised when the audio side of the blend cannot be produced."""


def ensure_wav(
    audio_path: Path | str,
    destination: Optional[Path | str] = None,
    *,
    ffmpeg_path: str = "ffmpeg",
) -> Path:
    """Return a 16 kHz mono WAV for `audio_path`, decoding it once if needed.

    The feature extractor reads WAV frames directly; the ingest artifact is
    m4a. Decoding the whole two-hour source takes about a second and is done
    once per run rather than once per candidate.
    """
    source = Path(audio_path)
    if not source.is_file():
        raise AudioTextScoringError(f"audio artifact not found: {source}")
    if source.suffix.lower() == ".wav":
        return source

    target = Path(destination) if destination else source.with_suffix(".wav")
    if target.is_file() and target.stat().st_size > 0:
        logger.info("reusing decoded audio at %s", target)
        return target

    if shutil.which(ffmpeg_path) is None:
        raise AudioTextScoringError(f"'{ffmpeg_path}' not found; install ffmpeg to score audio")

    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg_path, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-ac", "1", "-ar", str(WAV_SAMPLE_RATE),
            "-c:a", "pcm_s16le", str(target),
        ],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 or not target.is_file():
        raise AudioTextScoringError(
            f"could not decode {source} to WAV: {result.stderr.decode('utf-8', 'ignore')[:200]}"
        )
    return target


def rank_fractions(values: Sequence[float]) -> list[float]:
    """Map values onto 0..1 by rank, averaging ties.

    Rank space is what makes the blend meaningful: the text score is bounded
    0-100 and heavily tied at its floor, while RMS spread is unbounded and
    continuous. Averaging them raw would let whichever has the wider numeric
    range decide the outcome.
    """
    count = len(values)
    if count == 0:
        return []
    if count == 1:
        return [0.5]

    order = sorted(range(count), key=lambda index: values[index])
    ranks = [0.0] * count
    position = 0
    while position < count:
        end = position
        while end + 1 < count and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return [rank / (count - 1) for rank in ranks]


class AudioTextScorer(HighlightScorer):
    """Ranks candidates by a rank-space blend of a text score and prosody."""

    def __init__(
        self,
        wav_path: Path | str,
        *,
        text_scores: Optional[dict[str, float]] = None,
        text_scorer: Optional[HighlightScorer] = None,
        audio_weight: float = DEFAULT_AUDIO_WEIGHT,
        audio_feature: str = DEFAULT_AUDIO_FEATURE,
        source_fingerprint: str = "unknown",
        profile_cache: Optional[Path | str] = None,
    ) -> None:
        if not 0.0 <= audio_weight <= 1.0:
            raise ValueError(f"audio_weight must be in [0, 1], got {audio_weight}")
        if text_scores is None and text_scorer is None and audio_weight < 1.0:
            raise ValueError(
                "a text side is required unless audio_weight is 1.0; "
                "pass text_scores or text_scorer"
            )

        self.name = "audio_text"
        self.version = AUDIO_TEXT_SCORER_VERSION
        self.wav_path = Path(wav_path)
        self.text_scores = text_scores
        self.text_scorer = text_scorer
        self.audio_weight = audio_weight
        self.audio_feature = audio_feature
        self.source_fingerprint = source_fingerprint
        self.profile_cache = profile_cache

    def score(
        self,
        candidate: CandidateWindow,
        context: Optional[dict[str, Any]] = None,
    ) -> HighlightScore:
        """Not available: a rank blend needs the whole candidate set."""
        raise NotImplementedError(
            "AudioTextScorer ranks candidates against each other, so it has no "
            "meaningful score for one candidate in isolation. Use score_batch()."
        )

    def audio_values(self, candidates: Sequence[CandidateWindow]) -> list[float]:
        """Extract the chosen audio feature for every candidate window."""
        profile = compute_source_audio_profile(
            self.wav_path, self.source_fingerprint, cache_file=self.profile_cache
        )
        values: list[float] = []
        for candidate in candidates:
            features = extract_candidate_audio_features(
                self.wav_path, candidate.start, candidate.end, source_profile=profile
            )
            value = getattr(features, self.audio_feature, None)
            if value is None:
                raise AudioTextScoringError(
                    f"audio feature '{self.audio_feature}' is not available; "
                    f"expected one of {sorted(features.model_dump())}"
                )
            values.append(float(value))
        return values

    def text_values(
        self,
        candidates: Sequence[CandidateWindow],
        transcript: Optional[Any],
    ) -> list[float]:
        """The text side, either handed in precomputed or produced on the spot."""
        if self.text_scores is not None:
            missing = [c.id for c in candidates if c.id not in self.text_scores]
            if missing:
                raise AudioTextScoringError(
                    f"{len(missing)} candidate(s) have no text score, e.g. {missing[:3]}"
                )
            return [float(self.text_scores[c.id]) for c in candidates]
        assert self.text_scorer is not None  # guarded in __init__
        return [s.score for s in self.text_scorer.score_batch(list(candidates), transcript)]

    def score_batch(
        self,
        candidates: list[CandidateWindow],
        transcript: Optional[Any] = None,
    ) -> list[HighlightScore]:
        if not candidates:
            return []

        audio = rank_fractions(self.audio_values(candidates))
        if self.audio_weight >= 1.0:
            blended = audio
            text_ranks = [0.0] * len(candidates)
        else:
            text_ranks = rank_fractions(self.text_values(candidates, transcript))
            weight = self.audio_weight
            blended = [weight * a + (1.0 - weight) * t for a, t in zip(audio, text_ranks)]

        logger.info(
            "[audio_text] blended %d candidates (%.0f%% audio '%s', %.0f%% text)",
            len(candidates), self.audio_weight * 100, self.audio_feature,
            (1.0 - self.audio_weight) * 100,
        )

        scores: list[HighlightScore] = []
        for candidate, value, audio_rank, text_rank in zip(candidates, blended, audio, text_ranks):
            points = round(value * 100.0, 4)
            scores.append(
                HighlightScore(
                    score=points,
                    hook_score=points,
                    standalone_score=points,
                    emotion_score=points,
                    information_score=points,
                    shareability_score=points,
                    reason=(
                        f"blend={points:.1f} "
                        f"(audio rank {audio_rank:.2f} x {self.audio_weight:.2f}, "
                        f"text rank {text_rank:.2f} x {1.0 - self.audio_weight:.2f})"
                    ),
                    fallback_used=False,
                    fallback_reason=None,
                    final_score=points,
                    scorer_version=self.version,
                    subscores={
                        "audio_rank": round(audio_rank, 4),
                        "text_rank": round(text_rank, 4),
                        "audio_weight": self.audio_weight,
                    },
                )
            )
        return scores
