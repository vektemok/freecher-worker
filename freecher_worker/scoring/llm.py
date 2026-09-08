"""OpenAI-compatible LLM Highlight Scorer (highlight_v2) with decoupled feature extraction and versioned scoring."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional

import httpx

from freecher_worker.evaluation.models import ScoreDistributionDiagnostics
from freecher_worker.highlights.models import CandidateWindow, HighlightScore
from .base import HighlightScorer
from .heuristic import HeuristicScorer

logger = logging.getLogger("freecher_worker")

SCORER_VERSION_V2 = "highlight_v2"
PROMPT_VERSION_V2 = "highlight_v2_prompt_v1"
SCORE_FORMULA_VERSION_V2 = "highlight_v2_formula_v1"

SCORER_VERSION_V2_1 = "highlight_v2_1"
PROMPT_VERSION_V2_1 = "highlight_v2_1_prompt_v1"
SCORE_FORMULA_VERSION_V2_1 = "highlight_v2_1_formula_v1"

# Defaults maintain backward compatibility with v2
SCORER_VERSION = SCORER_VERSION_V2
PROMPT_VERSION = PROMPT_VERSION_V2
SCORE_FORMULA_VERSION = SCORE_FORMULA_VERSION_V2

SYSTEM_PROMPT_V2 = """You are a world-class viral short-form video editor and content strategist (TikTok, Instagram Reels, YouTube Shorts).
Your job is to critically evaluate whether a speech transcript segment from a long stream or video works as an independent, high-retention short clip.

CORE QUESTION:
"Would a person who does not know this creator or the surrounding stream stop scrolling and watch this to the end as an independent TikTok/Reel/Short?"

CONTEXT BOUNDARIES (CRITICAL):
You are provided with:
- CANDIDATE TEXT: The exact video segment under evaluation.
- PREVIOUS CONTEXT: Speech immediately preceding the candidate (up to 45s).
- NEXT CONTEXT: Speech immediately following the candidate (up to 45s).

RULE: Previous and following contexts are provided ONLY to help you understand the candidate.
DO NOT reward information, jokes, revelations, payoff, or punchlines that occur OUTSIDE the candidate boundaries.
If a setup or intrigue begins in the candidate but the payoff or answer only happens in NEXT CONTEXT:
you MUST set outside_payoff=true and setup_only=true, and NOT give credit for the outside payoff.

SURFACE SIGNALS WARNING:
DO NOT give a high score simply because the speaker:
- asks a rhetorical question ("Did you know?", "Why do 90% fail?")
- addresses the viewer ("Subscribe!", "Look at this!")
- uses loud emotional exclamations ("OMG!", "No way!")
- cites numbers or statistics without real substance.
These are weak surface tricks, not viral substance.

ROUTINE BANTER & LOGISTICS PENALTY:
STRONGLY PENALIZE and flag transitional=true / boringness=high for:
- mundane conversational filler (walking around talking casually without a point)
- stream logistics, scheduling, discussing future plans ("Let's check sound", "Can chat hear me?")
- greetings or goodbyes that lack a punchline or payoff
- reading chat questions or inside jokes with no standalone value
- wandering thoughts with no coherent narrative arc or payoff.

EVALUATION DIMENSIONS (0 to 100):
Positive dimensions:
- hook: Does the opening 3-5 seconds immediately arrest attention and create curiosity?
- standalone: Is it fully understandable without knowing the creator or watching the full stream?
- story_payoff: Does the narrative or argument reach a satisfying conclusion or punchline within the clip?
- emotion: Genuine emotional energy (shock, excitement, empathy, passion, tension).
- humor: Genuinely funny, witty, or entertaining.
- surprise: Unexpected twist, counter-intuitive insight, or shocking revelation.
- retention: Compelling pacing that makes viewer stay until the final second.
- shareability: Strong impulse to share with friends, comment, or save.

Negative dimensions (0 to 100):
- boringness: Monotonous delivery, repetitive chatter, rambling, or lack of dynamic energy.
- context_dependency: Requires outside stream knowledge, inside lore, or previous events to make sense.

Categorical flags (true/false):
- setup_only: true if this candidate is purely background buildup or premise without the payoff.
- transitional: true if this is just moving between scenes, stream chatter, soundcheck, or filler.
- outside_payoff: true if the actual payoff/answer happens in NEXT CONTEXT instead of inside the candidate.

Overall impression:
- llm_quality_score: 0-100 overall subjective rating of clip viability as a standalone short.
- reason: 1-2 concise sentences explaining your judgment.

OUTPUT FORMAT:
Reply ONLY with a raw JSON object matching this schema (no markdown, no backticks):
{
  "candidate_id": "<str>",
  "hook": <float 0-100>,
  "standalone": <float 0-100>,
  "story_payoff": <float 0-100>,
  "emotion": <float 0-100>,
  "humor": <float 0-100>,
  "surprise": <float 0-100>,
  "retention": <float 0-100>,
  "shareability": <float 0-100>,
  "boringness": <float 0-100>,
  "context_dependency": <float 0-100>,
  "setup_only": <bool>,
  "transitional": <bool>,
  "outside_payoff": <bool>,
  "llm_quality_score": <float 0-100>,
  "reason": "<str>"
}"""

PROMPT_HASH_V2 = hashlib.sha256(SYSTEM_PROMPT_V2.encode("utf-8")).hexdigest()[:16]

SYSTEM_PROMPT_V2_1 = """You are a world-class viral short-form video editor and streamer content strategist (TikTok, Instagram Reels, YouTube Shorts).
Your job is to evaluate whether a speech transcript segment from a stream or video works as an engaging short-form clip.

CORE QUESTION:
"Would a stranger who does not know this creator find this specific moment entertaining, surprising, funny, useful, emotionally engaging, or memorable enough to keep watching?"

MULTIPLE VALID HIGHLIGHT ARCHETYPES:
A short-form clip does NOT require a traditional narrative arc (exposition, conflict, resolution). Any of the following archetypes are fully valid and can score highly:
- story / anecdote / narrative progression
- joke / punchline / comedic timing
- absurd, witty, or funny streamer banter
- genuine, heightened reaction
- interpersonal conflict / argument / competitive tension
- unexpected surprise / shocking twist
- memorable, quotable line or strong opinion
- funny, awkward, or embarrassing moment
- impressive skill, clutch play, or demonstration
- genuinely useful or eye-opening insight
- unusual or spontaneous interaction
- visually or emotionally energetic moment.

DO NOT require every good candidate to contain:
- exposition or formal introductory setup
- traditional three-act story structure
- explicit moral or formal conclusion
- 100% complete standalone context.
Partial context dependency is entirely acceptable when the moment itself is funny, shocking, entertaining, or compelling.

NOISY ASR RESILIENCE:
The transcript was generated by automatic speech recognition (Whisper) and may contain transcription errors, phonetic mistakes, misheard slang/gamer terminology, and missing punctuation.
Do not interpret obvious ASR corruption, slang transcription mistakes, or punctuation errors as evidence that the underlying video is incoherent or boring. Judge the likely spoken interaction and communicative intent rather than grammatical transcript quality.

RECALIBRATED NEGATIVE DIMENSIONS:
- boringness (0-100):
  * 0-30: highly dynamic, punchy, funny, or eventful.
  * 35-60: normal casual dialogue or typical streamer banter.
  * 80-100: ONLY for genuinely low-event filler, silence, idle pauses, or unedited stream setup with zero entertainment value.
  Casual conversation, jokes, or lively streamer chat are NOT high boringness.
- context_dependency (0-100):
  * 0-30: completely standalone and instantly understood by any stranger.
  * 35-60: slight familiarity helps, but the core humor/emotion/insight works on its own.
  * 80-100: practically IMPOSSIBLE to appreciate without deep prior lore or watching the full stream.
  Casual streamer familiarity does NOT mean extreme context dependency.

CONTEXT BOUNDARIES (CRITICAL):
You are provided with:
- CANDIDATE TEXT: The exact video segment under evaluation.
- PREVIOUS CONTEXT: Speech immediately preceding the candidate (up to 45s).
- NEXT CONTEXT: Speech immediately following the candidate (up to 45s).

RULE: Previous and following contexts are provided ONLY to help you understand what is happening.
Evaluate ONLY what is contained within the CANDIDATE TEXT.
If a setup, intrigue, or joke begins in the candidate text, but the punchline, resolution, or answer only happens in NEXT CONTEXT:
you MUST set outside_payoff=true and setup_only=true, and NOT reward the clip for the payoff outside.

SURFACE SIGNALS WARNING:
Do not give high scores merely for cheap surface cues (rhetorical questions, yelling, exclamation marks, random numbers) unless real substance, humor, or entertainment is present.

CATEGORICAL FLAGS:
- setup_only: true if this candidate is purely a premise or background buildup without its own payoff or entertainment value.
- transitional: true if this is just moving between scenes, stream technical checks, sound checks, or mindless filler.
- outside_payoff: true if the actual payoff/answer happens in NEXT CONTEXT instead of inside the candidate.

OUTPUT FORMAT:
Reply ONLY with a raw JSON object matching this schema (no markdown, no backticks). ALL fields are strictly REQUIRED:
{
  "candidate_id": "<str>",
  "hook": <float 0-100>,
  "standalone": <float 0-100>,
  "story_payoff": <float 0-100>,
  "emotion": <float 0-100>,
  "humor": <float 0-100>,
  "surprise": <float 0-100>,
  "retention": <float 0-100>,
  "shareability": <float 0-100>,
  "boringness": <float 0-100>,
  "context_dependency": <float 0-100>,
  "setup_only": <bool>,
  "transitional": <bool>,
  "outside_payoff": <bool>,
  "llm_quality_score": <float 0-100>,
  "reason": "<str>"
}"""

PROMPT_HASH_V2_1 = hashlib.sha256(SYSTEM_PROMPT_V2_1.encode("utf-8")).hexdigest()[:16]

# Defaults alias to v2 for baseline preservation
SYSTEM_PROMPT = SYSTEM_PROMPT_V2
PROMPT_HASH = PROMPT_HASH_V2


def extract_surrounding_context(
    candidate: CandidateWindow,
    transcript: Optional[Any],
    context_window_seconds: float = 45.0,
) -> dict[str, Any]:
    """Extract surrounding transcript context and lightweight metadata for a candidate."""
    if transcript is None or not hasattr(transcript, "segments") or not transcript.segments:
        return {
            "previous_context": "",
            "next_context": "",
            "speech_rate_wpm": 0.0,
            "silence_before_sec": None,
            "silence_after_sec": None,
        }

    segments = transcript.segments
    cand_start = candidate.start
    cand_end = candidate.end

    prev_texts: List[str] = []
    next_texts: List[str] = []
    last_prev_end: Optional[float] = None
    first_next_start: Optional[float] = None

    for seg in segments:
        if seg.end <= cand_start:
            if cand_start - seg.start <= context_window_seconds:
                prev_texts.append(seg.text.strip())
            last_prev_end = seg.end
        elif seg.start >= cand_end:
            if seg.end - cand_end <= context_window_seconds:
                next_texts.append(seg.text.strip())
            if first_next_start is None:
                first_next_start = seg.start

    words = [w for w in candidate.text.split() if w]
    duration_min = max(0.1, candidate.duration / 60.0)
    wpm = round(len(words) / duration_min, 1)

    silence_before = round(cand_start - last_prev_end, 2) if last_prev_end is not None else None
    silence_after = round(first_next_start - cand_end, 2) if first_next_start is not None else None

    return {
        "previous_context": " ".join(prev_texts).strip(),
        "next_context": " ".join(next_texts).strip(),
        "speech_rate_wpm": wpm,
        "silence_before_sec": silence_before,
        "silence_after_sec": silence_after,
    }


def highlight_v2_score_formula_v1(features: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Deterministic, versioned formula combining positive dimensions and penalties for highlight_v2.

    Returns:
        (final_score, subscores_dict)
    """
    hook = float(features.get("hook", 50.0))
    standalone = float(features.get("standalone", 50.0))
    story_payoff = float(features.get("story_payoff", 50.0))
    emotion = float(features.get("emotion", 50.0))
    humor = float(features.get("humor", 50.0))
    surprise = float(features.get("surprise", 50.0))
    retention = float(features.get("retention", 50.0))
    shareability = float(features.get("shareability", 50.0))
    boringness = float(features.get("boringness", 50.0))
    context_dep = float(features.get("context_dependency", 50.0))
    llm_quality = float(features.get("llm_quality_score", 50.0))

    setup_only = bool(features.get("setup_only", False))
    transitional = bool(features.get("transitional", False))
    outside_payoff = bool(features.get("outside_payoff", False))

    # Clamping inputs strictly to [0.0, 100.0]
    hook = max(0.0, min(100.0, hook))
    standalone = max(0.0, min(100.0, standalone))
    story_payoff = max(0.0, min(100.0, story_payoff))
    emotion = max(0.0, min(100.0, emotion))
    humor = max(0.0, min(100.0, humor))
    surprise = max(0.0, min(100.0, surprise))
    retention = max(0.0, min(100.0, retention))
    shareability = max(0.0, min(100.0, shareability))
    boringness = max(0.0, min(100.0, boringness))
    context_dep = max(0.0, min(100.0, context_dep))
    llm_quality = max(0.0, min(100.0, llm_quality))

    # Positive weighted score
    pos_weights = {
        "story_payoff": 1.4,
        "retention": 1.4,
        "hook": 1.2,
        "standalone": 1.1,
        "shareability": 1.0,
        "humor": 0.8,
        "surprise": 0.8,
        "emotion": 0.7,
    }
    pos_sum = (
        story_payoff * pos_weights["story_payoff"]
        + retention * pos_weights["retention"]
        + hook * pos_weights["hook"]
        + standalone * pos_weights["standalone"]
        + shareability * pos_weights["shareability"]
        + humor * pos_weights["humor"]
        + surprise * pos_weights["surprise"]
        + emotion * pos_weights["emotion"]
    )
    total_pos_weight = sum(pos_weights.values())  # 8.4
    positive_score = pos_sum / total_pos_weight

    # Penalties
    penalty = 0.0
    if boringness > 35.0:
        penalty += (boringness - 35.0) * 0.7
    if context_dep > 35.0:
        penalty += (context_dep - 35.0) * 0.5
    if setup_only:
        penalty += 25.0
    if transitional:
        penalty += 30.0
    if outside_payoff:
        penalty += 20.0

    raw_score = max(0.0, positive_score - penalty)
    # Blend 75% calculated dimensions with 25% LLM overall impression
    blended = 0.75 * raw_score + 0.25 * llm_quality

    # Hard guardrail ceilings
    if transitional:
        blended = min(blended, 35.0)
    if setup_only:
        blended = min(blended, 40.0)
    if outside_payoff:
        blended = min(blended, 45.0)
    if boringness >= 75.0:
        blended = min(blended, 35.0)
    if context_dep >= 75.0:
        blended = min(blended, 40.0)

    final_score = round(max(0.0, min(100.0, blended)), 2)

    subscores = {
        "hook": hook,
        "hook_score": hook,
        "standalone": standalone,
        "standalone_score": standalone,
        "story_payoff": story_payoff,
        "emotion": emotion,
        "emotion_score": emotion,
        "humor": humor,
        "surprise": surprise,
        "retention": retention,
        "shareability": shareability,
        "shareability_score": shareability,
        "boringness": boringness,
        "context_dependency": context_dep,
        "llm_quality_score": llm_quality,
        "final_score": final_score,
    }
    return final_score, subscores


REQUIRED_FIELDS_V2_1 = (
    "hook",
    "standalone",
    "story_payoff",
    "emotion",
    "humor",
    "surprise",
    "retention",
    "shareability",
    "boringness",
    "context_dependency",
    "setup_only",
    "transitional",
    "outside_payoff",
    "reason",
)


def highlight_v2_1_formula_v1(
    features: dict[str, Any],
) -> tuple[float, dict[str, float], dict[str, Any]]:
    """Deterministic, versioned formula for highlight_v2_1.

    Calculates:
        1. positive_score = weighted sum of positive dimensions (weights sum to 1.00)
        2. additive negative penalty = boringness * 0.25 + context_dependency * 0.12
        3. additive boolean penalties = setup_only (-15) + transitional (-12)
        4. selective hard caps on raw_score:
            - setup_only and story_payoff < 30 -> <= 40
            - transitional and retention < 30 -> <= 35
            - boringness >= 85 and story_payoff < 25 -> <= 30
        5. clamp to [0.0, 100.0]

    Returns:
        (final_score, subscores, diagnostics)
    """
    hook = float(features["hook"])
    standalone = float(features["standalone"])
    story_payoff = float(features["story_payoff"])
    emotion = float(features["emotion"])
    humor = float(features["humor"])
    surprise = float(features["surprise"])
    retention = float(features["retention"])
    shareability = float(features["shareability"])
    boringness = float(features["boringness"])
    context_dep = float(features["context_dependency"])
    llm_quality = float(features.get("llm_quality_score", 50.0))

    setup_only = bool(features["setup_only"])
    transitional = bool(features["transitional"])
    outside_payoff = bool(features["outside_payoff"])

    # Clamping raw input dimensions strictly to [0.0, 100.0]
    hook = max(0.0, min(100.0, hook))
    standalone = max(0.0, min(100.0, standalone))
    story_payoff = max(0.0, min(100.0, story_payoff))
    emotion = max(0.0, min(100.0, emotion))
    humor = max(0.0, min(100.0, humor))
    surprise = max(0.0, min(100.0, surprise))
    retention = max(0.0, min(100.0, retention))
    shareability = max(0.0, min(100.0, shareability))
    boringness = max(0.0, min(100.0, boringness))
    context_dep = max(0.0, min(100.0, context_dep))
    llm_quality = max(0.0, min(100.0, llm_quality))

    # Positive score (0-100)
    positive_score = (
        hook * 0.12
        + standalone * 0.14
        + story_payoff * 0.18
        + emotion * 0.08
        + humor * 0.10
        + surprise * 0.08
        + retention * 0.18
        + shareability * 0.12
    )

    # Additive negative penalties
    base_penalty = boringness * 0.25 + context_dep * 0.12
    boolean_penalty = (15.0 if setup_only else 0.0) + (12.0 if transitional else 0.0)
    total_penalty = base_penalty + boolean_penalty

    raw_score = positive_score - total_penalty

    # Hard caps (applied sequentially if condition triggers and score exceeds cap)
    caps: list[tuple[str, float]] = []
    if setup_only and story_payoff < 30.0:
        caps.append(("setup_only_low_payoff", 40.0))
    if transitional and retention < 30.0:
        caps.append(("transitional_low_retention", 35.0))
    if boringness >= 85.0 and story_payoff < 25.0:
        caps.append(("extreme_boringness", 30.0))

    capped_score = raw_score
    applied_caps: list[str] = []
    for cap_name, cap_val in caps:
        capped_score = min(capped_score, cap_val)
        applied_caps.append(f"{cap_name}:{int(cap_val)}")

    final_score = round(max(0.0, min(100.0, capped_score)), 2)

    raw_pos = {
        "hook": hook,
        "standalone": standalone,
        "story_payoff": story_payoff,
        "emotion": emotion,
        "humor": humor,
        "surprise": surprise,
        "retention": retention,
        "shareability": shareability,
    }
    raw_neg = {
        "boringness": boringness,
        "context_dependency": context_dep,
    }

    subscores = {
        **raw_pos,
        "hook_score": hook,
        "standalone_score": standalone,
        "emotion_score": emotion,
        "shareability_score": shareability,
        **raw_neg,
        "llm_quality_score": llm_quality,
        "positive_score": positive_score,
        "total_penalty": total_penalty,
        "final_score": final_score,
    }

    diagnostics = {
        "positive_score": positive_score,
        "total_penalty": total_penalty,
        "raw_score": raw_score,
        "applied_caps": applied_caps,
        "raw_positive_dimensions": raw_pos,
        "raw_negative_dimensions": raw_neg,
    }

    return final_score, subscores, diagnostics


def compute_score_distribution(
    scores: list[float],
    unrounded_scores: Optional[list[float]] = None,
) -> ScoreDistributionDiagnostics:
    """Compute statistical distribution diagnostics for score predictions with fixed linear interpolation."""
    if not scores:
        return ScoreDistributionDiagnostics(
            min=0.0,
            p10=0.0,
            p25=0.0,
            median=0.0,
            p75=0.0,
            p90=0.0,
            max=0.0,
            unique_score_count_raw=0,
            unique_score_count_rounded=0,
            unique_score_count=0,
            zero_score_count=0,
            standard_deviation=0.0,
            warning="No predictions to compute distribution.",
        )

    sorted_scores = sorted(scores)
    n = len(sorted_scores)

    def _linear_percentile(sorted_vals: list[float], p: float) -> float:
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        idx = (len(sorted_vals) - 1) * (p / 100.0)
        low = int(idx)
        high = min(low + 1, len(sorted_vals) - 1)
        weight = idx - low
        return sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight

    min_val = round(sorted_scores[0], 2)
    p10 = round(_linear_percentile(sorted_scores, 10.0), 2)
    p25 = round(_linear_percentile(sorted_scores, 25.0), 2)
    median = round(_linear_percentile(sorted_scores, 50.0), 2)
    p75 = round(_linear_percentile(sorted_scores, 75.0), 2)
    p90 = round(_linear_percentile(sorted_scores, 90.0), 2)
    max_val = round(sorted_scores[-1], 2)

    raw_list = unrounded_scores if unrounded_scores is not None else scores
    unique_raw = len(set(raw_list))
    unique_rounded = len(set(round(s, 2) for s in scores))
    zero_count = sum(1 for s in scores if round(s, 2) == 0.0)

    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    std_dev = round(math.sqrt(variance), 2)

    warnings = []
    if n >= 10 and unique_rounded < 10:
        warnings.append(
            f"Score collapse warning: only {unique_rounded} unique rounded scores among {n} candidates"
        )
    if std_dev < 2.0:
        warnings.append(
            f"Low variance warning: standard deviation is {std_dev:.2f} (expected spread across 0-100)"
        )
    if n > 0 and (zero_count / n) >= 0.30:
        warnings.append(
            f"Zero-score collapse warning: {zero_count}/{n} ({zero_count/n:.1%}) candidates scored 0.0"
        )

    warning_str = "; ".join(warnings) if warnings else None

    return ScoreDistributionDiagnostics(
        min=min_val,
        p10=p10,
        p25=p25,
        median=median,
        p75=p75,
        p90=p90,
        max=max_val,
        unique_score_count_raw=unique_raw,
        unique_score_count_rounded=unique_rounded,
        unique_score_count=unique_rounded,
        zero_score_count=zero_count,
        standard_deviation=std_dev,
        warning=warning_str,
    )


class OpenAILLMScorer(HighlightScorer):
    """Highlight scorer connecting to an OpenAI-compatible API endpoint (highlight_v2)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 30.0,
        allow_fallback: bool = True,
        max_retries: int = 3,
        temperature: float = 0.1,
        context_window_seconds: float = 45.0,
        scorer_version: str = SCORER_VERSION_V2,
    ) -> None:
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.allow_fallback = allow_fallback
        self.max_retries = max(1, max_retries)
        self.temperature = temperature
        self.context_window_seconds = context_window_seconds

        if scorer_version == SCORER_VERSION_V2:
            self.name = "highlight_v2"
            self.version = SCORER_VERSION_V2
            self.prompt_version = PROMPT_VERSION_V2
            self.prompt_hash = PROMPT_HASH_V2
            self.score_formula_version = SCORE_FORMULA_VERSION_V2
            self.system_prompt = SYSTEM_PROMPT_V2
        else:
            self.name = "highlight_v2_1"
            self.version = SCORER_VERSION_V2_1
            self.prompt_version = PROMPT_VERSION_V2_1
            self.prompt_hash = PROMPT_HASH_V2_1
            self.score_formula_version = SCORE_FORMULA_VERSION_V2_1
            self.system_prompt = SYSTEM_PROMPT_V2_1

        self.fallback_scorer = HeuristicScorer()

    def score(
        self,
        candidate: CandidateWindow,
        context: Optional[dict[str, Any]] = None,
    ) -> HighlightScore:
        """Score candidate using OpenAI-compatible API with retries and formula computation."""
        if not self.api_key:
            reason = "FREECHER_LLM_API_KEY is not set"
            if not self.allow_fallback:
                raise RuntimeError(
                    f"LLM scoring failed for candidate {candidate.id}: {reason}, and allow_fallback=False."
                )
            logger.warning(
                f"[scoring] LLM fallback engaged for {candidate.id}: {reason}. Falling back to HeuristicScorer."
            )
            heuristic_res = self.fallback_scorer.score(candidate)
            return HighlightScore(
                score=heuristic_res.score,
                hook_score=heuristic_res.hook_score,
                standalone_score=heuristic_res.standalone_score,
                emotion_score=heuristic_res.emotion_score,
                information_score=heuristic_res.information_score,
                shareability_score=heuristic_res.shareability_score,
                reason=f"{heuristic_res.reason} [Fallback: {reason}]",
                fallback_used=True,
                fallback_reason=reason,
                scorer_version=self.version,
                requested_model=self.model,
                actual_model="heuristic_v1",
            )

        prev_ctx = context.get("previous_context", "") if context else ""
        next_ctx = context.get("next_context", "") if context else ""
        wpm = context.get("speech_rate_wpm") if context else None

        user_content_lines = [
            f"Candidate ID: {candidate.id}",
            f"Timestamps: {candidate.start:.2f}s - {candidate.end:.2f}s (Duration: {candidate.duration:.1f}s)",
        ]
        if wpm is not None and wpm > 0:
            user_content_lines.append(f"Estimated speech rate: {wpm} WPM")
        user_content_lines.append("")
        user_content_lines.append("--- PREVIOUS CONTEXT (up to 45s before candidate) ---")
        user_content_lines.append(prev_ctx if prev_ctx else "[None - start of video or no preceding speech]")
        user_content_lines.append("")
        user_content_lines.append("--- CANDIDATE TEXT (segment to evaluate) ---")
        user_content_lines.append(f'"{candidate.text}"')
        user_content_lines.append("")
        user_content_lines.append("--- NEXT CONTEXT (up to 45s after candidate) ---")
        user_content_lines.append(next_ctx if next_ctx else "[None - end of video or no subsequent speech]")

        user_content = "\n".join(user_content_lines)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
        }

        url = f"{self.base_url}/chat/completions"

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                choice_content = data["choices"][0]["message"]["content"].strip()
                # Strip markdown fences if returned
                choice_content = re.sub(r"^```(?:json)?\s*", "", choice_content)
                choice_content = re.sub(r"\s*```$", "", choice_content).strip()

                parsed = json.loads(choice_content)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected JSON dictionary, got {type(parsed)}")

                # In v2.1, validate all required ranking fields strictly (no silent imputation)
                if self.version == SCORER_VERSION_V2_1:
                    for req_f in REQUIRED_FIELDS_V2_1:
                        if req_f not in parsed or parsed[req_f] is None:
                            raise ValueError(
                                f"Missing required ranking field '{req_f}' in LLM response"
                            )
                    for num_f in (
                        "hook",
                        "standalone",
                        "story_payoff",
                        "emotion",
                        "humor",
                        "surprise",
                        "retention",
                        "shareability",
                        "boringness",
                        "context_dependency",
                    ):
                        try:
                            float(parsed[num_f])
                        except (TypeError, ValueError) as ex:
                            raise ValueError(
                                f"Invalid numeric value for required field '{num_f}': {parsed[num_f]}"
                            ) from ex

                # Compute final score, subscores, and diagnostics using deterministic versioned formula
                if self.version == SCORER_VERSION_V2:
                    final_score, subscores = highlight_v2_score_formula_v1(parsed)
                    diagnostics: dict[str, Any] = {
                        "positive_score": None,
                        "total_penalty": None,
                        "raw_score": final_score,
                        "applied_caps": [],
                        "raw_positive_dimensions": None,
                        "raw_negative_dimensions": None,
                    }
                else:
                    final_score, subscores, diagnostics = highlight_v2_1_formula_v1(parsed)

                flags = {
                    "setup_only": bool(parsed.get("setup_only", False)),
                    "transitional": bool(parsed.get("transitional", False)),
                    "outside_payoff": bool(parsed.get("outside_payoff", False)),
                }

                reason_text = parsed.get("reason", f"Evaluated by {self.version}")

                return HighlightScore(
                    score=final_score,
                    hook_score=subscores["hook"],
                    standalone_score=subscores["standalone"],
                    emotion_score=subscores["emotion"],
                    information_score=subscores["story_payoff"],
                    shareability_score=subscores["shareability"],
                    story_payoff_score=subscores["story_payoff"],
                    humor_score=subscores["humor"],
                    surprise_score=subscores["surprise"],
                    retention_score=subscores["retention"],
                    boringness_score=subscores["boringness"],
                    context_dependency_score=subscores["context_dependency"],
                    llm_quality_score=subscores.get("llm_quality_score"),
                    final_score=final_score,
                    positive_score=diagnostics.get("positive_score"),
                    total_penalty=diagnostics.get("total_penalty"),
                    applied_caps=diagnostics.get("applied_caps"),
                    raw_positive_dimensions=diagnostics.get("raw_positive_dimensions"),
                    raw_negative_dimensions=diagnostics.get("raw_negative_dimensions"),
                    subscores=subscores,
                    flags=flags,
                    reason=f"{self.version} ({self.model}): {reason_text}",
                    fallback_used=False,
                    fallback_reason=None,
                    scorer_version=self.version,
                    requested_model=self.model,
                    actual_model=self.model,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"[scoring] Attempt {attempt}/{self.max_retries} for candidate {candidate.id} failed: {exc}"
                )

        # If all retries failed
        fallback_err = f"{type(last_error).__name__}: {last_error}"
        if not self.allow_fallback:
            raise RuntimeError(
                f"LLM scoring failed for candidate {candidate.id} after {self.max_retries} attempts: {fallback_err}. "
                f"Fallback disabled for benchmark."
            )

        logger.warning(
            f"[scoring] All {self.max_retries} attempts failed for {candidate.id}: {fallback_err}. Falling back to HeuristicScorer."
        )
        heuristic_res = self.fallback_scorer.score(candidate)
        return HighlightScore(
            score=heuristic_res.score,
            hook_score=heuristic_res.hook_score,
            standalone_score=heuristic_res.standalone_score,
            emotion_score=heuristic_res.emotion_score,
            information_score=heuristic_res.information_score,
            shareability_score=heuristic_res.shareability_score,
            reason=f"{heuristic_res.reason} [Fallback: {fallback_err}]",
            fallback_used=True,
            fallback_reason=fallback_err,
            scorer_version=self.version,
            requested_model=self.model,
            actual_model="heuristic_v1",
        )

    def score_batch(
        self,
        candidates: list[CandidateWindow],
        transcript: Optional[Any] = None,
    ) -> list[HighlightScore]:
        """Evaluate multiple candidate windows sequentially, utilizing transcript context when available."""
        scores: list[HighlightScore] = []
        for cand in candidates:
            ctx = (
                extract_surrounding_context(cand, transcript, self.context_window_seconds)
                if transcript
                else None
            )
            scores.append(self.score(cand, context=ctx))
        return scores

