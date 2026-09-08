"""OpenAI-compatible LLM Highlight Scorer (highlight_v2) with decoupled feature extraction and versioned scoring."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from freecher_worker.highlights.models import CandidateWindow, HighlightScore
from .base import HighlightScorer
from .heuristic import HeuristicScorer

logger = logging.getLogger("freecher_worker")

SCORER_VERSION = "highlight_v2"
PROMPT_VERSION = "highlight_v2_prompt_v1"
SCORE_FORMULA_VERSION = "highlight_v2_formula_v1"

SYSTEM_PROMPT = """You are a world-class viral short-form video editor and content strategist (TikTok, Instagram Reels, YouTube Shorts).
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

PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]


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
    ) -> None:
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.allow_fallback = allow_fallback
        self.max_retries = max(1, max_retries)
        self.temperature = temperature
        self.context_window_seconds = context_window_seconds

        self.name = "highlight_v2"
        self.version = SCORER_VERSION
        self.prompt_version = PROMPT_VERSION
        self.prompt_hash = PROMPT_HASH
        self.score_formula_version = SCORE_FORMULA_VERSION

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
                {"role": "system", "content": SYSTEM_PROMPT},
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

                # Compute final score and subscores using deterministic versioned formula
                final_score, subscores = highlight_v2_score_formula_v1(parsed)

                flags = {
                    "setup_only": bool(parsed.get("setup_only", False)),
                    "transitional": bool(parsed.get("transitional", False)),
                    "outside_payoff": bool(parsed.get("outside_payoff", False)),
                }

                reason_text = parsed.get("reason", "Evaluated by highlight_v2")

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
                    llm_quality_score=subscores["llm_quality_score"],
                    final_score=final_score,
                    subscores=subscores,
                    flags=flags,
                    reason=f"highlight_v2 ({self.model}): {reason_text}",
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

