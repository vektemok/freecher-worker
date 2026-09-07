"""OpenAI-compatible LLM Highlight Scorer with JSON validation and graceful fallback observability."""

from __future__ import annotations

import json
import logging
from typing import Optional

import httpx

from freecher_worker.highlights.models import CandidateWindow, HighlightScore
from .base import HighlightScorer
from .heuristic import HeuristicScorer

logger = logging.getLogger("freecher_worker")

SCORER_VERSION = "1.1.0"

SYSTEM_PROMPT = """You are an expert short-form video editor (Shorts, Reels, TikTok).
Evaluate whether the following speech transcript fragment is suitable for a standalone viral clip.

Evaluation Criteria (each scored 0 to 100):
1. hook_score: Strong opening hook, immediately catches attention within the first seconds.
2. standalone_score: Clear and understandable without needing previous/subsequent video context.
3. emotion_score: High energy, surprise, excitement, conflict, or emotional relatability.
4. information_score: Contains a concrete insight, lesson, revelation, numbers, or useful story.
5. shareability_score: High urge for viewers to like, save, or share with friends.
6. score: Overall rating from 0 to 100 representing clip viability.

You MUST reply ONLY with a valid JSON object matching this schema:
{
  "score": <float between 0 and 100>,
  "hook_score": <float between 0 and 100>,
  "standalone_score": <float between 0 and 100>,
  "emotion_score": <float between 0 and 100>,
  "information_score": <float between 0 and 100>,
  "shareability_score": <float between 0 and 100>,
  "reason": "<one or two sentences explaining why this moment is or isn't a great short clip>"
}
Do not include any Markdown fencing like ```json, just pure raw JSON."""


class OpenAILLMScorer(HighlightScorer):
    """Highlight scorer connecting to an OpenAI-compatible API endpoint with explicit fallback observability."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.name = "llm"
        self.version = SCORER_VERSION
        self.fallback_scorer = HeuristicScorer()

    def score(self, candidate: CandidateWindow) -> HighlightScore:
        """Score candidate using OpenAI-compatible API, recording fallback metadata if failed."""
        if not self.api_key:
            reason = "FREECHER_LLM_API_KEY is not set"
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
            )

        user_content = (
            f"Candidate ID: {candidate.id}\n"
            f"Duration: {candidate.duration:.1f}s (from {candidate.start:.1f}s to {candidate.end:.1f}s)\n"
            f"Transcript text:\n\"\"\"{candidate.text}\"\"\""
        )

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
            "temperature": 0.2,
        }

        url = f"{self.base_url}/chat/completions"

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

            choice_content = data["choices"][0]["message"]["content"].strip()
            # Strip potential ```json ``` fences if model added them
            if choice_content.startswith("```"):
                choice_content = choice_content.strip("`")
                if choice_content.startswith("json"):
                    choice_content = choice_content[4:].strip()

            parsed = json.loads(choice_content)
            return HighlightScore(
                score=float(parsed["score"]),
                hook_score=float(parsed.get("hook_score", parsed["score"])),
                standalone_score=float(parsed.get("standalone_score", parsed["score"])),
                emotion_score=float(parsed.get("emotion_score", parsed["score"])),
                information_score=float(parsed.get("information_score", parsed["score"])),
                shareability_score=float(parsed.get("shareability_score", parsed["score"])),
                reason=f"LLM ({self.model}): {parsed.get('reason', 'Evaluated by LLM')}",
                fallback_used=False,
                fallback_reason=None,
            )
        except Exception as exc:
            fallback_err = f"{type(exc).__name__}: {exc}"
            logger.warning(
                f"[scoring] LLM fallback engaged for {candidate.id}: {fallback_err}. Falling back to HeuristicScorer."
            )
            heuristic_result = self.fallback_scorer.score(candidate)
            return HighlightScore(
                score=heuristic_result.score,
                hook_score=heuristic_result.hook_score,
                standalone_score=heuristic_result.standalone_score,
                emotion_score=heuristic_result.emotion_score,
                information_score=heuristic_result.information_score,
                shareability_score=heuristic_result.shareability_score,
                reason=f"{heuristic_result.reason} [Fallback: {fallback_err}]",
                fallback_used=True,
                fallback_reason=fallback_err,
            )
