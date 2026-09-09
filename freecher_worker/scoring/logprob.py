"""Highlight scorer that reads the model's confidence instead of its arithmetic.

`highlight_v2_1` asks the model to write a 0-100 number. Measured against the
frozen gold set that number carries almost no ranking signal: Spearman +0.068
against human labels, Pearson -0.014. The ranking literature says why -- a
score the model *writes out* is the weakest form of pointwise ranking, below
reading the probability it assigns to a decision token (Zhuang et al.,
"A Setwise Approach...", SIGIR 2024: NDCG@10 0.654 for yes/no logprobs against
0.557 for a generated quality number).

So this asks for one word and takes the score from `logprobs`, as the
probability-weighted average of the answer's possible values. The model never
does the arithmetic, and the result is continuous.

Two modes, because a 15-candidate pilot did not separate them:

- `binary` (yes/no) separated the human groups monotonically -- 9.3 / 0.00 /
  0.00 mean for strong / borderline / reject -- but a confident "no" pushes
  `yes` out of the top-k, so rejected candidates tie at exactly 0.
- `graded` (STRONG/DECENT/WEAK/DEAD) avoids the pile-up at zero but moved the
  ties to 33.3 and lost the monotonic separation.

Both roughly doubled the v2_1 baseline correlation on that pilot. Which is
actually better is a question for the full frozen set, not for 15 candidates.

Requires a model that returns logprobs. The gpt-5.5 and gpt-5.6 families do
not, and are rejected with a clear error rather than silently degraded.
"""

from __future__ import annotations

import logging
import math
import re
import time
from typing import Any, Optional

from freecher_worker.highlights.models import CandidateWindow, HighlightScore
from freecher_worker.scoring.base import HighlightScorer
from freecher_worker.scoring.llm import extract_surrounding_context

logger = logging.getLogger("freecher_worker")

LOGPROB_SCORER_VERSION = "logprob_v1"
LOGPROB_PROMPT_VERSION = "logprob_v1_prompt_v1"

MODE_GRADED = "graded"
MODE_BINARY = "binary"

GRADE_VALUES = {"strong": 100.0, "decent": 66.667, "weak": 33.333, "dead": 0.0}
BINARY_VALUES = {"yes": 100.0, "no": 0.0}

# The product question, not a topical-relevance one: the target is a clip that
# holds a stranger on TikTok/Shorts/Reels, and most of an ordinary conversation
# does not. The bar is stated explicitly because the benchmark is
# negative-heavy by design and a permissive judge would pass everything.
_SHARED_CRITERIA = """You decide whether one clip from a long stream would work as a standalone short-form video on TikTok, YouTube Shorts or Instagram Reels.

The viewer is a stranger scrolling. They did not choose this streamer, they have no context, and they leave in about two seconds unless something holds them.

A clip works only if ALL of these hold:
- the opening seconds give a stranger a reason to stay: a question, a claim, a confession, conflict, or something absurd;
- it makes sense with no knowledge of what came before;
- something actually lands inside the clip -- a punchline, a reveal, a reaction, a resolution -- rather than merely being set up;
- a stranger would plausibly react: laugh, be surprised, be outraged, or want to reply.

Ordinary competent conversation does not work, however pleasant or coherent: chat that needs prior context, pure setup whose payoff falls outside the window, filler, logistics, or anything whose interest depends on already following this streamer."""

GRADED_SYSTEM_PROMPT = _SHARED_CRITERIA + """

Reply with exactly one word, and nothing else:

STRONG - you would bet on this clip
DECENT - it could work with a better cut
WEAK - ordinary conversation, no reason for a stranger to stay
DEAD - unusable: no context, no payoff, filler

Most clips are WEAK or DEAD."""

BINARY_SYSTEM_PROMPT = _SHARED_CRITERIA + """

Reply with exactly one word: yes or no.

Answer "yes" only when you would bet on this clip. Most clips are "no"."""

_USER_BODY = """--- WHAT CAME BEFORE (up to 45s, context only, NOT part of the clip) ---
{previous}

--- THE CLIP ({duration:.0f}s) ---
"{candidate}"

--- WHAT CAME AFTER (up to 45s, context only, NOT part of the clip) ---
{next_context}

Would this clip hold a scrolling stranger on TikTok / Shorts / Reels? """

GRADED_USER_TEMPLATE = _USER_BODY + "Reply with exactly one word: STRONG, DECENT, WEAK or DEAD."
BINARY_USER_TEMPLATE = _USER_BODY + "Reply with exactly one word: yes or no."


class LogprobScoringError(Exception):
    """Raised when the model cannot be asked for, or does not return, logprobs."""


def normalize_token(token: str) -> str:
    """Strip whitespace and punctuation so ' Yes.' and 'yes' compare equal."""
    return re.sub(r"[^\w]", "", token).strip().lower()


def expected_value(
    top_logprobs: list[dict[str, Any]],
    values: dict[str, float],
) -> Optional[float]:
    """Probability-weighted average of the answer tokens, or None if none appeared.

    Weighting by probability rather than taking the token the model emitted is
    what keeps the score continuous: two candidates it both calls WEAK are
    still separated by how nearly it called them DECENT.
    """
    mass: dict[str, float] = {}
    for entry in top_logprobs:
        token = normalize_token(str(entry.get("token", "")))
        if token in values:
            mass[token] = mass.get(token, 0.0) + math.exp(float(entry["logprob"]))

    total = sum(mass.values())
    if total <= 0.0:
        return None
    return sum(values[token] * weight for token, weight in mass.items()) / total


class LogprobBinaryScorer(HighlightScorer):
    """Scores a candidate by the probability mass the model puts on each answer."""

    def __init__(
        self,
        base_url: Optional[str],
        api_key: Optional[str],
        model: str = "gpt-4o-mini",
        *,
        mode: str = MODE_BINARY,
        temperature: Optional[float] = 0.0,
        top_logprobs: int = 20,
        context_window_seconds: float = 45.0,
        timeout_seconds: float = 90.0,
        # A 146-candidate run is ~10 minutes of wall clock, long enough to meet
        # a DNS blip. Three attempts over six seconds is not enough to ride one
        # out, and losing the whole run at candidate 9 wastes every call before
        # it, so the ladder is longer and the backoff steeper.
        max_retries: int = 6,
        retry_backoff_seconds: float = 3.0,
    ) -> None:
        if mode not in (MODE_GRADED, MODE_BINARY):
            raise ValueError(f"unknown mode '{mode}'; expected '{MODE_GRADED}' or '{MODE_BINARY}'")
        self.mode = mode
        self.values = GRADE_VALUES if mode == MODE_GRADED else BINARY_VALUES
        self.system_prompt = GRADED_SYSTEM_PROMPT if mode == MODE_GRADED else BINARY_SYSTEM_PROMPT
        self.user_template = GRADED_USER_TEMPLATE if mode == MODE_GRADED else BINARY_USER_TEMPLATE

        self.name = "logprob"
        self.version = f"{LOGPROB_SCORER_VERSION}_{mode}"
        self.prompt_version = f"{LOGPROB_PROMPT_VERSION}_{mode}"
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.top_logprobs = top_logprobs
        self.context_window_seconds = context_window_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def build_user_content(
        self, candidate: CandidateWindow, context: Optional[dict[str, Any]]
    ) -> str:
        context = context or {}
        return self.user_template.format(
            previous=context.get("previous_context") or "[nothing - start of stream]",
            candidate=candidate.text.strip(),
            duration=candidate.duration,
            next_context=context.get("next_context") or "[nothing - end of stream]",
        )

    def _request(self, user_content: str) -> tuple[float, dict[str, Any]]:
        import httpx

        if not self.api_key:
            raise LogprobScoringError("no API key configured for the logprob scorer")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "max_completion_tokens": 4,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(url, headers=headers, json=payload)
                if response.status_code == 400:
                    # A model that cannot do this at all must fail loudly rather
                    # than be retried into the same wall.
                    detail = response.json().get("error", {}).get("message", "")
                    raise LogprobScoringError(f"model '{self.model}' rejected the request: {detail}")
                response.raise_for_status()
                data = response.json()

                content = (data["choices"][0].get("logprobs") or {}).get("content")
                if not content:
                    raise LogprobScoringError(f"model '{self.model}' returned no logprobs")

                expected = expected_value(content[0].get("top_logprobs") or [], self.values)
                if expected is None:
                    raise LogprobScoringError(
                        "no answer token appeared among the model's alternatives"
                    )
                return expected, {
                    "answer": data["choices"][0]["message"]["content"].strip(),
                    "usage": data.get("usage"),
                    "actual_model": data.get("model", self.model),
                }
            except LogprobScoringError:
                raise
            except Exception as exc:  # transport, DNS, 5xx, rate limit
                last_error = exc
                if attempt < self.max_retries:
                    delay = self.retry_backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "[logprob] attempt %d/%d failed (%s); retrying in %.0fs",
                        attempt, self.max_retries, type(exc).__name__, delay,
                    )
                    time.sleep(delay)

        raise LogprobScoringError(f"request failed after {self.max_retries} attempts: {last_error}")

    def score(
        self,
        candidate: CandidateWindow,
        context: Optional[dict[str, Any]] = None,
    ) -> HighlightScore:
        expected, meta = self._request(self.build_user_content(candidate, context))
        value = round(expected, 4)

        return HighlightScore(
            # One dimension is all this scorer measures. The rest exist because
            # HighlightScore requires them and are set to the same value rather
            # than invented independently.
            score=value,
            hook_score=value,
            standalone_score=value,
            emotion_score=value,
            information_score=value,
            shareability_score=value,
            reason=f"E[{self.mode}]={expected:.3f} from '{meta['answer']}' ({meta['actual_model']})",
            fallback_used=False,
            fallback_reason=None,
            llm_quality_score=value,
            final_score=value,
            scorer_version=self.version,
            requested_model=self.model,
            actual_model=meta["actual_model"],
        )

    def score_batch(
        self,
        candidates: list[CandidateWindow],
        transcript: Optional[Any] = None,
    ) -> list[HighlightScore]:
        """Score every candidate, reusing the existing context extraction."""
        scores: list[HighlightScore] = []
        for index, candidate in enumerate(candidates, start=1):
            context = (
                extract_surrounding_context(candidate, transcript, self.context_window_seconds)
                if transcript is not None
                else None
            )
            scores.append(self.score(candidate, context=context))
            if index % 25 == 0:
                logger.info("[logprob/%s] scored %d/%d", self.mode, index, len(candidates))
        return scores
