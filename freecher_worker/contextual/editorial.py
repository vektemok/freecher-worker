"""Editorial classification and REJECT filter for contextual_reranker_v1.

The model classifies each candidate (REJECT / WEAK / GOOD / STRONG) and must state a
concrete reason a stranger would keep watching. A candidate that cannot produce one is
demoted here, deterministically, after the model has answered.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .cache import ContextualCache
from .candidate_context import render_candidate_prompt
from .models import (
    SURVIVING_CLASSES,
    CandidateContextPackage,
    EditorialAnalysis,
    GlobalContext,
)
from .prompts import get_stage_prompt
from .provider import ContextualProvider, ContextualProviderError
from .versions import EDITORIAL_SCHEMA_VERSION

logger = logging.getLogger("freecher_worker")

VALID_CLASSES = ("REJECT", "WEAK", "GOOD", "STRONG")

#: A reason_to_watch shorter than this cannot be concrete.
MIN_REASON_WORDS = 4

#: Phrases that describe continuation rather than a reason to watch.
VAGUE_REASON_PATTERNS = (
    r"^\s*(they|the participants?|the speakers?|the hosts?)\s+(continue|keep|are|is)\b",
    r"^\s*(a|the)?\s*(general|casual|normal|ordinary)\s+(conversation|discussion|chat)\b",
    r"^\s*(discussion|conversation|talk|chat)\s+about\b",
    r"\bcontinues? (?:to )?(?:discuss|talk|chat)\b",
)


def _coerce_unit(value: Any, default: float = 0.0) -> float:
    """Coerce a model value into [0, 1], accepting 0-100 inputs."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if num > 1.0:
        num = num / 100.0
    return max(0.0, min(1.0, num))


def _clean_reason(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("null", "none", "n/a", "-"):
        return None
    return text


def is_vague_reason(reason: Optional[str]) -> bool:
    """True when a reason_to_watch does not name anything concrete."""
    if not reason:
        return True
    text = reason.strip()
    if len(text.split()) < MIN_REASON_WORDS:
        return True
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in VAGUE_REASON_PATTERNS)


def parse_editorial_response(
    parsed: Dict[str, Any],
    candidate_id: str,
    model: Optional[str],
    prompt_version: Optional[str],
) -> EditorialAnalysis:
    """Turn a raw model JSON object into a validated EditorialAnalysis."""
    raw_class = str(parsed.get("editorial_class") or "").strip().upper()
    editorial_class = raw_class if raw_class in VALID_CLASSES else "WEAK"

    reject_reasons: List[str] = []
    raw_reasons = parsed.get("reject_reasons")
    if isinstance(raw_reasons, list):
        reject_reasons = [str(r).strip() for r in raw_reasons if str(r).strip()][:8]
    elif isinstance(raw_reasons, str) and raw_reasons.strip():
        reject_reasons = [raw_reasons.strip()]

    if raw_class and raw_class not in VALID_CLASSES:
        reject_reasons.append(f"unrecognized editorial_class '{raw_class}' treated as WEAK")

    analysis = EditorialAnalysis(
        candidate_id=candidate_id,
        editorial_class=editorial_class,  # type: ignore[arg-type]
        scroll_stop=_coerce_unit(parsed.get("scroll_stop")),
        hook=_coerce_unit(parsed.get("hook")),
        payoff=_coerce_unit(parsed.get("payoff")),
        surprise=_coerce_unit(parsed.get("surprise")),
        humor=_coerce_unit(parsed.get("humor")),
        tension=_coerce_unit(parsed.get("tension")),
        emotion=_coerce_unit(parsed.get("emotion")),
        visual_interest=_coerce_unit(parsed.get("visual_interest")),
        novelty=_coerce_unit(parsed.get("novelty")),
        self_contained=_coerce_unit(parsed.get("self_contained")),
        shareability=_coerce_unit(parsed.get("shareability")),
        context_dependency=_coerce_unit(parsed.get("context_dependency")),
        dead_air=_coerce_unit(parsed.get("dead_air")),
        reason_to_watch=_clean_reason(parsed.get("reason_to_watch")),
        reason_to_skip=_clean_reason(parsed.get("reason_to_skip")),
        reject_reasons=reject_reasons,
        confidence=_coerce_unit(parsed.get("confidence")),
        prompt_version=prompt_version,
        model=model,
    )
    return apply_reason_gate(analysis)


def apply_reason_gate(analysis: EditorialAnalysis) -> EditorialAnalysis:
    """Demote candidates whose reason to watch is missing or vague.

    This is not a scoring formula: it enforces the stated contract that a highlight must
    come with a concrete reason a stranger would keep watching.
    """
    if analysis.editorial_class == "REJECT":
        if not analysis.reject_reasons:
            analysis.reject_reasons = ["model rejected without naming a reason"]
        return analysis

    if is_vague_reason(analysis.reason_to_watch):
        if analysis.reason_to_watch is None:
            analysis.reject_reasons.append("no reason to watch was produced")
        else:
            analysis.reject_reasons.append(
                "reason to watch describes continuation, not a concrete moment"
            )
        analysis.editorial_class = "REJECT"  # type: ignore[assignment]
    return analysis


def degraded_analysis(
    candidate_id: str,
    error: str,
    model: Optional[str],
    prompt_version: Optional[str],
) -> EditorialAnalysis:
    """Safe result for an unusable API response.

    A transport or parse failure is never allowed to reject a candidate outright, nor to
    promote one: the candidate continues as WEAK with zero confidence.
    """
    return EditorialAnalysis(
        candidate_id=candidate_id,
        editorial_class="WEAK",
        reason_to_watch=None,
        reason_to_skip=None,
        reject_reasons=[],
        confidence=0.0,
        prompt_version=prompt_version,
        model=model,
        parse_failed=True,
        error=error[:300],
    )


def analyze_candidate(
    package: CandidateContextPackage,
    global_context: GlobalContext,
    provider: Optional[ContextualProvider],
    cache: ContextualCache,
) -> EditorialAnalysis:
    """Run (or reuse a cached) editorial classification for one candidate."""
    prompt_version, prompt_hash, system_prompt = get_stage_prompt("editorial")
    model = getattr(provider, "model", None)

    req_hash = cache.request_hash(
        stage="editorial",
        prompt_version=prompt_version,
        prompt_hash=prompt_hash,
        schema_version=EDITORIAL_SCHEMA_VERSION,
        payload={"package_hash": package.package_hash, "candidate_id": package.candidate_id},
    )

    cached = cache.load("editorial", req_hash)
    if cached is not None:
        if provider is not None:
            provider.note_cache_hit("editorial")
        return parse_editorial_response(cached, package.candidate_id, model, prompt_version)

    if provider is None:
        return degraded_analysis(
            package.candidate_id, "no provider configured", model, prompt_version
        )

    user_content = render_candidate_prompt(package, global_context)
    try:
        parsed = provider.complete_json(system_prompt, user_content, stage="editorial")
    except (ContextualProviderError, ValueError) as exc:
        logger.warning(
            f"[contextual-editorial] Analysis failed for {package.candidate_id}: {exc}"
        )
        return degraded_analysis(package.candidate_id, str(exc), model, prompt_version)

    cache.store("editorial", req_hash, parsed)
    return parse_editorial_response(parsed, package.candidate_id, model, prompt_version)


def survives_reject_filter(analysis: EditorialAnalysis) -> bool:
    """True when a candidate may continue to the comparative stage."""
    return analysis.editorial_class in SURVIVING_CLASSES
