"""Salvage-aware editorial assessment for the contextual reranker.

Only fundamentally unusable source windows are removed. Missing setup, rough ASR, or an
imperfect 60-second boundary remain soft quality signals for comparative ranking.
"""

from __future__ import annotations

import logging
import math
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

VALID_CLASSES = ("FATAL_REJECT", "WEAK", "MAYBE", "GOOD", "STRONG")

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


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
    return default


def _coerce_nonnegative(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


def _coerce_penalty(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number > 0:
        number = -number
    return max(-100.0, min(0.0, number))


def _string_list(value: Any, limit: int = 8) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()][:limit]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def is_vague_reason(reason: Optional[str]) -> bool:
    """True when a reason_to_watch does not name anything concrete."""
    if not reason:
        return True
    text = reason.strip()
    if len(text.split()) < MIN_REASON_WORDS:
        return True
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in VAGUE_REASON_PATTERNS)


def has_strong_fatal_evidence(analysis: EditorialAnalysis) -> bool:
    """Validate that claimed fatal evidence names a legitimate unusability mode."""
    evidence = " ".join(analysis.fatal_reject_evidence).lower()
    if not evidence:
        return False
    direct_patterns = (
        "dead air",
        "duplicate",
        "near-duplicate",
        "technical corruption",
        "cannot be decoded",
        "no meaningful content",
        "no understandable event",
        "no useful event",
    )
    if any(pattern in evidence for pattern in direct_patterns):
        return True
    if "payoff" in evidence and "outside" in evidence and (
        "no useful" in evidence or "nothing" in evidence
    ):
        return True
    if ("severe" in evidence or "unintelligible" in evidence) and (
        "transcript" in evidence or "asr" in evidence
    ) and ("no visual" in evidence or "no useful visual" in evidence):
        return True
    return False


def parse_editorial_response(
    parsed: Dict[str, Any],
    candidate_id: str,
    model: Optional[str],
    prompt_version: Optional[str],
) -> EditorialAnalysis:
    """Turn a raw model JSON object into a validated EditorialAnalysis."""
    raw_class = str(parsed.get("editorial_class") or "").strip().upper()
    # REJECT was the v1 spelling. Accept it only as a legacy synonym so stale or
    # third-party providers fail closed at the schema boundary, not at validation.
    normalized_class = "FATAL_REJECT" if raw_class == "REJECT" else raw_class
    editorial_class = normalized_class if normalized_class in VALID_CLASSES else "WEAK"

    reject_reasons = _string_list(parsed.get("reject_reasons"))

    if raw_class and raw_class not in (*VALID_CLASSES, "REJECT"):
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
        salvageable=_coerce_bool(
            parsed.get("salvageable"), default=editorial_class != "FATAL_REJECT"
        ),
        best_internal_moment_present=_coerce_bool(parsed.get("best_internal_moment_present")),
        needs_more_setup=_coerce_bool(parsed.get("needs_more_setup")),
        needs_boundary_refinement=_coerce_bool(parsed.get("needs_boundary_refinement")),
        required_setup_seconds_estimate=_coerce_nonnegative(
            parsed.get("required_setup_seconds_estimate")
        ),
        payoff_inside_candidate=_coerce_bool(parsed.get("payoff_inside_candidate")),
        standalone_after_refinement_probability=_coerce_unit(
            parsed.get("standalone_after_refinement_probability")
        ),
        quality_penalty=_coerce_penalty(
            parsed.get("quality_penalty", parsed.get("penalty", 0.0))
        ),
        fatal_reject_evidence=_string_list(
            parsed.get("fatal_reject_evidence", parsed.get("fatal_evidence"))
        ),
        reason_to_watch=_clean_reason(parsed.get("reason_to_watch")),
        reason_to_skip=_clean_reason(parsed.get("reason_to_skip")),
        reject_reasons=reject_reasons,
        confidence=_coerce_unit(parsed.get("confidence")),
        prompt_version=prompt_version,
        model=model,
    )
    return apply_editorial_safety_gates(analysis)


def apply_editorial_safety_gates(analysis: EditorialAnalysis) -> EditorialAnalysis:
    """Prevent context/ASR/boundary uncertainty from becoming accidental deletion."""
    derived_penalty = 0.0
    if analysis.needs_more_setup:
        derived_penalty -= 10.0
    if analysis.needs_boundary_refinement:
        derived_penalty -= 5.0
    if analysis.context_dependency > 0.5:
        derived_penalty -= 20.0 * (analysis.context_dependency - 0.5)
    if analysis.best_internal_moment_present and not analysis.payoff_inside_candidate:
        derived_penalty -= 5.0
    analysis.quality_penalty = min(analysis.quality_penalty, max(-40.0, derived_penalty))

    if analysis.editorial_class == "FATAL_REJECT":
        if not analysis.reject_reasons:
            analysis.reject_reasons = ["model rejected without naming a reason"]
        # Positive internal evidence contradicts a claim that the entire source window
        # is fundamentally unusable. Preserve it for the comparative editor.
        if (
            analysis.salvageable
            or analysis.best_internal_moment_present
            or analysis.payoff_inside_candidate
            or not has_strong_fatal_evidence(analysis)
        ):
            analysis.editorial_class = "WEAK"  # type: ignore[assignment]
            analysis.quality_penalty = min(analysis.quality_penalty, -30.0)
            analysis.recovered_for_comparison = True
            analysis.reject_reasons.append(
                "fatal rejection downgraded: source window contains recoverable evidence"
            )
        return analysis

    if is_vague_reason(analysis.reason_to_watch):
        if analysis.reason_to_watch is None:
            analysis.reject_reasons.append("no reason to watch was produced")
        else:
            analysis.reject_reasons.append(
                "reason to watch describes continuation, not a concrete moment"
            )
        # An unconvincing absolute rationale is a quality penalty. It is not evidence
        # that comparison cannot find the best moment in a mediocre-looking pool.
        analysis.editorial_class = "WEAK"  # type: ignore[assignment]
        analysis.quality_penalty = min(analysis.quality_penalty, -25.0)
    return analysis


# Backwards-compatible public name used by callers and older integrations.
apply_reason_gate = apply_editorial_safety_gates


def recover_pathological_rejection_distribution(
    retrieval_ids: List[str],
    analyses: Dict[str, EditorialAnalysis],
) -> tuple[List[str], Optional[str]]:
    """Recover borderline fatal calls when an absolute judge collapses the pool.

    This is deliberately conditional, not an always-keep-N quota. It activates only
    when fewer than a quarter of a meaningful retrieval set survives, and it never
    restores a high-confidence, explicitly evidenced, non-salvageable fatal window.
    """
    survivors = [cid for cid in retrieval_ids if survives_reject_filter(analyses[cid])]
    count = len(retrieval_ids)
    pre_recovered = sum(analyses[cid].recovered_for_comparison for cid in retrieval_ids)
    fatal_claims = sum(
        analyses[cid].editorial_class == "FATAL_REJECT" for cid in retrieval_ids
    ) + pre_recovered
    pool_floor = max(2, math.ceil(count * 0.25))
    pathological = len(survivors) < pool_floor or fatal_claims >= math.ceil(count * 0.75)
    if count < 4 or not pathological:
        return survivors, None

    desired_pool = min(16, max(2, math.ceil(count * 0.25)))
    recoverable: List[str] = []
    for cid in retrieval_ids:
        analysis = analyses[cid]
        if analysis.editorial_class != "FATAL_REJECT":
            continue
        strongly_fatal = (
            not analysis.salvageable
            and not analysis.best_internal_moment_present
            and not analysis.payoff_inside_candidate
            and has_strong_fatal_evidence(analysis)
            and analysis.confidence >= 0.85
        )
        if not strongly_fatal:
            recoverable.append(cid)

    recoverable.sort(
        key=lambda cid: (
            not analyses[cid].best_internal_moment_present,
            not analyses[cid].payoff_inside_candidate,
            not analyses[cid].salvageable,
            -analyses[cid].standalone_after_refinement_probability,
            -analyses[cid].scroll_stop,
            analyses[cid].dead_air,
            cid,
        )
    )
    restored = recoverable[: max(0, desired_pool - len(survivors))]
    for cid in restored:
        analysis = analyses[cid]
        has_positive_evidence = (
            analysis.salvageable
            or analysis.best_internal_moment_present
            or analysis.payoff_inside_candidate
        )
        analysis.editorial_class = "MAYBE" if has_positive_evidence else "WEAK"  # type: ignore[assignment]
        analysis.quality_penalty = min(analysis.quality_penalty, -35.0)
        analysis.recovered_for_comparison = True
        analysis.reject_reasons.append("recovered after pathological rejection distribution")

    survivors = [cid for cid in retrieval_ids if survives_reject_filter(analyses[cid])]
    total_recovered = pre_recovered + len(restored)
    warning = (
        "rejection_distribution_warning: absolute editorial assessment attempted to reject "
        f"{fatal_claims}/{count}; recovered {total_recovered} borderline windows for a "
        f"comparative pool of {len(survivors)}."
    )
    return survivors, warning


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
