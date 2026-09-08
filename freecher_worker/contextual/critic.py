"""Penalty-oriented critic pass for the contextual reranker.

The critic normally demotes questionable candidates without deleting them. A hard
rejection requires an explicit unusability claim, concrete fatal evidence, and high
confidence.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cache import ContextualCache
from .candidate_context import render_candidate_prompt
from .models import CandidateContextPackage, CriticResult, GlobalContext
from .prompts import get_stage_prompt
from .provider import ContextualProvider, ContextualProviderError
from .versions import CRITIC_SCHEMA_VERSION

logger = logging.getLogger("freecher_worker")

HARD_REJECT_CONFIDENCE = 0.85


def _coerce_penalty(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number > 0:
        number = -number
    return max(-100.0, min(0.0, number))


def _string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()][:8]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


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


def _has_strong_fatal_evidence(failure_modes: List[str], fatal_evidence: List[str]) -> bool:
    evidence = " ".join([*failure_modes, *fatal_evidence]).lower()
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
    return ("severe" in evidence or "unintelligible" in evidence) and (
        "transcript" in evidence or "asr" in evidence
    ) and ("no visual" in evidence or "no useful visual" in evidence)


def parse_critic_response(
    parsed: Dict[str, Any],
    candidate_id: str,
    model: Optional[str],
    prompt_version: Optional[str],
) -> CriticResult:
    """Turn a raw critic JSON object into a validated CriticResult."""
    raw = str(parsed.get("decision") or "").strip().upper()

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence / 100.0 if confidence > 1.0 else confidence))

    failure_modes = _string_list(parsed.get("failure_modes"))
    fatal_evidence = _string_list(
        parsed.get("fatal_evidence", parsed.get("fatal_reject_evidence"))
    )
    explicit_hard_reject = _coerce_bool(
        parsed.get("hard_reject", parsed.get("fatal_reject", False))
    )
    requested_keep = _coerce_bool(parsed.get("keep_for_comparison", True), default=True)

    # Legacy binary REJECT is translated into a demotion. Only the v1.1 three-part
    # contract can delete: explicit hard flag + evidence + high confidence.
    hard_reject = (
        explicit_hard_reject
        and not bool(requested_keep)
        and _has_strong_fatal_evidence(failure_modes, fatal_evidence)
        and confidence >= HARD_REJECT_CONFIDENCE
    )
    decision = "REJECT" if hard_reject else "KEEP"
    default_penalty = -30.0 if raw in ("REJECT", "NO", "FALSE", "CUT", "DROP") else 0.0
    penalty = _coerce_penalty(
        parsed.get("penalty", parsed.get("critic_penalty", default_penalty)),
        default=default_penalty,
    )

    recognized = raw in ("", "KEEP", "REJECT", "YES", "TRUE", "PUBLISH", "NO", "FALSE", "CUT", "DROP")
    new_contract_present = any(
        key in parsed for key in ("penalty", "critic_penalty", "failure_modes", "keep_for_comparison")
    )

    return CriticResult(
        candidate_id=candidate_id,
        decision=decision,  # type: ignore[arg-type]
        reason=str(parsed.get("reason") or "").strip(),
        confidence=confidence,
        penalty=penalty,
        failure_modes=failure_modes,
        keep_for_comparison=not hard_reject,
        hard_reject=hard_reject,
        fatal_evidence=fatal_evidence,
        prompt_version=prompt_version,
        model=model,
        parse_failed=not recognized or (not raw and not new_contract_present),
    )


def criticize_candidate(
    package: CandidateContextPackage,
    global_context: GlobalContext,
    provider: Optional[ContextualProvider],
    cache: ContextualCache,
) -> CriticResult:
    """Run (or reuse a cached) critic verdict for one surviving candidate."""
    prompt_version, prompt_hash, system_prompt = get_stage_prompt("critic")
    model = getattr(provider, "model", None)

    req_hash = cache.request_hash(
        stage="critic",
        prompt_version=prompt_version,
        prompt_hash=prompt_hash,
        schema_version=CRITIC_SCHEMA_VERSION,
        payload={"package_hash": package.package_hash, "candidate_id": package.candidate_id},
    )

    cached = cache.load("critic", req_hash)
    if cached is not None:
        if provider is not None:
            provider.note_cache_hit("critic")
        return parse_critic_response(cached, package.candidate_id, model, prompt_version)

    if provider is None:
        return CriticResult(
            candidate_id=package.candidate_id,
            decision="KEEP",
            reason="Critic skipped: no provider configured.",
            confidence=0.0,
            penalty=0.0,
            keep_for_comparison=True,
            prompt_version=prompt_version,
            model=model,
            parse_failed=True,
            error="no provider configured",
        )

    user_content = render_candidate_prompt(package, global_context)
    try:
        parsed = provider.complete_json(system_prompt, user_content, stage="critic")
    except (ContextualProviderError, ValueError) as exc:
        logger.warning(f"[contextual-critic] Critic failed for {package.candidate_id}: {exc}")
        return CriticResult(
            candidate_id=package.candidate_id,
            decision="KEEP",
            reason="Critic unavailable; candidate retained.",
            confidence=0.0,
            penalty=0.0,
            keep_for_comparison=True,
            prompt_version=prompt_version,
            model=model,
            parse_failed=True,
            error=str(exc)[:300],
        )

    cache.store("critic", req_hash, parsed)
    return parse_critic_response(parsed, package.candidate_id, model, prompt_version)


def run_critic_pass(
    survivors: Sequence[str],
    packages: Dict[str, CandidateContextPackage],
    global_context: GlobalContext,
    provider: Optional[ContextualProvider],
    cache: ContextualCache,
    enabled: bool = True,
) -> Tuple[List[str], Dict[str, CriticResult]]:
    """Filter survivors through the critic.

    Returns the retained candidate ids (input order preserved) and every verdict.
    """
    if not enabled:
        return list(survivors), {}

    verdicts: Dict[str, CriticResult] = {}
    kept: List[str] = []
    for cid in survivors:
        package = packages.get(cid)
        if package is None:
            kept.append(cid)
            continue
        verdict = criticize_candidate(package, global_context, provider, cache)
        verdicts[cid] = verdict
        if verdict.keep_for_comparison:
            kept.append(cid)
    return kept, verdicts
