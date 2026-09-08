"""False-positive critic pass for contextual_reranker_v1.

The critic sees only the surviving candidates and looks exclusively for reasons to cut
them. It never sees human ratings, model scores, or upstream ranks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cache import ContextualCache
from .candidate_context import render_candidate_prompt
from .models import CandidateContextPackage, CriticResult, EditorialAnalysis, GlobalContext
from .prompts import get_stage_prompt
from .provider import ContextualProvider, ContextualProviderError
from .versions import CRITIC_SCHEMA_VERSION

logger = logging.getLogger("freecher_worker")


def parse_critic_response(
    parsed: Dict[str, Any],
    candidate_id: str,
    model: Optional[str],
    prompt_version: Optional[str],
) -> CriticResult:
    """Turn a raw critic JSON object into a validated CriticResult."""
    raw = str(parsed.get("decision") or "").strip().upper()
    if raw in ("KEEP", "REJECT"):
        decision = raw
    elif raw in ("YES", "TRUE", "PUBLISH"):
        decision = "KEEP"
    elif raw in ("NO", "FALSE", "CUT", "DROP"):
        decision = "REJECT"
    else:
        # An unreadable verdict must not silently cut a candidate.
        decision = "KEEP"

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence / 100.0 if confidence > 1.0 else confidence))

    return CriticResult(
        candidate_id=candidate_id,
        decision=decision,  # type: ignore[arg-type]
        reason=str(parsed.get("reason") or "").strip(),
        confidence=confidence,
        prompt_version=prompt_version,
        model=model,
        parse_failed=raw not in ("KEEP", "REJECT"),
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
        if verdict.decision == "KEEP":
            kept.append(cid)
    return kept, verdicts


def guard_against_empty_survivors(
    kept: Sequence[str],
    survivors: Sequence[str],
    analyses: Dict[str, EditorialAnalysis],
) -> Tuple[List[str], Optional[str]]:
    """Keep the pipeline usable if the critic rejects everything.

    Returns (final ids, warning). The strongest editorial candidates are restored so a
    run never produces an empty ranking, and the caller reports why.
    """
    if kept:
        return list(kept), None
    if not survivors:
        return [], None

    from .models import EDITORIAL_CLASS_ORDER

    ordered = sorted(
        survivors,
        key=lambda cid: (
            -EDITORIAL_CLASS_ORDER.get(
                analyses[cid].editorial_class if cid in analyses else "WEAK", 0
            ),
            -(analyses[cid].scroll_stop if cid in analyses else 0.0),
            cid,
        ),
    )
    restored = ordered[: min(3, len(ordered))]
    warning = (
        f"Critic rejected all {len(survivors)} survivors; restored the "
        f"{len(restored)} strongest editorial candidates so the ranking is not empty."
    )
    return restored, warning
