"""Comparative reranking for contextual_reranker_v1.

Survivors are ordered by comparing them against each other, never by summing their
observability numbers. The schedule is O(N log N):

    listwise batches (N / batch_size calls)
        -> Swiss tournament, ceil(log2(N)) rounds of N/2 pairings
        -> round-robin among the top group for pairs not yet played

For N = 16 that is ~39 comparisons instead of the 120 a full pairwise matrix needs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cache import ContextualCache
from .candidate_context import render_candidate_digest
from .models import (
    EDITORIAL_CLASS_ORDER,
    CandidateContextPackage,
    ComparisonResult,
    EditorialAnalysis,
    ListwiseBatchResult,
)
from .prompts import get_stage_prompt
from .provider import ContextualProvider, ContextualProviderError
from .versions import COMPARISON_SCHEMA_VERSION, LISTWISE_SCHEMA_VERSION, RANKING_ALGORITHM_VERSION

logger = logging.getLogger("freecher_worker")

COMPARISON_MODES = ("full", "swiss", "listwise", "none")

DEFAULT_LISTWISE_BATCH_SIZE = 7
DEFAULT_FINAL_PAIRWISE_TOP = 4

WIN_POINTS = 1.0
TIE_POINTS = 0.5


@dataclass
class ComparativeRanking:
    """Outcome of the comparative stage."""

    ordered_ids: List[str] = field(default_factory=list)
    points: Dict[str, float] = field(default_factory=dict)
    wins: Dict[str, int] = field(default_factory=dict)
    losses: Dict[str, int] = field(default_factory=dict)
    ties: Dict[str, int] = field(default_factory=dict)
    listwise_points: Dict[str, float] = field(default_factory=dict)
    head_to_head: Dict[str, Dict[str, str]] = field(default_factory=dict)
    comparisons: List[ComparisonResult] = field(default_factory=list)
    listwise_batches: List[ListwiseBatchResult] = field(default_factory=list)
    algorithm_version: str = RANKING_ALGORITHM_VERSION
    mode: str = "full"


def seed_order(
    survivors: Sequence[str],
    analyses: Dict[str, EditorialAnalysis],
) -> List[str]:
    """Deterministic starting order: editorial class, then scroll_stop, then id."""

    def key(cid: str) -> Tuple[int, float, str]:
        analysis = analyses.get(cid)
        cls_rank = EDITORIAL_CLASS_ORDER.get(analysis.editorial_class, 0) if analysis else 0
        scroll = analysis.scroll_stop if analysis else 0.0
        return (-cls_rank, -scroll, cid)

    return sorted(survivors, key=key)


def _canonical_pair(a: str, b: str) -> Tuple[str, str]:
    """Always compare a pair in one orientation so the result is cacheable and stable."""
    return (a, b) if a <= b else (b, a)


def build_listwise_batches(
    ordered: Sequence[str],
    batch_size: int,
) -> List[List[str]]:
    """Deal the seeded order round-robin into batches so each batch mixes strengths."""
    ids = list(ordered)
    if not ids:
        return []
    batch_size = max(2, batch_size)
    if len(ids) <= batch_size:
        return [ids]

    batch_count = math.ceil(len(ids) / batch_size)
    batches: List[List[str]] = [[] for _ in range(batch_count)]
    for index, cid in enumerate(ids):
        batches[index % batch_count].append(cid)
    return [b for b in batches if len(b) >= 2] or [ids]


def _listwise_points(ordering: Sequence[str]) -> Dict[str, float]:
    """Normalized Borda points in [0, 1] for one batch ordering."""
    size = len(ordering)
    if size <= 1:
        return {cid: 1.0 for cid in ordering}
    return {
        cid: round((size - 1 - position) / (size - 1), 4)
        for position, cid in enumerate(ordering)
    }


def run_listwise_batch(
    batch_index: int,
    batch: Sequence[str],
    packages: Dict[str, CandidateContextPackage],
    provider: Optional[ContextualProvider],
    cache: ContextualCache,
) -> ListwiseBatchResult:
    """Order one small batch of candidates with a single request."""
    prompt_version, prompt_hash, system_prompt = get_stage_prompt("listwise")
    ids = list(batch)

    payload = {
        "candidates": [
            {"candidate_id": cid, "package_hash": packages[cid].package_hash}
            for cid in ids
            if cid in packages
        ]
    }
    req_hash = cache.request_hash(
        stage="listwise",
        prompt_version=prompt_version,
        prompt_hash=prompt_hash,
        schema_version=LISTWISE_SCHEMA_VERSION,
        payload=payload,
    )

    parsed: Optional[Dict[str, Any]] = cache.load("listwise", req_hash)
    error: Optional[str] = None

    if parsed is not None:
        if provider is not None:
            provider.note_cache_hit("listwise")
    elif provider is None:
        error = "no provider configured"
    else:
        blocks = [
            render_candidate_digest(packages[cid], label=f"CANDIDATE {cid}")
            for cid in ids
            if cid in packages
        ]
        user_content = "\n\n".join(
            [
                "Order these candidate clips from the same video, best first.",
                "Candidate ids to order (use each exactly once): " + ", ".join(ids),
                "",
                *blocks,
            ]
        )
        try:
            parsed = provider.complete_json(system_prompt, user_content, stage="listwise")
            cache.store("listwise", req_hash, parsed)
        except (ContextualProviderError, ValueError) as exc:
            logger.warning(f"[contextual-listwise] Batch {batch_index} failed: {exc}")
            error = str(exc)[:300]

    ordering: List[str] = []
    reason: Optional[str] = None
    parse_failed = False

    if parsed is not None:
        raw = parsed.get("ordering")
        if isinstance(raw, list):
            seen = set()
            for entry in raw:
                cid = str(entry).strip()
                if cid in ids and cid not in seen:
                    seen.add(cid)
                    ordering.append(cid)
        reason = str(parsed.get("reason") or "").strip() or None

    missing = [cid for cid in ids if cid not in ordering]
    if missing:
        # Any id the model dropped keeps its seeded position at the tail.
        parse_failed = bool(ordering) or parsed is not None
        ordering.extend(missing)
    if not parsed:
        parse_failed = True

    return ListwiseBatchResult(
        batch_index=batch_index,
        candidate_ids=ids,
        ordering=ordering,
        reason=reason,
        request_hash=req_hash,
        parse_failed=parse_failed,
        error=error,
    )


def run_pairwise_comparison(
    candidate_a: str,
    candidate_b: str,
    packages: Dict[str, CandidateContextPackage],
    provider: Optional[ContextualProvider],
    cache: ContextualCache,
    stage_label: str,
) -> ComparisonResult:
    """Compare two survivors head to head. Never called with a == b."""
    if candidate_a == candidate_b:
        raise ValueError(f"Refusing to compare candidate '{candidate_a}' with itself.")

    prompt_version, prompt_hash, system_prompt = get_stage_prompt("pairwise")
    a_pkg = packages.get(candidate_a)
    b_pkg = packages.get(candidate_b)

    payload = {
        "a": {"candidate_id": candidate_a, "package_hash": a_pkg.package_hash if a_pkg else None},
        "b": {"candidate_id": candidate_b, "package_hash": b_pkg.package_hash if b_pkg else None},
    }
    req_hash = cache.request_hash(
        stage="pairwise",
        prompt_version=prompt_version,
        prompt_hash=prompt_hash,
        schema_version=COMPARISON_SCHEMA_VERSION,
        payload=payload,
    )

    parsed: Optional[Dict[str, Any]] = cache.load("pairwise", req_hash)
    error: Optional[str] = None

    if parsed is not None:
        if provider is not None:
            provider.note_cache_hit("pairwise")
    elif provider is None or a_pkg is None or b_pkg is None:
        error = "no provider configured" if provider is None else "missing candidate package"
    else:
        user_content = "\n\n".join(
            [
                "Two candidate clips from the same video. Choose the one a cold viewer is "
                "more likely to watch to the end.",
                render_candidate_digest(a_pkg, label="CANDIDATE A"),
                render_candidate_digest(b_pkg, label="CANDIDATE B"),
            ]
        )
        try:
            parsed = provider.complete_json(system_prompt, user_content, stage="pairwise")
            cache.store("pairwise", req_hash, parsed)
        except (ContextualProviderError, ValueError) as exc:
            logger.warning(
                f"[contextual-pairwise] Comparison {candidate_a} vs {candidate_b} failed: {exc}"
            )
            error = str(exc)[:300]

    winner = "TIE"
    confidence = 0.0
    reason = ""
    a_strength = ""
    b_strength = ""
    parse_failed = True

    if parsed is not None:
        raw_winner = str(parsed.get("winner") or "").strip().upper()
        if raw_winner in ("A", "B", "TIE"):
            winner = raw_winner
            parse_failed = False
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence / 100.0 if confidence > 1.0 else confidence))
        reason = str(parsed.get("reason") or "").strip()
        a_strength = str(parsed.get("a_strength") or "").strip()
        b_strength = str(parsed.get("b_strength") or "").strip()

    return ComparisonResult(
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        winner=winner,  # type: ignore[arg-type]
        confidence=confidence,
        reason=reason,
        a_strength=a_strength,
        b_strength=b_strength,
        stage=stage_label,
        request_hash=req_hash,
        parse_failed=parse_failed,
        error=error,
    )


def _pair_round(
    standings: Sequence[str],
    played: set[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """Pair adjacent players in the standings, skipping rematches where possible."""
    remaining = list(standings)
    pairs: List[Tuple[str, str]] = []

    while len(remaining) >= 2:
        first = remaining.pop(0)
        opponent_index = None
        for index, other in enumerate(remaining):
            if _canonical_pair(first, other) not in played:
                opponent_index = index
                break
        if opponent_index is None:
            # Every remaining opponent was already faced. Pair with the nearest one so the
            # round still consumes the standings; the caller drops the duplicate pairing.
            opponent_index = 0
        opponent = remaining.pop(opponent_index)
        pairs.append((first, opponent))

    bye = remaining[0] if remaining else None
    return pairs, bye


def run_comparative_ranking(
    survivors: Sequence[str],
    packages: Dict[str, CandidateContextPackage],
    analyses: Dict[str, EditorialAnalysis],
    provider: Optional[ContextualProvider],
    cache: ContextualCache,
    mode: str = "full",
    listwise_batch_size: int = DEFAULT_LISTWISE_BATCH_SIZE,
    final_pairwise_top: int = DEFAULT_FINAL_PAIRWISE_TOP,
) -> ComparativeRanking:
    """Rank survivors by comparing them with each other."""
    if mode not in COMPARISON_MODES:
        raise ValueError(f"Unknown comparison mode '{mode}'. Valid: {list(COMPARISON_MODES)}")

    ids = seed_order(survivors, analyses)
    ranking = ComparativeRanking(mode=mode)
    ranking.points = {cid: 0.0 for cid in ids}
    ranking.wins = {cid: 0 for cid in ids}
    ranking.losses = {cid: 0 for cid in ids}
    ranking.ties = {cid: 0 for cid in ids}
    ranking.listwise_points = {cid: 0.0 for cid in ids}
    ranking.head_to_head = {cid: {} for cid in ids}

    if not ids:
        return ranking
    if mode == "none" or len(ids) == 1:
        ranking.ordered_ids = ids
        return ranking

    seed_index = {cid: position for position, cid in enumerate(ids)}

    # --- Stage A: listwise batches -----------------------------------------------------
    if mode in ("full", "listwise"):
        for batch_index, batch in enumerate(
            build_listwise_batches(ids, listwise_batch_size), start=1
        ):
            result = run_listwise_batch(batch_index, batch, packages, provider, cache)
            ranking.listwise_batches.append(result)
            for cid, points in _listwise_points(result.ordering).items():
                ranking.listwise_points[cid] = points

    if mode == "listwise":
        ranking.ordered_ids = _final_order(ids, ranking, analyses, seed_index)
        return ranking

    # --- Stage B: Swiss tournament ------------------------------------------------------
    played: set[Tuple[str, str]] = set()
    byes: set[str] = set()
    rounds = max(1, math.ceil(math.log2(len(ids))))

    for round_index in range(1, rounds + 1):
        standings = sorted(
            ids,
            key=lambda cid: (
                -ranking.points[cid],
                -ranking.listwise_points[cid],
                seed_index[cid],
            ),
        )
        # Give the bye to the lowest-standing player who has not had one.
        if len(standings) % 2 == 1:
            for cid in reversed(standings):
                if cid not in byes:
                    byes.add(cid)
                    standings = [c for c in standings if c != cid]
                    break
            else:
                standings = standings[:-1]

        pairs, _ = _pair_round(standings, played)
        for left, right in pairs:
            a, b = _canonical_pair(left, right)
            if (a, b) in played:
                # Every possible opponent was already faced; a rematch would double-count.
                continue
            played.add((a, b))
            result = run_pairwise_comparison(
                a, b, packages, provider, cache, stage_label=f"swiss_round_{round_index}"
            )
            _apply_comparison(ranking, result)

    # --- Stage C: round-robin among the top group ---------------------------------------
    if mode == "full" and final_pairwise_top >= 2:
        standings = sorted(
            ids,
            key=lambda cid: (
                -ranking.points[cid],
                -ranking.listwise_points[cid],
                seed_index[cid],
            ),
        )
        top_group = standings[: min(final_pairwise_top, len(standings))]
        for i in range(len(top_group)):
            for j in range(i + 1, len(top_group)):
                a, b = _canonical_pair(top_group[i], top_group[j])
                if (a, b) in played:
                    continue
                played.add((a, b))
                result = run_pairwise_comparison(
                    a, b, packages, provider, cache, stage_label="final_pairwise"
                )
                _apply_comparison(ranking, result)

    ranking.ordered_ids = _final_order(ids, ranking, analyses, seed_index)
    return ranking


def _apply_comparison(ranking: ComparativeRanking, result: ComparisonResult) -> None:
    """Fold one comparison outcome into the standings."""
    ranking.comparisons.append(result)
    a, b = result.candidate_a, result.candidate_b
    if result.winner == "A":
        ranking.points[a] += WIN_POINTS
        ranking.wins[a] += 1
        ranking.losses[b] += 1
        ranking.head_to_head[a][b] = "WIN"
        ranking.head_to_head[b][a] = "LOSS"
    elif result.winner == "B":
        ranking.points[b] += WIN_POINTS
        ranking.wins[b] += 1
        ranking.losses[a] += 1
        ranking.head_to_head[b][a] = "WIN"
        ranking.head_to_head[a][b] = "LOSS"
    else:
        ranking.points[a] += TIE_POINTS
        ranking.points[b] += TIE_POINTS
        ranking.ties[a] += 1
        ranking.ties[b] += 1
        ranking.head_to_head[a][b] = "TIE"
        ranking.head_to_head[b][a] = "TIE"


def _head_to_head_score(ranking: ComparativeRanking, cid: str, group: Sequence[str]) -> float:
    """Points won against the other members of a tied group."""
    total = 0.0
    for other in group:
        if other == cid:
            continue
        outcome = ranking.head_to_head.get(cid, {}).get(other)
        if outcome == "WIN":
            total += WIN_POINTS
        elif outcome == "TIE":
            total += TIE_POINTS
    return total


def _final_order(
    ids: Sequence[str],
    ranking: ComparativeRanking,
    analyses: Dict[str, EditorialAnalysis],
    seed_index: Dict[str, int],
) -> List[str]:
    """Deterministic total order: points, head-to-head, listwise, editorial, seed."""
    tie_groups: Dict[float, List[str]] = {}
    for cid in ids:
        tie_groups.setdefault(ranking.points[cid], []).append(cid)

    def key(cid: str) -> Tuple[float, float, float, int, float, int]:
        analysis = analyses.get(cid)
        group = tie_groups[ranking.points[cid]]
        return (
            -ranking.points[cid],
            -_head_to_head_score(ranking, cid, group),
            -ranking.listwise_points[cid],
            -EDITORIAL_CLASS_ORDER.get(analysis.editorial_class, 0) if analysis else 0,
            -(analysis.scroll_stop if analysis else 0.0),
            seed_index[cid],
        )

    return sorted(ids, key=key)
