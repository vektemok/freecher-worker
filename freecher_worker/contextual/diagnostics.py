"""Diagnostics for locating where good highlights are lost.

``inspect_moment`` answers "what happened to the moment a human liked at 00:18:34?".
``build_blind_diagnostic`` produces a de-identified, deterministically shuffled review
package that separates a candidate-generation problem from a retrieval problem from a
reranking problem.
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from freecher_worker.evaluation.models import ScorerPredictionDocument, ScorerPredictionItem
from freecher_worker.highlights.models import CandidateWindow

from .models import (
    BlindDiagnosticDocument,
    BlindDiagnosticItem,
    BlindDiagnosticMapping,
    BlindDiagnosticMappingEntry,
    MomentInspection,
    MomentInspectionScorerRow,
)
from .reranker import (
    load_candidate_document,
    load_scorer_document,
    load_shortlist_ids,
)
from .versions import SCORER_VERSION_CONTEXTUAL_V1

logger = logging.getLogger("freecher_worker")

NO_COVERAGE_MESSAGE = "NO CANDIDATE COVERAGE"

#: Scorers reported by inspect-moment, in pipeline order.
INSPECTED_SCORERS = (
    "heuristic_v1",
    "highlight_v2_1",
    "multimodal_v1_1",
    SCORER_VERSION_CONTEXTUAL_V1,
)

DEFAULT_BLIND_SEED = 1337


def parse_timestamp(value: str) -> float:
    """Parse ``SS``, ``MM:SS``, or ``HH:MM:SS`` (fractional seconds allowed) into seconds."""
    text = str(value).strip()
    if not text:
        raise ValueError("Empty timestamp")
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Unrecognized timestamp '{value}'. Use SS, MM:SS, or HH:MM:SS.")
    try:
        numbers = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Unrecognized timestamp '{value}'. Use SS, MM:SS, or HH:MM:SS.") from exc
    if len(numbers) == 2:
        minutes, seconds = numbers
        hours = 0.0
    else:
        hours, minutes, seconds = numbers
    return hours * 3600.0 + minutes * 60.0 + seconds


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def find_covering_candidates(
    candidates: Sequence[CandidateWindow],
    timestamp: float,
) -> List[CandidateWindow]:
    """All candidates whose window contains the timestamp, best-centred first."""
    covering = [c for c in candidates if c.start <= timestamp <= c.end]
    covering.sort(key=lambda c: (abs(((c.start + c.end) / 2.0) - timestamp), c.id))
    return covering


def inspect_moment(run_dir: Path | str, timestamp: float) -> MomentInspection:
    """Report where a human-identified timestamp landed in every pipeline stage."""
    r_dir = Path(run_dir).resolve()
    cand_doc = load_candidate_document(r_dir)

    inspection = MomentInspection(
        run_dir=str(r_dir),
        timestamp=round(timestamp, 3),
        timestamp_label=format_timestamp(timestamp),
    )

    covering = find_covering_candidates(cand_doc.candidates, timestamp)
    if not covering:
        inspection.covered = False
        inspection.message = (
            f"{NO_COVERAGE_MESSAGE}: no candidate window contains "
            f"{inspection.timestamp_label}. This is a candidate-generation gap, "
            f"not a ranking problem."
        )
        return inspection

    primary = covering[0]
    inspection.covered = True
    inspection.candidate_id = primary.id
    inspection.candidate_start = primary.start
    inspection.candidate_end = primary.end
    inspection.candidate_text = primary.text
    inspection.overlapping_candidate_ids = [c.id for c in covering]

    shortlist_ids = load_shortlist_ids(r_dir)
    if shortlist_ids is not None:
        inspection.in_retrieval_shortlist = primary.id in shortlist_ids

    for scorer in INSPECTED_SCORERS:
        doc = load_scorer_document(r_dir, scorer)
        if doc is None:
            inspection.scorers.append(MomentInspectionScorerRow(scorer=scorer, present=False))
            continue
        item = next((p for p in doc.predictions if p.candidate_id == primary.id), None)
        inspection.scorers.append(
            MomentInspectionScorerRow(
                scorer=scorer,
                rank=item.rank if item else None,
                score=item.score if item else None,
                present=item is not None,
            )
        )
        if scorer == SCORER_VERSION_CONTEXTUAL_V1 and item is not None:
            inspection.contextual_rank = item.rank
            inspection.contextual_status = item.status
            inspection.editorial_class = item.editorial_class
            inspection.reason_to_watch = item.reason_to_watch
            inspection.reject_reasons = list(item.reject_reasons or [])
            inspection.critic_result = item.critic_result

    if inspection.contextual_status and inspection.contextual_status != "ranked":
        reasons = "; ".join(inspection.reject_reasons) or "no reason recorded"
        inspection.message = (
            f"Candidate {primary.id} was rejected by the contextual reranker "
            f"({inspection.contextual_status}): {reasons}"
        )
    elif inspection.contextual_rank is not None:
        inspection.message = (
            f"Candidate {primary.id} is ranked #{inspection.contextual_rank} by "
            f"{SCORER_VERSION_CONTEXTUAL_V1}."
        )
    else:
        inspection.message = (
            f"Candidate {primary.id} covers {inspection.timestamp_label} but has no "
            f"{SCORER_VERSION_CONTEXTUAL_V1} prediction. Run contextual-rerank first."
        )
    return inspection


def _prediction_rank_map(doc: Optional[ScorerPredictionDocument]) -> Dict[str, ScorerPredictionItem]:
    if doc is None:
        return {}
    return {item.candidate_id: item for item in doc.predictions}


def select_blind_groups(
    run_dir: Path | str,
    group_a_range: Tuple[int, int] = (16, 32),
    group_a_size: int = 4,
    group_b_size: int = 4,
    group_c_size: int = 0,
    seed: int = DEFAULT_BLIND_SEED,
) -> Dict[str, List[str]]:
    """Pick the three diagnostic groups deterministically.

    A: candidates ranked inside ``group_a_range`` by multimodal_v1_1.
    B: candidates never nominated into the retrieval shortlist.
    C: optional top-ranked contextual candidates.
    """
    r_dir = Path(run_dir).resolve()
    cand_doc = load_candidate_document(r_dir)
    all_ids = [c.id for c in cand_doc.candidates]

    multimodal = load_scorer_document(r_dir, "multimodal_v1_1")
    multimodal_items = _prediction_rank_map(multimodal)
    shortlist_ids = set(load_shortlist_ids(r_dir) or multimodal_items.keys())

    low, high = group_a_range
    group_a_pool = sorted(
        [cid for cid, item in multimodal_items.items() if low <= item.rank <= high],
        key=lambda cid: (multimodal_items[cid].rank, cid),
    )
    group_b_pool = sorted(cid for cid in all_ids if cid not in shortlist_ids)

    contextual = load_scorer_document(r_dir, SCORER_VERSION_CONTEXTUAL_V1)
    contextual_items = _prediction_rank_map(contextual)
    group_c_pool = sorted(
        [
            cid
            for cid, item in contextual_items.items()
            if item.status in (None, "ranked")
        ],
        key=lambda cid: (contextual_items[cid].rank, cid),
    )

    rng = random.Random(seed)

    def take(pool: Sequence[str], size: int, ordered: bool) -> List[str]:
        if size <= 0 or not pool:
            return []
        if ordered:
            return list(pool[:size])
        picks = list(pool)
        rng.shuffle(picks)
        return sorted(picks[:size])

    return {
        "A": take(group_a_pool, group_a_size, ordered=True),
        "B": take(group_b_pool, group_b_size, ordered=False),
        "C": take(group_c_pool, group_c_size, ordered=True),
    }


def build_blind_diagnostic(
    run_dir: Path | str,
    group_a_size: int = 4,
    group_b_size: int = 4,
    group_c_size: int = 0,
    group_a_range: Tuple[int, int] = (16, 32),
    seed: int = DEFAULT_BLIND_SEED,
) -> Tuple[BlindDiagnosticDocument, BlindDiagnosticMapping]:
    """Build a blind review package and its private mapping.

    The reviewer document carries no candidate ids, no ranks, and no scores.
    Both documents are fully reproducible from ``seed``.
    """
    r_dir = Path(run_dir).resolve()
    cand_doc = load_candidate_document(r_dir)
    cand_map = {c.id: c for c in cand_doc.candidates}

    groups = select_blind_groups(
        r_dir,
        group_a_range=group_a_range,
        group_a_size=group_a_size,
        group_b_size=group_b_size,
        group_c_size=group_c_size,
        seed=seed,
    )

    multimodal_items = _prediction_rank_map(load_scorer_document(r_dir, "multimodal_v1_1"))
    contextual_items = _prediction_rank_map(
        load_scorer_document(r_dir, SCORER_VERSION_CONTEXTUAL_V1)
    )

    # Deduplicate across groups, first group wins, in a fixed group order.
    selected: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for group in ("A", "B", "C"):
        for cid in groups.get(group, []):
            if cid in seen or cid not in cand_map:
                continue
            seen.add(cid)
            selected.append((cid, group))

    rng = random.Random(seed)
    shuffled = list(selected)
    rng.shuffle(shuffled)

    items: List[BlindDiagnosticItem] = []
    entries: List[BlindDiagnosticMappingEntry] = []
    for index, (cid, group) in enumerate(shuffled, start=1):
        candidate = cand_map[cid]
        blind_id = f"blind_{index:03d}"
        items.append(
            BlindDiagnosticItem(
                blind_id=blind_id,
                start=candidate.start,
                end=candidate.end,
                duration=candidate.duration,
                text=candidate.text,
            )
        )
        mm = multimodal_items.get(cid)
        ctx = contextual_items.get(cid)
        entries.append(
            BlindDiagnosticMappingEntry(
                blind_id=blind_id,
                candidate_id=cid,
                group=group,
                multimodal_rank=mm.rank if mm else None,
                multimodal_score=mm.score if mm else None,
                contextual_rank=ctx.rank if ctx else None,
                contextual_status=ctx.status if ctx else None,
                editorial_class=ctx.editorial_class if ctx else None,
            )
        )

    document = BlindDiagnosticDocument(
        candidate_set_id=cand_doc.candidate_set_id,
        seed=seed,
        total_items=len(items),
        items=items,
    )
    mapping = BlindDiagnosticMapping(
        candidate_set_id=cand_doc.candidate_set_id,
        seed=seed,
        group_sizes={group: len(ids) for group, ids in groups.items()},
        entries=entries,
    )
    return document, mapping
