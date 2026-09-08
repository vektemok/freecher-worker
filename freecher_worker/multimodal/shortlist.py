"""Deterministic shortlist generation for multimodal highlight reranking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from freecher_worker.highlights.models import CandidateDocument
from freecher_worker.evaluation.models import ScorerPredictionDocument
from freecher_worker.utils.json_io import load_json, save_json
from .models import ShortlistDocument, ShortlistItem

logger = logging.getLogger("freecher_worker")


def generate_shortlist(
    run_dir: Path | str,
    heuristic_top_k: Optional[int] = None,
    llm_top_k: Optional[int] = None,
    max_candidates: Optional[int] = None,
    allow_missing_llm: bool = False,
    output_file: Optional[Path | str] = None,
    shortlist_version: str = "v1",
) -> ShortlistDocument:
    """Generate a deterministic shortlist of candidates without using human evaluation labels.

    Args:
        run_dir: Path to run directory.
        heuristic_top_k: Number of top candidates from heuristic_v1 (default 12 for v1, 20 for v1.1).
        llm_top_k: Number of top candidates from highlight_v2_1 (default 12 for v1, 20 for v1.1).
        max_candidates: Maximum shortlist capacity (default 20 for v1, 32 for v1.1).
        allow_missing_llm: If False (benchmark mode), raises FileNotFoundError if highlight_v2_1 is missing.
        output_file: Optional custom path for saving shortlist JSON.
        shortlist_version: "v1" (alternating Top-12) or "v1_1" (high-recall Top-20 union with rank provenance).

    Returns:
        ShortlistDocument with candidate_ids and metadata.
    """
    is_v1_1 = shortlist_version in ("v1_1", "multimodal_v1_1")
    actual_heur_top_k = heuristic_top_k if heuristic_top_k is not None else (20 if is_v1_1 else 12)
    actual_llm_top_k = llm_top_k if llm_top_k is not None else (20 if is_v1_1 else 12)
    actual_max_candidates = max_candidates if max_candidates is not None else (32 if is_v1_1 else 20)

    r_dir = Path(run_dir).resolve()
    cand_file = r_dir / "candidates.json"
    if not cand_file.is_file():
        raise FileNotFoundError(f"Missing candidates file: {cand_file}")

    cand_doc = CandidateDocument.model_validate(load_json(cand_file))
    valid_cand_ids = {c.id for c in cand_doc.candidates}

    # 1. Load heuristic_v1 predictions
    heur_file = r_dir / "scores" / "heuristic_v1.json"
    if not heur_file.is_file():
        raise FileNotFoundError(
            f"Missing heuristic predictions at {heur_file}. "
            "Benchmark mode requires existing scores/heuristic_v1.json."
        )

    heur_doc = ScorerPredictionDocument.model_validate(load_json(heur_file))
    sorted_heur = sorted(heur_doc.predictions, key=lambda p: (p.rank, -p.score))
    heur_candidates = [p.candidate_id for p in sorted_heur if p.candidate_id in valid_cand_ids]
    top_heur_ids = heur_candidates[:actual_heur_top_k]

    # 2. Load highlight_v2_1 predictions
    v2_1_file = r_dir / "scores" / "highlight_v2_1.json"
    top_llm_ids: List[str] = []
    if not v2_1_file.is_file():
        if not allow_missing_llm:
            raise FileNotFoundError(
                f"Missing highlight_v2_1 predictions at {v2_1_file}. "
                "Benchmark mode requires existing scores/highlight_v2_1.json without silent substitution."
            )
        logger.warning(
            f"[multimodal-shortlist] {v2_1_file} not found. Running in heuristic-only retrieval mode."
        )
    else:
        llm_doc = ScorerPredictionDocument.model_validate(load_json(v2_1_file))

        def _llm_sort_key(p):
            q = getattr(p, "llm_quality_score", None)
            f = getattr(p, "final_score", None)
            return (-(q if q is not None else (f if f is not None else p.score)), p.rank)

        sorted_llm = sorted(llm_doc.predictions, key=_llm_sort_key)
        llm_candidates = [p.candidate_id for p in sorted_llm if p.candidate_id in valid_cand_ids]
        top_llm_ids = llm_candidates[:actual_llm_top_k]

    if is_v1_1:
        # High-recall strategy (v1.1)
        heur_rank_map = {cid: rank for rank, cid in enumerate(top_heur_ids, start=1)}
        llm_rank_map = {cid: rank for rank, cid in enumerate(top_llm_ids, start=1)}
        all_unique_cids = sorted(list(set(top_heur_ids).union(set(top_llm_ids))))

        shortlist_items: List[ShortlistItem] = []
        for cid in all_unique_cids:
            h_actual = heur_rank_map.get(cid)
            l_actual = llm_rank_map.get(cid)

            # Sorting ranks: missing rank = respective top_k + 1
            h_sort = h_actual if h_actual is not None else (actual_heur_top_k + 1)
            l_sort = l_actual if l_actual is not None else (actual_llm_top_k + 1)

            best_rank = min(h_sort, l_sort)
            sum_rank = h_sort + l_sort

            sources: List[str] = []
            if h_actual is not None:
                sources.append("heuristic_v1")
            if l_actual is not None:
                sources.append("highlight_v2_1")

            shortlist_items.append(
                ShortlistItem(
                    candidate_id=cid,
                    heuristic_rank=h_actual,
                    llm_rank=l_actual,
                    best_rank=best_rank,
                    sum_rank=sum_rank,
                    retrieval_sources=sources,
                )
            )

        # Sort deterministically by (best_rank, sum_rank, candidate_id)
        shortlist_items.sort(key=lambda item: (item.best_rank, item.sum_rank, item.candidate_id))
        final_items = shortlist_items[:actual_max_candidates]
        final_shortlist_ids = [it.candidate_id for it in final_items]
        strategy_name = "high_recall_union_v1_1"

        shortlist_doc = ShortlistDocument(
            candidate_set_id=cand_doc.candidate_set_id,
            strategy=strategy_name,
            heuristic_top_k=actual_heur_top_k,
            llm_top_k=actual_llm_top_k,
            max_candidates=actual_max_candidates,
            candidate_ids=final_shortlist_ids,
            total_unique=len(final_shortlist_ids),
            items=final_items,
        )
        default_filename = "shortlist_v1_1.json"
    else:
        # Alternating strategy (v1)
        merged_ids: List[str] = []
        seen: set[str] = set()

        max_iter = max(len(top_llm_ids), len(top_heur_ids))
        for i in range(max_iter):
            if i < len(top_llm_ids):
                cid = top_llm_ids[i]
                if cid not in seen:
                    seen.add(cid)
                    merged_ids.append(cid)
            if i < len(top_heur_ids):
                cid = top_heur_ids[i]
                if cid not in seen:
                    seen.add(cid)
                    merged_ids.append(cid)

        final_shortlist_ids = merged_ids[:actual_max_candidates]
        strategy_name = "union_alternating" if top_llm_ids else "heuristic_only"

        shortlist_doc = ShortlistDocument(
            candidate_set_id=cand_doc.candidate_set_id,
            strategy=strategy_name,
            heuristic_top_k=actual_heur_top_k,
            llm_top_k=actual_llm_top_k,
            max_candidates=actual_max_candidates,
            candidate_ids=final_shortlist_ids,
            total_unique=len(final_shortlist_ids),
        )
        default_filename = "shortlist_v1.json"

    # Persist to run directory
    target_path = Path(output_file) if output_file else (r_dir / "multimodal" / default_filename)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(shortlist_doc, target_path)

    logger.info(
        f"[multimodal-shortlist] Generated shortlist of {len(final_shortlist_ids)} candidates "
        f"saved to {target_path} (Strategy: {strategy_name})"
    )
    return shortlist_doc
