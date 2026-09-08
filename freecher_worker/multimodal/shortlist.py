"""Deterministic shortlist generation for multimodal highlight reranking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from freecher_worker.highlights.models import CandidateDocument
from freecher_worker.evaluation.models import ScorerPredictionDocument
from freecher_worker.utils.json_io import load_json, save_json
from .models import ShortlistDocument

logger = logging.getLogger("freecher_worker")


def generate_shortlist(
    run_dir: Path | str,
    heuristic_top_k: int = 12,
    llm_top_k: int = 12,
    max_candidates: int = 20,
    allow_missing_llm: bool = False,
    output_file: Optional[Path | str] = None,
) -> ShortlistDocument:
    """Generate a deterministic shortlist of candidates without using human evaluation labels.

    Merges Top-K candidates from heuristic_v1 and Top-K candidates from highlight_v2_1 using
    deterministic alternating priority, ensuring no unordered set behavior.

    Args:
        run_dir: Path to run directory.
        heuristic_top_k: Number of top candidates to take from heuristic_v1 (default 12).
        llm_top_k: Number of top candidates to take from highlight_v2_1 (default 12).
        max_candidates: Maximum shortlist capacity (default 20).
        allow_missing_llm: If False (benchmark mode), raises FileNotFoundError if highlight_v2_1 is missing.
        output_file: Optional custom path for saving shortlist_v1.json.

    Returns:
        ShortlistDocument with candidate_ids and metadata.
    """
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
    # Sort heuristic by rank ascending (or score descending)
    sorted_heur = sorted(heur_doc.predictions, key=lambda p: (p.rank, -p.score))
    heur_candidates = [p.candidate_id for p in sorted_heur if p.candidate_id in valid_cand_ids]
    top_heur_ids = heur_candidates[:heuristic_top_k]

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
        # Sort LLM candidates by llm_quality_score descending if present, fallback to final_score/score
        def _llm_sort_key(p):
            q = getattr(p, "llm_quality_score", None)
            f = getattr(p, "final_score", None)
            return (-(q if q is not None else (f if f is not None else p.score)), p.rank)

        sorted_llm = sorted(llm_doc.predictions, key=_llm_sort_key)
        llm_candidates = [p.candidate_id for p in sorted_llm if p.candidate_id in valid_cand_ids]
        top_llm_ids = llm_candidates[:llm_top_k]

    # 3. Deterministic alternating merge (no unordered set behavior)
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

    # 4. Deterministic truncation to max_candidates
    final_shortlist_ids = merged_ids[:max_candidates]

    strategy_name = "union_alternating" if top_llm_ids else "heuristic_only"

    shortlist_doc = ShortlistDocument(
        candidate_set_id=cand_doc.candidate_set_id,
        strategy=strategy_name,
        heuristic_top_k=heuristic_top_k,
        llm_top_k=llm_top_k,
        max_candidates=max_candidates,
        candidate_ids=final_shortlist_ids,
        total_unique=len(final_shortlist_ids),
    )

    # 5. Persist to run directory
    target_path = Path(output_file) if output_file else (r_dir / "multimodal" / "shortlist_v1.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(shortlist_doc, target_path)

    logger.info(
        f"[multimodal-shortlist] Generated shortlist of {len(final_shortlist_ids)} candidates "
        f"saved to {target_path} (Strategy: {strategy_name})"
    )
    return shortlist_doc
