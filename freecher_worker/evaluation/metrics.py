"""Ranking and evaluation metrics for highlight scoring models."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Union
from .models import (
    BlindEvaluationDocument,
    BlindEvaluationItem,
    EvaluationMetrics,
    ScorerPredictionDocument,
    ScorerPredictionItem,
)

RELEVANCE_THRESHOLD = 3  # Ratings 3 (good) and 4 (excellent) are considered relevant highlights


def calculate_dcg(relevances: List[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain at rank K using exponential gain 2^rel - 1."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            gain = (2.0 ** rel) - 1.0
            discount = math.log2(i + 1)
            dcg += gain / discount
    return dcg


def compute_evaluation_metrics(
    eval_doc: BlindEvaluationDocument,
    prediction_doc: ScorerPredictionDocument,
    k_values: Optional[List[int]] = None,
    hit_rate_k_values: Optional[List[int]] = None,
) -> EvaluationMetrics:
    """Compute comprehensive evaluation metrics comparing scorer predictions against human judgments.

    Fails if candidate_set_id does not match between predictions and evaluation document.
    """
    if eval_doc.candidate_set_id != prediction_doc.candidate_set_id:
        raise ValueError(
            f"Candidate set ID mismatch! Evaluation document has '{eval_doc.candidate_set_id}', "
            f"but scorer prediction document has '{prediction_doc.candidate_set_id}'."
        )

    if k_values is None:
        k_values = [5, 10]
    if hit_rate_k_values is None:
        hit_rate_k_values = [1, 3, 5]

    # Map evaluation items by candidate_id
    eval_by_id: Dict[str, BlindEvaluationItem] = {item.candidate_id: item for item in eval_doc.items}
    total_candidates = len(eval_doc.items)
    labeled_candidates = sum(1 for item in eval_doc.items if item.human_score is not None)

    # Predictions sorted by rank ascending (1, 2, 3...)
    sorted_preds = sorted(prediction_doc.predictions, key=lambda p: p.rank)

    # Extract human scores and publishable flags in ranked order
    ranked_relevances: List[int] = []
    ranked_publishable: List[bool] = []
    for pred in sorted_preds:
        item = eval_by_id.get(pred.candidate_id)
        if item is not None and item.human_score is not None:
            ranked_relevances.append(item.human_score)
        else:
            ranked_relevances.append(0)

        if item is not None and item.publishable is True:
            ranked_publishable.append(True)
        else:
            ranked_publishable.append(False)

    # Ideal ranking across the available candidate pool for nDCG calculation
    all_pool_scores = [item.human_score for item in eval_doc.items if item.human_score is not None]
    ideal_sorted_scores = sorted(all_pool_scores, reverse=True)

    precision_at_k: Dict[int, float] = {}
    ndcg_at_k: Dict[int, float] = {}
    mean_human_score_at_k: Dict[int, float] = {}
    publishable_rate_at_k: Dict[int, float] = {}

    for k in k_values:
        # Precision@K
        cutoff_relevances = ranked_relevances[:k]
        relevant_count = sum(1 for r in cutoff_relevances if r >= RELEVANCE_THRESHOLD)
        precision_at_k[k] = round(relevant_count / k, 4) if k > 0 else 0.0

        # nDCG@K
        dcg = calculate_dcg(ranked_relevances, k)
        idcg = calculate_dcg(ideal_sorted_scores, k)
        if idcg > 0.0:
            ndcg_at_k[k] = round(min(1.0, dcg / idcg), 4)
        else:
            ndcg_at_k[k] = 0.0

        # MeanHumanScore@K (mean of labeled candidates in top K)
        top_k_preds = sorted_preds[:k]
        top_k_scores = [
            eval_by_id[p.candidate_id].human_score
            for p in top_k_preds
            if p.candidate_id in eval_by_id and eval_by_id[p.candidate_id].human_score is not None
        ]
        if top_k_scores:
            mean_human_score_at_k[k] = round(sum(top_k_scores) / len(top_k_scores), 3)
        else:
            mean_human_score_at_k[k] = 0.0

        # PublishableRate@K
        pub_count = sum(1 for p in ranked_publishable[:k] if p)
        publishable_rate_at_k[k] = round(pub_count / k, 4) if k > 0 else 0.0

    # HitRate@K (K=1, 3, 5)
    hit_rate_at_k: Dict[int, float] = {}
    for k in hit_rate_k_values:
        cutoff_relevances = ranked_relevances[:k]
        has_hit = any(r >= RELEVANCE_THRESHOLD for r in cutoff_relevances)
        hit_rate_at_k[k] = 1.0 if has_hit else 0.0

    # Recall@K Guard
    recall_at_k: Optional[Dict[int, float]] = None
    recall_message: Optional[str] = None

    if total_candidates > 0 and labeled_candidates == total_candidates:
        total_relevant_in_pool = sum(1 for score in all_pool_scores if score >= RELEVANCE_THRESHOLD)
        if total_relevant_in_pool > 0:
            recall_at_k = {}
            for k in k_values:
                cutoff_relevances = ranked_relevances[:k]
                rel_retrieved = sum(1 for r in cutoff_relevances if r >= RELEVANCE_THRESHOLD)
                recall_at_k[k] = round(rel_retrieved / total_relevant_in_pool, 4)
        else:
            recall_at_k = {k: 0.0 for k in k_values}
            recall_message = "No candidates in candidate pool met relevance threshold (human_score >= 3)"
    else:
        recall_message = (
            f"Recall requires 100% labeled candidate pool (currently {labeled_candidates}/{total_candidates} labeled)"
        )

    return EvaluationMetrics(
        candidate_set_id=eval_doc.candidate_set_id,
        scorer=prediction_doc.scorer,
        scorer_version=prediction_doc.scorer_version,
        total_candidates=total_candidates,
        labeled_candidates=labeled_candidates,
        k_values=k_values,
        precision_at_k=precision_at_k,
        ndcg_at_k=ndcg_at_k,
        mean_human_score_at_k=mean_human_score_at_k,
        hit_rate_at_k=hit_rate_at_k,
        publishable_rate_at_k=publishable_rate_at_k,
        recall_at_k=recall_at_k,
        recall_message=recall_message,
    )
