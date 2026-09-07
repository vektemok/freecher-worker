"""Identification of false positives, false negatives, and scorer divergence."""

from __future__ import annotations

from typing import Dict, List, Optional
from .models import (
    BlindEvaluationDocument,
    BlindEvaluationItem,
    DisagreementItem,
    DisagreementReport,
    ScorerPredictionDocument,
    ScorerPredictionItem,
)


def extract_disagreements(
    eval_doc: BlindEvaluationDocument,
    pred_a: ScorerPredictionDocument,
    pred_b: Optional[ScorerPredictionDocument] = None,
    top_k_fp: int = 5,
    fn_rank_threshold: int = 5,
    divergence_rank_diff: int = 3,
) -> DisagreementReport:
    """Identify discrepancies between human judgments and scorer predictions."""
    if eval_doc.candidate_set_id != pred_a.candidate_set_id:
        raise ValueError(
            f"Candidate set ID mismatch! Evaluation: '{eval_doc.candidate_set_id}', Scorer A: '{pred_a.candidate_set_id}'"
        )
    if pred_b is not None and eval_doc.candidate_set_id != pred_b.candidate_set_id:
        raise ValueError(
            f"Candidate set ID mismatch! Evaluation: '{eval_doc.candidate_set_id}', Scorer B: '{pred_b.candidate_set_id}'"
        )

    eval_by_id: Dict[str, BlindEvaluationItem] = {item.candidate_id: item for item in eval_doc.items}
    preds_a_by_id: Dict[str, ScorerPredictionItem] = {p.candidate_id: p for p in pred_a.predictions}
    preds_b_by_id: Dict[str, ScorerPredictionItem] = {p.candidate_id: p for p in (pred_b.predictions if pred_b else [])}

    false_positives: List[DisagreementItem] = []
    false_negatives: List[DisagreementItem] = []
    scorer_divergences: List[DisagreementItem] = []

    # 1. False Positives: Model ranked in top_k_fp (e.g. top 5), but human rated <= 1
    sorted_a = sorted(pred_a.predictions, key=lambda p: p.rank)
    for p in sorted_a[:top_k_fp]:
        item = eval_by_id.get(p.candidate_id)
        if item is not None and item.human_score is not None and item.human_score <= 1.0:
            false_positives.append(
                DisagreementItem(
                    candidate_id=p.candidate_id,
                    disagreement_type="false_positive",
                    start=item.start,
                    end=item.end,
                    duration=item.duration,
                    text=item.text,
                    human_score=item.human_score,
                    publishable=item.publishable,
                    human_notes=item.human_notes,
                    model_rank=p.rank,
                    model_score=p.score,
                    model_reason=p.reason,
                )
            )

    # 2. False Negatives: Human scored >= 3, but model ranked low (rank > fn_rank_threshold)
    for item in eval_doc.items:
        if item.human_score is not None and item.human_score >= 3.0:
            pred_item = preds_a_by_id.get(item.candidate_id)
            if pred_item is None or pred_item.rank > fn_rank_threshold:
                false_negatives.append(
                    DisagreementItem(
                        candidate_id=item.candidate_id,
                        disagreement_type="false_negative",
                        start=item.start,
                        end=item.end,
                        duration=item.duration,
                        text=item.text,
                        human_score=item.human_score,
                        publishable=item.publishable,
                        human_notes=item.human_notes,
                        model_rank=pred_item.rank if pred_item else None,
                        model_score=pred_item.score if pred_item else None,
                        model_reason=pred_item.reason if pred_item else None,
                    )
                )

    # 3. Scorer Divergence (when comparing 2 scorers)
    if pred_b is not None:
        for cid, item_a in preds_a_by_id.items():
            item_b = preds_b_by_id.get(cid)
            if item_b is not None:
                rank_diff = abs(item_a.rank - item_b.rank)
                if rank_diff >= divergence_rank_diff:
                    eval_item = eval_by_id.get(cid)
                    scorer_divergences.append(
                        DisagreementItem(
                            candidate_id=cid,
                            disagreement_type="scorer_divergence",
                            start=eval_item.start if eval_item else 0.0,
                            end=eval_item.end if eval_item else 0.0,
                            duration=eval_item.duration if eval_item else 0.0,
                            text=eval_item.text if eval_item else "",
                            human_score=eval_item.human_score if eval_item else None,
                            publishable=eval_item.publishable if eval_item else None,
                            human_notes=eval_item.human_notes if eval_item else None,
                            model_rank=item_a.rank,
                            model_score=item_a.score,
                            model_reason=item_a.reason,
                            model_b_rank=item_b.rank,
                            model_b_score=item_b.score,
                        )
                    )
        # Sort divergences by largest rank difference
        scorer_divergences.sort(
            key=lambda d: abs((d.model_rank or 0) - (d.model_b_rank or 0)),
            reverse=True,
        )

    return DisagreementReport(
        candidate_set_id=eval_doc.candidate_set_id,
        scorer_a=f"{pred_a.scorer} ({pred_a.scorer_version})",
        scorer_b=f"{pred_b.scorer} ({pred_b.scorer_version})" if pred_b else None,
        false_positives=false_positives,
        false_negatives=false_negatives,
        scorer_divergences=scorer_divergences,
    )
