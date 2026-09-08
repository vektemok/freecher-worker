"""Editorial-decision metrics for contextual_reranker_v1.

These complement the existing ranking metrics: instead of asking "is the order good?",
they ask "were the REJECT and STRONG decisions themselves correct?". Human labels are
used here only, after scoring, never inside the scorer.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .models import BlindEvaluationDocument, ScorerPredictionDocument

#: human_score >= 3 is a relevant highlight, matching evaluation.metrics.
RELEVANCE_THRESHOLD = 3.0
#: human_score >= 4 is a "perfect" clip.
PERFECT_THRESHOLD = 4.0
#: human_score <= 2 is a bad clip.
BAD_THRESHOLD = 2.0


class EditorialDecisionMetrics(BaseModel):
    """Precision of the reranker's own editorial decisions against human labels."""

    candidate_set_id: str
    scorer: str
    scorer_version: str

    labeled_predictions: int = Field(default=0, description="Predictions with a human label")

    reject_count: int = 0
    reject_labeled: int = 0
    reject_precision: Optional[float] = Field(
        default=None, description="Share of REJECT decisions that are human-bad (<= 2)"
    )

    strong_count: int = 0
    strong_labeled: int = 0
    strong_precision: Optional[float] = Field(
        default=None, description="Share of STRONG decisions that are publishable or perfect"
    )

    class_distribution: Dict[str, int] = Field(default_factory=dict)
    mean_human_by_class: Dict[str, float] = Field(default_factory=dict)
    critic_reject_count: int = 0
    critic_reject_precision: Optional[float] = Field(
        default=None, description="Share of critic REJECT decisions that are human-bad (<= 2)"
    )
    message: Optional[str] = None


def compute_editorial_metrics(
    eval_doc: BlindEvaluationDocument,
    prediction_doc: ScorerPredictionDocument,
) -> EditorialDecisionMetrics:
    """Compute RejectPrecision, StrongPrecision, and per-class human agreement."""
    if eval_doc.candidate_set_id != prediction_doc.candidate_set_id:
        raise ValueError(
            f"Candidate set ID mismatch! Evaluation document has '{eval_doc.candidate_set_id}', "
            f"but scorer prediction document has '{prediction_doc.candidate_set_id}'."
        )

    labels = {item.candidate_id: item for item in eval_doc.items}

    class_distribution: Dict[str, int] = {}
    class_scores: Dict[str, List[float]] = {}

    reject_labeled: List[float] = []
    reject_total = 0
    strong_labeled: List[tuple[float, bool]] = []
    strong_total = 0
    critic_reject_labeled: List[float] = []
    critic_reject_total = 0
    labeled_predictions = 0

    for pred in prediction_doc.predictions:
        editorial_class = pred.editorial_class
        if editorial_class:
            class_distribution[editorial_class] = class_distribution.get(editorial_class, 0) + 1

        label = labels.get(pred.candidate_id)
        human = label.human_score if label is not None else None
        publishable = bool(label.publishable) if label is not None else False
        if human is not None:
            labeled_predictions += 1
            if editorial_class:
                class_scores.setdefault(editorial_class, []).append(float(human))

        # A candidate is REJECT either by editorial class or by the critic.
        is_rejected = editorial_class == "REJECT" or (pred.status or "").startswith("rejected")
        if is_rejected:
            reject_total += 1
            if human is not None:
                reject_labeled.append(float(human))

        if editorial_class == "STRONG":
            strong_total += 1
            if human is not None:
                strong_labeled.append((float(human), publishable))

        if pred.critic_result == "REJECT":
            critic_reject_total += 1
            if human is not None:
                critic_reject_labeled.append(float(human))

    reject_precision = (
        round(sum(1 for s in reject_labeled if s <= BAD_THRESHOLD) / len(reject_labeled), 4)
        if reject_labeled
        else None
    )
    strong_precision = (
        round(
            sum(1 for s, pub in strong_labeled if pub or s >= PERFECT_THRESHOLD)
            / len(strong_labeled),
            4,
        )
        if strong_labeled
        else None
    )
    critic_reject_precision = (
        round(
            sum(1 for s in critic_reject_labeled if s <= BAD_THRESHOLD)
            / len(critic_reject_labeled),
            4,
        )
        if critic_reject_labeled
        else None
    )

    messages: List[str] = []
    if not class_distribution:
        messages.append(
            "No editorial_class fields found; this prediction document was not produced by "
            "contextual_reranker_v1."
        )
    if reject_total and not reject_labeled:
        messages.append("No REJECT decision has a human label; RejectPrecision is undefined.")
    if strong_total and not strong_labeled:
        messages.append("No STRONG decision has a human label; StrongPrecision is undefined.")

    return EditorialDecisionMetrics(
        candidate_set_id=eval_doc.candidate_set_id,
        scorer=prediction_doc.scorer,
        scorer_version=prediction_doc.scorer_version,
        labeled_predictions=labeled_predictions,
        reject_count=reject_total,
        reject_labeled=len(reject_labeled),
        reject_precision=reject_precision,
        strong_count=strong_total,
        strong_labeled=len(strong_labeled),
        strong_precision=strong_precision,
        class_distribution=class_distribution,
        mean_human_by_class={
            key: round(sum(values) / len(values), 3) for key, values in class_scores.items()
        },
        critic_reject_count=critic_reject_total,
        critic_reject_precision=critic_reject_precision,
        message="; ".join(messages) if messages else None,
    )
