"""Evaluation and benchmarking harness for highlight scoring models."""

from .models import (
    BlindEvaluationItem,
    BlindEvaluationDocument,
    ScorerPredictionItem,
    ScorerPredictionDocument,
    EvaluationMetrics,
    DisagreementItem,
    DisagreementReport,
)
from .metrics import compute_evaluation_metrics, calculate_dcg, RELEVANCE_THRESHOLD
from .disagreements import extract_disagreements
from .annotator import run_terminal_annotator, preview_clip

__all__ = [
    "BlindEvaluationItem",
    "BlindEvaluationDocument",
    "ScorerPredictionItem",
    "ScorerPredictionDocument",
    "EvaluationMetrics",
    "DisagreementItem",
    "DisagreementReport",
    "compute_evaluation_metrics",
    "calculate_dcg",
    "RELEVANCE_THRESHOLD",
    "extract_disagreements",
    "run_terminal_annotator",
    "preview_clip",
]
