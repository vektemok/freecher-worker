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
from .editorial_metrics import EditorialDecisionMetrics, compute_editorial_metrics
from .human_dimensions import (
    DIMENSION_FIELDS,
    LOWER_IS_BETTER,
    DimensionStats,
    HumanDimensionSummary,
    summarize_human_dimensions,
)
from .audio_preview import (
    AudioPreviewError,
    AudioPreviewer,
    CachedAudio,
    build_playback_command,
    cache_path_for,
    ensure_audio_artifact,
)
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
    "EditorialDecisionMetrics",
    "compute_editorial_metrics",
    "compute_evaluation_metrics",
    "calculate_dcg",
    "RELEVANCE_THRESHOLD",
    "DIMENSION_FIELDS",
    "LOWER_IS_BETTER",
    "DimensionStats",
    "HumanDimensionSummary",
    "summarize_human_dimensions",
    "AudioPreviewError",
    "AudioPreviewer",
    "CachedAudio",
    "build_playback_command",
    "cache_path_for",
    "ensure_audio_artifact",
    "extract_disagreements",
    "run_terminal_annotator",
    "preview_clip",
]
