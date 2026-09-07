"""Data models for blind human evaluation, scorer predictions, and evaluation metrics."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BlindEvaluationItem(BaseModel):
    """An individual candidate window presented for blind human evaluation."""

    candidate_id: str = Field(description="Unique candidate identifier")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    duration: float = Field(description="Duration in seconds")
    text: str = Field(description="Candidate transcript text")
    segment_ids: list[int] = Field(default_factory=list, description="Transcript segment IDs")

    # Human annotations (MUST be empty during initial blind export)
    human_score: Optional[int] = Field(
        default=None,
        ge=0,
        le=4,
        description="Blind rating: 0=unusable, 1=weak, 2=acceptable, 3=good, 4=excellent",
    )
    publishable: Optional[bool] = Field(
        default=None,
        description="Whether this clip could be posted to social media without major rework",
    )
    human_notes: Optional[str] = Field(
        default=None,
        description="Free text commentary explaining the rating",
    )


class BlindEvaluationDocument(BaseModel):
    """A collection of candidate windows prepared for blind human evaluation."""

    candidate_set_id: str = Field(description="Immutable candidate set identifier matching CandidateDocument")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp")
    source_video: Optional[str] = Field(default=None, description="Path or name of source video")
    total_candidates: int = Field(description="Total number of candidates in this set")
    labeled_candidates: int = Field(default=0, description="Number of candidates labeled so far")
    seed: Optional[int] = Field(default=None, description="Random seed used to shuffle items")
    items: list[BlindEvaluationItem] = Field(default_factory=list, description="List of candidates to evaluate")

    def update_labeled_count(self) -> int:
        """Recalculate and update the count of labeled candidates."""
        self.labeled_candidates = sum(1 for item in self.items if item.human_score is not None)
        return self.labeled_candidates

    @property
    def is_fully_labeled(self) -> bool:
        """Check if 100% of candidates have been assigned a human score."""
        return len(self.items) > 0 and all(item.human_score is not None for item in self.items)


class ScorerPredictionItem(BaseModel):
    """Prediction for a single candidate highlight from an automated scorer."""

    candidate_id: str = Field(description="Candidate identifier")
    rank: int = Field(ge=1, description="1-based rank according to this scorer")
    score: float = Field(ge=0.0, le=100.0, description="Overall score normalized to 0-100")
    reason: Optional[str] = Field(default=None, description="Explanation for score")
    subscores: Optional[Dict[str, float]] = Field(default=None, description="Breakdown of subscores")


class ScorerPredictionDocument(BaseModel):
    """Container for predictions produced by a scorer on a frozen candidate set."""

    candidate_set_id: str = Field(description="Candidate set identifier matching CandidateDocument")
    scorer: str = Field(description="Scorer type name (e.g. heuristic, llm)")
    scorer_version: str = Field(description="Version string of the scorer")
    model: Optional[str] = Field(default=None, description="Underlying model name if applicable")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp")
    predictions: list[ScorerPredictionItem] = Field(default_factory=list, description="Ranked predictions")


class EvaluationMetrics(BaseModel):
    """Summary of ranking quality metrics comparing predictions to human labels."""

    candidate_set_id: str = Field(description="Candidate set identifier")
    scorer: str = Field(description="Scorer name")
    scorer_version: str = Field(description="Scorer version")
    total_candidates: int = Field(description="Total candidates in candidate set")
    labeled_candidates: int = Field(description="Number of candidates with human ratings")
    k_values: list[int] = Field(default_factory=lambda: [5, 10], description="Cutoff values evaluated")
    precision_at_k: Dict[int, float] = Field(default_factory=dict, description="Precision@K for human_score >= 3")
    ndcg_at_k: Dict[int, float] = Field(default_factory=dict, description="nDCG@K using 2^rel - 1 gain")
    mean_human_score_at_k: Dict[int, float] = Field(default_factory=dict, description="Mean human score in top K")
    hit_rate_at_k: Dict[int, float] = Field(default_factory=dict, description="HitRate@K (at least 1 score >= 3)")
    publishable_rate_at_k: Dict[int, float] = Field(default_factory=dict, description="Publishable rate in top K")
    recall_at_k: Optional[Dict[int, float]] = Field(
        default=None,
        description="Recall@K (only populated if 100% of candidate pool is labeled)",
    )
    recall_message: Optional[str] = Field(
        default=None,
        description="Explanation if Recall@K could not be computed",
    )


class DisagreementItem(BaseModel):
    """A highlight candidate where human judgment and model prediction diverge significantly."""

    candidate_id: str
    disagreement_type: str = Field(description="false_positive, false_negative, or scorer_divergence")
    start: float
    end: float
    duration: float
    text: str
    human_score: Optional[int] = None
    publishable: Optional[bool] = None
    human_notes: Optional[str] = None
    model_rank: Optional[int] = None
    model_score: Optional[float] = None
    model_reason: Optional[str] = None
    model_b_rank: Optional[int] = None
    model_b_score: Optional[float] = None


class DisagreementReport(BaseModel):
    """Structured report on false positives, false negatives, and scorer divergences."""

    candidate_set_id: str
    scorer_a: str
    scorer_b: Optional[str] = None
    false_positives: list[DisagreementItem] = Field(default_factory=list)
    false_negatives: list[DisagreementItem] = Field(default_factory=list)
    scorer_divergences: list[DisagreementItem] = Field(default_factory=list)
