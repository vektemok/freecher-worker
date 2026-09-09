"""Summary statistics over the structured human dimensions.

Deliberately separate from `metrics.py`: these describe the labels themselves —
what the rater saw across the candidate pool — and never enter a ranking
metric. `human_score` remains the only relevance signal any scorer is judged
against, and nothing here derives or adjusts it.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .models import BlindEvaluationDocument, BlindEvaluationItem

#: The dimensions recorded alongside human_score, in the order they are asked.
DIMENSION_FIELDS = (
    "hook_score",
    "standalone_score",
    "payoff_score",
    "value_score",
    "context_dependency",
)

#: Dimensions where a high value is a problem rather than a virtue.
LOWER_IS_BETTER = frozenset({"context_dependency"})

#: Matching the tiers the rater is asked for, and RELEVANCE_THRESHOLD in metrics.py.
STRONG_THRESHOLD = 3.0
BORDERLINE_SCORE = 2.0


class DimensionStats(BaseModel):
    """Distribution of one 0-4 dimension across the labeled pool."""

    field: str
    labeled: int = 0
    mean: Optional[float] = None
    median: Optional[float] = None
    histogram: Dict[int, int] = Field(default_factory=dict, description="Count per integer bucket 0-4")
    lower_is_better: bool = False


class HumanDimensionSummary(BaseModel):
    """What the human labels say about the candidate pool as a whole."""

    candidate_set_id: str
    total_candidates: int
    labeled_candidates: int = Field(description="Candidates carrying a human_score")
    dimension_labeled_candidates: int = Field(default=0, description="Candidates carrying any dimension")

    strong_count: int = Field(default=0, description="human_score >= 3")
    borderline_count: int = Field(default=0, description="human_score == 2")
    reject_count: int = Field(default=0, description="human_score <= 1")
    publishable_count: int = Field(default=0)

    human_score_mean: Optional[float] = None
    human_score_histogram: Dict[int, int] = Field(default_factory=dict)

    dimensions: List[DimensionStats] = Field(default_factory=list)

    bad_start_count: int = 0
    bad_end_count: int = 0
    boundary_labeled: int = Field(default=0, description="Candidates where bad_start/bad_end were recorded")

    @property
    def is_complete(self) -> bool:
        return self.total_candidates > 0 and self.labeled_candidates == self.total_candidates

    def rate(self, count: int, denominator: Optional[int] = None) -> Optional[float]:
        """A share of the labeled pool, or None when nothing is labeled yet."""
        base = self.labeled_candidates if denominator is None else denominator
        if not base:
            return None
        return round(count / base, 4)


def _values(items: List[BlindEvaluationItem], field: str) -> List[float]:
    return [
        float(getattr(item, field))
        for item in items
        if getattr(item, field) is not None
    ]


def _histogram(values: List[float]) -> Dict[int, int]:
    buckets = {bucket: 0 for bucket in range(5)}
    for value in values:
        bucket = int(round(value))
        if 0 <= bucket <= 4:
            buckets[bucket] += 1
    return buckets


def summarize_human_dimensions(eval_doc: BlindEvaluationDocument) -> HumanDimensionSummary:
    """Describe the human labels across a blind evaluation document."""
    items = eval_doc.items
    scored = [item for item in items if item.human_score is not None]
    human_scores = [float(item.human_score) for item in scored]

    summary = HumanDimensionSummary(
        candidate_set_id=eval_doc.candidate_set_id,
        total_candidates=len(items),
        labeled_candidates=len(scored),
        dimension_labeled_candidates=sum(1 for item in items if item.has_dimensions),
        strong_count=sum(1 for value in human_scores if value >= STRONG_THRESHOLD),
        borderline_count=sum(1 for value in human_scores if value == BORDERLINE_SCORE),
        reject_count=sum(1 for value in human_scores if value <= 1.0),
        publishable_count=sum(1 for item in items if item.publishable is True),
        human_score_mean=round(statistics.mean(human_scores), 3) if human_scores else None,
        human_score_histogram=_histogram(human_scores),
    )

    for field in DIMENSION_FIELDS:
        values = _values(items, field)
        summary.dimensions.append(
            DimensionStats(
                field=field,
                labeled=len(values),
                mean=round(statistics.mean(values), 3) if values else None,
                median=round(statistics.median(values), 3) if values else None,
                histogram=_histogram(values),
                lower_is_better=field in LOWER_IS_BETTER,
            )
        )

    summary.bad_start_count = sum(1 for item in items if item.bad_start is True)
    summary.bad_end_count = sum(1 for item in items if item.bad_end is True)
    summary.boundary_labeled = sum(
        1 for item in items if item.bad_start is not None or item.bad_end is not None
    )

    return summary
