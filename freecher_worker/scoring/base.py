"""Base abstract interface for highlight scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from freecher_worker.highlights.models import CandidateWindow, HighlightScore


class HighlightScorer(ABC):
    """Abstract base class for highlight scoring implementations."""

    @abstractmethod
    def score(self, candidate: CandidateWindow) -> HighlightScore:
        """Evaluate a single candidate window and return a HighlightScore."""
        pass

    def score_batch(self, candidates: list[CandidateWindow]) -> list[HighlightScore]:
        """Evaluate multiple candidate windows sequentially."""
        return [self.score(cand) for cand in candidates]
