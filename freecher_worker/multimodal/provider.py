"""Abstract base provider for multimodal highlight evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .models import MultimodalCandidatePackage, MultimodalModelResult, MultimodalUsage

PROMPT_VERSION_MULTIMODAL_V1 = "multimodal_v1_prompt_v1"
PROMPT_VERSION_MULTIMODAL_V1_1 = "multimodal_v1_1_prompt_v1"


class MultimodalProvider(ABC):
    """Abstract interface for multimodal highlight reranking providers."""

    name: str = "abstract_multimodal_provider"
    model: str = "default"
    prompt_version: str = PROMPT_VERSION_MULTIMODAL_V1
    reasoning_effort: Optional[str] = None
    temperature: Optional[float] = None

    @abstractmethod
    def score_candidate(self, package: MultimodalCandidatePackage) -> MultimodalModelResult:
        """Evaluate and score a multimodal candidate package.

        Args:
            package: Self-contained multimodal package with frames, audio, and transcript.

        Returns:
            MultimodalModelResult with structured dimensional scores and diagnostics.
        """
        pass

    @abstractmethod
    def get_usage(self) -> MultimodalUsage:
        """Return cumulative token usage and request statistics."""
        pass
