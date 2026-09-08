"""Data models for candidate highlight windows and ranked highlights."""

from __future__ import annotations

import hashlib
from typing import Optional
from pydantic import BaseModel, Field, model_validator

SEGMENTATION_VERSION = "1.0.0"


def compute_candidate_set_id(
    transcript_hash: str,
    min_seconds: float,
    target_seconds: float,
    max_seconds: float,
    overlap_seconds: float,
    segmentation_version: str = SEGMENTATION_VERSION,
) -> str:
    """Compute an immutable, deterministic identifier for a candidate window set."""
    canonical_meta = (
        f"{transcript_hash}|{min_seconds:.3f}|{target_seconds:.3f}|"
        f"{max_seconds:.3f}|{overlap_seconds:.3f}|{segmentation_version}"
    )
    digest = hashlib.sha256(canonical_meta.encode("utf-8")).hexdigest()[:16]
    return f"cset_{digest}"


class CandidateWindow(BaseModel):
    """A multi-segment candidate highlight window (temporal windowing bounded by Whisper segments)."""

    id: str
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    duration: float = Field(description="Duration in seconds (end - start)")
    text: str = Field(description="Aggregated text from included segments")
    segment_ids: list[int] = Field(default_factory=list, description="IDs of included transcript segments")


class CandidateDocument(BaseModel):
    """Container for candidates with generation parameters and immutable candidate_set_id."""

    candidate_set_id: str = Field(default="", description="Immutable identifier of this candidate set")
    transcript_hash: str = Field(description="Hash of source transcript segments")
    segmentation_version: str = Field(default=SEGMENTATION_VERSION, description="Version of the segmentation algorithm")
    min_seconds: float = Field(description="Minimum duration in seconds")
    target_seconds: float = Field(description="Target duration in seconds")
    max_seconds: float = Field(description="Maximum duration in seconds")
    overlap_seconds: float = Field(description="Overlap duration in seconds")
    candidates: list[CandidateWindow] = Field(default_factory=list)

    @model_validator(mode="after")
    def populate_candidate_set_id(self) -> CandidateDocument:
        if not self.candidate_set_id:
            self.candidate_set_id = compute_candidate_set_id(
                transcript_hash=self.transcript_hash,
                min_seconds=self.min_seconds,
                target_seconds=self.target_seconds,
                max_seconds=self.max_seconds,
                overlap_seconds=self.overlap_seconds,
                segmentation_version=self.segmentation_version,
            )
        return self


class HighlightScore(BaseModel):
    """Multi-dimensional score of a highlight candidate."""

    score: float = Field(ge=0, le=100, description="Overall score normalized to 0-100")
    hook_score: float = Field(ge=0, le=100, description="Hook potential score (0-100)")
    standalone_score: float = Field(ge=0, le=100, description="Context-independence score (0-100)")
    emotion_score: float = Field(ge=0, le=100, description="Emotional resonance score (0-100)")
    information_score: float = Field(ge=0, le=100, description="Information value score (0-100)")
    shareability_score: float = Field(ge=0, le=100, description="Shareability / viral potential score (0-100)")
    reason: str = Field(description="Human-readable explanation of the score")
    fallback_used: bool = Field(default=False, description="Whether fallback scoring was engaged (e.g. LLM failure)")
    fallback_reason: Optional[str] = Field(default=None, description="Reason why fallback was triggered")

    # Extended dimensions & metadata for Highlight Intelligence v2
    story_payoff_score: Optional[float] = Field(default=None, ge=0, le=100)
    humor_score: Optional[float] = Field(default=None, ge=0, le=100)
    surprise_score: Optional[float] = Field(default=None, ge=0, le=100)
    retention_score: Optional[float] = Field(default=None, ge=0, le=100)
    boringness_score: Optional[float] = Field(default=None, ge=0, le=100)
    context_dependency_score: Optional[float] = Field(default=None, ge=0, le=100)
    llm_quality_score: Optional[float] = Field(default=None, ge=0, le=100)
    final_score: Optional[float] = Field(default=None, ge=0, le=100)
    subscores: Optional[dict[str, float]] = None
    flags: Optional[dict[str, bool]] = None
    scorer_version: Optional[str] = None
    requested_model: Optional[str] = None
    actual_model: Optional[str] = None


class Highlight(BaseModel):
    """Ranked and selected highlight ready for clipping."""

    rank: int = Field(description="Rank position (1 to K)")
    start: float = Field(description="Original start time in seconds")
    end: float = Field(description="Original end time in seconds")
    duration: float = Field(description="Original highlight duration in seconds")
    score: float = Field(description="Overall highlight score")
    reason: str = Field(description="Score reason")
    candidate_id: str = Field(description="ID of source candidate")
    text: str = Field(description="Transcript text of highlight")
    file: Optional[str] = Field(default=None, description="Relative path to clipped video file")
    padded_start: Optional[float] = Field(default=None, description="Start timestamp with contextual padding")
    padded_end: Optional[float] = Field(default=None, description="End timestamp with contextual padding")
    score_breakdown: Optional[HighlightScore] = Field(default=None, description="Detailed score dimensions")
