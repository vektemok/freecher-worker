"""Data models for candidate highlight windows and ranked highlights."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class CandidateWindow(BaseModel):
    """A multi-segment candidate highlight window (temporal windowing bounded by Whisper segments)."""

    id: str
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    duration: float = Field(description="Duration in seconds (end - start)")
    text: str = Field(description="Aggregated text from included segments")
    segment_ids: list[int] = Field(default_factory=list, description="IDs of included transcript segments")


class CandidateDocument(BaseModel):
    """Container for candidates with generation parameters for cache validation."""

    transcript_hash: str = Field(description="Hash of source transcript segments")
    min_seconds: float = Field(description="Minimum duration in seconds")
    target_seconds: float = Field(description="Target duration in seconds")
    max_seconds: float = Field(description="Maximum duration in seconds")
    overlap_seconds: float = Field(description="Overlap duration in seconds")
    candidates: list[CandidateWindow] = Field(default_factory=list)


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
