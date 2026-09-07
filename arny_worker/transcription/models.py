"""Data models for normalized speech transcription."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """Normalized single speech segment."""

    id: int
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text")
    avg_logprob: Optional[float] = Field(default=None, description="Average log probability")
    no_speech_prob: Optional[float] = Field(default=None, description="Probability that segment has no speech")


class Transcript(BaseModel):
    """Normalized complete transcript document."""

    language: str = Field(description="Detected or configured language code (e.g. 'ru')")
    language_probability: float = Field(default=1.0, description="Confidence of language detection")
    duration: float = Field(description="Total audio duration in seconds")
    model: str = Field(description="Whisper model name used")
    compute_type: str = Field(description="Compute type used (e.g. int8_float16)")
    device: str = Field(description="Device used (cuda or cpu)")
    segments: list[TranscriptSegment] = Field(default_factory=list)
