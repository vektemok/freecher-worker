"""Data models for normalized speech transcription."""

from __future__ import annotations

import hashlib
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

    source_fingerprint_id: Optional[str] = Field(default=None, description="ID of source media fingerprint")
    language: str = Field(description="Detected or configured language code (e.g. 'ru')")
    language_probability: float = Field(default=1.0, description="Confidence of language detection")
    duration: float = Field(description="Total audio duration in seconds")
    model: str = Field(description="Whisper model name used")
    compute_type: str = Field(description="Compute type used (e.g. int8_float16)")
    device: str = Field(description="Device used (cuda or cpu)")
    beam_size: int = Field(default=5, description="Beam size used during decoding")
    vad_filter: bool = Field(default=True, description="Whether Silero VAD was enabled")
    segments: list[TranscriptSegment] = Field(default_factory=list)

    def compute_transcript_hash(self) -> str:
        """Compute a deterministic hash of all transcript segments for downstream cache invalidation."""
        hasher = hashlib.sha256()
        hasher.update(f"{self.language}|{self.duration:.3f}|{len(self.segments)}".encode("utf-8"))
        for s in self.segments:
            hasher.update(f"|{s.id}:{s.start:.3f}-{s.end:.3f}:{s.text}".encode("utf-8"))
        return hasher.hexdigest()[:16]
