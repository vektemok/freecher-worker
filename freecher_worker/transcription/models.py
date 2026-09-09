"""Data models for normalized speech transcription."""

from __future__ import annotations

import hashlib
from typing import Optional
from pydantic import BaseModel, Field

# Bumped when the shape of transcript.json changes in a way a consumer would
# have to notice. Word timestamps arrived additively, so 1.0 covers both.
TRANSCRIPT_SCHEMA_VERSION = "1.0"


class TranscriptWord(BaseModel):
    """One word with its own timestamps, on the same timeline as the segment."""

    start: float = Field(description="Word start time in seconds")
    end: float = Field(description="Word end time in seconds")
    word: str = Field(description="Word text, including its leading space as Whisper emits it")
    probability: Optional[float] = Field(default=None, description="Model confidence for this word")


class TranscriptSegment(BaseModel):
    """Normalized single speech segment."""

    id: int
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text")
    avg_logprob: Optional[float] = Field(default=None, description="Average log probability")
    no_speech_prob: Optional[float] = Field(default=None, description="Probability that segment has no speech")
    # None means word timestamps were not requested; an empty list means they
    # were requested and this segment carried none.
    words: Optional[list[TranscriptWord]] = Field(
        default=None, description="Word-level timestamps when they were requested"
    )


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
    word_timestamps: bool = Field(default=False, description="Whether word-level timestamps were requested")
    segments: list[TranscriptSegment] = Field(default_factory=list)

    # Provenance for a transcript produced from an R2 audio artifact rather
    # than from a local run directory. All optional, so a pipeline transcript
    # serialises exactly as it did before.
    schema_version: str = Field(default=TRANSCRIPT_SCHEMA_VERSION, description="Transcript JSON schema version")
    source_bucket: Optional[str] = Field(default=None, description="Bucket holding the audio this came from")
    source_audio_key: Optional[str] = Field(default=None, description="Object key of the audio this came from")
    audio_duration: Optional[float] = Field(default=None, description="Duration of the audio artifact as probed")
    audio_codec: Optional[str] = Field(default=None, description="Codec of the audio artifact")
    audio_sample_rate: Optional[int] = Field(default=None, description="Sample rate of the audio artifact in Hz")
    audio_channels: Optional[int] = Field(default=None, description="Channel count of the audio artifact")
    processing_seconds: Optional[float] = Field(default=None, description="Wall-clock seconds spent transcribing")
    created_at: Optional[str] = Field(default=None, description="UTC ISO-8601 timestamp of completion")

    @property
    def word_count(self) -> int:
        return sum(len(segment.words or ()) for segment in self.segments)

    def compute_transcript_hash(self) -> str:
        """Compute a deterministic hash of all transcript segments for downstream cache invalidation."""
        hasher = hashlib.sha256()
        hasher.update(f"{self.language}|{self.duration:.3f}|{len(self.segments)}".encode("utf-8"))
        for s in self.segments:
            hasher.update(f"|{s.id}:{s.start:.3f}-{s.end:.3f}:{s.text}".encode("utf-8"))
        return hasher.hexdigest()[:16]
