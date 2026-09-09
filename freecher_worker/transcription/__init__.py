"""Transcription module."""

from .models import TRANSCRIPT_SCHEMA_VERSION, Transcript, TranscriptSegment, TranscriptWord
from .r2 import (
    AudioArtifactInfo,
    DownloadProgress,
    TranscriptionResult,
    TranscriptionWorkflowError,
    audio_key_for,
    download_audio_artifact,
    existing_transcript,
    serialize_transcript,
    transcribe_from_r2,
    transcript_key_for,
    transcript_object_metadata,
    validate_audio_artifact,
)
from .whisper import (
    BaseTranscriber,
    TranscriptionError,
    TranscriptionProgress,
    WhisperTranscriber,
)

__all__ = [
    "TRANSCRIPT_SCHEMA_VERSION",
    "AudioArtifactInfo",
    "BaseTranscriber",
    "DownloadProgress",
    "Transcript",
    "TranscriptSegment",
    "TranscriptWord",
    "TranscriptionError",
    "TranscriptionProgress",
    "TranscriptionResult",
    "TranscriptionWorkflowError",
    "WhisperTranscriber",
    "audio_key_for",
    "download_audio_artifact",
    "existing_transcript",
    "serialize_transcript",
    "transcribe_from_r2",
    "transcript_key_for",
    "transcript_object_metadata",
    "validate_audio_artifact",
]
