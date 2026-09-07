"""Whisper transcription service based on faster-whisper."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .models import Transcript, TranscriptSegment

logger = logging.getLogger("arny_worker")


class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass


class BaseTranscriber(ABC):
    """Abstract interface for transcription services.

    Allows plugging in single-pass Whisper, future two-pass Whisper, or other ASR backends.
    """

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path | str,
        language: Optional[str] = None,
        source_fingerprint_id: Optional[str] = None,
    ) -> Transcript:
        """Transcribe an audio file into a normalized Transcript."""
        pass


class WhisperTranscriber(BaseTranscriber):
    """Transcription service using faster-whisper (CTranslate2)."""

    def __init__(
        self,
        model_name: str = "small",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model = None

    def _get_model(self):
        """Lazy load WhisperModel to avoid GPU memory allocation until required."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise TranscriptionError(
                    "faster-whisper is not installed. Please install it with 'pip install faster-whisper'."
                ) from exc

            try:
                logger.info(
                    f"[transcription] Loading WhisperModel(model='{self.model_name}', device='{self.device}', compute_type='{self.compute_type}')..."
                )
                self._model = WhisperModel(
                    model_size_or_path=self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as exc:
                err_str = str(exc)
                if "CUDA" in err_str or "cuDNN" in err_str or "cublas" in err_str:
                    hint = (
                        "Hint: CUDA or cuBLAS/cuDNN library was not found or failed to initialize. "
                        "Make sure scripts/cuda_env.sh has been sourced, or run with --device cpu."
                    )
                    raise TranscriptionError(f"Failed to load Whisper on GPU ({self.device}): {exc}\n{hint}") from exc
                raise TranscriptionError(f"Failed to load WhisperModel '{self.model_name}': {exc}") from exc

        return self._model

    def transcribe(
        self,
        audio_path: Path | str,
        language: Optional[str] = None,
        source_fingerprint_id: Optional[str] = None,
    ) -> Transcript:
        """Transcribe audio file to normalized Transcript object.

        Materializes all segments from faster-whisper generator into memory.
        """
        path = Path(audio_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file for transcription not found: {path}")

        model = self._get_model()

        try:
            raw_segments, info = model.transcribe(
                str(path),
                language=language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
            )

            # CRITICAL: faster-whisper returns a lazy generator.
            # We must materialize it completely during the transcription stage.
            materialized = list(raw_segments)
        except Exception as exc:
            err_msg = str(exc)
            if "out of memory" in err_msg.lower():
                raise TranscriptionError(
                    f"CUDA Out of Memory during transcription: {exc}. "
                    f"Try using a smaller model (e.g. 'small'), or compute_type 'int8'."
                ) from exc
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

        segments: list[TranscriptSegment] = []
        for idx, seg in enumerate(materialized):
            text = seg.text.strip()
            if not text:
                continue
            avg_logprob = getattr(seg, "avg_logprob", None)
            no_speech_prob = getattr(seg, "no_speech_prob", None)
            segments.append(
                TranscriptSegment(
                    id=idx,
                    start=round(seg.start, 3),
                    end=round(seg.end, 3),
                    text=text,
                    avg_logprob=round(avg_logprob, 4) if avg_logprob is not None else None,
                    no_speech_prob=round(no_speech_prob, 4) if no_speech_prob is not None else None,
                )
            )

        duration = getattr(info, "duration", 0.0)
        if duration <= 0.0 and segments:
            duration = segments[-1].end

        return Transcript(
            source_fingerprint_id=source_fingerprint_id,
            language=info.language if hasattr(info, "language") else (language or "unknown"),
            language_probability=round(getattr(info, "language_probability", 1.0), 3),
            duration=round(duration, 3),
            model=self.model_name,
            compute_type=self.compute_type,
            device=self.device,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            segments=segments,
        )
