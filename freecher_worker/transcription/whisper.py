"""Whisper transcription service based on faster-whisper."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .models import Transcript, TranscriptSegment, TranscriptWord

logger = logging.getLogger("freecher_worker")


@dataclass
class TranscriptionProgress:
    """Snapshot emitted after each segment faster-whisper yields.

    Decoding a two-hour source is a single long call, so the only honest
    measure of progress is how far along the audio timeline the decoder has
    reached — segment counts say nothing about how much is left.
    """

    segment_count: int
    current_seconds: float
    audio_seconds: Optional[float]
    elapsed_seconds: float

    @property
    def percent(self) -> Optional[float]:
        if not self.audio_seconds:
            return None
        return min(100.0, self.current_seconds / self.audio_seconds * 100.0)

    @property
    def speed(self) -> float:
        """Audio seconds decoded per wall-clock second (realtime factor)."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.current_seconds / self.elapsed_seconds

    @property
    def eta_seconds(self) -> Optional[float]:
        if not self.audio_seconds or self.speed <= 0:
            return None
        return max(0.0, (self.audio_seconds - self.current_seconds) / self.speed)


ProgressCallback = Callable[[TranscriptionProgress], None]


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
        on_progress: Optional[ProgressCallback] = None,
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
        word_timestamps: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.word_timestamps = word_timestamps
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
        on_progress: Optional[ProgressCallback] = None,
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
                word_timestamps=self.word_timestamps,
            )

            # CRITICAL: faster-whisper returns a lazy generator, and decoding
            # only happens as it is consumed. Draining it by hand rather than
            # with list() is what lets progress be reported on the way through;
            # it is still materialized completely before returning.
            audio_seconds = getattr(info, "duration", None)
            started_at = time.perf_counter()
            materialized = []
            for segment in raw_segments:
                materialized.append(segment)
                if on_progress is not None:
                    on_progress(
                        TranscriptionProgress(
                            segment_count=len(materialized),
                            current_seconds=float(getattr(segment, "end", 0.0) or 0.0),
                            audio_seconds=audio_seconds,
                            elapsed_seconds=time.perf_counter() - started_at,
                        )
                    )
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
                    words=_words_from_segment(seg) if self.word_timestamps else None,
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
            word_timestamps=self.word_timestamps,
            segments=segments,
        )


def _words_from_segment(segment) -> list[TranscriptWord]:
    """Map faster-whisper's per-word timings onto the normalized model.

    Word timings share the segment's timeline, so they are rounded but never
    shifted. A segment can legitimately carry none (VAD-clipped audio), which
    is an empty list rather than None: the distinction says whether word
    timestamps were asked for at all.
    """
    raw_words = getattr(segment, "words", None) or []
    words: list[TranscriptWord] = []
    for word in raw_words:
        text = getattr(word, "word", None)
        if text is None:
            continue
        probability = getattr(word, "probability", None)
        words.append(
            TranscriptWord(
                start=round(float(word.start), 3),
                end=round(float(word.end), 3),
                word=text,
                probability=round(float(probability), 4) if probability is not None else None,
            )
        )
    return words
