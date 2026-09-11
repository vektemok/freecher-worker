"""Refined ASR transcription with word-level timestamps for selected highlight clips."""

from __future__ import annotations

import hashlib
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional
from pydantic import BaseModel, Field

from freecher_worker.utils.json_io import load_json, save_json

logger = logging.getLogger("freecher_worker")


class WordItem(BaseModel):
    """Word-level timestamp and confidence."""

    word: str = Field(description="Recognized word text")
    start: float = Field(description="Clip-relative start time in seconds")
    end: float = Field(description="Clip-relative end time in seconds")
    probability: float = Field(default=1.0, description="Word recognition probability")


class RefinedWordsDocument(BaseModel):
    """Collection of word-level timestamps for a single highlight clip."""

    cache_key: str = Field(default="", description="Hash of source fingerprint and transcription params")
    language: str = Field(description="Detected or specified spoken language")
    start_offset: float = Field(description="Absolute start timestamp in source video (seconds)")
    duration: float = Field(description="Clip duration in seconds")
    model: str = Field(description="Whisper model name used")
    compute_type: str = Field(description="Compute type used")
    words: List[WordItem] = Field(default_factory=list, description="Clip-relative word timestamps")


def compute_refinement_cache_key(
    source_fingerprint_id: str,
    start: float,
    end: float,
    model: str,
    compute_type: str,
    language: Optional[str] = None,
) -> str:
    """Compute deterministic cache key for refined word timestamps."""
    raw = f"{source_fingerprint_id}|{start:.3f}|{end:.3f}|{model}|{compute_type}|{language or 'auto'}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def extract_audio_chunk(
    source_media: Path,
    start: float,
    duration: float,
    output_wav: Path,
) -> Path:
    """Extract a 16kHz mono WAV chunk for a specific temporal window using FFmpeg."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(source_media),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav),
    ]

    subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=120,
    )
    return output_wav


class HighlightWordTranscriber:
    """Produces refined word timestamps for highlight clips.

    The model is loaded lazily on first use and then held for the lifetime of
    this object, so one instance shared across a render batch loads it once. A
    per-clip instance therefore pays the load per clip -- roughly 115 s each on
    the two-core ARM host -- which is why `render_top_n` and
    `render_highlights_for_run` now construct exactly one and pass it down.

    Batch-scoped, deliberately not a module-level singleton: the model is over
    2 GiB resident, and a process that renders once should not keep it for the
    rest of its life. Use `release()`, or the context manager, to drop it.
    """

    def __init__(
        self,
        model_name: str = "medium",
        device: str = "cuda",
        compute_type: str = "int8",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None
        #: How many times a model was actually constructed. One shared instance
        #: across a batch must never exceed 1; asserted by the test suite.
        self.load_count = 0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def release(self) -> None:
        """Drop the loaded model so its memory can be reclaimed.

        Safe to call when nothing was ever loaded, and safe to call twice; the
        next transcribe_highlight simply loads again.
        """
        if self._model is not None:
            logger.info("[refinement] releasing '%s' after %d load(s)",
                        self.model_name, self.load_count)
        self._model = None

    def __enter__(self) -> "HighlightWordTranscriber":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            import ctranslate2

            dev = self.device
            comp = self.compute_type
            if dev == "cuda":
                try:
                    if ctranslate2.get_cuda_device_count() == 0:
                        logger.warning("No CUDA devices detected. Falling back to CPU for HighlightWordTranscriber.")
                        dev = "cpu"
                        comp = "int8"
                except Exception:
                    logger.warning("CUDA check failed. Falling back to CPU for HighlightWordTranscriber.")
                    dev = "cpu"
                    comp = "int8"

            logger.info(f"Loading faster-whisper model '{self.model_name}' on {dev} ({comp})...")
            try:
                self._model = WhisperModel(
                    self.model_name,
                    device=dev,
                    compute_type=comp,
                )
            except Exception as exc:
                if dev != "cpu":
                    logger.warning(f"Failed loading Whisper on {dev}: {exc}. Retrying on CPU...")
                    self._model = WhisperModel(
                        self.model_name,
                        device="cpu",
                        compute_type="int8",
                    )
                else:
                    raise
            self.load_count += 1
        return self._model

    def transcribe_highlight(
        self,
        source_media: Path,
        refined_start: float,
        refined_end: float,
        source_fingerprint_id: str,
        language: Optional[str] = None,
        cache_path: Optional[Path] = None,
        force: bool = False,
    ) -> RefinedWordsDocument:
        """Transcribe a highlight segment with word-level timestamps."""
        duration = round(refined_end - refined_start, 3)
        cache_key = compute_refinement_cache_key(
            source_fingerprint_id=source_fingerprint_id,
            start=refined_start,
            end=refined_end,
            model=self.model_name,
            compute_type=self.compute_type,
            language=language,
        )

        if not force and cache_path and cache_path.is_file():
            try:
                cached_doc = RefinedWordsDocument.model_validate(load_json(cache_path))
                if cached_doc.cache_key == cache_key and len(cached_doc.words) > 0:
                    logger.info(f"Reused refined words from cache: {cache_path}")
                    return cached_doc
            except Exception as e:
                logger.warning(f"Failed reading cached words from {cache_path}: {e}")

        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_wav = Path(tmpdir) / "highlight_chunk.wav"
            extract_audio_chunk(
                source_media=source_media,
                start=refined_start,
                duration=duration,
                output_wav=chunk_wav,
            )

            model = self._get_model()
            segments_gen, info = model.transcribe(
                str(chunk_wav),
                language=language,
                word_timestamps=True,
                vad_filter=True,
            )

            words: List[WordItem] = []
            for seg in segments_gen:
                if getattr(seg, "words", None):
                    for w in seg.words:
                        cleaned = w.word.strip()
                        if cleaned:
                            words.append(
                                WordItem(
                                    word=cleaned,
                                    start=round(w.start, 3),
                                    end=round(w.end, 3),
                                    probability=round(w.probability, 3),
                                )
                            )

            detected_lang = language or getattr(info, "language", "en")
            doc = RefinedWordsDocument(
                cache_key=cache_key,
                language=detected_lang,
                start_offset=refined_start,
                duration=duration,
                model=self.model_name,
                compute_type=self.compute_type,
                words=words,
            )

            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                save_json(doc, cache_path)

            return doc
