"""Transcribe an R2 audio artifact and write transcript.json back beside it.

The counterpart to the ingest milestone: ingest puts a mono 16 kHz m4a at
processing/{source_id}/audio.m4a, and this reads that object — never the
multi-gigabyte source.mp4 — runs faster-whisper over it on GPU, and writes
processing/{source_id}/transcript.json.

The audio is small enough (roughly 0.5 MB per minute) to land in a temporary
file, which is what faster-whisper wants anyway; it is always removed. The
transcript goes up as a single PutObject, so an interrupted run leaves either
the previous object or nothing — never a half-written one.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from freecher_worker.ingest.audio import AudioArtifactError, probe_audio_file
from freecher_worker.ingest.service import object_exists
from freecher_worker.transcription.models import Transcript
from freecher_worker.transcription.whisper import (
    BaseTranscriber,
    ProgressCallback,
    TranscriptionError,
    WhisperTranscriber,
)

logger = logging.getLogger("freecher_worker")

PROCESSING_PREFIX = "processing"
AUDIO_FILENAME = "audio.m4a"
TRANSCRIPT_FILENAME = "transcript.json"

TRANSCRIPT_CONTENT_TYPE = "application/json"

# A mono 16 kHz artifact should decode to exactly this; anything else means we
# fetched something other than what ingest produced.
EXPECTED_SAMPLE_RATE = 16_000
EXPECTED_CHANNELS = 1

# The downloaded artifact has to line up with the source the same way ingest
# guaranteed it did. Same bounded policy, same reasoning: a drift that matters
# is the same size whatever the length of the VOD.
MIN_DURATION_TOLERANCE_SECONDS = 2.0
MAX_DURATION_TOLERANCE_SECONDS = 5.0
DURATION_TOLERANCE_RATIO = 0.001


class TranscriptionWorkflowError(Exception):
    """Raised when the R2-backed transcription workflow cannot complete."""


@dataclass
class DownloadProgress:
    """Snapshot emitted while the audio artifact is being fetched."""

    downloaded_bytes: int
    total_bytes: Optional[int]

    @property
    def percent(self) -> Optional[float]:
        if not self.total_bytes:
            return None
        return min(100.0, self.downloaded_bytes / self.total_bytes * 100.0)


@dataclass
class AudioArtifactInfo:
    """What the downloaded artifact actually turned out to be."""

    path: Path
    size_bytes: int
    codec: str
    sample_rate: int
    channels: int
    duration_seconds: float
    source_duration_seconds: Optional[float] = None


@dataclass
class TranscriptionResult:
    """Outcome of a completed (or deliberately skipped) transcription."""

    bucket: str
    source_id: str
    audio_key: str
    transcript_key: str
    transcript: Optional[Transcript] = None
    skipped: bool = False
    uploaded_bytes: int = 0
    download_seconds: float = 0.0
    transcribe_seconds: float = 0.0
    total_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def transcript_uri(self) -> str:
        return f"s3://{self.bucket}/{self.transcript_key}"

    @property
    def audio_uri(self) -> str:
        return f"s3://{self.bucket}/{self.audio_key}"


def audio_key_for(source_id: str) -> str:
    """The audio artifact ingest wrote for this source."""
    return f"{PROCESSING_PREFIX}/{_clean_source_id(source_id)}/{AUDIO_FILENAME}"


def transcript_key_for(source_id: str) -> str:
    """The transcript object that sits beside that audio."""
    return f"{PROCESSING_PREFIX}/{_clean_source_id(source_id)}/{TRANSCRIPT_FILENAME}"


def _clean_source_id(source_id: str) -> str:
    cleaned = source_id.strip().strip("/")
    if not cleaned or "/" in cleaned:
        raise ValueError(
            f"source id must be a single path segment, got '{source_id}'"
        )
    return cleaned


def duration_tolerance_seconds(source_seconds: float) -> float:
    """Bounded drift allowance, mirroring the ingest-side check."""
    return max(
        MIN_DURATION_TOLERANCE_SECONDS,
        min(MAX_DURATION_TOLERANCE_SECONDS, source_seconds * DURATION_TOLERANCE_RATIO),
    )


def download_audio_artifact(
    client: Any,
    bucket: str,
    key: str,
    destination: Path | str,
    *,
    on_progress: Optional[Callable[[DownloadProgress], None]] = None,
    chunk_size: int = 8 * 1024 * 1024,
) -> tuple[int, dict[str, str]]:
    """Stream one object to a local file, returning its size and metadata.

    Returns the byte count actually written rather than trusting the header,
    so a short read is caught by the caller instead of being fed to the model.
    """
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        raise TranscriptionWorkflowError(
            f"could not read s3://{bucket}/{key}: {exc}"
        ) from exc

    total = response.get("ContentLength")
    metadata = {str(k): str(v) for k, v in (response.get("Metadata") or {}).items()}
    body = response["Body"]

    written = 0
    try:
        with target.open("wb") as handle:
            while True:
                chunk = body.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                if on_progress is not None:
                    on_progress(DownloadProgress(downloaded_bytes=written, total_bytes=total))
    finally:
        close = getattr(body, "close", None)
        if close is not None:
            close()

    if total is not None and written != total:
        raise TranscriptionWorkflowError(
            f"s3://{bucket}/{key} is {total} bytes but only {written} arrived; "
            "the download was truncated"
        )
    return written, metadata


def validate_audio_artifact(
    path: Path | str,
    *,
    metadata: Optional[dict[str, str]] = None,
    ffprobe_path: str = "ffprobe",
) -> AudioArtifactInfo:
    """Check the downloaded file really is decodable speech audio.

    Feeding a truncated or empty file to Whisper produces a short transcript
    rather than an error, so the artifact is probed — and cross-checked against
    the duration ingest recorded on the object — before any GPU time is spent.
    """
    audio_path = Path(path)
    if not audio_path.is_file():
        raise TranscriptionWorkflowError(f"downloaded audio is missing: {audio_path}")

    size_bytes = audio_path.stat().st_size
    if size_bytes == 0:
        raise TranscriptionWorkflowError(f"downloaded audio is empty: {audio_path}")

    try:
        probed = probe_audio_file(audio_path, ffprobe_path=ffprobe_path)
    except AudioArtifactError as exc:
        raise TranscriptionWorkflowError(f"downloaded audio is not decodable: {exc}") from exc

    duration = probed["duration_seconds"]
    if not duration or duration <= 0:
        raise TranscriptionWorkflowError("downloaded audio has no readable duration")
    if probed["channels"] < 1 or probed["sample_rate"] <= 0:
        raise TranscriptionWorkflowError(
            f"downloaded audio has no usable audio stream "
            f"({probed['channels']} ch, {probed['sample_rate']} Hz)"
        )

    source_duration = _float_or_none((metadata or {}).get("source-duration-seconds"))
    if source_duration:
        drift = abs(duration - source_duration)
        tolerance = duration_tolerance_seconds(source_duration)
        if drift > tolerance:
            raise TranscriptionWorkflowError(
                f"downloaded audio is {duration:.1f}s but the object records a "
                f"{source_duration:.1f}s source ({drift:.1f}s off, tolerance "
                f"{tolerance:.1f}s); transcript timestamps would not line up"
            )

    return AudioArtifactInfo(
        path=audio_path,
        size_bytes=size_bytes,
        codec=probed["codec"],
        sample_rate=probed["sample_rate"],
        channels=probed["channels"],
        duration_seconds=duration,
        source_duration_seconds=source_duration,
    )


def _float_or_none(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def transcript_object_metadata(transcript: Transcript, source_id: str) -> dict[str, str]:
    """ASCII-safe user metadata describing the transcript object."""
    metadata = {
        "artifact": "transcript",
        "source-id": source_id,
        "schema-version": transcript.schema_version,
        "asr-model": transcript.model,
        "asr-device": transcript.device,
        "asr-compute-type": transcript.compute_type,
        "language": transcript.language,
        "language-probability": f"{transcript.language_probability:.3f}",
        "segment-count": str(len(transcript.segments)),
        "word-timestamps": "true" if transcript.word_timestamps else "false",
    }
    if transcript.source_audio_key:
        metadata["source-key"] = transcript.source_audio_key
    if transcript.duration:
        metadata["audio-duration-seconds"] = f"{transcript.duration:.3f}"
    return metadata


def serialize_transcript(transcript: Transcript) -> bytes:
    """Render transcript.json deterministically.

    Key order follows the model definition and never the insertion order of a
    dict, so re-running an unchanged transcription produces an identical body.
    """
    payload = transcript.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False).encode("utf-8")


def existing_transcript(client: Any, bucket: str, key: str) -> Optional[Transcript]:
    """Return the transcript already at `key`, or None if there is none.

    A stored object that will not parse is treated as absent rather than as a
    reason to stop: it cannot be the valid transcript we refuse to overwrite.
    """
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except Exception:
        return None
    try:
        body = response["Body"].read()
        return Transcript.model_validate(json.loads(body.decode("utf-8")))
    except Exception as exc:
        logger.warning("existing transcript at %s is unreadable (%s); treating as absent", key, exc)
        return None


def transcribe_from_r2(
    source_id: str,
    *,
    client: Any,
    bucket: str,
    audio_key: Optional[str] = None,
    transcript_key: Optional[str] = None,
    transcriber: Optional[BaseTranscriber] = None,
    model_name: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    beam_size: int = 5,
    vad_filter: bool = True,
    word_timestamps: bool = True,
    language: Optional[str] = None,
    overwrite: bool = False,
    staging_dir: Optional[str] = None,
    keep_audio: bool = False,
    ffprobe_path: str = "ffprobe",
    on_download_progress: Optional[Callable[[DownloadProgress], None]] = None,
    on_transcribe_progress: Optional[ProgressCallback] = None,
) -> TranscriptionResult:
    """Download the audio artifact, transcribe it, and publish transcript.json.

    Idempotent: an existing, parseable transcript is left alone unless
    `overwrite` is set. Nothing is written to R2 until the transcript is
    complete, so a failure anywhere leaves the previous object untouched.
    """
    started_at = time.perf_counter()
    warnings: list[str] = []
    resolved_audio_key = audio_key or audio_key_for(source_id)
    resolved_transcript_key = transcript_key or transcript_key_for(source_id)

    if resolved_transcript_key == resolved_audio_key:
        raise ValueError("the transcript key would overwrite the audio artifact")

    if not overwrite:
        existing = existing_transcript(client, bucket, resolved_transcript_key)
        if existing is not None:
            logger.info("transcript already present, skipping: %s", resolved_transcript_key)
            return TranscriptionResult(
                bucket=bucket,
                source_id=source_id,
                audio_key=resolved_audio_key,
                transcript_key=resolved_transcript_key,
                transcript=existing,
                skipped=True,
                total_seconds=round(time.perf_counter() - started_at, 3),
            )

    if not object_exists(client, bucket, resolved_audio_key):
        raise TranscriptionWorkflowError(
            f"no audio artifact at s3://{bucket}/{resolved_audio_key}; run ingest first"
        )

    staging_root = tempfile.mkdtemp(prefix="freecher-transcribe-", dir=staging_dir)
    audio_path = Path(staging_root) / AUDIO_FILENAME

    try:
        download_start = time.perf_counter()
        size_bytes, metadata = download_audio_artifact(
            client,
            bucket,
            resolved_audio_key,
            audio_path,
            on_progress=on_download_progress,
        )
        download_seconds = time.perf_counter() - download_start
        logger.info(
            "downloaded %s (%.1f MB) in %.1fs",
            resolved_audio_key,
            size_bytes / 1024 / 1024,
            download_seconds,
        )

        audio = validate_audio_artifact(
            audio_path, metadata=metadata, ffprobe_path=ffprobe_path
        )
        if audio.sample_rate != EXPECTED_SAMPLE_RATE or audio.channels != EXPECTED_CHANNELS:
            # Not fatal: Whisper resamples anyway. Worth saying out loud,
            # because it means the artifact did not come from this ingest.
            warnings.append(
                f"audio is {audio.sample_rate} Hz / {audio.channels} ch, expected "
                f"{EXPECTED_SAMPLE_RATE} Hz mono"
            )

        active = transcriber or WhisperTranscriber(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
        )

        transcribe_start = time.perf_counter()
        transcript = active.transcribe(
            audio_path,
            language=language,
            on_progress=on_transcribe_progress,
        )
        transcribe_seconds = time.perf_counter() - transcribe_start

        transcript.source_bucket = bucket
        transcript.source_audio_key = resolved_audio_key
        transcript.audio_duration = round(audio.duration_seconds, 3)
        transcript.audio_codec = audio.codec
        transcript.audio_sample_rate = audio.sample_rate
        transcript.audio_channels = audio.channels
        transcript.processing_seconds = round(transcribe_seconds, 3)
        transcript.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if not transcript.segments:
            # A complete transcript of silence is legitimate, but on a long
            # source it almost always means the wrong audio or a broken decode,
            # and it would otherwise look like a perfectly valid result.
            warnings.append(
                f"no speech was found in {audio.duration_seconds:.0f}s of audio; "
                "the transcript is empty"
            )

        body = serialize_transcript(transcript)
        # One PutObject: R2 makes an object visible only once it is complete,
        # so an interrupted run can never leave a half-written transcript.
        client.put_object(
            Bucket=bucket,
            Key=resolved_transcript_key,
            Body=body,
            ContentType=TRANSCRIPT_CONTENT_TYPE,
            Metadata=transcript_object_metadata(transcript, source_id),
        )
        logger.info(
            "transcript uploaded: %s/%s (%d segments, %d words, lang=%s)",
            bucket,
            resolved_transcript_key,
            len(transcript.segments),
            transcript.word_count,
            transcript.language,
        )

        return TranscriptionResult(
            bucket=bucket,
            source_id=source_id,
            audio_key=resolved_audio_key,
            transcript_key=resolved_transcript_key,
            transcript=transcript,
            uploaded_bytes=len(body),
            download_seconds=round(download_seconds, 3),
            transcribe_seconds=round(transcribe_seconds, 3),
            total_seconds=round(time.perf_counter() - started_at, 3),
            warnings=warnings,
        )
    finally:
        if keep_audio:
            logger.info("keeping staged audio at %s", audio_path)
        else:
            shutil.rmtree(staging_root, ignore_errors=True)
