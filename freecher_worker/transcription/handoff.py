"""GPU transcription handoff: R2 is the queue.

The Oracle ARM VM that runs ingest and orchestration has no CUDA, and CPU
faster-whisper on 2 ARM cores is not a production path for a multi-hour VOD.
The already-proven GPU path is the Kaggle T4 notebook running
`transcription.r2.transcribe_from_r2`, which reads processing/{id}/audio.m4a
from R2 and writes processing/{id}/transcript.json back.

Nothing about Whisper is reimplemented here. This module only adds the smallest
contract that lets the two hosts hand work to each other through the bucket they
already share:

    processing/{id}/transcribe_request.json   written by the Oracle worker
    processing/{id}/transcript.json           written by the GPU host  (existing)

The request object is the queue entry. The transcript object is the completion
signal -- there is no second "done" flag to keep in sync, because the artifact
the pipeline actually needs is the only truth worth reading.

Kaggle invocation is not automated: Kaggle offers no API this host can drive to
start a T4 session on demand. The handoff is therefore explicit, and a job that
is waiting says so (AWAITING_TRANSCRIPT) instead of failing or hanging.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, NamedTuple, Optional

from freecher_worker.transcription.r2 import (
    PROCESSING_PREFIX,
    audio_key_for,
    transcript_key_for,
)

logger = logging.getLogger("freecher_worker")

REQUEST_FILENAME = "transcribe_request.json"
REQUEST_CONTENT_TYPE = "application/json"

#: Version the contract so a GPU host running older code can refuse politely
#: instead of misreading a field it does not know about.
REQUEST_SCHEMA = "freecher.transcribe_request/1"

BACKEND_LOCAL = "local"
BACKEND_REMOTE = "remote"
BACKEND_AUTO = "auto"
BACKENDS = (BACKEND_AUTO, BACKEND_LOCAL, BACKEND_REMOTE)


class TranscriptionHandoffError(Exception):
    """The handoff contract could not be honoured."""


class CpuTranscriptionRefused(Exception):
    """A long audio would have been transcribed on CPU without being asked to.

    Deliberately not retryable: retrying changes nothing. The message names the
    two ways out (point the job at the GPU host, or opt in explicitly).
    """


def request_key_for(source_id: str) -> str:
    cleaned = source_id.strip().strip("/")
    if not cleaned or "/" in cleaned:
        raise ValueError(f"source id must be a single path segment, got '{source_id}'")
    return f"{PROCESSING_PREFIX}/{cleaned}/{REQUEST_FILENAME}"


def cuda_available() -> bool:
    """True only if a CUDA device is actually visible, not merely importable."""
    try:
        import ctranslate2  # noqa: PLC0415

        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:  # noqa: BLE001 - absence is the normal case, not an error
        return False


def resolve_backend(settings: Any) -> str:
    """The configured placement, with 'auto' resolved against the local device.

    'auto' answers only the hardware question. Whether a CPU host may attempt a
    particular audio is a separate, length-dependent decision -- see
    `plan_transcription`.
    """
    backend = str(getattr(settings, "transcribe_backend", BACKEND_AUTO) or BACKEND_AUTO).lower()
    if backend not in BACKENDS:
        raise ValueError(
            f"transcribe_backend must be one of {', '.join(BACKENDS)}, got '{backend}'"
        )
    if backend != BACKEND_AUTO:
        return backend
    return BACKEND_LOCAL if cuda_available() else BACKEND_REMOTE


class TranscriptionPlan(NamedTuple):
    """Where this particular transcription runs, and why."""

    backend: str
    reason: str


def plan_transcription(settings: Any, audio_seconds: Optional[float] = None) -> TranscriptionPlan:
    """Decide placement for one source, refusing slow CPU work that was never asked for.

    The rule the milestone asks for: a long video must not quietly start a
    multi-hour CPU transcription. But a 40-second clip on a laptop should still
    just work, so the length guard -- not the mere absence of a GPU -- is what
    diverts work to the GPU host.

      explicit 'local'  : run here; refuse only if it is CPU, long, and not opted in
      explicit 'remote' : always hand off
      'auto'            : GPU here -> local; else short enough -> local; else remote

    `audio_seconds` of None means the duration could not be read, which is
    treated as "possibly long" for the auto decision but never as grounds to
    refuse an explicit instruction.
    """
    configured = str(getattr(settings, "transcribe_backend", BACKEND_AUTO) or BACKEND_AUTO).lower()
    if configured not in BACKENDS:
        raise ValueError(
            f"transcribe_backend must be one of {', '.join(BACKENDS)}, got '{configured}'"
        )
    limit = float(getattr(settings, "cpu_transcription_max_seconds", 900.0))
    allowed = bool(getattr(settings, "allow_cpu_transcription", False))

    if configured == BACKEND_REMOTE:
        return TranscriptionPlan(BACKEND_REMOTE, "configured for handoff")
    if cuda_available():
        return TranscriptionPlan(BACKEND_LOCAL, "CUDA device present")
    if allowed:
        return TranscriptionPlan(BACKEND_LOCAL, "CPU transcription explicitly allowed")
    if audio_seconds is not None and audio_seconds <= limit:
        return TranscriptionPlan(
            BACKEND_LOCAL, f"CPU, but audio is {audio_seconds:.0f}s <= {limit:.0f}s limit")

    described = "unknown length" if audio_seconds is None else f"{audio_seconds:.0f}s"
    if configured == BACKEND_AUTO:
        return TranscriptionPlan(
            BACKEND_REMOTE, f"CPU host and audio is {described} (> {limit:.0f}s limit)")

    raise CpuTranscriptionRefused(
        f"refusing to transcribe {described} of audio on CPU: it would take hours. "
        f"Either set FREECHER_TRANSCRIBE_BACKEND=remote to hand off to the GPU host, "
        f"or set FREECHER_ALLOW_CPU_TRANSCRIPTION=true to accept the wall time "
        f"(current limit: FREECHER_CPU_TRANSCRIPTION_MAX_SECONDS={limit:.0f})."
    )


def build_request(source_id: str, settings: Any, *, job_id: Optional[str] = None) -> dict[str, Any]:
    """The queue entry. Carries the GPU-side decoding config, not the local one.

    `asr_*` describes the modest-hardware `process` pipeline; `transcribe_*` is
    the T4 profile the R2 workflow was built for, so that is what the GPU host
    is asked to use.
    """
    return {
        "schema": REQUEST_SCHEMA,
        "source_id": source_id,
        "audio_key": audio_key_for(source_id),
        "transcript_key": transcript_key_for(source_id),
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "requested_by_job": job_id,
        "model": settings.transcribe_model,
        "device": settings.transcribe_device,
        "compute_type": settings.transcribe_compute_type,
        "beam_size": settings.transcribe_beam_size,
        "vad_filter": settings.transcribe_vad_filter,
        "word_timestamps": settings.transcribe_word_timestamps,
        "language": settings.asr_language,
    }


def publish_request(client: Any, bucket: str, source_id: str, settings: Any, *,
                    job_id: Optional[str] = None) -> str:
    """Put the request object and return its key. Idempotent by overwrite.

    Overwriting is correct: a re-queued job wants the newest decoding config and
    the newest job id, and a stale request for an already-transcribed source is
    harmless because the transcript, not the request, ends the wait.
    """
    from freecher_worker.ingest.service import object_exists

    key = request_key_for(source_id)
    if not object_exists(client, bucket, audio_key_for(source_id)):
        raise TranscriptionHandoffError(
            f"cannot request transcription for '{source_id}': "
            f"{audio_key_for(source_id)} is missing. Ingest must run first."
        )
    payload = json.dumps(build_request(source_id, settings, job_id=job_id), indent=2)
    client.put_object(Bucket=bucket, Key=key, Body=payload.encode("utf-8"),
                      ContentType=REQUEST_CONTENT_TYPE)
    logger.info("[transcribe] handoff requested: s3://%s/%s (model=%s device=%s)",
                bucket, key, settings.transcribe_model, settings.transcribe_device)
    return key


def read_request(client: Any, bucket: str, source_id: str) -> Optional[dict[str, Any]]:
    """The pending request for a source, or None."""
    try:
        body = client.get_object(Bucket=bucket, Key=request_key_for(source_id))["Body"].read()
        return json.loads(body)
    except Exception:  # noqa: BLE001 - absent or unreadable both mean "no request"
        return None


def clear_request(client: Any, bucket: str, source_id: str) -> None:
    """Remove a satisfied request. Failure to delete is not an error."""
    try:
        client.delete_object(Bucket=bucket, Key=request_key_for(source_id))
    except Exception as exc:  # noqa: BLE001
        logger.debug("[transcribe] could not clear request for %s: %s", source_id, exc)


def pending_requests(client: Any, bucket: str, limit: int = 50) -> list[dict[str, Any]]:
    """Requests whose transcript has not been written yet, oldest first.

    This is what the GPU host polls. It lists only the small request objects, so
    the scan costs nothing next to the transcription itself.
    """
    from freecher_worker.ingest.service import object_exists

    out: list[dict[str, Any]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{PROCESSING_PREFIX}/"):
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key", "")
            if not key.endswith(f"/{REQUEST_FILENAME}"):
                continue
            source_id = key.split("/")[-2]
            if object_exists(client, bucket, transcript_key_for(source_id)):
                continue                      # already satisfied; nothing to do
            request = read_request(client, bucket, source_id) or {"source_id": source_id}
            request.setdefault("source_id", source_id)
            request["_key"] = key
            request["_last_modified"] = str(obj.get("LastModified", ""))
            out.append(request)
    out.sort(key=lambda r: r.get("requested_at") or r.get("_last_modified") or "")
    return out[:limit]
