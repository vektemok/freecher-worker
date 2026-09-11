"""URL -> finished clips orchestration.

Calls the existing stage implementations; reimplements none of them:

  INGESTING     ingest.service.ingest_to_r2      (streams URL -> R2, + audio sidecar)
  TRANSCRIBING  transcription.r2.transcribe_from_r2
  DISCOVERING   highlights.r2.discover_from_r2   (candidates + scoring + ranking)
  RANKING       same call -- discovery and ranking are one upstream workflow, so
                RANKING is reported as a distinct state but not re-executed
  RENDERING     rendering.batch.render_top_n     (renders + validates)
  UPLOADING     same call -- render_top_n publishes each validated clip to R2

Every stage is preceded by an artifact check that logs exactly one of

    SKIP <stage>: valid artifact already exists
    RUN  <stage>: artifact missing/invalid

so a resumed job's behaviour is readable from the log alone.
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from freecher_worker.config import Settings, get_settings
from freecher_worker.ops.disk import InsufficientDiskSpaceError, require_disk_space
from freecher_worker.jobs.models import Job, JobStatus, StageTiming
from freecher_worker.jobs.store import JobStore

logger = logging.getLogger("freecher_worker")

#: Errors worth retrying: transient transport/service problems. A malformed URL
#: or a missing binary will fail identically on a retry, so it is not retryable.
_RETRYABLE_TYPES = {
    "ConnectionError", "ReadTimeout", "ConnectTimeout", "Timeout",
    "EndpointConnectionError", "ClientError", "IncompleteRead",
    "ChunkedEncodingError", "ProtocolError", "SSLError", "OSError",
    "R2UploadError", "TimeoutExpired",
}
_NON_RETRYABLE_TYPES = {
    "InvalidUrlError", "ValueError", "FileNotFoundError", "KeyError",
    "SubtitleBurnUnsupportedError", "FFmpegNotFoundError", "R2ConfigurationError",
    # Retrying a full disk fills it again. An operator (or `cleanup`) must act.
    "InsufficientDiskSpaceError",
}


def is_retryable(exc: BaseException) -> bool:
    """Classify by the exception's own name, then by its base classes.

    Walking the MRO matters: builtins like TimeoutError and ConnectionResetError
    subclass OSError, and botocore raises many transport errors that derive from
    a small number of retryable bases. Non-retryable wins on a tie -- a specific
    "this will fail again" beats a generic retryable base.
    """
    names = [cls.__name__ for cls in type(exc).__mro__]
    if any(n in _NON_RETRYABLE_TYPES for n in names[:1]):
        return False
    if any(n in _RETRYABLE_TYPES for n in names):
        return True
    if any(n in _NON_RETRYABLE_TYPES for n in names):
        return False
    return False


#: Stages a worker can be interrupted in. A job sitting in one of these with no
#: worker running is orphaned: the process that owned it is gone.
IN_FLIGHT = {
    JobStatus.INGESTING, JobStatus.TRANSCRIBING, JobStatus.DISCOVERING,
    JobStatus.RANKING, JobStatus.RENDERING, JobStatus.UPLOADING,
}


def _requeue(job: Job) -> None:
    """Return a job to the queue without losing what it already achieved."""
    job.status = JobStatus.QUEUED
    job.stage = JobStatus.QUEUED
    job.error_stage = job.error_type = job.error_message = None
    job.retryable = None
    job.completed_at = None


class StageSkipped(Exception):
    """Internal control-flow marker; never surfaces to callers."""


class AwaitingTranscript(Exception):
    """The job is parked on the GPU host, not failed.

    Raised by _transcribe when work has been handed to the Kaggle T4 path. run()
    catches it and records AWAITING_TRANSCRIPT, so the record distinguishes
    "waiting for something that was actually asked for" from "broken".
    """

    def __init__(self, request_key: str, reason: str):
        super().__init__(f"handed off to the GPU host ({reason})")
        self.request_key = request_key
        self.reason = reason


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _object_exists(client: Any, bucket: str, key: str) -> Optional[int]:
    try:
        head = client.head_object(Bucket=bucket, Key=key)
        return int(head.get("ContentLength") or 0)
    except Exception:  # noqa: BLE001
        return None


def _estimated_source_bytes(url: str) -> Optional[int]:
    """A best-effort size for the pending download, or None.

    Used only to tighten the free-space check, never to replace it: yt-dlp's
    `filesize_approx` is missing for many sites and wrong for some.
    """
    try:
        from freecher_worker.ingest.source import probe_source

        info = probe_source(url)
    except Exception:  # noqa: BLE001 - probing is advisory
        return None
    for attr in ("filesize", "filesize_approx"):
        value = getattr(info, attr, None)
        if value:
            return int(value)
    return None


def _audio_duration_seconds(client: Any, bucket: str, key: str) -> Optional[float]:
    """Length of the audio artifact, from the metadata ingest already wrote.

    Metadata-only: reading the object itself just to time it would defeat the
    point of deciding *before* spending the transfer.
    """
    try:
        meta = client.head_object(Bucket=bucket, Key=key).get("Metadata") or {}
    except Exception:  # noqa: BLE001 - absent audio is handled by the stage itself
        return None
    for field in ("audio-duration-seconds", "source-duration-seconds", "duration-seconds"):
        raw = meta.get(field)
        if raw:
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
    return None


def _valid_json_object(client: Any, bucket: str, key: str, min_bytes: int = 2) -> bool:
    size = _object_exists(client, bucket, key)
    if size is None or size < min_bytes:
        return False
    try:
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        json.loads(body)
        return True
    except Exception:  # noqa: BLE001
        return False


class Orchestrator:
    """Runs one job to completion (or failure), persisting every transition."""

    def __init__(self, store: JobStore, settings: Optional[Settings] = None):
        self.store = store
        self.cfg = settings or get_settings()

    # ------------------------------------------------------------- bookkeeping
    def _enter(self, job_id: str, stage: JobStatus) -> float:
        self.store.update(job_id, lambda j: (
            setattr(j, "status", stage), setattr(j, "stage", stage),
            j.stage_timings.append(StageTiming(stage=stage.value, started_at=_now())),
        ) and None)
        return time.perf_counter()

    def _leave(self, job_id: str, t0: float, *, skipped: bool = False,
               reason: Optional[str] = None) -> None:
        seconds = round(time.perf_counter() - t0, 2)

        def mutate(j: Job) -> None:
            if j.stage_timings:
                t = j.stage_timings[-1]
                t.completed_at = _now()
                t.seconds = seconds
                t.skipped = skipped
                t.reason = reason

        self.store.update(job_id, mutate)

    def _log_gate(self, stage: str, skip: bool, reason: str) -> None:
        if skip:
            logger.info("SKIP %s: valid artifact already exists (%s)", stage, reason)
        else:
            logger.info("RUN  %s: artifact missing/invalid (%s)", stage, reason)

    # ------------------------------------------------------------------- stages
    def run(self, job_id: str) -> Job:
        job = self.store.get(job_id)
        cfg = self.cfg
        from freecher_worker.ingest.r2 import build_r2_client
        client = build_r2_client(cfg.r2_endpoint, cfg.r2_access_key_id, cfg.r2_secret_access_key)
        bucket = cfg.r2_bucket

        try:
            self.store.update(job_id, lambda j: (
                setattr(j, "started_at", j.started_at or _now()),
                setattr(j, "error_stage", None), setattr(j, "error_type", None),
                setattr(j, "error_message", None), setattr(j, "retryable", None),
            ) and None)

            source_id = self._ingest(job_id, client, bucket)
            self._transcribe(job_id, source_id, client, bucket)
            run_dir = self._discover_and_rank(job_id, source_id, client, bucket)
            self._render_and_publish(job_id, source_id, run_dir, client, bucket)

            self.store.update(job_id, lambda j: (
                setattr(j, "status", JobStatus.DONE), setattr(j, "stage", JobStatus.DONE),
                setattr(j, "completed_at", _now()),
            ) and None)
            logger.info("[job %s] DONE", job_id)
            self._cleanup_if_configured(job_id, client)
        except (KeyboardInterrupt, SystemExit):
            # An operator stopped the service; the pipeline did not fail. Put the
            # job back in the queue so the restarted worker resumes it (every
            # completed stage will skip), then let the signal do its job.
            # Re-raising matters as much as the requeue: swallowing it left the
            # worker loop polling after SIGINT until systemd escalated to SIGKILL.
            logger.warning("[job %s] interrupted in %s; requeueing", job_id,
                           self.store.get(job_id).stage.value)
            self.store.update(job_id, _requeue)
            raise
        except AwaitingTranscript as parked:
            def park(j: Job) -> None:
                j.status = JobStatus.AWAITING_TRANSCRIPT
                j.stage = JobStatus.AWAITING_TRANSCRIPT
                j.transcribe_request_key = parked.request_key

            self.store.update(job_id, park)
            logger.info("[job %s] AWAITING_TRANSCRIPT: %s", job_id, parked.reason)
        except BaseException as exc:  # noqa: BLE001 - every failure must be recorded
            stage = self.store.get(job_id).stage
            logger.error("[job %s] FAILED in %s\n%s", job_id, stage.value,
                         traceback.format_exc())          # full traceback: logs only
            retry = is_retryable(exc)

            def fail(j: Job) -> None:
                j.status = JobStatus.FAILED
                j.error_stage = stage.value
                j.error_type = type(exc).__name__
                j.error_message = str(exc)[:500] or type(exc).__name__
                j.retryable = retry
                j.completed_at = _now()

            self.store.update(job_id, fail)
        return self.store.get(job_id)

    # --------------------------------------------------------------- INGESTING
    def _ingest(self, job_id: str, client: Any, bucket: str) -> str:
        from freecher_worker.ingest.service import ingest_to_r2, render_key
        from freecher_worker.ingest.source import probe_source

        job = self.store.get(job_id)
        t0 = self._enter(job_id, JobStatus.INGESTING)

        # Resolve the prospective source_id BEFORE transferring anything. A brand
        # new job for an already-ingested URL has no source_id of its own, so
        # without this probe it would re-download the source and then fail on the
        # existing object. The probe is metadata-only.
        candidate_id = job.source_id
        if not candidate_id:
            try:
                info = probe_source(job.source_url)
                candidate_id = render_key(self.cfg.ingest_key_template, info).split("/")[-2]
            except Exception as exc:  # noqa: BLE001 - probing is best-effort
                logger.info("[ingest] could not pre-resolve source id (%s); ingesting", exc)

        if candidate_id:
            key = f"input/{candidate_id}/source.mp4"
            size = _object_exists(client, bucket, key)
            if size and size > 0:
                self._log_gate("INGESTING", True, f"{key} = {size} bytes")
                self.store.update(job_id, lambda j: setattr(j, "source_id", candidate_id))
                self._leave(job_id, t0, skipped=True, reason="source object present")
                return candidate_id

        self._log_gate("INGESTING", False, "no existing source object for this URL")
        # Before pulling potentially gigabytes: a full disk turns into a
        # truncated source and a confusing failure three stages later.
        require_disk_space(Path(self.cfg.runs_dir), "INGESTING", settings=self.cfg,
                           expected_bytes=_estimated_source_bytes(job.source_url))
        result = ingest_to_r2(
            job.source_url, client=client, bucket=bucket,
            key_template=self.cfg.ingest_key_template,
            quality=self.cfg.ingest_quality if hasattr(self.cfg, "ingest_quality") else "best",
            extract_audio=True,
            public_base_url=self.cfg.r2_public_base_url,
            overwrite=False,
        )
        source_id = (result.video.video_id if result.video
                     else result.key.split("/")[-2])
        self.store.update(job_id, lambda j: setattr(j, "source_id", source_id))
        self._leave(job_id, t0)
        return source_id

    # ------------------------------------------------------------ TRANSCRIBING
    def _transcribe(self, job_id: str, source_id: str, client: Any, bucket: str) -> None:
        from freecher_worker.transcription.handoff import (
            BACKEND_REMOTE, clear_request, plan_transcription, publish_request,
        )
        from freecher_worker.transcription.r2 import (
            audio_key_for, transcribe_from_r2, transcript_key_for,
        )

        t0 = self._enter(job_id, JobStatus.TRANSCRIBING)
        key = transcript_key_for(source_id)
        if _valid_json_object(client, bucket, key, min_bytes=64):
            self._log_gate("TRANSCRIBING", True, key)
            # A satisfied request is queue litter; drop it so the GPU host's
            # pending list stays an accurate to-do.
            clear_request(client, bucket, source_id)
            self.store.update(job_id, lambda j: setattr(j, "transcribe_request_key", None))
            self._leave(job_id, t0, skipped=True, reason="transcript present and parseable")
            return

        self._log_gate("TRANSCRIBING", False, key)
        plan = plan_transcription(
            self.cfg, _audio_duration_seconds(client, bucket, audio_key_for(source_id)))
        logger.info("[transcribe] placement=%s (%s)", plan.backend, plan.reason)

        if plan.backend == BACKEND_REMOTE:
            request_key = publish_request(client, bucket, source_id, self.cfg, job_id=job_id)
            self._leave(job_id, t0, skipped=True, reason=f"handed off: {plan.reason}")
            raise AwaitingTranscript(request_key, plan.reason)

        transcribe_from_r2(
            source_id, client=client, bucket=bucket,
            model_name=self.cfg.asr_model, device=self.cfg.asr_device,
            compute_type=self.cfg.asr_compute_type, beam_size=self.cfg.asr_beam_size,
            vad_filter=self.cfg.asr_vad_filter, language=self.cfg.asr_language,
            overwrite=False,
        )
        self._leave(job_id, t0)

    # ------------------------------------------------ DISCOVERING + RANKING
    def _discover_and_rank(self, job_id: str, source_id: str, client: Any, bucket: str) -> Path:
        from freecher_worker.highlights.r2 import (
            candidates_key_for, discover_from_r2, highlights_key_for, manifest_key_for,
        )

        run_dir = Path("runs") / source_id
        t0 = self._enter(job_id, JobStatus.DISCOVERING)
        keys = (candidates_key_for(source_id), highlights_key_for(source_id),
                manifest_key_for(source_id))
        complete = all(_valid_json_object(client, bucket, k) for k in keys)

        if complete and all((run_dir / n).is_file()
                            for n in ("candidates.json", "highlights.json",
                                      "transcript.json", "manifest.json")):
            self._log_gate("DISCOVERING", True, "candidates/highlights/manifest present")
            self._leave(job_id, t0, skipped=True, reason="artifact set complete")
        else:
            self._log_gate("DISCOVERING", False,
                           "manifest missing or local mirror incomplete")
            discover_from_r2(
                source_id, client=client, bucket=bucket,
                min_seconds=self.cfg.highlight_min_seconds,
                target_seconds=self.cfg.highlight_target_seconds,
                max_seconds=self.cfg.highlight_max_seconds,
                overlap_seconds=self.cfg.highlight_overlap_seconds,
                top_k=self.cfg.highlight_top_k,
                dedup_threshold=self.cfg.dedup_overlap_threshold,
                overwrite=False, local_dir=run_dir,
            )
            self._leave(job_id, t0)

        # Ranking is produced by the same upstream workflow. Report the state for
        # observability, but never re-run scoring -- the scorer is frozen.
        t1 = self._enter(job_id, JobStatus.RANKING)
        self._log_gate("RANKING", True, "produced by discover_from_r2 (frozen scorer)")
        self._leave(job_id, t1, skipped=True, reason="ranked by the discovery workflow")
        self.store.update(job_id, lambda j: setattr(j, "run_dir", str(run_dir)))
        return run_dir

    # ----------------------------------------------- RENDERING + UPLOADING
    def _render_and_publish(self, job_id: str, source_id: str, run_dir: Path,
                            client: Any, bucket: str) -> None:
        from freecher_worker.rendering.batch import (
            MANIFEST_KEY_TEMPLATE, ClipStatus, render_top_n,
        )

        job = self.store.get(job_id)
        source_video = self._ensure_local_source(source_id, run_dir, client, bucket)

        t0 = self._enter(job_id, JobStatus.RENDERING)
        require_disk_space(run_dir, "RENDERING", settings=self.cfg)
        self._log_gate("RENDERING", False,
                       "render_top_n performs its own per-clip artifact reuse")
        manifest = render_top_n(
            run_dir=run_dir, source_video=source_video, source_id=source_id,
            top_n=job.top_n, publish=True, settings=self.cfg,
        )
        self._leave(job_id, t0)

        # render_top_n validates and publishes in the same pass; UPLOADING is the
        # state in which we verify the published result rather than a second send.
        t1 = self._enter(job_id, JobStatus.UPLOADING)
        key = MANIFEST_KEY_TEMPLATE.format(source_id=source_id)
        if not _valid_json_object(client, bucket, key):
            raise RuntimeError(f"clips manifest {key} missing or unreadable after render")
        done = [c for c in manifest.clips if c.status is ClipStatus.DONE]
        if manifest.status != "DONE" or not done:
            failed = [f"{c.clip_id}: {c.error_type}: {c.error_message}"
                      for c in manifest.clips if c.status is not ClipStatus.DONE]
            raise RuntimeError("clip rendering incomplete -- " + "; ".join(failed)[:400])
        for c in done:
            if not c.r2_key or _object_exists(client, bucket, c.r2_key) is None:
                raise RuntimeError(f"clip {c.clip_id} is not present in R2")

        payload = [c.model_dump(mode="json") for c in done]
        self.store.update(job_id, lambda j: (
            setattr(j, "clips", payload), setattr(j, "clips_manifest_key", key),
        ) and None)
        self._leave(job_id, t1)

    # ------------------------------------------------------- post-DONE cleanup
    def _cleanup_if_configured(self, job_id: str, client: Any) -> None:
        """Reclaim regenerable local files once the outputs are safely in R2.

        Opt-in via FREECHER_CLEANUP_AFTER_DONE. Failure here is logged and
        swallowed: the job succeeded, and housekeeping must not turn a delivered
        result into a reported failure.
        """
        if not getattr(self.cfg, "cleanup_after_done", False):
            return
        try:
            from freecher_worker.ops.cleanup import cleanup_completed_job

            result = cleanup_completed_job(self.store.get(job_id), settings=self.cfg,
                                           store=self.store, client=client)
            logger.info("[job %s] post-DONE cleanup freed %.2f GiB across %d item(s)",
                        job_id, result.freed_bytes / (1 << 30), result.deleted)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[job %s] post-DONE cleanup skipped: %s: %s",
                           job_id, type(exc).__name__, exc)

    # ------------------------------------------------------ crash recovery
    def reclaim_orphans(self) -> list[str]:
        """Requeue jobs left mid-stage by a worker that died.

        Called once when a worker starts. A clean stop requeues its own job, but
        a SIGKILL, an OOM or a power loss cannot; without this the record sits in
        RENDERING for ever and no worker will ever claim it, because
        claim_next_queued only takes QUEUED.

        Safe only because the supported topology is one worker per job
        directory -- see the note in jobs.store. A second concurrent worker would
        reclaim the first one's live job.
        """
        orphans = [j for j in self.store.list() if j.status in IN_FLIGHT]
        for job in orphans:
            logger.warning("[job %s] orphaned in %s; requeueing", job.job_id, job.stage.value)
            self.store.update(job.job_id, _requeue)
        return [j.job_id for j in orphans]

    # -------------------------------------------------------- parked jobs
    def resume_awaiting(self) -> list[str]:
        """Requeue every parked job whose transcript has since appeared.

        Called from the worker's idle loop. Cheap: one HEAD per parked job, and
        there is normally at most a handful. Returns the job ids requeued so the
        caller can log them.
        """
        from freecher_worker.transcription.r2 import transcript_key_for

        parked = [j for j in self.store.list() if j.status is JobStatus.AWAITING_TRANSCRIPT]
        if not parked:
            return []

        from freecher_worker.ingest.r2 import build_r2_client
        cfg = self.cfg
        client = build_r2_client(cfg.r2_endpoint, cfg.r2_access_key_id, cfg.r2_secret_access_key)

        resumed: list[str] = []
        for job in parked:
            if not job.source_id:
                continue
            key = transcript_key_for(job.source_id)
            if not _valid_json_object(client, cfg.r2_bucket, key, min_bytes=64):
                continue
            logger.info("[job %s] transcript arrived (%s); requeueing", job.job_id, key)
            self.store.update(job.job_id, lambda j: (
                setattr(j, "status", JobStatus.QUEUED),
                setattr(j, "stage", JobStatus.QUEUED),
                setattr(j, "transcribe_request_key", None),
            ) and None)
            resumed.append(job.job_id)
        return resumed

    # ------------------------------------------------------------------ helper
    def _ensure_local_source(self, source_id: str, run_dir: Path, client: Any,
                             bucket: str) -> Path:
        """The renderer needs the source on disk; fetch it once, then reuse."""
        run_dir.mkdir(parents=True, exist_ok=True)
        local = run_dir / "source.mp4"
        key = f"input/{source_id}/source.mp4"
        remote_size = _object_exists(client, bucket, key)
        if remote_size is None:
            raise FileNotFoundError(f"source object missing in R2: {key}")
        if local.is_file() and local.stat().st_size == remote_size:
            logger.info("SKIP source download: %s already matches R2 (%s bytes)",
                        local, remote_size)
            return local
        logger.info("RUN  source download: %s -> %s (%s bytes)", key, local, remote_size)
        client.download_file(bucket, key, str(local))
        return local
