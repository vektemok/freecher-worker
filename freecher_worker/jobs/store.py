"""Durable JSON job store.

One file per job under a deterministic path, written atomically (temp file in the
same directory + os.replace, which is atomic on POSIX and Windows). That is
enough for this milestone: jobs are few, writes are small and infrequent, and the
store must survive a process restart. No Redis/Postgres/Celery is introduced --
none of them are existing dependencies.

Concurrency: a single in-process lock serialises writes, and `update()` re-reads
the record before mutating so a worker and the API do not clobber each other's
fields. That is sufficient for the supported topology (one API, one worker on one
host). It is NOT a distributed lock, and the docstring says so deliberately.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

from freecher_worker.jobs.models import Job, JobStatus

_LOCK = threading.RLock()
DEFAULT_ROOT = Path("runs") / "_jobs"


class JobNotFound(KeyError):
    pass


class JobStore:
    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def path_for(self, job_id: str) -> Path:
        safe = "".join(c for c in job_id if c.isalnum() or c in "-_")
        if not safe or safe != job_id:
            raise ValueError(f"unsafe job id: {job_id!r}")
        return self.root / f"{safe}.json"

    # ------------------------------------------------------------------- CRUD
    def create(self, source_url: str, top_n: int = 5, job_id: Optional[str] = None) -> Job:
        job = Job(job_id=job_id or uuid.uuid4().hex, source_url=source_url, top_n=top_n)
        self._write(job)
        return job

    def get(self, job_id: str) -> Job:
        path = self.path_for(job_id)
        if not path.is_file():
            raise JobNotFound(job_id)
        return Job.model_validate_json(path.read_text())

    def exists(self, job_id: str) -> bool:
        try:
            return self.path_for(job_id).is_file()
        except ValueError:
            return False

    def list(self) -> list[Job]:
        out = []
        for p in sorted(self.root.glob("*.json")):
            try:
                out.append(Job.model_validate_json(p.read_text()))
            except Exception:  # noqa: BLE001 - a corrupt file must not hide the rest
                continue
        return sorted(out, key=lambda j: j.created_at, reverse=True)

    def update(self, job_id: str, mutate: Callable[[Job], None]) -> Job:
        """Re-read, mutate, write atomically. `mutate` must not raise."""
        with _LOCK:
            job = self.get(job_id)
            mutate(job)
            job.updated_at = datetime.now(timezone.utc).isoformat()
            self._write(job)
            return job

    def claim_next_queued(self) -> Optional[Job]:
        """Atomically take the oldest QUEUED job, for a worker loop."""
        with _LOCK:
            for job in sorted(self.list(), key=lambda j: j.created_at):
                if job.status is JobStatus.QUEUED:
                    job.status = JobStatus.INGESTING
                    job.stage = JobStatus.INGESTING
                    job.started_at = job.started_at or datetime.now(timezone.utc).isoformat()
                    job.attempts += 1
                    job.updated_at = datetime.now(timezone.utc).isoformat()
                    self._write(job)
                    return job
        return None

    # ------------------------------------------------------------------ write
    def _write(self, job: Job) -> None:
        path = self.path_for(job.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".job-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(job.model_dump_json(indent=2))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)          # atomic
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def iter_jobs(self) -> Iterator[Job]:
        yield from self.list()
