"""Minimal HTTP API over the job store.

Deliberately thin: it validates input, creates/reads job records, and shapes the
response. It never imports the pipeline or runs work in the request path -- a
separate `freecher-worker worker` process drains the queue. That keeps the HTTP
layer isolated from pipeline implementation and means an API restart cannot
strand a running job.

Errors reaching the client are typed and message-only. Tracebacks stay in logs.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional
from urllib.parse import urlparse

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from freecher_worker.api.auth import AuthenticatedUser, get_current_user
from freecher_worker.jobs.models import Job, JobStatus
from freecher_worker.jobs.store import JobNotFound, JobStore

logger = logging.getLogger("freecher_worker")

ALLOWED_SCHEMES = {"http", "https"}
MAX_TOP_N = 20


def _store() -> JobStore:
    from freecher_worker.jobs.store import DEFAULT_ROOT
    return JobStore(os.environ.get("FREECHER_JOBS_DIR") or DEFAULT_ROOT)


class CreateJobRequest(BaseModel):
    url: str = Field(description="Public video URL to ingest")
    top_n: int = Field(default=5, ge=1, le=MAX_TOP_N)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("url must not be empty")
        parsed = urlparse(v)
        if parsed.scheme.lower() not in ALLOWED_SCHEMES:
            raise ValueError("url must be http(s)")
        if not parsed.netloc:
            raise ValueError("url must include a host")
        return v


class CreateJobResponse(BaseModel):
    job_id: str
    status: str


class ClipOut(BaseModel):
    clip_id: str
    rank: int
    duration_seconds: float
    url: Optional[str] = None
    r2_key: Optional[str] = None
    bytes: Optional[int] = None
    sha256: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    progress: int
    source_url: Optional[str] = None
    owner_user_id: Optional[str] = None
    source_id: Optional[str] = None
    error: Optional[dict[str, Any]] = None
    clips: list[ClipOut] = Field(default_factory=list)


def _to_response(job: Job) -> JobResponse:
    clips = [
        ClipOut(
            clip_id=c.get("clip_id", ""), rank=int(c.get("rank", 0)),
            duration_seconds=float(c.get("duration_seconds", 0.0)),
            url=c.get("r2_url"), r2_key=c.get("r2_key"),
            bytes=c.get("bytes"), sha256=c.get("sha256"),
        )
        for c in job.clips
    ]
    return JobResponse(
        job_id=job.job_id, status=job.status.value, stage=job.stage.value,
        progress=job.progress, source_url=job.source_url, owner_user_id=job.owner_user_id, source_id=job.source_id,
        error=job.public_error(), clips=clips,
    )


def create_app(store: Optional[JobStore] = None) -> FastAPI:
    app = FastAPI(title="Freecher", version="0.2.0",
                  description="URL in, finished vertical clips out.")
    app.state.store = store or _store()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # noqa: ANN202
        # The client gets a type and a message; the traceback goes to the log.
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500,
                            content={"detail": "internal error", "type": type(exc).__name__})

    @app.get("/health")
    def health(deep: bool = True) -> JSONResponse:
        """Deployment diagnostics, not just liveness.

        A 200 here means this host can actually run a job: R2 answers, ffmpeg
        exists and can burn subtitles, the job directory is writable. A degraded
        host answers 503 with the same body, so a monitor sees the failure and an
        operator reading it sees which check failed and why.

        `?deep=false` skips the R2 round trip for a pure liveness probe.
        """
        from freecher_worker.ops.diagnostics import collect

        diag = collect(jobs_dir=app.state.store.root, include_network=deep)
        return JSONResponse(status_code=200 if diag.ok else 503, content=diag.as_dict())

    @app.post("/jobs", response_model=CreateJobResponse, status_code=201)
    def create_job(
        body: CreateJobRequest,
        background_tasks: BackgroundTasks,
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> CreateJobResponse:
        job = app.state.store.create(
            source_url=body.url, top_n=body.top_n, owner_user_id=user.uid
        )
        logger.info(
            "[api] queued job %s for %s by user %s (top_n=%d)",
            job.job_id, body.url, user.uid, body.top_n
        )
        if os.environ.get("FREECHER_INLINE_WORKER", "1") == "1":
            from freecher_worker.jobs.orchestrator import Orchestrator
            background_tasks.add_task(Orchestrator(app.state.store).run, job.job_id)
        return CreateJobResponse(job_id=job.job_id, status=job.status.value)

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    def get_job(
        job_id: str,
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> JobResponse:
        try:
            job = app.state.store.get(job_id)
        except (JobNotFound, ValueError):
            raise HTTPException(status_code=404, detail="job not found")
        if job.owner_user_id is not None and job.owner_user_id != user.uid:
            raise HTTPException(status_code=404, detail="job not found")
        return _to_response(job)

    @app.get("/jobs", response_model=list[JobResponse])
    def list_jobs(
        limit: int = 50,
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> list[JobResponse]:
        return [
            _to_response(j)
            for j in app.state.store.list(owner_user_id=user.uid)[: max(1, min(limit, 200))]
        ]

    @app.post("/jobs/{job_id}/retry", response_model=JobResponse)
    def retry_job(
        job_id: str,
        background_tasks: BackgroundTasks,
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> JobResponse:
        try:
            job = app.state.store.get(job_id)
        except (JobNotFound, ValueError):
            raise HTTPException(status_code=404, detail="job not found")
        if job.owner_user_id is not None and job.owner_user_id != user.uid:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status is JobStatus.DONE:
            raise HTTPException(status_code=409, detail="job already completed")
        if job.status not in (JobStatus.FAILED, JobStatus.QUEUED):
            raise HTTPException(status_code=409,
                                detail=f"job is {job.status.value}; only FAILED jobs can be retried")

        def reset(j: Job) -> None:
            j.status = JobStatus.QUEUED
            j.stage = JobStatus.QUEUED
            j.error_stage = j.error_type = j.error_message = None
            j.retryable = None
            j.completed_at = None

        # Resume, not restart: source_id, run_dir and clips are kept so the
        # orchestrator's per-stage artifact checks can skip completed work.
        updated_job = app.state.store.update(job_id, reset)
        if os.environ.get("FREECHER_INLINE_WORKER", "1") == "1":
            from freecher_worker.jobs.orchestrator import Orchestrator
            background_tasks.add_task(Orchestrator(app.state.store).run, job_id)
        return _to_response(updated_job)

    return app


app = create_app()
