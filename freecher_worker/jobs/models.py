"""Job state model for the URL -> finished clips pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    INGESTING = "INGESTING"
    TRANSCRIBING = "TRANSCRIBING"
    AWAITING_TRANSCRIPT = "AWAITING_TRANSCRIPT"
    DISCOVERING = "DISCOVERING"
    RANKING = "RANKING"
    RENDERING = "RENDERING"
    UPLOADING = "UPLOADING"
    DONE = "DONE"
    FAILED = "FAILED"


#: Coarse, honest progress. These are stage markers, not measured fractions --
#: the pipeline cannot measure true completion inside a stage, so we do not
#: pretend to. RANKING shares DISCOVERING's call, hence the small step.
STAGE_PROGRESS: dict[JobStatus, int] = {
    JobStatus.QUEUED: 0,
    JobStatus.INGESTING: 10,
    JobStatus.TRANSCRIBING: 30,
    # Parked, not stalled: the GPU host has been asked and the job resumes the
    # moment transcript.json appears. Same progress as TRANSCRIBING because it
    # is the same stage, waiting rather than computing.
    JobStatus.AWAITING_TRANSCRIPT: 30,
    JobStatus.DISCOVERING: 55,
    JobStatus.RANKING: 65,
    JobStatus.RENDERING: 75,
    JobStatus.UPLOADING: 95,
    JobStatus.DONE: 100,
    JobStatus.FAILED: 0,
}

TERMINAL = {JobStatus.DONE, JobStatus.FAILED}

#: Order the orchestrator walks. FAILED/QUEUED are not stages.
STAGE_ORDER = [
    JobStatus.INGESTING,
    JobStatus.TRANSCRIBING,
    JobStatus.DISCOVERING,
    JobStatus.RANKING,
    JobStatus.RENDERING,
    JobStatus.UPLOADING,
]


class StageTiming(BaseModel):
    stage: str
    started_at: str
    completed_at: Optional[str] = None
    seconds: Optional[float] = None
    skipped: bool = False
    reason: Optional[str] = None


class Job(BaseModel):
    job_id: str
    source_url: str
    source_id: Optional[str] = None
    status: JobStatus = JobStatus.QUEUED
    stage: JobStatus = JobStatus.QUEUED
    top_n: int = 5

    created_at: str = Field(default_factory=_now)
    started_at: Optional[str] = None
    updated_at: str = Field(default_factory=_now)
    completed_at: Optional[str] = None

    #: Set while status is AWAITING_TRANSCRIPT: the R2 key of the request the
    #: GPU host is expected to pick up. Cleared once a transcript exists.
    transcribe_request_key: Optional[str] = None

    error_stage: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    retryable: Optional[bool] = None

    attempts: int = 0
    stage_timings: list[StageTiming] = Field(default_factory=list)
    clips: list[dict[str, Any]] = Field(default_factory=list)
    run_dir: Optional[str] = None
    clips_manifest_key: Optional[str] = None

    @property
    def progress(self) -> int:
        if self.status is JobStatus.FAILED:
            # Report how far it got, not zero -- otherwise a late failure looks
            # identical to one that never started.
            return STAGE_PROGRESS.get(self.stage, 0)
        return STAGE_PROGRESS.get(self.status, 0)

    def public_error(self) -> Optional[dict[str, Any]]:
        """User-facing error. Never carries a traceback."""
        if self.status is not JobStatus.FAILED:
            return None
        return {
            "stage": self.error_stage,
            "type": self.error_type,
            "message": self.error_message,
            "retryable": bool(self.retryable),
        }
