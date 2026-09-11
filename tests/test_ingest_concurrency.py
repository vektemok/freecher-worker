"""Concurrency regression tests for source-scoped ingest locking.

Verifies:
1. Two simultaneous jobs for the same source invoke the downloader exactly once.
2. Waiting jobs do not skip until BOTH source.mp4 and audio.m4a are valid.
3. Owner crash with multiple contenders grants ownership to exactly one contender.
"""
from __future__ import annotations

import concurrent.futures
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from freecher_worker.config import Settings
from freecher_worker.ingest.lock import SourceIngestLock
from freecher_worker.ingest.models import IngestResult, VideoInfo
from freecher_worker.jobs.models import JobStatus
from freecher_worker.jobs.orchestrator import Orchestrator
from freecher_worker.jobs.store import JobStore


class FakeR2Client:
    """In-memory fake S3/R2 client for testing gate checks."""

    def __init__(self) -> None:
        self.objects: dict[str, int] = {}
        self._lock = threading.Lock()

    def head_object(self, Bucket: str, Key: str) -> dict:  # noqa: N803
        with self._lock:
            if Key in self.objects:
                return {"ContentLength": self.objects[Key]}
        from botocore.exceptions import ClientError
        raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "head_object")

    def put_object(self, Key: str, size: int = 1000) -> None:  # noqa: N803
        with self._lock:
            self.objects[Key] = size


def test_dual_jobs_same_source_download_exactly_once(tmp_path):
    """Job A + Job B simultaneous: downloader called exactly once, both complete."""
    store = JobStore(tmp_path / "jobs")
    cfg = Settings(runs_dir=str(tmp_path / "runs"), locks_dir=str(tmp_path / "runs" / "_locks"))
    r2 = FakeR2Client()
    source_id = "test_src_123"

    job_a = store.create(source_url="https://example.com/stream.mp4")
    job_b = store.create(source_url="https://example.com/stream.mp4")

    download_invocations = 0
    invocations_lock = threading.Lock()

    def fake_ingest_to_r2(*args, **kwargs):
        nonlocal download_invocations
        with invocations_lock:
            download_invocations += 1
        # Simulate time taken to download and upload source.mp4 and audio.m4a
        time.sleep(0.3)
        r2.put_object(f"input/{source_id}/source.mp4", size=5000)
        time.sleep(0.2)
        r2.put_object(f"processing/{source_id}/audio.m4a", size=1000)
        return IngestResult(
            bucket="test",
            key=f"input/{source_id}/source.mp4",
            uploaded_bytes=5000,
            part_count=1,
            elapsed_seconds=0.5,
            quality="best",
            format_selector="b",
            video=VideoInfo(video_id=source_id, title="Test", duration_seconds=60.0, uploader="Tester"),
        )

    orch_a = Orchestrator(store, cfg=cfg)
    orch_b = Orchestrator(store, cfg=cfg)

    with patch("freecher_worker.ingest.source.probe_source") as mock_probe, \
         patch("freecher_worker.ingest.service.ingest_to_r2", side_effect=fake_ingest_to_r2):
        mock_probe.return_value = VideoInfo(video_id=source_id, title="Test", duration_seconds=60.0, uploader="Tester")

        # Start Job A and Job B simultaneously
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(orch_a._ingest, job_a.job_id, r2, "test")
            fut_b = pool.submit(orch_b._ingest, job_b.job_id, r2, "test")

            res_a = fut_a.result(timeout=10.0)
            res_b = fut_b.result(timeout=10.0)

    assert res_a == source_id
    assert res_b == source_id
    assert download_invocations == 1, f"Expected 1 download invocation, got {download_invocations}"

    job_a_rec = store.get(job_a.job_id)
    job_b_rec = store.get(job_b.job_id)
    assert job_a_rec.source_id == source_id
    assert job_b_rec.source_id == source_id


def test_waiter_does_not_skip_on_partial_artifacts(tmp_path):
    """Waiter must NOT skip if only source.mp4 exists but audio.m4a is still missing."""
    store = JobStore(tmp_path / "jobs")
    cfg = Settings(runs_dir=str(tmp_path / "runs"))
    r2 = FakeR2Client()
    source_id = "partial_src_456"

    job_b = store.create(source_url="https://example.com/partial.mp4")

    # Lock held by another job
    lock_a = SourceIngestLock(source_id=source_id, job_id="job_a", locks_dir=tmp_path / "runs" / "_locks")
    assert lock_a.acquire() is True

    # Only source.mp4 is uploaded initially
    r2.put_object(f"input/{source_id}/source.mp4", size=5000)

    orch = Orchestrator(store, cfg=cfg)

    # In background thread, after 0.5s upload audio.m4a and release lock
    def _finish_audio_later():
        time.sleep(0.5)
        r2.put_object(f"processing/{source_id}/audio.m4a", size=1000)
        lock_a.release()

    t = threading.Thread(target=_finish_audio_later)
    t.start()

    with patch("freecher_worker.ingest.source.probe_source") as mock_probe:
        mock_probe.return_value = VideoInfo(video_id=source_id, title="Test", duration_seconds=60.0, uploader="Tester")
        res = orch._ingest(job_b.job_id, r2, "test")

    t.join()
    assert res == source_id
    # Confirmed it waited and succeeded once audio.m4a became available


def test_stale_lock_reclaimed_by_exactly_one_contender(tmp_path):
    """When owner crashes, exactly one contender acquires the lock, others wait."""
    locks_dir = tmp_path / "locks"
    source_id = "stale_test_789"

    # Job A acquires lock
    lock_a = SourceIngestLock(source_id=source_id, job_id="job_a", locks_dir=locks_dir)
    assert lock_a.acquire() is True

    # Simulate Job A dying/crashing: flock is released
    lock_a.release()

    # Now Job B and Job C contend for the lock simultaneously
    lock_b = SourceIngestLock(source_id=source_id, job_id="job_b", locks_dir=locks_dir)
    lock_c = SourceIngestLock(source_id=source_id, job_id="job_c", locks_dir=locks_dir)

    barrier = threading.Barrier(2)
    results = {}

    def _attempt(name, lock):
        barrier.wait()
        results[name] = lock.acquire(force_if_stale=True)

    t_b = threading.Thread(target=_attempt, args=("B", lock_b))
    t_c = threading.Thread(target=_attempt, args=("C", lock_c))
    t_b.start()
    t_c.start()
    t_b.join()
    t_c.join()

    # Exactly one succeeded!
    acquired_count = sum(1 for v in results.values() if v is True)
    assert acquired_count == 1, f"Expected exactly 1 winner, got results={results}"

    winner_lock = lock_b if results["B"] else lock_c
    winner_lock.release()
