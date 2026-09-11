"""Job store, orchestration state machine, and the HTTP API."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import freecher_worker.rendering  # noqa: F401  (import-order; subtitles<->rendering cycle)
from freecher_worker.api.app import create_app
from freecher_worker.jobs.models import STAGE_PROGRESS, Job, JobStatus
from freecher_worker.jobs.orchestrator import Orchestrator, is_retryable
from freecher_worker.jobs.store import JobNotFound, JobStore

URL = "https://example.com/video.mp4"


@pytest.fixture
def store(tmp_path) -> JobStore:
    return JobStore(tmp_path / "jobs")


@pytest.fixture
def client(store) -> TestClient:
    return TestClient(create_app(store))


# ------------------------------------------------------------------ job store
def test_create_and_get_round_trip(store):
    job = store.create(URL, top_n=3)
    again = store.get(job.job_id)
    assert again.source_url == URL and again.top_n == 3
    assert again.status is JobStatus.QUEUED


def test_job_survives_a_process_restart(store, tmp_path):
    job = store.create(URL)
    store.update(job.job_id, lambda j: setattr(j, "source_id", "abc"))
    reloaded = JobStore(tmp_path / "jobs").get(job.job_id)   # fresh instance
    assert reloaded.source_id == "abc"


def test_unknown_job_raises(store):
    with pytest.raises(JobNotFound):
        store.get("does-not-exist")


def test_unsafe_job_id_is_rejected(store):
    with pytest.raises(ValueError):
        store.path_for("../escape")


def test_writes_are_atomic_leaving_no_temp_files(store):
    job = store.create(URL)
    for i in range(5):
        store.update(job.job_id, lambda j, i=i: setattr(j, "top_n", i + 1))
    leftovers = list(store.root.glob(".job-*"))
    assert not leftovers, f"temp files left behind: {leftovers}"
    assert store.get(job.job_id).top_n == 5


def test_claim_next_queued_takes_one_job_once(store):
    a = store.create(URL)
    claimed = store.claim_next_queued()
    assert claimed and claimed.job_id == a.job_id
    assert claimed.status is JobStatus.INGESTING
    assert store.claim_next_queued() is None, "an in-flight job must not be claimed twice"


def test_progress_is_monotonic_across_the_stage_order():
    order = [JobStatus.QUEUED, JobStatus.INGESTING, JobStatus.TRANSCRIBING,
             JobStatus.DISCOVERING, JobStatus.RANKING, JobStatus.RENDERING,
             JobStatus.UPLOADING, JobStatus.DONE]
    values = [STAGE_PROGRESS[s] for s in order]
    assert values == sorted(values)
    assert values[0] == 0 and values[-1] == 100


def test_failed_job_reports_the_progress_it_reached():
    job = Job(job_id="x", source_url=URL, status=JobStatus.FAILED,
              stage=JobStatus.RENDERING)
    assert job.progress == STAGE_PROGRESS[JobStatus.RENDERING]


# ------------------------------------------------------------ retry semantics
@pytest.mark.parametrize("exc,expected", [
    (ConnectionError("net"), True),
    (TimeoutError("slow"), True),
    (ValueError("bad url"), False),
    (FileNotFoundError("missing"), False),
])
def test_retryability_classification(exc, expected):
    assert is_retryable(exc) is expected


# -------------------------------------------------------------- orchestration
def test_failure_is_persisted_with_stage_and_type(store, monkeypatch):
    job = store.create(URL)
    orch = Orchestrator(store)
    monkeypatch.setattr("freecher_worker.ingest.r2.build_r2_client",
                        lambda *a, **k: object())
    monkeypatch.setattr(Orchestrator, "_ingest",
                        lambda self, *a, **k: (_ for _ in ()).throw(ConnectionError("boom")))
    result = orch.run(job.job_id)
    assert result.status is JobStatus.FAILED
    assert result.error_type == "ConnectionError"
    assert "boom" in result.error_message
    assert result.retryable is True
    assert result.error_stage  # the stage it died in is recorded


def test_failure_message_carries_no_traceback(store, monkeypatch):
    job = store.create(URL)
    monkeypatch.setattr("freecher_worker.ingest.r2.build_r2_client", lambda *a, **k: object())
    monkeypatch.setattr(Orchestrator, "_ingest",
                        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("kaput")))
    result = Orchestrator(store).run(job.job_id)
    blob = json.dumps(result.public_error())
    assert "Traceback" not in blob and "File \"" not in blob


def test_successful_run_records_every_stage(store, monkeypatch):
    job = store.create(URL)
    monkeypatch.setattr("freecher_worker.ingest.r2.build_r2_client", lambda *a, **k: object())
    monkeypatch.setattr(Orchestrator, "_ingest", lambda self, jid, *a: "src1")
    monkeypatch.setattr(Orchestrator, "_transcribe", lambda self, *a: None)
    monkeypatch.setattr(Orchestrator, "_discover_and_rank", lambda self, *a: Path("runs/src1"))
    monkeypatch.setattr(Orchestrator, "_render_and_publish", lambda self, *a: None)
    result = Orchestrator(store).run(job.job_id)
    assert result.status is JobStatus.DONE
    assert result.completed_at


# ----------------------------------------------------------------------- API
def test_create_job_returns_queued(client):
    r = client.post("/jobs", json={"url": URL, "top_n": 3})
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "QUEUED" and body["job_id"]


@pytest.mark.parametrize("url", ["", "   ", "ftp://x/y", "not-a-url", "https://"])
def test_invalid_urls_are_rejected(client, url):
    assert client.post("/jobs", json={"url": url}).status_code == 422


@pytest.mark.parametrize("top_n", [0, -1, 999])
def test_out_of_range_top_n_is_rejected(client, top_n):
    assert client.post("/jobs", json={"url": URL, "top_n": top_n}).status_code == 422


def test_unknown_job_returns_404_without_internals(client):
    r = client.get("/jobs/deadbeefdeadbeef")
    assert r.status_code == 404
    assert r.json() == {"detail": "job not found"}


def test_unsafe_job_id_returns_404_not_500(client):
    assert client.get("/jobs/..%2F..%2Fetc").status_code in (404, 400)


def test_get_reports_stage_and_progress(client, store):
    jid = client.post("/jobs", json={"url": URL}).json()["job_id"]
    store.update(jid, lambda j: (setattr(j, "status", JobStatus.RENDERING),
                                 setattr(j, "stage", JobStatus.RENDERING)) and None)
    body = client.get(f"/jobs/{jid}").json()
    assert body["status"] == "RENDERING" and body["stage"] == "RENDERING"
    assert body["progress"] == 75 and body["error"] is None


def test_done_response_exposes_clip_metadata(client, store):
    jid = client.post("/jobs", json={"url": URL}).json()["job_id"]
    clip = {"clip_id": "c1", "rank": 1, "duration_seconds": 60.5,
            "r2_key": "output/s/clips/c1.mp4", "r2_url": "https://cdn/c1.mp4",
            "bytes": 123, "sha256": "abc"}
    store.update(jid, lambda j: (setattr(j, "status", JobStatus.DONE),
                                 setattr(j, "stage", JobStatus.DONE),
                                 setattr(j, "clips", [clip])) and None)
    body = client.get(f"/jobs/{jid}").json()
    assert body["status"] == "DONE" and body["progress"] == 100
    assert len(body["clips"]) == 1
    got = body["clips"][0]
    assert got["clip_id"] == "c1" and got["rank"] == 1
    assert got["url"] == "https://cdn/c1.mp4" and got["r2_key"] == "output/s/clips/c1.mp4"
    assert got["bytes"] == 123 and got["sha256"] == "abc"


def test_failed_response_is_typed_and_traceback_free(client, store):
    jid = client.post("/jobs", json={"url": URL}).json()["job_id"]
    store.update(jid, lambda j: (
        setattr(j, "status", JobStatus.FAILED), setattr(j, "stage", JobStatus.TRANSCRIBING),
        setattr(j, "error_stage", "TRANSCRIBING"), setattr(j, "error_type", "ConnectionError"),
        setattr(j, "error_message", "connection reset"), setattr(j, "retryable", True),
    ) and None)
    body = client.get(f"/jobs/{jid}").json()
    assert body["status"] == "FAILED"
    assert body["error"]["type"] == "ConnectionError"
    assert body["error"]["retryable"] is True
    assert "Traceback" not in json.dumps(body)


def test_retry_requeues_a_failed_job_and_clears_the_error(client, store):
    jid = client.post("/jobs", json={"url": URL}).json()["job_id"]
    store.update(jid, lambda j: (
        setattr(j, "status", JobStatus.FAILED), setattr(j, "source_id", "src1"),
        setattr(j, "error_type", "ConnectionError"), setattr(j, "error_message", "x"),
    ) and None)
    body = client.post(f"/jobs/{jid}/retry").json()
    assert body["status"] == "QUEUED" and body["error"] is None
    # resume, not restart: identity of prior work is preserved
    assert store.get(jid).source_id == "src1"


def test_retry_of_a_completed_job_is_refused(client, store):
    jid = client.post("/jobs", json={"url": URL}).json()["job_id"]
    store.update(jid, lambda j: setattr(j, "status", JobStatus.DONE))
    assert client.post(f"/jobs/{jid}/retry").status_code == 409


def test_retry_of_an_unknown_job_is_404(client):
    assert client.post("/jobs/nope/retry").status_code == 404


def test_api_never_runs_the_pipeline_in_the_request(client, store):
    """POST must only enqueue; a worker does the work."""
    jid = client.post("/jobs", json={"url": URL}).json()["job_id"]
    assert store.get(jid).status is JobStatus.QUEUED
    assert store.get(jid).source_id is None
