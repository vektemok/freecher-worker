"""Deployment and execution-placement behaviour.

These cover the things that only break on a real host: an ffmpeg build older
than the developer's, a transcription stage that has no GPU under it, a health
endpoint that has to tell an operator which of six things is wrong, and unit
files whose paths must agree with the environment template they ship beside.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import freecher_worker.rendering  # noqa: F401  (import-order; subtitles<->rendering cycle)
from freecher_worker.api.app import create_app
from freecher_worker.jobs.models import STAGE_PROGRESS, JobStatus
from freecher_worker.jobs.orchestrator import AwaitingTranscript, Orchestrator
from freecher_worker.jobs.store import JobStore
from freecher_worker.media import ffmpeg_env
from freecher_worker.ops import diagnostics
from freecher_worker.transcription import handoff

DEPLOY = Path(__file__).resolve().parent.parent / "deploy"
URL = "https://example.com/video.mp4"


class Cfg:
    """Minimal settings stand-in; only the fields under test."""

    transcribe_backend = "auto"
    cpu_transcription_max_seconds = 900.0
    allow_cpu_transcription = False
    transcribe_model = "large-v3"
    transcribe_device = "cuda"
    transcribe_compute_type = "float16"
    transcribe_beam_size = 5
    transcribe_vad_filter = True
    transcribe_word_timestamps = True
    asr_language = None
    r2_bucket = "b"
    r2_endpoint = "https://acct-id.r2.cloudflarestorage.com"
    r2_access_key_id = "AKIAsecret"
    r2_secret_access_key = "topsecretvalue"


# ------------------------------------------------------- ffmpeg listing parser
#: FFmpeg 6.1 (Ubuntu 24.04, the production ARM host) prints no separator row.
FFMPEG_61_FILTERS = """Filters:
  T.. = Timeline support
  .S. = Slice threading
  ..C = Command support
  A = Audio input/output
  | = Source or sink filter
 ... abench            A->A       Benchmark part of a filtergraph.
 ..C crop              V->V       Crop the input video.
 ... ass               V->V       Render ASS subtitles onto input video using libass.
 ... sendcmd           V->V       Send commands to filters.
 ..C scale             V->V       Scale the input video size.
 ... loudnorm          A->A       EBU R128 loudness normalization
"""

#: FFmpeg 7+ does print one. Both must parse identically.
FFMPEG_9_FILTERS = FFMPEG_61_FILTERS.replace(
    " ... abench", "  ------\n ... abench")

ENCODERS = """Encoders:
 V..... = Video
 A..... = Audio
 ------
 V....D libx264              libx264 H.264 / AVC
 A....D aac                  AAC (Advanced Audio Coding)
"""


def _stub_ffmpeg(monkeypatch, filters: str, encoders: str = ENCODERS) -> None:
    class Result:
        def __init__(self, out): self.stdout, self.stderr = out, ""

    def run(cmd, **kwargs):
        return Result(filters if "-filters" in cmd else encoders)

    monkeypatch.setattr(ffmpeg_env.subprocess, "run", run)


@pytest.mark.parametrize("listing,label", [(FFMPEG_61_FILTERS, "6.1"), (FFMPEG_9_FILTERS, "9")])
def test_filters_parse_with_and_without_a_separator_row(monkeypatch, listing, label):
    """The deployment bug: requiring '------' hid every filter on Ubuntu's ffmpeg."""
    _stub_ffmpeg(monkeypatch, listing)
    names = ffmpeg_env._list_section("ffmpeg", "-filters")
    assert {"crop", "scale", "loudnorm", "sendcmd", "ass"} <= names, label


def test_legend_rows_are_not_mistaken_for_entries(monkeypatch):
    _stub_ffmpeg(monkeypatch, FFMPEG_61_FILTERS)
    names = ffmpeg_env._list_section("ffmpeg", "-filters")
    assert "=" not in names and "Timeline" not in names and "------" not in names


def test_encoders_parse_the_same_way(monkeypatch):
    _stub_ffmpeg(monkeypatch, FFMPEG_61_FILTERS)
    assert {"libx264", "aac"} <= ffmpeg_env._list_section("ffmpeg", "-encoders")


def test_capabilities_on_an_ubuntu_build_report_subtitle_support(monkeypatch, tmp_path):
    _stub_ffmpeg(monkeypatch, FFMPEG_61_FILTERS)
    monkeypatch.setattr(ffmpeg_env, "resolve_ffmpeg", lambda s=None: "/usr/bin/ffmpeg")
    monkeypatch.setattr(ffmpeg_env, "resolve_ffprobe", lambda s=None: "/usr/bin/ffprobe")
    ffmpeg_env.clear_capability_cache()
    caps = ffmpeg_env.probe_capabilities(refresh=True)
    assert caps.supports_subtitle_burn and caps.subtitle_filter == "ass"
    assert not caps.missing_filters() and not caps.missing_encoders()
    ffmpeg_env.clear_capability_cache()


# ------------------------------------------------------- transcription routing
@pytest.mark.parametrize("backend,seconds,allow,expected", [
    ("remote", 10,   False, "remote"),   # explicit handoff wins regardless of length
    ("auto",   40,   False, "local"),    # short audio still runs on a CPU laptop
    ("auto",   7200, False, "remote"),   # a 2-hour VOD goes to the GPU host
    ("auto",   None, False, "remote"),   # unknown length is treated as long
    ("local",  7200, True,  "local"),    # opted in explicitly
])
def test_placement_matrix(monkeypatch, backend, seconds, allow, expected):
    monkeypatch.setattr(handoff, "cuda_available", lambda: False)
    cfg = Cfg()
    cfg.transcribe_backend, cfg.allow_cpu_transcription = backend, allow
    assert handoff.plan_transcription(cfg, seconds).backend == expected


def test_explicit_local_refuses_long_cpu_audio(monkeypatch):
    monkeypatch.setattr(handoff, "cuda_available", lambda: False)
    cfg = Cfg()
    cfg.transcribe_backend = "local"
    with pytest.raises(handoff.CpuTranscriptionRefused) as excinfo:
        handoff.plan_transcription(cfg, 7200)
    message = str(excinfo.value)
    assert "FREECHER_TRANSCRIBE_BACKEND=remote" in message
    assert "FREECHER_ALLOW_CPU_TRANSCRIPTION" in message


def test_a_gpu_host_always_runs_locally(monkeypatch):
    monkeypatch.setattr(handoff, "cuda_available", lambda: True)
    assert handoff.plan_transcription(Cfg(), 999_999).backend == "local"


def test_unknown_backend_is_rejected():
    cfg = Cfg()
    cfg.transcribe_backend = "kaggle"
    with pytest.raises(ValueError):
        handoff.plan_transcription(cfg, 10)


# ------------------------------------------------------------ handoff contract
class FakeR2:
    """Just enough S3 surface for the handoff, with recorded writes."""

    def __init__(self, objects=()):
        self.objects: dict[str, bytes] = {k: b"{}" for k in objects}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            # Shaped like botocore's ClientError, because object_exists only
            # treats a 404-shaped failure as "absent" and re-raises anything
            # else -- a fake that raises a bare exception would test nothing.
            error = Exception("Not Found")
            error.response = {"Error": {"Code": "404"},
                              "ResponseMetadata": {"HTTPStatusCode": 404}}
            raise error
        return {"ContentLength": len(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kw):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        class Body:
            def __init__(self, b): self._b = b
            def read(self): return self._b
        return {"Body": Body(self.objects[Key])}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)

    def get_paginator(self, _name):
        outer = self

        class P:
            def paginate(self, Bucket, Prefix):
                return [{"Contents": [{"Key": k, "LastModified": ""}
                                      for k in outer.objects if k.startswith(Prefix)]}]
        return P()


def test_request_carries_the_gpu_profile_not_the_local_one():
    request = handoff.build_request("src1", Cfg(), job_id="job1")
    assert request["model"] == "large-v3" and request["device"] == "cuda"
    assert request["audio_key"] == "processing/src1/audio.m4a"
    assert request["transcript_key"] == "processing/src1/transcript.json"
    assert request["requested_by_job"] == "job1"
    assert request["schema"] == handoff.REQUEST_SCHEMA


def test_publishing_a_request_without_audio_is_refused():
    with pytest.raises(handoff.TranscriptionHandoffError):
        handoff.publish_request(FakeR2(), "b", "src1", Cfg())


def test_publish_then_read_round_trips():
    client = FakeR2(["processing/src1/audio.m4a"])
    key = handoff.publish_request(client, "b", "src1", Cfg(), job_id="j")
    assert key == "processing/src1/transcribe_request.json"
    assert handoff.read_request(client, "b", "src1")["source_id"] == "src1"


def test_pending_queue_hides_requests_that_are_already_satisfied():
    client = FakeR2(["processing/a/audio.m4a", "processing/b/audio.m4a"])
    handoff.publish_request(client, "bucket", "a", Cfg())
    handoff.publish_request(client, "bucket", "b", Cfg())
    client.objects["processing/b/transcript.json"] = b"{}"
    pending = handoff.pending_requests(client, "bucket")
    assert [r["source_id"] for r in pending] == ["a"]


def test_clearing_a_request_is_safe_when_it_is_already_gone():
    handoff.clear_request(FakeR2(), "b", "src1")     # must not raise


# ------------------------------------------------------- parked job lifecycle
def test_awaiting_transcript_is_a_stage_not_a_failure():
    assert STAGE_PROGRESS[JobStatus.AWAITING_TRANSCRIPT] == STAGE_PROGRESS[JobStatus.TRANSCRIBING]
    assert JobStatus.AWAITING_TRANSCRIPT.value == "AWAITING_TRANSCRIPT"


def test_a_handed_off_job_parks_instead_of_failing(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    job = store.create(URL)
    monkeypatch.setattr("freecher_worker.ingest.r2.build_r2_client", lambda *a, **k: object())
    monkeypatch.setattr(Orchestrator, "_ingest", lambda self, jid, *a: "src1")
    monkeypatch.setattr(Orchestrator, "_transcribe", lambda self, *a: (_ for _ in ()).throw(
        AwaitingTranscript("processing/src1/transcribe_request.json", "no local GPU")))
    result = Orchestrator(store).run(job.job_id)
    assert result.status is JobStatus.AWAITING_TRANSCRIPT
    assert result.error_type is None, "parking is not an error"
    assert result.transcribe_request_key == "processing/src1/transcribe_request.json"
    assert result.progress == 30


def test_a_parked_job_is_requeued_once_the_transcript_lands(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    job = store.create(URL)
    store.update(job.job_id, lambda j: (setattr(j, "status", JobStatus.AWAITING_TRANSCRIPT),
                                        setattr(j, "source_id", "src1")) and None)
    client = FakeR2(["processing/src1/transcript.json"])
    # Deliberately a plausible transcript: the resume gate requires a parseable
    # object of at least 64 bytes, so that a truncated write does not read as done.
    client.objects["processing/src1/transcript.json"] = json.dumps(
        {"language": "en", "duration": 42.0, "segments": [
            {"start": 0.0, "end": 2.0, "text": "hello there"}]}).encode()
    monkeypatch.setattr("freecher_worker.ingest.r2.build_r2_client", lambda *a, **k: client)
    resumed = Orchestrator(store, settings=Cfg()).resume_awaiting()
    assert resumed == [job.job_id]
    assert store.get(job.job_id).status is JobStatus.QUEUED
    assert store.get(job.job_id).transcribe_request_key is None


def test_a_parked_job_stays_parked_while_the_transcript_is_missing(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    job = store.create(URL)
    store.update(job.job_id, lambda j: (setattr(j, "status", JobStatus.AWAITING_TRANSCRIPT),
                                        setattr(j, "source_id", "src1")) and None)
    monkeypatch.setattr("freecher_worker.ingest.r2.build_r2_client", lambda *a, **k: FakeR2())
    assert Orchestrator(store, settings=Cfg()).resume_awaiting() == []
    assert store.get(job.job_id).status is JobStatus.AWAITING_TRANSCRIPT


def test_the_api_reports_a_parked_job_plainly(tmp_path):
    store = JobStore(tmp_path / "jobs")
    client = TestClient(create_app(store))
    jid = client.post("/jobs", json={"url": URL}).json()["job_id"]
    store.update(jid, lambda j: (setattr(j, "status", JobStatus.AWAITING_TRANSCRIPT),
                                 setattr(j, "stage", JobStatus.AWAITING_TRANSCRIPT)) and None)
    body = client.get(f"/jobs/{jid}").json()
    assert body["status"] == "AWAITING_TRANSCRIPT" and body["error"] is None


# ----------------------------------------------------------------- diagnostics
def test_r2_check_names_the_missing_variables_without_inventing_values():
    class Empty:
        r2_bucket = r2_endpoint = r2_access_key_id = r2_secret_access_key = None

    check = diagnostics.check_r2(Empty(), use_cache=False)
    assert not check.ok and "FREECHER_R2_BUCKET" in check.detail


def test_diagnostics_never_serialise_a_credential(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "check_r2",
                        lambda cfg, **kw: diagnostics.Check("r2", True, "bucket 'b' reachable"))
    blob = json.dumps(diagnostics.collect(Cfg(), jobs_dir=tmp_path).as_dict())
    assert Cfg.r2_secret_access_key not in blob
    assert Cfg.r2_access_key_id not in blob
    assert "acct-id" not in blob, "the endpoint embeds the account id"


def test_a_degraded_host_is_reported_as_degraded(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "check_ffmpeg",
                        lambda cfg: [diagnostics.Check("ffmpeg", False, "not found")])
    monkeypatch.setattr(diagnostics, "check_r2",
                        lambda cfg, **kw: diagnostics.Check("r2", True, "ok"))
    diag = diagnostics.collect(Cfg(), jobs_dir=tmp_path)
    assert diag.status == "degraded" and [c.name for c in diag.failures()] == ["ffmpeg"]


def test_jobs_check_fails_on_an_unwritable_directory(tmp_path):
    target = tmp_path / "ro" / "jobs"
    target.parent.mkdir()
    target.parent.chmod(0o500)
    try:
        assert not diagnostics.check_jobs_dir(target).ok
    finally:
        target.parent.chmod(0o700)


def test_health_returns_503_when_the_host_cannot_work(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "check_ffmpeg",
                        lambda cfg: [diagnostics.Check("ffmpeg", False, "not found")])
    client = TestClient(create_app(JobStore(tmp_path / "jobs")))
    response = client.get("/health?deep=false")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert any(c["name"] == "ffmpeg" and not c["ok"] for c in response.json()["checks"])


def test_health_can_skip_the_network_probe(tmp_path, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("deep=false must not touch the network")

    monkeypatch.setattr(diagnostics, "check_r2", explode)
    client = TestClient(create_app(JobStore(tmp_path / "jobs")))
    assert client.get("/health?deep=false").status_code in (200, 503)


# --------------------------------------------------------------- systemd units
@pytest.mark.parametrize("unit", ["freecher-api.service", "freecher-worker.service"])
def test_units_restart_and_start_at_boot(unit):
    text = (DEPLOY / unit).read_text()
    assert "Restart=always" in text
    assert "WantedBy=multi-user.target" in text, "must survive a reboot"
    assert "EnvironmentFile=/etc/freecher/freecher.env" in text
    assert "ExecStart=/opt/freecher/venv/bin/freecher-worker" in text
    assert "User=freecher" in text, "must not run as root"


@pytest.mark.parametrize("unit", ["freecher-api.service", "freecher-worker.service"])
def test_units_keep_a_writable_state_directory(unit):
    text = (DEPLOY / unit).read_text()
    assert "ProtectSystem=strict" in text
    assert "ReadWritePaths=/var/lib/freecher" in text
    # ProtectSystem=strict makes /opt read-only, so the process must not run
    # from the app directory it would try to write ./runs into.
    assert "WorkingDirectory=/var/lib/freecher" in text


def test_the_env_template_and_the_units_agree_on_the_job_directory():
    env = (DEPLOY / "freecher.env.example").read_text()
    assert "FREECHER_JOBS_DIR=/var/lib/freecher/jobs" in env
    for unit in ("freecher-api.service", "freecher-worker.service"):
        assert "--jobs-dir /var/lib/freecher/jobs" in (DEPLOY / unit).read_text()


def test_the_env_template_carries_no_real_secret():
    env = (DEPLOY / "freecher.env.example").read_text()
    for line in env.splitlines():
        if line.startswith("FREECHER_R2_SECRET") or line.startswith("FREECHER_R2_ACCESS"):
            assert line.split("=", 1)[1].startswith("<"), f"placeholder expected: {line}"


def test_the_installer_refuses_to_start_a_degraded_host():
    script = (DEPLOY / "install.sh").read_text()
    assert "preflight" in script
    assert "Nothing was enabled or started." in script


# ------------------------------------------------------------ restart safety
def test_an_operator_stop_requeues_rather_than_failing(tmp_path, monkeypatch):
    """systemd sends SIGINT on stop; that is not a pipeline failure."""
    store = JobStore(tmp_path / "jobs")
    job = store.create(URL)
    monkeypatch.setattr("freecher_worker.ingest.r2.build_r2_client", lambda *a, **k: object())
    def ingest(self, jid, *a):
        self.store.update(jid, lambda j: setattr(j, "source_id", "src1"))
        return "src1"

    monkeypatch.setattr(Orchestrator, "_ingest", ingest)
    monkeypatch.setattr(Orchestrator, "_transcribe",
                        lambda self, *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        Orchestrator(store).run(job.job_id)      # must propagate, or the unit hangs
    again = store.get(job.job_id)
    assert again.status is JobStatus.QUEUED
    assert again.error_type is None
    assert again.source_id == "src1", "resume, not restart"


@pytest.mark.parametrize("stage", [JobStatus.INGESTING, JobStatus.RENDERING,
                                   JobStatus.UPLOADING])
def test_a_worker_reclaims_jobs_orphaned_by_a_kill(tmp_path, stage):
    """SIGKILL leaves no chance to requeue, so the next worker must notice."""
    store = JobStore(tmp_path / "jobs")
    job = store.create(URL)
    store.update(job.job_id, lambda j: (setattr(j, "status", stage),
                                        setattr(j, "stage", stage)) and None)
    assert Orchestrator(store).reclaim_orphans() == [job.job_id]
    assert store.get(job.job_id).status is JobStatus.QUEUED


@pytest.mark.parametrize("stage", [JobStatus.DONE, JobStatus.FAILED,
                                   JobStatus.QUEUED, JobStatus.AWAITING_TRANSCRIPT])
def test_reclaim_leaves_settled_and_parked_jobs_alone(tmp_path, stage):
    store = JobStore(tmp_path / "jobs")
    job = store.create(URL)
    store.update(job.job_id, lambda j: setattr(j, "status", stage))
    assert Orchestrator(store).reclaim_orphans() == []
    assert store.get(job.job_id).status is stage
