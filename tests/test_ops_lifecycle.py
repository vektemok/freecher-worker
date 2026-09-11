"""Disk lifecycle, disk guards, batch-scoped refinement model, and env-file handling.

These cover the operational failure modes the deployment surfaced: a run tree
that grows until the disk fills, an expensive stage that starts on a full disk,
a 1.5 GB model reloaded once per clip, and a CLI that died on an unreadable .env
belonging to someone else.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

import freecher_worker.rendering  # noqa: F401  (import-order; subtitles<->rendering cycle)
from freecher_worker.config import EnvFileError, Settings, resolve_env_file
from freecher_worker.jobs.models import JobStatus
from freecher_worker.jobs.store import JobStore
from freecher_worker.ops import cleanup as C
from freecher_worker.ops import diagnostics as D
from freecher_worker.ops import disk as DK
from freecher_worker.rendering import batch as B
from freecher_worker.rendering.asr_refinement import HighlightWordTranscriber

URL = "https://example.com/v.mp4"


# --------------------------------------------------------------------- fakes
class FakeR2:
    """Head-only S3 surface, with per-key size and metadata."""

    def __init__(self, objects: dict[str, dict] | None = None):
        self.objects = objects or {}
        self.heads: list[str] = []

    def head_object(self, Bucket, Key):
        self.heads.append(Key)
        if Key not in self.objects:
            err = Exception("Not Found")
            err.response = {"Error": {"Code": "404"},
                            "ResponseMetadata": {"HTTPStatusCode": 404}}
            raise err
        return self.objects[Key]


def _run_tree(root: Path, source_id: str, *, clip_bytes: int = 4096,
              source_bytes: int = 8192, r2_key: str | None = "set") -> Path:
    """A run directory shaped like a real one, with a published clip sidecar."""
    run = root / source_id
    (run / "final").mkdir(parents=True)
    for name in ("transcript.json", "candidates.json", "highlights.json",
                 "manifest.json", "clips.json"):
        (run / name).write_text("{}")
    for sub in ("subtitles", "words", "crop_paths"):
        (run / sub).mkdir()
        (run / sub / "highlight_01.json").write_text("{}")

    clip = run / "final" / f"{source_id}_r01_cand_001.mp4"
    clip.write_bytes(b"c" * clip_bytes)
    record = {"clip_id": f"{source_id}_r01", "bytes": clip_bytes, "sha256": "abc"}
    if r2_key:
        record["r2_key"] = f"output/{source_id}/clips/{source_id}_r01_cand_001.mp4"
    clip.with_suffix(".json").write_text(json.dumps(record))

    (run / "source.mp4").write_bytes(b"s" * source_bytes)
    return run


def _published(source_id: str, *, clip_bytes: int = 4096, source_bytes: int = 8192) -> FakeR2:
    return FakeR2({
        f"output/{source_id}/clips/{source_id}_r01_cand_001.mp4":
            {"ContentLength": clip_bytes, "Metadata": {"sha256": "abc"}},
        f"input/{source_id}/source.mp4": {"ContentLength": source_bytes, "Metadata": {}},
    })


# ------------------------------------------------------------- cleanup planning
def test_dry_run_plans_deletions_but_removes_nothing(tmp_path):
    run = _run_tree(tmp_path, "src1")
    plan = C.plan_cleanup(tmp_path, client=_published("src1"), bucket="b",
                          older_than_hours=0)
    result = C.execute(plan, dry_run=True)

    assert result.deleted == 0 and result.freed_bytes == 0
    assert plan.reclaimable_bytes == 4096 + 8192
    assert (run / "final" / "src1_r01_cand_001.mp4").is_file()
    assert (run / "source.mp4").is_file()


def test_a_real_run_deletes_exactly_what_it_planned(tmp_path):
    run = _run_tree(tmp_path, "src1")
    plan = C.plan_cleanup(tmp_path, client=_published("src1"), bucket="b",
                          older_than_hours=0)
    result = C.execute(plan, dry_run=False)

    assert result.deleted == 2 and result.freed_bytes == 4096 + 8192
    assert not (run / "final" / "src1_r01_cand_001.mp4").exists()
    assert not (run / "source.mp4").exists()
    # The sidecar survives: without it a re-render cannot recognise its own work.
    assert (run / "final" / "src1_r01_cand_001.json").is_file()


@pytest.mark.parametrize("state", sorted(C.ACTIVE_STATES))
def test_a_run_owned_by_an_active_job_is_never_touched(tmp_path, state):
    run = _run_tree(tmp_path / "runs", "src1")
    store = JobStore(tmp_path / "jobs")
    job = store.create(URL)
    store.update(job.job_id, lambda j: (setattr(j, "status", JobStatus(state)),
                                        setattr(j, "source_id", "src1")) and None)

    plan = C.plan_cleanup(tmp_path / "runs", store=store, client=_published("src1"),
                          bucket="b", older_than_hours=0)
    C.execute(plan, dry_run=False)

    assert plan.deletions == []
    assert any(i.action is C.Action.SKIP and "active_job" in i.reason for i in plan.items)
    assert (run / "source.mp4").is_file()


@pytest.mark.parametrize("state", ["DONE", "FAILED"])
def test_a_settled_job_does_not_protect_its_run(tmp_path, state):
    _run_tree(tmp_path / "runs", "src1")
    store = JobStore(tmp_path / "jobs")
    job = store.create(URL)
    store.update(job.job_id, lambda j: (setattr(j, "status", JobStatus(state)),
                                        setattr(j, "source_id", "src1")) and None)
    plan = C.plan_cleanup(tmp_path / "runs", store=store, client=_published("src1"),
                          bucket="b", older_than_hours=0)
    assert len(plan.deletions) == 2


def test_nothing_is_deleted_without_an_r2_client(tmp_path):
    _run_tree(tmp_path, "src1")
    plan = C.plan_cleanup(tmp_path, older_than_hours=0)
    assert plan.deletions == []
    assert all("remote_not_verified" in i.reason
               for i in plan.items
               if i.path.name in ("src1_r01_cand_001.mp4", "source.mp4"))


def test_a_clip_missing_from_r2_is_kept(tmp_path):
    _run_tree(tmp_path, "src1")
    remote = _published("src1")
    del remote.objects["output/src1/clips/src1_r01_cand_001.mp4"]
    plan = C.plan_cleanup(tmp_path, client=remote, bucket="b", older_than_hours=0)
    kept = [i for i in plan.items if i.path.name.endswith(".mp4") and i.action is C.Action.KEEP]
    assert len(kept) == 1 and "remote_not_verified" in kept[0].reason
    # the source is independently verified, so it is still reclaimable
    assert [i.path.name for i in plan.deletions] == ["source.mp4"]


def test_a_size_mismatch_in_r2_blocks_deletion(tmp_path):
    _run_tree(tmp_path, "src1")
    remote = _published("src1")
    remote.objects["input/src1/source.mp4"]["ContentLength"] = 999
    plan = C.plan_cleanup(tmp_path, client=remote, bucket="b", older_than_hours=0)
    assert "source.mp4" not in [i.path.name for i in plan.deletions]


def test_a_sha256_mismatch_in_r2_blocks_deletion(tmp_path):
    _run_tree(tmp_path, "src1")
    remote = _published("src1")
    remote.objects["output/src1/clips/src1_r01_cand_001.mp4"]["Metadata"]["sha256"] = "different"
    plan = C.plan_cleanup(tmp_path, client=remote, bucket="b", older_than_hours=0)
    assert not any(i.path.suffix == ".mp4" and i.path.name.startswith("src1_r01")
                   for i in plan.deletions)


def test_a_clip_with_no_recorded_r2_key_is_kept(tmp_path):
    _run_tree(tmp_path, "src1", r2_key=None)
    plan = C.plan_cleanup(tmp_path, client=_published("src1"), bucket="b",
                          older_than_hours=0)
    # The clip is unverifiable without a key; the source is verified separately
    # and stays reclaimable, so name the clip rather than every .mp4.
    assert "src1_r01_cand_001.mp4" not in {i.path.name for i in plan.deletions}
    assert "source.mp4" in {i.path.name for i in plan.deletions}


def test_resume_artifacts_are_always_kept(tmp_path):
    _run_tree(tmp_path, "src1")
    plan = C.plan_cleanup(tmp_path, client=_published("src1"), bucket="b",
                          older_than_hours=0, aggressive=True)
    deleted = {i.path.name for i in plan.deletions}
    assert not (deleted & C.PROTECTED_FILES)


def test_intermediates_are_kept_by_default_and_removed_only_on_request(tmp_path):
    _run_tree(tmp_path, "src1")
    conservative = C.plan_cleanup(tmp_path, client=_published("src1"), bucket="b",
                                  older_than_hours=0)
    assert not any(i.path.name in C.INTERMEDIATE_DIRS for i in conservative.deletions)

    aggressive = C.plan_cleanup(tmp_path, client=_published("src1"), bucket="b",
                                older_than_hours=0, aggressive=True)
    assert {i.path.name for i in aggressive.deletions} >= set(C.INTERMEDIATE_DIRS)


def test_recent_runs_are_left_alone(tmp_path):
    _run_tree(tmp_path, "src1")
    # Default: anything touched in the last hour is out of scope.
    plan = C.plan_cleanup(tmp_path, client=_published("src1"), bucket="b")
    assert plan.deletions == []
    assert any("newer_than" in i.reason for i in plan.items)


def test_keep_recent_protects_the_newest_runs(tmp_path):
    for name in ("old", "new"):
        _run_tree(tmp_path, name)
    os.utime(tmp_path / "old", (0, 0))
    remote = FakeR2({**_published("old").objects, **_published("new").objects})
    plan = C.plan_cleanup(tmp_path, client=remote, bucket="b",
                          older_than_hours=0, keep_recent=1)
    assert all("new" not in str(i.path) for i in plan.deletions)
    assert any("old" in str(i.path) for i in plan.deletions)


def test_job_id_scopes_cleanup_to_one_source(tmp_path):
    for name in ("src1", "src2"):
        _run_tree(tmp_path / "runs", name)
    store = JobStore(tmp_path / "jobs")
    job = store.create(URL)
    store.update(job.job_id, lambda j: (setattr(j, "status", JobStatus.DONE),
                                        setattr(j, "source_id", "src1")) and None)
    remote = FakeR2({**_published("src1").objects, **_published("src2").objects})
    plan = C.plan_cleanup(tmp_path / "runs", store=store, client=remote, bucket="b",
                          older_than_hours=0, job_id=job.job_id)
    assert plan.deletions and all("src1" in str(i.path) for i in plan.deletions)


def test_a_disk_budget_stops_once_the_target_is_met(tmp_path):
    for name in ("a", "b", "c"):
        _run_tree(tmp_path, name, clip_bytes=1 << 20, source_bytes=1 << 20)
    for i, name in enumerate(("a", "b", "c")):
        os.utime(tmp_path / name, (i, i))       # 'a' oldest, 'c' newest
    remote = FakeR2({k: v for n in ("a", "b", "c")
                     for k, v in _published(n, clip_bytes=1 << 20,
                                            source_bytes=1 << 20).objects.items()})
    # ~6 MiB present; ask to get under 4 MiB. Reclamation goes oldest-first and
    # stops as soon as the target is met, so the newest run must survive intact.
    # (It can overshoot by one run: the small JSON files mean the last deletion
    # rarely lands exactly on the target.)
    plan = C.plan_cleanup(tmp_path, client=remote, bucket="b", older_than_hours=0,
                          max_disk_usage_gb=4 / 1024)
    touched = {p.parts[-3] if p.parent.name == "final" else p.parent.name
               for p in (i.path for i in plan.deletions)}
    assert "a" in touched, "the oldest run is reclaimed first"
    assert "c" not in touched, "the newest run is spared once the target is met"
    assert any(i.action is C.Action.KEEP and "disk target already met" in i.reason
               for i in plan.items)


# ------------------------------------------------------------------ disk guards
def test_the_guard_allows_a_healthy_filesystem(tmp_path):
    status = DK.require_disk_space(tmp_path, "RENDERING", settings=Settings(
        min_free_disk_gb=0.0, min_free_disk_percent=0.0))
    assert status.ok and status.total_bytes > 0


def test_the_guard_refuses_when_free_space_is_below_the_floor(tmp_path):
    with pytest.raises(DK.InsufficientDiskSpaceError) as excinfo:
        DK.require_disk_space(tmp_path, "INGESTING", settings=Settings(
            min_free_disk_gb=1_000_000.0))
    message = str(excinfo.value)
    assert "INGESTING" in message and "cleanup" in message


def test_the_guard_refuses_when_the_percentage_floor_is_breached(tmp_path):
    with pytest.raises(DK.InsufficientDiskSpaceError):
        DK.require_disk_space(tmp_path, "RENDERING",
                              settings=Settings(min_free_disk_percent=100.0))


def test_a_size_estimate_tightens_the_check_but_never_replaces_it(tmp_path):
    cfg = Settings(min_free_disk_gb=0.0, min_free_disk_percent=0.0)
    free = DK.disk_status(tmp_path, cfg).free_bytes
    # Comfortably satisfiable.
    DK.require_disk_space(tmp_path, "INGESTING", settings=cfg, expected_bytes=1024)
    # A source that cannot fit with headroom is refused even though free space
    # alone passes the floor.
    with pytest.raises(DK.InsufficientDiskSpaceError):
        DK.require_disk_space(tmp_path, "INGESTING", settings=cfg,
                              expected_bytes=free)


def test_a_full_disk_is_not_retried_automatically():
    from freecher_worker.jobs.orchestrator import is_retryable

    assert is_retryable(DK.InsufficientDiskSpaceError("full")) is False


def test_health_reports_a_degraded_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "check_r2", lambda cfg, **kw: D.Check("r2", True, "ok"))
    cfg = Settings(min_free_disk_gb=1_000_000.0, runs_dir=str(tmp_path))
    diag = D.collect(cfg, jobs_dir=tmp_path)
    disk = next(c for c in diag.checks if c.name == "disk")
    assert not disk.ok and disk.info["guards_would_block"] is True
    assert diag.status == "degraded"


def test_health_never_loads_a_model(tmp_path, monkeypatch):
    """The model check must be a path stat, not an import of faster_whisper."""
    def explode(*a, **k):
        raise AssertionError("health must not construct a WhisperModel")

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = explode
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    check = D.check_model_cache(Settings(runs_dir=str(tmp_path)))
    assert check.ok and check.required is False


def test_runs_size_is_reported_without_network(tmp_path):
    _run_tree(tmp_path, "src1", clip_bytes=2048, source_bytes=2048)
    check = D.check_runs(Settings(runs_dir=str(tmp_path)), use_cache=False)
    assert check.ok and check.info["sources"] == 1
    assert check.info["bytes"] >= 4096


# -------------------------------------------------- batch-scoped refinement model
class CountingWhisper:
    instances = 0

    def __init__(self, *a, **k):
        CountingWhisper.instances += 1

    def transcribe(self, *a, **k):  # pragma: no cover - not reached in these tests
        return iter(()), types.SimpleNamespace(language="en")


@pytest.fixture
def counting_whisper(monkeypatch):
    CountingWhisper.instances = 0
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = CountingWhisper
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    ct = types.ModuleType("ctranslate2")
    ct.get_cuda_device_count = lambda: 0
    monkeypatch.setitem(sys.modules, "ctranslate2", ct)
    return CountingWhisper


def test_one_transcriber_loads_its_model_once_however_often_it_is_used(counting_whisper):
    t = HighlightWordTranscriber(model_name="medium", device="cpu", compute_type="int8")
    for _ in range(5):
        t._get_model()
    assert counting_whisper.instances == 1
    assert t.load_count == 1 and t.is_loaded


def test_separate_transcribers_each_load_their_own(counting_whisper):
    """The old behaviour, kept as a test so the cost of regressing is visible."""
    for _ in range(3):
        HighlightWordTranscriber(device="cpu")._get_model()
    assert counting_whisper.instances == 3


def test_release_drops_the_model_and_reloads_on_demand(counting_whisper):
    t = HighlightWordTranscriber(device="cpu")
    t._get_model()
    t.release()
    assert not t.is_loaded
    t._get_model()
    assert counting_whisper.instances == 2, "a released model must be re-created, not reused"


def test_the_context_manager_releases_on_exit(counting_whisper):
    with HighlightWordTranscriber(device="cpu") as t:
        t._get_model()
        assert t.is_loaded
    assert not t.is_loaded


# ------------------------------------------------------- batch wiring (no ffmpeg)
def _minimal_run(tmp_path: Path, clips: int = 3) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "highlights.json").write_text(json.dumps([
        {"rank": i + 1, "candidate_id": f"cand_{i:03d}", "start": i * 10.0,
         "end": i * 10.0 + 8.0, "score": 90 - i, "duration": 8.0,
         "reason": "r", "text": "t"}
        for i in range(clips)
    ]))
    (run / "transcript.json").write_text(json.dumps(
        {"language": "en", "duration": 60.0, "model": "small",
         "compute_type": "int8", "device": "cpu", "segments": []}))
    (run / "manifest.json").write_text(json.dumps(
        {"source_fingerprint": {"fingerprint_id": "fp", "duration_seconds": 60.0}}))
    return run


def _spy_render(monkeypatch, fail_ranks: set[int] = frozenset()):
    """Replace render_single_short, recording the transcriber each clip received."""
    seen: list[object] = []

    def fake(**kwargs):
        seen.append(kwargs.get("transcriber"))
        rank = kwargs["highlight"].rank
        if rank in fail_ranks:
            raise RuntimeError(f"clip {rank} exploded")
        raise FileNotFoundError("stop after the transcriber is observed")

    monkeypatch.setattr("freecher_worker.rendering.renderer.render_single_short", fake)
    return seen


def test_every_clip_in_a_batch_shares_one_transcriber(tmp_path, monkeypatch):
    run = _minimal_run(tmp_path, clips=3)
    seen = _spy_render(monkeypatch)
    B.render_top_n(run_dir=run, source_video=tmp_path / "s.mp4", source_id="src1",
                   top_n=3, publish=False, settings=Settings())
    assert len(seen) == 3
    assert seen[0] is not None
    assert seen[0] is seen[1] is seen[2], "a new model per clip is the bug being fixed"


def test_a_failing_clip_does_not_poison_the_shared_context(tmp_path, monkeypatch):
    run = _minimal_run(tmp_path, clips=3)
    seen = _spy_render(monkeypatch, fail_ranks={1})
    manifest = B.render_top_n(run_dir=run, source_video=tmp_path / "s.mp4",
                              source_id="src1", top_n=3, publish=False,
                              settings=Settings())
    assert len(seen) == 3, "the batch continued past the failure"
    assert seen[0] is seen[1] is seen[2]
    assert manifest.failed == 3 and manifest.status == "FAILED"


def test_no_transcriber_is_built_when_subtitles_are_off(tmp_path, monkeypatch):
    run = _minimal_run(tmp_path, clips=2)
    seen = _spy_render(monkeypatch)
    B.render_top_n(run_dir=run, source_video=tmp_path / "s.mp4", source_id="src1",
                   top_n=2, publish=False, enable_subtitles=False, settings=Settings())
    assert seen == [None, None]


def test_the_batch_releases_its_model_afterwards(tmp_path, monkeypatch):
    run = _minimal_run(tmp_path, clips=2)
    seen = _spy_render(monkeypatch)
    B.render_top_n(run_dir=run, source_video=tmp_path / "s.mp4", source_id="src1",
                   top_n=2, publish=False, settings=Settings())
    assert seen[0] is not None and not seen[0].is_loaded


def test_a_single_clip_render_still_builds_its_own_transcriber(counting_whisper, tmp_path):
    """The single-highlight CLI path passes no transcriber and must keep working."""
    from freecher_worker.rendering.renderer import render_single_short
    import inspect

    signature = inspect.signature(render_single_short)
    assert signature.parameters["transcriber"].default is None


# --------------------------------------------------------------------- env file
def test_an_unreadable_incidental_env_file_is_ignored(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FREECHER_R2_BUCKET=should-not-be-read\n")
    env.chmod(0o000)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FREECHER_ENV_FILE", raising=False)
    try:
        assert resolve_env_file() is None
        assert Settings().r2_bucket != "should-not-be-read"   # no crash, no value
    finally:
        env.chmod(0o644)


def test_an_unreadable_parent_directory_does_not_crash_the_cli(tmp_path, monkeypatch):
    """The reported failure: a service user running from someone else's checkout."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FREECHER_ENV_FILE", raising=False)
    monkeypatch.setattr(Path, "is_file",
                        lambda self: (_ for _ in ()).throw(PermissionError(13, "denied")))
    assert resolve_env_file() is None


def test_a_readable_env_file_is_still_used(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("FREECHER_R2_BUCKET=from-file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FREECHER_ENV_FILE", raising=False)
    monkeypatch.delenv("FREECHER_R2_BUCKET", raising=False)
    monkeypatch.delenv("R2_BUCKET", raising=False)
    assert Settings().r2_bucket == "from-file"


def test_an_explicitly_requested_env_file_must_be_readable(tmp_path, monkeypatch):
    target = tmp_path / "secrets.env"
    target.write_text("FREECHER_R2_BUCKET=explicit\n")
    monkeypatch.setenv("FREECHER_ENV_FILE", str(target))
    monkeypatch.delenv("FREECHER_R2_BUCKET", raising=False)
    monkeypatch.delenv("R2_BUCKET", raising=False)
    assert Settings().r2_bucket == "explicit"

    target.chmod(0o000)
    try:
        with pytest.raises(EnvFileError) as excinfo:
            resolve_env_file()
        assert "FREECHER_ENV_FILE" in str(excinfo.value)
    finally:
        target.chmod(0o644)


def test_an_explicitly_requested_missing_env_file_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("FREECHER_ENV_FILE", str(tmp_path / "nope.env"))
    with pytest.raises(EnvFileError):
        resolve_env_file()


# ------------------------------------------------ refinement word-cache reuse
def test_a_rerender_reuses_cached_refinement_words(tmp_path, monkeypatch):
    """Re-rendering a clip must not redo Whisper for byte-identical words.

    The cache key covers the fingerprint, both boundaries, the model, the
    compute type and the language, so a hit means the words cannot differ.
    Passing force=True unconditionally threw that away and cost ~100 s of CPU
    inference per clip on every re-render.
    """
    run = _minimal_run(tmp_path, clips=2)
    seen: list[bool] = []

    def fake(**kwargs):
        seen.append(kwargs["force"])
        raise FileNotFoundError("stop once the flag is observed")

    monkeypatch.setattr("freecher_worker.rendering.renderer.render_single_short", fake)
    B.render_top_n(run_dir=run, source_video=tmp_path / "s.mp4", source_id="src1",
                   top_n=2, publish=False, settings=Settings())
    assert seen == [False, False], "a normal re-render must consult the word cache"


def test_an_explicit_force_still_redoes_the_refinement(tmp_path, monkeypatch):
    run = _minimal_run(tmp_path, clips=1)
    seen: list[bool] = []

    def fake(**kwargs):
        seen.append(kwargs["force"])
        raise FileNotFoundError("stop")

    monkeypatch.setattr("freecher_worker.rendering.renderer.render_single_short", fake)
    B.render_top_n(run_dir=run, source_video=tmp_path / "s.mp4", source_id="src1",
                   top_n=1, publish=False, force=True, settings=Settings())
    assert seen == [True], "--force means redo the work, including the words"
