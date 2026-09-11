"""Disk lifecycle for run artifacts.

`runs/<source_id>/` grows without bound: seven short test jobs on the Oracle host
already held 599 MB, and a single multi-hour VOD would dwarf that. Nothing in the
pipeline ever deletes anything, because until now nothing could tell which files
were safe to lose.

This module answers that question, and only that question. Artifacts fall into
three classes:

  DURABLE / REMOTE   the published clip in R2, the transcript / candidates /
                     highlights / manifest objects in R2, and the job records.
                     Never deleted here; the job store is not touched at all.

  REGENERABLE LOCAL  the downloaded source, the local copy of a published clip,
                     and the small intermediates (ASS files, word caches, crop
                     paths). Deletable -- but a local file is only "regenerable"
                     once the thing it can be regenerated *from* is verified to
                     exist remotely.

  SHARED CACHE       model weights and detector ONNX under ~/.cache. Expensive,
                     not per-job, and never a cleanup target.

Two guards decide every deletion, and both must pass:

  1. the run's source must not belong to a job in a non-terminal state;
  2. the remote replacement must be verified by a live HEAD -- matching size, and
     matching sha256 where the uploader recorded one.

Everything is planned before anything is removed, so `--dry-run` shows exactly
what a real run would do.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger("freecher_worker")

GIB = 1 << 30

#: Job states in which a run directory is being read or written right now.
#: Cleaning one of these races the worker, so they are skipped outright.
ACTIVE_STATES = frozenset({
    "QUEUED", "INGESTING", "TRANSCRIBING", "AWAITING_TRANSCRIPT",
    "DISCOVERING", "RANKING", "RENDERING", "UPLOADING",
})

#: Small, cheap to keep, and expensive or annoying to regenerate:
#: `words/` is the refinement-ASR cache (minutes of Whisper per clip),
#: `final/*.json` are the sidecars that make clip reuse work at all.
#: Removed only with `aggressive=True`.
INTERMEDIATE_DIRS = ("subtitles", "words", "crop_paths")

#: Never deleted: the artifacts that let a resumed job skip completed stages
#: without re-downloading anything. All are a few KB.
PROTECTED_FILES = frozenset({
    "transcript.json", "candidates.json", "highlights.json",
    "manifest.json", "clips.json",
})


class Action(str, Enum):
    DELETE = "DELETE"
    KEEP = "KEEP"
    SKIP = "SKIP"


@dataclass
class Item:
    action: Action
    path: Path
    bytes: int = 0
    reason: str = ""

    def log(self) -> None:
        if self.action is Action.DELETE:
            logger.info("DELETE %s (%s bytes)", self.path, f"{self.bytes:,}")
        else:
            logger.info("%s %s reason=%s", self.action.value, self.path, self.reason)


@dataclass
class Plan:
    items: list[Item] = field(default_factory=list)
    root: Optional[Path] = None

    @property
    def deletions(self) -> list[Item]:
        return [i for i in self.items if i.action is Action.DELETE]

    @property
    def reclaimable_bytes(self) -> int:
        return sum(i.bytes for i in self.deletions)

    def log(self) -> None:
        for item in self.items:
            item.log()


@dataclass
class Result:
    deleted: int = 0
    freed_bytes: int = 0
    failed: list[tuple[Path, str]] = field(default_factory=list)


# ------------------------------------------------------------------ remote checks
def _remote_matches(client: Any, bucket: str, key: str, size: int,
                    sha256: Optional[str] = None) -> tuple[bool, str]:
    """Verify the remote replacement really is there and really is this file."""
    if not key:
        return False, "no r2 key recorded"
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001 - absent or unreachable both fail closed
        return False, f"remote head failed ({type(exc).__name__})"
    remote_size = int(head.get("ContentLength") or 0)
    if size and remote_size != size:
        return False, f"remote size {remote_size} != local {size}"
    remote_sha = (head.get("Metadata") or {}).get("sha256")
    if sha256 and remote_sha and remote_sha != sha256:
        return False, "remote sha256 differs"
    return True, key


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------- job state
def _active_source_ids(store: Any) -> dict[str, str]:
    """source_id -> the state that makes it untouchable."""
    active: dict[str, str] = {}
    if store is None:
        return active
    for job in store.list():
        state = job.status.value if hasattr(job.status, "value") else str(job.status)
        if state in ACTIVE_STATES and job.source_id:
            active[job.source_id] = state
    return active


def _source_ids_for_job(store: Any, job_id: str) -> set[str]:
    job = store.get(job_id)
    return {job.source_id} if job.source_id else set()


# ---------------------------------------------------------------------- planning
def _dir_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _run_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted((d for d in root.iterdir()
                   if d.is_dir() and not d.name.startswith("_")),
                  key=lambda d: d.stat().st_mtime, reverse=True)


def plan_cleanup(
    root: Path | str,
    *,
    store: Any = None,
    client: Any = None,
    bucket: Optional[str] = None,
    older_than_hours: float = 1.0,
    keep_recent: int = 0,
    job_id: Optional[str] = None,
    max_disk_usage_gb: Optional[float] = None,
    aggressive: bool = False,
    verify_sha256: bool = False,
) -> Plan:
    """Decide what may be removed. Reads only; nothing is deleted here.

    `max_disk_usage_gb` caps the total size of the runs tree: directories are
    considered oldest-first and the plan stops as soon as the projected size is
    under the cap, so a routine cleanup removes the least useful data and no
    more.
    """
    root = Path(root)
    plan = Plan(root=root)
    now = time.time()
    cutoff = now - older_than_hours * 3600.0
    active = _active_source_ids(store)
    wanted = _source_ids_for_job(store, job_id) if (job_id and store) else None

    dirs = _run_dirs(root)
    protected_recent = {d for d in dirs[:keep_recent]} if keep_recent > 0 else set()

    budget_remaining: Optional[int] = None
    if max_disk_usage_gb is not None:
        current = sum(_dir_bytes(d) for d in dirs)
        budget_remaining = max(0, current - int(max_disk_usage_gb * GIB))
        logger.info("[cleanup] runs tree is %.2f GiB; target %.2f GiB; "
                    "%.2f GiB must be reclaimed",
                    current / GIB, max_disk_usage_gb, budget_remaining / GIB)
        # Oldest first: the least likely to be reused again.
        dirs = list(reversed(dirs))

    for run_dir in dirs:
        source_id = run_dir.name

        if wanted is not None and source_id not in wanted:
            continue
        if source_id in active:
            plan.items.append(Item(Action.SKIP, run_dir, reason=f"active_job ({active[source_id]})"))
            continue
        if run_dir in protected_recent:
            plan.items.append(Item(Action.KEEP, run_dir, reason="keep_recent"))
            continue
        if run_dir.stat().st_mtime > cutoff:
            plan.items.append(Item(Action.KEEP, run_dir, reason=f"newer_than {older_than_hours}h"))
            continue
        if budget_remaining is not None and budget_remaining <= 0:
            plan.items.append(Item(Action.KEEP, run_dir, reason="disk target already met"))
            continue

        before = len(plan.items)
        _plan_one_run(plan, run_dir, source_id, client, bucket, aggressive, verify_sha256)
        if budget_remaining is not None:
            budget_remaining -= sum(i.bytes for i in plan.items[before:]
                                   if i.action is Action.DELETE)

    return plan


def _plan_one_run(plan: Plan, run_dir: Path, source_id: str, client: Any,
                  bucket: Optional[str], aggressive: bool, verify_sha256: bool) -> None:
    # --- published clips: deletable once the R2 object is verified -------------
    final_dir = run_dir / "final"
    if final_dir.is_dir():
        for clip in sorted(final_dir.glob("*.mp4")):
            sidecar = clip.with_suffix(".json")
            record: dict[str, Any] = {}
            if sidecar.is_file():
                try:
                    record = json.loads(sidecar.read_text())
                except (json.JSONDecodeError, OSError):
                    record = {}
            size = clip.stat().st_size
            if not record.get("r2_key"):
                plan.items.append(Item(Action.KEEP, clip, size,
                                       "remote_not_verified (no sidecar r2_key)"))
                continue
            if client is None or not bucket:
                plan.items.append(Item(Action.KEEP, clip, size,
                                       "remote_not_verified (no R2 client)"))
                continue
            local_sha = record.get("sha256")
            if verify_sha256:
                local_sha = _sha256(clip)
            ok, why = _remote_matches(client, bucket, record["r2_key"],
                                      int(record.get("bytes") or size), local_sha)
            if ok:
                plan.items.append(Item(Action.DELETE, clip, size,
                                       f"published at {record['r2_key']}"))
            else:
                plan.items.append(Item(Action.KEEP, clip, size, f"remote_not_verified ({why})"))

    # --- the downloaded source: deletable once R2 still holds the original -----
    source = run_dir / "source.mp4"
    if source.is_file():
        size = source.stat().st_size
        key = f"input/{source_id}/source.mp4"
        if client is None or not bucket:
            plan.items.append(Item(Action.KEEP, source, size,
                                   "remote_not_verified (no R2 client)"))
        else:
            ok, why = _remote_matches(client, bucket, key, size)
            if ok:
                plan.items.append(Item(Action.DELETE, source, size, f"re-fetchable from {key}"))
            else:
                plan.items.append(Item(Action.KEEP, source, size, f"remote_not_verified ({why})"))

    # --- small intermediates: only on request ---------------------------------
    for name in INTERMEDIATE_DIRS:
        target = run_dir / name
        if not target.is_dir():
            continue
        size = _dir_bytes(target)
        if aggressive:
            plan.items.append(Item(Action.DELETE, target, size, "regenerable intermediate"))
        else:
            plan.items.append(Item(Action.KEEP, target, size,
                                   "cheap to keep, expensive to regenerate"))

    for name in sorted(PROTECTED_FILES):
        path = run_dir / name
        if path.is_file():
            plan.items.append(Item(Action.KEEP, path, path.stat().st_size,
                                   "needed to resume without re-downloading"))


# --------------------------------------------------------------------- execution
def execute(plan: Plan, *, dry_run: bool = True) -> Result:
    """Carry out a plan. With dry_run the log is identical but nothing is removed."""
    result = Result()
    for item in plan.items:
        item.log()
        if item.action is not Action.DELETE or dry_run:
            continue
        try:
            if item.path.is_dir():
                shutil.rmtree(item.path)
            else:
                item.path.unlink()
            result.deleted += 1
            result.freed_bytes += item.bytes
        except OSError as exc:
            result.failed.append((item.path, str(exc)))
            logger.error("FAILED to delete %s: %s", item.path, exc)
    if dry_run:
        logger.info("[cleanup] dry run: %d item(s), %.2f GiB reclaimable",
                    len(plan.deletions), plan.reclaimable_bytes / GIB)
    else:
        logger.info("[cleanup] removed %d item(s), %.2f GiB freed",
                    result.deleted, result.freed_bytes / GIB)
    return result


def cleanup_completed_job(job: Any, *, settings: Any = None, store: Any = None,
                          client: Any = None) -> Result:
    """Post-DONE cleanup for one job, used when cleanup_after_done is enabled.

    Deliberately not called on worker startup or on a schedule: automatic
    destruction that a process performs merely because it started is exactly the
    behaviour this module is meant to avoid.
    """
    from freecher_worker.config import get_settings
    from freecher_worker.jobs.store import DEFAULT_ROOT

    cfg = settings or get_settings()
    root = Path(str(getattr(cfg, "runs_dir", None) or DEFAULT_ROOT.parent))
    if client is None:
        from freecher_worker.ingest.r2 import build_r2_client

        client = build_r2_client(cfg.r2_endpoint, cfg.r2_access_key_id,
                                 cfg.r2_secret_access_key)
    plan = plan_cleanup(root, store=store, client=client, bucket=cfg.r2_bucket,
                        older_than_hours=0.0, job_id=job.job_id)
    return execute(plan, dry_run=False)
