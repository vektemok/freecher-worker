"""Source-scoped ingest ownership and lease locking.

Ensures that multiple concurrent jobs requesting the same source_id do not
run duplicate ingest (yt-dlp/ffmpeg -> R2) pipelines. Exactly one job holds
the lease, while concurrent waiters either reuse the completed artifacts
(source.mp4 and audio.m4a) or take over if the owner dies.

Topology: Single-host only.
Like `freecher_worker.jobs.store._LOCK`, this lock is backed by local filesystem
`fcntl.flock` and local PID validation (`ps -p <pid> -o lstart=`). It is
explicitly scoped to Freecher's supported single-host deployment model (one API
and worker co-located on one host). It is NOT a distributed cross-host lock.
If multi-host workers are deployed in the future, a shared coordinator
(e.g., Postgres advisory locks or Redis Redlock) would be required.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("freecher_worker")

DEFAULT_LOCKS_DIR = Path("runs") / "_locks"
DEFAULT_STALE_HEARTBEAT_SECONDS = 120.0
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 10.0


def get_process_start_time(pid: int) -> Optional[str]:
    """Return process start time string via ps, guarding against PID reuse."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2.0,
            check=False,
        )
        if result.returncode == 0:
            out = result.stdout.strip()
            return out if out else None
    except Exception:
        pass
    return None


def is_pid_alive(pid: int, expected_start_time: Optional[str] = None) -> bool:
    """Check if process with pid is alive and matches expected start time."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False

    if expected_start_time is not None:
        current_start = get_process_start_time(pid)
        if current_start is None or current_start != expected_start_time:
            return False
    return True


class SourceIngestLock:
    """A durable, source_id-scoped lock backed by a local file and kernel flock.

    Single-host only: scoped to Freecher's current single-host architecture.
    The lock is owned by a single process and job. Kernel flock ensures atomic
    acquisition and instant release on process crash/kill. Heartbeating metadata
    in JSON inside the lock file tracks the owner and detects stale or stalled jobs.
    """

    def __init__(
        self,
        source_id: str,
        job_id: str,
        locks_dir: Path | str = DEFAULT_LOCKS_DIR,
        stale_heartbeat_seconds: float = DEFAULT_STALE_HEARTBEAT_SECONDS,
    ) -> None:
        self.source_id = source_id
        self.job_id = job_id
        self.locks_dir = Path(locks_dir)
        self.stale_heartbeat_seconds = stale_heartbeat_seconds
        self.path = self.locks_dir / f"{source_id}.lock"

        self._fd: Optional[int] = None
        self._heartbeat_stop: Optional[threading.Event] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

    def acquire(self, force_if_stale: bool = False) -> bool:
        """Attempt to atomically acquire the lock without blocking.

        If force_if_stale is True and the lock is held by a stale/deadlocked owner,
        reclaims the lock. Kernel flock guarantees only one contender succeeds.

        Returns True if acquired, False if currently held by an active process.
        """
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as exc:
            logger.warning("[lock:%s] failed to open lock file %s: %s", self.source_id, self.path, exc)
            return False

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            # Another process holds flock. Check if it is stale if force_if_stale requested.
            if force_if_stale and self.is_stale():
                meta = self.read_metadata() or {}
                stale_pid = meta.get("owner_pid")
                stale_start = meta.get("owner_lstart")
                if isinstance(stale_pid, int) and stale_pid > 0 and is_pid_alive(stale_pid, stale_start):
                    logger.warning(
                        "[lock:%s] terminating deadlocked stale owner PID %d (job %s)",
                        self.source_id, stale_pid, meta.get("owner_job_id")
                    )
                    try:
                        os.kill(stale_pid, 9)  # SIGKILL
                    except OSError:
                        pass
                # Retry acquiring flock after terminating/reclaiming stale owner
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (BlockingIOError, OSError):
                    os.close(fd)
                    return False
            else:
                os.close(fd)
                return False

        # Flock acquired!
        self._fd = fd
        self._write_metadata(is_initial=True)
        logger.info(
            "[lock:%s] acquired source ingest lock for job %s (pid %d)",
            self.source_id, self.job_id, os.getpid()
        )
        return True

    def read_metadata(self) -> Optional[dict[str, Any]]:
        """Read the lock file metadata if readable."""
        if not self.path.is_file():
            return None
        try:
            content = self.path.read_text(encoding="utf-8").strip()
            if content:
                return json.loads(content)
        except Exception:
            pass
        return None

    def is_stale(self) -> bool:
        """Check if the current lock holder is stale (dead process or missing heartbeats)."""
        meta = self.read_metadata()
        if not meta:
            return True

        pid = meta.get("owner_pid")
        if not isinstance(pid, int) or pid <= 0:
            return True

        start_time = meta.get("owner_lstart")
        if not is_pid_alive(pid, start_time):
            logger.warning(
                "[lock:%s] owner process %d (job %s) is no longer alive",
                self.source_id, pid, meta.get("owner_job_id")
            )
            return True

        # Owner is alive; check heartbeat timestamp
        hb_str = meta.get("heartbeat_at") or meta.get("acquired_at")
        if hb_str:
            try:
                hb_time = datetime.fromisoformat(hb_str)
                now = datetime.now(timezone.utc)
                age = (now - hb_time).total_seconds()
                if age > self.stale_heartbeat_seconds:
                    logger.warning(
                        "[lock:%s] owner heartbeat is stale (age=%.1fs > limit=%.1fs)",
                        self.source_id, age, self.stale_heartbeat_seconds
                    )
                    return True
            except Exception:
                pass

        return False

    def heartbeat(self) -> None:
        """Update the heartbeat_at timestamp in the lock file."""
        if self._fd is None:
            return
        self._write_metadata(is_initial=False)

    def _write_metadata(self, is_initial: bool = False) -> None:
        if self._fd is None:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        acquired_at = now_iso
        if not is_initial:
            existing = self.read_metadata()
            if existing and "acquired_at" in existing:
                acquired_at = existing["acquired_at"]

        meta = {
            "source_id": self.source_id,
            "owner_job_id": self.job_id,
            "owner_pid": os.getpid(),
            "owner_lstart": get_process_start_time(os.getpid()),
            "acquired_at": acquired_at,
            "heartbeat_at": now_iso,
        }
        payload = json.dumps(meta, indent=2).encode("utf-8")
        try:
            os.lseek(self._fd, 0, os.SEEK_SET)
            os.write(self._fd, payload)
            os.ftruncate(self._fd, len(payload))
            os.fsync(self._fd)
        except OSError as exc:
            logger.warning("[lock:%s] failed writing heartbeat: %s", self.source_id, exc)

    def start_heartbeat(self, interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS) -> None:
        """Start a background thread periodically heartbeating while the lock is held."""
        if self._heartbeat_thread is not None:
            return
        self._heartbeat_stop = threading.Event()

        def _runner() -> None:
            assert self._heartbeat_stop is not None
            while not self._heartbeat_stop.wait(interval_seconds):
                try:
                    self.heartbeat()
                except Exception as exc:
                    logger.debug("[lock:%s] heartbeat error: %s", self.source_id, exc)

        self._heartbeat_thread = threading.Thread(
            target=_runner, name=f"lock-hb-{self.source_id}", daemon=True
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        """Stop the background heartbeat thread."""
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None
        self._heartbeat_stop = None

    def release(self) -> None:
        """Stop heartbeats, unlock flock, close fd, and remove lock file."""
        self.stop_heartbeat()
        if self._fd is not None:
            try:
                self.path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None
            logger.info(
                "[lock:%s] released source ingest lock for job %s",
                self.source_id, self.job_id
            )

    def __enter__(self) -> SourceIngestLock:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()
