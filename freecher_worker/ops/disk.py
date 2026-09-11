"""Free-space accounting and the guard that runs before expensive stages.

Ingest writes the whole source video; rendering writes one 1080x1920 MP4 per
clip plus ffmpeg intermediates. Both can consume several gigabytes, and both
fail late and messily when the filesystem fills -- a truncated MP4 that still
looks like a file, or an ffmpeg error whose text says nothing about disk. The
cheap fix is to ask before starting.

The authority is always an actual `statvfs` of the directory being written to.
Source metadata, where available, only adds a second, stricter check.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("freecher_worker")

GIB = 1 << 30

#: Ingest keeps the source in R2 and on disk, and rendering reads it while
#: writing clips, so a source needs room for itself several times over before
#: the job is finished. Applied only when a size estimate exists.
SOURCE_HEADROOM_FACTOR = 3.0


class InsufficientDiskSpaceError(RuntimeError):
    """Not enough free space to start a stage that would produce large files.

    Not retryable: the next attempt fails identically until an operator (or
    `freecher-worker cleanup`) frees space. The message says which.
    """


@dataclass(frozen=True)
class DiskStatus:
    path: str
    total_bytes: int
    free_bytes: int
    min_free_bytes: int
    min_free_percent: float

    @property
    def free_gib(self) -> float:
        return self.free_bytes / GIB

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB

    @property
    def free_percent(self) -> float:
        return (self.free_bytes / self.total_bytes * 100.0) if self.total_bytes else 0.0

    @property
    def ok(self) -> bool:
        return (self.free_bytes >= self.min_free_bytes
                and self.free_percent >= self.min_free_percent)

    @property
    def summary(self) -> str:
        return (f"{self.free_gib:.1f} GiB free of {self.total_gib:.1f} GiB "
                f"({self.free_percent:.1f}%), floor "
                f"{self.min_free_bytes / GIB:.1f} GiB / {self.min_free_percent:.1f}%")


def _existing_ancestor(path: Path) -> Path:
    """statvfs needs a path that exists; a run directory may not yet."""
    target = path.resolve()
    while not target.exists() and target != target.parent:
        target = target.parent
    return target


def disk_status(path: Path | str, settings: Any = None) -> DiskStatus:
    from freecher_worker.config import get_settings

    cfg = settings or get_settings()
    target = _existing_ancestor(Path(path))
    usage = shutil.disk_usage(target)
    return DiskStatus(
        path=str(target),
        total_bytes=usage.total,
        free_bytes=usage.free,
        min_free_bytes=int(float(getattr(cfg, "min_free_disk_gb", 5.0)) * GIB),
        min_free_percent=float(getattr(cfg, "min_free_disk_percent", 5.0)),
    )


def require_disk_space(path: Path | str, stage: str, *, settings: Any = None,
                       expected_bytes: Optional[int] = None) -> DiskStatus:
    """Refuse to start `stage` when the filesystem is too full. Returns the status.

    `expected_bytes` is an optional size estimate for what the stage will write
    (a source's reported file size, say). It is an *additional* constraint, never
    a substitute for the real free-space reading -- an estimate that happens to
    look fine does not make a full disk usable.
    """
    status = disk_status(path, settings)
    if not status.ok:
        raise InsufficientDiskSpaceError(
            f"refusing to start {stage}: {status.summary} at {status.path}. "
            f"Free space (`freecher-worker cleanup --dry-run` shows what is "
            f"reclaimable) or lower FREECHER_MIN_FREE_DISK_GB / "
            f"FREECHER_MIN_FREE_DISK_PERCENT."
        )

    if expected_bytes:
        needed = int(expected_bytes * SOURCE_HEADROOM_FACTOR)
        if status.free_bytes - status.min_free_bytes < needed:
            raise InsufficientDiskSpaceError(
                f"refusing to start {stage}: needs roughly "
                f"{needed / GIB:.1f} GiB above the safety floor for a "
                f"{expected_bytes / GIB:.1f} GiB source, but only "
                f"{(status.free_bytes - status.min_free_bytes) / GIB:.1f} GiB is "
                f"available at {status.path}."
            )

    logger.info("[disk] %s may start: %s", stage, status.summary)
    return status
