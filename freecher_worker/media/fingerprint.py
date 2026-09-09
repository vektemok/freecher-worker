"""Source media fingerprinting for reliable cache validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


LOCAL_FILE_SOURCE = "local_file"
R2_TRANSCRIPT_SOURCE = "r2_transcript"


class SourceFingerprint(BaseModel):
    """Deterministic fingerprint identifying a specific source state.

    Two kinds, distinguished by `kind`. A `local_file` fingerprint identifies a
    video on disk by size, mtime and sampled content. An `r2_transcript`
    fingerprint identifies a run that has no local video at all: its identity
    is the remote one — source id, transcript object key and transcript hash.
    The remote fields are left unset for a local run and vice versa, so nothing
    here is ever a stand-in for a value that does not exist.
    """

    path: str = Field(description="Absolute path of the source file, or its s3:// URI for a remote run")
    file_size: int = Field(description="Size in bytes of the object this identifies")
    duration_seconds: float = Field(description="Media duration in seconds")
    content_hash: str = Field(description="Lightweight content hash, or the transcript hash for a remote run")
    fingerprint_id: str = Field(description="Short unique fingerprint hash")
    kind: str = Field(default=LOCAL_FILE_SOURCE, description="local_file or r2_transcript")
    # Local-only: meaningless without a file on disk, so absent for a remote run.
    mtime_ns: Optional[int] = Field(default=None, description="Modification time in nanoseconds (local runs only)")
    # Remote-only: the stable identity of an R2-backed run.
    source_id: Optional[str] = Field(default=None, description="Source id, i.e. the processing/{id}/ folder")
    bucket: Optional[str] = Field(default=None, description="Bucket holding the remote artifacts")
    transcript_key: Optional[str] = Field(default=None, description="Object key of the transcript this run reads")

    @property
    def has_local_video(self) -> bool:
        return self.kind == LOCAL_FILE_SOURCE


def compute_lightweight_content_hash(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute a deterministic hash using head, middle, and tail chunks of the file.

    Fast even for multi-gigabyte video files.
    """
    hasher = hashlib.sha256()
    size = file_path.stat().st_size

    if size <= chunk_size * 3:
        with open(file_path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    with open(file_path, "rb") as f:
        # Head chunk
        hasher.update(f.read(chunk_size))

        # Middle chunk
        mid_pos = (size // 2) - (chunk_size // 2)
        f.seek(mid_pos)
        hasher.update(f.read(chunk_size))

        # Tail chunk
        f.seek(size - chunk_size)
        hasher.update(f.read(chunk_size))

    return hasher.hexdigest()


def compute_source_fingerprint(video_path: Path | str, duration_seconds: float) -> SourceFingerprint:
    """Generate a deterministic SourceFingerprint for a video file."""
    path = Path(video_path).resolve()
    stat = path.stat()
    content_hash = compute_lightweight_content_hash(path)

    raw_signature = (
        f"{str(path)}|{stat.st_size}|{stat.st_mtime_ns}|{duration_seconds:.3f}|{content_hash}"
    )
    fp_id = hashlib.sha256(raw_signature.encode("utf-8")).hexdigest()[:16]

    return SourceFingerprint(
        path=str(path),
        file_size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        duration_seconds=round(duration_seconds, 3),
        content_hash=content_hash,
        fingerprint_id=fp_id,
        kind=LOCAL_FILE_SOURCE,
    )


def compute_r2_transcript_fingerprint(
    *,
    source_id: str,
    bucket: str,
    transcript_key: str,
    transcript_hash: str,
    duration_seconds: float,
    transcript_bytes: int,
) -> SourceFingerprint:
    """Identify an R2-backed run by what actually identifies it.

    There is no local video to hash, and inventing one would make the manifest
    lie. The stable identity of such a run is the source id, the transcript
    object it reads and that transcript's content hash — all three deterministic,
    so re-running over an unchanged transcript reproduces the same id.
    """
    raw_signature = f"{R2_TRANSCRIPT_SOURCE}|{source_id}|{bucket}|{transcript_key}|{transcript_hash}"
    fp_id = hashlib.sha256(raw_signature.encode("utf-8")).hexdigest()[:16]

    return SourceFingerprint(
        path=f"s3://{bucket}/{transcript_key}",
        file_size=transcript_bytes,
        duration_seconds=round(duration_seconds, 3),
        content_hash=transcript_hash,
        fingerprint_id=fp_id,
        kind=R2_TRANSCRIPT_SOURCE,
        source_id=source_id,
        bucket=bucket,
        transcript_key=transcript_key,
    )
