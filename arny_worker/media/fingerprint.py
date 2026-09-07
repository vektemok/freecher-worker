"""Source media fingerprinting for reliable cache validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from pydantic import BaseModel, Field


class SourceFingerprint(BaseModel):
    """Deterministic fingerprint identifying a specific media file state."""

    path: str = Field(description="Resolved absolute path of the source file")
    file_size: int = Field(description="File size in bytes")
    mtime_ns: int = Field(description="Modification timestamp in nanoseconds")
    duration_seconds: float = Field(description="Media duration reported by ffprobe")
    content_hash: str = Field(description="Lightweight content hash (head/mid/tail chunks)")
    fingerprint_id: str = Field(description="Short unique fingerprint hash")


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
    )
