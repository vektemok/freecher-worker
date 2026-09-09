"""Data models for the streaming ingest subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# S3/R2 multipart constraints. Every part except the last must be at least
# 5 MiB and — on R2 specifically — every part except the last must be the
# same size, which is why the reader always fills a full chunk before upload.
MIN_PART_SIZE = 5 * 1024 * 1024
MAX_PARTS = 10_000
DEFAULT_PART_SIZE = 16 * 1024 * 1024

QUALITY_BEST = "best"
QUALITY_PROGRESSIVE = "progressive"
AVAILABLE_QUALITY_MODES = (QUALITY_BEST, QUALITY_PROGRESSIVE)


# Protocols yt-dlp can hand over as one contiguous byte range. Anything else
# (HLS/DASH segment lists, as Twitch and most live platforms serve) has to pass
# through ffmpeg to become a single playable container.
DIRECT_PROTOCOLS = frozenset({"http", "https", "ftp", "ftps", "file"})


@dataclass
class VideoInfo:
    """Metadata resolved from the source site before the transfer starts."""

    video_id: str
    title: str
    duration_seconds: Optional[float]
    uploader: Optional[str]
    width: Optional[int] = None
    height: Optional[int] = None
    filesize_approx: Optional[int] = None
    webpage_url: Optional[str] = None
    extractor: Optional[str] = None
    protocol: Optional[str] = None
    acodec: Optional[str] = None
    requested_format_ids: list[str] = field(default_factory=list)
    is_live: bool = False

    @property
    def duration_label(self) -> str:
        if self.duration_seconds is None:
            return "unknown"
        total = int(self.duration_seconds)
        return f"{total // 3600:d}:{(total % 3600) // 60:02d}:{total % 60:02d}"

    @property
    def needs_remux(self) -> bool:
        """True when the bytes cannot be piped through untouched.

        Two tracks to merge, or a segmented protocol: either way ffmpeg has to
        assemble the stream before it can go out as one mp4 object.
        """
        if len(self.requested_format_ids) > 1:
            return True
        if self.protocol is None:
            return False
        # A merged selection shows up as "https+https".
        return any(part not in DIRECT_PROTOCOLS for part in self.protocol.split("+"))

    @property
    def needs_adts_to_asc(self) -> bool:
        """True when AAC audio arrives as ADTS frames and mp4 needs ASC.

        HLS carries AAC inside MPEG-TS as ADTS, which the mp4 muxer refuses.
        yt-dlp applies this filter itself only on its own mp4 branch, and a
        stdout target never reaches that branch, so we have to ask for it.
        """
        if not self.protocol or "m3u8" not in self.protocol:
            return False
        if not self.acodec or self.acodec == "none":
            return True
        return self.acodec.split(".")[0] in {"aac", "mp4a"}


@dataclass
class UploadProgress:
    """Snapshot handed to the progress callback after each completed part."""

    part_number: int
    part_bytes: int
    uploaded_bytes: int
    elapsed_seconds: float
    expected_bytes: Optional[int] = None

    @property
    def megabytes_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.uploaded_bytes / self.elapsed_seconds / 1024 / 1024

    @property
    def percent(self) -> Optional[float]:
        if not self.expected_bytes:
            return None
        return min(100.0, self.uploaded_bytes / self.expected_bytes * 100.0)


@dataclass
class IngestResult:
    """Outcome of a completed YouTube -> R2 transfer."""

    bucket: str
    key: str
    uploaded_bytes: int
    part_count: int
    elapsed_seconds: float
    quality: str
    format_selector: str
    remuxed: bool = False
    video: Optional[VideoInfo] = None
    public_url: Optional[str] = None
    etag: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    @property
    def megabytes(self) -> float:
        return self.uploaded_bytes / 1024 / 1024

    @property
    def average_megabytes_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.megabytes / self.elapsed_seconds
