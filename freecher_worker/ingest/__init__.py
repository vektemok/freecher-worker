"""Streaming ingest: any yt-dlp source -> Cloudflare R2, nothing staged on disk."""

from freecher_worker.ingest.models import (
    AVAILABLE_QUALITY_MODES,
    DEFAULT_PART_SIZE,
    DIRECT_PROTOCOLS,
    MAX_PARTS,
    MIN_PART_SIZE,
    QUALITY_BEST,
    QUALITY_PROGRESSIVE,
    IngestResult,
    UploadProgress,
    VideoInfo,
)
from freecher_worker.ingest.r2 import (
    R2ConfigurationError,
    R2UploadError,
    StreamingMultipartUploader,
    build_r2_client,
    read_exact,
)
from freecher_worker.ingest.service import (
    DEFAULT_KEY_TEMPLATE,
    ingest_to_r2,
    object_exists,
    render_key,
    resolve_remux,
)
from freecher_worker.ingest.source import (
    SourceStreamError,
    build_format_selector,
    build_stream_command,
    open_source_stream,
    probe_source,
)

__all__ = [
    "AVAILABLE_QUALITY_MODES",
    "DEFAULT_KEY_TEMPLATE",
    "DEFAULT_PART_SIZE",
    "DIRECT_PROTOCOLS",
    "MAX_PARTS",
    "MIN_PART_SIZE",
    "QUALITY_BEST",
    "QUALITY_PROGRESSIVE",
    "IngestResult",
    "R2ConfigurationError",
    "R2UploadError",
    "SourceStreamError",
    "StreamingMultipartUploader",
    "UploadProgress",
    "VideoInfo",
    "build_format_selector",
    "build_r2_client",
    "build_stream_command",
    "ingest_to_r2",
    "object_exists",
    "open_source_stream",
    "probe_source",
    "read_exact",
    "render_key",
    "resolve_remux",
]
