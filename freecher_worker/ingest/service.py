"""Glue between the yt-dlp source stream and the R2 multipart sink."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Sequence
from urllib.parse import quote

from freecher_worker.ingest.models import (
    DEFAULT_PART_SIZE,
    QUALITY_BEST,
    IngestResult,
    VideoInfo,
)
from freecher_worker.ingest.r2 import (
    ProgressCallback,
    R2UploadError,
    StreamingMultipartUploader,
)
from freecher_worker.ingest.source import (
    build_format_selector,
    open_source_stream,
    probe_source,
)

logger = logging.getLogger("freecher_worker")

DEFAULT_KEY_TEMPLATE = "input/{video_id}/source.mp4"

# S3 user metadata must be US-ASCII. Percent-encoding keeps a Cyrillic or emoji
# title recoverable instead of blanking it out, and leaves a plain ASCII title
# almost untouched. quote() escapes '%' itself, so decoding is unambiguous.
_METADATA_SAFE_CHARS = " ._-()[]@"


def render_key(
    template: str,
    video: Optional[VideoInfo],
    project_id: Optional[str] = None,
) -> str:
    """Expand {video_id}/{project_id}/{extractor}/{date} placeholders in a key."""
    video_id = video.video_id if video else "unknown"
    extractor = (video.extractor if video and video.extractor else "source").lower()
    return template.format(
        video_id=video_id,
        project_id=project_id or video_id,
        extractor=extractor,
        date=time.strftime("%Y-%m-%d"),
    )


def object_exists(client: Any, bucket: str, key: str) -> bool:
    """Check whether the target key is already occupied."""
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:  # botocore raises ClientError with a 404/NoSuchKey
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = response.get("Error", {}).get("Code")
            if status == 404 or code in {"404", "NoSuchKey", "NotFound"}:
                return False
        raise


def resolve_remux(quality: str, video: Optional[VideoInfo], override: Optional[bool]) -> bool:
    """Decide whether ffmpeg has to assemble the stream.

    An explicit override wins. Otherwise a probed source answers for itself
    (two tracks, or a segmented protocol such as Twitch's HLS); with no probe
    the safe reading is that 'best' merges and 'progressive' does not.
    """
    if override is not None:
        return override
    if video is not None:
        return video.needs_remux
    return quality == QUALITY_BEST


def ascii_metadata_value(value: str, limit: int) -> str:
    """Percent-encode a value so it survives an ASCII-only metadata header."""
    return quote(value, safe=_METADATA_SAFE_CHARS)[:limit]


def _object_metadata(video: Optional[VideoInfo], quality: str) -> dict[str, str]:
    """Build ASCII-safe user metadata; R2 rejects non-ASCII header values."""
    metadata = {"quality": quality}
    if video is None:
        return metadata
    metadata["source"] = (video.extractor or "unknown").lower()
    metadata["video-id"] = ascii_metadata_value(video.video_id, 256)
    metadata["title"] = ascii_metadata_value(video.title, 512)
    if video.duration_seconds is not None:
        metadata["duration-seconds"] = str(int(video.duration_seconds))
    if video.uploader:
        metadata["uploader"] = ascii_metadata_value(video.uploader, 256)
    return metadata


def ingest_to_r2(
    url: str,
    *,
    client: Any,
    bucket: str,
    key: Optional[str] = None,
    key_template: str = DEFAULT_KEY_TEMPLATE,
    project_id: Optional[str] = None,
    quality: str = QUALITY_BEST,
    max_height: Optional[int] = None,
    remux: Optional[bool] = None,
    part_size: int = DEFAULT_PART_SIZE,
    concurrency: int = 4,
    probe: bool = True,
    overwrite: bool = False,
    public_base_url: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[str] = None,
    ytdlp_path: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    on_progress: Optional[ProgressCallback] = None,
    on_video_info: Optional[Callable[[VideoInfo], None]] = None,
) -> IngestResult:
    """Stream a video from any yt-dlp-supported site into an R2 object.

    Nothing is staged on local disk: yt-dlp downloads and the uploader consumes
    its pipe at the same time, so the only bytes held in memory are the parts
    currently in flight.
    """
    warnings: list[str] = []
    format_selector = build_format_selector(quality, max_height)

    video: Optional[VideoInfo] = None
    if probe:
        video = probe_source(
            url,
            format_selector=format_selector,
            ytdlp_path=ytdlp_path,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
            extra_args=extra_args,
        )
        if video.is_live:
            warnings.append(
                "the source is a live stream; the transfer runs until the broadcast ends"
            )
        if on_video_info is not None:
            on_video_info(video)

    should_remux = resolve_remux(quality, video, remux)
    target_key = key or render_key(key_template, video, project_id)

    if not overwrite and object_exists(client, bucket, target_key):
        raise R2UploadError(
            f"{bucket}/{target_key} already exists; pass overwrite=True to replace it"
        )

    uploader = StreamingMultipartUploader(
        client,
        bucket,
        target_key,
        part_size=part_size,
        concurrency=concurrency,
        content_type="video/mp4",
        metadata=_object_metadata(video, quality),
    )

    # A remux rewrites the container, so the source-side estimate no longer
    # describes the object being written.
    expected_bytes = None if should_remux else (video.filesize_approx if video else None)
    started_at = time.perf_counter()

    with open_source_stream(
        url,
        format_selector=format_selector,
        remux=should_remux,
        adts_to_asc=should_remux and video is not None and video.needs_adts_to_asc,
        ytdlp_path=ytdlp_path,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
        extra_args=extra_args,
    ) as stream:
        uploaded_bytes, part_count, etag = uploader.upload_stream(
            stream.stdout,
            on_progress=on_progress,
            expected_bytes=expected_bytes,
            # The pipe is closed by now; only assemble the object if yt-dlp
            # closed it because it finished, not because it died.
            before_complete=stream.wait_and_check,
        )

    elapsed = time.perf_counter() - started_at

    if expected_bytes and uploaded_bytes < expected_bytes * 0.5:
        warnings.append(
            f"uploaded {uploaded_bytes} bytes but yt-dlp estimated ~{expected_bytes}; "
            "the object may be truncated"
        )

    public_url = None
    if public_base_url:
        public_url = f"{public_base_url.rstrip('/')}/{target_key}"

    return IngestResult(
        bucket=bucket,
        key=target_key,
        uploaded_bytes=uploaded_bytes,
        part_count=part_count,
        elapsed_seconds=elapsed,
        quality=quality,
        format_selector=format_selector,
        remuxed=should_remux,
        video=video,
        public_url=public_url,
        etag=etag,
        warnings=warnings,
    )
