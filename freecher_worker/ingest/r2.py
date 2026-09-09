"""Cloudflare R2 client construction and streaming multipart upload."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, BinaryIO, Callable, Optional

from freecher_worker.ingest.models import (
    DEFAULT_PART_SIZE,
    MAX_PARTS,
    MIN_PART_SIZE,
    UploadProgress,
)

logger = logging.getLogger("freecher_worker")

ProgressCallback = Callable[[UploadProgress], None]


class R2ConfigurationError(Exception):
    """Raised when R2 credentials or endpoint are missing/invalid."""


class R2UploadError(Exception):
    """Raised when a multipart upload cannot be completed."""


# Retried inside the worker rather than left to botocore: a part is already
# fully in memory, so replaying it is free, and on a thin uplink a single
# dropped connection would otherwise throw away the whole transfer.
DEFAULT_PART_ATTEMPTS = 5
RETRY_BACKOFF_SECONDS = 2.0


def is_retryable(error: BaseException) -> bool:
    """True for transport failures and server-side errors, not for bad requests."""
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if isinstance(status, int):
            # 4xx means the request itself is wrong (bad key, expired upload),
            # and replaying it verbatim will fail exactly the same way.
            return status >= 500 or status in {408, 429}
    # No HTTP response at all: the connection dropped or timed out.
    return True


def build_r2_client(
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    *,
    max_attempts: int = 5,
    connect_timeout: float = 15.0,
    read_timeout: float = 300.0,
) -> Any:
    """Create a boto3 S3 client pointed at an R2 account endpoint."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise R2ConfigurationError(
            "boto3 is required for R2 ingest. Install it with: pip install boto3"
        ) from exc

    if not endpoint_url:
        raise R2ConfigurationError("R2 endpoint URL is not configured (R2_ENDPOINT).")
    if not access_key_id or not secret_access_key:
        raise R2ConfigurationError(
            "R2 credentials are not configured (R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY)."
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url.rstrip("/"),
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
        config=Config(
            retries={"max_attempts": max_attempts, "mode": "standard"},
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            # R2 rejects the streaming checksum trailers newer botocore versions
            # add by default; keep the plain SigV4 payload signing it expects.
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            signature_version="s3v4",
        ),
    )


def read_exact(stream: BinaryIO, size: int) -> bytes:
    """Read exactly `size` bytes unless the stream ends first.

    Parts other than the last must all be the same size for R2, so a short read
    from a pipe must never be turned into a short part.
    """
    buffer = bytearray()
    while len(buffer) < size:
        chunk = stream.read(size - len(buffer))
        if not chunk:
            break
        buffer.extend(chunk)
    return bytes(buffer)


class StreamingMultipartUploader:
    """Uploads an unseekable byte stream to R2 as a multipart object.

    Data is consumed in fixed-size parts and each part is handed to a worker
    thread, so the producer (yt-dlp) keeps downloading while earlier parts are
    still in flight. Memory stays bounded at roughly part_size * concurrency.
    """

    def __init__(
        self,
        client: Any,
        bucket: str,
        key: str,
        *,
        part_size: int = DEFAULT_PART_SIZE,
        concurrency: int = 4,
        content_type: str = "video/mp4",
        metadata: Optional[dict[str, str]] = None,
        part_attempts: int = DEFAULT_PART_ATTEMPTS,
    ) -> None:
        if part_size < MIN_PART_SIZE:
            raise ValueError(
                f"part_size must be at least {MIN_PART_SIZE} bytes (5 MiB), got {part_size}"
            )
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        self.client = client
        self.bucket = bucket
        self.key = key
        self.part_size = part_size
        self.concurrency = concurrency
        self.content_type = content_type
        self.metadata = metadata or {}
        self.part_attempts = max(1, part_attempts)

    def upload_stream(
        self,
        stream: BinaryIO,
        *,
        on_progress: Optional[ProgressCallback] = None,
        expected_bytes: Optional[int] = None,
        before_complete: Optional[Callable[[], None]] = None,
    ) -> tuple[int, int, Optional[str]]:
        """Stream `stream` into the object, returning (bytes, part_count, etag).

        `before_complete` runs once every part has landed but before the object
        is assembled, so a producer that died mid-stream can abort the upload
        instead of publishing a truncated object.
        """
        create_kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": self.key,
            "ContentType": self.content_type,
        }
        if self.metadata:
            create_kwargs["Metadata"] = self.metadata

        upload_id = self.client.create_multipart_upload(**create_kwargs)["UploadId"]
        logger.info("r2 multipart upload started: %s/%s (%s)", self.bucket, self.key, upload_id)

        parts: dict[int, str] = {}
        uploaded_bytes = 0
        state_lock = threading.Lock()
        slots = threading.Semaphore(self.concurrency)
        futures: list[Future] = []
        started_at = time.perf_counter()

        def finish_part(future: Future) -> None:
            slots.release()
            if future.cancelled() or future.exception() is not None:
                return
            number, etag, size = future.result()
            with state_lock:
                nonlocal uploaded_bytes
                parts[number] = etag
                uploaded_bytes += size
                snapshot = UploadProgress(
                    part_number=number,
                    part_bytes=size,
                    uploaded_bytes=uploaded_bytes,
                    elapsed_seconds=time.perf_counter() - started_at,
                    expected_bytes=expected_bytes,
                )
            if on_progress is not None:
                on_progress(snapshot)

        try:
            with ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="r2-part") as pool:
                part_number = 1
                while True:
                    chunk = read_exact(stream, self.part_size)
                    if not chunk:
                        break
                    if part_number > MAX_PARTS:
                        raise R2UploadError(
                            f"stream exceeds the {MAX_PARTS}-part limit at part_size="
                            f"{self.part_size}; use a larger --part-size"
                        )
                    _raise_first_failure(futures)
                    slots.acquire()
                    future = pool.submit(self._upload_part, upload_id, part_number, chunk)
                    future.add_done_callback(finish_part)
                    futures.append(future)
                    part_number += 1

            _raise_first_failure(futures)

            if not parts:
                raise R2UploadError("the source stream produced no data")

            if before_complete is not None:
                before_complete()

            manifest = [
                {"PartNumber": number, "ETag": parts[number]} for number in sorted(parts)
            ]
            response = self.client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=self.key,
                UploadId=upload_id,
                MultipartUpload={"Parts": manifest},
            )
        except BaseException:
            self._abort(upload_id)
            raise

        logger.info(
            "r2 multipart upload completed: %s/%s (%d parts, %d bytes)",
            self.bucket,
            self.key,
            len(parts),
            uploaded_bytes,
        )
        return uploaded_bytes, len(parts), response.get("ETag")

    def _upload_part(self, upload_id: str, part_number: int, chunk: bytes) -> tuple[int, str, int]:
        for attempt in range(1, self.part_attempts + 1):
            try:
                response = self.client.upload_part(
                    Bucket=self.bucket,
                    Key=self.key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=chunk,
                )
            except Exception as error:
                if attempt >= self.part_attempts or not is_retryable(error):
                    raise
                delay = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "part %d failed (attempt %d/%d): %s; retrying in %.0fs",
                    part_number,
                    attempt,
                    self.part_attempts,
                    error,
                    delay,
                )
                time.sleep(delay)
                continue
            return part_number, response["ETag"], len(chunk)
        raise R2UploadError(f"part {part_number} exhausted its retries")

    def _abort(self, upload_id: str) -> None:
        try:
            self.client.abort_multipart_upload(
                Bucket=self.bucket, Key=self.key, UploadId=upload_id
            )
            logger.warning("aborted r2 multipart upload %s for %s/%s", upload_id, self.bucket, self.key)
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            logger.error("failed to abort r2 multipart upload %s: %s", upload_id, exc)


def _raise_first_failure(futures: list[Future]) -> None:
    """Surface the first part failure so the reader stops pulling data."""
    for future in futures:
        if future.done() and not future.cancelled():
            error = future.exception()
            if error is not None:
                raise R2UploadError(f"part upload failed: {error}") from error
