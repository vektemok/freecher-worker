"""Glue between the yt-dlp source stream and the R2 multipart sink."""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from urllib.parse import quote

from freecher_worker.ingest.audio import (
    AudioArtifact,
    AudioArtifactError,
    AudioEncodeSettings,
    AudioSidecar,
    TeeReader,
    duration_mismatch,
    probe_audio_file,
)
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

# The audio key is derived from the resolved video key rather than rendered
# from its own template, so the two artifacts of one ingest can never drift
# apart: input/<source_id>/source.mp4 -> processing/<source_id>/audio.m4a.
SOURCE_KEY_PREFIX = "input"
AUDIO_KEY_PREFIX = "processing"
AUDIO_FILENAME = "audio.m4a"

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


def derive_audio_key(video_key: str) -> str:
    """Map a source video key onto the audio artifact key beside it.

    The leading "input/" segment becomes "processing/" and the filename becomes
    audio.m4a, so the default template lands on processing/{source_id}/audio.m4a
    and a custom key still gets a collision-free companion.
    """
    segments = [part for part in video_key.split("/") if part]
    directory = segments[:-1]
    if directory and directory[0] == SOURCE_KEY_PREFIX:
        directory = directory[1:]
    audio_key = "/".join([AUDIO_KEY_PREFIX, *directory, AUDIO_FILENAME])
    if audio_key == video_key:
        raise ValueError(
            f"the audio key would overwrite the source video at '{video_key}'; "
            "pass an explicit audio_key"
        )
    return audio_key


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


def ascii_metadata_value(value: str, limit: int, *, safe: str = _METADATA_SAFE_CHARS) -> str:
    """Percent-encode a value so it survives an ASCII-only metadata header."""
    return quote(value, safe=safe)[:limit]


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


def _audio_object_metadata(
    video: Optional[VideoInfo],
    quality: str,
    *,
    bucket: str,
    source_key: str,
    codec: str,
    sample_rate: int,
    channels: int,
    source_duration_seconds: Optional[float],
    audio_duration_seconds: Optional[float],
) -> dict[str, str]:
    """User metadata for the audio artifact; every value must be US-ASCII.

    Carries enough to reconstruct the pairing without a database: which object
    it was cut from, how long that source runs, and exactly how the audio was
    encoded.
    """
    metadata = _object_metadata(video, quality)
    metadata["artifact"] = "audio"
    metadata["source-bucket"] = bucket
    # An object key is ASCII already; keeping '/' unescaped leaves it usable
    # as-is rather than as %2F noise. The cap keeps the whole header set inside
    # the 2 KB S3/R2 allows for user metadata once title and uploader are in it.
    metadata["source-key"] = ascii_metadata_value(
        source_key, 512, safe=_METADATA_SAFE_CHARS + "/"
    )
    metadata["audio-codec"] = codec
    metadata["audio-sample-rate"] = str(sample_rate)
    metadata["audio-channels"] = str(channels)
    if source_duration_seconds:
        metadata["source-duration-seconds"] = f"{source_duration_seconds:.3f}"
    if audio_duration_seconds:
        metadata["audio-duration-seconds"] = f"{audio_duration_seconds:.3f}"
    return metadata


def _finish_audio_artifact(
    *,
    client: Any,
    bucket: str,
    audio_key: str,
    source_key: str,
    audio_path: Path,
    sidecar: AudioSidecar,
    tee: Optional[TeeReader],
    settings: AudioEncodeSettings,
    video: Optional[VideoInfo],
    quality: str,
    part_size: int,
    concurrency: int,
    ffprobe_path: str,
    on_progress: Optional[ProgressCallback],
) -> AudioArtifact:
    """Finish the encode, verify the timeline, and upload the artifact.

    Raises rather than returns on any problem; the caller degrades to a warning
    so a bad audio branch can never cost an already-uploaded source video.
    """
    if tee is not None:
        tee.wait_for_mirror()
    sidecar.wait()

    probed = probe_audio_file(audio_path, ffprobe_path=ffprobe_path)
    audio_duration = probed["duration_seconds"]
    source_duration = video.duration_seconds if video else None

    # ffmpeg exits 0 on a truncated input, so length is the real check that the
    # artifact still lines up with the source timeline end to end.
    problem = duration_mismatch(source_duration, audio_duration)
    if problem is not None:
        raise AudioArtifactError(problem)

    metadata = _audio_object_metadata(
        video,
        quality,
        bucket=bucket,
        source_key=source_key,
        codec=probed["codec"],
        sample_rate=probed["sample_rate"],
        channels=probed["channels"],
        # Without a probe the artifact itself is the only measure of how long
        # the source runs, and it covers the whole of it.
        source_duration_seconds=source_duration or audio_duration,
        audio_duration_seconds=audio_duration,
    )

    uploader = StreamingMultipartUploader(
        client,
        bucket,
        audio_key,
        part_size=part_size,
        concurrency=concurrency,
        content_type=settings.content_type,
        metadata=metadata,
    )
    size_bytes = audio_path.stat().st_size
    with audio_path.open("rb") as handle:
        uploaded_bytes, part_count, etag = uploader.upload_stream(
            handle, on_progress=on_progress, expected_bytes=size_bytes
        )

    logger.info(
        "audio artifact uploaded: %s/%s (%d bytes, %s Hz, %d ch)",
        bucket,
        audio_key,
        uploaded_bytes,
        probed["sample_rate"],
        probed["channels"],
    )

    return AudioArtifact(
        bucket=bucket,
        key=audio_key,
        source_key=source_key,
        size_bytes=uploaded_bytes,
        codec=probed["codec"],
        sample_rate=probed["sample_rate"],
        channels=probed["channels"],
        source_duration_seconds=source_duration or audio_duration,
        audio_duration_seconds=audio_duration,
        part_count=part_count,
        etag=etag,
    )


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
    # Off by default so this stays a single-artifact primitive; the CLI turns
    # it on, since a transcription copy is what the pipeline downstream wants.
    extract_audio: bool = False,
    audio_key: Optional[str] = None,
    audio_settings: Optional[AudioEncodeSettings] = None,
    audio_staging_dir: Optional[str] = None,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
    on_audio_progress: Optional[ProgressCallback] = None,
    network_timeout: float = 30.0,
    network_retries: int = 5,
    stall_timeout: float = 60.0,
) -> IngestResult:
    """Stream a video from any yt-dlp-supported site into an R2 object.

    No video is staged on local disk: yt-dlp downloads and the uploader consumes
    its pipe at the same time, so the only bytes held in memory are the parts
    currently in flight.

    With `extract_audio`, the same bytes are mirrored into ffmpeg on the way
    past and a mono speech-rate m4a is written beside the source video. The
    audio is the only thing that touches the disk, and only until it is
    uploaded. It is strictly a second artifact: if it cannot be produced or does
    not span the whole source, the source video still lands and the failure is
    reported through `IngestResult.audio_error`.
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

    settings = audio_settings or AudioEncodeSettings()
    audio_artifact: Optional[AudioArtifact] = None
    audio_error: Optional[str] = None
    audio_target_key: Optional[str] = None
    produce_audio = extract_audio

    if produce_audio:
        audio_target_key = audio_key or derive_audio_key(target_key)
        if audio_target_key == target_key:
            raise ValueError(
                f"the audio key would overwrite the source video at '{target_key}'"
            )
        if not overwrite and object_exists(client, bucket, audio_target_key):
            # Idempotent: re-running an ingest leaves an artifact that is
            # already there alone rather than re-encoding and re-uploading it.
            produce_audio = False
            audio_artifact = AudioArtifact(
                bucket=bucket,
                key=audio_target_key,
                source_key=target_key,
                size_bytes=0,
                codec=settings.codec,
                sample_rate=settings.sample_rate,
                channels=settings.channels,
                source_duration_seconds=video.duration_seconds if video else None,
                skipped=True,
            )
            logger.info("audio artifact already present, skipping: %s", audio_target_key)

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

    staging_dir: Optional[str] = None
    audio_path: Optional[Path] = None
    sidecar: Optional[AudioSidecar] = None
    tee: Optional[TeeReader] = None

    if produce_audio:
        staging_dir = tempfile.mkdtemp(prefix="freecher-ingest-audio-", dir=audio_staging_dir)
        audio_path = Path(staging_dir) / AUDIO_FILENAME

    try:
        with open_source_stream(
            url,
            format_selector=format_selector,
            remux=should_remux,
            adts_to_asc=should_remux and video is not None and video.needs_adts_to_asc,
            ytdlp_path=ytdlp_path,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
            extra_args=extra_args,
            network_timeout=network_timeout,
            network_retries=network_retries,
            stall_timeout=stall_timeout,
        ) as stream:
            upload_source = stream.stdout
            if produce_audio and audio_path is not None:
                try:
                    sidecar = AudioSidecar(audio_path, settings, ffmpeg_path=ffmpeg_path)
                except Exception as exc:
                    # A branch that cannot even start must not stop the video.
                    produce_audio = False
                    audio_error = str(exc)
                    warnings.append(f"audio artifact was not produced: {exc}")
                else:
                    tee = TeeReader(stream.stdout, sidecar)
                    upload_source = tee

            uploaded_bytes, part_count, etag = uploader.upload_stream(
                upload_source,
                on_progress=on_progress,
                expected_bytes=expected_bytes,
                # The pipe is closed by now; only assemble the object if yt-dlp
                # closed it because it finished, not because it died.
                before_complete=stream.wait_and_check,
            )
    except BaseException:
        # The source video never landed, so there is nothing for an audio
        # artifact to accompany; drop the branch without touching R2.
        if tee is not None:
            tee.abandon()
        if sidecar is not None:
            sidecar.terminate()
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    elapsed = time.perf_counter() - started_at

    if (
        produce_audio
        and sidecar is not None
        and audio_path is not None
        and audio_target_key is not None
    ):
        try:
            audio_artifact = _finish_audio_artifact(
                client=client,
                bucket=bucket,
                audio_key=audio_target_key,
                source_key=target_key,
                audio_path=audio_path,
                sidecar=sidecar,
                tee=tee,
                settings=settings,
                video=video,
                quality=quality,
                part_size=part_size,
                concurrency=concurrency,
                ffprobe_path=ffprobe_path,
                on_progress=on_audio_progress,
            )
        except Exception as exc:
            audio_error = str(exc)
            warnings.append(f"audio artifact was not produced: {exc}")
            logger.error("audio artifact failed for %s: %s", target_key, exc)
        finally:
            sidecar.terminate()

    if staging_dir is not None:
        shutil.rmtree(staging_dir, ignore_errors=True)

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
        audio=audio_artifact,
        audio_error=audio_error,
    )
