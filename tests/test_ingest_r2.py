"""Tests for the streaming source -> R2 ingest path."""

from __future__ import annotations

import io
import json
import os
import stat
import threading

import pytest

from freecher_worker.ingest import (
    MIN_PART_SIZE,
    QUALITY_BEST,
    QUALITY_PROGRESSIVE,
    R2UploadError,
    StreamingMultipartUploader,
    build_format_selector,
    build_stream_command,
    ingest_to_r2,
    object_exists,
    probe_source,
    read_exact,
    render_key,
    resolve_remux,
)
from freecher_worker.ingest.models import VideoInfo
from freecher_worker.ingest.r2 import is_retryable
from freecher_worker.ingest.service import ascii_metadata_value
from freecher_worker.ingest.source import video_info_from_payload

PART_SIZE = MIN_PART_SIZE


class FakeS3Client:
    """Minimal in-memory stand-in for the boto3 S3 client."""

    def __init__(self, fail_on_part: int | None = None, existing_keys: set[str] | None = None) -> None:
        self.fail_on_part = fail_on_part
        self.existing_keys = existing_keys or set()
        self.parts: dict[int, bytes] = {}
        self.completed: list[dict] = []
        self.aborted: list[str] = []
        self.created: list[dict] = []
        self.lock = threading.Lock()

    def create_multipart_upload(self, **kwargs):
        self.created.append(kwargs)
        return {"UploadId": "upload-1"}

    def upload_part(self, *, Bucket, Key, UploadId, PartNumber, Body):
        if self.fail_on_part == PartNumber:
            raise RuntimeError(f"synthetic failure on part {PartNumber}")
        with self.lock:
            self.parts[PartNumber] = bytes(Body)
        return {"ETag": f'"etag-{PartNumber}"'}

    def complete_multipart_upload(self, **kwargs):
        self.completed.append(kwargs)
        return {"ETag": '"final-etag"'}

    def abort_multipart_upload(self, *, Bucket, Key, UploadId):
        self.aborted.append(UploadId)

    def head_object(self, *, Bucket, Key):
        if Key in self.existing_keys:
            return {"ContentLength": 1}
        raise _missing_key_error()

    @property
    def assembled(self) -> bytes:
        return b"".join(self.parts[number] for number in sorted(self.parts))


def _missing_key_error() -> Exception:
    error = Exception("Not Found")
    error.response = {"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}
    return error


class DribbleStream(io.RawIOBase):
    """A stream that returns fewer bytes than asked for, like a real pipe."""

    def __init__(self, payload: bytes, dribble: int) -> None:
        self.payload = payload
        self.dribble = dribble
        self.position = 0

    def read(self, size=-1):  # type: ignore[override]
        take = min(self.dribble, size if size and size > 0 else self.dribble)
        chunk = self.payload[self.position:self.position + take]
        self.position += len(chunk)
        return chunk


class FakeYtDlp:
    """An executable stand-in for yt-dlp that records how it was invoked."""

    def __init__(self, tmp_path, payload: bytes, info: dict | None = None, exit_code: int = 0) -> None:
        payload_path = tmp_path / "payload.bin"
        payload_path.write_bytes(payload)
        info_path = tmp_path / "info.json"
        info_path.write_text(json.dumps(info or {}))
        self.args_log = tmp_path / "args.log"

        script = tmp_path / "fake-yt-dlp"
        script.write_text(
            "#!/bin/sh\n"
            f'echo "$@" >> "{self.args_log}"\n'
            'for arg in "$@"; do\n'
            '  if [ "$arg" = "-J" ]; then\n'
            f'    cat "{info_path}"\n'
            "    exit 0\n"
            "  fi\n"
            "done\n"
            'echo "fake yt-dlp: streaming" >&2\n'
            f'cat "{payload_path}"\n'
            f"exit {exit_code}\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        self.path = str(script)

    @property
    def download_args(self) -> str:
        """The argv of the streaming call (the probe runs first)."""
        return [line for line in self.args_log.read_text().splitlines() if "-J" not in line][-1]


def test_read_exact_fills_a_full_part_from_a_dribbling_stream():
    payload = os.urandom(1000)
    stream = DribbleStream(payload, dribble=64)

    assert read_exact(stream, 1000) == payload
    assert read_exact(stream, 1000) == b""


def test_uploader_splits_stream_into_equal_parts_and_preserves_bytes():
    payload = os.urandom(PART_SIZE * 2 + 1234)
    client = FakeS3Client()
    uploader = StreamingMultipartUploader(client, "bucket", "input/video.mp4", part_size=PART_SIZE)

    uploaded, part_count, etag = uploader.upload_stream(io.BytesIO(payload))

    assert uploaded == len(payload)
    assert part_count == 3
    assert etag == '"final-etag"'
    assert client.assembled == payload
    # R2 requires every part but the last to be the same size.
    assert [len(client.parts[n]) for n in sorted(client.parts)] == [PART_SIZE, PART_SIZE, 1234]
    assert client.completed[0]["MultipartUpload"]["Parts"] == [
        {"PartNumber": 1, "ETag": '"etag-1"'},
        {"PartNumber": 2, "ETag": '"etag-2"'},
        {"PartNumber": 3, "ETag": '"etag-3"'},
    ]
    assert client.aborted == []


def test_uploader_reports_progress_per_part():
    payload = os.urandom(PART_SIZE + 10)
    client = FakeS3Client()
    uploader = StreamingMultipartUploader(client, "bucket", "k", part_size=PART_SIZE, concurrency=1)

    seen = []
    uploader.upload_stream(io.BytesIO(payload), on_progress=seen.append, expected_bytes=len(payload))

    assert [p.part_number for p in seen] == [1, 2]
    assert seen[-1].uploaded_bytes == len(payload)
    assert seen[-1].percent == pytest.approx(100.0)


def test_uploader_aborts_once_a_part_exhausts_its_retries(monkeypatch):
    monkeypatch.setattr("freecher_worker.ingest.r2.RETRY_BACKOFF_SECONDS", 0.0)
    payload = os.urandom(PART_SIZE * 3)
    client = FakeS3Client(fail_on_part=2)
    uploader = StreamingMultipartUploader(
        client, "bucket", "k", part_size=PART_SIZE, concurrency=1, part_attempts=3
    )

    with pytest.raises(R2UploadError):
        uploader.upload_stream(io.BytesIO(payload))

    assert client.aborted == ["upload-1"]
    assert client.completed == []


def test_uploader_aborts_on_an_empty_stream():
    client = FakeS3Client()
    uploader = StreamingMultipartUploader(client, "bucket", "k", part_size=PART_SIZE)

    with pytest.raises(R2UploadError):
        uploader.upload_stream(io.BytesIO(b""))

    assert client.aborted == ["upload-1"]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (500, True), (503, True), (429, True), (408, True),
        # Replaying a malformed or unauthorised request changes nothing.
        (403, False), (404, False), (400, False),
    ],
)
def test_is_retryable_separates_transport_faults_from_bad_requests(status, expected):
    error = Exception("boom")
    error.response = {"ResponseMetadata": {"HTTPStatusCode": status}}
    assert is_retryable(error) is expected


def test_is_retryable_treats_a_dropped_connection_as_transient():
    # No HTTP response at all is what a closed connection looks like.
    assert is_retryable(ConnectionError("Connection was closed")) is True


class FlakyS3Client(FakeS3Client):
    """Drops the connection on the first N attempts of every part."""

    def __init__(self, failures: int, status: int | None = None) -> None:
        super().__init__()
        self.remaining = failures
        self.status = status
        self.attempts = 0

    def upload_part(self, **kwargs):
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            error = Exception("Connection was closed before we received a valid response")
            if self.status is not None:
                error.response = {"ResponseMetadata": {"HTTPStatusCode": self.status}}
            raise error
        return super().upload_part(**kwargs)


def test_uploader_replays_a_part_after_a_dropped_connection(monkeypatch):
    monkeypatch.setattr("freecher_worker.ingest.r2.RETRY_BACKOFF_SECONDS", 0.0)
    payload = os.urandom(PART_SIZE + 100)
    client = FlakyS3Client(failures=2)
    uploader = StreamingMultipartUploader(client, "bucket", "k", part_size=PART_SIZE, concurrency=1)

    uploaded, part_count, _ = uploader.upload_stream(io.BytesIO(payload))

    assert uploaded == len(payload)
    assert part_count == 2
    assert client.attempts == 4  # two failures replayed, then both parts land
    assert client.assembled == payload
    assert client.aborted == []


def test_uploader_gives_up_immediately_on_a_permission_error(monkeypatch):
    monkeypatch.setattr("freecher_worker.ingest.r2.RETRY_BACKOFF_SECONDS", 0.0)
    client = FlakyS3Client(failures=99, status=403)
    uploader = StreamingMultipartUploader(client, "bucket", "k", part_size=PART_SIZE, concurrency=1)

    with pytest.raises(R2UploadError):
        uploader.upload_stream(io.BytesIO(os.urandom(PART_SIZE + 1)))

    # One attempt, not five: a 403 will never turn into a 200.
    assert client.attempts == 1
    assert client.aborted == ["upload-1"]


def test_uploader_rejects_a_part_size_below_the_r2_minimum():
    with pytest.raises(ValueError):
        StreamingMultipartUploader(FakeS3Client(), "bucket", "k", part_size=1024)


def test_format_selector_merges_for_best_and_stays_single_stream_for_progressive():
    best = build_format_selector(QUALITY_BEST, max_height=1080)
    assert best.startswith("bv*[ext=mp4][vcodec^=avc1][height<=1080]+ba[ext=m4a]")
    assert build_format_selector(QUALITY_PROGRESSIVE) == "b[ext=mp4]/b"

    with pytest.raises(ValueError):
        build_format_selector("ultra")


def test_stream_command_only_asks_ffmpeg_for_a_fragmented_mp4_when_remuxing():
    remuxed = build_stream_command("URL", format_selector="bv*+ba", remux=True, ytdlp_path="/bin/yt-dlp")
    assert "--downloader" in remuxed and "ffmpeg" in remuxed
    assert any("frag_keyframe+empty_moov" in arg for arg in remuxed)
    assert not any("aac_adtstoasc" in arg for arg in remuxed)
    assert remuxed[-3:] == ["-o", "-", "URL"]

    hls = build_stream_command(
        "URL", format_selector="b", remux=True, adts_to_asc=True, ytdlp_path="/bin/yt-dlp"
    )
    assert any("aac_adtstoasc" in arg for arg in hls)

    direct = build_stream_command("URL", format_selector="b", remux=False, ytdlp_path="/bin/yt-dlp")
    assert "--downloader" not in direct


def test_video_info_reads_the_selected_formats_not_just_the_video():
    merged = video_info_from_payload({
        "id": "v1",
        "title": "t",
        "requested_formats": [
            {"format_id": "137", "protocol": "https", "filesize": 100},
            {"format_id": "140", "protocol": "https", "filesize": 20},
        ],
    })
    assert merged.requested_format_ids == ["137", "140"]
    assert merged.protocol == "https+https"
    assert merged.filesize_approx == 120

    single = video_info_from_payload({"id": "v2", "title": "t", "format_id": "22", "protocol": "https"})
    assert single.requested_format_ids == ["22"]
    assert single.protocol == "https"


@pytest.mark.parametrize(
    ("protocol", "format_ids", "expected"),
    [
        ("https", ["22"], False),
        ("https+https", ["137", "140"], True),
        # Twitch and other live platforms serve segment lists, not a byte range.
        ("m3u8_native", ["720p60"], True),
        ("http_dash_segments", ["dash-1"], True),
        (None, [], False),
    ],
)
def test_needs_remux_follows_the_selected_protocol(protocol, format_ids, expected):
    video = VideoInfo(
        video_id="v",
        title="t",
        duration_seconds=None,
        uploader=None,
        protocol=protocol,
        requested_format_ids=format_ids,
    )
    assert video.needs_remux is expected


@pytest.mark.parametrize(
    ("protocol", "acodec", "expected"),
    [
        # HLS carries AAC as ADTS frames, which the mp4 muxer rejects outright.
        ("m3u8_native", "aac", True),
        ("m3u8_native", None, True),
        ("m3u8_native", "mp4a.40.2", True),
        ("m3u8_native", "opus", False),
        # Anything not segmented already hands over ASC.
        ("https", "aac", False),
        ("https+https", "mp4a.40.2", False),
        (None, "aac", False),
    ],
)
def test_needs_adts_to_asc_only_for_aac_inside_hls(protocol, acodec, expected):
    video = VideoInfo(
        video_id="v",
        title="t",
        duration_seconds=None,
        uploader=None,
        protocol=protocol,
        acodec=acodec,
    )
    assert video.needs_adts_to_asc is expected


def test_resolve_remux_prefers_the_probe_then_the_quality_mode():
    hls = VideoInfo(
        video_id="v", title="t", duration_seconds=None, uploader=None,
        protocol="m3u8_native", requested_format_ids=["720p60"],
    )
    direct = VideoInfo(
        video_id="v", title="t", duration_seconds=None, uploader=None,
        protocol="https", requested_format_ids=["22"],
    )

    # A probed source answers for itself, whatever the quality mode says.
    assert resolve_remux(QUALITY_PROGRESSIVE, hls, None) is True
    assert resolve_remux(QUALITY_BEST, direct, None) is False
    # An explicit flag always wins.
    assert resolve_remux(QUALITY_PROGRESSIVE, hls, False) is False
    # Without a probe, 'best' is assumed to merge and 'progressive' is not.
    assert resolve_remux(QUALITY_BEST, None, None) is True
    assert resolve_remux(QUALITY_PROGRESSIVE, None, None) is False


def test_render_key_expands_placeholders():
    video = VideoInfo(
        video_id="abc123", title="t", duration_seconds=10, uploader="u", extractor="Twitch",
    )

    assert render_key("input/{video_id}/source.mp4", video) == "input/abc123/source.mp4"
    assert render_key("input/{project_id}/source.mp4", video, "proj-7") == "input/proj-7/source.mp4"
    assert render_key("{extractor}/{video_id}.mp4", video) == "twitch/abc123.mp4"


@pytest.mark.parametrize(
    "title",
    [
        "Детектор лжи с @leva2k 🔞",   # Cyrillic + emoji
        "Plain ASCII title (2024)",
        "100% real",                    # a literal percent must stay unambiguous
        "",
    ],
)
def test_metadata_values_survive_the_ascii_only_header(title):
    from urllib.parse import unquote

    encoded = ascii_metadata_value(title, 512)

    # S3/R2 reject anything outside US-ASCII in a metadata header.
    assert encoded.isascii()
    assert unquote(encoded) == title


def test_metadata_leaves_a_plain_ascii_title_readable():
    # Percent-encoding must not turn an ordinary title into noise.
    assert ascii_metadata_value("Best of stream (part 2)", 512) == "Best of stream (part 2)"


def test_object_exists_distinguishes_missing_from_present():
    client = FakeS3Client(existing_keys={"input/taken.mp4"})

    assert object_exists(client, "bucket", "input/taken.mp4") is True
    assert object_exists(client, "bucket", "input/free.mp4") is False


def test_probe_source_maps_ytdlp_metadata(tmp_path):
    ytdlp = FakeYtDlp(
        tmp_path,
        b"",
        info={
            "id": "abc", "title": "Clip", "duration": 3725, "uploader": "Chan",
            "height": 1080, "extractor_key": "Twitch", "format_id": "1080p60",
            "protocol": "m3u8_native",
        },
    )

    video = probe_source("URL", ytdlp_path=ytdlp.path)

    assert (video.video_id, video.title, video.uploader) == ("abc", "Clip", "Chan")
    assert video.duration_label == "1:02:05"
    assert video.extractor == "Twitch"
    assert video.needs_remux is True


def test_ingest_streams_the_whole_video_into_one_object(tmp_path):
    payload = os.urandom(PART_SIZE * 2 + 4096)
    ytdlp = FakeYtDlp(tmp_path, payload, info={
        "id": "vid42", "title": "Demo", "duration": 60,
        "extractor_key": "Youtube", "format_id": "22", "protocol": "https",
    })
    client = FakeS3Client()

    result = ingest_to_r2(
        "https://youtu.be/vid42",
        client=client,
        bucket="freecher",
        quality=QUALITY_PROGRESSIVE,
        part_size=PART_SIZE,
        ytdlp_path=ytdlp.path,
        public_base_url="https://cdn.example.com",
    )

    assert result.key == "input/vid42/source.mp4"
    assert result.uploaded_bytes == len(payload)
    assert result.part_count == 3
    assert result.remuxed is False
    assert client.assembled == payload
    assert result.public_url == "https://cdn.example.com/input/vid42/source.mp4"
    assert result.video is not None and result.video.title == "Demo"
    assert client.created[0]["ContentType"] == "video/mp4"
    assert client.created[0]["Metadata"]["video-id"] == "vid42"
    assert client.created[0]["Metadata"]["source"] == "youtube"
    # A direct byte range needs no ffmpeg in the middle.
    assert "--downloader" not in ytdlp.download_args
    assert "aac_adtstoasc" not in ytdlp.download_args


def test_ingest_puts_ffmpeg_in_the_middle_for_an_hls_source(tmp_path):
    payload = os.urandom(PART_SIZE + 512)
    ytdlp = FakeYtDlp(tmp_path, payload, info={
        "id": "2211", "title": "VOD", "duration": 300,
        "extractor_key": "Twitch", "format_id": "1080p60", "protocol": "m3u8_native",
    })
    client = FakeS3Client()

    result = ingest_to_r2(
        "https://www.twitch.tv/videos/2211",
        client=client,
        bucket="freecher",
        quality=QUALITY_PROGRESSIVE,
        part_size=PART_SIZE,
        ytdlp_path=ytdlp.path,
    )

    # Segment lists cannot be piped through untouched, even in progressive mode.
    assert result.remuxed is True
    assert "--downloader ffmpeg" in ytdlp.download_args
    assert "frag_keyframe+empty_moov" in ytdlp.download_args
    # Without this the mp4 muxer rejects HLS's ADTS-framed AAC outright.
    assert "aac_adtstoasc" in ytdlp.download_args
    assert client.assembled == payload


def test_ingest_aborts_when_ytdlp_exits_nonzero(tmp_path):
    ytdlp = FakeYtDlp(tmp_path, os.urandom(PART_SIZE + 1), info={"id": "bad"}, exit_code=1)
    client = FakeS3Client()

    with pytest.raises(Exception) as excinfo:
        ingest_to_r2(
            "https://youtu.be/bad",
            client=client,
            bucket="freecher",
            quality=QUALITY_PROGRESSIVE,
            part_size=PART_SIZE,
            ytdlp_path=ytdlp.path,
        )

    assert "yt-dlp exited with code 1" in str(excinfo.value)
    # A truncated object must never be published.
    assert client.completed == []
    assert client.aborted == ["upload-1"]


def test_ingest_refuses_to_clobber_an_existing_key(tmp_path):
    ytdlp = FakeYtDlp(tmp_path, b"x" * 16, info={"id": "vid42"})
    client = FakeS3Client(existing_keys={"input/vid42/source.mp4"})

    with pytest.raises(R2UploadError, match="already exists"):
        ingest_to_r2(
            "https://youtu.be/vid42",
            client=client,
            bucket="freecher",
            quality=QUALITY_PROGRESSIVE,
            part_size=PART_SIZE,
            ytdlp_path=ytdlp.path,
        )
