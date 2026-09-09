"""Tests for the transcription-ready audio artifact produced during ingest."""

from __future__ import annotations

import io
import json
import os
import shutil
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest

from freecher_worker.ingest import (
    MIN_PART_SIZE,
    QUALITY_PROGRESSIVE,
    AudioArtifactError,
    AudioEncodeSettings,
    AudioSidecar,
    TeeReader,
    build_audio_command,
    derive_audio_key,
    duration_mismatch,
    ingest_to_r2,
    probe_audio_file,
)

PART_SIZE = MIN_PART_SIZE
SAMPLE_DURATION = 12.0

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe are not installed",
)


# --------------------------------------------------------------------------
# doubles
# --------------------------------------------------------------------------


class KeyedFakeS3Client:
    """In-memory S3 stand-in that keeps each object's parts separate."""

    def __init__(self, existing_keys: set[str] | None = None) -> None:
        self.existing_keys = set(existing_keys or ())
        self.parts: dict[str, dict[int, bytes]] = {}
        self.created: dict[str, dict] = {}
        self.completed: list[str] = []
        self.aborted: list[str] = []
        self.lock = threading.Lock()
        self._upload_keys: dict[str, str] = {}

    def create_multipart_upload(self, **kwargs):
        key = kwargs["Key"]
        upload_id = f"upload-{len(self._upload_keys) + 1}"
        self.created[key] = kwargs
        self._upload_keys[upload_id] = key
        self.parts.setdefault(key, {})
        return {"UploadId": upload_id}

    def upload_part(self, *, Bucket, Key, UploadId, PartNumber, Body):
        with self.lock:
            self.parts[Key][PartNumber] = bytes(Body)
        return {"ETag": f'"etag-{PartNumber}"'}

    def complete_multipart_upload(self, **kwargs):
        self.completed.append(kwargs["Key"])
        self.existing_keys.add(kwargs["Key"])
        return {"ETag": '"final-etag"'}

    def abort_multipart_upload(self, *, Bucket, Key, UploadId):
        self.aborted.append(Key)

    def head_object(self, *, Bucket, Key):
        if Key in self.existing_keys:
            return {"ContentLength": 1}
        error = Exception("Not Found")
        error.response = {"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}
        raise error

    def body(self, key: str) -> bytes:
        stored = self.parts.get(key, {})
        return b"".join(stored[number] for number in sorted(stored))


class FakeYtDlp:
    """An executable stand-in for yt-dlp that streams a fixed payload."""

    def __init__(self, tmp_path: Path, payload: bytes, info: dict, exit_code: int = 0) -> None:
        payload_path = tmp_path / "payload.bin"
        payload_path.write_bytes(payload)
        info_path = tmp_path / "info.json"
        info_path.write_text(json.dumps(info))

        script = tmp_path / "fake-yt-dlp"
        script.write_text(
            "#!/bin/sh\n"
            'for arg in "$@"; do\n'
            '  if [ "$arg" = "-J" ]; then\n'
            f'    cat "{info_path}"\n'
            "    exit 0\n"
            "  fi\n"
            "done\n"
            f'cat "{payload_path}"\n'
            f"exit {exit_code}\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        self.path = str(script)


class RecordingSink:
    """Collects everything the tee mirrors, with an optional per-write delay."""

    def __init__(self, delay: float = 0.0, fail_after: int | None = None) -> None:
        self.chunks: list[bytes] = []
        self.delay = delay
        self.fail_after = fail_after
        self.input_closed = False

    def write(self, chunk: bytes) -> None:
        if self.fail_after is not None and len(self.chunks) >= self.fail_after:
            raise BrokenPipeError("sink is gone")
        if self.delay:
            time.sleep(self.delay)
        self.chunks.append(chunk)

    def close_input(self) -> None:
        self.input_closed = True

    @property
    def body(self) -> bytes:
        return b"".join(self.chunks)


@pytest.fixture(scope="session")
def fragmented_source(tmp_path_factory) -> Path:
    """A synthetic fragmented mp4: what a remuxed ingest actually streams out.

    Fragmented because that is the only mp4 shape ffmpeg can read off an
    unseekable pipe, which is what the audio branch is handed.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    destination = tmp_path_factory.mktemp("fixture") / "fragmented.mp4"
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=15:duration={SAMPLE_DURATION:g}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={SAMPLE_DURATION:g}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "44100", "-ac", "2", "-shortest",
            "-movflags", "frag_keyframe+empty_moov+default_base_moof",
            "-f", "mp4", str(destination),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 or not destination.is_file():
        pytest.skip(f"could not build the mp4 fixture: {result.stderr.decode(errors='ignore')}")
    return destination


# --------------------------------------------------------------------------
# key derivation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("video_key", "expected"),
    [
        ("input/vid42/source.mp4", "processing/vid42/audio.m4a"),
        ("input/2866049874/source.mp4", "processing/2866049874/audio.m4a"),
        # A custom key still gets a companion under processing/.
        ("archive/twitch/clip.mp4", "processing/archive/twitch/audio.m4a"),
        ("input/source.mp4", "processing/audio.m4a"),
        ("source.mp4", "processing/audio.m4a"),
    ],
)
def test_audio_key_sits_beside_the_source_under_processing(video_key, expected):
    assert derive_audio_key(video_key) == expected


def test_ingest_refuses_an_audio_key_that_lands_on_the_source_video(tmp_path):
    client = KeyedFakeS3Client()

    with pytest.raises(ValueError, match="overwrite the source"):
        _ingest(
            tmp_path,
            client,
            b"x" * 32,
            extract_audio=True,
            audio_key="input/vid42/source.mp4",
        )

    assert client.created == {}


# --------------------------------------------------------------------------
# encoder settings and command
# --------------------------------------------------------------------------


def test_default_settings_are_speech_shaped_not_archival():
    settings = AudioEncodeSettings()
    assert (settings.channels, settings.sample_rate) == (1, 16000)
    assert settings.codec == "aac"
    assert settings.content_type == "audio/mp4"


@pytest.mark.parametrize("kwargs", [{"sample_rate": 0}, {"channels": 0}])
def test_settings_reject_impossible_encodes(kwargs):
    with pytest.raises(ValueError):
        AudioEncodeSettings(**kwargs)


def test_audio_command_keeps_the_source_timeline_and_drops_everything_but_speech():
    command = build_audio_command("/tmp/audio.m4a", AudioEncodeSettings())

    assert command[:1] == ["ffmpeg"]
    assert command[-1] == "/tmp/audio.m4a"
    assert "-i" in command and command[command.index("-i") + 1] == "pipe:0"
    # Mono at ASR sample rate, nothing else in the container.
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    for flag in ("-vn", "-sn", "-dn"):
        assert flag in command
    # Padding, never shifting: a late-starting track is aligned back to zero.
    assert command[command.index("-af") + 1] == "aresample=async=1:first_pts=0"
    # ffmpeg exits 0 on a demux fault otherwise.
    assert "-xerror" in command
    # Trimming the source would break alignment outright.
    assert "-ss" not in command and "-t" not in command


def test_audio_command_carries_configured_settings():
    command = build_audio_command(
        "/tmp/a.m4a",
        AudioEncodeSettings(codec="libopus", sample_rate=24000, channels=2, bitrate="32k"),
    )
    assert command[command.index("-c:a") + 1] == "libopus"
    assert command[command.index("-ar") + 1] == "24000"
    assert command[command.index("-ac") + 1] == "2"
    assert command[command.index("-b:a") + 1] == "32k"


# --------------------------------------------------------------------------
# completeness check
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "audio", "complains"),
    [
        (3600.0, 3600.0, False),
        (3600.0, 3599.0, False),      # inside the 1% band
        (3600.0, 3000.0, True),       # truncated
        (10.0, 11.5, False),          # inside the 2s floor
        (10.0, 20.0, True),
        (None, 12.0, False),          # nothing to compare against
        (0.0, 12.0, False),
        (3600.0, None, True),         # unreadable length is not a pass
    ],
)
def test_duration_mismatch_flags_only_a_real_drift(source, audio, complains):
    assert (duration_mismatch(source, audio) is not None) is complains


# --------------------------------------------------------------------------
# the tee
# --------------------------------------------------------------------------


def test_tee_hands_the_reader_every_byte_and_mirrors_the_same_bytes():
    payload = os.urandom(300_000)
    sink = RecordingSink()
    tee = TeeReader(io.BytesIO(payload), sink)

    read_back = b""
    while True:
        chunk = tee.read(4096)
        if not chunk:
            break
        read_back += chunk
    tee.wait_for_mirror()

    assert read_back == payload
    assert sink.body == payload
    assert sink.input_closed


def test_tee_throttles_instead_of_dropping_when_the_sink_lags():
    payload = os.urandom(200_000)
    # A budget far below the payload: the reader has to wait for the sink.
    sink = RecordingSink(delay=0.002)
    tee = TeeReader(io.BytesIO(payload), sink, buffer_bytes=8192)

    read_back = b""
    while True:
        chunk = tee.read(4096)
        if not chunk:
            break
        read_back += chunk
    tee.wait_for_mirror()

    assert read_back == payload
    assert sink.body == payload  # nothing was dropped to keep up


def test_tee_keeps_feeding_the_reader_after_the_sink_dies():
    payload = os.urandom(120_000)
    sink = RecordingSink(fail_after=2)
    tee = TeeReader(io.BytesIO(payload), sink, buffer_bytes=4096)

    read_back = b""
    while True:
        chunk = tee.read(4096)
        if not chunk:
            break
        read_back += chunk

    # The primary path is untouched; the failure surfaces only on the mirror.
    assert read_back == payload
    with pytest.raises(AudioArtifactError):
        tee.wait_for_mirror()


def test_tee_admits_a_chunk_larger_than_its_whole_budget():
    payload = os.urandom(50_000)
    sink = RecordingSink()
    tee = TeeReader(io.BytesIO(payload), sink, buffer_bytes=1024)

    assert tee.read(50_000) == payload
    assert tee.read(4096) == b""
    tee.wait_for_mirror()
    assert sink.body == payload


# --------------------------------------------------------------------------
# ffmpeg-backed extraction
# --------------------------------------------------------------------------


@needs_ffmpeg
def test_sidecar_extracts_mono_speech_audio_spanning_the_whole_source(tmp_path, fragmented_source):
    destination = tmp_path / "audio.m4a"
    sidecar = AudioSidecar(destination, AudioEncodeSettings())
    tee = TeeReader(fragmented_source.open("rb"), sidecar)

    while tee.read(64 * 1024):
        pass
    tee.wait_for_mirror()
    sidecar.wait()

    probed = probe_audio_file(destination)
    assert probed["channels"] == 1
    assert probed["sample_rate"] == 16000
    assert probed["codec"] == "aac"
    # Same length as the source, so a timestamp means the same thing in both.
    assert probed["duration_seconds"] == pytest.approx(SAMPLE_DURATION, abs=0.2)
    assert duration_mismatch(SAMPLE_DURATION, probed["duration_seconds"]) is None
    # Speech-shaped, not archival: far smaller than the video it came from.
    assert destination.stat().st_size < fragmented_source.stat().st_size


@needs_ffmpeg
def test_sidecar_fails_loudly_on_a_stream_it_cannot_demux(tmp_path):
    destination = tmp_path / "audio.m4a"
    sidecar = AudioSidecar(destination, AudioEncodeSettings())
    try:
        try:
            sidecar.write(os.urandom(256 * 1024))
        except BrokenPipeError:
            pass  # ffmpeg may already have given up
        with pytest.raises(AudioArtifactError):
            sidecar.wait()
    finally:
        sidecar.terminate()


def test_sidecar_reports_a_missing_ffmpeg_instead_of_crashing(tmp_path):
    with pytest.raises(AudioArtifactError, match="not found"):
        AudioSidecar(tmp_path / "a.m4a", AudioEncodeSettings(), ffmpeg_path="/nonexistent/ffmpeg")


@needs_ffmpeg
def test_probe_rejects_a_file_with_no_audio_stream(tmp_path):
    empty = tmp_path / "empty.m4a"
    empty.write_bytes(b"not an audio file")
    with pytest.raises(AudioArtifactError):
        probe_audio_file(empty)


# --------------------------------------------------------------------------
# ingest end to end
# --------------------------------------------------------------------------


def _ingest(tmp_path, client, payload, *, info=None, exit_code=0, **kwargs):
    ytdlp = FakeYtDlp(
        tmp_path,
        payload,
        info or {
            "id": "vid42", "title": "Demo", "duration": SAMPLE_DURATION,
            "extractor_key": "Twitch", "format_id": "1080p60", "protocol": "https",
        },
        exit_code=exit_code,
    )
    return ingest_to_r2(
        "https://example.com/vid42",
        client=client,
        bucket="freecher",
        quality=QUALITY_PROGRESSIVE,
        part_size=PART_SIZE,
        ytdlp_path=ytdlp.path,
        **kwargs,
    )


def test_ingest_without_the_flag_still_writes_only_the_video(tmp_path):
    client = KeyedFakeS3Client()

    result = _ingest(tmp_path, client, os.urandom(PART_SIZE + 64))

    assert result.audio is None and result.audio_error is None
    assert list(client.created) == ["input/vid42/source.mp4"]


@needs_ffmpeg
def test_ingest_writes_the_audio_artifact_beside_the_source(tmp_path, fragmented_source):
    payload = fragmented_source.read_bytes()
    client = KeyedFakeS3Client()

    result = _ingest(tmp_path, client, payload, extract_audio=True)

    video_key = "input/vid42/source.mp4"
    audio_key = "processing/vid42/audio.m4a"
    assert result.key == video_key
    assert sorted(client.completed) == [video_key, audio_key]
    # The source object is still byte-for-byte what yt-dlp produced.
    assert client.body(video_key) == payload

    artifact = result.audio
    assert artifact is not None and not artifact.skipped
    assert artifact.key == audio_key
    assert artifact.source_key == video_key
    assert (artifact.codec, artifact.sample_rate, artifact.channels) == ("aac", 16000, 1)
    assert artifact.audio_duration_seconds == pytest.approx(SAMPLE_DURATION, abs=0.2)
    assert artifact.size_bytes == len(client.body(audio_key))
    assert artifact.size_bytes < len(payload)

    # The uploaded bytes really are a playable, aligned audio file.
    written = tmp_path / "roundtrip.m4a"
    written.write_bytes(client.body(audio_key))
    probed = probe_audio_file(written)
    assert probed["duration_seconds"] == pytest.approx(SAMPLE_DURATION, abs=0.2)
    assert (probed["codec"], probed["sample_rate"], probed["channels"]) == ("aac", 16000, 1)


@needs_ffmpeg
def test_audio_object_carries_the_pairing_and_the_encode_in_its_metadata(tmp_path, fragmented_source):
    client = KeyedFakeS3Client()

    _ingest(tmp_path, client, fragmented_source.read_bytes(), extract_audio=True)

    created = client.created["processing/vid42/audio.m4a"]
    metadata = created["Metadata"]
    assert created["ContentType"] == "audio/mp4"
    assert metadata["artifact"] == "audio"
    # Everything needed to find the video this was cut from.
    assert metadata["source-key"] == "input/vid42/source.mp4"
    assert metadata["source-bucket"] == "freecher"
    assert metadata["video-id"] == "vid42"
    # ... how long that source runs ...
    assert float(metadata["source-duration-seconds"]) == pytest.approx(SAMPLE_DURATION, abs=0.2)
    # ... and exactly how the audio was encoded.
    assert metadata["audio-codec"] == "aac"
    assert metadata["audio-sample-rate"] == "16000"
    assert metadata["audio-channels"] == "1"
    assert float(metadata["audio-duration-seconds"]) == pytest.approx(SAMPLE_DURATION, abs=0.2)
    # R2 rejects a non-ASCII metadata header outright.
    assert all(value.isascii() for value in metadata.values())


@needs_ffmpeg
def test_neither_object_is_published_publicly(tmp_path, fragmented_source):
    client = KeyedFakeS3Client()

    result = _ingest(tmp_path, client, fragmented_source.read_bytes(), extract_audio=True)

    for created in client.created.values():
        assert "ACL" not in created
        assert "GrantRead" not in created
        assert not any("public" in str(value).lower() for value in created.values())
    # No readable URL is invented for the audio artifact either.
    assert not hasattr(result.audio, "public_url")


@needs_ffmpeg
def test_a_second_ingest_leaves_an_existing_audio_artifact_alone(tmp_path, fragmented_source):
    client = KeyedFakeS3Client(existing_keys={"processing/vid42/audio.m4a"})

    result = _ingest(tmp_path, client, fragmented_source.read_bytes(), extract_audio=True)

    artifact = result.audio
    assert artifact is not None and artifact.skipped
    assert artifact.key == "processing/vid42/audio.m4a"
    # Nothing was re-encoded or rewritten; only the video object was created.
    assert list(client.created) == ["input/vid42/source.mp4"]
    assert client.completed == ["input/vid42/source.mp4"]


@needs_ffmpeg
def test_overwrite_replaces_the_audio_artifact_too(tmp_path, fragmented_source):
    client = KeyedFakeS3Client(
        existing_keys={"input/vid42/source.mp4", "processing/vid42/audio.m4a"}
    )

    result = _ingest(tmp_path, client, fragmented_source.read_bytes(), extract_audio=True, overwrite=True)

    assert result.audio is not None and not result.audio.skipped
    assert sorted(client.completed) == ["input/vid42/source.mp4", "processing/vid42/audio.m4a"]


def test_a_failed_audio_branch_never_costs_the_source_video(tmp_path):
    # Random bytes are not a container ffmpeg can demux.
    payload = os.urandom(PART_SIZE + 4096)
    client = KeyedFakeS3Client()

    result = _ingest(tmp_path, client, payload, extract_audio=True)

    assert result.key == "input/vid42/source.mp4"
    assert client.completed == ["input/vid42/source.mp4"]
    assert client.body("input/vid42/source.mp4") == payload
    assert result.audio is None
    assert result.audio_error
    assert any("audio artifact was not produced" in w for w in result.warnings)


def test_a_missing_ffmpeg_downgrades_to_a_warning_not_a_failed_ingest(tmp_path):
    payload = os.urandom(PART_SIZE + 16)
    client = KeyedFakeS3Client()

    result = _ingest(tmp_path, client, payload, extract_audio=True, ffmpeg_path="/nonexistent/ffmpeg")

    assert client.completed == ["input/vid42/source.mp4"]
    assert result.audio is None
    assert "not found" in (result.audio_error or "")


@needs_ffmpeg
def test_a_dead_source_publishes_neither_object(tmp_path, fragmented_source):
    client = KeyedFakeS3Client()

    with pytest.raises(Exception, match="yt-dlp exited with code 1"):
        _ingest(tmp_path, client, fragmented_source.read_bytes(), extract_audio=True, exit_code=1)

    assert client.completed == []
    assert client.aborted == ["input/vid42/source.mp4"]
    assert "processing/vid42/audio.m4a" not in client.created


@needs_ffmpeg
def test_an_explicit_audio_key_is_honoured(tmp_path, fragmented_source):
    client = KeyedFakeS3Client()

    result = _ingest(
        tmp_path,
        client,
        fragmented_source.read_bytes(),
        extract_audio=True,
        audio_key="processing/custom/speech.m4a",
    )

    assert result.audio is not None
    assert result.audio.key == "processing/custom/speech.m4a"
    assert "processing/custom/speech.m4a" in client.completed


@needs_ffmpeg
def test_nothing_but_the_audio_is_left_on_local_disk(tmp_path, fragmented_source):
    staging = tmp_path / "staging"
    staging.mkdir()
    client = KeyedFakeS3Client()

    _ingest(
        tmp_path,
        client,
        fragmented_source.read_bytes(),
        extract_audio=True,
        audio_staging_dir=str(staging),
    )

    # The staged audio is removed once it is in R2; the video never touched disk.
    assert list(staging.iterdir()) == []
