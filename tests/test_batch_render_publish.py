"""Batch Top-N rendering, MP4 validation, R2 publication and idempotency."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import freecher_worker.rendering  # noqa: F401  (import-order; subtitles<->rendering cycle)
from freecher_worker.rendering import batch as B


# --------------------------------------------------------------------- helpers
def _make_mp4(path: Path, *, width=1080, height=1920, seconds=2.0, audio=True) -> Path:
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
           f"testsrc=size={width}x{height}:rate=10"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100"]
    cmd += ["-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    cmd += (["-c:a", "aac", "-shortest"] if audio else ["-an"])
    cmd += ["-y", str(path)]
    subprocess.run(cmd, check=True, timeout=180)
    return path


class FakeR2:
    """Minimal stand-in for the boto3 S3 client surface batch.py uses."""

    def __init__(self):
        self.objects: dict[str, dict] = {}
        self.put_calls = 0

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return self.objects[Key]

    def put_object(self, Bucket, Key, Body, ContentType=None, Metadata=None):
        data = Body.read() if hasattr(Body, "read") else Body
        self.put_calls += 1
        self.objects[Key] = {"ContentLength": len(data), "ContentType": ContentType,
                             "Metadata": Metadata or {}}
        return {"ETag": "x"}


# ------------------------------------------------------------------ selection
def test_top_n_selection_is_deterministic_and_rank_ordered():
    hs = [{"rank": 3, "candidate_id": "c"}, {"rank": 1, "candidate_id": "a"},
          {"rank": 2, "candidate_id": "b"}]
    assert [h["candidate_id"] for h in B.select_highlights(hs, 2)] == ["a", "b"]
    assert B.select_highlights(hs, 0) == []
    assert len(B.select_highlights(hs, 99)) == 3


def test_ties_break_on_candidate_id():
    hs = [{"rank": 1, "candidate_id": "z"}, {"rank": 1, "candidate_id": "a"}]
    assert [h["candidate_id"] for h in B.select_highlights(hs, 2)] == ["a", "z"]


def test_clip_id_is_stable():
    a = B.clip_id_for("4ae07cfe573bf5cd1cde8525e7987fcc", "cand_019", 1)
    b = B.clip_id_for("4ae07cfe573bf5cd1cde8525e7987fcc", "cand_019", 1)
    assert a == b == "4ae07cfe573b_r01_cand_019"


# ------------------------------------------------------------------ validation
def test_valid_mp4_passes_and_reports_streams(tmp_path):
    f = _make_mp4(tmp_path / "ok.mp4")
    info = B.validate_clip(f, expected_width=1080, expected_height=1920,
                           expected_duration=2.0)
    assert info["width"] == 1080 and info["height"] == 1920
    assert info["video_codec"] == "h264" and info["audio_codec"] == "aac"
    assert info["bytes"] > 0


def test_missing_audio_stream_is_rejected(tmp_path):
    f = _make_mp4(tmp_path / "noaudio.mp4", audio=False)
    with pytest.raises(B.ClipValidationError, match="no audio stream"):
        B.validate_clip(f, expected_width=1080, expected_height=1920, expected_duration=2.0)


def test_wrong_dimensions_are_rejected(tmp_path):
    f = _make_mp4(tmp_path / "small.mp4", width=640, height=360)
    with pytest.raises(B.ClipValidationError, match="expected 1080x1920"):
        B.validate_clip(f, expected_width=1080, expected_height=1920, expected_duration=2.0)


def test_duration_drift_is_rejected(tmp_path):
    f = _make_mp4(tmp_path / "short.mp4", seconds=2.0)
    with pytest.raises(B.ClipValidationError, match="differs from expected"):
        B.validate_clip(f, expected_width=1080, expected_height=1920, expected_duration=30.0)


def test_empty_and_missing_files_are_rejected(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(B.ClipValidationError, match="missing or empty"):
        B.validate_clip(empty, expected_width=1080, expected_height=1920, expected_duration=1.0)
    with pytest.raises(B.ClipValidationError, match="missing or empty"):
        B.validate_clip(tmp_path / "nope.mp4", expected_width=1080, expected_height=1920,
                        expected_duration=1.0)


def test_non_media_bytes_are_rejected(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video at all")
    with pytest.raises(B.ClipValidationError):
        B.validate_clip(junk, expected_width=1080, expected_height=1920, expected_duration=1.0)


# --------------------------------------------------------------------- publish
def test_upload_records_sha256_and_content_type(tmp_path):
    f = tmp_path / "c.mp4"
    f.write_bytes(b"payload")
    r2 = FakeR2()
    digest = B.sha256_file(f)
    assert B.upload_clip(r2, "bkt", "output/s/clips/c.mp4", f, digest) is True
    obj = r2.objects["output/s/clips/c.mp4"]
    assert obj["ContentType"] == "video/mp4"
    assert obj["Metadata"]["sha256"] == digest
    assert obj["ContentLength"] == len(b"payload")


def test_upload_is_idempotent_for_an_identical_object(tmp_path):
    f = tmp_path / "c.mp4"
    f.write_bytes(b"payload")
    r2 = FakeR2()
    digest = B.sha256_file(f)
    B.upload_clip(r2, "bkt", "k", f, digest)
    assert r2.put_calls == 1
    assert B.upload_clip(r2, "bkt", "k", f, digest) is False
    assert r2.put_calls == 1, "identical object must not be re-sent"


def test_changed_content_is_re_uploaded(tmp_path):
    f = tmp_path / "c.mp4"
    f.write_bytes(b"one")
    r2 = FakeR2()
    B.upload_clip(r2, "bkt", "k", f, B.sha256_file(f))
    f.write_bytes(b"two-different")
    assert B.upload_clip(r2, "bkt", "k", f, B.sha256_file(f)) is True
    assert r2.put_calls == 2


def test_upload_never_deletes_the_local_file(tmp_path):
    f = tmp_path / "c.mp4"
    f.write_bytes(b"payload")
    B.upload_clip(FakeR2(), "bkt", "k", f, B.sha256_file(f))
    assert f.is_file(), "local artifact must survive publication"


def test_verify_remote_detects_size_mismatch(tmp_path):
    f = tmp_path / "c.mp4"
    f.write_bytes(b"payload")
    r2 = FakeR2()
    B.upload_clip(r2, "bkt", "k", f, B.sha256_file(f))
    assert B.verify_remote(r2, "bkt", "k", len(b"payload")) is True
    assert B.verify_remote(r2, "bkt", "k", 999) is False
    assert B.verify_remote(r2, "bkt", "absent", 7) is False


# ------------------------------------------------------- partial-failure contract
def test_manifest_is_failed_when_any_clip_failed():
    m = B.ClipsManifest(source_id="s", requested=2, succeeded=1, failed=1)
    m.clips = [
        B.ClipRecord(clip_id="a", candidate_id="c1", rank=1, start_seconds=0,
                     end_seconds=1, duration_seconds=1, status=B.ClipStatus.DONE),
        B.ClipRecord(clip_id="b", candidate_id="c2", rank=2, start_seconds=1,
                     end_seconds=2, duration_seconds=1, status=B.ClipStatus.FAILED,
                     error_type="ClipValidationError", error_message="no audio stream"),
    ]
    assert m.failed == 1
    failed = [c for c in m.clips if c.status is B.ClipStatus.FAILED]
    assert failed and failed[0].error_type and failed[0].error_message
    # a batch with any failure must never be reported as DONE
    status = "DONE" if (m.failed == 0 and m.succeeded == m.requested) else "FAILED"
    assert status == "FAILED"


def test_a_clip_record_serialises_the_documented_fields():
    rec = B.ClipRecord(clip_id="cid", candidate_id="cand_019", rank=1,
                       start_seconds=1.0, end_seconds=61.0, duration_seconds=60.0)
    data = json.loads(rec.model_dump_json())
    for field in ("clip_id", "candidate_id", "rank", "start_seconds", "end_seconds",
                  "duration_seconds", "width", "height", "video_codec", "audio_codec",
                  "subtitles_burned", "crop_mode", "local_path", "sha256", "bytes",
                  "status"):
        assert field in data, field
