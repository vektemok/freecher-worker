"""Tests for R2-backed highlight discovery (transcript -> ranked highlights)."""

from __future__ import annotations

import io
import itertools
import json
import random

import pytest

from freecher_worker.highlights.dedup import calculate_overlap_ratio
from freecher_worker.highlights.models import CandidateDocument, Highlight, HighlightScore
from freecher_worker.highlights.r2 import (
    DiscoveryResult,
    HighlightDiscoveryError,
    candidates_key_for,
    discover_from_r2,
    highlights_key_for,
    load_transcript_from_r2,
    manifest_key_for,
    serialize_document,
)
from freecher_worker.highlights.segmenter import generate_candidate_windows
from freecher_worker.media.fingerprint import (
    LOCAL_FILE_SOURCE,
    R2_TRANSCRIPT_SOURCE,
    compute_r2_transcript_fingerprint,
)
from freecher_worker.pipeline.processor import (
    LOCAL_VIDEO_RUN,
    R2_TRANSCRIPT_RUN,
    Manifest,
)
from freecher_worker.scoring.base import HighlightScorer
from freecher_worker.transcription.models import Transcript, TranscriptSegment

SOURCE_ID = "v2866049874"
BUCKET = "freecher"
TRANSCRIPT_KEY = "processing/v2866049874/transcript.json"
CANDIDATES_KEY = "processing/v2866049874/candidates.json"
HIGHLIGHTS_KEY = "processing/v2866049874/highlights.json"
MANIFEST_KEY = "processing/v2866049874/manifest.json"


class FakeS3Client:
    """In-memory object store with the three calls discovery uses."""

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects: dict[str, bytes] = dict(objects or {})
        self.puts: list[dict] = []
        self.get_calls: list[str] = []

    def head_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise _missing_key_error()
        return {"ContentLength": len(self.objects[Key])}

    def get_object(self, *, Bucket, Key):
        self.get_calls.append(Key)
        if Key not in self.objects:
            raise _missing_key_error()
        return {"Body": io.BytesIO(self.objects[Key]), "ContentLength": len(self.objects[Key])}

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        self.objects[kwargs["Key"]] = kwargs["Body"]
        return {"ETag": '"etag"'}

    @property
    def put_keys(self) -> list[str]:
        return [put["Key"] for put in self.puts]


def _missing_key_error() -> Exception:
    error = Exception("Not Found")
    error.response = {"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}
    return error


def build_transcript(
    *, total_seconds: float = 600.0, segment_count: int = 260, seed: int = 7
) -> Transcript:
    """A transcript shaped like a real stream: many short, varied segments."""
    rng = random.Random(seed)
    phrases = [
        "почему это вообще работает", "смотрите что сейчас будет", "и вот тут секрет",
        "это невероятно круто получилось", "потому что результат был другой",
        "ну короче я не знаю", "представьте себе такую картину",
        "статистика говорит о трёх миллионах", "вау это просто шок",
        "а потом всё пошло не так",
    ]
    cursor, segments = 0.0, []
    for index in range(segment_count):
        duration = max(0.4, rng.gauss(total_seconds / segment_count, 0.7))
        segments.append(
            TranscriptSegment(
                id=index,
                start=round(cursor, 3),
                end=round(cursor + duration, 3),
                text=rng.choice(phrases),
                avg_logprob=-0.2,
                no_speech_prob=0.01,
            )
        )
        cursor += duration
    scale = total_seconds / cursor
    for segment in segments:
        segment.start = round(segment.start * scale, 3)
        segment.end = round(segment.end * scale, 3)

    return Transcript(
        language="ru",
        language_probability=0.85,
        duration=total_seconds,
        model="large-v3",
        compute_type="float16",
        device="cuda",
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        segments=segments,
        source_bucket=BUCKET,
        source_audio_key="processing/v2866049874/audio.m4a",
        processing_seconds=991.3,
    )


@pytest.fixture
def transcript() -> Transcript:
    return build_transcript()


@pytest.fixture
def client(transcript) -> FakeS3Client:
    return FakeS3Client({TRANSCRIPT_KEY: serialize_document(transcript)})


class CountingScorer(HighlightScorer):
    """Scores by position so the expected ranking is known exactly."""

    def __init__(self) -> None:
        self.name = "counting"
        self.version = "counting_v1"
        self.seen: list[str] = []

    def score(self, candidate, context=None):
        self.seen.append(candidate.id)
        # Later candidates score higher, so the tail should win.
        value = min(100.0, float(len(self.seen)))
        return HighlightScore(
            score=value, hook_score=value, standalone_score=value, emotion_score=value,
            information_score=value, shareability_score=value, reason="counting",
        )


# --------------------------------------------------------------------------
# keys and identity
# --------------------------------------------------------------------------


def test_artifact_keys_sit_beside_the_transcript():
    assert candidates_key_for(SOURCE_ID) == CANDIDATES_KEY
    assert highlights_key_for(SOURCE_ID) == HIGHLIGHTS_KEY
    assert manifest_key_for(SOURCE_ID) == MANIFEST_KEY
    assert len({candidates_key_for(SOURCE_ID), highlights_key_for(SOURCE_ID),
                manifest_key_for(SOURCE_ID), TRANSCRIPT_KEY}) == 4


@pytest.mark.parametrize("source_id", ["", "  ", "a/b", "/"])
def test_a_source_id_must_be_a_single_path_segment(source_id):
    with pytest.raises(ValueError):
        candidates_key_for(source_id)


def test_run_identity_comes_from_remote_identity_not_a_local_file():
    fingerprint = compute_r2_transcript_fingerprint(
        source_id=SOURCE_ID, bucket=BUCKET, transcript_key=TRANSCRIPT_KEY,
        transcript_hash="abc123", duration_seconds=6817.878, transcript_bytes=2530406,
    )

    assert fingerprint.kind == R2_TRANSCRIPT_SOURCE
    assert fingerprint.has_local_video is False
    # The path is the object's real URI, not an invented local file.
    assert fingerprint.path == f"s3://{BUCKET}/{TRANSCRIPT_KEY}"
    assert not fingerprint.path.startswith("/")
    # Local-only facts are absent rather than faked.
    assert fingerprint.mtime_ns is None
    assert fingerprint.content_hash == "abc123"
    assert (fingerprint.source_id, fingerprint.bucket) == (SOURCE_ID, BUCKET)


def test_run_identity_is_deterministic_and_tracks_the_transcript():
    def build(transcript_hash: str, key: str = TRANSCRIPT_KEY):
        return compute_r2_transcript_fingerprint(
            source_id=SOURCE_ID, bucket=BUCKET, transcript_key=key,
            transcript_hash=transcript_hash, duration_seconds=100.0, transcript_bytes=10,
        ).fingerprint_id

    # Same inputs, same id, every time.
    assert build("hash-a") == build("hash-a")
    # A different transcript is a different run...
    assert build("hash-a") != build("hash-b")
    # ... and so is the same transcript in a different place.
    assert build("hash-a") != build("hash-a", "processing/other/transcript.json")


def test_a_local_fingerprint_still_declares_itself_local(tmp_path):
    from freecher_worker.media.fingerprint import compute_source_fingerprint

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x" * 4096)
    fingerprint = compute_source_fingerprint(video, 12.0)

    assert fingerprint.kind == LOCAL_FILE_SOURCE
    assert fingerprint.has_local_video is True
    assert fingerprint.mtime_ns is not None


# --------------------------------------------------------------------------
# loading the transcript
# --------------------------------------------------------------------------


def test_the_transcript_is_loaded_and_measured(client, transcript):
    loaded, size = load_transcript_from_r2(client, BUCKET, TRANSCRIPT_KEY)

    assert len(loaded.segments) == len(transcript.segments)
    assert loaded.language == "ru"
    assert size == len(client.objects[TRANSCRIPT_KEY])


def test_a_missing_transcript_is_a_hard_stop():
    with pytest.raises(HighlightDiscoveryError, match="could not read"):
        load_transcript_from_r2(FakeS3Client(), BUCKET, TRANSCRIPT_KEY)


def test_an_unparseable_transcript_is_a_hard_stop_not_a_silent_skip():
    # Unlike the output artifacts, the input must never be treated as absent.
    client = FakeS3Client({TRANSCRIPT_KEY: b"{ not json"})
    with pytest.raises(HighlightDiscoveryError, match="not a valid transcript"):
        load_transcript_from_r2(client, BUCKET, TRANSCRIPT_KEY)


def test_a_transcript_with_no_segments_is_refused():
    empty = build_transcript(segment_count=1)
    empty.segments = []
    client = FakeS3Client({TRANSCRIPT_KEY: serialize_document(empty)})
    with pytest.raises(HighlightDiscoveryError, match="no segments"):
        load_transcript_from_r2(client, BUCKET, TRANSCRIPT_KEY)


# --------------------------------------------------------------------------
# the workflow
# --------------------------------------------------------------------------


def test_discovery_writes_all_three_artifacts_with_the_manifest_last(client):
    result = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)

    # The manifest is the completion marker, so it must land after the rest.
    assert client.put_keys == [CANDIDATES_KEY, HIGHLIGHTS_KEY, MANIFEST_KEY]
    assert result.candidates and result.highlights
    assert all(put["ContentType"] == "application/json" for put in client.puts)


def test_the_source_video_is_never_fetched(client):
    client.objects["input/v2866049874/source.mp4"] = b"the enormous video"

    discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)

    assert "input/v2866049874/source.mp4" not in client.get_calls
    # Only the transcript and the idempotency probes were read.
    assert set(client.get_calls) <= {TRANSCRIPT_KEY, CANDIDATES_KEY, HIGHLIGHTS_KEY, MANIFEST_KEY}


def test_candidates_match_the_generator_run_directly(client, transcript):
    result = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)

    # The workflow must not fork the algorithm: same input, same windows.
    expected = generate_candidate_windows(transcript, 30.0, 60.0, 90.0, 15.0)
    assert [c.model_dump() for c in result.candidates] == [c.model_dump() for c in expected]


def test_candidate_windows_respect_the_configured_bounds(client):
    result = discover_from_r2(
        SOURCE_ID, client=client, bucket=BUCKET,
        min_seconds=20.0, target_seconds=40.0, max_seconds=60.0, overlap_seconds=10.0,
    )

    durations = [c.duration for c in result.candidates]
    assert durations
    assert max(durations) <= 60.0
    # Every window but a trailing remainder clears the minimum.
    assert all(d >= 20.0 for d in durations[:-1])
    # Windows advance in time and stay inside the transcript.
    starts = [c.start for c in result.candidates]
    assert starts == sorted(starts)


def test_the_candidate_set_id_is_deterministic_and_configuration_aware(client):
    first = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)
    again = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET, overwrite=True)
    assert first.candidate_set_id == again.candidate_set_id

    changed = discover_from_r2(
        SOURCE_ID, client=client, bucket=BUCKET, overwrite=True, target_seconds=45.0
    )
    assert changed.candidate_set_id != first.candidate_set_id


def test_highlights_are_ranked_deduplicated_and_capped(client):
    result = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET, top_k=5)

    assert len(result.highlights) <= 5
    assert [h.rank for h in result.highlights] == list(range(1, len(result.highlights) + 1))
    scores = [h.score for h in result.highlights]
    assert scores == sorted(scores, reverse=True)

    # Deduplication bounds how much any two selected highlights may share.
    # They are ordered by score, not by time, and partial overlap below the
    # threshold is allowed by design, so the invariant is the ratio itself.
    by_id = {c.id: c for c in result.candidates}
    for first, second in itertools.combinations(result.highlights, 2):
        ratio = calculate_overlap_ratio(by_id[first.candidate_id], by_id[second.candidate_id])
        assert ratio < 0.60, f"{first.candidate_id} and {second.candidate_id} overlap {ratio:.2f}"


def test_a_custom_scorer_is_used_for_every_candidate(client):
    scorer = CountingScorer()

    result = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET, scorer=scorer)

    assert scorer.seen == [c.id for c in result.candidates]
    assert result.scorer_version == "counting_v1"
    assert result.manifest.scoring.scorer == "counting"


def test_the_default_scorer_is_heuristic_v1_with_no_llm(client):
    result = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)

    assert result.scorer_version == "heuristic_v1"
    assert result.manifest.scoring.llm_model is None
    assert result.manifest.scoring.fallback_used is False


# --------------------------------------------------------------------------
# the synthesized manifest
# --------------------------------------------------------------------------


def test_the_manifest_says_plainly_that_there_is_no_local_video(client):
    result = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)
    manifest = result.manifest

    assert manifest.run_kind == R2_TRANSCRIPT_RUN
    assert manifest.source_video_available is False
    assert manifest.source == f"s3://{BUCKET}/{TRANSCRIPT_KEY}"
    assert manifest.source_fingerprint.kind == R2_TRANSCRIPT_SOURCE
    # Nothing was clipped, so no highlight claims a file.
    assert all(item.file is None for item in manifest.highlights)
    # Stages that never ran report no time rather than an invented one.
    assert manifest.timings.probe_seconds == 0.0
    assert manifest.timings.audio_seconds == 0.0
    assert manifest.timings.clipping_seconds == 0.0


def test_the_manifest_carries_the_asr_settings_the_transcript_records(client, transcript):
    manifest = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET).manifest

    assert manifest.asr.model == "large-v3"
    assert manifest.asr.device == "cuda"
    assert manifest.asr.compute_type == "float16"
    assert manifest.asr.language == "ru"
    # Transcription time is carried over, not re-measured or zeroed.
    assert manifest.timings.transcription_seconds == transcript.processing_seconds


def test_the_manifest_records_where_every_artifact_lives(client):
    manifest = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET).manifest

    assert manifest.r2 is not None
    assert manifest.r2.bucket == BUCKET
    assert manifest.r2.source_id == SOURCE_ID
    assert manifest.r2.transcript_key == TRANSCRIPT_KEY
    assert manifest.r2.candidates_key == CANDIDATES_KEY
    assert manifest.r2.highlights_key == HIGHLIGHTS_KEY
    assert manifest.r2.manifest_key == MANIFEST_KEY


def test_the_manifest_statistics_describe_the_run(client, transcript):
    result = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)
    stats = result.manifest.statistics

    assert stats.transcript_segment_count == len(transcript.segments)
    assert stats.candidate_count == len(result.candidates)
    assert stats.selected_highlight_count == len(result.highlights)


def test_a_manifest_written_by_the_local_pipeline_still_validates():
    # The new fields are additive, so an older manifest keeps its meaning.
    legacy = {
        "pipeline_version": "0.2.0", "created_at": "2026-01-01T00:00:00", "source": "/tmp/a.mp4",
        "source_fingerprint": {
            "path": "/tmp/a.mp4", "file_size": 10, "mtime_ns": 5,
            "duration_seconds": 12.0, "content_hash": "h", "fingerprint_id": "fp",
        },
        "environment": {"python_version": "3.12.0", "platform": "test"},
        "asr": {"model": "small", "device": "cpu", "compute_type": "int8"},
        "candidate_config": {"min_seconds": 30, "target_seconds": 60, "max_seconds": 90, "overlap": 15},
        "scoring": {"scorer": "heuristic", "scorer_version": "heuristic_v1"},
        "ranking": {"top_k": 5, "dedup_threshold": 0.6},
        "timings": {}, "statistics": {}, "highlights": [],
    }
    manifest = Manifest.model_validate(legacy)

    assert manifest.run_kind == LOCAL_VIDEO_RUN
    assert manifest.source_video_available is True
    assert manifest.source_fingerprint.kind == LOCAL_FILE_SOURCE
    assert manifest.r2 is None


# --------------------------------------------------------------------------
# idempotency and atomicity
# --------------------------------------------------------------------------


def test_a_complete_artifact_set_is_left_alone(client):
    discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)
    client.puts.clear()

    again = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)

    assert again.skipped
    assert client.puts == []
    assert again.candidates and again.highlights
    assert again.manifest is not None


def test_overwrite_rebuilds_a_complete_set(client):
    discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)
    client.puts.clear()

    again = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET, overwrite=True)

    assert not again.skipped
    assert client.put_keys == [CANDIDATES_KEY, HIGHLIGHTS_KEY, MANIFEST_KEY]


@pytest.mark.parametrize("missing", [CANDIDATES_KEY, HIGHLIGHTS_KEY, MANIFEST_KEY])
def test_a_partial_set_is_rebuilt_rather_than_trusted(client, missing):
    discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)
    del client.objects[missing]
    client.puts.clear()

    result = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)

    assert not result.skipped
    assert client.put_keys == [CANDIDATES_KEY, HIGHLIGHTS_KEY, MANIFEST_KEY]


def test_an_unreadable_artifact_does_not_block_a_rerun(client):
    discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)
    client.objects[MANIFEST_KEY] = b"{ corrupted"
    client.puts.clear()

    result = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)

    assert not result.skipped
    assert MANIFEST_KEY in client.put_keys


def test_a_missing_transcript_publishes_nothing():
    empty = FakeS3Client()

    with pytest.raises(HighlightDiscoveryError, match="run transcribe first"):
        discover_from_r2(SOURCE_ID, client=empty, bucket=BUCKET)

    assert empty.puts == []


def test_a_failing_scorer_publishes_nothing(client):
    class Exploding(HighlightScorer):
        name, version = "boom", "boom_v1"

        def score(self, candidate, context=None):
            raise RuntimeError("scorer exploded")

    with pytest.raises(RuntimeError, match="exploded"):
        discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET, scorer=Exploding())

    # No half-finished artifact set is left for a later run to trust.
    assert client.puts == []
    assert MANIFEST_KEY not in client.objects


# --------------------------------------------------------------------------
# serialization and downstream compatibility
# --------------------------------------------------------------------------


def test_the_artifacts_serialize_deterministically(client):
    first = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)
    bodies = {put["Key"]: put["Body"] for put in client.puts}
    client.puts.clear()

    discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET, overwrite=True)
    again = {put["Key"]: put["Body"] for put in client.puts}

    # Candidates depend only on the transcript, so they are byte-identical.
    assert bodies[CANDIDATES_KEY] == again[CANDIDATES_KEY]
    assert bodies[HIGHLIGHTS_KEY] == again[HIGHLIGHTS_KEY]
    # The manifest differs only by its timestamp and timings.
    assert first.candidate_set_id


def test_the_stored_artifacts_round_trip_through_their_models(client):
    discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)

    candidates = CandidateDocument.model_validate(json.loads(client.objects[CANDIDATES_KEY]))
    highlights = [Highlight.model_validate(h) for h in json.loads(client.objects[HIGHLIGHTS_KEY])]
    manifest = Manifest.model_validate(json.loads(client.objects[MANIFEST_KEY]))

    assert candidates.candidate_set_id == manifest.candidate_config.candidate_set_id
    assert len(highlights) == manifest.statistics.selected_highlight_count
    assert {h.candidate_id for h in highlights} <= {c.id for c in candidates.candidates}


def test_non_ascii_transcript_text_survives_into_the_artifacts(client):
    discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)

    body = client.objects[CANDIDATES_KEY].decode("utf-8")
    assert "почему" in body


def test_the_artifact_metadata_identifies_the_run(client):
    discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)

    for put in client.puts:
        metadata = put["Metadata"]
        assert metadata["source-id"] == SOURCE_ID
        assert metadata["run-kind"] == R2_TRANSCRIPT_RUN
        assert metadata["scorer-version"] == "heuristic_v1"
        assert all(value.isascii() for value in metadata.values())
    assert [put["Metadata"]["artifact"] for put in client.puts] == [
        "candidates", "highlights", "manifest",
    ]


def test_mirroring_locally_produces_a_run_directory_inspect_can_read(client, tmp_path):
    from freecher_worker.cli import _resolve_run_path

    run_dir = tmp_path / "run"
    discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET, local_dir=run_dir)

    # The layout process() produces, minus everything derived from video.
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "candidates.json").is_file()
    assert (run_dir / "highlights.json").is_file()
    assert (run_dir / "transcript.json").is_file()
    assert not (run_dir / "audio.wav").exists()
    # inspect and export-eval both resolve a run through this gate.
    assert _resolve_run_path(run_dir) == run_dir


def test_candidate_duration_summary_matches_the_candidates(client):
    result = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)
    low, mean, high = result.candidate_durations

    durations = [c.duration for c in result.candidates]
    assert low == min(durations)
    assert high == max(durations)
    assert mean == pytest.approx(sum(durations) / len(durations))


def test_an_empty_result_reports_zeroed_durations():
    assert DiscoveryResult(
        bucket=BUCKET, source_id=SOURCE_ID, transcript_key=TRANSCRIPT_KEY,
        candidates_key=CANDIDATES_KEY, highlights_key=HIGHLIGHTS_KEY, manifest_key=MANIFEST_KEY,
    ).candidate_durations == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# mirroring is reported from what happened, not what was asked for
# --------------------------------------------------------------------------


def test_a_skipped_run_still_writes_the_local_mirror(client, tmp_path):
    # Skipping the work must not skip the mirror: whether R2 already held the
    # artifacts is beside the point if a local run directory was requested.
    discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)
    run_dir = tmp_path / "run"

    again = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET, local_dir=run_dir)

    assert again.skipped is True
    assert again.mirrored_to == str(run_dir)
    for name in ("manifest.json", "candidates.json", "highlights.json", "transcript.json"):
        assert (run_dir / name).is_file(), name


def test_a_skipped_mirror_carries_the_same_candidate_set(client, tmp_path):
    first = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET)
    run_dir = tmp_path / "run"

    discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET, local_dir=run_dir)

    mirrored = CandidateDocument.model_validate(json.loads((run_dir / "candidates.json").read_text()))
    assert mirrored.candidate_set_id == first.candidate_set_id
    assert len(mirrored.candidates) == len(first.candidates)


def test_mirroring_is_only_reported_when_it_actually_happened(client, tmp_path):
    # Nothing requested, nothing claimed.
    assert discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET).mirrored_to is None

    run_dir = tmp_path / "run"
    fresh = discover_from_r2(SOURCE_ID, client=client, bucket=BUCKET, overwrite=True, local_dir=run_dir)
    assert fresh.mirrored_to == str(run_dir)
