"""Tests for the R2-backed transcription workflow."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from freecher_worker.transcription import (
    TRANSCRIPT_SCHEMA_VERSION,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    TranscriptionProgress,
    TranscriptionWorkflowError,
    audio_key_for,
    download_audio_artifact,
    existing_transcript,
    serialize_transcript,
    transcribe_from_r2,
    transcript_key_for,
    transcript_object_metadata,
    validate_audio_artifact,
)
from freecher_worker.transcription.r2 import duration_tolerance_seconds
from freecher_worker.transcription.whisper import WhisperTranscriber, _words_from_segment

SOURCE_ID = "v2866049874"
AUDIO_KEY = "processing/v2866049874/audio.m4a"
TRANSCRIPT_KEY = "processing/v2866049874/transcript.json"
AUDIO_SECONDS = 12.0

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe are not installed",
)


# --------------------------------------------------------------------------
# doubles
# --------------------------------------------------------------------------


class FakeS3Client:
    """In-memory object store with the four calls the workflow uses."""

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects: dict[str, bytes] = dict(objects or {})
        self.metadata: dict[str, dict[str, str]] = {}
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
        payload = self.objects[Key]
        return {
            "Body": io.BytesIO(payload),
            "ContentLength": len(payload),
            "Metadata": self.metadata.get(Key, {}),
        }

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        self.objects[kwargs["Key"]] = kwargs["Body"]
        self.metadata[kwargs["Key"]] = kwargs.get("Metadata", {})
        return {"ETag": '"put-etag"'}


def _missing_key_error() -> Exception:
    error = Exception("Not Found")
    error.response = {"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}
    return error


class LyingLengthClient(FakeS3Client):
    """Reports more bytes than it actually hands over."""

    def get_object(self, *, Bucket, Key):
        response = super().get_object(Bucket=Bucket, Key=Key)
        response["ContentLength"] = len(self.objects[Key]) + 4096
        return response


class FakeWord:
    def __init__(self, start, end, word, probability=None):
        self.start, self.end, self.word, self.probability = start, end, word, probability


class FakeSegment:
    def __init__(self, start, end, text, words=None):
        self.start, self.end, self.text = start, end, text
        self.avg_logprob, self.no_speech_prob = -0.2, 0.01
        self.words = words


class FakeInfo:
    def __init__(self, language="ru", language_probability=0.97, duration=AUDIO_SECONDS):
        self.language = language
        self.language_probability = language_probability
        self.duration = duration


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel."""

    def __init__(self, segments=None, info=None) -> None:
        self.segments = segments if segments is not None else _default_segments()
        self.info = info or FakeInfo()
        self.calls: list[dict] = []

    def transcribe(self, path, **kwargs):
        self.calls.append({"path": path, **kwargs})
        return iter(self.segments), self.info


def _default_segments():
    return [
        FakeSegment(0.0, 4.5, " Привет мир", words=[
            FakeWord(0.0, 1.2, " Привет", 0.94),
            FakeWord(1.3, 4.5, " мир", 0.88),
        ]),
        FakeSegment(4.8, 12.0, " Это тест", words=[
            FakeWord(4.8, 7.0, " Это", 0.91),
            FakeWord(7.1, 12.0, " тест", 0.86),
        ]),
    ]


class StubTranscriber:
    """A BaseTranscriber that returns a canned transcript."""

    def __init__(self, transcript: Transcript | None = None, error: Exception | None = None) -> None:
        self.transcript = transcript or _canned_transcript()
        self.error = error
        self.calls: list[dict] = []

    def transcribe(self, audio_path, language=None, source_fingerprint_id=None, on_progress=None):
        self.calls.append({"audio_path": Path(audio_path), "language": language})
        if self.error is not None:
            raise self.error
        if on_progress is not None:
            on_progress(TranscriptionProgress(1, 6.0, AUDIO_SECONDS, 1.0))
        return self.transcript.model_copy(deep=True)


def _canned_transcript() -> Transcript:
    return Transcript(
        language="ru",
        language_probability=0.97,
        duration=AUDIO_SECONDS,
        model="large-v3",
        compute_type="float16",
        device="cuda",
        word_timestamps=True,
        segments=[
            TranscriptSegment(
                id=0, start=0.0, end=4.5, text="Привет мир",
                words=[
                    TranscriptWord(start=0.0, end=1.2, word=" Привет", probability=0.94),
                    TranscriptWord(start=1.3, end=4.5, word=" мир", probability=0.88),
                ],
            ),
        ],
    )


@pytest.fixture(scope="session")
def speech_audio(tmp_path_factory) -> Path:
    """A 12s mono 16 kHz m4a, shaped exactly like the ingest artifact."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    destination = tmp_path_factory.mktemp("audio") / "audio.m4a"
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=330:duration={AUDIO_SECONDS:g}",
            "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "64k",
            "-f", "ipod", str(destination),
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0 or not destination.is_file():
        pytest.skip("could not build the audio fixture")
    return destination


@pytest.fixture
def audio_client(speech_audio) -> FakeS3Client:
    client = FakeS3Client({AUDIO_KEY: speech_audio.read_bytes()})
    client.metadata[AUDIO_KEY] = {
        "artifact": "audio",
        "source-key": "input/v2866049874/source.mp4",
        "source-duration-seconds": f"{AUDIO_SECONDS:.3f}",
        "audio-codec": "aac",
        "audio-sample-rate": "16000",
        "audio-channels": "1",
    }
    return client


# --------------------------------------------------------------------------
# keys
# --------------------------------------------------------------------------


def test_keys_are_derived_from_the_source_id():
    assert audio_key_for(SOURCE_ID) == AUDIO_KEY
    assert transcript_key_for(SOURCE_ID) == TRANSCRIPT_KEY
    # The transcript never lands on the audio it was made from.
    assert audio_key_for(SOURCE_ID) != transcript_key_for(SOURCE_ID)


@pytest.mark.parametrize("source_id", ["", "  ", "a/b", "/", "nested/id"])
def test_a_source_id_must_be_a_single_path_segment(source_id):
    with pytest.raises(ValueError):
        audio_key_for(source_id)


def test_surrounding_slashes_and_spaces_are_tolerated():
    assert audio_key_for("  v123  ") == "processing/v123/audio.m4a"


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------


def test_download_writes_the_object_and_reports_progress(tmp_path):
    payload = b"x" * 20_000_000
    client = FakeS3Client({AUDIO_KEY: payload})
    client.metadata[AUDIO_KEY] = {"audio-codec": "aac"}
    destination = tmp_path / "audio.m4a"

    seen: list = []
    written, metadata = download_audio_artifact(
        client, "freecher", AUDIO_KEY, destination, on_progress=seen.append
    )

    assert written == len(payload)
    assert destination.read_bytes() == payload
    assert metadata["audio-codec"] == "aac"
    assert seen and seen[-1].downloaded_bytes == len(payload)
    assert seen[-1].percent == pytest.approx(100.0)


def test_download_refuses_a_short_read(tmp_path):
    # A truncated download must never reach the model as if it were complete.
    client = LyingLengthClient({AUDIO_KEY: b"y" * 1024})

    with pytest.raises(TranscriptionWorkflowError, match="truncated"):
        download_audio_artifact(client, "freecher", AUDIO_KEY, tmp_path / "a.m4a")


def test_download_reports_a_missing_object_clearly(tmp_path):
    with pytest.raises(TranscriptionWorkflowError, match="could not read"):
        download_audio_artifact(FakeS3Client(), "freecher", AUDIO_KEY, tmp_path / "a.m4a")


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@needs_ffmpeg
def test_validation_accepts_the_ingest_artifact(speech_audio):
    info = validate_audio_artifact(speech_audio)

    assert (info.codec, info.sample_rate, info.channels) == ("aac", 16000, 1)
    assert info.duration_seconds == pytest.approx(AUDIO_SECONDS, abs=0.2)
    assert info.size_bytes == speech_audio.stat().st_size


@needs_ffmpeg
def test_validation_cross_checks_the_duration_recorded_on_the_object(speech_audio):
    # The object says the source runs an hour; the audio is twelve seconds.
    with pytest.raises(TranscriptionWorkflowError, match="would not line up"):
        validate_audio_artifact(
            speech_audio, metadata={"source-duration-seconds": "3600.0"}
        )

    # A duration that agrees passes.
    info = validate_audio_artifact(
        speech_audio, metadata={"source-duration-seconds": f"{AUDIO_SECONDS:.3f}"}
    )
    assert info.source_duration_seconds == pytest.approx(AUDIO_SECONDS)


def test_validation_refuses_an_empty_file(tmp_path):
    empty = tmp_path / "audio.m4a"
    empty.write_bytes(b"")
    with pytest.raises(TranscriptionWorkflowError, match="empty"):
        validate_audio_artifact(empty)


def test_validation_refuses_a_missing_file(tmp_path):
    with pytest.raises(TranscriptionWorkflowError, match="missing"):
        validate_audio_artifact(tmp_path / "nope.m4a")


@needs_ffmpeg
def test_validation_refuses_bytes_that_are_not_audio(tmp_path):
    junk = tmp_path / "audio.m4a"
    junk.write_bytes(b"this is not an audio file at all" * 100)
    with pytest.raises(TranscriptionWorkflowError, match="not decodable"):
        validate_audio_artifact(junk)


@pytest.mark.parametrize(
    ("source", "expected"),
    [(60.0, 2.0), (3600.0, 3.6), (6817.0, 5.0), (7200.0, 5.0)],
)
def test_the_drift_allowance_matches_the_ingest_side_policy(source, expected):
    assert duration_tolerance_seconds(source) == pytest.approx(expected)


# --------------------------------------------------------------------------
# word timestamps
# --------------------------------------------------------------------------


def test_word_timestamps_are_requested_and_mapped_onto_the_segment_timeline():
    model = FakeWhisperModel()
    transcriber = WhisperTranscriber(word_timestamps=True)
    transcriber._model = model

    transcript = transcriber.transcribe(__file__)

    assert model.calls[0]["word_timestamps"] is True
    assert transcript.word_timestamps is True
    words = transcript.segments[0].words
    assert words is not None and [w.word for w in words] == [" Привет", " мир"]
    assert (words[0].start, words[0].end) == (0.0, 1.2)
    assert words[0].probability == 0.94
    # Word timings stay inside the segment that carries them.
    for segment in transcript.segments:
        for word in segment.words or ():
            assert segment.start <= word.start <= word.end <= segment.end


def test_words_are_absent_not_empty_when_not_requested():
    model = FakeWhisperModel()
    transcriber = WhisperTranscriber(word_timestamps=False)
    transcriber._model = model

    transcript = transcriber.transcribe(__file__)

    assert model.calls[0]["word_timestamps"] is False
    assert transcript.word_timestamps is False
    # None means "not asked for", which is different from "asked for, none found".
    assert all(segment.words is None for segment in transcript.segments)
    assert transcript.word_count == 0


def test_a_segment_with_no_words_maps_to_an_empty_list_not_none():
    assert _words_from_segment(FakeSegment(0.0, 1.0, "x", words=None)) == []
    assert _words_from_segment(FakeSegment(0.0, 1.0, "x", words=[])) == []


def test_language_detection_is_carried_through_with_its_probability():
    model = FakeWhisperModel(info=FakeInfo(language="ru", language_probability=0.9712))
    transcriber = WhisperTranscriber()
    transcriber._model = model

    transcript = transcriber.transcribe(__file__)

    assert transcript.language == "ru"
    assert transcript.language_probability == 0.971
    # Auto-detection means no language was forced on the model.
    assert model.calls[0]["language"] is None


def test_progress_is_reported_as_the_decoder_advances_through_the_audio():
    model = FakeWhisperModel()
    transcriber = WhisperTranscriber()
    transcriber._model = model

    seen: list[TranscriptionProgress] = []
    transcriber.transcribe(__file__, on_progress=seen.append)

    assert [p.segment_count for p in seen] == [1, 2]
    assert seen[-1].current_seconds == 12.0
    assert seen[-1].percent == pytest.approx(100.0)
    assert seen[0].percent == pytest.approx(37.5)  # 4.5s of 12s


def test_progress_estimates_speed_and_eta_for_a_long_source():
    # Two hours of audio, 600s in, decoding at 20x realtime.
    progress = TranscriptionProgress(
        segment_count=900, current_seconds=1200.0, audio_seconds=7200.0, elapsed_seconds=60.0
    )
    assert progress.speed == pytest.approx(20.0)
    assert progress.eta_seconds == pytest.approx(300.0)
    assert progress.percent == pytest.approx(16.667, abs=0.01)


def test_progress_degrades_gracefully_without_a_known_duration():
    progress = TranscriptionProgress(1, 10.0, None, 1.0)
    assert progress.percent is None
    assert progress.eta_seconds is None


# --------------------------------------------------------------------------
# serialization
# --------------------------------------------------------------------------


def test_the_transcript_json_schema_is_deterministic():
    transcript = _canned_transcript()

    first = serialize_transcript(transcript)
    second = serialize_transcript(transcript.model_copy(deep=True))

    assert first == second
    payload = json.loads(first.decode("utf-8"))
    assert payload["schema_version"] == TRANSCRIPT_SCHEMA_VERSION
    # Key order is the model's, not a dict's insertion order.
    assert list(payload) == list(json.loads(second.decode("utf-8")))
    assert list(payload["segments"][0]["words"][0]) == ["start", "end", "word", "probability"]


def test_non_ascii_transcript_text_survives_serialization():
    payload = json.loads(serialize_transcript(_canned_transcript()).decode("utf-8"))
    assert payload["segments"][0]["text"] == "Привет мир"
    assert payload["segments"][0]["words"][0]["word"] == " Привет"


def test_a_serialized_transcript_round_trips_through_the_model():
    original = _canned_transcript()
    restored = Transcript.model_validate(json.loads(serialize_transcript(original).decode("utf-8")))
    assert restored == original


# --------------------------------------------------------------------------
# the workflow
# --------------------------------------------------------------------------


@needs_ffmpeg
def test_transcription_publishes_the_transcript_beside_the_audio(audio_client):
    stub = StubTranscriber()

    result = transcribe_from_r2(
        SOURCE_ID, client=audio_client, bucket="freecher", transcriber=stub
    )

    assert not result.skipped
    assert result.transcript_key == TRANSCRIPT_KEY
    assert result.audio_key == AUDIO_KEY
    assert audio_client.puts[0]["Key"] == TRANSCRIPT_KEY
    assert audio_client.puts[0]["ContentType"] == "application/json"

    stored = Transcript.model_validate(json.loads(audio_client.objects[TRANSCRIPT_KEY].decode("utf-8")))
    # Provenance: which object this came from, and how it was produced.
    assert stored.source_audio_key == AUDIO_KEY
    assert stored.source_bucket == "freecher"
    assert stored.model == "large-v3"
    assert stored.language == "ru"
    assert stored.language_probability == 0.97
    assert stored.processing_seconds is not None
    assert stored.created_at is not None
    assert stored.audio_sample_rate == 16000 and stored.audio_channels == 1
    assert stored.audio_duration == pytest.approx(AUDIO_SECONDS, abs=0.2)
    assert stored.word_timestamps is True


@needs_ffmpeg
def test_the_six_gigabyte_source_video_is_never_touched(audio_client):
    audio_client.objects["input/v2866049874/source.mp4"] = b"the enormous video"

    transcribe_from_r2(SOURCE_ID, client=audio_client, bucket="freecher", transcriber=StubTranscriber())

    # Only the audio artifact was ever fetched.
    assert audio_client.get_calls == [TRANSCRIPT_KEY, AUDIO_KEY]
    assert "input/v2866049874/source.mp4" not in audio_client.get_calls


@needs_ffmpeg
def test_an_existing_transcript_is_left_alone(audio_client):
    audio_client.objects[TRANSCRIPT_KEY] = serialize_transcript(_canned_transcript())

    result = transcribe_from_r2(
        SOURCE_ID, client=audio_client, bucket="freecher", transcriber=StubTranscriber()
    )

    assert result.skipped
    assert result.transcript is not None
    # Nothing was downloaded, decoded or written.
    assert audio_client.puts == []
    assert AUDIO_KEY not in audio_client.get_calls


@needs_ffmpeg
def test_overwrite_replaces_an_existing_transcript(audio_client):
    audio_client.objects[TRANSCRIPT_KEY] = serialize_transcript(_canned_transcript())
    stub = StubTranscriber()

    result = transcribe_from_r2(
        SOURCE_ID, client=audio_client, bucket="freecher", transcriber=stub, overwrite=True
    )

    assert not result.skipped
    assert len(audio_client.puts) == 1
    assert stub.calls


@needs_ffmpeg
def test_an_unreadable_stored_transcript_does_not_block_a_rerun(audio_client):
    # Garbage at the key is not a valid transcript, so it is not protected.
    audio_client.objects[TRANSCRIPT_KEY] = b"{ this is not json"

    result = transcribe_from_r2(
        SOURCE_ID, client=audio_client, bucket="freecher", transcriber=StubTranscriber()
    )

    assert not result.skipped
    assert audio_client.puts


def test_a_missing_audio_artifact_is_reported_before_any_gpu_work():
    stub = StubTranscriber()

    with pytest.raises(TranscriptionWorkflowError, match="run ingest first"):
        transcribe_from_r2(SOURCE_ID, client=FakeS3Client(), bucket="freecher", transcriber=stub)

    assert stub.calls == []


@needs_ffmpeg
def test_a_failed_transcription_publishes_nothing(audio_client):
    stub = StubTranscriber(error=RuntimeError("CUDA out of memory"))

    with pytest.raises(RuntimeError, match="out of memory"):
        transcribe_from_r2(SOURCE_ID, client=audio_client, bucket="freecher", transcriber=stub)

    # No partial transcript is left behind for a later run to trust.
    assert audio_client.puts == []
    assert TRANSCRIPT_KEY not in audio_client.objects


@needs_ffmpeg
def test_the_staged_audio_is_always_removed(audio_client, tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()

    transcribe_from_r2(
        SOURCE_ID, client=audio_client, bucket="freecher",
        transcriber=StubTranscriber(), staging_dir=str(staging),
    )
    assert list(staging.iterdir()) == []

    # ... including when the run fails.
    with pytest.raises(RuntimeError):
        transcribe_from_r2(
            SOURCE_ID, client=audio_client, bucket="freecher", overwrite=True,
            transcriber=StubTranscriber(error=RuntimeError("boom")),
            staging_dir=str(staging),
        )
    assert list(staging.iterdir()) == []


@needs_ffmpeg
def test_keep_audio_leaves_the_download_in_place(audio_client, tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()

    transcribe_from_r2(
        SOURCE_ID, client=audio_client, bucket="freecher", transcriber=StubTranscriber(),
        staging_dir=str(staging), keep_audio=True,
    )

    staged = list(staging.rglob("audio.m4a"))
    assert len(staged) == 1 and staged[0].stat().st_size > 0


@needs_ffmpeg
def test_the_transcript_object_carries_searchable_metadata(audio_client):
    transcribe_from_r2(SOURCE_ID, client=audio_client, bucket="freecher", transcriber=StubTranscriber())

    metadata = audio_client.puts[0]["Metadata"]
    assert metadata["artifact"] == "transcript"
    assert metadata["source-id"] == SOURCE_ID
    assert metadata["source-key"] == AUDIO_KEY
    assert metadata["asr-model"] == "large-v3"
    assert metadata["language"] == "ru"
    assert metadata["word-timestamps"] == "true"
    assert metadata["schema-version"] == TRANSCRIPT_SCHEMA_VERSION
    # R2 rejects a non-ASCII metadata header.
    assert all(value.isascii() for value in metadata.values())


def test_transcript_metadata_stays_ascii_for_a_non_latin_language():
    transcript = _canned_transcript()
    transcript.source_audio_key = AUDIO_KEY
    metadata = transcript_object_metadata(transcript, SOURCE_ID)
    assert all(value.isascii() for value in metadata.values())


@needs_ffmpeg
def test_a_forced_language_is_passed_through(audio_client):
    stub = StubTranscriber()

    transcribe_from_r2(
        SOURCE_ID, client=audio_client, bucket="freecher", transcriber=stub, language="ru"
    )

    assert stub.calls[0]["language"] == "ru"


@needs_ffmpeg
def test_an_empty_transcript_is_published_but_flagged(audio_client):
    silent = _canned_transcript()
    silent.segments = []

    result = transcribe_from_r2(
        SOURCE_ID, client=audio_client, bucket="freecher",
        transcriber=StubTranscriber(transcript=silent),
    )

    # It is still a complete result, so it is written...
    assert audio_client.puts
    # ... but an empty transcript on real audio is worth saying out loud.
    assert any("no speech was found" in w for w in result.warnings)


def test_existing_transcript_returns_none_when_the_key_is_free():
    assert existing_transcript(FakeS3Client(), "freecher", TRANSCRIPT_KEY) is None
