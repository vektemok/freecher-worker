"""Tests for structured human dimensions and audio-assisted annotation."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest

from freecher_worker.evaluation import (
    DIMENSION_FIELDS,
    AudioPreviewError,
    AudioPreviewer,
    BlindEvaluationDocument,
    BlindEvaluationItem,
    build_playback_command,
    cache_path_for,
    compute_evaluation_metrics,
    ensure_audio_artifact,
    summarize_human_dimensions,
)
from freecher_worker.evaluation.annotator import (
    DIMENSION_PROMPTS,
    collect_candidate_ids,
    prompt_boolean,
    prompt_dimension,
    run_dimension_backfill,
    run_terminal_annotator,
    select_backfill_items,
)
from freecher_worker.evaluation.models import ScorerPredictionDocument, ScorerPredictionItem

SOURCE_ID = "v2866049874"
AUDIO_KEY = "processing/v2866049874/audio.m4a"
CSET = "cset_72b311f2cb41727f"


def make_item(candidate_id: str, **overrides) -> BlindEvaluationItem:
    base = dict(
        candidate_id=candidate_id, start=0.0, end=60.0, duration=60.0,
        text="какой-то текст", segment_ids=[1, 2],
    )
    base.update(overrides)
    return BlindEvaluationItem(**base)


def make_doc(items) -> BlindEvaluationDocument:
    return BlindEvaluationDocument(
        candidate_set_id=CSET, total_candidates=len(items), items=items
    )


# --------------------------------------------------------------------------
# schema: additive and backwards compatible
# --------------------------------------------------------------------------


def test_a_document_labeled_before_the_dimensions_existed_still_validates():
    legacy = {
        "candidate_set_id": CSET, "total_candidates": 1, "labeled_candidates": 1,
        "items": [{
            "candidate_id": "cand_001", "start": 0.0, "end": 60.0, "duration": 60.0,
            "text": "x", "segment_ids": [], "human_score": 3, "publishable": True,
            "human_notes": "good",
        }],
    }
    doc = BlindEvaluationDocument.model_validate(legacy)
    item = doc.items[0]

    assert item.human_score == 3 and item.publishable is True and item.human_notes == "good"
    # Absent dimensions are None, never zero: unrecorded is not "scored badly".
    assert all(getattr(item, field) is None for field in DIMENSION_FIELDS)
    assert item.bad_start is None and item.bad_end is None
    assert item.has_dimensions is False


def test_the_original_fields_are_untouched_by_the_new_ones():
    item = make_item("cand_001", human_score=4, publishable=True, human_notes="n",
                     hook_score=1, standalone_score=0, payoff_score=0,
                     value_score=0, context_dependency=4)

    # A weak set of components must not drag the canonical label down.
    assert item.human_score == 4
    assert item.publishable is True
    assert item.human_notes == "n"


@pytest.mark.parametrize("field", DIMENSION_FIELDS)
@pytest.mark.parametrize("value", [-0.5, 4.5, 10])
def test_dimensions_are_bounded_to_the_same_zero_to_four_scale(field, value):
    with pytest.raises(Exception):
        make_item("cand_001", **{field: value})


def test_has_dimensions_notices_a_boundary_flag_alone():
    assert make_item("c", bad_start=True).has_dimensions is True
    assert make_item("c", bad_end=False).has_dimensions is True
    assert make_item("c", human_score=4).has_dimensions is False


def test_the_document_counts_dimension_coverage_separately_from_scores():
    doc = make_doc([
        make_item("a", human_score=3, hook_score=4),
        make_item("b", human_score=2),
        make_item("c"),
    ])

    assert doc.update_labeled_count() == 2
    assert doc.update_dimension_count() == 1


def test_existing_ranking_metrics_are_unaffected_by_the_new_fields():
    doc = make_doc([
        make_item("cand_001", human_score=4, publishable=True, hook_score=0, context_dependency=4),
        make_item("cand_002", human_score=1, hook_score=4, context_dependency=0),
    ])
    predictions = ScorerPredictionDocument(
        candidate_set_id=CSET, scorer="heuristic", scorer_version="heuristic_v1",
        predictions=[
            ScorerPredictionItem(candidate_id="cand_001", rank=1, score=90.0),
            ScorerPredictionItem(candidate_id="cand_002", rank=2, score=10.0),
        ],
    )

    metrics = compute_evaluation_metrics(doc, predictions, k_values=[1, 2])

    # Relevance still comes from human_score alone; the dimensions, which point
    # the other way here, do not enter the calculation.
    assert metrics.precision_at_k[1] == 1.0
    assert metrics.mean_human_score_at_k[1] == 4.0


# --------------------------------------------------------------------------
# summary statistics
# --------------------------------------------------------------------------


def test_the_summary_splits_the_pool_into_the_three_tiers():
    doc = make_doc([
        make_item("a", human_score=4), make_item("b", human_score=3),
        make_item("c", human_score=2), make_item("d", human_score=1),
        make_item("e", human_score=0), make_item("f"),
    ])

    summary = summarize_human_dimensions(doc)

    assert summary.total_candidates == 6
    assert summary.labeled_candidates == 5
    assert (summary.strong_count, summary.borderline_count, summary.reject_count) == (2, 1, 2)
    assert summary.is_complete is False
    assert summary.human_score_mean == 2.0
    assert summary.human_score_histogram == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}


def test_the_summary_describes_each_dimension():
    doc = make_doc([
        make_item("a", human_score=4, hook_score=4, standalone_score=3,
                  payoff_score=4, value_score=4, context_dependency=0),
        make_item("b", human_score=2, hook_score=2, standalone_score=1,
                  payoff_score=0, value_score=2, context_dependency=4),
    ])

    summary = summarize_human_dimensions(doc)
    by_field = {d.field: d for d in summary.dimensions}

    assert by_field["hook_score"].mean == 3.0
    assert by_field["hook_score"].labeled == 2
    assert by_field["hook_score"].histogram[4] == 1
    assert by_field["context_dependency"].mean == 2.0
    # Context dependency is the one where a high number is bad.
    assert by_field["context_dependency"].lower_is_better is True
    assert by_field["hook_score"].lower_is_better is False


def test_the_summary_counts_boundary_problems_against_what_was_judged():
    doc = make_doc([
        make_item("a", human_score=3, bad_start=True, bad_end=False),
        make_item("b", human_score=3, bad_start=False, bad_end=False),
        make_item("c", human_score=3),  # boundaries never judged
    ])

    summary = summarize_human_dimensions(doc)

    assert summary.bad_start_count == 1
    assert summary.bad_end_count == 0
    # The denominator is what was judged, not the whole pool.
    assert summary.boundary_labeled == 2
    assert summary.rate(summary.bad_start_count, summary.boundary_labeled) == 0.5


def test_the_summary_is_safe_on_an_unlabeled_document():
    summary = summarize_human_dimensions(make_doc([make_item("a"), make_item("b")]))

    assert summary.labeled_candidates == 0
    assert summary.human_score_mean is None
    assert summary.rate(0) is None
    assert all(d.mean is None for d in summary.dimensions)


def test_a_fully_labeled_pool_reports_complete():
    doc = make_doc([make_item("a", human_score=3), make_item("b", human_score=1)])
    assert summarize_human_dimensions(doc).is_complete is True


# --------------------------------------------------------------------------
# annotator prompts
# --------------------------------------------------------------------------


def test_a_dimension_prompt_records_a_value(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "3")
    assert prompt_dimension("Hook", "none", "great", None) == 3


def test_an_empty_dimension_prompt_leaves_the_value_alone(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    # Unset stays unset ...
    assert prompt_dimension("Hook", "none", "great", None) is None
    # ... and an existing answer is kept rather than cleared.
    assert prompt_dimension("Hook", "none", "great", 2) == 2


def test_a_dimension_prompt_reasks_until_the_answer_is_in_range(monkeypatch):
    answers = iter(["9", "abc", "-1", "4"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert prompt_dimension("Hook", "none", "great", None) == 4


@pytest.mark.parametrize(
    ("typed", "expected"), [("y", True), ("n", False), ("yes", True), ("no", False)]
)
def test_a_boolean_prompt_reads_yes_and_no(monkeypatch, typed, expected):
    monkeypatch.setattr("builtins.input", lambda _: typed)
    assert prompt_boolean("Bad start?", None) is expected


def test_an_empty_boolean_prompt_keeps_the_current_answer(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert prompt_boolean("Bad start?", True) is True
    assert prompt_boolean("Bad start?", None) is None


def test_every_declared_dimension_is_actually_asked():
    asked = {attribute for attribute, *_ in DIMENSION_PROMPTS}
    assert asked == set(DIMENSION_FIELDS)


# --------------------------------------------------------------------------
# the annotation loop
# --------------------------------------------------------------------------


def _run_annotator(tmp_path, answers, **kwargs):
    doc = make_doc([make_item("cand_001"), make_item("cand_002")])
    path = tmp_path / "evaluation_blind.json"
    stream = iter(answers)

    import builtins

    original_input = builtins.input
    builtins.input = lambda _="": next(stream)
    try:
        run_terminal_annotator(eval_doc=doc, eval_file_path=path, **kwargs)
    finally:
        builtins.input = original_input
    return doc, path


def test_the_loop_records_the_score_then_every_dimension(tmp_path):
    doc, path = _run_annotator(
        tmp_path,
        ["4", "y", "great clip", "4", "3", "4", "2", "1", "n", "n", "q"],
        capture_dimensions=True,
    )
    item = doc.items[0]

    assert item.human_score == 4
    assert item.publishable is True
    assert item.human_notes == "great clip"
    assert (item.hook_score, item.standalone_score, item.payoff_score) == (4, 3, 4)
    assert (item.value_score, item.context_dependency) == (2, 1)
    assert item.bad_start is False and item.bad_end is False
    # Saved to disk as it goes, so an interrupted session keeps its work.
    assert json.loads(path.read_text())["items"][0]["hook_score"] == 4


def test_the_loop_does_not_ask_for_dimensions_unless_asked(tmp_path):
    # The library default stays off so the pre-existing annotator behaviour --
    # and the tests written against it -- are untouched. The CLI opts in.
    doc, _ = _run_annotator(tmp_path, ["3", "y", "note", "q"])

    assert doc.items[0].human_score == 3
    assert doc.items[0].human_notes == "note"
    assert doc.items[0].has_dimensions is False


def test_dimensions_can_be_switched_off(tmp_path):
    doc, _ = _run_annotator(tmp_path, ["3", "y", "", "q"], capture_dimensions=False)
    item = doc.items[0]

    assert item.human_score == 3
    assert item.has_dimensions is False


def test_skipping_a_dimension_leaves_it_unrecorded(tmp_path):
    doc, _ = _run_annotator(
        tmp_path,
        ["2", "n", "", "3", "", "", "", "", "", "", "q"],
        capture_dimensions=True,
    )
    item = doc.items[0]

    assert item.human_score == 2
    assert item.hook_score == 3
    assert item.standalone_score is None
    assert item.bad_start is None


def test_the_saved_document_tracks_dimension_coverage(tmp_path):
    doc, path = _run_annotator(
        tmp_path, ["3", "y", "", "3", "3", "3", "3", "1", "n", "n", "q"],
        capture_dimensions=True,
    )
    stored = json.loads(path.read_text())

    assert stored["labeled_candidates"] == 1
    assert stored["dimension_labeled_candidates"] == 1


# --------------------------------------------------------------------------
# audio preview
# --------------------------------------------------------------------------


class FakeS3Client:
    def __init__(self, objects=None, metadata=None) -> None:
        self.objects = dict(objects or {})
        self.metadata = dict(metadata or {})
        self.get_calls: list[str] = []

    def get_object(self, *, Bucket, Key):
        self.get_calls.append(Key)
        if Key not in self.objects:
            error = Exception("Not Found")
            error.response = {"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}
            raise error
        payload = self.objects[Key]
        return {
            "Body": io.BytesIO(payload),
            "ContentLength": len(payload),
            "Metadata": self.metadata.get(Key, {}),
        }


@pytest.fixture(scope="session")
def speech_audio(tmp_path_factory) -> Path:
    if not subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
        pytest.skip("ffmpeg is not installed")
    destination = tmp_path_factory.mktemp("audio") / "audio.m4a"
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=330:duration=12",
         "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "64k",
         "-f", "ipod", str(destination)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        pytest.skip("could not build the audio fixture")
    return destination


def test_playback_asks_for_exactly_the_candidate_range():
    command = build_playback_command("/tmp/audio.m4a", 4967.0, 5036.0)

    assert command[0] == "ffplay"
    # Seek before input, so a window at 1:22:47 starts as fast as one at zero.
    assert command.index("-ss") < command.index("/tmp/audio.m4a")
    assert command[command.index("-ss") + 1] == "4967.000"
    assert command[command.index("-t") + 1] == "69.000"
    # No window, and it exits on its own at the end of the range.
    assert "-nodisp" in command and "-autoexit" in command


def test_a_zero_length_range_never_asks_for_negative_duration():
    command = build_playback_command("/tmp/a.m4a", 10.0, 5.0)
    assert command[command.index("-t") + 1] == "0.000"


def test_the_artifact_is_fetched_once_and_then_reused(tmp_path, speech_audio):
    client = FakeS3Client({AUDIO_KEY: speech_audio.read_bytes()})

    first = ensure_audio_artifact(client, "freecher", SOURCE_ID, cache_dir=tmp_path)
    second = ensure_audio_artifact(client, "freecher", SOURCE_ID, cache_dir=tmp_path)
    third = ensure_audio_artifact(client, "freecher", SOURCE_ID, cache_dir=tmp_path)

    # One download for the whole session, not one per candidate.
    assert client.get_calls == [AUDIO_KEY]
    assert first.reused is False
    assert second.reused is True and third.reused is True
    assert first.path == cache_path_for(SOURCE_ID, tmp_path)
    assert first.path.read_bytes() == speech_audio.read_bytes()


def test_the_source_video_is_never_requested(tmp_path, speech_audio):
    client = FakeS3Client({
        AUDIO_KEY: speech_audio.read_bytes(),
        "input/v2866049874/source.mp4": b"the enormous video",
    })

    ensure_audio_artifact(client, "freecher", SOURCE_ID, cache_dir=tmp_path)

    assert client.get_calls == [AUDIO_KEY]
    assert "input/v2866049874/source.mp4" not in client.get_calls


def test_a_damaged_cache_entry_is_replaced_rather_than_played(tmp_path, speech_audio):
    client = FakeS3Client({AUDIO_KEY: speech_audio.read_bytes()})
    ensure_audio_artifact(client, "freecher", SOURCE_ID, cache_dir=tmp_path)
    cache_path_for(SOURCE_ID, tmp_path).write_bytes(b"corrupted")

    recovered = ensure_audio_artifact(client, "freecher", SOURCE_ID, cache_dir=tmp_path)

    assert recovered.reused is False
    assert client.get_calls == [AUDIO_KEY, AUDIO_KEY]
    assert recovered.path.read_bytes() == speech_audio.read_bytes()


def test_refresh_forces_a_new_download(tmp_path, speech_audio):
    client = FakeS3Client({AUDIO_KEY: speech_audio.read_bytes()})
    ensure_audio_artifact(client, "freecher", SOURCE_ID, cache_dir=tmp_path)

    refreshed = ensure_audio_artifact(client, "freecher", SOURCE_ID, cache_dir=tmp_path, force=True)

    assert refreshed.reused is False
    assert client.get_calls == [AUDIO_KEY, AUDIO_KEY]


def test_a_missing_artifact_is_reported_clearly(tmp_path):
    with pytest.raises(AudioPreviewError, match="could not fetch"):
        ensure_audio_artifact(FakeS3Client(), "freecher", SOURCE_ID, cache_dir=tmp_path)


def test_the_previewer_reports_a_missing_player_instead_of_crashing(tmp_path, speech_audio):
    from freecher_worker.evaluation.audio_preview import CachedAudio

    previewer = AudioPreviewer(
        CachedAudio(path=speech_audio, source_id=SOURCE_ID, bucket="b", key=AUDIO_KEY),
        ffplay_path="/nonexistent/ffplay",
    )

    assert previewer.available is False
    with pytest.raises(AudioPreviewError, match="not found"):
        previewer.play(0.0, 5.0)


def test_stopping_when_nothing_plays_is_harmless(speech_audio):
    from freecher_worker.evaluation.audio_preview import CachedAudio

    previewer = AudioPreviewer(
        CachedAudio(path=speech_audio, source_id=SOURCE_ID, bucket="b", key=AUDIO_KEY)
    )
    previewer.stop()
    previewer.stop()
    assert previewer.is_playing is False


def test_preview_never_writes_to_the_candidate(tmp_path, speech_audio):
    """Audio is annotation assistance; it must not touch recorded data."""
    from freecher_worker.evaluation.audio_preview import CachedAudio

    class RecordingPreviewer(AudioPreviewer):
        def __init__(self, audio):
            super().__init__(audio)
            self.played: list[tuple[float, float]] = []

        def play(self, start, end):
            self.played.append((start, end))

        def stop(self):
            pass

    previewer = RecordingPreviewer(
        CachedAudio(path=speech_audio, source_id=SOURCE_ID, bucket="b", key=AUDIO_KEY)
    )
    doc, _ = _run_annotator(tmp_path, ["a", "x", "q"], previewer=previewer)

    # The candidate was played over its exact range and left entirely unlabeled.
    assert previewer.played == [(0.0, 60.0)]
    assert doc.items[0].human_score is None
    assert doc.items[0].has_dimensions is False


# --------------------------------------------------------------------------
# pass 2: dimension backfill over already-labeled candidates
# --------------------------------------------------------------------------


def pass_one_doc(count: int = 3, **overrides) -> BlindEvaluationDocument:
    """The state after `label-eval --no-dimensions` over the whole pool."""
    items = []
    for index in range(1, count + 1):
        base = dict(
            human_score=3, publishable=True, human_notes=f"pass 1 note {index}"
        )
        base.update(overrides)
        items.append(make_item(f"cand_{index:03d}", **base))
    return make_doc(items)


def _run_backfill(tmp_path, answers, doc=None, **kwargs):
    import builtins

    doc = doc if doc is not None else pass_one_doc()
    path = tmp_path / "evaluation_blind.json"
    stream = iter(answers)
    original_input = builtins.input
    builtins.input = lambda _="": next(stream)
    try:
        run_dimension_backfill(eval_doc=doc, eval_file_path=path, **kwargs)
    finally:
        builtins.input = original_input
    return doc, path


def test_backfill_targets_exactly_the_candidates_missing_dimensions():
    doc = pass_one_doc(3)
    doc.items[1].hook_score = 4  # already has dimensions

    targets = select_backfill_items(doc.items)

    assert [item.candidate_id for item in targets] == ["cand_001", "cand_003"]


def test_backfill_never_targets_an_unlabeled_candidate():
    doc = make_doc([
        make_item("cand_001", human_score=3),
        make_item("cand_002"),  # pass 1 skipped this one
    ])

    # A backfill adds detail to a judgement, it does not make one.
    assert [item.candidate_id for item in select_backfill_items(doc.items)] == ["cand_001"]


def test_backfill_can_be_restricted_to_chosen_candidates():
    doc = pass_one_doc(4)

    targets = select_backfill_items(doc.items, candidate_ids={"cand_002", "cand_004"})

    assert [item.candidate_id for item in targets] == ["cand_002", "cand_004"]


def test_redoing_dimensions_revisits_completed_candidates():
    doc = pass_one_doc(2)
    doc.items[0].hook_score = 4

    assert len(select_backfill_items(doc.items)) == 1
    assert len(select_backfill_items(doc.items, include_complete=True)) == 2


def test_backfill_records_dimensions_without_touching_the_canonical_label(tmp_path):
    doc, path = _run_backfill(tmp_path, ["", "4", "3", "2", "4", "1", "n", "y", "q"])
    item = doc.items[0]

    # The whole point: pass-1 answers survive untouched.
    assert item.human_score == 3
    assert item.publishable is True
    assert item.human_notes == "pass 1 note 1"
    # ... and the dimensions are now recorded.
    assert (item.hook_score, item.standalone_score, item.payoff_score) == (4, 3, 2)
    assert (item.value_score, item.context_dependency) == (4, 1)
    assert item.bad_start is False and item.bad_end is True
    assert json.loads(path.read_text())["items"][0]["human_score"] == 3


@pytest.mark.parametrize("keystrokes", [
    ["0", "q"],          # a digit, which in pass 1 would have been a new score
    ["4", "q"],
    ["s", "q"],          # skip
    ["b", "q"],          # back
    ["nonsense", "q"],
])
def test_no_keystroke_in_this_pass_can_change_the_canonical_label(tmp_path, keystrokes):
    # Pass 1 lost a label to a mistyped digit; this loop has no code path that
    # can assign human_score, publishable or human_notes at all.
    doc, _ = _run_backfill(tmp_path, keystrokes)

    for item in doc.items:
        assert item.human_score == 3
        assert item.publishable is True
        assert item.human_notes.startswith("pass 1 note")


def test_backfill_preserves_a_publishable_false_and_empty_notes(tmp_path):
    doc = pass_one_doc(1, human_score=1, publishable=False, human_notes=None)

    doc, _ = _run_backfill(tmp_path, ["", "0", "0", "0", "0", "4", "y", "y", "q"], doc=doc)
    item = doc.items[0]

    assert item.human_score == 1
    assert item.publishable is False
    assert item.human_notes is None
    assert item.context_dependency == 4


def test_backfill_leaves_an_answer_alone_when_the_rater_presses_enter(tmp_path):
    doc = pass_one_doc(1)
    doc.items[0].hook_score = 2

    doc, _ = _run_backfill(
        tmp_path, ["", "", "3", "", "", "", "", "", "q"],
        doc=doc, include_complete=True,
    )
    item = doc.items[0]

    assert item.hook_score == 2          # kept
    assert item.standalone_score == 3    # recorded
    assert item.payoff_score is None     # still unrecorded, not zero


def test_backfill_resumes_over_what_is_still_missing(tmp_path):
    doc = pass_one_doc(3)

    doc, path = _run_backfill(tmp_path, ["", "4", "4", "4", "4", "0", "n", "n", "q"], doc=doc)
    assert doc.update_dimension_count() == 1

    # A second pass picks up the two that are still missing, not the finished one.
    remaining = select_backfill_items(doc.items)
    assert [item.candidate_id for item in remaining] == ["cand_002", "cand_003"]


def test_backfill_on_a_finished_pool_does_nothing_and_says_so(tmp_path, capsys):
    doc = pass_one_doc(2)
    for item in doc.items:
        item.hook_score = 3

    doc, _ = _run_backfill(tmp_path, [], doc=doc)

    assert "No candidates need dimension labeling" in capsys.readouterr().out


def test_backfill_saves_after_each_candidate(tmp_path):
    doc, path = _run_backfill(
        tmp_path,
        ["", "4", "4", "4", "4", "0", "n", "n",
         "", "1", "1", "1", "1", "4", "y", "y", "q"],
    )
    stored = json.loads(path.read_text())

    assert stored["dimension_labeled_candidates"] == 2
    assert stored["items"][0]["hook_score"] == 4
    assert stored["items"][1]["hook_score"] == 1
    # Canonical labels untouched throughout.
    assert [item["human_score"] for item in stored["items"]] == [3, 3, 3]


def test_backfill_plays_audio_without_recording_anything(tmp_path, speech_audio):
    from freecher_worker.evaluation.audio_preview import CachedAudio

    class RecordingPreviewer(AudioPreviewer):
        def __init__(self, audio):
            super().__init__(audio)
            self.played = []

        def play(self, start, end):
            self.played.append((start, end))

        def stop(self):
            pass

    previewer = RecordingPreviewer(
        CachedAudio(path=speech_audio, source_id=SOURCE_ID, bucket="b", key=AUDIO_KEY)
    )
    doc, _ = _run_backfill(tmp_path, ["a", "x", "q"], previewer=previewer)

    assert previewer.played == [(0.0, 60.0)]
    assert doc.items[0].has_dimensions is False
    assert doc.items[0].human_score == 3


def test_the_backfill_loop_cannot_assign_a_canonical_label():
    """A structural guarantee, not just a behavioural one."""
    import inspect

    from freecher_worker.evaluation.annotator import run_dimension_backfill

    source = inspect.getsource(run_dimension_backfill)
    for attribute in ("human_score", "publishable", "human_notes"):
        assert f"item.{attribute} =" not in source
        assert f"setattr(item, \"{attribute}\"" not in source


# --------------------------------------------------------------------------
# candidate selection
# --------------------------------------------------------------------------


def test_ids_can_be_given_inline():
    assert collect_candidate_ids("cand_001,cand_002") == {"cand_001", "cand_002"}
    assert collect_candidate_ids(" cand_001 , cand_002 ") == {"cand_001", "cand_002"}
    assert collect_candidate_ids("cand_001") == {"cand_001"}


def test_ids_can_come_from_a_json_file(tmp_path):
    path = tmp_path / "subset.json"
    path.write_text(json.dumps(["cand_003", "cand_007"]))

    assert collect_candidate_ids(str(path)) == {"cand_003", "cand_007"}


def test_a_json_file_that_is_not_a_list_of_ids_is_rejected(tmp_path):
    path = tmp_path / "subset.json"
    path.write_text(json.dumps({"candidate_id": "cand_001"}))

    with pytest.raises(ValueError, match="array of candidate id strings"):
        collect_candidate_ids(str(path))


# --------------------------------------------------------------------------
# anchoring: the rater's own pass-1 answer is hidden by default
# --------------------------------------------------------------------------


def test_the_backfill_screen_hides_the_pass_one_answer_by_default(tmp_path, capsys):
    doc = pass_one_doc(1, human_score=4, publishable=True, human_notes="unmistakable note")

    doc, _ = _run_backfill(tmp_path, ["", "1", "1", "1", "1", "4", "y", "y", "q"], doc=doc)
    rendered = capsys.readouterr().out

    # Seeing "I called this a 4" would pull every dimension towards agreeing.
    assert "score=4" not in rendered
    assert "unmistakable note" not in rendered
    assert "publishable=True" not in rendered
    assert "hidden so they do not anchor" in rendered
    # The candidate itself is still shown, or there would be nothing to judge.
    assert "cand_001" in rendered

    # Hiding is presentation only: the values are untouched in the document...
    assert doc.items[0].human_score == 4
    assert doc.items[0].publishable is True
    assert doc.items[0].human_notes == "unmistakable note"
    # ... and the dimensions, which disagree with it, were recorded as given.
    assert doc.items[0].hook_score == 1
    assert doc.items[0].context_dependency == 4


def test_the_pass_one_answer_can_be_shown_on_request(tmp_path, capsys):
    doc = pass_one_doc(1, human_score=4, publishable=True, human_notes="unmistakable note")

    doc, _ = _run_backfill(
        tmp_path, ["", "1", "1", "1", "1", "4", "y", "y", "q"],
        doc=doc, show_canonical_label=True,
    )
    rendered = capsys.readouterr().out

    assert "score=4" in rendered
    assert "publishable=True" in rendered
    assert "unmistakable note" in rendered

    # Same preserved values either way; only the rendering differs.
    assert doc.items[0].human_score == 4
    assert doc.items[0].publishable is True
    assert doc.items[0].human_notes == "unmistakable note"


def test_showing_the_label_changes_nothing_but_the_screen(tmp_path):
    answers = ["", "2", "3", "2", "3", "1", "n", "n", "q"]

    hidden, hidden_path = _run_backfill(tmp_path / "a", answers.copy(), doc=pass_one_doc(1))
    shown, shown_path = _run_backfill(
        tmp_path / "b", answers.copy(), doc=pass_one_doc(1), show_canonical_label=True
    )

    # Every recorded value is identical; only the document's own creation
    # timestamp differs, since these are two separately built documents.
    assert json.loads(hidden_path.read_text())["items"] == json.loads(shown_path.read_text())["items"]
    assert hidden.dimension_labeled_candidates == shown.dimension_labeled_candidates
