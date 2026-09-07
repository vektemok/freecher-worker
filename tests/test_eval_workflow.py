"""End-to-end unit tests for Phase 1.2 evaluation workflow, CLI commands, and annotator."""

import json
from pathlib import Path
from typer.testing import CliRunner

from freecher_worker.cli import app
from freecher_worker.evaluation.annotator import run_terminal_annotator
from freecher_worker.evaluation.disagreements import extract_disagreements
from freecher_worker.evaluation.models import (
    BlindEvaluationDocument,
    BlindEvaluationItem,
    ScorerPredictionDocument,
    ScorerPredictionItem,
)
from freecher_worker.highlights.models import (
    CandidateDocument,
    CandidateWindow,
    compute_candidate_set_id,
)
from freecher_worker.utils.json_io import load_json, save_json

runner = CliRunner()


def _create_mock_run_with_candidates(run_dir: Path) -> str:
    """Create a mock run directory with 6 candidate windows and a candidate_set_id."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cands = [
        CandidateWindow(id="c01", start=0.0, end=30.0, duration=30.0, text="Introductory remarks and welcome", segment_ids=[0]),
        CandidateWindow(id="c02", start=30.0, end=65.0, duration=35.0, text="Key insight: how to scale workers 10x", segment_ids=[1]),
        CandidateWindow(id="c03", start=65.0, end=100.0, duration=35.0, text="Technical breakdown of ffmpeg nvenc", segment_ids=[2]),
        CandidateWindow(id="c04", start=100.0, end=130.0, duration=30.0, text="Off-topic chatter about coffee breaks", segment_ids=[3]),
        CandidateWindow(id="c05", start=130.0, end=165.0, duration=35.0, text="Explosive story about production outage", segment_ids=[4]),
        CandidateWindow(id="c06", start=165.0, end=195.0, duration=30.0, text="Outro and thank you for watching", segment_ids=[5]),
    ]

    cset_id = compute_candidate_set_id(
        transcript_hash="dummy_transcript_hash_42",
        min_seconds=30.0,
        target_seconds=60.0,
        max_seconds=90.0,
        overlap_seconds=15.0,
    )

    cand_doc = CandidateDocument(
        candidate_set_id=cset_id,
        transcript_hash="dummy_transcript_hash_42",
        min_seconds=30.0,
        target_seconds=60.0,
        max_seconds=90.0,
        overlap_seconds=15.0,
        candidates=cands,
    )
    save_json(cand_doc, run_dir / "candidates.json")

    manifest = {
        "pipeline_version": "0.1.2",
        "created_at": "2026-09-07T10:00:00",
        "source": "/mock/video.mp4",
        "candidate_config": {
            "candidate_set_id": cset_id,
            "min_seconds": 30.0,
            "target_seconds": 60.0,
            "max_seconds": 90.0,
            "overlap": 15.0,
        },
    }
    save_json(manifest, run_dir / "manifest.json")
    return cset_id


def test_candidate_set_id_determinism():
    """Ensure candidate_set_id is deterministic and distinguishes parameters."""
    id1 = compute_candidate_set_id("hash_a", 30.0, 60.0, 90.0, 15.0)
    id2 = compute_candidate_set_id("hash_a", 30.0, 60.0, 90.0, 15.0)
    assert id1 == id2
    assert id1.startswith("cset_")

    # Different transcript hash
    id3 = compute_candidate_set_id("hash_b", 30.0, 60.0, 90.0, 15.0)
    assert id1 != id3

    # Different duration param
    id4 = compute_candidate_set_id("hash_a", 25.0, 60.0, 90.0, 15.0)
    assert id1 != id4


def test_export_eval_blind_command(tmp_path):
    """Test 'export-eval --blind' creates blind set with no model scores and deterministic order."""
    run_dir = tmp_path / "test_run"
    cset_id = _create_mock_run_with_candidates(run_dir)

    # Export with seed 42
    res_42 = runner.invoke(app, ["export-eval", str(run_dir), "--blind", "--seed", "42"])
    assert res_42.exit_code == 0
    blind_path = run_dir / "evaluation_blind.json"
    assert blind_path.is_file()

    doc_42 = load_json(blind_path)
    assert doc_42["candidate_set_id"] == cset_id
    assert doc_42["total_candidates"] == 6
    assert doc_42["seed"] == 42
    assert doc_42["labeled_candidates"] == 0

    order_42 = [item["candidate_id"] for item in doc_42["items"]]
    for item in doc_42["items"]:
        # Verify complete absence of any model scores or ranks
        assert "score" not in item
        assert "rank" not in item
        assert "reason" not in item
        assert "subscores" not in item
        assert item["human_score"] is None
        assert item["publishable"] is None
        assert item["human_notes"] is None

    # Export with seed 99 -> order should differ from seed 42
    out_99 = run_dir / "blind_99.json"
    res_99 = runner.invoke(app, ["export-eval", str(run_dir), "--blind", "--seed", "99", "-o", str(out_99)])
    assert res_99.exit_code == 0
    doc_99 = load_json(out_99)
    order_99 = [item["candidate_id"] for item in doc_99["items"]]

    assert set(order_42) == set(order_99)  # same candidate set
    assert order_42 != order_99            # shuffled differently


def test_score_run_command(tmp_path):
    """Test 'score-run' scores frozen candidate set and creates prediction file."""
    run_dir = tmp_path / "test_run"
    cset_id = _create_mock_run_with_candidates(run_dir)

    result = runner.invoke(app, ["score-run", str(run_dir), "--scorer", "heuristic"])
    assert result.exit_code == 0
    assert "Predictions saved to" in result.output

    scores_file = run_dir / "scores" / "heuristic_v1.json"
    assert scores_file.is_file()

    score_doc = load_json(scores_file)
    assert score_doc["candidate_set_id"] == cset_id
    assert score_doc["scorer"] == "heuristic"
    assert score_doc["scorer_version"] == "heuristic_v1"
    assert len(score_doc["predictions"]) == 6

    # Verify predictions are ranked 1 to 6 descending
    ranks = [p["rank"] for p in score_doc["predictions"]]
    assert ranks == [1, 2, 3, 4, 5, 6]
    scores = [p["score"] for p in score_doc["predictions"]]
    assert scores == sorted(scores, reverse=True)


def test_evaluate_command_and_mismatch_guard(tmp_path):
    """Test CLI 'evaluate' command and ensure candidate_set_id mismatch is rejected."""
    run_dir = tmp_path / "test_run"
    cset_id = _create_mock_run_with_candidates(run_dir)

    # 1. Run score-run
    runner.invoke(app, ["score-run", str(run_dir), "--scorer", "heuristic"])
    scores_file = run_dir / "scores" / "heuristic_v1.json"

    # 2. Create labeled blind evaluation document
    eval_items = [
        BlindEvaluationItem(candidate_id="c01", start=0.0, end=30.0, duration=30.0, text="Intro", human_score=1, publishable=False),
        BlindEvaluationItem(candidate_id="c02", start=30.0, end=65.0, duration=35.0, text="Insight", human_score=4, publishable=True),
        BlindEvaluationItem(candidate_id="c03", start=65.0, end=100.0, duration=35.0, text="Tech", human_score=3, publishable=True),
        BlindEvaluationItem(candidate_id="c04", start=100.0, end=130.0, duration=30.0, text="Chatter", human_score=0, publishable=False),
        BlindEvaluationItem(candidate_id="c05", start=130.0, end=165.0, duration=35.0, text="Outage", human_score=4, publishable=True),
        BlindEvaluationItem(candidate_id="c06", start=165.0, end=195.0, duration=30.0, text="Outro", human_score=1, publishable=False),
    ]
    eval_doc = BlindEvaluationDocument(
        candidate_set_id=cset_id,
        total_candidates=6,
        labeled_candidates=6,
        items=eval_items,
    )
    eval_file = run_dir / "evaluation_blind.json"
    save_json(eval_doc, eval_file)

    # 3. Successful evaluate
    res_eval = runner.invoke(app, ["evaluate", str(eval_file), str(scores_file), "--k", "3,5"])
    assert res_eval.exit_code == 0
    assert "Evaluation Report" in res_eval.output
    assert "Precision (rel>=3)" in res_eval.output
    assert "nDCG" in res_eval.output

    # JSON output mode
    res_json = runner.invoke(app, ["evaluate", str(eval_file), str(scores_file), "--json"])
    assert res_json.exit_code == 0
    parsed = json.loads(res_json.output)
    assert parsed["candidate_set_id"] == cset_id
    assert parsed["scorer"] == "heuristic"

    # 4. Mismatch test: modify candidate_set_id in eval_file
    mismatch_doc = eval_doc.model_copy(update={"candidate_set_id": "cset_mismatched_xyz"})
    mismatch_file = run_dir / "eval_mismatch.json"
    save_json(mismatch_doc, mismatch_file)

    res_mismatch = runner.invoke(app, ["evaluate", str(mismatch_file), str(scores_file)])
    assert res_mismatch.exit_code == 1
    assert "Candidate Set ID Mismatch" in res_mismatch.output


def test_compare_scorers_and_disagreements(tmp_path):
    """Test comparing two scorer predictions and exporting disagreements."""
    run_dir = tmp_path / "test_run"
    cset_id = _create_mock_run_with_candidates(run_dir)

    eval_file = run_dir / "evaluation_blind.json"
    eval_items = [
        BlindEvaluationItem(candidate_id="c01", start=0.0, end=30.0, duration=30.0, text="Intro", human_score=1, publishable=False),
        BlindEvaluationItem(candidate_id="c02", start=30.0, end=65.0, duration=35.0, text="Insight", human_score=4, publishable=True),
        BlindEvaluationItem(candidate_id="c03", start=65.0, end=100.0, duration=35.0, text="Tech", human_score=3, publishable=True),
        BlindEvaluationItem(candidate_id="c04", start=100.0, end=130.0, duration=30.0, text="Chatter", human_score=0, publishable=False),
        BlindEvaluationItem(candidate_id="c05", start=130.0, end=165.0, duration=35.0, text="Outage", human_score=4, publishable=True),
        BlindEvaluationItem(candidate_id="c06", start=165.0, end=195.0, duration=30.0, text="Outro", human_score=1, publishable=False),
    ]
    eval_doc = BlindEvaluationDocument(
        candidate_set_id=cset_id,
        total_candidates=6,
        labeled_candidates=6,
        items=eval_items,
    )
    save_json(eval_doc, eval_file)

    # Scorer 1: ranks c04 (human=0) at #1 (false positive!), c02 at #2
    s1_doc = ScorerPredictionDocument(
        candidate_set_id=cset_id,
        scorer="scorer_alpha",
        scorer_version="v1.0",
        predictions=[
            ScorerPredictionItem(candidate_id="c04", rank=1, score=95.0, reason="Mistaken noise"),
            ScorerPredictionItem(candidate_id="c02", rank=2, score=90.0, reason="Good"),
            ScorerPredictionItem(candidate_id="c03", rank=3, score=85.0, reason="Decent"),
            ScorerPredictionItem(candidate_id="c01", rank=4, score=70.0, reason="Intro"),
            ScorerPredictionItem(candidate_id="c06", rank=5, score=60.0, reason="Outro"),
            ScorerPredictionItem(candidate_id="c05", rank=6, score=50.0, reason="Missed outage"),  # False negative
        ],
    )
    s1_path = run_dir / "s1.json"
    save_json(s1_doc, s1_path)

    # Scorer 2: ranks c02 at #1, c05 at #2 (much better)
    s2_doc = ScorerPredictionDocument(
        candidate_set_id=cset_id,
        scorer="scorer_beta",
        scorer_version="v2.0",
        predictions=[
            ScorerPredictionItem(candidate_id="c02", rank=1, score=99.0, reason="Strong hook"),
            ScorerPredictionItem(candidate_id="c05", rank=2, score=95.0, reason="Great story"),
            ScorerPredictionItem(candidate_id="c03", rank=3, score=88.0, reason="Technical"),
            ScorerPredictionItem(candidate_id="c01", rank=4, score=60.0, reason="Intro"),
            ScorerPredictionItem(candidate_id="c06", rank=5, score=55.0, reason="Outro"),
            ScorerPredictionItem(candidate_id="c04", rank=6, score=20.0, reason="Chatter"),
        ],
    )
    s2_path = run_dir / "s2.json"
    save_json(s2_doc, s2_path)

    disagreements_file = run_dir / "disagreements.json"
    res_comp = runner.invoke(
        app,
        ["compare-scorers", str(eval_file), str(s1_path), str(s2_path), "--disagreements", str(disagreements_file)],
    )
    assert res_comp.exit_code == 0
    assert "Scorer Comparison Report" in res_comp.output
    assert "scorer_alpha" in res_comp.output
    assert "scorer_beta" in res_comp.output
    assert disagreements_file.is_file()

    diag = load_json(disagreements_file)
    # c04 is in top 5 of s1, but human score is 0 -> false positive
    fp_ids = [fp["candidate_id"] for fp in diag["false_positives"]]
    assert "c04" in fp_ids

    # c05 is human score 4, but s1 rank is 6 (> 5) -> false negative
    fn_ids = [fn["candidate_id"] for fn in diag["false_negatives"]]
    assert "c05" in fn_ids

    # c04 rank diff: 1 in s1, 6 in s2 (|1-6| = 5 >= 3) -> scorer divergence
    div_ids = [div["candidate_id"] for div in diag["scorer_divergences"]]
    assert "c04" in div_ids
    assert "c05" in div_ids


def test_terminal_annotator_resume_and_atomic_save(tmp_path, monkeypatch):
    """Test interactive annotator input loop, atomic save, and resume functionality."""
    eval_file = tmp_path / "eval_session.json"
    items = [
        BlindEvaluationItem(candidate_id="c01", start=0.0, end=10.0, duration=10.0, text="First", human_score=None),
        BlindEvaluationItem(candidate_id="c02", start=10.0, end=20.0, duration=10.0, text="Second", human_score=None),
        BlindEvaluationItem(candidate_id="c03", start=20.0, end=30.0, duration=10.0, text="Third", human_score=None),
    ]
    eval_doc = BlindEvaluationDocument(
        candidate_set_id="cset_test",
        total_candidates=3,
        items=items,
    )
    save_json(eval_doc, eval_file)

    # Simulate user labeling:
    # Candidate 1: score=3, pub=y, notes=Good hook
    # Candidate 2: score=q (quit early)
    inputs = iter([
        "3",            # score c01
        "y",            # pub c01
        "Good hook",    # note c01
        "q",            # quit at c02
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    labeled = run_terminal_annotator(eval_doc, eval_file)
    assert labeled == 1

    # Verify file was saved atomically with c01 labeled and c02/c03 unlabeled
    saved_data = load_json(eval_file)
    assert saved_data["items"][0]["human_score"] == 3
    assert saved_data["items"][0]["publishable"] is True
    assert saved_data["items"][0]["human_notes"] == "Good hook"
    assert saved_data["items"][1]["human_score"] is None

    # Resume session: should resume automatically from c02
    # Candidate 2: score=4, pub=y, notes=Awesome
    # Candidate 3: score=1, pub=n, notes=Dull
    resume_doc = BlindEvaluationDocument.model_validate(saved_data)
    inputs_resume = iter([
        "4",        # score c02
        "y",        # pub c02
        "Awesome",  # note c02
        "1",        # score c03
        "n",        # pub c03
        "Dull",     # note c03
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs_resume))

    labeled_resumed = run_terminal_annotator(resume_doc, eval_file)
    assert labeled_resumed == 3
    assert resume_doc.is_fully_labeled


def test_preview_candidate_command(tmp_path, monkeypatch):
    """Test preview-candidate command CLI wrapper and arguments."""
    dummy_video = tmp_path / "mock_video.mp4"
    dummy_video.write_bytes(b"dummy video content")
    out_clip = tmp_path / "preview.mp4"

    called_args = {}

    def mock_preview_clip(video_path, start, end, output_path=None, open_player=True):
        called_args["video_path"] = video_path
        called_args["start"] = start
        called_args["end"] = end
        called_args["output_path"] = output_path
        called_args["open_player"] = open_player
        output_path.write_bytes(b"clipped mp4")
        return output_path

    monkeypatch.setattr("freecher_worker.cli.preview_clip", mock_preview_clip)

    res = runner.invoke(
        app,
        [
            "preview-candidate",
            str(dummy_video),
            "--start",
            "10.5",
            "--end",
            "42.0",
            "-o",
            str(out_clip),
            "--no-open",
        ],
    )
    assert res.exit_code == 0
    assert "Preview clip generated" in res.output
    assert called_args["start"] == 10.5
    assert called_args["end"] == 42.0
    assert called_args["open_player"] is False
    assert out_clip.is_file()
