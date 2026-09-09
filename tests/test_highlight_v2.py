"""Comprehensive tests for Highlight Intelligence v2 (highlight_v2)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from freecher_worker.cli import app
from freecher_worker.config import Settings
from freecher_worker.evaluation.metrics import compute_evaluation_metrics
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
from freecher_worker.scoring.heuristic import HeuristicScorer
from freecher_worker.scoring.llm import (
    OpenAILLMScorer,
    extract_surrounding_context,
    highlight_v2_score_formula_v1,
    SCORER_VERSION,
    PROMPT_VERSION,
    SCORE_FORMULA_VERSION,
)
from freecher_worker.transcription.models import Transcript, TranscriptSegment
from freecher_worker.utils.json_io import load_json, save_json

runner = CliRunner()


def _make_eval_item(cid: str, score: float | None) -> BlindEvaluationItem:
    return BlindEvaluationItem(
        candidate_id=cid,
        start=0.0,
        end=30.0,
        duration=30.0,
        text=f"Candidate text for {cid}",
        human_score=score,
        human_notes=f"Confidential note for {cid}",
    )


def _make_pred_item(cid: str, rank: int, score: float) -> ScorerPredictionItem:
    return ScorerPredictionItem(candidate_id=cid, rank=rank, score=score)


# ==============================================================================
# 1. Metric Definition: BadRate <= 2.0, PerfectRate >= 4.0
# ==============================================================================
def test_bad_rate_and_perfect_rate_definitions():
    """Verify PerfectRate is score >= 4.0 and BadRate is score <= 2.0."""
    cset_id = "cset_rates_test"
    eval_doc = BlindEvaluationDocument(
        candidate_set_id=cset_id,
        total_candidates=5,
        items=[
            _make_eval_item("c1", 4.0),  # Perfect
            _make_eval_item("c2", 3.0),  # Publishable, not bad
            _make_eval_item("c3", 2.5),  # Borderline, not bad (<=2 is bad)
            _make_eval_item("c4", 2.0),  # Bad
            _make_eval_item("c5", 1.0),  # Bad
        ],
    )
    eval_doc.update_labeled_count()

    pred_doc = ScorerPredictionDocument(
        candidate_set_id=cset_id,
        scorer="highlight_v2",
        scorer_version="highlight_v2",
        predictions=[
            _make_pred_item("c1", 1, 95.0),
            _make_pred_item("c2", 2, 85.0),
            _make_pred_item("c3", 3, 75.0),
            _make_pred_item("c4", 4, 65.0),
            _make_pred_item("c5", 5, 55.0),
        ],
    )

    metrics = compute_evaluation_metrics(eval_doc, pred_doc, k_values=[3, 5])

    # Top 3: c1(4.0), c2(3.0), c3(2.5)
    # Perfect count: 1 (c1) -> 1/3 = 33.33%
    # Bad count (<= 2.0): 0 -> 0%
    assert metrics.perfect_rate_at_k[3] == 0.3333
    assert metrics.bad_rate_at_k[3] == 0.0

    # Top 5: c1(4.0), c2(3.0), c3(2.5), c4(2.0), c5(1.0)
    # Perfect count: 1 (c1) -> 1/5 = 20%
    # Bad count (<= 2.0): 2 (c4, c5) -> 2/5 = 40%
    assert metrics.perfect_rate_at_k[5] == 0.20
    assert metrics.bad_rate_at_k[5] == 0.40


# ==============================================================================
# 2. Human labels are NEVER passed into scorer prompt
# ==============================================================================
def test_human_labels_never_passed_into_scorer_prompt():
    """Verify human scores, notes, and labels are never leaked into the LLM prompt."""
    cand = CandidateWindow(
        id="cand_test_leak",
        start=10.0,
        end=40.0,
        duration=30.0,
        text="A very interesting discussion on astronomy.",
    )

    captured_payload = {}

    def mock_post(url, headers, json):
        nonlocal captured_payload
        captured_payload = json
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json_payload_sample()
                    }
                }
            ]
        }
        return mock_resp

    scorer = OpenAILLMScorer(api_key="test-key", model="gpt-4o-mini")
    with patch("httpx.Client.post", side_effect=mock_post):
        scorer.score(cand)

    user_message = captured_payload["messages"][1]["content"]
    system_message = captured_payload["messages"][0]["content"]
    combined_text = user_message + " " + system_message

    # Ensure no human evaluation terms or values leaked
    forbidden_terms = [
        "human_score",
        "human_label",
        "human_notes",
        "Confidential note",
        "publishable",
    ]
    for term in forbidden_terms:
        assert term not in combined_text, f"Leaked forbidden term '{term}' into LLM prompt!"


def json_payload_sample():
    return json.dumps({
        "candidate_id": "cand_test_leak",
        "hook": 85.0,
        "standalone": 90.0,
        "story_payoff": 80.0,
        "emotion": 70.0,
        "humor": 60.0,
        "surprise": 75.0,
        "retention": 85.0,
        "shareability": 80.0,
        "boringness": 20.0,
        "context_dependency": 25.0,
        "setup_only": False,
        "transitional": False,
        "outside_payoff": False,
        "llm_quality_score": 82.0,
        "reason": "Strong hook and satisfying standalone conclusion.",
    })


# ==============================================================================
# 3. Candidate set ID preserved & mismatch guard in score-run
# ==============================================================================
def _create_mock_run_directory(run_dir: Path) -> str:
    run_dir.mkdir(parents=True, exist_ok=True)
    cset_id = compute_candidate_set_id(
        transcript_hash="hash_12345",
        min_seconds=30.0,
        target_seconds=60.0,
        max_seconds=90.0,
        overlap_seconds=15.0,
        segmentation_version="1.0.0",
    )
    cand_doc = CandidateDocument(
        candidate_set_id=cset_id,
        transcript_hash="hash_12345",
        min_seconds=30.0,
        target_seconds=60.0,
        max_seconds=90.0,
        overlap_seconds=15.0,
        candidates=[
            CandidateWindow(id="c01", start=0.0, end=30.0, duration=30.0, text="Intro clip"),
            CandidateWindow(id="c02", start=30.0, end=65.0, duration=35.0, text="Insight clip"),
        ],
    )
    save_json(cand_doc, run_dir / "candidates.json")

    transcript = Transcript(
        text="Intro clip Insight clip",
        language="ru",
        duration=65.0,
        model="small",
        compute_type="float16",
        device="cpu",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=30.0, text="Intro clip"),
            TranscriptSegment(id=1, start=30.0, end=65.0, text="Insight clip"),
        ],
    )
    save_json(transcript, run_dir / "transcript.json")

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


def test_candidate_set_id_unchanged_after_score_run(tmp_path):
    """Verify candidate_set_id in score-run output exactly matches source candidate_set_id."""
    run_dir = tmp_path / "run_set_id_test"
    cset_id = _create_mock_run_directory(run_dir)

    res = runner.invoke(app, ["score-run", str(run_dir), "--scorer", "heuristic"])
    assert res.exit_code == 0

    scores_path = run_dir / "scores" / "heuristic_v1.json"
    assert scores_path.is_file()
    data = load_json(scores_path)
    assert data["candidate_set_id"] == cset_id


def test_score_run_refuses_on_tampered_candidate_set_id(tmp_path):
    """Verify score-run aborts if candidate_set_id does not match its computed integrity hash."""
    run_dir = tmp_path / "run_tampered_test"
    _create_mock_run_directory(run_dir)

    # Tamper with candidate_set_id
    cand_path = run_dir / "candidates.json"
    data = load_json(cand_path)
    data["candidate_set_id"] = "cset_tampered_invalid"
    save_json(data, cand_path)

    res = runner.invoke(app, ["score-run", str(run_dir), "--scorer", "heuristic"])
    assert res.exit_code == 1
    assert "candidate_set_id integrity check failed" in res.output


# ==============================================================================
# 4. Benchmark score-run cannot silently mix fallback
# ==============================================================================
def test_benchmark_score_run_fails_without_allow_fallback(tmp_path, monkeypatch):
    """Verify score-run without --allow-fallback FAILS if LLM cannot score, preventing silent mixing."""
    run_dir = tmp_path / "run_fallback_purity"
    _create_mock_run_directory(run_dir)

    # Unset API key to simulate failure
    monkeypatch.delenv("FREECHER_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ARNY_LLM_API_KEY", raising=False)
    # delenv only clears the environment; Settings also reads the developer's
    # .env, so without this the test silently stops simulating "no API key"
    # the moment a real key is configured locally.
    monkeypatch.setattr(
        "freecher_worker.cli.get_settings", lambda: Settings(_env_file=None)
    )

    # 1. Default (no --allow-fallback): must fail!
    res = runner.invoke(app, ["score-run", str(run_dir), "--scorer", "highlight_v2"])
    assert res.exit_code != 0

    # 2. With --allow-fallback: allowed to fallback
    res_fallback = runner.invoke(app, ["score-run", str(run_dir), "--scorer", "highlight_v2", "--allow-fallback"])
    assert res_fallback.exit_code == 0

    scores_doc = load_json(run_dir / "scores" / "highlight_v2.json")
    assert scores_doc["predictions"][0]["fallback_used"] is True


# ==============================================================================
# 5. Formula, Clamping, Ceilings, and Flags
# ==============================================================================
def test_score_formula_penalties_and_ceilings():
    """Verify score formula applies strict penalties for transitional, setup_only, outside_payoff, and boringness."""
    # Base great clip
    good_features = {
        "hook": 90.0,
        "standalone": 90.0,
        "story_payoff": 90.0,
        "retention": 90.0,
        "shareability": 85.0,
        "humor": 80.0,
        "surprise": 80.0,
        "emotion": 80.0,
        "boringness": 10.0,
        "context_dependency": 10.0,
        "llm_quality_score": 90.0,
    }
    score_good, _ = highlight_v2_score_formula_v1(good_features)
    assert score_good >= 85.0

    # Transitional clip (e.g. stream banter / setup) must have ceiling <= 35.0
    transitional_features = dict(good_features)
    transitional_features["transitional"] = True
    score_trans, _ = highlight_v2_score_formula_v1(transitional_features)
    assert score_trans <= 35.0

    # Setup only clip must have ceiling <= 40.0
    setup_features = dict(good_features)
    setup_features["setup_only"] = True
    score_setup, _ = highlight_v2_score_formula_v1(setup_features)
    assert score_setup <= 40.0

    # Outside payoff must have ceiling <= 45.0
    outside_features = dict(good_features)
    outside_features["outside_payoff"] = True
    score_outside, _ = highlight_v2_score_formula_v1(outside_features)
    assert score_outside <= 45.0

    # High boringness (>= 75) must have ceiling <= 35.0
    boring_features = dict(good_features)
    boring_features["boringness"] = 80.0
    score_boring, _ = highlight_v2_score_formula_v1(boring_features)
    assert score_boring <= 35.0


def test_score_formula_clamping():
    """Verify out-of-range feature values are clamped into [0.0, 100.0]."""
    extreme_features = {
        "hook": 999.0,
        "standalone": -50.0,
        "story_payoff": 150.0,
        "retention": 200.0,
        "boringness": -100.0,
        "llm_quality_score": 120.0,
    }
    score, breakdown = highlight_v2_score_formula_v1(extreme_features)
    assert 0.0 <= score <= 100.0
    for val in breakdown.values():
        assert 0.0 <= val <= 100.0


# ==============================================================================
# 6. JSON Parsing & Markdown Stripping & Retries
# ==============================================================================
def test_highlight_v2_markdown_stripping():
    """Verify OpenAILLMScorer properly strips ```json markdown fences."""
    cand = CandidateWindow(id="c01", start=0.0, end=10.0, duration=10.0, text="Sample text")

    fenced_content = f"```json\n{json_payload_sample()}\n```"

    def mock_post(*args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": fenced_content}}]}
        return resp

    scorer = OpenAILLMScorer(api_key="sk-test", model="gpt-4o-mini")
    with patch("httpx.Client.post", side_effect=mock_post):
        score_res = scorer.score(cand)

    assert score_res.fallback_used is False
    assert score_res.score > 0
    assert score_res.scorer_version == "highlight_v2"


def test_highlight_v2_retries_on_malformed_response():
    """Verify OpenAILLMScorer retries when LLM returns invalid JSON."""
    cand = CandidateWindow(id="c01", start=0.0, end=10.0, duration=10.0, text="Sample text")

    attempts = 0

    def mock_post(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        resp = MagicMock()
        resp.status_code = 200
        if attempts == 1:
            resp.json.return_value = {"choices": [{"message": {"content": "Not valid json {"}}]}
        else:
            resp.json.return_value = {"choices": [{"message": {"content": json_payload_sample()}}]}
        return resp

    scorer = OpenAILLMScorer(api_key="sk-test", model="gpt-4o-mini", max_retries=3)
    with patch("httpx.Client.post", side_effect=mock_post):
        score_res = scorer.score(cand)

    assert attempts == 2
    assert score_res.fallback_used is False


# ==============================================================================
# 7. Context Extraction Boundaries
# ==============================================================================
def test_extract_surrounding_context():
    """Verify surrounding context accurately extracts preceding and following segments within window."""
    transcript = Transcript(
        text="p1 p2 c1 c2 n1 n2",
        language="ru",
        duration=100.0,
        model="small",
        compute_type="float16",
        device="cpu",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=10.0, text="p1"),
            TranscriptSegment(id=1, start=10.0, end=20.0, text="p2"),
            TranscriptSegment(id=2, start=25.0, end=40.0, text="c1"),
            TranscriptSegment(id=3, start=40.0, end=55.0, text="c2"),
            TranscriptSegment(id=4, start=60.0, end=75.0, text="n1"),
            TranscriptSegment(id=5, start=75.0, end=90.0, text="n2"),
        ],
    )
    cand = CandidateWindow(id="cand_test", start=25.0, end=55.0, duration=30.0, text="c1 c2")

    ctx = extract_surrounding_context(cand, transcript, context_window_seconds=40.0)
    # Segments before cand_start (25.0):
    # seg 1: [10, 20], 25 - 10 = 15 <= 40 -> included
    # seg 0: [0, 10], 25 - 0 = 25 <= 40 -> included
    assert "p1" in ctx["previous_context"]
    assert "p2" in ctx["previous_context"]

    # Segments after cand_end (55.0):
    # seg 4: [60, 75], 75 - 55 = 20 <= 40 -> included
    # seg 5: [75, 90], 90 - 55 = 35 <= 40 -> included
    assert "n1" in ctx["next_context"]
    assert "n2" in ctx["next_context"]

    assert ctx["silence_before_sec"] == 5.0  # 25.0 - 20.0
    assert ctx["silence_after_sec"] == 5.0   # 60.0 - 55.0


# ==============================================================================
# 8. Baseline Heuristic V1 Preservation
# ==============================================================================
def test_heuristic_v1_baseline_preserved():
    """Verify HeuristicScorer version is heuristic_v1 and produces expected deterministic score."""
    scorer = HeuristicScorer()
    assert scorer.name == "heuristic"
    assert scorer.version == "heuristic_v1"

    cand = CandidateWindow(
        id="c01",
        start=0.0,
        end=30.0,
        duration=30.0,
        text="Почему 90% совершают эту ошибку? Секрет в том, что результат исследования просто шок!",
    )
    score = scorer.score(cand)
    assert score.score > 0
    assert score.hook_score > 0
