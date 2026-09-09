"""Tests for Highlight Intelligence v2.1:
- score calibration & resolution (0-100)
- streamer-aware archetypes (humor without narrative arc)
- negative penalty calibration & caps
- ASR corruption resilience & recalibrated negative dimensions in prompt
- strict field parsing without silent imputation
- distribution diagnostics & collapse detection (including zero-score collapse)
- human-label isolation
- benchmark purity (no silent fallback)
- preservation of heuristic_v1 and highlight_v2 baselines
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from freecher_worker.cli import app
from freecher_worker.config import Settings
from freecher_worker.utils.json_io import load_json, save_json
from freecher_worker.highlights.models import (
    CandidateDocument,
    CandidateWindow,
    compute_candidate_set_id,
)
from freecher_worker.scoring.heuristic import HeuristicScorer
from freecher_worker.scoring.llm import (
    PROMPT_HASH_V2,
    PROMPT_HASH_V2_1,
    PROMPT_VERSION_V2,
    PROMPT_VERSION_V2_1,
    REQUIRED_FIELDS_V2_1,
    SCORER_VERSION_V2,
    SCORER_VERSION_V2_1,
    SYSTEM_PROMPT_V2,
    SYSTEM_PROMPT_V2_1,
    OpenAILLMScorer,
    compute_score_distribution,
    highlight_v2_1_formula_v1,
    highlight_v2_score_formula_v1,
)
from freecher_worker.transcription.models import Transcript, TranscriptSegment

runner = CliRunner()


def _create_mock_run_directory(run_dir: Path) -> str:
    run_dir.mkdir(parents=True, exist_ok=True)
    cset_id = compute_candidate_set_id(
        transcript_hash="hash_mock_123",
        min_seconds=30.0,
        target_seconds=60.0,
        max_seconds=90.0,
        overlap_seconds=15.0,
        segmentation_version="1.0.0",
    )
    cand_doc = CandidateDocument(
        candidate_set_id=cset_id,
        transcript_hash="hash_mock_123",
        min_seconds=30.0,
        target_seconds=60.0,
        max_seconds=90.0,
        overlap_seconds=15.0,
        candidates=[
            CandidateWindow(id="c01", start=0.0, end=30.0, duration=30.0, text="Intro streamer clip"),
            CandidateWindow(id="c02", start=30.0, end=65.0, duration=35.0, text="Funny banter clip"),
        ],
    )
    save_json(cand_doc, run_dir / "candidates.json")

    transcript = Transcript(
        text="Intro streamer clip Funny banter clip",
        language="ru",
        duration=65.0,
        model="small",
        compute_type="float16",
        device="cpu",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=30.0, text="Intro streamer clip"),
            TranscriptSegment(id=1, start=30.0, end=65.0, text="Funny banter clip"),
        ],
    )
    save_json(transcript, run_dir / "transcript.json")

    manifest = {
        "pipeline_version": "0.1.2",
        "created_at": "2026-09-08T10:00:00",
        "source": "/mock/stream.mp4",
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


# ==============================================================================
# 1. Score Calibration & No Collapse
# ==============================================================================
def test_score_formula_resolution_and_no_severe_multiplier():
    """Verify that 25 quality points do not become 6.25 and ordinary candidates are not blanket reduced."""
    features = {
        "hook": 50.0,
        "standalone": 50.0,
        "story_payoff": 50.0,
        "emotion": 50.0,
        "humor": 50.0,
        "surprise": 50.0,
        "retention": 50.0,
        "shareability": 50.0,
        "boringness": 20.0,
        "context_dependency": 25.0,
        "setup_only": False,
        "transitional": False,
        "outside_payoff": False,
        "llm_quality_score": 50.0,
    }
    # positive_score = 50.0 * 1.0 = 50.0
    # penalty = 20 * 0.25 + 25 * 0.12 = 5.0 + 3.0 = 8.0
    # raw_score = 50.0 - 8.0 = 42.0
    final_score, subscores, diags = highlight_v2_1_formula_v1(features)
    assert final_score == 42.0
    # In v2, this would have been crushed by multiplier/blending; here it is cleanly 42.0
    assert final_score > 20.0


def test_streamer_humor_without_narrative_arc_scores_well():
    """Verify that a strong comedic streamer moment with low story_payoff can score > 60."""
    features = {
        "hook": 85.0,
        "standalone": 80.0,
        "story_payoff": 20.0,  # No traditional narrative arc!
        "emotion": 60.0,
        "humor": 90.0,
        "surprise": 65.0,
        "retention": 85.0,
        "shareability": 80.0,
        "boringness": 15.0,
        "context_dependency": 20.0,
        "setup_only": False,
        "transitional": False,
        "outside_payoff": False,
    }
    # positive = 85*0.12 + 80*0.14 + 20*0.18 + 60*0.08 + 90*0.10 + 65*0.08 + 85*0.18 + 80*0.12
    #          = 10.2 + 11.2 + 3.6 + 4.8 + 9.0 + 5.2 + 15.3 + 9.6 = 68.9
    # penalty = 15*0.25 + 20*0.12 = 3.75 + 2.4 = 6.15
    # raw_score = 68.9 - 6.15 = 62.75
    final_score, subscores, diags = highlight_v2_1_formula_v1(features)
    assert final_score == 62.75
    assert final_score > 60.0
    assert diags["applied_caps"] == []


# ==============================================================================
# 2. Penalties and Hard Caps
# ==============================================================================
def test_setup_only_content_penalized_and_capped():
    """Verify setup_only=True applies -15 penalty and caps at 40 if payoff < 30."""
    features = {
        "hook": 80.0,
        "standalone": 80.0,
        "story_payoff": 25.0,  # < 30
        "emotion": 50.0,
        "humor": 50.0,
        "surprise": 50.0,
        "retention": 80.0,
        "shareability": 70.0,
        "boringness": 10.0,
        "context_dependency": 10.0,
        "setup_only": True,
        "transitional": False,
        "outside_payoff": True,
    }
    final_score, subscores, diags = highlight_v2_1_formula_v1(features)
    assert diags["total_penalty"] > 15.0  # includes 15 boolean penalty
    assert "setup_only_low_payoff:40" in diags["applied_caps"]
    assert final_score <= 40.0


def test_transitional_content_penalized_and_capped():
    """Verify transitional=True applies -12 penalty and caps at 35 if retention < 30."""
    features = {
        "hook": 60.0,
        "standalone": 60.0,
        "story_payoff": 40.0,
        "emotion": 40.0,
        "humor": 40.0,
        "surprise": 40.0,
        "retention": 20.0,  # < 30
        "shareability": 40.0,
        "boringness": 20.0,
        "context_dependency": 20.0,
        "setup_only": False,
        "transitional": True,
        "outside_payoff": False,
    }
    final_score, subscores, diags = highlight_v2_1_formula_v1(features)
    assert "transitional_low_retention:35" in diags["applied_caps"]
    assert final_score <= 35.0


def test_extreme_boringness_capped():
    """Verify boringness >= 85 and story_payoff < 25 caps score at 30."""
    features = {
        "hook": 50.0,
        "standalone": 50.0,
        "story_payoff": 20.0,  # < 25
        "emotion": 30.0,
        "humor": 20.0,
        "surprise": 20.0,
        "retention": 40.0,
        "shareability": 30.0,
        "boringness": 90.0,  # >= 85
        "context_dependency": 10.0,
        "setup_only": False,
        "transitional": False,
        "outside_payoff": False,
    }
    final_score, subscores, diags = highlight_v2_1_formula_v1(features)
    assert "extreme_boringness:30" in diags["applied_caps"]
    assert final_score <= 30.0


def test_clamping_to_zero_and_hundred():
    """Verify final score is strictly clamped to [0.0, 100.0]."""
    # Extremely negative features
    low_features = {
        "hook": 0.0,
        "standalone": 0.0,
        "story_payoff": 0.0,
        "emotion": 0.0,
        "humor": 0.0,
        "surprise": 0.0,
        "retention": 0.0,
        "shareability": 0.0,
        "boringness": 100.0,
        "context_dependency": 100.0,
        "setup_only": True,
        "transitional": True,
        "outside_payoff": True,
    }
    low_score, _, _ = highlight_v2_1_formula_v1(low_features)
    assert low_score == 0.0

    # Perfect features
    high_features = {
        "hook": 100.0,
        "standalone": 100.0,
        "story_payoff": 100.0,
        "emotion": 100.0,
        "humor": 100.0,
        "surprise": 100.0,
        "retention": 100.0,
        "shareability": 100.0,
        "boringness": 0.0,
        "context_dependency": 0.0,
        "setup_only": False,
        "transitional": False,
        "outside_payoff": False,
    }
    high_score, _, _ = highlight_v2_1_formula_v1(high_features)
    assert high_score == 100.0


# ==============================================================================
# 3. Prompt Guidance: ASR Corruption & Recalibrated Dimensions
# ==============================================================================
def test_prompt_asr_corruption_resilience():
    """Verify SYSTEM_PROMPT_V2_1 explicitly guides model to ignore ASR corruption."""
    assert "Do not interpret obvious ASR corruption" in SYSTEM_PROMPT_V2_1
    assert "slang transcription mistakes" in SYSTEM_PROMPT_V2_1
    assert "punctuation errors" in SYSTEM_PROMPT_V2_1
    assert "Judge the likely spoken interaction" in SYSTEM_PROMPT_V2_1


def test_prompt_recalibrated_negative_dimensions():
    """Verify SYSTEM_PROMPT_V2_1 defines 80+ boringness/context_dep only for extreme low-event/lore cases."""
    assert "boringness (0-100):" in SYSTEM_PROMPT_V2_1
    assert "context_dependency (0-100):" in SYSTEM_PROMPT_V2_1
    assert "80-100: ONLY for genuinely low-event filler" in SYSTEM_PROMPT_V2_1
    assert "80-100: practically IMPOSSIBLE to appreciate without deep prior lore" in SYSTEM_PROMPT_V2_1
    assert "Casual streamer familiarity does NOT mean extreme context dependency." in SYSTEM_PROMPT_V2_1


def test_prompt_streamer_archetypes():
    """Verify SYSTEM_PROMPT_V2_1 instructs model that traditional story arc is not mandatory."""
    assert "MULTIPLE VALID HIGHLIGHT ARCHETYPES:" in SYSTEM_PROMPT_V2_1
    assert "joke / punchline / comedic timing" in SYSTEM_PROMPT_V2_1
    assert "absurd, witty, or funny streamer banter" in SYSTEM_PROMPT_V2_1
    assert "A short-form clip does NOT require a traditional narrative arc" in SYSTEM_PROMPT_V2_1


# ==============================================================================
# 4. Strict Parsing & Retry on Missing Required Fields
# ==============================================================================
def test_missing_required_fields_trigger_retry():
    """Verify that omitting required fields raises ValueError and triggers retry."""
    scorer = OpenAILLMScorer(
        api_key="mock_key",
        scorer_version=SCORER_VERSION_V2_1,
        max_retries=2,
    )
    cand = CandidateWindow(id="c01", start=0.0, end=30.0, duration=30.0, text="Hello world")

    # Incomplete response missing 'humor' and 'reason'
    incomplete_json = {
        "candidate_id": "c01",
        "hook": 80.0,
        "standalone": 80.0,
        "story_payoff": 80.0,
        "emotion": 80.0,
        "surprise": 80.0,
        "retention": 80.0,
        "shareability": 80.0,
        "boringness": 10.0,
        "context_dependency": 10.0,
        "setup_only": False,
        "transitional": False,
        "outside_payoff": False,
        # 'humor' is missing!
        # 'reason' is missing!
    }

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps(incomplete_json)}}]
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        # allow_fallback=False should raise RuntimeError because all retries failed due to missing fields
        scorer.allow_fallback = False
        with pytest.raises(RuntimeError, match="Missing required ranking field 'humor'"):
            scorer.score(cand)


# ==============================================================================
# 5. Distribution Diagnostics & Zero-Score Collapse Detection
# ==============================================================================
def test_distribution_diagnostics_calculation_and_collapse_warning():
    """Verify percentiles, unique counts, std dev, and collapse detection."""
    # Varied spread
    scores = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    unrounded = [10.12, 20.34, 30.56, 40.78, 50.90, 60.11, 70.22, 80.33, 90.44, 99.88]
    diag = compute_score_distribution(scores, unrounded)

    assert diag.min == 10.0
    assert diag.max == 100.0
    assert diag.median == 55.0
    assert diag.unique_score_count_raw == 10
    assert diag.unique_score_count_rounded == 10
    assert diag.zero_score_count == 0
    assert diag.warning is None

    # Test collapse scenario: scores collapsed into few identical values
    collapsed_scores = [3.75] * 30 + [5.0] * 12
    diag_collapsed = compute_score_distribution(collapsed_scores)
    assert diag_collapsed.unique_score_count_rounded == 2
    assert diag_collapsed.warning is not None
    assert "Score collapse warning" in diag_collapsed.warning

    # Test zero-score collapse scenario
    zero_collapsed = [0.0] * 20 + [50.0] * 10
    diag_zero = compute_score_distribution(zero_collapsed)
    assert diag_zero.zero_score_count == 20
    assert diag_zero.warning is not None
    assert "Zero-score collapse warning" in diag_zero.warning


# ==============================================================================
# 6. Human-Label Isolation & Benchmark Purity
# ==============================================================================
def test_human_label_isolation_score_run_never_reads_labels(tmp_path):
    """Verify score-run NEVER reads evaluation_blind.json or injects human labels into scorer."""
    run_dir = tmp_path / "run_isolation_test"
    _create_mock_run_directory(run_dir)

    # Place an evaluation_blind.json with sensitive labels
    eval_doc = {
        "candidate_set_id": "cset_mock_123",
        "items": [
            {"candidate_id": "c01", "human_score": 4.0, "publishable": True, "human_notes": "Secret annotation"},
            {"candidate_id": "c02", "human_score": 0.0, "publishable": False, "human_notes": "Garbage"},
        ],
    }
    save_json(eval_doc, run_dir / "evaluation_blind.json")

    # Spy on OpenAILLMScorer.score to check that prompt user_content never receives "Secret annotation" or 4.0
    with patch.object(OpenAILLMScorer, "score") as mock_score:
        mock_score.return_value = MagicMock(
            score=75.0,
            hook_score=75.0,
            standalone_score=75.0,
            emotion_score=75.0,
            information_score=75.0,
            shareability_score=75.0,
            reason="Mocked reason",
            fallback_used=False,
            fallback_reason=None,
            final_score=75.0,
            subscores={},
            flags={},
            positive_score=75.0,
            total_penalty=0.0,
            applied_caps=[],
            raw_positive_dimensions={},
            raw_negative_dimensions={},
            actual_model="mock-model",
        )
        res = runner.invoke(app, ["score-run", str(run_dir), "--scorer", "highlight_v2_1"])
        assert res.exit_code == 0

        # Check call arguments to ensure no human annotations leaked
        for call_args in mock_score.call_args_list:
            cand_arg = call_args[0][0]
            ctx_arg = call_args[1].get("context") if "context" in call_args[1] else None
            cand_str = str(cand_arg) + str(ctx_arg)
            assert "Secret annotation" not in cand_str
            assert "human_score" not in cand_str
            assert "publishable" not in cand_str


def test_benchmark_score_run_cannot_silently_mix_llm_and_heuristic(tmp_path, monkeypatch):
    """Verify score-run fails non-zero if LLM fails and does not produce disguised heuristic predictions."""
    run_dir = tmp_path / "run_purity_test"
    _create_mock_run_directory(run_dir)

    monkeypatch.delenv("FREECHER_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ARNY_LLM_API_KEY", raising=False)
    # delenv only clears the environment; Settings also reads the developer's
    # .env, so without this the test silently stops simulating "no API key"
    # the moment a real key is configured locally.
    monkeypatch.setattr(
        "freecher_worker.cli.get_settings", lambda: Settings(_env_file=None)
    )

    # 1. Default (no --allow-fallback): MUST FAIL!
    res = runner.invoke(app, ["score-run", str(run_dir), "--scorer", "highlight_v2_1"])
    assert res.exit_code != 0
    # Destination file should NOT exist
    assert not (run_dir / "scores" / "highlight_v2_1.json").is_file()

    # 2. With --allow-fallback: produces output with explicit fallback flags
    res_fallback = runner.invoke(
        app, ["score-run", str(run_dir), "--scorer", "highlight_v2_1", "--allow-fallback"]
    )
    assert res_fallback.exit_code == 0
    pred_path = run_dir / "scores" / "highlight_v2_1.json"
    assert pred_path.is_file()
    saved_doc = load_json(pred_path)
    assert saved_doc["predictions"][0]["fallback_used"] is True
    assert saved_doc["predictions"][0]["actual_model"] == "heuristic_v1"


# ==============================================================================
# 7. Scorer Metadata and Aliases
# ==============================================================================
def test_requested_scorer_alias_metadata_in_prediction_document(tmp_path):
    """Verify metadata reflects requested_scorer='llm' -> actual_scorer='highlight_v2_1'."""
    run_dir = tmp_path / "run_metadata_test"
    _create_mock_run_directory(run_dir)

    mock_score = MagicMock(
        score=70.0,
        hook_score=70.0,
        standalone_score=70.0,
        emotion_score=70.0,
        information_score=70.0,
        shareability_score=70.0,
        reason="Mocked reason",
        fallback_used=False,
        fallback_reason=None,
        final_score=70.0,
        subscores={},
        flags={},
        positive_score=70.0,
        total_penalty=0.0,
        applied_caps=[],
        raw_positive_dimensions={},
        raw_negative_dimensions={},
        actual_model="mock-model",
    )

    with patch.object(OpenAILLMScorer, "score", return_value=mock_score):
        # 1. Test alias --scorer llm
        res = runner.invoke(app, ["score-run", str(run_dir), "--scorer", "llm", "--model", "mock-model"])
        assert res.exit_code == 0
        doc_v2_1 = load_json(run_dir / "scores" / "highlight_v2_1.json")
        assert doc_v2_1["requested_scorer"] == "llm"
        assert doc_v2_1["actual_scorer"] == "highlight_v2_1"
        assert doc_v2_1["scorer_version"] == "highlight_v2_1"

        # 2. Test explicit --scorer highlight_v2
        res_v2 = runner.invoke(
            app, ["score-run", str(run_dir), "--scorer", "highlight_v2", "--model", "mock-model"]
        )
        assert res_v2.exit_code == 0
        doc_v2 = load_json(run_dir / "scores" / "highlight_v2.json")
        assert doc_v2["requested_scorer"] == "highlight_v2"
        assert doc_v2["actual_scorer"] == "highlight_v2"
        assert doc_v2["scorer_version"] == "highlight_v2"


# ==============================================================================
# 8. Baseline Preservations
# ==============================================================================
def test_baselines_preserved():
    """Verify heuristic_v1 and highlight_v2 baselines remain identical and reproducible."""
    h_scorer = HeuristicScorer()
    assert h_scorer.version == "heuristic_v1"

    assert SCORER_VERSION_V2 == "highlight_v2"
    assert PROMPT_VERSION_V2 == "highlight_v2_prompt_v1"
    assert len(PROMPT_HASH_V2) == 16

    assert SCORER_VERSION_V2_1 == "highlight_v2_1"
    assert PROMPT_VERSION_V2_1 == "highlight_v2_1_prompt_v1"
    assert len(PROMPT_HASH_V2_1) == 16
