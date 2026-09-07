"""Tests for CLI inspect and export-eval commands."""

from pathlib import Path
from typer.testing import CliRunner
from freecher_worker.cli import app
from freecher_worker.highlights.models import CandidateDocument, CandidateWindow, Highlight, HighlightScore
from freecher_worker.media.fingerprint import SourceFingerprint
from freecher_worker.pipeline.processor import (
    AsrManifestInfo,
    CandidateConfigInfo,
    EnvironmentInfo,
    HighlightManifestItem,
    Manifest,
    PipelineStatistics,
    PipelineTimings,
    RankingManifestInfo,
    ScoringManifestInfo,
)
from freecher_worker.utils.json_io import load_json, save_json

runner = CliRunner()


def _create_mock_run(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    c1 = CandidateWindow(id="cand_001", start=10.0, end=40.0, duration=30.0, text="First candidate text", segment_ids=[0])
    c2 = CandidateWindow(id="cand_002", start=50.0, end=90.0, duration=40.0, text="Second candidate text", segment_ids=[1])

    cand_doc = CandidateDocument(
        transcript_hash="hash123",
        min_seconds=30.0,
        target_seconds=60.0,
        max_seconds=90.0,
        overlap_seconds=15.0,
        candidates=[c1, c2],
    )
    save_json(cand_doc, run_dir / "candidates.json")

    score_item = HighlightScore(
        score=85.0,
        hook_score=90.0,
        standalone_score=80.0,
        emotion_score=75.0,
        information_score=85.0,
        shareability_score=88.0,
        reason="Strong hook and numbers",
        fallback_used=False,
    )

    hl = Highlight(
        rank=1,
        start=50.0,
        end=90.0,
        duration=40.0,
        score=85.0,
        reason="Strong hook and numbers",
        candidate_id="cand_002",
        text="Second candidate text",
        file="clips/clip_01.mp4",
        score_breakdown=score_item,
    )
    save_json([hl], run_dir / "highlights.json")

    manifest = Manifest(
        pipeline_version="0.1.1",
        created_at="2026-09-07T09:00:00",
        source="/path/to/video.mp4",
        source_fingerprint=SourceFingerprint(
            path="/path/to/video.mp4",
            file_size=1024,
            mtime_ns=123456789,
            duration_seconds=100.0,
            content_hash="abc123hash",
            fingerprint_id="fp_test123",
        ),
        environment=EnvironmentInfo(python_version="3.12.0", platform="Linux"),
        asr=AsrManifestInfo(model="small", device="cuda", compute_type="int8_float16", language="ru"),
        candidate_config=CandidateConfigInfo(min_seconds=30.0, target_seconds=60.0, max_seconds=90.0, overlap=15.0),
        scoring=ScoringManifestInfo(scorer="heuristic", scorer_version="1.1.0"),
        ranking=RankingManifestInfo(top_k=5, dedup_threshold=0.60),
        timings=PipelineTimings(total_seconds=42.5),
        statistics=PipelineStatistics(transcript_segment_count=10, candidate_count=2, selected_highlight_count=1),
        highlights=[
            HighlightManifestItem(
                rank=1,
                start=50.0,
                end=90.0,
                duration=40.0,
                score=85.0,
                reason="Strong hook and numbers",
                file="clips/clip_01.mp4",
                candidate_id="cand_002",
                score_breakdown=score_item,
            )
        ],
    )
    save_json(manifest, run_dir / "manifest.json")
    return run_dir


def test_inspect_command(tmp_path):
    run_dir = _create_mock_run(tmp_path / "mock_run")
    result = runner.invoke(app, ["inspect", str(run_dir)])

    assert result.exit_code == 0
    assert "Run Inspection" in result.output
    assert "#1" in result.output
    assert "Score: 85.0" in result.output
    assert "Strong hook and numbers" in result.output


def test_export_eval_command(tmp_path):
    run_dir = _create_mock_run(tmp_path / "mock_run")
    eval_file = run_dir / "evaluation.json"

    result = runner.invoke(app, ["export-eval", str(run_dir)])
    assert result.exit_code == 0
    assert eval_file.is_file()

    eval_data = load_json(eval_file)
    assert len(eval_data) == 2  # both candidates exported

    top = eval_data[0]
    assert top["candidate_id"] == "cand_002"
    assert top["rank"] == 1
    assert top["score"] == 85.0
    assert top["human_label"] is None
    assert top["human_score"] is None
    assert top["human_notes"] is None
    assert top["subscores"]["hook_score"] == 90.0

    # Second candidate was not ranked in top highlights
    second = eval_data[1]
    assert second["candidate_id"] == "cand_001"
    assert second["human_score"] is None
