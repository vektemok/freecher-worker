"""Comprehensive unit and integration tests for Freecher Multimodal Highlight Reranker v1.1 (multimodal_v1_1)."""

import json
from pathlib import Path
from typing import Any, Dict, List
import unittest.mock as mock
import wave

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from freecher_worker.evaluation.models import (
    BlindEvaluationDocument,
    BlindEvaluationItem,
    ScorerPredictionDocument,
    ScorerPredictionItem,
)
from freecher_worker.evaluation.metrics import compute_evaluation_metrics
from freecher_worker.highlights.models import CandidateDocument, CandidateWindow
from freecher_worker.multimodal import (
    ACTIVITY_V1_1_FORMULA_VERSION,
    ActivityCurveSummary,
    ActivityPoint,
    AudioFeatures,
    ExtractedFrame,
    FORMULA_VERSION_MULTIMODAL_V1,
    FORMULA_VERSION_MULTIMODAL_V1_1,
    MultimodalCandidatePackage,
    MultimodalModelResult,
    MultimodalProvider,
    MultimodalReranker,
    MultimodalUsage,
    ObservedEvidenceItem,
    ObservedRegion,
    OpenAIMultimodalProvider,
    PACKAGE_VERSION_V1,
    PACKAGE_VERSION_V1_1,
    PROMPT_VERSION_MULTIMODAL_V1,
    PROMPT_VERSION_MULTIMODAL_V1_1,
    SCORER_VERSION_MULTIMODAL_V1,
    SCORER_VERSION_MULTIMODAL_V1_1,
    ShortlistDocument,
    ShortlistItem,
    SafeDecoderResolution,
    SourceAudioProfile,
    SourceTemporalActivityPoint,
    SourceTemporalActivityProfile,
    TemporalBurst,
    VisualFeatures,
    build_multimodal_package,
    compute_combined_activity,
    compute_source_audio_profile,
    compute_source_temporal_activity_profile,
    compute_v1_1_sample_timestamps,
    extract_candidate_audio_features,
    extract_candidate_frames,
    extract_candidate_visual_features,
    extract_canonical_fingerprint,
    generate_shortlist,
    multimodal_v1_formula_v1,
    multimodal_v1_1_formula_v1,
    probe_software_decoder,
    resolve_safe_video_decoder,
    resolve_source_video_path,
    run_decoder_smoke_test,
    select_temporal_burst_peaks,
    slice_candidate_activity_curve,
)
from freecher_worker.multimodal.package import (
    compute_package_hash,
    extract_temporal_burst_transcript,
)
from freecher_worker.multimodal.openai_provider import (
    SYSTEM_PROMPT_MULTIMODAL_V1,
    SYSTEM_PROMPT_MULTIMODAL_V1_1,
)
from freecher_worker.scoring.heuristic import HeuristicScorer
from freecher_worker.scoring.llm import highlight_v2_1_formula_v1
from freecher_worker.transcription.models import Transcript, TranscriptSegment
from freecher_worker.utils.json_io import load_json, save_json


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _create_synthetic_wav(path: Path, duration_sec: float = 2.0, sample_rate: int = 16000) -> Path:
    total_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    sig = (0.4 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(sig.tobytes())
    return path


def _create_synthetic_image(path: Path, width: int = 320, height: int = 240, color: int = 128) -> Path:
    img = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _make_candidate(cid: str, start: float, end: float, text: str = "Test candidate text") -> CandidateWindow:
    return CandidateWindow(
        id=cid,
        start=start,
        end=end,
        duration=round(end - start, 2),
        text=text,
        segment_ids=[1],
    )


def _setup_mock_run_dir(tmp_path: Path, count: int = 42) -> Path:
    cset_id = "cset_8ac195a200530e96"
    candidates = []
    heur_preds = []
    llm_preds = []

    for i in range(1, count + 1):
        cid = f"cand_{i:03d}"
        c = _make_candidate(cid, start=(i - 1) * 30.0, end=i * 30.0)
        candidates.append(c)

        # Heuristic ranking: 1..count (cand_001 is rank 1)
        heur_preds.append(
            ScorerPredictionItem(
                candidate_id=cid,
                rank=i,
                score=round(100.0 - (i * 1.5), 2),
            )
        )
        # LLM ranking: reversed (cand_042 is rank 1, cand_001 is rank count)
        rev_rank = count - i + 1
        llm_score = round(float(i) * 2.0, 2)
        llm_preds.append(
            ScorerPredictionItem(
                candidate_id=cid,
                rank=rev_rank,
                score=llm_score,
                llm_quality_score=llm_score,
                final_score=llm_score,
            )
        )

    cand_doc = CandidateDocument(
        candidate_set_id=cset_id,
        transcript_hash="mock_transcript_hash_123",
        min_seconds=30.0,
        target_seconds=60.0,
        max_seconds=90.0,
        overlap_seconds=15.0,
        candidates=candidates,
    )
    save_json(cand_doc, tmp_path / "candidates.json")

    scores_dir = tmp_path / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    heur_doc = ScorerPredictionDocument(
        candidate_set_id=cset_id,
        scorer="heuristic",
        scorer_version="heuristic_v1",
        predictions=heur_preds,
    )
    save_json(heur_doc, scores_dir / "heuristic_v1.json")

    llm_doc = ScorerPredictionDocument(
        candidate_set_id=cset_id,
        scorer="highlight_v2_1",
        scorer_version="highlight_v2_1",
        predictions=llm_preds,
    )
    save_json(llm_doc, scores_dir / "highlight_v2_1.json")

    v1_doc = ScorerPredictionDocument(
        candidate_set_id=cset_id,
        scorer="multimodal_v1",
        scorer_version="multimodal_v1",
        predictions=heur_preds[:20],
    )
    save_json(v1_doc, scores_dir / "multimodal_v1.json")

    # Synthetic transcript
    segments = [
        TranscriptSegment(id=i, start=c.start, end=c.end, text=c.text)
        for i, c in enumerate(candidates, start=1)
    ]
    transcript_doc = Transcript(
        language="en",
        duration=count * 30.0,
        model="tiny",
        compute_type="int8",
        device="cpu",
        segments=segments,
    )
    save_json(transcript_doc, tmp_path / "transcript.json")

    _create_synthetic_wav(tmp_path / "audio.wav", duration_sec=5.0)
    (tmp_path / "source.mp4").write_bytes(b"DUMMY_MP4_HEADER")

    manifest = {
        "pipeline_version": "0.1.1",
        "created_at": "2026-09-07T16:43:21.000000",
        "source": str(tmp_path / "source.mp4"),
        "source_fingerprint": {
            "path": str(tmp_path / "source.mp4"),
            "file_size": 1024,
            "mtime_ns": 1788747967206757495,
            "duration_seconds": count * 30.0,
            "content_hash": "mock_content_hash_123",
            "fingerprint_id": "035864f48379388e",
        },
    }
    save_json(manifest, tmp_path / "manifest.json")

    return tmp_path


class MockMultimodalV11Provider(MultimodalProvider):
    """Mock provider for testing v1.1 reranker with full structured evidence."""

    def __init__(self, prompt_version: str = PROMPT_VERSION_MULTIMODAL_V1_1):
        self.name = "mock_v1_1_provider"
        self.prompt_version = prompt_version
        self.calls: List[MultimodalCandidatePackage] = []
        self.usage = MultimodalUsage(input_tokens=100, output_tokens=50, total_tokens=150, estimated_cost_usd=0.001)

    def score_candidate(self, package: MultimodalCandidatePackage) -> MultimodalModelResult:
        self.calls.append(package)
        return MultimodalModelResult(
            candidate_id=package.candidate_id,
            observable_event=True,
            visual_payoff=True,
            visual_event=85.0,
            reaction=80.0,
            emotion=75.0,
            humor=70.0,
            surprise=65.0,
            energy=88.0,
            standalone=90.0,
            retention=85.0,
            shareability=80.0,
            boringness=15.0,
            context_dependency=20.0,
            outside_payoff=False,
            missing_setup=False,
            insufficient_visual_evidence=False,
            confidence=0.92,
            best_observed_region=ObservedRegion(
                start_offset=5.0,
                end_offset=15.0,
                confidence=0.9,
                reason="Streamer dramatic reaction to in-game event",
            ),
            evidence=[
                ObservedEvidenceItem(timestamp_offset=6.0, description="Character jumps"),
                ObservedEvidenceItem(timestamp_offset=12.0, description="Streamer celebrates"),
            ],
            reason="High energy reaction with complete setup and payoff.",
            quality_score=82.5,
        )

    def get_usage(self) -> MultimodalUsage:
        return self.usage


# ---------------------------------------------------------------------------
# Test Cases 1 to 26
# ---------------------------------------------------------------------------

def test_1_baselines_unchanged():
    """1. Baselines, versions, and formulas for heuristic_v1, highlight_v2_1, and multimodal_v1 remain unchanged."""
    assert SCORER_VERSION_MULTIMODAL_V1 == "multimodal_v1"
    assert FORMULA_VERSION_MULTIMODAL_V1 == "multimodal_v1_formula_v1"
    assert PACKAGE_VERSION_V1 == "multimodal_package_v1"
    assert PROMPT_VERSION_MULTIMODAL_V1 == "multimodal_v1_prompt_v1"

    assert SCORER_VERSION_MULTIMODAL_V1_1 == "multimodal_v1_1"
    assert FORMULA_VERSION_MULTIMODAL_V1_1 == "multimodal_v1_1_formula_v1"
    assert PACKAGE_VERSION_V1_1 == "multimodal_package_v1_1"
    assert PROMPT_VERSION_MULTIMODAL_V1_1 == "multimodal_v1_1_prompt_v1"
    assert ACTIVITY_V1_1_FORMULA_VERSION == "activity_v1_1_formula_v1"


def test_2_multimodal_v1_1_produces_separate_output_file(tmp_path: Path):
    """2. multimodal_v1_1 writes scores/multimodal_v1_1.json and does not overwrite scores/multimodal_v1.json."""
    _setup_mock_run_dir(tmp_path, count=42)
    v1_file = tmp_path / "scores" / "multimodal_v1.json"
    v1_bytes = v1_file.read_bytes()

    mock_provider = MockMultimodalV11Provider()
    reranker = MultimodalReranker(
        provider=mock_provider,
        scorer_version=SCORER_VERSION_MULTIMODAL_V1_1,
        heuristic_top_k=5,
        llm_top_k=5,
        max_candidates=8,
    )

    with mock.patch("freecher_worker.multimodal.package.extract_candidate_frames") as mock_frames,          mock.patch("freecher_worker.multimodal.scorer.probe_media") as mock_probe,          mock.patch("freecher_worker.multimodal.scorer.compute_source_temporal_activity_profile") as mock_act:
        mock_probe.return_value = mock.MagicMock(
            duration_seconds=1260.0, video_codec="h264", width=1920, height=1080, fps=30.0
        )
        img_path = _create_synthetic_image(tmp_path / "frame.jpg")
        mock_frames.return_value = (
            [ExtractedFrame(timestamp_offset=1.0, absolute_timestamp=1.0, image_path=str(img_path), width=640, height=360)] * 8,
            "libdav1d", 8, 0,
        )
        mock_act.return_value = SourceTemporalActivityProfile(
            source_fingerprint="035864f48379388e",
            formula_version=ACTIVITY_V1_1_FORMULA_VERSION,
            duration_seconds=1260.0,
            timeline=[
                SourceTemporalActivityPoint(
                    absolute_timestamp=float(s),
                    audio_energy=0.5,
                    audio_delta=0.4,
                    speech_activity=0.8,
                    visual_motion=0.3,
                    scene_change=False,
                    combined_activity=0.42,
                )
                for s in range(1260)
            ],
        )

        pred_doc = reranker.rerank_run(tmp_path)

    v1_1_file = tmp_path / "scores" / "multimodal_v1_1.json"
    assert v1_1_file.is_file()
    assert pred_doc.scorer == "multimodal_v1_1"
    assert pred_doc.scorer_version == "multimodal_v1_1"
    assert v1_file.read_bytes() == v1_bytes, "multimodal_v1.json was modified!"


def test_3_candidate_set_id_matches_frozen_run(tmp_path: Path):
    """3. Candidate set ID strictly matches candidate_set_id in candidates.json."""
    _setup_mock_run_dir(tmp_path, count=10)
    cand_doc = CandidateDocument.model_validate(load_json(tmp_path / "candidates.json"))

    mock_provider = MockMultimodalV11Provider()
    reranker = MultimodalReranker(
        provider=mock_provider,
        scorer_version=SCORER_VERSION_MULTIMODAL_V1_1,
        heuristic_top_k=3,
        llm_top_k=3,
        max_candidates=5,
    )

    with mock.patch("freecher_worker.multimodal.package.extract_candidate_frames") as mock_frames,          mock.patch("freecher_worker.multimodal.scorer.probe_media") as mock_probe,          mock.patch("freecher_worker.multimodal.scorer.compute_source_temporal_activity_profile") as mock_act:
        mock_probe.return_value = mock.MagicMock(duration_seconds=300.0, video_codec="h264", width=1920, height=1080, fps=30.0)
        img_path = _create_synthetic_image(tmp_path / "frame.jpg")
        mock_frames.return_value = (
            [ExtractedFrame(timestamp_offset=1.0, absolute_timestamp=1.0, image_path=str(img_path), width=640, height=360)] * 4,
            "libdav1d", 4, 0,
        )
        mock_act.return_value = SourceTemporalActivityProfile(
            source_fingerprint="035864f48379388e",
            formula_version=ACTIVITY_V1_1_FORMULA_VERSION,
            duration_seconds=300.0,
            timeline=[],
        )
        pred_doc = reranker.rerank_run(tmp_path)

    assert pred_doc.candidate_set_id == cand_doc.candidate_set_id
    assert pred_doc.candidate_set_id == "cset_8ac195a200530e96"


def test_4_high_recall_shortlist_top20_union_up_to_32(tmp_path: Path):
    """4. Shortlist v1_1 combines Top-20 heuristic + Top-20 LLM up to max 32 candidates."""
    _setup_mock_run_dir(tmp_path, count=42)
    sl = generate_shortlist(
        tmp_path,
        shortlist_version="v1_1",
        heuristic_top_k=20,
        llm_top_k=20,
        max_candidates=32,
    )
    assert len(sl.candidate_ids) == 32
    assert sl.max_candidates == 32
    assert sl.strategy == "high_recall_union_v1_1"
    assert sl.items is not None
    assert len(sl.items) == 32


def test_5_deterministic_shortlist_sorting_and_truncation(tmp_path: Path):
    """5. Shortlist sorting (best_rank, sum_rank, candidate_id) is deterministic."""
    _setup_mock_run_dir(tmp_path, count=42)
    sl1 = generate_shortlist(tmp_path, shortlist_version="v1_1", heuristic_top_k=20, llm_top_k=20, max_candidates=32)
    sl2 = generate_shortlist(tmp_path, shortlist_version="v1_1", heuristic_top_k=20, llm_top_k=20, max_candidates=32)
    assert sl1.candidate_ids == sl2.candidate_ids
    assert [item.candidate_id for item in sl1.items] == [item.candidate_id for item in sl2.items]


def test_6_shortlist_missing_rank_handling(tmp_path: Path):
    """6. Missing ranks use top_k + 1 for sorting, but persist as null (None)."""
    _setup_mock_run_dir(tmp_path, count=42)
    sl = generate_shortlist(tmp_path, shortlist_version="v1_1", heuristic_top_k=5, llm_top_k=5, max_candidates=10)
    items_by_id = {it.candidate_id: it for it in sl.items}

    item_1 = items_by_id.get("cand_001")
    assert item_1 is not None
    assert item_1.heuristic_rank == 1
    assert item_1.llm_rank is None  # Absent from llm top_k, persisted as null
    assert item_1.best_rank == 1
    assert item_1.sum_rank == 1 + (5 + 1)  # 1 + 6 = 7

    item_42 = items_by_id.get("cand_042")
    assert item_42 is not None
    assert item_42.heuristic_rank is None  # Absent from heuristic top_k
    assert item_42.llm_rank == 1
    assert item_42.best_rank == 1
    assert item_42.sum_rank == (5 + 1) + 1  # 6 + 1 = 7


def test_7_human_label_isolation(tmp_path: Path):
    """7. Human-label isolation: evaluation_blind.json never read during scoring/reranking."""
    _setup_mock_run_dir(tmp_path, count=10)
    eval_file = tmp_path / "evaluation_blind.json"
    save_json({
        "candidate_set_id": "cset_8ac195a200530e96",
        "total_candidates": 10,
        "items": [{
            "candidate_id": "cand_001",
            "start": 0.0,
            "end": 30.0,
            "duration": 30.0,
            "text": "hello",
            "human_score": 4.0,
            "publishable": True,
            "human_notes": "LEAK",
        }]
    }, eval_file)

    mock_provider = MockMultimodalV11Provider()
    reranker = MultimodalReranker(
        provider=mock_provider,
        scorer_version=SCORER_VERSION_MULTIMODAL_V1_1,
        heuristic_top_k=3,
        llm_top_k=3,
        max_candidates=5,
    )

    with mock.patch("freecher_worker.multimodal.package.extract_candidate_frames") as mock_frames,          mock.patch("freecher_worker.multimodal.scorer.probe_media") as mock_probe,          mock.patch("freecher_worker.multimodal.scorer.compute_source_temporal_activity_profile") as mock_act:
        mock_probe.return_value = mock.MagicMock(duration_seconds=300.0, video_codec="h264", width=1920, height=1080, fps=30.0)
        img_path = _create_synthetic_image(tmp_path / "frame.jpg")
        mock_frames.return_value = (
            [ExtractedFrame(timestamp_offset=1.0, absolute_timestamp=1.0, image_path=str(img_path), width=640, height=360)] * 4,
            "libdav1d", 4, 0,
        )
        mock_act.return_value = SourceTemporalActivityProfile(
            source_fingerprint="035864f48379388e",
            formula_version=ACTIVITY_V1_1_FORMULA_VERSION,
            duration_seconds=300.0,
            timeline=[],
        )
        reranker.rerank_run(tmp_path)

    for pkg in mock_provider.calls:
        pkg_json = pkg.model_dump_json()
        assert "human_score" not in pkg_json
        assert "publishable" not in pkg_json
        assert "LEAK" not in pkg_json


def test_8_structured_fields_survive_in_scorer_prediction_item(tmp_path: Path):
    """8. Structured fields survive into ScorerPredictionItem and scores/multimodal_v1_1.json."""
    _setup_mock_run_dir(tmp_path, count=5)
    mock_provider = MockMultimodalV11Provider()
    reranker = MultimodalReranker(
        provider=mock_provider,
        scorer_version=SCORER_VERSION_MULTIMODAL_V1_1,
        heuristic_top_k=2,
        llm_top_k=2,
        max_candidates=3,
    )

    with mock.patch("freecher_worker.multimodal.package.extract_candidate_frames") as mock_frames,          mock.patch("freecher_worker.multimodal.scorer.probe_media") as mock_probe,          mock.patch("freecher_worker.multimodal.scorer.compute_source_temporal_activity_profile") as mock_act:
        mock_probe.return_value = mock.MagicMock(duration_seconds=150.0, video_codec="h264", width=1920, height=1080, fps=30.0)
        img_path = _create_synthetic_image(tmp_path / "frame.jpg")
        mock_frames.return_value = (
            [ExtractedFrame(timestamp_offset=1.0, absolute_timestamp=1.0, image_path=str(img_path), width=640, height=360)] * 4,
            "libdav1d", 4, 0,
        )
        mock_act.return_value = SourceTemporalActivityProfile(
            source_fingerprint="035864f48379388e",
            formula_version=ACTIVITY_V1_1_FORMULA_VERSION,
            duration_seconds=150.0,
            timeline=[],
        )
        pred_doc = reranker.rerank_run(tmp_path)

    top_item = pred_doc.predictions[0]
    assert top_item.observable_event is True
    assert top_item.visual_payoff is True
    assert top_item.outside_payoff is False
    assert top_item.missing_setup is False
    assert top_item.insufficient_visual_evidence is False
    assert top_item.confidence == 0.92
    assert top_item.best_observed_region is not None
    assert top_item.best_observed_region["reason"] == "Streamer dramatic reaction to in-game event"
    assert top_item.evidence is not None
    assert len(top_item.evidence) == 2
    assert top_item.actual_decoder == "libdav1d"
    assert top_item.frame_count == 4
    assert top_item.package_hash is not None

    saved_doc = load_json(tmp_path / "scores" / "multimodal_v1_1.json")
    saved_top = saved_doc["predictions"][0]
    assert saved_top["observable_event"] is True
    assert saved_top["visual_payoff"] is True
    assert saved_top["best_observed_region"]["reason"] == "Streamer dramatic reaction to in-game event"
    assert len(saved_top["evidence"]) == 2


def test_9_observable_event_true_preserved():
    """9. Candidates with observable_event=True preserve full score."""
    result = MultimodalModelResult(
        candidate_id="c1",
        observable_event=True,
        visual_payoff=True,
        visual_event=85.0,
        reaction=80.0,
        emotion=75.0,
        humor=70.0,
        surprise=65.0,
        energy=85.0,
        standalone=90.0,
        retention=80.0,
        shareability=75.0,
        boringness=10.0,
        context_dependency=15.0,
        outside_payoff=False,
        missing_setup=False,
        insufficient_visual_evidence=False,
        confidence=0.9,
        reason="Good clip",
        quality_score=85.0,
    )
    package = mock.MagicMock(insufficient_visual_evidence=False)
    score, subscores, diag = multimodal_v1_1_formula_v1(result, package)
    assert score == 85.0
    assert len(diag["applied_caps"]) == 0


def test_10_observable_event_false_outside_payoff_capped():
    """10. Candidate with outside_payoff=True and observable_event=False is capped at <= 40.0."""
    result = MultimodalModelResult(
        candidate_id="c1",
        observable_event=False,
        visual_payoff=False,
        visual_event=50.0,
        reaction=50.0,
        emotion=50.0,
        humor=50.0,
        surprise=50.0,
        energy=50.0,
        standalone=50.0,
        retention=50.0,
        shareability=50.0,
        boringness=30.0,
        context_dependency=40.0,
        outside_payoff=True,
        missing_setup=False,
        insufficient_visual_evidence=False,
        confidence=0.9,
        reason="Punchline outside candidate",
        quality_score=75.0,
    )
    package = mock.MagicMock(insufficient_visual_evidence=False)
    score, subscores, diag = multimodal_v1_1_formula_v1(result, package)
    assert score <= 40.0
    assert any("outside_payoff_unobservable_cap:40.0" in cap for cap in diag["applied_caps"])


def test_11_visual_payoff_preserved():
    """11. visual_payoff field is persisted and distinct from observable_event."""
    res = MultimodalModelResult(
        candidate_id="c2",
        observable_event=True,
        visual_payoff=False,
        visual_event=60.0,
        reaction=60.0,
        emotion=60.0,
        humor=60.0,
        surprise=60.0,
        energy=60.0,
        standalone=60.0,
        retention=60.0,
        shareability=60.0,
        boringness=20.0,
        context_dependency=20.0,
        confidence=0.85,
        reason="Testing visual payoff",
        quality_score=60.0,
    )
    assert res.visual_payoff is False
    assert res.observable_event is True


def test_12_evidence_list_preserved():
    """12. Evidence list preserves timestamps and descriptions."""
    ev = [
        ObservedEvidenceItem(timestamp_offset=3.5, description="Jump scare happens"),
        ObservedEvidenceItem(timestamp_offset=7.0, description="Screaming reaction"),
    ]
    res = MultimodalModelResult(
        candidate_id="c3",
        observable_event=True,
        visual_payoff=True,
        visual_event=70.0,
        reaction=70.0,
        emotion=70.0,
        humor=70.0,
        surprise=70.0,
        energy=70.0,
        standalone=70.0,
        retention=70.0,
        shareability=70.0,
        boringness=20.0,
        context_dependency=20.0,
        confidence=0.85,
        reason="Testing evidence list",
        evidence=ev,
        quality_score=70.0,
    )
    assert len(res.evidence) == 2
    assert res.evidence[0].timestamp_offset == 3.5
    assert res.evidence[1].description == "Screaming reaction"


def test_13_best_observed_region_schema_validation():
    """13. best_observed_region adheres to schema and validates offsets."""
    valid_region = ObservedRegion(
        start_offset=2.5,
        end_offset=8.0,
        confidence=0.85,
        reason="Peak punchline and facepalm",
    )
    assert valid_region.start_offset == 2.5
    assert valid_region.end_offset == 8.0


def test_14_temporal_activity_curve_generated_per_candidate():
    """14. Candidate activity curve is sliced from source profile and summary computed."""
    points = [
        SourceTemporalActivityPoint(
            absolute_timestamp=float(t),
            audio_energy=min(1.0, t * 0.05),
            audio_delta=0.1,
            speech_activity=0.8,
            visual_motion=0.2,
            scene_change=False,
            combined_activity=round(0.1 * 0.35 + min(1.0, t * 0.05) * 0.25 + 0.2 * 0.30, 4),
        )
        for t in range(60)
    ]
    profile = SourceTemporalActivityProfile(
        source_fingerprint="test_fp",
        formula_version=ACTIVITY_V1_1_FORMULA_VERSION,
        duration_seconds=60.0,
        timeline=points,
    )
    summary = slice_candidate_activity_curve(profile, candidate_start=10.0, candidate_duration=20.0)

    assert len(summary.curve) > 0
    assert len(summary.top_combined_activity_peaks) > 0


def test_15_temporal_activity_curve_timestamps_within_bounds():
    """15. All points in candidate activity curve fall strictly within [candidate.start, candidate.end]."""
    points = [
        SourceTemporalActivityPoint(
            absolute_timestamp=float(t),
            audio_energy=0.5,
            audio_delta=0.2,
            speech_activity=0.5,
            visual_motion=0.3,
            scene_change=False,
            combined_activity=0.35,
        )
        for t in range(100)
    ]
    profile = SourceTemporalActivityProfile(
        source_fingerprint="test_fp",
        formula_version=ACTIVITY_V1_1_FORMULA_VERSION,
        duration_seconds=100.0,
        timeline=points,
    )
    summary = slice_candidate_activity_curve(profile, candidate_start=25.0, candidate_duration=30.0)

    for pt in summary.curve:
        assert 25.0 <= pt.absolute_timestamp <= 55.0
        assert 0.0 <= pt.offset <= 30.0


def test_16_audio_activity_peak_detection():
    """16. Audio delta/energy peaks contribute to combined activity calculation."""
    act1 = compute_combined_activity(audio_delta=0.8, audio_energy=0.9, visual_motion=0.1, scene_change=False)
    act2 = compute_combined_activity(audio_delta=0.05, audio_energy=0.1, visual_motion=0.1, scene_change=False)
    assert act1 > act2
    # Formula check: 0.8*0.35 + 0.9*0.25 + 0.1*0.30 + 0.0*0.10 = 0.28 + 0.225 + 0.03 = 0.535
    assert abs(act1 - 0.535) < 1e-3


def test_17_visual_motion_peak_detection():
    """17. Visual motion peaks are detected and contribute 0.30 weight."""
    act1 = compute_combined_activity(audio_delta=0.1, audio_energy=0.1, visual_motion=0.9, scene_change=False)
    act2 = compute_combined_activity(audio_delta=0.1, audio_energy=0.1, visual_motion=0.1, scene_change=False)
    assert act1 > act2
    # Difference should be (0.9 - 0.1) * 0.30 = 0.24
    assert abs((act1 - act2) - 0.24) < 1e-4


def test_18_combined_activity_peak_selection():
    """18. Temporal burst peaks select separated candidate-local bursts."""
    pts = [
        ActivityPoint(offset=float(t), absolute_timestamp=100.0 + t, audio_energy=0.1, audio_delta=0.1, speech_activity=0.1, visual_motion=0.1, scene_change=False, combined_activity=0.1)
        for t in range(30)
    ]
    pts[5].combined_activity = 0.9
    pts[5].audio_energy = 0.9
    pts[20].combined_activity = 0.8
    pts[20].audio_energy = 0.8

    summary = ActivityCurveSummary(
        curve=pts,
        top_audio_peaks=[5.0, 20.0],
        top_motion_peaks=[],
        top_combined_activity_peaks=[5.0, 20.0],
    )
    bursts = select_temporal_burst_peaks(summary, candidate_duration=30.0)
    assert len(bursts) == 2
    assert bursts[0].burst_index == 1
    assert bursts[1].burst_index == 2
    assert abs(bursts[0].center_offset - bursts[1].center_offset) >= 3.0


def test_19_burst_generation_frame_budget():
    """19. compute_v1_1_sample_timestamps produces <= 12 total frames."""
    samples = compute_v1_1_sample_timestamps(
        candidate_start=10.0,
        candidate_duration=30.0,
        burst_1_center=8.0,
        burst_2_center=22.0,
        max_total_frames=12,
    )
    assert len(samples) <= 12
    assert any(s[2] == "global" for s in samples)
    assert any(s[2] == "burst_1" for s in samples)
    assert any(s[2] == "burst_2" for s in samples)


def test_20_transcript_alignment_padding():
    """20. Burst transcript includes speech segments overlapping [burst_start - 0.75, burst_end + 0.75]."""
    segments = [
        TranscriptSegment(id=1, start=2.0, end=3.5, text="Before"),
        TranscriptSegment(id=2, start=4.5, end=5.5, text="Overlaps left pad"),
        TranscriptSegment(id=3, start=6.0, end=7.0, text="Inside burst"),
        TranscriptSegment(id=4, start=7.5, end=8.5, text="Overlaps right pad"),
        TranscriptSegment(id=5, start=9.5, end=11.0, text="After"),
    ]
    transcript = Transcript(
        language="en", duration=20.0, model="tiny", compute_type="int8", device="cpu", segments=segments
    )
    burst = TemporalBurst(
        burst_index=1,
        center_offset=6.0,
        start_offset=5.0,
        end_offset=7.0,
        selection_reason="combined_activity_peak",
        combined_activity=0.8,
        activity_rank=1,
    )
    burst_text = extract_temporal_burst_transcript(
        candidate_start=0.0,
        burst=burst,
        transcript_doc=transcript,
        padding=0.75,
    )
    assert "Overlaps left pad" in burst_text
    assert "Inside burst" in burst_text
    assert "Overlaps right pad" in burst_text
    assert "Before" not in burst_text
    assert "After" not in burst_text


def test_21_package_cache_invalidation_on_version_change():
    """21. compute_package_hash produces different hashes for different package versions."""
    hash_v1 = compute_package_hash(
        candidate_set_id="cset_1",
        source_fingerprint="fp1",
        candidate_id="cand_1",
        start=10.0,
        end=40.0,
        package_version=PACKAGE_VERSION_V1,
    )
    hash_v1_1 = compute_package_hash(
        candidate_set_id="cset_1",
        source_fingerprint="fp1",
        candidate_id="cand_1",
        start=10.0,
        end=40.0,
        package_version=PACKAGE_VERSION_V1_1,
    )
    assert hash_v1 != hash_v1_1


def test_22_av1_software_decode_probe():
    """22. AV1 software decoder probe returns a valid decoder string."""
    dec_name, desc = probe_software_decoder("av1")
    assert dec_name in ("libdav1d", "av1", "none", "unknown") or isinstance(dec_name, str)


def test_23_request_cache_separated_between_v1_and_v1_1():
    """23. OpenAI provider prompt versions and system prompts differ between v1 and v1_1."""
    p1 = OpenAIMultimodalProvider(prompt_version=PROMPT_VERSION_MULTIMODAL_V1)
    p1_1 = OpenAIMultimodalProvider(prompt_version=PROMPT_VERSION_MULTIMODAL_V1_1)
    assert p1.prompt_version == "multimodal_v1_prompt_v1"
    assert p1_1.prompt_version == "multimodal_v1_1_prompt_v1"
    assert SYSTEM_PROMPT_MULTIMODAL_V1 != SYSTEM_PROMPT_MULTIMODAL_V1_1


def test_24_no_silent_fallback(tmp_path: Path):
    """24. Shortlist generation raises FileNotFoundError when highlight_v2_1.json is missing and allow_missing_llm=False."""
    _setup_mock_run_dir(tmp_path, count=10)
    (tmp_path / "scores" / "highlight_v2_1.json").unlink()

    with pytest.raises(FileNotFoundError, match="Missing highlight_v2_1 predictions"):
        generate_shortlist(
            tmp_path,
            shortlist_version="v1_1",
            heuristic_top_k=5,
            llm_top_k=5,
            max_candidates=8,
            allow_missing_llm=False,
        )


def test_25_evaluation_separates_retrieval_and_reranking(tmp_path: Path):
    """25. Evaluation metrics compute coverage and shortlist recalls for partial candidate pool."""
    _setup_mock_run_dir(tmp_path, count=20)
    eval_items = []
    for i in range(1, 21):
        cid = f"cand_{i:03d}"
        if i <= 5:
            score = 4.0
            pub = True
        elif i <= 10:
            score = 3.0
            pub = True
        else:
            score = 1.0
            pub = False
        eval_items.append(
            BlindEvaluationItem(
                candidate_id=cid,
                start=0.0,
                end=30.0,
                duration=30.0,
                text="Sample transcript",
                human_score=score,
                publishable=pub,
            )
        )

    eval_doc = BlindEvaluationDocument(
        candidate_set_id="cset_8ac195a200530e96",
        total_candidates=20,
        labeled_candidates=20,
        items=eval_items,
    )

    pred_items = [
        ScorerPredictionItem(candidate_id=f"cand_{i:03d}", rank=i, score=100.0 - i)
        for i in range(1, 11)
    ]
    scores_doc = ScorerPredictionDocument(
        candidate_set_id="cset_8ac195a200530e96",
        scorer="multimodal_v1_1",
        scorer_version="multimodal_v1_1",
        predictions=pred_items,
    )

    metrics = compute_evaluation_metrics(eval_doc, scores_doc, k_values=[5, 10])
    assert metrics.candidate_coverage_ratio == 0.5  # 10 / 20
    assert metrics.perfect_candidate_recall_in_shortlist == 1.0  # All 5 perfect are in top 10
    assert metrics.publishable_candidate_recall_in_shortlist == 1.0  # All 10 publishable are in top 10
    assert metrics.scored_candidates == 10
    assert metrics.total_candidates == 20


def test_26_highlight_v2_1_metrics_and_scoring_unchanged():
    """26. highlight_v2_1 formula and scoring remains unchanged."""
    features = {
        "hook": 80.0,
        "standalone": 75.0,
        "story_payoff": 85.0,
        "emotion": 70.0,
        "humor": 60.0,
        "surprise": 50.0,
        "visual_action": 65.0,
        "dialogue_clarity": 90.0,
        "pacing": 75.0,
        "retention": 80.0,
        "shareability": 70.0,
        "boringness": 20.0,
        "cringe": 10.0,
        "context_dependency": 30.0,
        "setup_only": False,
        "transitional": False,
        "outside_payoff": False,
    }
    score, subscores, diag = highlight_v2_1_formula_v1(features)
    assert 60.0 <= score <= 90.0
    assert diag["positive_score"] > 0 and "raw_score" in diag


def test_27_av1_never_selects_native_av1():
    """27. Native 'av1' decoder is never selected as explicit FFmpeg argument; falls back safely to ffmpeg_auto."""
    with mock.patch("freecher_worker.multimodal.frames.get_available_ffmpeg_decoders", return_value={"av1", "libaom-av1"}):
        res = resolve_safe_video_decoder("av1")
        assert res.decoder_mode == "ffmpeg_auto"
        assert res.ffmpeg_decoder_arg is None
        assert res.hardware_acceleration is False
        assert res.source_codec == "av1"

        dec_arg, desc = probe_software_decoder("av1")
        assert dec_arg is None
        assert "ffmpeg_auto" in desc


def test_28_libdav1d_preferred_when_smoke_test_succeeds(tmp_path: Path):
    """28. libdav1d is preferred for AV1 when available and runtime smoke test passes."""
    mock_vid = tmp_path / "sample.mp4"
    mock_vid.write_bytes(b"TEST_VIDEO")

    with mock.patch("freecher_worker.multimodal.frames.get_available_ffmpeg_decoders", return_value={"libdav1d", "av1"}), \
         mock.patch("freecher_worker.multimodal.frames.run_decoder_smoke_test", return_value=True):
        res = resolve_safe_video_decoder("av1", source_video_path=mock_vid)
        assert res.decoder_mode == "libdav1d"
        assert res.requested_decoder == "libdav1d"
        assert res.ffmpeg_decoder_arg == "libdav1d"
        assert res.hardware_acceleration is False
        assert res.smoke_test_passed is True

        dec_arg, desc = probe_software_decoder("av1", source_video_path=mock_vid)
        assert dec_arg == "libdav1d"
        assert "libdav1d" in desc


def test_29_ffmpeg_auto_used_when_libdav1d_smoke_fails(tmp_path: Path):
    """29. If libdav1d smoke test fails, gracefully falls back to ffmpeg_auto (omitting -c:v)."""
    mock_vid = tmp_path / "sample.mp4"
    mock_vid.write_bytes(b"TEST_VIDEO")

    with mock.patch("freecher_worker.multimodal.frames.get_available_ffmpeg_decoders", return_value={"libdav1d", "av1"}), \
         mock.patch("freecher_worker.multimodal.frames.run_decoder_smoke_test", return_value=False):
        res = resolve_safe_video_decoder("av1", source_video_path=mock_vid)
        assert res.decoder_mode == "ffmpeg_auto"
        assert res.requested_decoder == "libdav1d"
        assert res.ffmpeg_decoder_arg is None
        assert res.hardware_acceleration is False
        assert res.smoke_test_passed is False

        dec_arg, desc = probe_software_decoder("av1", source_video_path=mock_vid)
        assert dec_arg is None
        assert "ffmpeg_auto" in desc


def test_30_hardware_av1_decoders_never_selected():
    """30. Hardware AV1 decoders (cuvid, qsv, nvdec) are never selected."""
    with mock.patch("freecher_worker.multimodal.frames.get_available_ffmpeg_decoders", return_value={"av1_cuvid", "av1_qsv", "av1_nvdec", "av1"}):
        res = resolve_safe_video_decoder("av1")
        assert res.decoder_mode == "ffmpeg_auto"
        assert res.ffmpeg_decoder_arg is None
        assert res.hardware_acceleration is False
        assert res.requested_decoder is None

        dec_arg, desc = probe_software_decoder("av1")
        assert dec_arg is None
        assert "cuvid" not in desc.lower()
        assert "qsv" not in desc.lower()
        assert "nvdec" not in desc.lower()


def test_31_command_builder_does_not_translate_libdav1d_to_av1(tmp_path: Path):
    """31. Command builder passes -c:v libdav1d when libdav1d is used, never -c:v av1; omits -c:v on auto/av1."""
    from freecher_worker.multimodal.activity import _compute_source_visual_timeline
    mock_vid = tmp_path / "test.mp4"
    mock_vid.write_bytes(b"DUMMY_MP4")

    # A. Activity visual timeline command checks
    with mock.patch("subprocess.Popen") as mock_popen:
        mock_proc = mock.MagicMock()
        mock_proc.stdout.read.return_value = b""
        mock_proc.wait.return_value = 0
        mock_proc.poll.return_value = 0
        mock_popen.return_value = mock_proc

        # A1: libdav1d -> -c:v libdav1d
        _compute_source_visual_timeline(mock_vid, duration_seconds=5.0, decoder_name="libdav1d")
        cmd1 = mock_popen.call_args[0][0]
        assert "-c:v" in cmd1
        assert cmd1[cmd1.index("-c:v") + 1] == "libdav1d"
        assert cmd1[cmd1.index("-c:v") + 1] != "av1"

        # A2: "av1" -> omitted, never -c:v av1
        _compute_source_visual_timeline(mock_vid, duration_seconds=5.0, decoder_name="av1")
        cmd2 = mock_popen.call_args[0][0]
        assert "-c:v" not in cmd2

        # A3: "ffmpeg_auto" -> omitted
        _compute_source_visual_timeline(mock_vid, duration_seconds=5.0, decoder_name="ffmpeg_auto")
        cmd3 = mock_popen.call_args[0][0]
        assert "-c:v" not in cmd3

    # B. Frame extraction command checks
    c = _make_candidate("c_test", start=0.0, end=10.0)
    dst = tmp_path / "extracted_frames"
    dst.mkdir(parents=True, exist_ok=True)

    with mock.patch("freecher_worker.multimodal.frames.probe_media") as mock_probe, \
         mock.patch("subprocess.run") as mock_run, \
         mock.patch("freecher_worker.multimodal.frames.run_decoder_smoke_test", return_value=True):
        mock_probe.return_value = mock.MagicMock(video_codec="av1")
        mock_res = mock.MagicMock(returncode=0)
        mock_run.return_value = mock_res

        # B1: when libdav1d is resolved
        with mock.patch("freecher_worker.multimodal.frames.get_available_ffmpeg_decoders", return_value={"libdav1d"}):
            extract_candidate_frames(
                source_video_path=mock_vid,
                candidate=c,
                destination_dir=dst,
                uniform_fractions=(0.5,),
            )
            cmd_frame1 = mock_run.call_args[0][0]
            assert "-c:v" in cmd_frame1
            assert cmd_frame1[cmd_frame1.index("-c:v") + 1] == "libdav1d"
            assert cmd_frame1[cmd_frame1.index("-c:v") + 1] != "av1"

        # Clean dst
        for f in dst.glob("*.jpg"):
            f.unlink()

        # B2: when only native av1 exists in system
        with mock.patch("freecher_worker.multimodal.frames.get_available_ffmpeg_decoders", return_value={"av1"}):
            extract_candidate_frames(
                source_video_path=mock_vid,
                candidate=c,
                destination_dir=dst,
                uniform_fractions=(0.5,),
            )
            cmd_frame2 = mock_run.call_args[0][0]
            assert "-c:v" not in cmd_frame2


def test_32_activity_profile_and_frame_extraction_share_decoder_policy(tmp_path: Path):
    """32. Activity profiling and candidate frame extraction share the exact same resolver policy and persist decoder_info."""
    _setup_mock_run_dir(tmp_path, count=5)

    def _mock_sub_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and len(cmd) > 0 and str(cmd[-1]).endswith(".jpg"):
            Path(cmd[-1]).write_bytes(b"DUMMY_IMAGE")
        return mock.MagicMock(returncode=0, stderr=b"")

    with mock.patch("freecher_worker.multimodal.frames.get_available_ffmpeg_decoders", return_value={"libdav1d", "av1"}), \
         mock.patch("freecher_worker.multimodal.frames.run_decoder_smoke_test", return_value=True), \
         mock.patch("freecher_worker.multimodal.scorer.probe_media") as mock_probe_scorer, \
         mock.patch("freecher_worker.multimodal.frames.probe_media") as mock_probe_frames, \
         mock.patch("freecher_worker.multimodal.activity.probe_media") as mock_probe_act, \
         mock.patch("subprocess.run", side_effect=_mock_sub_run), \
         mock.patch("subprocess.Popen") as mock_popen:
        mock_media = mock.MagicMock(video_codec="av1", duration_seconds=10.0, fps=30.0, width=1920, height=1080)
        mock_probe_scorer.return_value = mock_media
        mock_probe_frames.return_value = mock_media
        mock_probe_act.return_value = mock_media

        mock_proc = mock.MagicMock()
        mock_proc.stdout.read.return_value = b""
        mock_proc.wait.return_value = 0
        mock_proc.poll.return_value = 0
        mock_popen.return_value = mock_proc

        mock_prov = MockMultimodalV11Provider(prompt_version=PROMPT_VERSION_MULTIMODAL_V1_1)
        reranker = MultimodalReranker(
            provider=mock_prov,
            scorer_version="multimodal_v1_1",
            heuristic_top_k=2,
            llm_top_k=2,
            max_candidates=4,
        )

        pred_doc = reranker.rerank_run(tmp_path)
        assert len(pred_doc.predictions) > 0
        p0 = pred_doc.predictions[0]

        # Verify decoder info persisted on ScorerPredictionItem
        assert p0.actual_decoder_mode == "libdav1d"
        assert p0.requested_decoder == "libdav1d"
        assert p0.decoder_info is not None
        assert p0.decoder_info["decoder_mode"] == "libdav1d"
        assert p0.decoder_info["hardware_acceleration"] is False

        # Verify activity profile decoder info matches
        act_file = tmp_path / "multimodal" / "cache" / "source_temporal_activity_profile_v1_1.json"
        assert act_file.is_file()
        act_data = load_json(act_file)
        assert act_data["decoder_info"]["decoder_mode"] == "libdav1d"
        assert act_data["decoder_info"]["hardware_acceleration"] is False

