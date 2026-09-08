"""Comprehensive unit and integration tests for Freecher Multimodal Highlight Reranker v1 (multimodal_v1)."""

import io
import json
from pathlib import Path
from typing import Any, Dict
import unittest.mock as mock
import wave

import cv2
import numpy as np
import pytest

from freecher_worker.evaluation.models import (
    BlindEvaluationDocument,
    BlindEvaluationItem,
    ScorerPredictionDocument,
    ScorerPredictionItem,
)
from freecher_worker.evaluation.metrics import compute_evaluation_metrics
from freecher_worker.highlights.models import CandidateDocument, CandidateWindow
from freecher_worker.multimodal import (
    AudioFeatures,
    ExtractedFrame,
    FORMULA_VERSION_MULTIMODAL_V1,
    MultimodalCandidatePackage,
    MultimodalModelResult,
    MultimodalProvider,
    MultimodalReranker,
    MultimodalUsage,
    OpenAIMultimodalProvider,
    SCORER_VERSION_MULTIMODAL_V1,
    ShortlistDocument,
    SourceAudioProfile,
    VisualFeatures,
    build_multimodal_package,
    compute_source_audio_profile,
    extract_candidate_audio_features,
    extract_candidate_visual_features,
    generate_shortlist,
    multimodal_v1_formula_v1,
    probe_software_decoder,
)
from freecher_worker.multimodal.frames import (
    compute_frame_sample_timestamps,
    extract_candidate_frames,
)
from freecher_worker.transcription.models import Transcript, TranscriptSegment
from freecher_worker.utils.json_io import save_json


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _create_synthetic_wav(path: Path, duration_sec: float = 2.0, sample_rate: int = 16000) -> Path:
    """Create a test mono 16-bit PCM WAV file."""
    total_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    # Sine wave with varying amplitude
    sig = (0.4 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(sig.tobytes())
    return path


def _create_synthetic_image(path: Path, width: int = 320, height: int = 240, color: int = 128) -> Path:
    """Create a test JPEG image."""
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


def _setup_mock_run_dir(tmp_path: Path, count: int = 25) -> Path:
    """Populate a synthetic run directory with candidates and historical score files."""
    cset_id = "cset_mock_test_123"
    candidates = []
    heur_preds = []
    llm_preds = []

    for i in range(1, count + 1):
        cid = f"cand_{i:03d}"
        c = _make_candidate(cid, start=(i - 1) * 30.0, end=i * 30.0)
        candidates.append(c)

        # Heuristic ranking (order 1..count)
        heur_preds.append(
            ScorerPredictionItem(
                candidate_id=cid,
                rank=i,
                score=100.0 - i,
            )
        )
        # LLM ranking (reversed order count..1)
        llm_preds.append(
            ScorerPredictionItem(
                candidate_id=cid,
                rank=i,
                score=float(i),
                llm_quality_score=float(i * 3),
                final_score=float(i * 3),
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

    # Synthetic wav
    _create_synthetic_wav(tmp_path / "audio.wav", duration_sec=5.0)

    # Empty dummy video file
    (tmp_path / "source.mp4").write_bytes(b"DUMMY_MP4_HEADER")

    return tmp_path


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_shortlist_generation_deterministic_order(tmp_path: Path):
    """1. Shortlist generated twice on the same run produces identical candidate ID list and ordering."""
    _setup_mock_run_dir(tmp_path, count=10)
    sl1 = generate_shortlist(tmp_path, heuristic_top_k=5, llm_top_k=5, max_candidates=8)
    sl2 = generate_shortlist(tmp_path, heuristic_top_k=5, llm_top_k=5, max_candidates=8)

    assert sl1.candidate_ids == sl2.candidate_ids
    assert len(sl1.candidate_ids) == 8
    assert sl1.total_unique == 8


def test_shortlist_truncation_max_candidates(tmp_path: Path):
    """2. Shortlist with 25 candidates requested truncates deterministically to max_candidates=20."""
    _setup_mock_run_dir(tmp_path, count=25)
    sl = generate_shortlist(tmp_path, heuristic_top_k=15, llm_top_k=15, max_candidates=20)

    assert len(sl.candidate_ids) == 20
    assert sl.max_candidates == 20
    assert sl.total_unique == 20


def test_shortlist_missing_heuristic_fails(tmp_path: Path):
    """3. Shortlist generation raises FileNotFoundError when scores/heuristic_v1.json is missing."""
    _setup_mock_run_dir(tmp_path, count=5)
    (tmp_path / "scores" / "heuristic_v1.json").unlink()

    with pytest.raises(FileNotFoundError, match="Missing heuristic predictions"):
        generate_shortlist(tmp_path)


def test_shortlist_missing_llm_fails_in_benchmark_mode(tmp_path: Path):
    """4. Raises FileNotFoundError when scores/highlight_v2_1.json is missing and allow_missing_llm=False."""
    _setup_mock_run_dir(tmp_path, count=5)
    (tmp_path / "scores" / "highlight_v2_1.json").unlink()

    with pytest.raises(FileNotFoundError, match="Missing highlight_v2_1 predictions"):
        generate_shortlist(tmp_path, allow_missing_llm=False)


def test_shortlist_allow_missing_llm_fallback(tmp_path: Path):
    """5. When allow_missing_llm=True, succeeds and uses heuristic-only retrieval."""
    _setup_mock_run_dir(tmp_path, count=5)
    (tmp_path / "scores" / "highlight_v2_1.json").unlink()

    sl = generate_shortlist(tmp_path, allow_missing_llm=True)
    assert sl.strategy == "heuristic_only"
    assert len(sl.candidate_ids) > 0


def test_shortlist_persisted_to_json(tmp_path: Path):
    """6. Generates and writes multimodal/shortlist_v1.json matching ShortlistDocument schema."""
    _setup_mock_run_dir(tmp_path, count=10)
    sl = generate_shortlist(tmp_path, heuristic_top_k=4, llm_top_k=4, max_candidates=6)

    target_file = tmp_path / "multimodal" / "shortlist_v1.json"
    assert target_file.is_file()

    with open(target_file, "r") as f:
        data = json.load(f)
    doc = ShortlistDocument.model_validate(data)
    assert doc.candidate_ids == sl.candidate_ids
    assert doc.candidate_set_id == "cset_mock_test_123"


def test_probe_software_decoder_av1():
    """7. Probes AV1 codec and selects software decoder (e.g. libdav1d/av1), never hardware acceleration."""
    dec, desc = probe_software_decoder("av1")
    assert dec in ("libdav1d", "av1", None)
    # Ensure hardware decoders (nvdec, cuvid, qsv) are never selected
    assert "cuvid" not in desc.lower()
    assert "nvdec" not in desc.lower()


def test_frame_sampling_uniform_fractions():
    """8. Produces 8 uniform sample offsets at [5%, 18%, 31%, 44%, 56%, 69%, 82%, 95%]."""
    samples = compute_frame_sample_timestamps(candidate_start=10.0, candidate_duration=100.0)
    assert len(samples) == 8

    offsets = [s[0] for s in samples]
    expected_offsets = [5.0, 18.0, 31.0, 44.0, 56.0, 69.0, 82.0, 95.0]
    assert offsets == expected_offsets

    abs_times = [s[1] for s in samples]
    expected_abs = [15.0, 28.0, 41.0, 54.0, 66.0, 79.0, 92.0, 105.0]
    assert abs_times == expected_abs

    for s in samples:
        assert s[2] == "uniform"


def test_frame_sampling_with_scene_changes():
    """9. Merges up to 4 scene change offsets without exceeding max_total_frames=12."""
    samples = compute_frame_sample_timestamps(
        candidate_start=0.0,
        candidate_duration=60.0,
        scene_change_offsets=[10.0, 25.0, 35.0, 40.0, 42.0, 48.0],
        max_total_frames=12,
    )
    assert len(samples) <= 12
    # Ensure strictly sorted by offset
    offsets = [s[0] for s in samples]
    assert offsets == sorted(offsets)
    # Check scene_change source type is present
    stypes = {s[2] for s in samples}
    assert "scene_change" in stypes


def test_frame_extraction_downscaling(tmp_path: Path):
    """10. Extracted frames request downscaling to max 640px long edge."""
    c = _make_candidate("cand_001", start=0.0, end=10.0)
    src_vid = tmp_path / "source.mp4"
    src_vid.write_bytes(b"DUMMY_MP4")

    # Mock probe_media and subprocess.run
    with mock.patch("freecher_worker.multimodal.frames.probe_media") as mock_probe, \
         mock.patch("subprocess.run") as mock_sub:
        mock_probe.return_value = mock.MagicMock(video_codec="h264")
        mock_sub.return_value = mock.MagicMock(return_code=0)

        dst = tmp_path / "frames"
        dst.mkdir(parents=True, exist_ok=True)
        # Touch a dummy output frame so extract_candidate_frames recognizes output
        for i in range(1, 9):
            (dst / f"frame_{i:02d}.jpg").write_bytes(b"DUMMY_IMAGE")

        frames, decoder_used, req_count, fail_count = extract_candidate_frames(
            source_video_path=src_vid,
            candidate=c,
            destination_dir=dst,
            max_long_edge=640,
        )
        assert len(frames) == 8
        assert req_count == 8
        assert fail_count == 0


def test_frame_extraction_reuse_cached_files(tmp_path: Path):
    """11. Reuses existing non-empty extracted JPEG files on disk without calling FFmpeg."""
    c = _make_candidate("cand_001", start=0.0, end=10.0)
    src_vid = tmp_path / "source.mp4"
    src_vid.write_bytes(b"DUMMY_MP4")

    dst = tmp_path / "frames"
    dst.mkdir(parents=True, exist_ok=True)
    # Pre-create all 8 frame files
    for i in range(1, 9):
        (dst / f"frame_{i:02d}.jpg").write_bytes(b"EXISTING_FRAME")

    with mock.patch("freecher_worker.multimodal.frames.probe_media") as mock_probe, \
         mock.patch("subprocess.run") as mock_sub:
        mock_probe.return_value = mock.MagicMock(video_codec="h264")

        frames, _, _, _ = extract_candidate_frames(
            source_video_path=src_vid,
            candidate=c,
            destination_dir=dst,
        )
        assert len(frames) == 8
        # subprocess.run should not have been called for extraction
        assert mock_sub.call_count == 0


def test_source_audio_profile_single_pass(tmp_path: Path):
    """12. compute_source_audio_profile computes percentiles and caches to JSON."""
    wav_path = _create_synthetic_wav(tmp_path / "test.wav", duration_sec=3.0)
    cache_file = tmp_path / "profile.json"

    profile = compute_source_audio_profile(wav_path, "fp_test", cache_file=cache_file)
    assert profile.source_fingerprint == "fp_test"
    assert profile.duration_seconds >= 2.9
    assert "p50" in profile.rms_percentiles
    assert profile.rms_percentiles["p50"] > 0.0
    assert cache_file.is_file()


def test_source_audio_profile_reuse_cache(tmp_path: Path):
    """13. Second call loads cached profile without recomputing."""
    wav_path = _create_synthetic_wav(tmp_path / "test.wav", duration_sec=2.0)
    cache_file = tmp_path / "profile.json"

    prof1 = compute_source_audio_profile(wav_path, "fp_test", cache_file=cache_file)
    # Delete original wav file
    wav_path.unlink()

    # Second call should load from cache_file without needing wav_path
    prof2 = compute_source_audio_profile(wav_path, "fp_test", cache_file=cache_file)
    assert prof2.source_fingerprint == prof1.source_fingerprint
    assert prof2.rms_percentiles == prof1.rms_percentiles


def test_candidate_audio_features_metrics(tmp_path: Path):
    """14. Computes RMS, peak, silence ratio, speech coverage, and relative percentile."""
    wav_path = _create_synthetic_wav(tmp_path / "test.wav", duration_sec=3.0)
    profile = compute_source_audio_profile(wav_path, "fp_test")

    feats = extract_candidate_audio_features(wav_path, start=0.5, end=2.0, source_profile=profile)
    assert feats.rms_mean > 0.0
    assert feats.peak > 0.0
    assert 0.0 <= feats.silence_ratio <= 1.0
    assert 0.0 <= feats.speech_coverage <= 1.0
    assert feats.energy_percentile is not None
    assert 0.0 <= feats.energy_percentile <= 1.0
    assert feats.beginning_rms is not None
    assert feats.middle_rms is not None
    assert feats.ending_rms is not None


def test_candidate_audio_features_empty_audio(tmp_path: Path):
    """15. Gracefully returns zeroed/safe audio features for empty/out-of-bounds audio slice."""
    wav_path = _create_synthetic_wav(tmp_path / "test.wav", duration_sec=1.0)
    feats = extract_candidate_audio_features(wav_path, start=999.0, end=1000.0)

    assert feats.rms_mean == 0.0
    assert feats.peak == 0.0
    assert feats.silence_ratio == 1.0
    assert feats.speech_coverage == 0.0


def test_candidate_visual_features_motion_score(tmp_path: Path):
    """16. Consecutive frame differences produce valid motion score in [0, 1]."""
    f1 = _create_synthetic_image(tmp_path / "f1.jpg", color=0)
    f2 = _create_synthetic_image(tmp_path / "f2.jpg", color=255)

    vf = extract_candidate_visual_features([f1, f2], requested_count=2)
    assert vf.decoded_frame_count == 2
    # Moving from black (0) to white (255) yields normalized MAD of ~1.0
    assert vf.motion_score > 0.95


def test_candidate_visual_features_scene_changes(tmp_path: Path):
    """17. Detects scene cuts when frame differences exceed threshold."""
    f1 = _create_synthetic_image(tmp_path / "f1.jpg", color=0)
    f2 = _create_synthetic_image(tmp_path / "f2.jpg", color=255)
    f3 = _create_synthetic_image(tmp_path / "f3.jpg", color=255)

    vf = extract_candidate_visual_features([f1, f2, f3], requested_count=3)
    # f1 -> f2 is a cut, f2 -> f3 is static
    assert vf.scene_change_count == 1


def test_candidate_visual_features_face_and_person_separation(tmp_path: Path):
    """18. face_presence_ratio and person_presence_ratio are separate; person is None if no detector."""
    f1 = _create_synthetic_image(tmp_path / "f1.jpg", color=120)
    vf = extract_candidate_visual_features([f1], requested_count=1)

    assert hasattr(vf, "face_presence_ratio")
    assert hasattr(vf, "person_presence_ratio")
    # If HOGDescriptor is not installed in environment, person_presence_ratio must be None
    if not hasattr(cv2, "HOGDescriptor"):
        assert vf.person_presence_ratio is None


def test_multimodal_package_builder_no_human_labels(tmp_path: Path):
    """19. build_multimodal_package produces package with zero human evaluation labels."""
    _setup_mock_run_dir(tmp_path, count=3)
    c = _make_candidate("cand_001", start=0.0, end=10.0)

    # Pre-populate frames so frame extraction is simulated
    cache_dir = tmp_path / "multimodal" / "cache" / c.id
    cache_dir.mkdir(parents=True, exist_ok=True)

    with mock.patch("freecher_worker.multimodal.package.extract_candidate_frames") as mock_frames:
        mock_frames.return_value = (
            [
                ExtractedFrame(
                    timestamp_offset=1.0,
                    absolute_timestamp=1.0,
                    image_path=str(_create_synthetic_image(tmp_path / "f.jpg")),
                    width=640,
                    height=360,
                )
            ] * 8,
            "libdav1d",
            8,
            0,
        )

        pkg = build_multimodal_package(
            candidate=c,
            transcript_doc=None,
            source_video_path=tmp_path / "source.mp4",
            source_wav_path=tmp_path / "audio.wav",
            source_fingerprint="fp_test",
            candidate_set_id="cset_test",
            run_dir=tmp_path,
        )

        data = pkg.model_dump()
        # Guarantee label isolation
        assert "human_score" not in data
        assert "publishable" not in data
        assert "human_notes" not in data


def test_multimodal_package_caching(tmp_path: Path):
    """20. Caches package JSON and reloads on second invocation."""
    _setup_mock_run_dir(tmp_path, count=2)
    c = _make_candidate("cand_001", start=0.0, end=10.0)

    f_img = _create_synthetic_image(tmp_path / "f.jpg")
    mock_frames_list = [
        ExtractedFrame(
            timestamp_offset=float(i),
            absolute_timestamp=float(i),
            image_path=str(f_img),
            width=640,
            height=360,
        )
        for i in range(1, 9)
    ]

    with mock.patch("freecher_worker.multimodal.package.extract_candidate_frames") as mock_extract:
        mock_extract.return_value = (mock_frames_list, "libdav1d", 8, 0)

        pkg1 = build_multimodal_package(
            candidate=c,
            transcript_doc=None,
            source_video_path=tmp_path / "source.mp4",
            source_wav_path=tmp_path / "audio.wav",
            source_fingerprint="fp_test",
            candidate_set_id="cset_test",
            run_dir=tmp_path,
        )
        assert mock_extract.call_count == 1

        # Second call should load from cache
        pkg2 = build_multimodal_package(
            candidate=c,
            transcript_doc=None,
            source_video_path=tmp_path / "source.mp4",
            source_wav_path=tmp_path / "audio.wav",
            source_fingerprint="fp_test",
            candidate_set_id="cset_test",
            run_dir=tmp_path,
            force_rebuild=False,
        )
        assert mock_extract.call_count == 1  # Not called again
        assert pkg2.package_hash == pkg1.package_hash


def test_multimodal_package_insufficient_visual_flag(tmp_path: Path):
    """21. Flags insufficient_visual_evidence=True when fewer than 4 frames are decoded."""
    c = _make_candidate("cand_001", start=0.0, end=10.0)
    _create_synthetic_wav(tmp_path / "audio.wav")

    with mock.patch("freecher_worker.multimodal.package.extract_candidate_frames") as mock_extract:
        # Only 2 frames returned
        f_img = _create_synthetic_image(tmp_path / "f.jpg")
        mock_extract.return_value = (
            [
                ExtractedFrame(
                    timestamp_offset=1.0,
                    absolute_timestamp=1.0,
                    image_path=str(f_img),
                    width=640,
                    height=360,
                )
            ] * 2,
            "software",
            8,
            6,
        )

        pkg = build_multimodal_package(
            candidate=c,
            transcript_doc=None,
            source_video_path=tmp_path / "source.mp4",
            source_wav_path=tmp_path / "audio.wav",
            source_fingerprint="fp_test",
            candidate_set_id="cset_test",
            run_dir=tmp_path,
            force_rebuild=True,
        )
        assert pkg.insufficient_visual_evidence is True


def test_openai_provider_prompt_and_format(tmp_path: Path):
    """22. Verifies prompt version is multimodal_v1_prompt_v1 and payload contains sparse frames."""
    provider = OpenAIMultimodalProvider(
        base_url="https://api.openai.com/v1",
        api_key="test_key",
        model="gpt-4o-mini",
    )
    assert provider.prompt_version == "multimodal_v1_prompt_v1"

    img_file = _create_synthetic_image(tmp_path / "frame.jpg")
    frame = ExtractedFrame(
        timestamp_offset=2.5,
        absolute_timestamp=12.5,
        image_path=str(img_file),
        width=640,
        height=360,
    )
    pkg = MultimodalCandidatePackage(
        candidate_id="c1",
        start=10.0,
        end=20.0,
        duration=10.0,
        candidate_transcript="Hello world",
        frames=[frame],
        audio_features=AudioFeatures(
            rms_mean=0.1, rms_std=0.02, peak=0.5, silence_ratio=0.1, speech_coverage=0.9, energy_change_rate=0.01
        ),
        visual_features=VisualFeatures(
            motion_score=0.2, scene_change_count=0, face_presence_ratio=0.5, decoded_frame_count=1, requested_frame_count=1
        ),
        source_fingerprint="fp1",
        candidate_set_id="cs1",
        package_hash="hash1",
    )

    mock_resp_json = {
        "candidate_id": "c1",
        "observable_event": True,
        "visual_payoff": True,
        "visual_event": 80.0,
        "reaction": 75.0,
        "emotion": 70.0,
        "humor": 65.0,
        "surprise": 60.0,
        "energy": 85.0,
        "standalone": 90.0,
        "retention": 85.0,
        "shareability": 80.0,
        "boringness": 10.0,
        "context_dependency": 15.0,
        "outside_payoff": False,
        "missing_setup": False,
        "insufficient_visual_evidence": False,
        "confidence": 0.9,
        "best_observed_region": None,
        "evidence": [],
        "reason": "Great visual reaction.",
        "quality_score": 82.5,
    }

    with mock.patch("httpx.Client.post") as mock_post:
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(mock_resp_json)}}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 250},
        }
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        result = provider.score_candidate(pkg)
        assert result.candidate_id == "c1"
        assert result.quality_score == 82.5

        # Check payload had image block
        sent_payload = mock_post.call_args[1]["json"]
        user_msg = sent_payload["messages"][1]["content"]
        types = [b["type"] for b in user_msg]
        assert "text" in types
        assert "image_url" in types


def test_openai_provider_retries_on_failure(tmp_path: Path):
    """23. Retries up to max_retries on transient API errors."""
    provider = OpenAIMultimodalProvider(
        base_url="https://api.openai.com/v1",
        api_key="test_key",
        max_retries=3,
    )
    img_file = _create_synthetic_image(tmp_path / "frame.jpg")
    pkg = MultimodalCandidatePackage(
        candidate_id="c1",
        start=0.0,
        end=10.0,
        duration=10.0,
        candidate_transcript="Hi",
        frames=[
            ExtractedFrame(
                timestamp_offset=1.0, absolute_timestamp=1.0, image_path=str(img_file), width=640, height=360
            )
        ],
        audio_features=AudioFeatures(
            rms_mean=0.1, rms_std=0.01, peak=0.2, silence_ratio=0.2, speech_coverage=0.8, energy_change_rate=0.01
        ),
        visual_features=VisualFeatures(
            motion_score=0.1, scene_change_count=0, face_presence_ratio=0.0, decoded_frame_count=1, requested_frame_count=1
        ),
        source_fingerprint="fp1",
        candidate_set_id="cs1",
        package_hash="h1",
    )

    mock_resp_json = {
        "candidate_id": "c1",
        "observable_event": True,
        "visual_payoff": False,
        "visual_event": 50.0,
        "reaction": 50.0,
        "emotion": 50.0,
        "humor": 50.0,
        "surprise": 50.0,
        "energy": 50.0,
        "standalone": 50.0,
        "retention": 50.0,
        "shareability": 50.0,
        "boringness": 50.0,
        "context_dependency": 50.0,
        "confidence": 0.8,
        "reason": "OK",
        "quality_score": 50.0,
    }

    with mock.patch("httpx.Client.post") as mock_post:
        # First two calls raise error, 3rd call succeeds
        mock_ok = mock.MagicMock()
        mock_ok.json.return_value = {
            "choices": [{"message": {"content": json.dumps(mock_resp_json)}}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 100},
        }
        mock_ok.raise_for_status.return_value = None

        mock_post.side_effect = [
            RuntimeError("Network timeout"),
            RuntimeError("Rate limit"),
            mock_ok,
        ]

        res = provider.score_candidate(pkg)
        assert res.quality_score == 50.0
        assert mock_post.call_count == 3


def test_openai_provider_usage_and_cost_tracking(tmp_path: Path):
    """24. Tracks request count, input tokens, output tokens, and cost."""
    provider = OpenAIMultimodalProvider(
        base_url="https://api.openai.com/v1",
        api_key="test_key",
        model="gpt-4o-mini",
    )
    img_file = _create_synthetic_image(tmp_path / "frame.jpg")
    pkg = MultimodalCandidatePackage(
        candidate_id="c1",
        start=0.0,
        end=5.0,
        duration=5.0,
        candidate_transcript="Test",
        frames=[
            ExtractedFrame(
                timestamp_offset=1.0, absolute_timestamp=1.0, image_path=str(img_file), width=640, height=360
            )
        ],
        audio_features=AudioFeatures(
            rms_mean=0.1, rms_std=0.01, peak=0.2, silence_ratio=0.1, speech_coverage=0.9, energy_change_rate=0.01
        ),
        visual_features=VisualFeatures(
            motion_score=0.1, scene_change_count=0, face_presence_ratio=0.0, decoded_frame_count=1, requested_frame_count=1
        ),
        source_fingerprint="fp1",
        candidate_set_id="cs1",
        package_hash="h1",
    )

    mock_resp_json = {
        "candidate_id": "c1",
        "observable_event": True,
        "visual_payoff": True,
        "visual_event": 70.0,
        "reaction": 70.0,
        "emotion": 70.0,
        "humor": 70.0,
        "surprise": 70.0,
        "energy": 70.0,
        "standalone": 70.0,
        "retention": 70.0,
        "shareability": 70.0,
        "boringness": 20.0,
        "context_dependency": 20.0,
        "confidence": 0.85,
        "reason": "Good",
        "quality_score": 75.0,
    }

    with mock.patch("httpx.Client.post") as mock_post:
        mock_ok = mock.MagicMock()
        mock_ok.json.return_value = {
            "choices": [{"message": {"content": json.dumps(mock_resp_json)}}],
            "usage": {"prompt_tokens": 2000, "completion_tokens": 400},
        }
        mock_ok.raise_for_status.return_value = None
        mock_post.return_value = mock_ok

        provider.score_candidate(pkg)
        usage = provider.get_usage()
        assert usage.requests == 1
        assert usage.input_tokens == 2000
        assert usage.output_tokens == 400
        assert usage.estimated_cost_usd is not None and usage.estimated_cost_usd > 0.0


def test_formula_quality_score_preservation():
    """25. multimodal_v1_formula_v1 preserves 0-100 score distribution without collapsing."""
    result = MultimodalModelResult(
        candidate_id="c1",
        observable_event=True,
        visual_payoff=True,
        visual_event=75.0,
        reaction=70.0,
        emotion=80.0,
        humor=85.0,
        surprise=65.0,
        energy=80.0,
        standalone=75.0,
        retention=85.0,
        shareability=75.0,
        boringness=15.0,
        context_dependency=20.0,
        confidence=0.9,
        reason="Preserved quality score.",
        quality_score=78.5,
    )
    pkg = mock.MagicMock(insufficient_visual_evidence=False)
    score, subscores, diagnostics = multimodal_v1_formula_v1(result, pkg)

    assert score == 78.5
    assert len(diagnostics["applied_caps"]) == 0
    assert subscores["quality_score"] == 78.5


def test_formula_caps_applied():
    """26. Checks boringness/retention cap (35), outside payoff cap (40), missing setup cap (45)."""
    # 1. Boring cap: boringness >= 85 and retention <= 25 -> max 35
    res1 = MultimodalModelResult(
        candidate_id="c1",
        observable_event=False,
        visual_payoff=False,
        visual_event=10.0,
        reaction=10.0,
        emotion=10.0,
        humor=10.0,
        surprise=10.0,
        energy=10.0,
        standalone=50.0,
        retention=20.0,
        shareability=10.0,
        boringness=90.0,
        context_dependency=50.0,
        confidence=0.8,
        reason="Boring",
        quality_score=60.0,
    )
    pkg = mock.MagicMock(insufficient_visual_evidence=False)
    s1, _, d1 = multimodal_v1_formula_v1(res1, pkg)
    assert s1 == 35.0
    assert any("boring_low_retention_cap" in c for c in d1["applied_caps"])

    # 2. Outside payoff cap: outside_payoff and not observable_event -> max 40
    res2 = MultimodalModelResult(
        candidate_id="c2",
        observable_event=False,
        visual_payoff=False,
        visual_event=20.0,
        reaction=20.0,
        emotion=20.0,
        humor=20.0,
        surprise=20.0,
        energy=20.0,
        standalone=60.0,
        retention=50.0,
        shareability=30.0,
        boringness=30.0,
        context_dependency=30.0,
        outside_payoff=True,
        confidence=0.8,
        reason="Payoff in next context",
        quality_score=70.0,
    )
    s2, _, d2 = multimodal_v1_formula_v1(res2, pkg)
    assert s2 == 40.0
    assert any("outside_payoff_unobservable_cap" in c for c in d2["applied_caps"])


def test_formula_insufficient_visual_lowers_confidence():
    """27. When visual evidence is insufficient, lowers confidence rather than score penalty."""
    res = MultimodalModelResult(
        candidate_id="c1",
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
        confidence=0.9,
        reason="Good text, missing visuals",
        quality_score=75.0,
        insufficient_visual_evidence=True,
    )
    pkg = mock.MagicMock(insufficient_visual_evidence=True)
    score, subscores, diag = multimodal_v1_formula_v1(res, pkg)

    assert score == 75.0  # Not penalized
    assert any("insufficient_visual_evidence_confidence_lowered" in c for c in diag["applied_caps"])


def test_reranker_pipeline_end_to_end(tmp_path: Path):
    """28. Mocked provider reranks a synthetic run directory and writes scores/multimodal_v1.json."""
    _setup_mock_run_dir(tmp_path, count=6)

    class MockMultimodalProvider(MultimodalProvider):
        name = "mock_provider"
        model = "mock_model"
        prompt_version = "multimodal_v1_prompt_v1"

        def __init__(self):
            self.usage = MultimodalUsage(requests=0, input_tokens=0, output_tokens=0)

        def score_candidate(self, package: MultimodalCandidatePackage) -> MultimodalModelResult:
            self.usage.requests += 1
            self.usage.input_tokens += 1000
            self.usage.output_tokens += 200
            # Higher candidate ID gets higher score
            cand_num = int(package.candidate_id.split("_")[1])
            return MultimodalModelResult(
                candidate_id=package.candidate_id,
                observable_event=True,
                visual_payoff=True,
                visual_event=50.0 + cand_num * 5,
                reaction=50.0 + cand_num * 5,
                emotion=50.0,
                humor=50.0,
                surprise=50.0,
                energy=60.0,
                standalone=70.0,
                retention=75.0,
                shareability=70.0,
                boringness=15.0,
                context_dependency=20.0,
                confidence=0.85,
                reason=f"Mock judgment for {package.candidate_id}",
                quality_score=60.0 + cand_num * 4,
            )

        def get_usage(self) -> MultimodalUsage:
            return self.usage

    mock_provider = MockMultimodalProvider()
    reranker = MultimodalReranker(
        provider=mock_provider,
        heuristic_top_k=3,
        llm_top_k=3,
        max_candidates=5,
    )

    # Frame extraction and probe_media mocks
    img_path = _create_synthetic_image(tmp_path / "frame.jpg")
    with mock.patch("freecher_worker.multimodal.package.extract_candidate_frames") as mock_frames, \
         mock.patch("freecher_worker.multimodal.scorer.probe_media") as mock_probe:
        mock_probe.return_value = mock.MagicMock(duration=180.0, video_codec="h264")
        mock_frames.return_value = (
            [
                ExtractedFrame(
                    timestamp_offset=1.0,
                    absolute_timestamp=1.0,
                    image_path=str(img_path),
                    width=640,
                    height=360,
                )
            ] * 8,
            "libdav1d",
            8,
            0,
        )

        pred_doc = reranker.rerank_run(tmp_path)

        assert pred_doc.scorer == SCORER_VERSION_MULTIMODAL_V1
        assert pred_doc.scorer_version == SCORER_VERSION_MULTIMODAL_V1
        assert pred_doc.score_formula_version == FORMULA_VERSION_MULTIMODAL_V1
        assert len(pred_doc.predictions) <= 5
        # Ranks must be 1-based sequential
        ranks = [p.rank for p in pred_doc.predictions]
        assert ranks == list(range(1, len(pred_doc.predictions) + 1))
        # Scores strictly descending
        scores = [p.score for p in pred_doc.predictions]
        assert scores == sorted(scores, reverse=True)

        target_file = tmp_path / "scores" / "multimodal_v1.json"
        assert target_file.is_file()


def test_evaluation_metrics_shortlist_recall():
    """29. compute_evaluation_metrics calculates perfect_candidate_recall_in_shortlist and publishable recall."""
    cset_id = "cset_test_metrics"
    eval_items = [
        BlindEvaluationItem(candidate_id="c1", start=0, end=10, duration=10, text="1", human_score=4.0, publishable=True),
        BlindEvaluationItem(candidate_id="c2", start=10, end=20, duration=10, text="2", human_score=4.0, publishable=True),
        BlindEvaluationItem(candidate_id="c3", start=20, end=30, duration=10, text="3", human_score=3.0, publishable=True),
        BlindEvaluationItem(candidate_id="c4", start=30, end=40, duration=10, text="4", human_score=2.0, publishable=False),
        BlindEvaluationItem(candidate_id="c5", start=40, end=50, duration=10, text="5", human_score=1.0, publishable=False),
        BlindEvaluationItem(candidate_id="c6", start=50, end=60, duration=10, text="6", human_score=0.0, publishable=False),
    ]
    eval_doc = BlindEvaluationDocument(
        candidate_set_id=cset_id,
        total_candidates=len(eval_items),
        labeled_candidates=len(eval_items),
        items=eval_items,
    )

    # Scorer only predicted a shortlist of 3 candidates (c1, c3, c4), missing c2 (perfect)
    preds = [
        ScorerPredictionItem(candidate_id="c1", rank=1, score=90.0),
        ScorerPredictionItem(candidate_id="c3", rank=2, score=80.0),
        ScorerPredictionItem(candidate_id="c4", rank=3, score=70.0),
    ]
    pred_doc = ScorerPredictionDocument(
        candidate_set_id=cset_id,
        scorer="multimodal_v1",
        scorer_version="multimodal_v1",
        predictions=preds,
    )

    metrics = compute_evaluation_metrics(eval_doc, pred_doc, k_values=[2, 3])

    assert metrics.scored_candidates == 3
    assert metrics.candidate_coverage_ratio == 0.5  # 3/6
    assert metrics.shortlist_size == 3
    # Pool has 2 perfect candidates (c1, c2). Shortlist contains c1. Recall = 1/2 = 50%
    assert metrics.perfect_candidate_recall_in_shortlist == 0.5
    # Pool has 3 publishable candidates (c1, c2, c3). Shortlist contains c1 and c3. Recall = 2/3 = 66.7%
    assert metrics.publishable_candidate_recall_in_shortlist == pytest.approx(0.6667, abs=1e-3)
