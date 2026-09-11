"""Integration tests for Phase 2 short-form vertical video rendering pipeline and CLI."""

import subprocess
from pathlib import Path
import pytest
from typer.testing import CliRunner

from freecher_worker.cli import app
from freecher_worker.highlights.models import Highlight, HighlightScore
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
from freecher_worker.rendering import (
    RefinedWordsDocument,
    WordItem,
    get_preset,
    render_highlights_for_run,
    render_single_short,
    validate_rendered_video,
)
from freecher_worker.transcription.models import Transcript, TranscriptSegment
from freecher_worker.utils.json_io import load_json, save_json

runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_whisper_words(monkeypatch):
    """Avoid downloading Whisper model in integration tests by returning synthetic word timestamps."""
    def fake_transcribe(
        self,
        source_media: Path,
        refined_start: float,
        refined_end: float,
        source_fingerprint_id: str,
        language: str = None,
        cache_path: Path = None,
        force: bool = False,
    ) -> RefinedWordsDocument:
        dur = round(refined_end - refined_start, 3)
        words = [
            WordItem(word="Привет", start=0.1, end=0.8, probability=0.99),
            WordItem(word="всем", start=0.9, end=1.4, probability=0.98),
            WordItem(word="Это", start=1.5, end=1.9, probability=0.99),
            WordItem(word="тест", start=2.0, end=2.5, probability=0.97),
        ]
        doc = RefinedWordsDocument(
            cache_key="mock_test_key",
            language=language or "ru",
            start_offset=refined_start,
            duration=dur,
            model="mock",
            compute_type="none",
            words=words,
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(doc, cache_path)
        return doc

    monkeypatch.setattr(
        "freecher_worker.rendering.renderer.HighlightWordTranscriber.transcribe_highlight",
        fake_transcribe,
    )


def _create_synthetic_av_video(output_mp4: Path, duration: float = 4.0) -> Path:
    """Create a 1920x1080 landscape video with synthetic video pattern and audio tone."""
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-f", "lavfi",
        "-i", f"testsrc=size=1920x1080:rate=25:duration={duration}",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_mp4),
    ]
    res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        pytest.skip(f"FFmpeg synthetic media generation failed: {res.stderr}")
    return output_mp4


def _setup_mock_run_with_media(tmp_path: Path) -> Path:
    """Setup a complete mock run directory with synthetic video and transcript."""
    run_dir = tmp_path / "run_20260907_render_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    source_video = _create_synthetic_av_video(tmp_path / "source.mp4", duration=4.0)

    transcript = Transcript(
        language="ru",
        duration=4.0,
        model="small",
        compute_type="int8_float16",
        device="cpu",
        segments=[
            TranscriptSegment(
                id=0,
                start=0.2,
                end=1.8,
                text="Привет всем.",
            ),
            TranscriptSegment(
                id=1,
                start=2.0,
                end=3.5,
                text="Это тест.",
            ),
        ],
    )
    save_json(transcript, run_dir / "transcript.json")

    hl1 = Highlight(
        rank=1,
        start=0.5,
        end=3.2,
        duration=2.7,
        score=92.5,
        reason="Captivating hook and punchline",
        candidate_id="cand_0001",
        text="Привет всем. Это тест.",
        file="clips/clip_01.mp4",
        score_breakdown=HighlightScore(
            score=92.5,
            hook_score=95.0,
            standalone_score=90.0,
            emotion_score=90.0,
            information_score=85.0,
            shareability_score=95.0,
            reason="Captivating hook",
        ),
    )
    save_json([hl1], run_dir / "highlights.json")

    manifest = Manifest(
        pipeline_version="0.2.0",
        created_at="2026-09-07T10:00:00",
        source=str(source_video),
        source_fingerprint=SourceFingerprint(
            path=str(source_video),
            file_size=source_video.stat().st_size,
            mtime_ns=int(source_video.stat().st_mtime_ns),
            duration_seconds=4.0,
            content_hash="mock_hash_render",
            fingerprint_id="fp_test_render",
        ),
        environment=EnvironmentInfo(python_version="3.12.0", platform="Linux"),
        asr=AsrManifestInfo(model="small", device="cuda", compute_type="int8_float16", language="ru"),
        candidate_config=CandidateConfigInfo(min_seconds=2.0, target_seconds=3.0, max_seconds=4.0, overlap=1.0),
        scoring=ScoringManifestInfo(scorer="heuristic", scorer_version="1.1.0"),
        ranking=RankingManifestInfo(top_k=1, dedup_threshold=0.60),
        timings=PipelineTimings(total_seconds=5.0),
        statistics=PipelineStatistics(transcript_segment_count=2, candidate_count=1, selected_highlight_count=1),
        highlights=[
            HighlightManifestItem(
                rank=1,
                start=0.5,
                end=3.2,
                duration=2.7,
                score=92.5,
                reason="Captivating hook",
                file="clips/clip_01.mp4",
                candidate_id="cand_0001",
            )
        ],
    )
    save_json(manifest, run_dir / "manifest.json")
    return run_dir


def test_render_single_short_end_to_end(tmp_path):
    run_dir = _setup_mock_run_with_media(tmp_path)
    manifest = load_json(run_dir / "manifest.json")
    source_video = Path(manifest["source"])
    transcript = Transcript.model_validate(load_json(run_dir / "transcript.json"))
    highlights = [Highlight.model_validate(h) for h in load_json(run_dir / "highlights.json")]

    preset = get_preset("shorts")

    item_manifest = render_single_short(
        source_video=source_video,
        highlight=highlights[0],
        transcript=transcript,
        source_fingerprint_id="fp_test_render",
        video_duration=4.0,
        run_dir=run_dir,
        preset=preset,
        enable_smart_crop=True,
        enable_subtitles=True,
        enable_audio_normalization=True,
        force=True,
    )

    assert item_manifest.rank == 1
    assert item_manifest.candidate_id == "cand_0001"
    assert item_manifest.resolution == "1080x1920"

    output_file = run_dir / item_manifest.file
    assert output_file.is_file()
    assert output_file.stat().st_size > 1000

    # Validate with ffprobe
    val = validate_rendered_video(output_file, expected_duration=item_manifest.duration)
    assert val.passed is True
    assert val.width == 1080
    assert val.height == 1920
    assert val.has_audio is True

    # Check cached artifacts
    crop_files = list((run_dir / "crop_paths").glob("*.json"))
    assert len(crop_files) >= 1
    ass_files = list((run_dir / "subtitles").glob("*.ass"))
    assert len(ass_files) >= 1
    words_files = list((run_dir / "words").glob("*.json"))
    assert len(words_files) >= 1

    # Verify idempotency / caching without force
    item_reused = render_single_short(
        source_video=source_video,
        highlight=highlights[0],
        transcript=transcript,
        source_fingerprint_id="fp_test_render",
        video_duration=4.0,
        run_dir=run_dir,
        preset=preset,
        force=False,
    )
    assert item_reused.file == item_manifest.file


def test_render_highlights_for_run(tmp_path):
    run_dir = _setup_mock_run_with_media(tmp_path)
    render_manifest = render_highlights_for_run(
        run_dir=run_dir,
        preset_name="shorts",
        top_k=1,
        force=True,
    )

    assert render_manifest.top_k == 1
    assert len(render_manifest.shorts) == 1
    assert (run_dir / "render_manifest.json").is_file()

    saved_man = load_json(run_dir / "render_manifest.json")
    assert saved_man["preset"] == "shorts"
    assert len(saved_man["shorts"]) == 1


def test_cli_render_command(tmp_path):
    run_dir = _setup_mock_run_with_media(tmp_path)
    result = runner.invoke(app, ["render", str(run_dir), "--top-k", "1", "--force"])

    assert result.exit_code == 0
    assert "Starting Short-Form Rendering MVP" in result.output
    assert "Successfully rendered 1 vertical video" in result.output
    assert (run_dir / "render_manifest.json").is_file()


def test_cli_render_highlight_command(tmp_path):
    run_dir = _setup_mock_run_with_media(tmp_path)
    result = runner.invoke(app, ["render-highlight", str(run_dir), "--rank", "1", "--force"])

    assert result.exit_code == 0
    assert "Rendering Highlight #1" in result.output
    assert "Rendered Short #1" in result.output
    assert "1080x1920" in result.output


def test_cli_inspect_displays_rendered_shorts(tmp_path):
    run_dir = _setup_mock_run_with_media(tmp_path)
    # Render first
    render_res = runner.invoke(app, ["render", str(run_dir), "--top-k", "1", "--force"])
    assert render_res.exit_code == 0

    # Now inspect
    res = runner.invoke(app, ["inspect", str(run_dir)])
    assert res.exit_code == 0
    assert "Rendered Vertical Shorts (1):" in res.output
    assert "1080x1920" in res.output


# ==============================================================================
# --source-video override (explicit only, no guessing)
# ==============================================================================


def _make_r2_style_run(tmp_path):
    """A run whose manifest records an s3:// URI, as an R2-backed run does.

    The video exists on disk but the manifest cannot name it, which is exactly
    the situation the override is for.
    """
    run_dir = _setup_mock_run_with_media(tmp_path)
    manifest = load_json(run_dir / "manifest.json")
    real_video = Path(manifest["source"])
    manifest["source"] = "s3://bucket/processing/abc123/transcript.json"
    save_json(manifest, run_dir / "manifest.json")
    return run_dir, real_video


def test_render_highlight_uses_the_explicit_source_video(tmp_path):
    run_dir, real_video = _make_r2_style_run(tmp_path)

    # Without the override the manifest is unusable and the command must refuse.
    without = runner.invoke(app, ["render-highlight", str(run_dir), "--rank", "1", "--force"])
    assert without.exit_code == 1
    assert "Source video does not exist" in without.output

    # With it, the render proceeds from the named file.
    with_flag = runner.invoke(app, [
        "render-highlight", str(run_dir), "--rank", "1", "--force",
        "--source-video", str(real_video),
    ])
    assert with_flag.exit_code == 0, with_flag.output


def test_render_batch_uses_the_explicit_source_video(tmp_path):
    run_dir, real_video = _make_r2_style_run(tmp_path)

    without = runner.invoke(app, ["render", str(run_dir), "--top-k", "1", "--force"])
    assert without.exit_code != 0

    with_flag = runner.invoke(app, [
        "render", str(run_dir), "--top-k", "1", "--force", "--source-video", str(real_video),
    ])
    assert with_flag.exit_code == 0, with_flag.output
    assert (run_dir / "render_manifest.json").is_file()


def test_a_missing_override_fails_with_a_clear_message(tmp_path):
    run_dir = _setup_mock_run_with_media(tmp_path)
    missing = tmp_path / "nowhere" / "absent.mp4"

    result = runner.invoke(app, [
        "render-highlight", str(run_dir), "--rank", "1", "--force", "--source-video", str(missing),
    ])

    assert result.exit_code == 1
    # Names the flag, so the reader knows which input was wrong.
    assert "--source-video" in result.output
    # Rich wraps at the console width, which is narrower on a server than in a
    # developer terminal -- there the path broke as "absent.\nmp4". Strip the
    # wrapping before matching, so this asserts the message content and not the
    # width of whatever terminal happened to run it.
    assert "absent.mp4" in "".join(result.output.split())


def test_a_directory_is_not_accepted_as_a_source_video(tmp_path):
    run_dir = _setup_mock_run_with_media(tmp_path)

    result = runner.invoke(app, [
        "render-highlight", str(run_dir), "--rank", "1", "--force", "--source-video", str(tmp_path),
    ])

    assert result.exit_code == 1
    assert "--source-video" in result.output


def test_the_override_is_not_consulted_when_absent(tmp_path):
    """Backward compatibility: without the flag the manifest still decides."""
    run_dir = _setup_mock_run_with_media(tmp_path)
    manifest_path = Path(load_json(run_dir / "manifest.json")["source"])

    result = runner.invoke(app, ["render-highlight", str(run_dir), "--rank", "1", "--force"])

    assert result.exit_code == 0, result.output
    # The file the manifest names is the one that was rendered from. Only the
    # basename is checked: rich wraps long paths across lines in the output.
    assert manifest_path.name in result.output
    assert "does not exist" not in result.output


def test_the_library_function_defaults_to_the_manifest(tmp_path):
    run_dir = _setup_mock_run_with_media(tmp_path)

    # No override argument at all: the pre-existing signature still works.
    render_manifest = render_highlights_for_run(run_dir=run_dir, top_k=1, force=True)

    assert len(render_manifest.shorts) == 1


def test_the_library_function_rejects_a_missing_override(tmp_path):
    run_dir = _setup_mock_run_with_media(tmp_path)

    with pytest.raises(FileNotFoundError, match="--source-video"):
        render_highlights_for_run(
            run_dir=run_dir, top_k=1, force=True,
            source_video_override=tmp_path / "absent.mp4",
        )


# ==============================================================================
# render-highlight: an explicit candidate outside the top-K
# ==============================================================================


def _add_unranked_candidate(run_dir, candidate_id="cand_0099", start=0.2, end=2.4):
    """Put a candidate in candidates.json that highlights.json does not list."""
    doc = {
        "candidate_set_id": "cset_render_test",
        "transcript_hash": "hash_render_test",
        "min_seconds": 2.0,
        "target_seconds": 3.0,
        "max_seconds": 4.0,
        "overlap_seconds": 1.0,
        "candidates": [
            {
                "id": candidate_id,
                "start": start,
                "end": end,
                "duration": round(end - start, 2),
                "text": "Кандидат вне топ-К.",
                "segment_ids": [0],
            }
        ],
    }
    save_json(doc, run_dir / "candidates.json")
    return doc["candidates"][0]


def test_a_candidate_present_in_highlights_still_renders(tmp_path):
    run_dir = _setup_mock_run_with_media(tmp_path)
    _add_unranked_candidate(run_dir)  # unrelated extra candidate

    result = runner.invoke(app, [
        "render-highlight", str(run_dir), "--candidate-id", "cand_0001", "--force",
    ])

    assert result.exit_code == 0, result.output
    # The ranked path is taken, so no fallback note appears.
    assert "not in highlights.json" not in result.output


def test_a_candidate_outside_the_top_k_renders_from_the_frozen_set(tmp_path):
    run_dir = _setup_mock_run_with_media(tmp_path)
    window = _add_unranked_candidate(run_dir, "cand_0099", start=0.2, end=2.4)

    result = runner.invoke(app, [
        "render-highlight", str(run_dir), "--candidate-id", "cand_0099", "--force",
    ])

    assert result.exit_code == 0, result.output
    assert "not in highlights.json" in result.output
    # The frozen boundaries are the ones used, not a re-derived window.
    assert f"{window['start']:.2f}-{window['end']:.2f}s" in result.output


def test_the_frozen_boundaries_survive_into_the_render(tmp_path):
    run_dir = _setup_mock_run_with_media(tmp_path)
    window = _add_unranked_candidate(run_dir, "cand_0099", start=0.3, end=2.1)

    result = runner.invoke(app, [
        "render-highlight", str(run_dir), "--candidate-id", "cand_0099", "--force",
    ])
    assert result.exit_code == 0, result.output

    # The window the renderer starts from is the frozen one. What it finally
    # cuts may differ -- boundary refinement is a separate existing feature --
    # so this pins the input, which is what "frozen" means here.
    assert f"{window['start']:.2f}-{window['end']:.2f}s" in result.output
    # rank 0 marks a candidate rendered outside any ranking.
    rendered = run_dir / "final" / "short_00.mp4"
    assert rendered.is_file(), f"no rendered file at {rendered}"


def test_a_candidate_in_neither_file_fails_clearly(tmp_path):
    run_dir = _setup_mock_run_with_media(tmp_path)
    _add_unranked_candidate(run_dir, "cand_0099")

    result = runner.invoke(app, [
        "render-highlight", str(run_dir), "--candidate-id", "cand_nope", "--force",
    ])

    assert result.exit_code == 1
    assert "cand_nope" in result.output
    # Both places that were searched are named, so the reader can check either.
    assert "highlights.json" in result.output and "candidates.json" in result.output


def test_the_fallback_does_not_apply_to_rank(tmp_path):
    """--rank only means something inside a ranking, so no fallback for it."""
    run_dir = _setup_mock_run_with_media(tmp_path)
    _add_unranked_candidate(run_dir, "cand_0099")

    result = runner.invoke(app, ["render-highlight", str(run_dir), "--rank", "99", "--force"])

    assert result.exit_code == 1
    assert "candidates.json" not in result.output


def test_batch_render_is_unaffected_by_the_fallback(tmp_path):
    """Only render-highlight gained this; `render` still renders the ranking."""
    run_dir = _setup_mock_run_with_media(tmp_path)
    _add_unranked_candidate(run_dir, "cand_0099")

    result = runner.invoke(app, ["render", str(run_dir), "--top-k", "5", "--force"])

    assert result.exit_code == 0, result.output
    rendered = {item["candidate_id"] for item in load_json(run_dir / "render_manifest.json")["shorts"]}
    assert "cand_0099" not in rendered


# ==============================================================================
# --no-refine-boundaries: cut the frozen window exactly
# ==============================================================================


def test_refinement_runs_by_default(tmp_path):
    """Backward compatibility: without the flag the window is still refined."""
    run_dir = _setup_mock_run_with_media(tmp_path)

    result = runner.invoke(app, ["render-highlight", str(run_dir), "--rank", "1", "--force"])

    assert result.exit_code == 0, result.output
    assert "boundary refinement disabled" not in result.output


def test_disabling_refinement_keeps_the_frozen_boundaries_exactly(tmp_path):
    run_dir = _setup_mock_run_with_media(tmp_path)
    window = _add_unranked_candidate(run_dir, "cand_0099", start=0.4, end=2.3)

    result = runner.invoke(app, [
        "render-highlight", str(run_dir), "--candidate-id", "cand_0099", "--force",
        "--no-refine-boundaries",
    ])

    assert result.exit_code == 0, result.output
    assert "boundary refinement disabled" in result.output
    # The rendered window is the frozen one, to the second.
    from freecher_worker.cli import _format_timestamp

    assert _format_timestamp(window["start"]) in result.output
    assert _format_timestamp(window["end"]) in result.output


def test_the_disabled_path_never_calls_the_refiner(monkeypatch, tmp_path):
    """Not merely 'the numbers match': the refinement logic must not run."""
    import freecher_worker.rendering.renderer as renderer

    calls = []
    original = renderer.refine_highlight
    monkeypatch.setattr(
        renderer, "refine_highlight",
        lambda **kwargs: calls.append(kwargs) or original(**kwargs),
    )

    run_dir = _setup_mock_run_with_media(tmp_path)
    _add_unranked_candidate(run_dir, "cand_0099", start=0.4, end=2.3)

    runner.invoke(app, [
        "render-highlight", str(run_dir), "--candidate-id", "cand_0099", "--force",
        "--no-refine-boundaries",
    ])
    assert calls == []

    runner.invoke(app, ["render-highlight", str(run_dir), "--rank", "1", "--force"])
    assert len(calls) == 1


def test_the_library_default_still_refines(tmp_path):
    """render_single_short keeps its old behaviour when the flag is absent."""
    run_dir = _setup_mock_run_with_media(tmp_path)
    manifest = load_json(run_dir / "manifest.json")
    transcript = Transcript.model_validate(load_json(run_dir / "transcript.json"))
    highlights = [Highlight.model_validate(h) for h in load_json(run_dir / "highlights.json")]

    item = render_single_short(
        source_video=Path(manifest["source"]),
        highlight=highlights[0],
        transcript=transcript,
        source_fingerprint_id="fp_test_render",
        video_duration=4.0,
        run_dir=run_dir,
        preset=get_preset("shorts"),
        force=True,
    )
    assert "disabled" not in item.refinement_reason


def test_batch_render_is_unaffected_by_the_refinement_flag(tmp_path):
    """The flag exists only on render-highlight."""
    result = runner.invoke(app, ["render", "--help"])
    assert "--no-refine-boundaries" not in result.output
