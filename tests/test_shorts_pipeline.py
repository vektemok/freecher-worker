"""End-to-end vertical short production: refinement -> reframing -> 1080x1920 MP4 -> metadata."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from freecher_worker.cli import app
from freecher_worker.config import Settings
from freecher_worker.shorts.render import (
    ENCODER_NVENC,
    ENCODER_X264,
    build_vertical_filter,
    escape_filtergraph_value,
    short_stem,
    load_run_context,
    render_short,
    render_shorts_for_run,
    resolve_encoder,
    resolve_timeframe,
    select_ranked_candidates,
)
import freecher_worker.shorts.render as render_short_module
from freecher_worker.shorts.trajectory import (
    CROP_DRIVER_SENDCMD,
    CROP_DRIVER_STATIC,
    CropDriverPlan,
)

FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg is not installed")

SOURCE_W, SOURCE_H = 1280, 720
SOURCE_SECONDS = 90


def make_source_video(path: Path) -> Path:
    """Synthesize a landscape clip with audio, wide enough for a 60 s candidate."""
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y",
            "-f", "lavfi", "-i", f"testsrc=size={SOURCE_W}x{SOURCE_H}:rate=25:duration={SOURCE_SECONDS}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={SOURCE_SECONDS}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(path),
        ],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return path


CANDIDATE_START = 10.0
CANDIDATE_END = 70.0


def transcript_segments():
    spans = [(0.0, 1.2, "Ну.")]
    spans += [
        (3.0, 8.0, "Так вот, представь себе ситуацию, мы стоим посреди дороги и ничего не понимаем."),
        (8.0, 14.0, "И тут он поворачивается, смотрит на нас совершенно спокойно и говорит невероятное."),
        (14.0, 20.0, "Он говорит что всю неделю вообще не спал и делал это исключительно ради спора."),
        (20.0, 26.0, "И в этот момент мы все просто взорвались от смеха, потому что это была правда!"),
    ]
    spans += [(float(i), float(i) + 1.0, "Ага.") for i in range(28, 60, 5)]
    return [
        {"id": i, "start": CANDIDATE_START + a, "end": CANDIDATE_START + b, "text": t}
        for i, (a, b, t) in enumerate(spans)
    ]


def build_run(tmp_path: Path, advisory: dict | None = None) -> Path:
    """Materialize a minimal but realistic run directory."""
    run_dir = tmp_path / "runs" / "benchmark_test"
    run_dir.mkdir(parents=True)
    source = make_source_video(tmp_path / "source.mp4")

    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "source_fingerprint": {"fingerprint_id": "fp_test", "duration_seconds": SOURCE_SECONDS},
            }
        )
    )
    (run_dir / "transcript.json").write_text(
        json.dumps(
            {
                "language": "ru", "duration": float(SOURCE_SECONDS), "model": "small",
                "compute_type": "int8", "device": "cpu", "segments": transcript_segments(),
            }
        )
    )
    (run_dir / "candidates.json").write_text(
        json.dumps(
            {
                "transcript_hash": "abc123", "min_seconds": 30.0, "target_seconds": 60.0,
                "max_seconds": 90.0, "overlap_seconds": 15.0,
                "candidates": [
                    {
                        "id": "cand_037", "start": CANDIDATE_START, "end": CANDIDATE_END,
                        "duration": 60.0, "text": "test candidate", "segment_ids": [],
                    },
                    {
                        "id": "cand_038", "start": 20.0, "end": 80.0,
                        "duration": 60.0, "text": "second candidate", "segment_ids": [],
                    },
                ],
            }
        )
    )
    (run_dir / "highlights.json").write_text(
        json.dumps(
            [
                {"rank": 1, "start": CANDIDATE_START, "end": CANDIDATE_END, "duration": 60.0,
                 "score": 88.0, "reason": "test", "candidate_id": "cand_037", "text": "test candidate"},
                {"rank": 2, "start": 20.0, "end": 80.0, "duration": 60.0,
                 "score": 71.0, "reason": "test", "candidate_id": "cand_038", "text": "second candidate"},
            ]
        )
    )
    if advisory is not None:
        scores_dir = run_dir / "scores"
        scores_dir.mkdir()
        (scores_dir / "multimodal_v1_1.json").write_text(
            json.dumps(
                {
                    "candidate_set_id": "cset_test", "scorer": "multimodal_v1_1",
                    "predictions": [
                        {"candidate_id": "cand_037", "rank": 1, "score": 88.0,
                         "best_observed_region": advisory}
                    ],
                }
            )
        )
    return run_dir


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
    )
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Encoder selection (must never require NVENC)
# ---------------------------------------------------------------------------


def test_libx264_can_always_be_forced(monkeypatch):
    monkeypatch.setattr("freecher_worker.shorts.render.is_nvenc_available", lambda: True)
    assert resolve_encoder(ENCODER_X264) == ENCODER_X264


def test_auto_falls_back_to_libx264_without_nvenc(monkeypatch):
    monkeypatch.setattr("freecher_worker.shorts.render.is_nvenc_available", lambda: False)
    assert resolve_encoder("auto") == ENCODER_X264


def test_explicit_nvenc_request_degrades_instead_of_failing(monkeypatch):
    """A GTX 1650 under WSL2 has no usable NVENC; asking for it must not break the render."""
    monkeypatch.setattr("freecher_worker.shorts.render.is_nvenc_available", lambda: False)
    assert resolve_encoder(ENCODER_NVENC) == ENCODER_X264


def test_auto_uses_nvenc_when_it_is_actually_present(monkeypatch):
    monkeypatch.setattr("freecher_worker.shorts.render.is_nvenc_available", lambda: True)
    assert resolve_encoder("auto") == ENCODER_NVENC


# ---------------------------------------------------------------------------
# Filter chain
# ---------------------------------------------------------------------------


def test_vertical_filter_always_ends_at_the_target_resolution():
    driver = CropDriverPlan(
        driver=CROP_DRIVER_STATIC, crop_w=404, crop_h=720, x_expr="200", y_expr="0", keyframes=1
    )
    chain = build_vertical_filter(driver, 1080, 1920)
    assert chain.startswith("crop=404:720:200:0")
    assert "scale=1080:1920:force_original_aspect_ratio=increase" in chain
    assert chain.endswith("crop=1080:1920,setsar=1")


def test_vertical_filter_prepends_sendcmd_for_dynamic_trajectories(tmp_path):
    driver = CropDriverPlan(
        driver=CROP_DRIVER_SENDCMD, crop_w=404, crop_h=720, x_expr="200", y_expr="0",
        script="0.0-0.2 crop x '200';\n", keyframes=2,
    )
    script = tmp_path / "cmds.txt"
    chain = build_vertical_filter(driver, 1080, 1920, script)
    assert chain.startswith(f"sendcmd=f={escape_filtergraph_value(str(script))},crop=404:720:")


def test_sendcmd_driver_requires_a_script_path():
    driver = CropDriverPlan(
        driver=CROP_DRIVER_SENDCMD, crop_w=404, crop_h=720, x_expr="0", y_expr="0", script="x"
    )
    with pytest.raises(ValueError, match="script path"):
        build_vertical_filter(driver, 1080, 1920)


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_run_context_exposes_candidates_ranking_and_advisory(tmp_path):
    run_dir = build_run(tmp_path, advisory={"start_offset": 3.0, "end_offset": 26.0, "confidence": 0.8})
    context = load_run_context(run_dir)

    assert context.transcript is not None
    assert set(context.candidates) == {"cand_037", "cand_038"}
    assert [h.candidate_id for h in context.highlights] == ["cand_037", "cand_038"]
    assert context.advisory_regions["cand_037"]["end_offset"] == 26.0
    assert select_ranked_candidates(context, top=1)[0] == ["cand_037"]

    timeframe = resolve_timeframe(context, "cand_037")
    assert timeframe.source_start_sec == CANDIDATE_START
    assert timeframe.duration_sec == pytest.approx(60.0)


@needs_ffmpeg
def test_unknown_candidate_is_reported_clearly(tmp_path):
    context = load_run_context(build_run(tmp_path))
    with pytest.raises(KeyError, match="cand_999"):
        resolve_timeframe(context, "cand_999")


# ---------------------------------------------------------------------------
# End-to-end render
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_render_short_produces_a_valid_1080x1920_mp4(tmp_path):
    run_dir = build_run(tmp_path)
    context = load_run_context(run_dir)
    timeframe = resolve_timeframe(context, "cand_037")
    output = run_dir / "shorts" / "short_cand_037.mp4"

    metadata = render_short(
        source_video=context.source_video,
        timeframe=timeframe,
        output_path=output,
        transcript=context.transcript,
        run_dir=run_dir,
        settings=Settings(reframe_analysis_fps=2.0),
        enable_audio_normalization=False,
        encoder=ENCODER_X264,
    )

    assert output.is_file() and output.stat().st_size > 0
    info = probe(output)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    audio = next(s for s in info["streams"] if s["codec_type"] == "audio")

    assert (video["width"], video["height"]) == (1080, 1920)
    assert video["codec_name"] == "h264"
    assert audio["codec_name"] == "aac"
    assert metadata.aspect_ratio == "9:16"
    assert metadata.encoder == ENCODER_X264
    assert metadata.validation is not None and metadata.validation.valid
    assert float(info["format"]["duration"]) == pytest.approx(metadata.duration_sec, abs=0.6)


@needs_ffmpeg
def test_rendered_short_is_much_shorter_than_the_candidate(tmp_path):
    """The whole point of the stage: a 60 s candidate must not become a 60 s short."""
    run_dir = build_run(tmp_path)
    manifest = render_shorts_for_run(
        run_dir=run_dir, candidate_ids=["cand_037"],
        settings=Settings(reframe_analysis_fps=2.0),
        enable_audio_normalization=False, encoder=ENCODER_X264,
    )
    item = manifest.shorts[0]

    assert item.candidate_duration_sec == pytest.approx(60.0)
    assert 8.0 <= item.duration_sec <= 45.0
    assert item.duration_sec < 40.0
    assert item.short_start_offset_sec > 1.0


@needs_ffmpeg
def test_metadata_json_matches_the_documented_schema(tmp_path):
    run_dir = build_run(tmp_path)
    render_shorts_for_run(
        run_dir=run_dir, candidate_ids=["cand_037"],
        settings=Settings(reframe_analysis_fps=2.0),
        enable_audio_normalization=False, encoder=ENCODER_X264,
    )
    payload = json.loads((run_dir / "shorts" / "short_cand_037.json").read_text())

    for key in (
        "candidate_id", "source_start_sec", "source_end_sec",
        "short_start_offset_sec", "short_end_offset_sec",
        "short_source_start_sec", "short_source_end_sec",
        "duration_sec", "aspect_ratio", "width", "height",
        "reframing_mode", "duration_mode",
    ):
        assert key in payload, f"missing {key}"

    assert payload["aspect_ratio"] == "9:16"
    assert (payload["width"], payload["height"]) == (1080, 1920)
    assert payload["duration_mode"] == "auto"
    assert payload["source_start_sec"] == pytest.approx(CANDIDATE_START)
    assert payload["short_source_start_sec"] == pytest.approx(
        payload["source_start_sec"] + payload["short_start_offset_sec"], abs=1e-3
    )
    assert payload["short_source_end_sec"] == pytest.approx(
        payload["source_start_sec"] + payload["short_end_offset_sec"], abs=1e-3
    )
    assert 0.0 <= payload["short_start_offset_sec"] < payload["short_end_offset_sec"] <= 60.0


@needs_ffmpeg
def test_metadata_records_the_required_quality_diagnostics(tmp_path):
    run_dir = build_run(tmp_path)
    manifest = render_shorts_for_run(
        run_dir=run_dir, candidate_ids=["cand_037"],
        settings=Settings(reframe_analysis_fps=2.0),
        enable_audio_normalization=False, encoder=ENCODER_X264,
    )
    item = manifest.shorts[0]

    assert item.subclip is not None and item.subclip.reason
    assert item.reframe is not None
    assert item.reframe.sampled_frames > 0
    assert item.reframe.dominant_subject_switches >= 0
    assert item.reframe.trajectory.crop_x_range >= 0
    assert item.timings["render_seconds"] > 0
    assert "subclip_refinement_seconds" in item.timings
    assert (run_dir / "shorts" / "short_cand_037_crop_trajectory.json").is_file()


@needs_ffmpeg
def test_ambiguous_advisory_region_is_refused_but_the_render_still_succeeds(tmp_path):
    """cand_037 starts at 10 s, so offsets 20-40 read as both offset and absolute."""
    run_dir = build_run(tmp_path, advisory={"start_offset": 20.0, "end_offset": 40.0})
    manifest = render_shorts_for_run(
        run_dir=run_dir, candidate_ids=["cand_037"],
        settings=Settings(reframe_analysis_fps=2.0),
        enable_audio_normalization=False, encoder=ENCODER_X264,
    )
    item = manifest.shorts[0]

    assert item.advisory_interpretation == "ambiguous"
    assert item.subclip.advisory_used is False
    assert item.validation.valid


@needs_ffmpeg
def test_batch_render_numbers_outputs_in_ranking_order(tmp_path):
    run_dir = build_run(tmp_path)
    manifest = render_shorts_for_run(
        run_dir=run_dir, top=2,
        settings=Settings(reframe_analysis_fps=2.0),
        enable_audio_normalization=False, encoder=ENCODER_X264,
    )

    assert [s.file for s in manifest.shorts] == ["short_01_cand_037.mp4", "short_02_cand_038.mp4"]
    assert [s.candidate_id for s in manifest.shorts] == ["cand_037", "cand_038"]
    assert [s.rank for s in manifest.shorts] == [1, 2]
    for short in manifest.shorts:
        assert (run_dir / "shorts" / short.file).is_file()
        assert (short.width, short.height) == (1080, 1920)
    assert (run_dir / "shorts" / "shorts_manifest.json").is_file()


@needs_ffmpeg
def test_center_crop_fallback_still_produces_a_valid_vertical_short(tmp_path):
    run_dir = build_run(tmp_path)
    manifest = render_shorts_for_run(
        run_dir=run_dir, candidate_ids=["cand_037"],
        settings=Settings(reframe_analysis_fps=2.0),
        enable_smart_reframe=False, enable_audio_normalization=False, encoder=ENCODER_X264,
    )
    item = manifest.shorts[0]

    assert item.reframing_mode == "center"
    video = next(s for s in probe(run_dir / "shorts" / item.file)["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)


@needs_ffmpeg
def test_configured_duration_bounds_are_honoured_end_to_end(tmp_path):
    run_dir = build_run(tmp_path)
    manifest = render_shorts_for_run(
        run_dir=run_dir, candidate_ids=["cand_037"],
        settings=Settings(
            reframe_analysis_fps=2.0,
            subclip_min_duration_sec=10.0,
            subclip_target_min_duration_sec=10.0,
            subclip_target_max_duration_sec=12.0,
            subclip_max_duration_sec=13.0,
        ),
        enable_audio_normalization=False, encoder=ENCODER_X264,
    )
    assert 10.0 <= manifest.shorts[0].duration_sec <= 13.0


@needs_ffmpeg
def test_render_is_deterministic_for_the_same_inputs(tmp_path):
    run_dir = build_run(tmp_path)
    settings = Settings(reframe_analysis_fps=2.0)
    kwargs = dict(candidate_ids=["cand_037"], settings=settings,
                  enable_audio_normalization=False, encoder=ENCODER_X264)

    first = render_shorts_for_run(run_dir=run_dir, **kwargs).shorts[0]
    second = render_shorts_for_run(run_dir=run_dir, **kwargs).shorts[0]

    assert first.short_source_start_sec == second.short_source_start_sec
    assert first.short_source_end_sec == second.short_source_end_sec
    assert first.subclip.score == second.subclip.score


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_render_short_cli(tmp_path, monkeypatch):
    run_dir = build_run(tmp_path)
    monkeypatch.setenv("FREECHER_REFRAME_ANALYSIS_FPS", "2.0")
    monkeypatch.setenv("FREECHER_AUDIO_NORMALIZE_LOUDNESS", "false")
    monkeypatch.setenv("FREECHER_SHORTS_ENCODER", "libx264")
    from freecher_worker.config import get_settings

    get_settings.cache_clear()

    result = CliRunner().invoke(
        app, ["render-short", str(run_dir), "--candidate", "cand_037", "--duration-mode", "auto"]
    )
    get_settings.cache_clear()

    assert result.exit_code == 0, result.output
    assert (run_dir / "shorts" / "short_cand_037.mp4").is_file()
    assert "9:16" in result.output


@needs_ffmpeg
def test_render_shorts_cli_batch(tmp_path, monkeypatch):
    run_dir = build_run(tmp_path)
    monkeypatch.setenv("FREECHER_REFRAME_ANALYSIS_FPS", "2.0")
    monkeypatch.setenv("FREECHER_AUDIO_NORMALIZE_LOUDNESS", "false")
    monkeypatch.setenv("FREECHER_SHORTS_ENCODER", "libx264")
    from freecher_worker.config import get_settings

    get_settings.cache_clear()

    result = CliRunner().invoke(app, ["render-shorts", str(run_dir), "--top", "2"])
    get_settings.cache_clear()

    assert result.exit_code == 0, result.output
    assert (run_dir / "shorts" / "short_01_cand_037.mp4").is_file()
    assert (run_dir / "shorts" / "short_02_cand_038.mp4").is_file()


def test_cli_rejects_unknown_duration_mode(tmp_path):
    result = CliRunner().invoke(
        app, ["render-shorts", str(tmp_path), "--duration-mode", "square"]
    )
    assert result.exit_code == 1
    assert "Unknown duration mode" in result.output


def test_cli_offers_no_aspect_ratio_option():
    """Freecher produces 9:16 only; there must be no way to ask for anything else."""
    output = CliRunner().invoke(app, ["render-shorts", "--help"]).output
    assert "--aspect" not in output
    assert "--preset" not in output


# ---------------------------------------------------------------------------
# Output naming, atomicity and batch resilience
# ---------------------------------------------------------------------------


def test_output_stems_carry_the_candidate_id():
    assert short_stem("cand_010") == "short_cand_010"
    assert short_stem("cand_080", 1) == "short_01_cand_080"
    assert short_stem("cand_010", 12) == "short_12_cand_010"


def test_output_stem_sanitizes_hostile_candidate_ids():
    assert short_stem("../../etc/passwd") == "short_______etc_passwd"
    assert "/" not in short_stem("a/b")


@needs_ffmpeg
def test_single_render_cannot_overwrite_a_batch_result(tmp_path):
    """The reported collision: `render-short` reusing `short_01.mp4` from a previous batch."""
    run_dir = build_run(tmp_path)
    settings = Settings(reframe_analysis_fps=2.0)
    shared = dict(settings=settings, enable_audio_normalization=False, encoder=ENCODER_X264)

    batch = render_shorts_for_run(run_dir=run_dir, top=2, **shared)
    batch_files = {s.file for s in batch.shorts}
    batch_bytes = {
        s.file: (run_dir / "shorts" / s.file).read_bytes()[:2048] for s in batch.shorts
    }

    single = render_shorts_for_run(run_dir=run_dir, candidate_ids=["cand_038"], **shared)

    assert single.shorts[0].file == "short_cand_038.mp4"
    assert single.shorts[0].file not in batch_files
    for name, head in batch_bytes.items():
        assert (run_dir / "shorts" / name).read_bytes()[:2048] == head, f"{name} was clobbered"


@needs_ffmpeg
def test_render_refuses_to_overwrite_another_candidates_output(tmp_path):
    run_dir = build_run(tmp_path)
    context = load_run_context(run_dir)
    timeframe = resolve_timeframe(context, "cand_037")
    output = run_dir / "shorts" / "short_01_cand_037.mp4"

    render_short(
        source_video=context.source_video, timeframe=timeframe, output_path=output,
        transcript=context.transcript, run_dir=run_dir,
        settings=Settings(reframe_analysis_fps=2.0),
        enable_audio_normalization=False, encoder=ENCODER_X264, index=1,
    )
    (output.with_suffix(".json")).write_text(json.dumps({"candidate_id": "cand_037"}))

    other = resolve_timeframe(context, "cand_038")
    with pytest.raises(FileExistsError, match="cand_037"):
        render_short(
            source_video=context.source_video, timeframe=other, output_path=output,
            transcript=context.transcript, run_dir=run_dir,
            settings=Settings(reframe_analysis_fps=2.0),
            enable_audio_normalization=False, encoder=ENCODER_X264, index=1,
        )


@needs_ffmpeg
def test_failed_render_leaves_no_zero_byte_file(tmp_path, monkeypatch):
    """A failing FFmpeg must not leave `short_03.mp4 = 0 bytes` looking like a valid short."""
    run_dir = build_run(tmp_path)
    context = load_run_context(run_dir)
    timeframe = resolve_timeframe(context, "cand_037")
    output = run_dir / "shorts" / "short_cand_037.mp4"

    def always_fails(cmd, timeout=1800.0):
        # Simulate FFmpeg dying after it has already created the output file.
        target = Path(cmd[-1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        return subprocess.CompletedProcess(cmd, 1, "", "Failed to configure input pad")

    monkeypatch.setattr("freecher_worker.shorts.render._run_ffmpeg", always_fails)

    with pytest.raises(RuntimeError, match="FFmpeg failed"):
        render_short(
            source_video=context.source_video, timeframe=timeframe, output_path=output,
            transcript=context.transcript, run_dir=run_dir,
            settings=Settings(reframe_analysis_fps=2.0),
            enable_audio_normalization=False, encoder=ENCODER_X264,
        )

    assert not output.exists()
    assert list(run_dir.glob("shorts/*.tmp.mp4")) == []
    assert [p for p in run_dir.glob("shorts/*.mp4") if p.stat().st_size == 0] == []


@needs_ffmpeg
def test_output_appears_atomically(tmp_path, monkeypatch):
    """The final path must never exist while FFmpeg is still writing."""
    run_dir = build_run(tmp_path)
    context = load_run_context(run_dir)
    timeframe = resolve_timeframe(context, "cand_037")
    output = run_dir / "shorts" / "short_cand_037.mp4"

    real = subprocess.run
    seen = []

    def spy(cmd, **kwargs):
        if cmd and cmd[0] == "ffmpeg" and str(output) not in cmd:
            seen.append(output.exists())
        return real(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    render_short(
        source_video=context.source_video, timeframe=timeframe, output_path=output,
        transcript=context.transcript, run_dir=run_dir,
        settings=Settings(reframe_analysis_fps=2.0),
        enable_audio_normalization=False, encoder=ENCODER_X264,
    )

    assert seen and not any(seen), "final output existed before the render completed"
    assert output.is_file() and output.stat().st_size > 0


@needs_ffmpeg
def test_dynamic_crop_failure_falls_back_to_static_within_the_same_render(tmp_path, monkeypatch):
    """A dynamic-crop FFmpeg failure must yield a static-crop short, not lose the short."""
    run_dir = build_run(tmp_path)
    context = load_run_context(run_dir)
    timeframe = resolve_timeframe(context, "cand_037")
    output = run_dir / "shorts" / "short_cand_037.mp4"

    real = subprocess.run

    def reject_dynamic(cmd, timeout=1800.0):
        graph = cmd[cmd.index("-filter_complex") + 1] if "-filter_complex" in cmd else ""
        if "sendcmd" in graph or "if(" in graph:
            return subprocess.CompletedProcess(
                cmd, 234, "", "[Parsed_crop_0] Failed to configure input pad: Invalid argument"
            )
        return real(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, timeout=timeout, check=False)

    monkeypatch.setattr("freecher_worker.shorts.render._run_ffmpeg", reject_dynamic)

    metadata = render_short(
        source_video=context.source_video, timeframe=timeframe, output_path=output,
        transcript=context.transcript, run_dir=run_dir,
        settings=Settings(reframe_analysis_fps=5.0),
        enable_audio_normalization=False, encoder=ENCODER_X264,
    )

    assert metadata.render_fallback_used is True
    assert metadata.status == "fallback"
    assert "Invalid argument" in (metadata.render_failure_reason or "")
    assert metadata.crop_driver == "static"
    video = next(s for s in probe(output)["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)


@needs_ffmpeg
def test_one_failing_candidate_does_not_abort_the_batch(tmp_path, monkeypatch):
    """The reported abort: `--top 5` stopping at the first bad candidate."""
    run_dir = build_run(tmp_path)
    real_render = render_short_module.render_short
    calls = {"n": 0}

    def fail_first(*args, **kwargs):
        if kwargs.get("timeframe").candidate_id == "cand_037":
            calls["n"] += 1
            raise RuntimeError("simulated hard failure for cand_037")
        return real_render(*args, **kwargs)

    monkeypatch.setattr(render_short_module, "render_short", fail_first)

    manifest = render_shorts_for_run(
        run_dir=run_dir, top=2, settings=Settings(reframe_analysis_fps=2.0),
        enable_audio_normalization=False, encoder=ENCODER_X264,
    )

    assert calls["n"] == 2, "the failing candidate should also have been retried statically"
    assert manifest.requested == 2
    assert manifest.failure_count == 1
    assert manifest.success_count == 1
    statuses = {r.candidate_id: r.status for r in manifest.results}
    assert statuses["cand_037"] == "failed"
    assert statuses["cand_038"] == "success"
    assert [s.candidate_id for s in manifest.shorts] == ["cand_038"]
    assert (run_dir / "shorts" / "short_02_cand_038.mp4").is_file()
    assert not (run_dir / "shorts" / "short_01_cand_037.mp4").exists()


@needs_ffmpeg
def test_manifest_reports_counts_and_per_candidate_status(tmp_path):
    run_dir = build_run(tmp_path)
    manifest = render_shorts_for_run(
        run_dir=run_dir, top=2, settings=Settings(reframe_analysis_fps=2.0),
        enable_audio_normalization=False, encoder=ENCODER_X264,
    )

    assert manifest.success_count + manifest.fallback_count + manifest.failure_count == manifest.requested
    assert len(manifest.results) == 2
    for result in manifest.results:
        assert result.status in {"success", "fallback", "failed"}
        assert result.candidate_id
        assert result.file and result.candidate_id in result.file

    payload = json.loads((run_dir / "shorts" / "shorts_manifest.json").read_text())
    for key in ("success_count", "fallback_count", "failure_count", "results"):
        assert key in payload


@needs_ffmpeg
def test_unknown_candidate_is_recorded_as_failed_not_raised(tmp_path):
    run_dir = build_run(tmp_path)
    manifest = render_shorts_for_run(
        run_dir=run_dir, candidate_ids=["cand_037", "cand_999"],
        settings=Settings(reframe_analysis_fps=2.0),
        enable_audio_normalization=False, encoder=ENCODER_X264,
    )

    assert manifest.failure_count == 1
    assert manifest.success_count == 1
    failed = next(r for r in manifest.results if r.candidate_id == "cand_999")
    assert failed.status == "failed"
    assert "cand_999" in (failed.reason or "")


@needs_ffmpeg
def test_metadata_records_the_crop_driver_and_trajectory_report(tmp_path):
    run_dir = build_run(tmp_path)
    manifest = render_shorts_for_run(
        run_dir=run_dir, candidate_ids=["cand_037"],
        settings=Settings(reframe_analysis_fps=5.0),
        enable_audio_normalization=False, encoder=ENCODER_X264,
    )
    item = manifest.shorts[0]

    assert item.crop_driver in {"sendcmd", "expression", "static"}
    assert item.crop_keyframes >= 1
    assert item.trajectory_report is not None
    report = item.trajectory_report
    assert report.valid
    assert report.crop_width <= report.source_width
    assert report.crop_height <= report.source_height
    assert 0 <= report.min_x <= report.max_x <= report.max_x_allowed
    assert item.reframe.detection_coverage <= 1.0
    assert item.reframe.tracking_coverage <= 1.0
    assert item.reframe.tracking_fallback_rate <= 1.0
