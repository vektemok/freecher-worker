"""Adaptive layout rendering: filtergraph construction and real 1080x1920 output.

The planner is unit-tested separately; here the visual analysis is scripted so that a chosen
layout is guaranteed, and FFmpeg is then asked to actually produce the file.
"""

import json
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from freecher_worker.cli import app
from freecher_worker.config import Settings
from freecher_worker.crop.models import CropPoint, CropTrajectory
import freecher_worker.shorts.render as render_module
from freecher_worker.shorts.layout import (
    LAYOUT_DUAL,
    LAYOUT_FULL,
    LAYOUT_MODE_ADAPTIVE,
    LAYOUT_MODE_FULL_FRAME,
    LAYOUT_MODE_SINGLE,
    LAYOUT_SINGLE,
    LayoutConfig,
    LayoutPlan,
    LayoutSegment,
    ViewportPlan,
    VIEWPORT_DUAL_BOTTOM,
    VIEWPORT_DUAL_TOP,
    VIEWPORT_SINGLE,
    build_dual_viewports,
    build_layout_plan,
)
from freecher_worker.shorts.layout_render import (
    LayoutRenderError,
    build_adaptive_render_plan,
    build_adaptive_video_filter,
    supports_named_filter_instances,
)
from freecher_worker.shorts.reframe import (
    REFRAME_MODE_SMART,
    ReframeDiagnostics,
    ReframePlan,
    calculate_vertical_crop,
)
from freecher_worker.shorts.render import render_shorts_for_run

from test_shorts_layout import FPS, SOURCE_H, SOURCE_W, face, make_frames
from test_shorts_pipeline import build_run

FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg is not installed")
needs_named_instances = pytest.mark.skipif(
    not FFMPEG or not supports_named_filter_instances(),
    reason="this FFmpeg build has no named filter instances",
)

CROP_W, CROP_H = calculate_vertical_crop(SOURCE_W, SOURCE_H)
OUT_W, OUT_H = 1080, 1920


def scripted_reframe_plan(script, duration: float) -> ReframePlan:
    """A reframe plan whose visual analysis is dictated rather than detected."""
    frames = make_frames(script, duration=duration)
    points = [
        CropPoint(
            time=frame.time,
            center_x=SOURCE_W / 2.0,
            center_y=SOURCE_H / 2.0,
            crop_x=((SOURCE_W - CROP_W) // 4) * 2 + (index % 3) * 2,
            crop_y=0,
            crop_w=CROP_W,
            crop_h=CROP_H,
            subject_type="face",
        )
        for index, frame in enumerate(frames)
    ]
    return ReframePlan(
        mode=REFRAME_MODE_SMART,
        trajectory=CropTrajectory(
            source_width=SOURCE_W,
            source_height=SOURCE_H,
            crop_w=CROP_W,
            crop_h=CROP_H,
            points=points,
        ),
        diagnostics=ReframeDiagnostics(
            detector="scripted", analysis_fps=FPS, sampled_frames=len(frames)
        ),
        frames=frames,
    )


def use_scripted_analysis(monkeypatch, script) -> None:
    """Replace detection with a script, keeping every other stage real."""

    def fake_plan(**kwargs):
        return scripted_reframe_plan(script, kwargs["duration_sec"])

    monkeypatch.setattr(render_module, "build_reframe_plan", fake_plan)


ONE_PERSON = lambda t: [face(1, 500)]
TWO_DISTANT = lambda t: [face(1, 200), face(2, 1080)]
THREE_PEOPLE = lambda t: [face(1, 220), face(2, 640), face(3, 1060)]
#: Two people who both drift, slowly enough to stay too far apart for one crop. Both viewports
#: are then genuinely dynamic, which is what makes the independent command streams observable.
MOVING_DISTANT = lambda t: [face(1, 420 + 5.0 * min(t, 12.0)), face(2, 850 - 5.0 * min(t, 12.0))]


def plan_from(script, duration: float, **kwargs) -> LayoutPlan:
    return build_layout_plan(
        frames=make_frames(script, duration=duration),
        duration=duration,
        source_width=SOURCE_W,
        source_height=SOURCE_H,
        crop_w=CROP_W,
        crop_h=CROP_H,
        mode=LAYOUT_MODE_ADAPTIVE,
        **kwargs,
    )


def single_trajectory() -> CropTrajectory:
    return scripted_reframe_plan(ONE_PERSON, 12.0).trajectory


# ---------------------------------------------------------------------------
# Filtergraph construction
# ---------------------------------------------------------------------------


def test_a_single_layout_plan_needs_no_overlays():
    plan = plan_from(ONE_PERSON, 12.0)
    render = build_adaptive_render_plan(plan, single_trajectory(), OUT_W, OUT_H)
    assert render.base_layout == LAYOUT_SINGLE
    assert render.overlay_layouts == []

    graph = build_adaptive_video_filter(render, {VIEWPORT_SINGLE: Path("/tmp/single.txt")})
    assert "overlay" not in graph
    assert graph.endswith("format=yuv420p[v]")
    assert f"crop@{VIEWPORT_SINGLE}=" in graph


def test_each_stacked_viewport_gets_its_own_command_stream():
    """Two viewports panning independently cannot share one sendcmd target."""
    frames = make_frames(MOVING_DISTANT, duration=12.0)
    plan = plan_from(MOVING_DISTANT, 12.0)
    viewports = build_dual_viewports(plan, frames, SOURCE_W, SOURCE_H, OUT_W, OUT_H)
    assert viewports is not None

    render = build_adaptive_render_plan(
        plan, single_trajectory(), OUT_W, OUT_H, dual_viewports=viewports
    )
    graph = build_adaptive_video_filter(
        render,
        {
            VIEWPORT_DUAL_TOP: Path("/tmp/top.txt"),
            VIEWPORT_DUAL_BOTTOM: Path("/tmp/bottom.txt"),
        },
    )
    assert f"crop@{VIEWPORT_DUAL_TOP}=" in graph
    assert f"crop@{VIEWPORT_DUAL_BOTTOM}=" in graph
    assert "vstack=inputs=2" in graph
    assert graph.count("sendcmd") == 2
    assert "/tmp/top.txt" in graph and "/tmp/bottom.txt" in graph


def test_stacked_viewports_follow_two_different_people():
    frames = make_frames(TWO_DISTANT, duration=12.0)
    plan = plan_from(TWO_DISTANT, 12.0)
    top, bottom = build_dual_viewports(plan, frames, SOURCE_W, SOURCE_H, OUT_W, OUT_H)
    assert top.track_ids == [1]
    assert bottom.track_ids == [2]
    # The left subject sits on top, so the two windows must be horizontally separated.
    assert max(p.crop_x for p in top.trajectory.points) < min(
        p.crop_x for p in bottom.trajectory.points
    )


def test_stacked_viewports_are_exactly_half_the_output_each():
    frames = make_frames(TWO_DISTANT, duration=12.0)
    plan = plan_from(TWO_DISTANT, 12.0)
    top, bottom = build_dual_viewports(plan, frames, SOURCE_W, SOURCE_H, OUT_W, OUT_H)
    for viewport in (top, bottom):
        ratio = viewport.trajectory.crop_w / viewport.trajectory.crop_h
        assert ratio == pytest.approx(OUT_W / (OUT_H / 2), rel=0.01)


def test_a_full_frame_layout_never_crops_the_source():
    plan = plan_from(THREE_PEOPLE, 12.0)
    render = build_adaptive_render_plan(plan, single_trajectory(), OUT_W, OUT_H)
    graph = build_adaptive_video_filter(render, {})
    assert "force_original_aspect_ratio=decrease" in graph
    assert "gblur" in graph


def test_the_full_frame_background_can_be_a_flat_colour():
    plan = plan_from(THREE_PEOPLE, 12.0)
    render = build_adaptive_render_plan(
        plan,
        single_trajectory(),
        OUT_W,
        OUT_H,
        config=LayoutConfig(full_frame_blur_background=False),
    )
    graph = build_adaptive_video_filter(render, {})
    assert "gblur" not in graph
    assert "pad=1080:1920" in graph


def test_only_one_layout_is_enabled_at_a_segment_boundary():
    """`between()` is inclusive at both ends; two layouts must never claim the same frame."""
    plan = LayoutPlan(
        duration=12.0,
        segments=[
            LayoutSegment(start=0.0, end=6.0, layout=LAYOUT_SINGLE, subject_ids=[1]),
            LayoutSegment(start=6.0, end=12.0, layout=LAYOUT_FULL),
        ],
    )
    render = build_adaptive_render_plan(plan, single_trajectory(), OUT_W, OUT_H)
    graph = build_adaptive_video_filter(render, {VIEWPORT_SINGLE: Path("/tmp/s.txt")})
    assert "gte(t,0.0000)*lt(t,6.0000)" in graph or "gte(t,6.0000)*lt(t,12.0000)" in graph
    assert "between(" not in graph


def test_the_longest_layout_carries_the_graph():
    plan = LayoutPlan(
        duration=12.0,
        segments=[
            LayoutSegment(start=0.0, end=2.0, layout=LAYOUT_SINGLE, subject_ids=[1]),
            LayoutSegment(start=2.0, end=12.0, layout=LAYOUT_FULL),
        ],
    )
    render = build_adaptive_render_plan(plan, single_trajectory(), OUT_W, OUT_H)
    assert render.base_layout == LAYOUT_FULL
    assert render.overlay_layouts == [LAYOUT_SINGLE]


def test_a_stack_without_viewports_degrades_to_the_full_frame():
    plan = LayoutPlan(
        duration=12.0,
        segments=[LayoutSegment(start=0.0, end=12.0, layout=LAYOUT_DUAL, subject_ids=[1, 2])],
    )
    render = build_adaptive_render_plan(plan, single_trajectory(), OUT_W, OUT_H)
    assert [s.layout for s in render.segments] == [LAYOUT_FULL]


def test_a_missing_command_script_is_refused_rather_than_rendered_wrong():
    plan = plan_from(ONE_PERSON, 12.0)
    render = build_adaptive_render_plan(plan, single_trajectory(), OUT_W, OUT_H)
    with pytest.raises(LayoutRenderError):
        build_adaptive_video_filter(render, {})


# ---------------------------------------------------------------------------
# End-to-end renders
# ---------------------------------------------------------------------------


def probe_size(path: Path):
    capture = cv2.VideoCapture(str(path))
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    return size


def read_frame(path: Path, at_seconds: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_MSEC, at_seconds * 1000.0)
    ok, frame = capture.read()
    capture.release()
    assert ok and frame is not None, f"could not read {path} at {at_seconds}s"
    return frame


def render(run_dir: Path, layout_mode: str, **settings_kwargs):
    return render_shorts_for_run(
        run_dir=run_dir,
        candidate_ids=["cand_037"],
        settings=Settings(reframe_analysis_fps=2.0, **settings_kwargs),
        enable_audio_normalization=False,
        encoder="libx264",
        layout_mode=layout_mode,
    )


@needs_named_instances
def test_a_stacked_layout_renders_two_different_viewports(tmp_path, monkeypatch):
    run_dir = build_run(tmp_path)
    use_scripted_analysis(monkeypatch, MOVING_DISTANT)
    manifest = render(run_dir, LAYOUT_MODE_ADAPTIVE)

    item = manifest.shorts[0]
    assert item.layout is not None
    assert [s.layout for s in item.layout.segments] == [LAYOUT_DUAL]
    output = run_dir / "shorts" / item.file
    assert probe_size(output) == (OUT_W, OUT_H)

    frame = read_frame(output, item.duration_sec / 2.0)
    top, bottom = frame[: OUT_H // 2], frame[OUT_H // 2 :]
    assert float(np.mean(np.abs(top.astype(float) - bottom.astype(float)))) > 5.0, (
        "the two halves are identical, so both viewports followed the same window"
    )
    for name in (VIEWPORT_DUAL_TOP, VIEWPORT_DUAL_BOTTOM):
        assert (run_dir / "shorts" / f"{output.stem}_crop_commands_{name}.txt").is_file()


@needs_named_instances
def test_a_full_frame_layout_keeps_the_whole_source_frame(tmp_path, monkeypatch):
    """A 16:9 source fitted into 9:16 must letterbox, not crop."""
    run_dir = build_run(tmp_path)
    use_scripted_analysis(monkeypatch, THREE_PEOPLE)
    manifest = render(run_dir, LAYOUT_MODE_ADAPTIVE, layout_full_frame_blur=False)

    item = manifest.shorts[0]
    assert [s.layout for s in item.layout.segments] == [LAYOUT_FULL]
    output = run_dir / "shorts" / item.file
    assert probe_size(output) == (OUT_W, OUT_H)

    frame = read_frame(output, item.duration_sec / 2.0)
    fitted_height = int(OUT_W * SOURCE_H / SOURCE_W)
    band = (OUT_H - fitted_height) // 2
    assert frame[: band - 20].max() < 24, "the reserved band should be the flat background"
    assert frame[OUT_H - band + 20 :].max() < 24
    assert frame[OUT_H // 2].std() > 10, "the fitted frame should carry the picture"


@needs_named_instances
def test_full_frame_mode_renders_without_any_analysis(tmp_path):
    run_dir = build_run(tmp_path)
    manifest = render(run_dir, LAYOUT_MODE_FULL_FRAME)
    item = manifest.shorts[0]
    assert [s.layout for s in item.layout.segments] == [LAYOUT_FULL]
    assert probe_size(run_dir / "shorts" / item.file) == (OUT_W, OUT_H)


@needs_ffmpeg
def test_the_layout_never_moves_the_published_seconds(tmp_path, monkeypatch):
    """Layout is presentation. The same candidate must publish the same source range."""
    baseline_dir = build_run(tmp_path / "a")
    baseline = render(baseline_dir, LAYOUT_MODE_SINGLE).shorts[0]

    adaptive_dir = build_run(tmp_path / "b")
    use_scripted_analysis(monkeypatch, TWO_DISTANT)
    adaptive = render(adaptive_dir, LAYOUT_MODE_ADAPTIVE).shorts[0]

    assert adaptive.short_source_start_sec == baseline.short_source_start_sec
    assert adaptive.short_source_end_sec == baseline.short_source_end_sec
    assert adaptive.duration_sec == baseline.duration_sec
    assert adaptive.candidate_id == baseline.candidate_id


@needs_ffmpeg
def test_the_default_render_is_unchanged_and_records_its_layout(tmp_path):
    run_dir = build_run(tmp_path)
    manifest = render(run_dir, LAYOUT_MODE_SINGLE)
    item = manifest.shorts[0]

    assert manifest.layout_mode == LAYOUT_MODE_SINGLE
    assert item.layout_mode_requested == LAYOUT_MODE_SINGLE
    assert [s.layout for s in item.layout.segments] == [LAYOUT_SINGLE]
    assert probe_size(run_dir / "shorts" / item.file) == (OUT_W, OUT_H)


@needs_ffmpeg
def test_the_manifest_carries_the_layout_plan(tmp_path, monkeypatch):
    run_dir = build_run(tmp_path)
    use_scripted_analysis(monkeypatch, TWO_DISTANT)
    render(run_dir, LAYOUT_MODE_ADAPTIVE)

    payload = json.loads((run_dir / "shorts" / "shorts_manifest.json").read_text())
    assert payload["layout_mode"] == LAYOUT_MODE_ADAPTIVE
    layout = payload["shorts"][0]["layout"]
    assert layout["version"] == "adaptive_layout_v1"
    assert layout["segments"]
    assert "duration_by_mode" in layout
    assert layout["persistent_track_count"] == 2
    assert layout["dominant_track_id"] is not None
    for segment in layout["segments"]:
        assert segment["reason"]


@needs_ffmpeg
def test_an_unrenderable_layout_falls_back_instead_of_losing_the_short(tmp_path, monkeypatch):
    run_dir = build_run(tmp_path)
    use_scripted_analysis(monkeypatch, TWO_DISTANT)
    monkeypatch.setattr(
        render_module,
        "build_adaptive_video_filter",
        lambda *a, **k: "[0:v]this_filter_does_not_exist[v];[0:a]anull[a]",
    )

    manifest = render(run_dir, LAYOUT_MODE_ADAPTIVE)
    item = manifest.shorts[0]
    assert item.layout_fallback_reason
    assert [s.layout for s in item.layout.segments] == [LAYOUT_SINGLE]
    assert probe_size(run_dir / "shorts" / item.file) == (OUT_W, OUT_H)


def test_the_cli_rejects_an_unknown_layout_mode(tmp_path):
    result = CliRunner().invoke(app, ["render-shorts", str(tmp_path), "--layout-mode", "grid"])
    assert result.exit_code == 1
    assert "Unknown layout mode" in result.output


def test_the_cli_offers_the_three_documented_layout_modes():
    output = CliRunner().invoke(app, ["render-shorts", "--help"]).output
    assert "--layout-mode" in output
    for mode in ("single", "adaptive", "full-frame"):
        assert mode in output
