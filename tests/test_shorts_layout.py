"""Adaptive layout planning: subject significance, layout rules, hysteresis and safety.

Every test drives the planner from synthesized ``FrameObservation``s rather than from pixels, so
the decision rules are exercised exactly and reproducibly.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from freecher_worker.shorts.layout import (
    LAYOUT_DUAL,
    LAYOUT_FULL,
    LAYOUT_MODE_ADAPTIVE,
    LAYOUT_MODE_FULL_FRAME,
    LAYOUT_MODE_SINGLE,
    LAYOUT_SINGLE,
    LAYOUT_VERSION,
    LayoutConfig,
    SemanticContext,
    build_layout_plan,
    calculate_aspect_crop,
    can_fit_subjects_in_single_vertical_crop,
    summarize_tracks,
)
from freecher_worker.shorts.reframe import (
    FrameObservation,
    TrackObservation,
    calculate_vertical_crop,
)

SOURCE_W, SOURCE_H = 1280, 720
FPS = 5.0
CROP_W, CROP_H = calculate_vertical_crop(SOURCE_W, SOURCE_H)

#: A face big enough to clear ``significant_min_area_ratio`` at this source size.
FACE = 80
FACE_Y = 300


def face(track_id: int, center_x: float, size: int = FACE, center_y: float = FACE_Y,
         confidence: float = 0.9, hits: int = 5) -> Tuple[int, float, float, int, float, int]:
    return (track_id, center_x, center_y, size, confidence, hits)


def make_frames(
    script,
    duration: float,
    fps: float = FPS,
    scene_cuts: Sequence[float] = (),
) -> List[FrameObservation]:
    """Build an analysis timeline. ``script(t)`` returns the subjects visible at time ``t``."""
    frames: List[FrameObservation] = []
    hits: Dict[int, int] = {}
    steps = int(round(duration * fps))
    for index in range(steps):
        t = round(index / fps, 3)
        tracks: List[TrackObservation] = []
        for track_id, cx, cy, size, confidence, _ in script(t):
            hits[track_id] = hits.get(track_id, 0) + 1
            tracks.append(
                TrackObservation(
                    track_id=track_id,
                    box=(int(cx - size / 2), int(cy - size / 2), size, size),
                    subject_type="face",
                    confidence=confidence,
                    speaking_score=0.5,
                    hits=hits[track_id],
                )
            )
        frames.append(
            FrameObservation(
                time=t,
                scene_cut=any(abs(t - cut) < 1e-6 for cut in scene_cuts),
                active_track_id=tracks[0].track_id if tracks else None,
                tracks=tracks,
            )
        )
    return frames


def plan_for(frames: List[FrameObservation], duration: float, **kwargs):
    config = kwargs.pop("config", None) or LayoutConfig()
    return build_layout_plan(
        frames=frames,
        duration=duration,
        source_width=SOURCE_W,
        source_height=SOURCE_H,
        crop_w=CROP_W,
        crop_h=CROP_H,
        mode=kwargs.pop("mode", LAYOUT_MODE_ADAPTIVE),
        config=config,
        **kwargs,
    )


def layouts_of(plan) -> List[str]:
    return [segment.layout for segment in plan.segments]


# ---------------------------------------------------------------------------
# Geometry and feasibility
# ---------------------------------------------------------------------------


def test_dual_viewport_crop_matches_a_stacked_half():
    """Each half of a 1080x1920 stack is 1080x960, so its source window must be 9:8."""
    crop_w, crop_h = calculate_aspect_crop(SOURCE_W, SOURCE_H, 1080, 960)
    assert crop_w % 2 == 0 and crop_h % 2 == 0
    assert crop_w <= SOURCE_W and crop_h <= SOURCE_H
    assert crop_w / crop_h == pytest.approx(1080 / 960, rel=0.01)


def test_two_close_subjects_fit_one_vertical_crop():
    boxes = [(520, 260, FACE, FACE), (680, 260, FACE, FACE)]
    fits, reason = can_fit_subjects_in_single_vertical_crop(
        boxes, CROP_W, CROP_H, SOURCE_W, SOURCE_H
    )
    assert fits, reason


def test_two_distant_subjects_do_not_fit_one_vertical_crop():
    boxes = [(160, 260, FACE, FACE), (1040, 260, FACE, FACE)]
    fits, reason = can_fit_subjects_in_single_vertical_crop(
        boxes, CROP_W, CROP_H, SOURCE_W, SOURCE_H
    )
    assert not fits
    assert "wider" in reason


def test_a_subject_in_the_caption_band_does_not_count_as_fitting():
    """The bottom of the frame belongs to captions and platform UI, not to a face."""
    boxes = [(600, SOURCE_H - 90, 60, 60)]
    fits, reason = can_fit_subjects_in_single_vertical_crop(
        boxes, CROP_W, CROP_H, SOURCE_W, SOURCE_H, config=LayoutConfig(safe_bottom_ratio=0.30)
    )
    assert not fits
    assert "caption" in reason


def test_a_tiny_subject_is_not_worth_cropping_to():
    boxes = [(600, 300, 8, 8)]
    fits, reason = can_fit_subjects_in_single_vertical_crop(
        boxes, CROP_W, CROP_H, SOURCE_W, SOURCE_H
    )
    assert not fits
    assert "small" in reason


# ---------------------------------------------------------------------------
# Layout rules
# ---------------------------------------------------------------------------


def test_one_stable_person_gets_a_single_subject_layout():
    frames = make_frames(lambda t: [face(1, 500)], duration=15.0)
    plan = plan_for(frames, 15.0)
    assert layouts_of(plan) == [LAYOUT_SINGLE]
    assert plan.segments[0].subjects == ["track_1"]


def test_two_neighbouring_people_stay_in_one_crop():
    frames = make_frames(lambda t: [face(1, 560), face(2, 720)], duration=15.0)
    plan = plan_for(frames, 15.0)
    assert layouts_of(plan) == [LAYOUT_SINGLE]


def test_two_distant_people_are_stacked():
    frames = make_frames(lambda t: [face(1, 200), face(2, 1080)], duration=15.0)
    plan = plan_for(frames, 15.0)
    assert layouts_of(plan) == [LAYOUT_DUAL]
    assert plan.segments[0].subjects == ["track_1", "track_2"]


def test_three_persistent_people_get_the_full_frame():
    frames = make_frames(
        lambda t: [face(1, 220), face(2, 640), face(3, 1060)], duration=15.0
    )
    plan = plan_for(frames, 15.0)
    assert layouts_of(plan) == [LAYOUT_FULL]
    assert plan.significant_track_count == 3


def test_low_detection_confidence_falls_back_to_the_full_frame():
    frames = make_frames(lambda t: [face(1, 500, confidence=0.2)], duration=15.0)
    plan = plan_for(frames, 15.0)
    assert layouts_of(plan) == [LAYOUT_FULL]


def test_sparse_detection_falls_back_to_the_full_frame():
    """A clip the detector barely sees is exactly the clip a crop would ruin."""
    frames = make_frames(
        lambda t: [face(1, 500)] if int(t * FPS) % 5 == 0 else [], duration=15.0
    )
    plan = plan_for(frames, 15.0)
    assert layouts_of(plan) == [LAYOUT_FULL]


def test_a_one_frame_face_cannot_create_a_second_subject():
    def script(t):
        subjects = [face(1, 500)]
        if abs(t - 6.0) < 1e-6:
            subjects.append(face(2, 1100))
        return subjects

    frames = make_frames(script, duration=15.0)
    plan = plan_for(frames, 15.0)
    assert layouts_of(plan) == [LAYOUT_SINGLE]
    assert [t.label for t in plan.tracks if t.significant] == ["track_1"]


def test_semantic_context_can_demand_the_full_frame():
    frames = make_frames(lambda t: [face(1, 500)], duration=15.0)
    plan = plan_for(frames, 15.0, semantic=SemanticContext(prefer_full_frame=True))
    assert layouts_of(plan) == [LAYOUT_FULL]


# ---------------------------------------------------------------------------
# Temporal stability
# ---------------------------------------------------------------------------


def test_a_brief_disappearance_does_not_change_the_layout():
    """One second of lost detection is a blink, not an edit."""
    def script(t):
        subjects = [face(1, 200)]
        if not (8.0 <= t < 9.0):
            subjects.append(face(2, 1080))
        return subjects

    frames = make_frames(script, duration=16.0)
    plan = plan_for(frames, 16.0)
    assert layouts_of(plan) == [LAYOUT_DUAL]


def test_a_long_disappearance_does_change_the_layout():
    def script(t):
        subjects = [face(1, 200)]
        if t < 8.0:
            subjects.append(face(2, 1080))
        return subjects

    frames = make_frames(script, duration=16.0)
    plan = plan_for(frames, 16.0)
    assert layouts_of(plan) == [LAYOUT_DUAL, LAYOUT_SINGLE]
    # The switch waits for sustained evidence rather than firing on the first quiet window.
    assert plan.segments[1].start >= 9.0


def test_a_scene_cut_may_switch_the_layout_at_once():
    def script(t):
        if t < 4.0:
            return [face(1, 500)]
        return [face(2, 220), face(3, 640), face(4, 1060)]

    with_cut = plan_for(make_frames(script, duration=12.0, scene_cuts=(4.0,)), 12.0)
    without_cut = plan_for(make_frames(script, duration=12.0), 12.0)

    cut_start = next(s.start for s in with_cut.segments if s.layout == LAYOUT_FULL)
    slow_start = next(s.start for s in without_cut.segments if s.layout == LAYOUT_FULL)
    assert cut_start <= 4.5
    assert slow_start > cut_start


def test_alternating_evidence_does_not_produce_a_strobing_layout():
    """The pathological case: evidence that flips every window must not flip the layout."""
    def script(t):
        subjects = [face(1, 200)]
        if int(t * FPS) % 2 == 0:
            subjects.append(face(2, 1080))
        return subjects

    plan = plan_for(make_frames(script, duration=30.0), 30.0)
    assert plan.switch_count <= 3
    for segment in plan.segments:
        assert segment.duration >= LayoutConfig().min_layout_duration_sec - 1e-6


def test_a_thirty_second_short_stays_within_a_few_layout_changes():
    def script(t):
        if t < 10.0:
            return [face(1, 500)]
        if t < 20.0:
            return [face(1, 200), face(2, 1080)]
        return [face(1, 220), face(2, 640), face(3, 1060)]

    plan = plan_for(make_frames(script, duration=30.0), 30.0)
    assert set(layouts_of(plan)) == {LAYOUT_SINGLE, LAYOUT_DUAL, LAYOUT_FULL}
    assert plan.switch_count <= 3


# ---------------------------------------------------------------------------
# Plan integrity
# ---------------------------------------------------------------------------


def test_segments_tile_the_clip_without_gaps_or_overlaps():
    def script(t):
        if t < 10.0:
            return [face(1, 500)]
        return [face(1, 200), face(2, 1080)]

    plan = plan_for(make_frames(script, duration=24.0), 24.0)
    assert plan.segments[0].start == 0.0
    assert plan.segments[-1].end == pytest.approx(24.0)
    for previous, following in zip(plan.segments, plan.segments[1:]):
        assert following.start == pytest.approx(previous.end)
    for segment in plan.segments:
        assert 0.0 <= segment.start < segment.end <= 24.0


def test_a_dual_stack_never_shows_the_same_person_twice():
    def script(t):
        if t < 12.0:
            return [face(1, 200), face(2, 1080)]
        return [face(3, 180), face(4, 1100)]

    plan = plan_for(make_frames(script, duration=24.0), 24.0)
    dual = plan.segments_for(LAYOUT_DUAL)
    assert dual, "expected at least one stacked segment"
    for segment in dual:
        assert len(segment.subject_ids) == 2
        assert len(set(segment.subject_ids)) == 2


def test_planning_is_deterministic():
    def script(t):
        if t < 8.0:
            return [face(1, 500)]
        return [face(1, 200), face(2, 1080)]

    frames = make_frames(script, duration=20.0)
    first = plan_for(frames, 20.0)
    second = plan_for(frames, 20.0)
    assert first.model_dump() == second.model_dump()


def test_the_plan_reports_what_it_decided_and_why():
    frames = make_frames(lambda t: [face(1, 500)], duration=15.0)
    plan = plan_for(frames, 15.0)
    assert plan.version == LAYOUT_VERSION
    assert plan.duration_by_mode[LAYOUT_SINGLE] == pytest.approx(15.0)
    assert plan.track_count == 1
    assert plan.persistent_track_count == 1
    assert plan.dominant_track_id == 1
    assert plan.mean_tracking_confidence > 0.5
    assert plan.analysis_windows > 0
    assert plan.segments[0].reason


# ---------------------------------------------------------------------------
# Modes and degenerate input
# ---------------------------------------------------------------------------


def test_single_mode_never_plans_anything_else():
    def script(t):
        return [face(1, 220), face(2, 640), face(3, 1060)]

    plan = plan_for(make_frames(script, duration=15.0), 15.0, mode=LAYOUT_MODE_SINGLE)
    assert layouts_of(plan) == [LAYOUT_SINGLE]


def test_full_frame_mode_never_plans_anything_else():
    frames = make_frames(lambda t: [face(1, 500)], duration=15.0)
    plan = plan_for(frames, 15.0, mode=LAYOUT_MODE_FULL_FRAME)
    assert layouts_of(plan) == [LAYOUT_FULL]


def test_no_analysis_frames_prefers_context_over_a_guessed_crop():
    plan = plan_for([], 15.0)
    assert layouts_of(plan) == [LAYOUT_FULL]
    assert plan.fallback_reason


def test_track_summaries_separate_real_subjects_from_noise():
    def script(t):
        subjects = [face(1, 500)]
        if t < 0.4:
            subjects.append(face(9, 1100))
        return subjects

    summaries = summarize_tracks(make_frames(script, duration=15.0), SOURCE_W, SOURCE_H)
    by_id = {s.track_id: s for s in summaries}
    assert by_id[1].persistent and by_id[1].significant
    assert not by_id[9].persistent
    assert by_id[1].visibility_ratio == pytest.approx(1.0)
    assert by_id[1].scene_ids == [0]
