"""Subject identity continuity: one physical person must keep one track id.

These are regressions for the benchmark_02 failure where a clip with 72% detection coverage
produced 37 tracks and zero persistent subjects, which the layout planner could only read as
"there is nobody here".
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from freecher_worker.crop.models import DetectedSubject
from freecher_worker.shorts.layout import LayoutConfig, summarize_tracks
from freecher_worker.shorts.reframe import (
    ReframeConfig,
    SubjectDetector,
    build_reframe_plan,
)

from test_shorts_reframe import SOURCE_H, SOURCE_W, write_video

FPS = 5.0
FACE = 90


@pytest.fixture(scope="module")
def clip() -> str:
    """One synthetic clip reused by every scenario; detections come from the script, not pixels."""
    return write_video(Path(tempfile.mkdtemp()) / "clip.mp4", seconds=32.0)


class TimelineDetector(SubjectDetector):
    """Detector driven by ``script(sample_index) -> [(cx, cy, size), ...]`` in source pixels."""

    name = "timeline"

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def detect(self, frame_bgr, timestamp: float) -> List[DetectedSubject]:
        index = self.calls
        self.calls += 1
        scale = frame_bgr.shape[1] / float(SOURCE_W)
        subjects = []
        for cx, cy, size in self.script(index):
            w = h = max(2, int(size * scale))
            x, y = int(cx * scale - w / 2), int(cy * scale - h / 2)
            subjects.append(
                DetectedSubject(
                    box=(x, y, w, h),
                    confidence=0.9,
                    subject_type="face",
                    area=float(w * h),
                    center_x=x + w / 2.0,
                    center_y=y + h / 2.0,
                )
            )
        return subjects


def analyse(clip: str, script, duration: float = 20.0, **config):
    plan = build_reframe_plan(
        video_path=clip,
        source_start_sec=0.0,
        duration_sec=duration,
        detector=TimelineDetector(script),
        config=ReframeConfig(analysis_fps=FPS, **config),
        source_width=SOURCE_W,
        source_height=SOURCE_H,
    )
    return plan


def identities(plan) -> Dict[int, List[float]]:
    """Track ids the planner was actually told about, and when they were seen."""
    seen: Dict[int, List[float]] = {}
    for frame in plan.frames:
        for track in frame.tracks:
            seen.setdefault(track.track_id, []).append(frame.time)
    return seen


# ---------------------------------------------------------------------------
# The reported failure
# ---------------------------------------------------------------------------


def test_one_person_with_a_flaky_detector_keeps_one_identity(clip):
    """cand_080: 14 s, ~72% coverage. One person must not become dozens of tracks."""
    present = [(index % 7) not in (5, 6) for index in range(200)]

    def script(index):
        if not present[index]:
            return []
        return [(300 + 8.0 * (index / FPS), 300, FACE)]

    plan = analyse(clip, script, duration=14.17)
    seen = identities(plan)

    assert len(seen) == 1, f"one physical person fragmented into {len(seen)} identities"
    assert plan.diagnostics.fragmentation.identities == 1
    assert plan.diagnostics.fragmentation.longest_lifetime_sec > 12.0
    assert plan.diagnostics.fragmentation.warning is None


#: Two positions far enough apart that no predicted motion can link them, fired far enough
#: apart in time that a candidate track has already expired before the next one appears. A
#: detector doing this is producing false positives, not following anybody.
GHOST_SPOTS = ((150, 300), (1130, 300))
GHOST_EVERY = 10  # samples, i.e. 2.0 s at 5 fps - longer than track_max_gap_sec


def ghost_at(index: int) -> Optional[Tuple[int, int, int]]:
    if index % GHOST_EVERY:
        return None
    spot = GHOST_SPOTS[(index // GHOST_EVERY) % 2]
    return (spot[0], spot[1], 70)


def test_single_frame_false_positives_never_become_identities(clip):
    """A detector that fires a stray box every couple of seconds must not invent people."""
    def script(index):
        subjects = [(400, 300, FACE)]
        ghost = ghost_at(index)
        if ghost:
            subjects.append(ghost)
        return subjects

    plan = analyse(clip, script, duration=20.0)
    seen = identities(plan)

    assert len(seen) == 1, f"stray detections created {len(seen)} identities"
    report = plan.diagnostics.fragmentation
    assert report.raw_candidates > report.identities, "the ghosts should still be counted as raw"
    assert report.discarded_tentative > 0


def test_the_report_says_when_detections_produced_no_identity_at_all(clip):
    """The diagnostic that distinguishes a tracking failure from an empty room."""
    def script(index):
        # Isolated in both space and time: nothing here can ever be associated with anything.
        ghost = ghost_at(index)
        return [ghost] if ghost else []

    plan = analyse(clip, script, duration=30.0)
    report = plan.diagnostics.fragmentation

    assert report.identities == 0
    assert report.detections_total >= 10
    assert report.warning is not None
    assert "no stable identity" in report.warning


def test_fragmentation_diagnostics_report_every_documented_metric(clip):
    def script(index):
        return [(400, 300, FACE)] if index % 6 != 5 else []

    report = analyse(clip, script, duration=20.0).diagnostics.fragmentation
    assert report.identities >= 1
    assert report.mean_lifetime_sec > 0
    assert report.median_lifetime_sec > 0
    assert report.longest_lifetime_sec >= report.median_lifetime_sec
    assert report.tracks_under_half_second >= 0
    assert report.tracks_under_one_second >= 0
    assert report.fragmentation_ratio >= 1.0
    assert report.detections_total > 0
    assert report.detections_per_identity > 0
    assert report.summary()


# ---------------------------------------------------------------------------
# Re-attachment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gap_sec", [0.5, 0.8, 1.0])
def test_a_subject_lost_for_up_to_a_second_recovers_its_identity(clip, gap_sec):
    hidden = {index for index in range(int(6.0 * FPS), int((6.0 + gap_sec) * FPS))}

    def script(index):
        if index in hidden:
            return []
        return [(300 + 10.0 * (index / FPS), 300, FACE)]

    plan = analyse(clip, script, duration=20.0)
    seen = identities(plan)

    assert len(seen) == 1, f"a {gap_sec}s dropout created {len(seen)} identities"
    assert plan.diagnostics.fragmentation.reattachments >= 1


def test_a_subject_gone_far_longer_than_the_grace_is_a_new_identity(clip):
    """Identity recovery must not become identity invention."""
    def script(index):
        if int(4.0 * FPS) <= index < int(9.0 * FPS):
            return []
        return [(300, 300, FACE)]

    plan = analyse(clip, script, duration=20.0, track_max_gap_sec=1.0)
    assert len(identities(plan)) == 2


def test_a_moving_subject_is_recovered_where_it_actually_went(clip):
    """The bug: gating a one-second dropout as though only one frame had elapsed."""
    hidden = {index for index in range(int(5.0 * FPS), int(6.0 * FPS))}

    def script(index):
        if index in hidden:
            return []
        return [(200 + 220.0 * (index / FPS), 300, FACE)]

    plan = analyse(clip, script, duration=8.0)
    assert len(identities(plan)) == 1


# ---------------------------------------------------------------------------
# Not merging different people
# ---------------------------------------------------------------------------


def test_two_people_standing_apart_keep_two_identities(clip):
    def script(index):
        return [(240, 300, FACE), (1040, 300, FACE)]

    plan = analyse(clip, script, duration=20.0)
    assert len(identities(plan)) == 2


def test_crossing_subjects_do_not_swap_or_merge(clip):
    """Two people who walk through each other must come out the other side as two people."""
    def script(index):
        t = index / FPS
        left = 240 + 60.0 * t
        right = 1040 - 60.0 * t
        return [(left, 260, FACE), (right, 340, FACE)]

    plan = analyse(clip, script, duration=12.0)
    seen = identities(plan)
    assert len(seen) == 2, f"crossing subjects produced {len(seen)} identities"
    # Both must survive the crossing rather than one absorbing the other.
    for times in seen.values():
        assert (max(times) - min(times)) > 8.0


def test_a_distant_detection_does_not_capture_a_lost_identity(clip):
    """When one person leaves and another appears across the frame, that is two people."""
    def script(index):
        if index < int(5.0 * FPS):
            return [(200, 300, FACE)]
        if index < int(5.6 * FPS):
            return []
        return [(1100, 300, FACE)]

    plan = analyse(clip, script, duration=14.0)
    assert len(identities(plan)) == 2


def test_faces_of_very_different_sizes_are_not_the_same_person(clip):
    """A close facecam and a distant face are different subjects even when they overlap."""
    def script(index):
        return [(500, 300, 40), (520, 320, 190)]

    plan = analyse(clip, script, duration=14.0)
    assert len(identities(plan)) == 2


# ---------------------------------------------------------------------------
# Scene cuts
# ---------------------------------------------------------------------------


def test_a_scene_cut_does_not_carry_a_lost_subject_into_the_new_shot(tmp_path):
    """After a cut, a subject that was already missing belongs to the previous shot."""
    from test_shorts_reframe import write_video as write

    cut_clip = write(tmp_path / "cut.mp4", seconds=16.0, scene_cut_at=6.0)

    def script(index):
        t = index / FPS
        if t < 5.4:
            return [(300, 300, FACE)]
        if t < 6.4:
            return []
        return [(300, 300, FACE)]

    plan = build_reframe_plan(
        video_path=cut_clip,
        source_start_sec=0.0,
        duration_sec=14.0,
        detector=TimelineDetector(script),
        config=ReframeConfig(analysis_fps=FPS),
        source_width=SOURCE_W,
        source_height=SOURCE_H,
    )
    assert plan.diagnostics.scene_cuts >= 1
    assert len(identities(plan)) == 2


def test_scene_cut_reset_can_be_turned_off(tmp_path):
    from test_shorts_reframe import write_video as write

    cut_clip = write(tmp_path / "cut.mp4", seconds=16.0, scene_cut_at=6.0)

    def script(index):
        t = index / FPS
        if 5.4 <= t < 6.4:
            return []
        return [(300, 300, FACE)]

    plan = build_reframe_plan(
        video_path=cut_clip,
        source_start_sec=0.0,
        duration_sec=14.0,
        detector=TimelineDetector(script),
        config=ReframeConfig(analysis_fps=FPS, track_scene_cut_reset=False),
        source_width=SOURCE_W,
        source_height=SOURCE_H,
    )
    assert len(identities(plan)) == 1


# ---------------------------------------------------------------------------
# Persistence classification
# ---------------------------------------------------------------------------


def test_three_seconds_on_screen_is_a_real_subject_in_a_thirty_second_clip(clip):
    """The old global-visibility rule erased anyone who was not present most of the clip."""
    def script(index):
        t = index / FPS
        if 4.0 <= t < 7.5:
            return [(400, 300, FACE), (900, 300, FACE)]
        return [(400, 300, FACE)]

    plan = analyse(clip, script, duration=30.0)
    summaries = {s.track_id: s for s in summarize_tracks(plan.frames, SOURCE_W, SOURCE_H)}
    guests = [s for s in summaries.values() if s.visibility_ratio < 0.3]

    assert guests, "expected a subject present for only part of the clip"
    guest = guests[0]
    assert guest.longest_continuous_sec >= 2.0
    assert guest.persistent, "a subject on screen for 3.5 s straight is not noise"
    assert "continuously present" in guest.persistence_reason


def test_a_blink_of_a_face_is_still_not_a_subject(clip):
    def script(index):
        subjects = [(400, 300, FACE)]
        if index == 20:
            subjects.append((1000, 300, FACE))
        return subjects

    plan = analyse(clip, script, duration=20.0)
    summaries = summarize_tracks(plan.frames, SOURCE_W, SOURCE_H)
    assert [s.persistent for s in summaries] == [True]


def test_continuous_presence_tolerates_a_short_dropout(clip):
    def script(index):
        t = index / FPS
        if 5.0 <= t < 5.4:
            return []
        return [(400, 300, FACE)]

    plan = analyse(clip, script, duration=20.0)
    summary = summarize_tracks(plan.frames, SOURCE_W, SOURCE_H)[0]
    assert summary.longest_continuous_sec > 15.0, "a 0.4 s dropout must not split the run"


def test_the_plan_warns_when_detections_produce_no_usable_subject(clip):
    """The cand_080 signature at the layout level: plenty of tracking, nothing to frame with."""
    from freecher_worker.shorts.layout import build_layout_plan
    from freecher_worker.shorts.reframe import calculate_vertical_crop

    #: Far enough apart that consecutive appearances cannot be the same person, and cycling
    #: slowly enough that a slot's own track has expired before that slot comes round again.
    spots = (100, 640, 1180)

    def script(index):
        slot = index // 5
        if index % 5 == 4:
            return []
        return [(spots[slot % len(spots)], 300, FACE)]

    plan = analyse(clip, script, duration=30.0)
    crop_w, crop_h = calculate_vertical_crop(SOURCE_W, SOURCE_H)
    layout = build_layout_plan(
        frames=plan.frames,
        duration=30.0,
        source_width=SOURCE_W,
        source_height=SOURCE_H,
        crop_w=crop_w,
        crop_h=crop_h,
        mode="adaptive",
        config=LayoutConfig(),
    )

    assert layout.track_count > 0, "the tracker did see subjects"
    assert layout.persistent_track_count == 0, "none of them stayed long enough to frame"
    assert layout.fragmentation_warning is not None
    assert "no persistent subject" in layout.fragmentation_warning
