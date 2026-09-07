"""Temporal crop tracking, multi-subject arbitration, and smoothing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional
import cv2

from .detector import SubjectDetector, get_subject_detector
from .models import CropPoint, CropTrajectory, DetectedSubject

logger = logging.getLogger("freecher_worker")


def calculate_crop_dimensions(source_width: int, source_height: int) -> tuple[int, int]:
    """Calculate even-integer 9:16 crop dimensions in source frame space."""
    crop_h = source_height
    # 9/16 aspect ratio: width = height * 9 / 16
    raw_w = (source_height * 9.0) / 16.0
    # Ensure even integer for H.264 / yuv420p
    crop_w = int(round(raw_w / 2.0)) * 2
    crop_w = min(crop_w, source_width)
    if crop_w % 2 != 0:
        crop_w -= 1
    if crop_h % 2 != 0:
        crop_h -= 1
    return crop_w, crop_h


def select_target_center(
    subjects: List[DetectedSubject],
    source_width: int,
    source_height: int,
    crop_w: int,
) -> tuple[float, float, str]:
    """Arbitrate multi-person observations to choose the dominant visual subject."""
    center_fallback_x = source_width / 2.0
    center_fallback_y = source_height / 2.0

    if not subjects:
        return center_fallback_x, center_fallback_y, "center_fallback"

    faces = [s for s in subjects if s.subject_type == "face"]
    active_pool = faces if faces else subjects

    # Rule 1: Single subject
    if len(active_pool) == 1:
        s = active_pool[0]
        return s.center_x, s.center_y, s.subject_type

    # Rule 2: Two nearby faces that fit comfortably in a 9:16 window
    if len(faces) == 2:
        f1, f2 = faces[0], faces[1]
        span = abs(f1.center_x - f2.center_x)
        max_box_w = max(f1.box[2], f2.box[2])
        if (span + max_box_w) <= (crop_w * 0.85):
            mid_x = (f1.center_x + f2.center_x) / 2.0
            mid_y = (f1.center_y + f2.center_y) / 2.0
            return mid_x, mid_y, "dual_faces"

    # Rule 3: Multiple separated subjects -> pick dominant visual subject by area and centrality
    def score_subject(s: DetectedSubject) -> float:
        dist_from_center = abs(s.center_x - center_fallback_x) / (center_fallback_x or 1.0)
        centrality_weight = max(0.5, 1.0 - 0.4 * dist_from_center)
        return s.area * centrality_weight

    dominant = max(active_pool, key=score_subject)
    return dominant.center_x, dominant.center_y, dominant.subject_type


def generate_crop_trajectory(
    video_path: Path,
    start_seconds: float,
    duration_seconds: float,
    detector: Optional[SubjectDetector] = None,
    analysis_fps: float = 2.0,
    deadzone_ratio: float = 0.03,
    max_velocity_px_per_sec: float = 200.0,
) -> CropTrajectory:
    """Analyze video at sampled intervals to build a smooth 9:16 crop trajectory."""
    if detector is None:
        detector = get_subject_detector("haar")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video for crop analysis: {video_path}")

    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w, crop_h = calculate_crop_dimensions(source_width, source_height)

    # Sampling timestamps
    interval = 1.0 / max(0.5, analysis_fps)
    num_steps = max(2, int(duration_seconds / interval) + 1)
    sample_times = [min(duration_seconds, i * interval) for i in range(num_steps)]
    if sample_times[-1] < duration_seconds:
        sample_times.append(duration_seconds)

    raw_observations: List[tuple[float, float, float, str]] = []
    deadzone_px = deadzone_ratio * source_width

    # 1. Sample and detect
    for t_rel in sample_times:
        t_abs = start_seconds + t_rel
        cap.set(cv2.CAP_PROP_POS_MSEC, t_abs * 1000.0)
        ret, frame = cap.read()
        if not ret or frame is None:
            raw_observations.append((t_rel, source_width / 2.0, source_height / 2.0, "center_fallback"))
            continue

        subjects = detector.detect(frame, t_abs)
        tgt_x, tgt_y, subj_type = select_target_center(subjects, source_width, source_height, crop_w)
        raw_observations.append((t_rel, tgt_x, tgt_y, subj_type))

    cap.release()

    # 2. Apply dead-zone, velocity clamping, and temporal smoothing
    points: List[CropPoint] = []
    curr_cx = source_width / 2.0
    curr_cy = source_height / 2.0

    for idx, (t, tgt_x, tgt_y, subj_type) in enumerate(raw_observations):
        if idx == 0:
            curr_cx = tgt_x
            curr_cy = tgt_y
        else:
            prev_t = raw_observations[idx - 1][0]
            dt = max(0.01, t - prev_t)

            # Dead zone check
            diff_x = tgt_x - curr_cx
            if abs(diff_x) > deadzone_px:
                # Velocity clamping: max distance allowed in dt
                max_step = max_velocity_px_per_sec * dt
                clamped_step = max(-max_step, min(max_step, diff_x))
                curr_cx += clamped_step

            # Center Y typically remains vertical middle
            curr_cy = (curr_cy + tgt_y) / 2.0

        # Calculate clamped top-left crop box
        top_left_x = int(round(curr_cx - crop_w / 2.0))
        top_left_x = max(0, min(top_left_x, source_width - crop_w))
        # Ensure even integer
        top_left_x = (top_left_x // 2) * 2
        top_left_y = 0  # Full height crop for 9:16 on landscape

        points.append(
            CropPoint(
                time=round(t, 3),
                center_x=round(curr_cx, 2),
                center_y=round(curr_cy, 2),
                crop_x=top_left_x,
                crop_y=top_left_y,
                crop_w=crop_w,
                crop_h=crop_h,
                subject_type=subj_type,
            )
        )

    return CropTrajectory(
        source_width=source_width,
        source_height=source_height,
        crop_w=crop_w,
        crop_h=crop_h,
        points=points,
    )
