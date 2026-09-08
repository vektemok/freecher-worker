"""Validation and FFmpeg delivery of dynamic crop trajectories.

FFmpeg's expression evaluator (``libavutil/eval.c``) caps a single expression at roughly one
hundred parsed nodes: ``av_expr_parse()`` starts with ``p.stack_index = 100`` and the parser
fails with ``AVERROR(EINVAL)`` once that budget is exhausted. A 31-second short sampled at 5 fps
produces ~155 trajectory keyframes, which blows the budget in either a nested ``if()`` chain or
a flat sum, and ``crop`` then reports::

    [Parsed_crop_0] Failed to configure input pad on Parsed_crop_0
    Invalid argument

The trajectory is therefore delivered through ``sendcmd`` instead: one short, independently
parsed linear expression per keyframe interval, with no practical limit on keyframe count and no
loss of tracking resolution. A budget-limited expression and a static center crop remain as
fallbacks for builds without ``sendcmd``.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from freecher_worker.crop.expression import (
    build_ffmpeg_crop_expression,
    fit_points_to_expression_budget,
)
from freecher_worker.crop.models import CropPoint, CropTrajectory

logger = logging.getLogger("freecher_worker")

CROP_DRIVER_SENDCMD = "sendcmd"
CROP_DRIVER_EXPRESSION = "expression"
CROP_DRIVER_STATIC = "static"

#: Measured ceiling of FFmpeg's expression node budget (fails at ~101 nodes on 6.x and 9.x).
FFMPEG_EXPRESSION_NODE_LIMIT = 100
#: Keyframes per axis the expression fallback may use, with headroom under that ceiling.
EXPRESSION_KEYFRAME_BUDGET = 45


class TrajectoryValidationError(ValueError):
    """Raised when a crop trajectory cannot be repaired into something FFmpeg can render."""


class TrajectoryReport(BaseModel):
    """Everything checked about a trajectory before it is handed to FFmpeg."""

    source_width: int
    source_height: int
    crop_width: int
    crop_height: int
    max_x_allowed: int
    max_y_allowed: int

    point_count_in: int = 0
    point_count_out: int = 0
    min_x: int = 0
    max_x: int = 0
    min_y: int = 0
    max_y: int = 0

    non_finite_dropped: int = 0
    out_of_range_clamped: int = 0
    odd_coordinates_fixed: int = 0
    non_monotonic_times_fixed: int = 0

    valid: bool = True
    errors: List[str] = Field(default_factory=list)

    def summary(self) -> str:
        """One-line description suitable for a log record before invoking FFmpeg."""
        return (
            f"source={self.source_width}x{self.source_height} "
            f"crop={self.crop_width}x{self.crop_height} "
            f"points={self.point_count_out}/{self.point_count_in} "
            f"x=[{self.min_x},{self.max_x}]<=|{self.max_x_allowed}| "
            f"y=[{self.min_y},{self.max_y}]<=|{self.max_y_allowed}| "
            f"repairs(nonfinite={self.non_finite_dropped}, clamped={self.out_of_range_clamped}, "
            f"odd={self.odd_coordinates_fixed}, time={self.non_monotonic_times_fixed})"
        )


def _finite(*values: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values)


def validate_and_sanitize_trajectory(
    trajectory: CropTrajectory,
) -> Tuple[CropTrajectory, TrajectoryReport]:
    """Check every trajectory point and repair what can be repaired.

    Non-finite points are dropped, out-of-range coordinates are clamped into the legal crop
    window, odd coordinates are snapped down to even (H.264 / yuv420p needs even offsets and
    sizes), and non-monotonic timestamps are nudged forward so interpolation stays well defined.

    Raises:
        TrajectoryValidationError: if the geometry itself is impossible (crop larger than the
            source, non-positive dimensions) or no usable point survives.
    """
    source_w = int(trajectory.source_width)
    source_h = int(trajectory.source_height)
    crop_w = int(trajectory.crop_w)
    crop_h = int(trajectory.crop_h)

    errors: List[str] = []
    if source_w <= 0 or source_h <= 0:
        errors.append(f"source dimensions are not positive: {source_w}x{source_h}")
    if crop_w <= 0 or crop_h <= 0:
        errors.append(f"crop dimensions are not positive: {crop_w}x{crop_h}")
    if crop_w > source_w:
        errors.append(f"crop width {crop_w} exceeds source width {source_w}")
    if crop_h > source_h:
        errors.append(f"crop height {crop_h} exceeds source height {source_h}")
    if crop_w % 2 or crop_h % 2:
        errors.append(f"crop size {crop_w}x{crop_h} is not even; H.264 yuv420p requires even dimensions")

    report = TrajectoryReport(
        source_width=source_w,
        source_height=source_h,
        crop_width=crop_w,
        crop_height=crop_h,
        max_x_allowed=max(0, source_w - crop_w),
        max_y_allowed=max(0, source_h - crop_h),
        point_count_in=len(trajectory.points),
    )

    if errors:
        report.valid = False
        report.errors = errors
        raise TrajectoryValidationError("; ".join(errors))

    max_x = report.max_x_allowed
    max_y = report.max_y_allowed

    clean: List[CropPoint] = []
    previous_time = -1.0
    for point in trajectory.points:
        if not _finite(point.time, point.crop_x, point.crop_y, point.center_x, point.center_y):
            report.non_finite_dropped += 1
            continue

        time = float(point.time)
        if time <= previous_time:
            time = previous_time + 1e-3
            report.non_monotonic_times_fixed += 1
        previous_time = time

        x, y = int(point.crop_x), int(point.crop_y)
        clamped_x = min(max(x, 0), max_x)
        clamped_y = min(max(y, 0), max_y)
        if clamped_x != x or clamped_y != y:
            report.out_of_range_clamped += 1

        even_x = (clamped_x // 2) * 2
        even_y = (clamped_y // 2) * 2
        if even_x != clamped_x or even_y != clamped_y:
            report.odd_coordinates_fixed += 1

        clean.append(
            point.model_copy(
                update={
                    "time": round(time, 4),
                    "crop_x": even_x,
                    "crop_y": even_y,
                    "crop_w": crop_w,
                    "crop_h": crop_h,
                }
            )
        )

    if not clean:
        report.valid = False
        report.errors = ["no finite trajectory points remained after validation"]
        raise TrajectoryValidationError(report.errors[0])

    report.point_count_out = len(clean)
    report.min_x = min(p.crop_x for p in clean)
    report.max_x = max(p.crop_x for p in clean)
    report.min_y = min(p.crop_y for p in clean)
    report.max_y = max(p.crop_y for p in clean)

    sanitized = trajectory.model_copy(update={"points": clean})
    return sanitized, report


class CropDriverPlan(BaseModel):
    """How a validated trajectory will actually reach FFmpeg."""

    driver: str = Field(description="sendcmd | expression | static")
    crop_w: int
    crop_h: int
    x_expr: str = Field(description="Initial/static crop x expression for the crop filter")
    y_expr: str = Field(description="Initial/static crop y expression for the crop filter")
    script: Optional[str] = Field(default=None, description="sendcmd command file contents")
    keyframes: int = Field(default=0, description="Keyframes actually delivered")
    keyframes_available: int = Field(default=0, description="Keyframes present in the trajectory")
    resolution_reduced: bool = Field(default=False, description="Keyframes were dropped to fit a limit")
    reason: str = ""


def _axis_values(points: List[CropPoint], axis: str) -> List[int]:
    return [p.crop_x if axis == "x" else p.crop_y for p in points]


def build_sendcmd_script(trajectory: CropTrajectory) -> str:
    """Render the trajectory as a ``sendcmd`` command file.

    Each interval installs a short linear expression for that interval only, so the number of
    keyframes is bounded by nothing but the file size, and interpolation stays exact.
    """
    points = trajectory.points
    lines: List[str] = []

    for axis, bound in (("x", "in_w-out_w"), ("y", "in_h-out_h")):
        values = _axis_values(points, axis)
        if len(set(values)) <= 1:
            # A constant axis needs no commands; the static crop option already carries it.
            continue
        for (p0, v0), (p1, v1) in zip(zip(points, values), zip(points[1:], values[1:])):
            start, end = p0.time, p1.time
            span = max(1e-3, end - start)
            if v1 == v0:
                inner = f"{v0}"
            else:
                inner = f"{v0}+{(v1 - v0) / span:.6f}*(t-{start:.4f})"
            lines.append(
                f"{start:.4f}-{end:.4f} crop {axis} '2*trunc(min(max(0,{inner}),{bound})/2)';"
            )
        # Hold the final value for any frame beyond the last keyframe.
        lines.append(f"{points[-1].time:.4f} crop {axis} '{values[-1]}';")

    return "\n".join(lines) + "\n"


def plan_crop_driver(
    trajectory: CropTrajectory,
    sendcmd_available: bool = True,
) -> CropDriverPlan:
    """Choose the most faithful delivery mechanism the current FFmpeg build supports."""
    points = trajectory.points
    x_values = _axis_values(points, "x")
    y_values = _axis_values(points, "y")
    dynamic_x = len(set(x_values)) > 1
    dynamic_y = len(set(y_values)) > 1
    available = len(points)

    if not dynamic_x and not dynamic_y:
        return CropDriverPlan(
            driver=CROP_DRIVER_STATIC,
            crop_w=trajectory.crop_w,
            crop_h=trajectory.crop_h,
            x_expr=str(x_values[0]),
            y_expr=str(y_values[0]),
            keyframes=1,
            keyframes_available=available,
            reason="trajectory is constant on both axes",
        )

    if sendcmd_available:
        return CropDriverPlan(
            driver=CROP_DRIVER_SENDCMD,
            crop_w=trajectory.crop_w,
            crop_h=trajectory.crop_h,
            # Static options seed the filter; sendcmd takes over from the first interval.
            x_expr=str(x_values[0]),
            y_expr=str(y_values[0]),
            script=build_sendcmd_script(trajectory),
            keyframes=available,
            keyframes_available=available,
            reason=f"sendcmd delivers all {available} keyframes at full tracking resolution",
        )

    # No sendcmd: fall back to a single expression, thinned to fit FFmpeg's node budget.
    fitted, reduced = fit_points_to_expression_budget(points, EXPRESSION_KEYFRAME_BUDGET)
    thinned = trajectory.model_copy(update={"points": fitted})
    logger.warning(
        f"[crop-driver] sendcmd filter unavailable; falling back to a crop expression limited to "
        f"{len(fitted)} of {available} keyframes"
    )
    return CropDriverPlan(
        driver=CROP_DRIVER_EXPRESSION,
        crop_w=trajectory.crop_w,
        crop_h=trajectory.crop_h,
        x_expr=build_ffmpeg_crop_expression(thinned, axis="x", escape_for_filter=True),
        y_expr=build_ffmpeg_crop_expression(thinned, axis="y", escape_for_filter=True),
        keyframes=len(fitted),
        keyframes_available=available,
        resolution_reduced=reduced,
        reason=(
            f"sendcmd unavailable; expression limited to {len(fitted)} of {available} keyframes "
            f"to stay under FFmpeg's ~{FFMPEG_EXPRESSION_NODE_LIMIT} node expression budget"
        ),
    )


def static_center_driver(source_width: int, source_height: int, crop_w: int, crop_h: int) -> CropDriverPlan:
    """Last-resort static center crop, used when dynamic delivery fails outright."""
    return CropDriverPlan(
        driver=CROP_DRIVER_STATIC,
        crop_w=crop_w,
        crop_h=crop_h,
        x_expr=str(((max(0, source_width - crop_w)) // 4) * 2),
        y_expr=str(((max(0, source_height - crop_h)) // 4) * 2),
        keyframes=1,
        reason="static center crop fallback",
    )
