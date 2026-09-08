"""FFmpeg expression generation from temporal crop trajectories."""

from __future__ import annotations

from typing import List
from .models import CropPoint, CropTrajectory


def simplify_trajectory_points(
    points: List[CropPoint],
    tolerance_px: float = 2.0,
    axis: str = "x",
) -> List[CropPoint]:
    """Reduce redundant points along linear segments to keep FFmpeg expression compact."""
    if len(points) <= 2:
        return points

    attr = "crop_x" if axis == "x" else "crop_y"
    simplified: List[CropPoint] = [points[0]]
    for i in range(1, len(points) - 1):
        prev_pt = simplified[-1]
        curr_pt = points[i]
        next_pt = points[i + 1]

        # Check if curr_pt is approximately collinear between prev_pt and next_pt
        dt_total = next_pt.time - prev_pt.time
        if dt_total > 1e-4:
            alpha = (curr_pt.time - prev_pt.time) / dt_total
            expected = getattr(prev_pt, attr) + alpha * (getattr(next_pt, attr) - getattr(prev_pt, attr))
            if abs(getattr(curr_pt, attr) - expected) > tolerance_px:
                simplified.append(curr_pt)
        else:
            simplified.append(curr_pt)

    simplified.append(points[-1])
    return simplified


def build_ffmpeg_crop_expression(
    trajectory: CropTrajectory,
    axis: str = "x",
    escape_for_filter: bool = True,
) -> str:
    """Convert one axis of a crop trajectory into a piecewise-linear FFmpeg expression.

    The result is clamped to the valid crop range for that axis and snapped to even integers,
    which keeps H.264 / yuv420p happy regardless of where the trajectory wandered.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    attr = "crop_x" if axis == "x" else "crop_y"
    bound = "in_w-out_w" if axis == "x" else "in_h-out_h"
    center_default = f"trunc(({bound})/2)"

    raw_points = trajectory.points
    if not raw_points:
        return center_default

    points = simplify_trajectory_points(raw_points, tolerance_px=2.0, axis=axis)
    first_value = getattr(points[0], attr)
    if len(points) == 1 or all(getattr(p, attr) == first_value for p in points):
        return f"{first_value}"

    expr = f"{getattr(points[-1], attr)}"
    for i in range(len(points) - 2, -1, -1):
        p0, p1 = points[i], points[i + 1]
        t0, v0 = p0.time, getattr(p0, attr)
        t1, v1 = p1.time, getattr(p1, attr)
        dt = max(0.001, t1 - t0)
        dv = v1 - v0
        segment = f"{v0}" if abs(dv) < 1 else f"({v0}+({dv:.1f})*(t-{t0:.2f})/{dt:.3f})"
        expr = f"if(lte(t,{t1:.2f}),{segment},{expr})"

    clamped = f"2*trunc(min(max(0,{expr}),{bound})/2)"
    return clamped.replace(",", r"\,") if escape_for_filter else clamped


def build_ffmpeg_crop_x_expression(trajectory: CropTrajectory, escape_for_filter: bool = True) -> str:
    """Convert crop trajectory into an FFmpeg-evaluable piecewise-linear expression x(t).

    Returns an expression string suitable for crop=w:h:x:y in FFmpeg,
    guaranteed to remain clamped within [0, in_w - out_w] and aligned to even integers.
    If escape_for_filter is True, commas are escaped as '\\,' so FFmpeg does not treat them
    as filtergraph separators.
    """
    raw_points = trajectory.points
    if not raw_points:
        return "trunc((in_w-out_w)/2)"

    # Simplify trajectory to avoid excessive expression nesting in FFmpeg
    points = simplify_trajectory_points(raw_points, tolerance_px=2.0)

    # If all points share the same crop_x coordinate, return static constant
    first_x = points[0].crop_x
    if all(p.crop_x == first_x for p in points):
        return f"{first_x}"

    if len(points) == 1:
        return f"{first_x}"

    # Build piecewise linear interpolation from right to left (innermost fallback to outermost if)
    # At t >= last point time, value is last point crop_x
    last_pt = points[-1]
    expr = f"{last_pt.crop_x}"

    # Build segments backwards: if(lte(t, t_i), interp_expr, fallback)
    for i in range(len(points) - 2, -1, -1):
        p0 = points[i]
        p1 = points[i + 1]

        t0, x0 = p0.time, p0.crop_x
        t1, x1 = p1.time, p1.crop_x

        dt = max(0.001, t1 - t0)
        dx = x1 - x0

        if abs(dx) < 1:
            segment_expr = f"{x0}"
        else:
            segment_expr = f"({x0}+({dx:.1f})*(t-{t0:.2f})/{dt:.3f})"

        expr = f"if(lte(t,{t1:.2f}),{segment_expr},{expr})"

    # Clamp to [0, in_w - out_w] and ensure even integer
    clamped_expr = f"2*trunc(min(max(0,{expr}),in_w-out_w)/2)"
    if escape_for_filter:
        clamped_expr = clamped_expr.replace(",", r"\,")
    return clamped_expr
