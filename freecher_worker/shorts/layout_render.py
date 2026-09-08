"""FFmpeg delivery of an adaptive layout plan — one pass, one encode, three possible layouts.

The plan is declarative; this module is the only place that knows how a layout becomes pixels.

Each layout is built as its own branch of a single filtergraph, all of them producing a full
``WIDTHxHEIGHT`` frame, and the plan's segments decide which branch is on screen at each moment::

    [0:v] split -> single  : sendcmd + crop@single           -> scale/crop 1080x1920
                -> dual    : sendcmd + crop@dual_top/@dual_bottom -> 2x 1080x960 -> vstack
                -> full    : blurred cover + fitted contain  -> overlay centred
          -> overlay ... enable='gte(t,s)*lt(t,e)' -> [v]

Two details make this work rather than merely look plausible:

* each moving viewport gets its **own** ``sendcmd`` script targeting its **own** named crop
  instance (``crop@dual_top``), because three viewports panning independently cannot share one
  command stream;
* the switch predicate is ``gte(t,s)*lt(t,e)`` rather than ``between``, so two adjacent segments
  can never both be enabled on the frame that sits exactly on their boundary.

Branches are only built when the plan actually uses them, so a plan that never leaves
``single_subject`` costs exactly what the non-adaptive path costs.
"""

from __future__ import annotations

import logging
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from freecher_worker.crop.models import CropTrajectory

from .layout import (
    LAYOUT_DUAL,
    LAYOUT_FULL,
    LAYOUT_SINGLE,
    LayoutConfig,
    LayoutPlan,
    LayoutSegment,
    ViewportPlan,
    VIEWPORT_DUAL_BOTTOM,
    VIEWPORT_DUAL_TOP,
    VIEWPORT_SINGLE,
)
from .trajectory import (
    CROP_DRIVER_SENDCMD,
    CROP_DRIVER_STATIC,
    build_sendcmd_script,
    escape_filtergraph_value,
    validate_and_sanitize_trajectory,
)

logger = logging.getLogger("freecher_worker")


class LayoutRenderError(ValueError):
    """Raised when a layout plan cannot be turned into a renderable filtergraph."""


class ViewportRender(BaseModel):
    """A viewport reduced to what FFmpeg needs: a crop window and how it moves."""

    name: str
    crop_w: int
    crop_h: int
    x_expr: str
    y_expr: str
    script: Optional[str] = None
    keyframes: int = 0

    @property
    def needs_script(self) -> bool:
        return self.script is not None


class AdaptiveRenderPlan(BaseModel):
    """Everything needed to render one short with a temporally varying layout."""

    width: int
    height: int
    segments: List[LayoutSegment] = Field(default_factory=list)
    viewports: Dict[str, ViewportRender] = Field(default_factory=dict)
    base_layout: str = LAYOUT_SINGLE
    overlay_layouts: List[str] = Field(default_factory=list)
    blur_background: bool = True
    blur_sigma: float = 12.0
    background_color: str = "black"

    @property
    def layouts_used(self) -> List[str]:
        seen: List[str] = []
        for segment in self.segments:
            if segment.layout not in seen:
                seen.append(segment.layout)
        return seen

    def script_names(self) -> List[str]:
        return [name for name, viewport in sorted(self.viewports.items()) if viewport.needs_script]


@lru_cache(maxsize=1)
def supports_named_filter_instances() -> bool:
    """Whether this FFmpeg build accepts ``crop@id`` and routes ``sendcmd`` to that instance.

    Independent viewports are impossible without it, so adaptive rendering degrades to the
    single-crop layout rather than silently producing a frozen stack.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error",
                "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
                "-filter_complex", "[0:v]crop@probe=32:32:0:0[v]",
                "-map", "[v]", "-frames:v", "1", "-f", "null", "-",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=15.0,
        )
        return result.returncode == 0
    except Exception as exc:  # pragma: no cover - probing must never break a render
        logger.warning(f"[layout-render] Could not probe named filter instances: {exc}")
        return False


def _viewport_render(name: str, trajectory: CropTrajectory) -> ViewportRender:
    """Validate one viewport's trajectory and turn it into a crop plus its command script."""
    sanitized, report = validate_and_sanitize_trajectory(trajectory)
    logger.info(f"[layout-render] viewport '{name}' trajectory {report.summary()}")
    xs = [p.crop_x for p in sanitized.points]
    ys = [p.crop_y for p in sanitized.points]
    dynamic = len(set(xs)) > 1 or len(set(ys)) > 1
    return ViewportRender(
        name=name,
        crop_w=sanitized.crop_w,
        crop_h=sanitized.crop_h,
        x_expr=str(xs[0]),
        y_expr=str(ys[0]),
        script=build_sendcmd_script(sanitized) if dynamic else None,
        keyframes=len(sanitized.points) if dynamic else 1,
    )


def build_adaptive_render_plan(
    plan: LayoutPlan,
    single_trajectory: CropTrajectory,
    width: int,
    height: int,
    dual_viewports: Optional[Tuple[ViewportPlan, ViewportPlan]] = None,
    config: Optional[LayoutConfig] = None,
) -> AdaptiveRenderPlan:
    """Reduce a layout plan and its trajectories to renderable branches.

    A plan that asks for ``dual_stack`` without usable viewports is rewritten to
    ``full_frame_context`` rather than rendered wrong: showing the whole frame is always honest.
    """
    cfg = config or LayoutConfig()
    if not plan.segments:
        raise LayoutRenderError("layout plan has no segments")

    segments = [segment.model_copy(deep=True) for segment in plan.segments]
    if any(s.layout == LAYOUT_DUAL for s in segments) and dual_viewports is None:
        logger.warning(
            "[layout-render] dual_stack was planned but no viewports were built; "
            "those segments fall back to full_frame_context"
        )
        for segment in segments:
            if segment.layout == LAYOUT_DUAL:
                segment.layout = LAYOUT_FULL
                segment.subject_ids = []
                segment.subjects = []
                segment.reason = "dual viewports unavailable; fell back to full frame"

    viewports: Dict[str, ViewportRender] = {}
    if any(s.layout == LAYOUT_SINGLE for s in segments):
        viewports[VIEWPORT_SINGLE] = _viewport_render(VIEWPORT_SINGLE, single_trajectory)
    if any(s.layout == LAYOUT_DUAL for s in segments) and dual_viewports is not None:
        top, bottom = dual_viewports
        viewports[VIEWPORT_DUAL_TOP] = _viewport_render(VIEWPORT_DUAL_TOP, top.trajectory)
        viewports[VIEWPORT_DUAL_BOTTOM] = _viewport_render(VIEWPORT_DUAL_BOTTOM, bottom.trajectory)

    used: Dict[str, float] = {}
    for segment in segments:
        used[segment.layout] = used.get(segment.layout, 0.0) + segment.duration
    # The layout that is on screen longest becomes the base, so the common case pays for the
    # fewest overlays.
    base = max(sorted(used), key=lambda layout: used[layout])
    overlays = [layout for layout in sorted(used) if layout != base]

    return AdaptiveRenderPlan(
        width=width,
        height=height,
        segments=segments,
        viewports=viewports,
        base_layout=base,
        overlay_layouts=overlays,
        blur_background=cfg.full_frame_blur_background,
        blur_sigma=cfg.full_frame_blur_sigma,
        background_color=cfg.full_frame_background_color,
    )


def _enable_expression(segments: Sequence[LayoutSegment], layout: str) -> str:
    """Timeline predicate that is true exactly while ``layout`` is on screen."""
    terms = [
        f"gte(t,{segment.start:.4f})*lt(t,{segment.end:.4f})"
        for segment in segments
        if segment.layout == layout
    ]
    return "+".join(terms) if terms else "0"


def _single_chain(viewport: ViewportRender, width: int, height: int, script: Optional[Path]) -> str:
    parts: List[str] = []
    if viewport.needs_script:
        if script is None:
            raise LayoutRenderError(f"viewport '{viewport.name}' needs a command script path")
        parts.append(f"sendcmd=f={escape_filtergraph_value(str(script))}")
    parts.append(
        f"crop@{viewport.name}={viewport.crop_w}:{viewport.crop_h}:{viewport.x_expr}:{viewport.y_expr}"
    )
    parts.append(f"scale={width}:{height}:force_original_aspect_ratio=increase")
    parts.append(f"crop={width}:{height}")
    parts.append("setsar=1")
    return ",".join(parts)


def _full_frame_chains(plan: AdaptiveRenderPlan, label_in: str, label_out: str) -> str:
    """Whole source frame fitted into 9:16 — nothing is ever cropped away here."""
    width, height = plan.width, plan.height
    if not plan.blur_background:
        return (
            f"[{label_in}]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={plan.background_color},"
            f"setsar=1[{label_out}]"
        )
    # Blur on a downscaled copy: visually identical to a wide-radius blur, a fraction of the cost.
    small_w = max(2, (width // 8) // 2 * 2)
    small_h = max(2, (height // 8) // 2 * 2)
    return (
        f"[{label_in}]split=2[{label_out}_bg][{label_out}_fg];"
        f"[{label_out}_bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},scale={small_w}:{small_h},gblur=sigma={plan.blur_sigma:g},"
        f"scale={width}:{height},setsar=1[{label_out}_bgb];"
        f"[{label_out}_fg]scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"setsar=1[{label_out}_fgf];"
        f"[{label_out}_bgb][{label_out}_fgf]overlay=(W-w)/2:(H-h)/2[{label_out}]"
    )


def build_adaptive_video_filter(
    plan: AdaptiveRenderPlan,
    script_paths: Optional[Dict[str, Path]] = None,
) -> str:
    """Build the complete video filtergraph for an adaptive plan, ending in ``[v]``."""
    scripts = script_paths or {}
    for name in plan.script_names():
        if name not in scripts:
            raise LayoutRenderError(f"missing command script path for viewport '{name}'")

    layouts = [plan.base_layout] + plan.overlay_layouts
    chains: List[str] = []
    inputs = [f"src{i}" for i in range(len(layouts))]
    if len(layouts) == 1:
        chains.append(f"[0:v]null[{inputs[0]}]")
    else:
        chains.append("[0:v]split=" + str(len(layouts)) + "".join(f"[{name}]" for name in inputs))

    produced: Dict[str, str] = {}
    for layout, source in zip(layouts, inputs):
        label = f"lay_{layout}"
        if layout == LAYOUT_SINGLE:
            viewport = plan.viewports[VIEWPORT_SINGLE]
            chains.append(
                f"[{source}]{_single_chain(viewport, plan.width, plan.height, scripts.get(viewport.name))}[{label}]"
            )
        elif layout == LAYOUT_DUAL:
            top = plan.viewports[VIEWPORT_DUAL_TOP]
            bottom = plan.viewports[VIEWPORT_DUAL_BOTTOM]
            half = plan.height // 2
            chains.append(f"[{source}]split=2[{label}_t][{label}_b]")
            for viewport, tag in ((top, "t"), (bottom, "b")):
                parts: List[str] = []
                if viewport.needs_script:
                    parts.append(f"sendcmd=f={escape_filtergraph_value(str(scripts[viewport.name]))}")
                parts.append(
                    f"crop@{viewport.name}={viewport.crop_w}:{viewport.crop_h}:"
                    f"{viewport.x_expr}:{viewport.y_expr}"
                )
                parts.append(f"scale={plan.width}:{half}:force_original_aspect_ratio=increase")
                parts.append(f"crop={plan.width}:{half}")
                parts.append("setsar=1")
                chains.append(f"[{label}_{tag}]{','.join(parts)}[{label}_{tag}v]")
            chains.append(f"[{label}_tv][{label}_bv]vstack=inputs=2,setsar=1[{label}]")
        elif layout == LAYOUT_FULL:
            chains.append(_full_frame_chains(plan, source, label))
        else:  # pragma: no cover - guarded by the planner's own vocabulary
            raise LayoutRenderError(f"unknown layout '{layout}'")
        produced[layout] = label

    current = produced[plan.base_layout]
    for index, layout in enumerate(plan.overlay_layouts):
        enable = _enable_expression(plan.segments, layout)
        out = f"ov{index}"
        chains.append(f"[{current}][{produced[layout]}]overlay=0:0:enable='{enable}'[{out}]")
        current = out

    chains.append(f"[{current}]format=yuv420p[v]")
    return ";".join(chains)


def describe_plan(plan: AdaptiveRenderPlan) -> str:
    """One-line summary logged before FFmpeg is invoked."""
    parts = [f"{s.layout}@{s.start:.1f}-{s.end:.1f}s" for s in plan.segments]
    return (
        f"base={plan.base_layout} overlays={plan.overlay_layouts or 'none'} "
        f"viewports={sorted(plan.viewports)} segments=[{', '.join(parts)}]"
    )


def crop_driver_of(plan: AdaptiveRenderPlan) -> str:
    """Which delivery mechanism the adaptive graph actually relies on."""
    return CROP_DRIVER_SENDCMD if plan.script_names() else CROP_DRIVER_STATIC
