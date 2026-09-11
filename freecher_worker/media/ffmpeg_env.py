"""FFmpeg/ffprobe discovery and capability probing.

Everything that shells out to FFmpeg resolves its binary here, so a deployment can
point at a build with the features the renderer needs instead of whatever happens
to be first on PATH. That is not hypothetical: Homebrew's default `ffmpeg` 9.0.1
ships without libass, while `ffmpeg-full` on the same machine has it, and the
renderer used to silently emit a subtitle-less MP4 on the former.

Resolution order for each binary:

1. ``FREECHER_FFMPEG_PATH`` / ``FREECHER_FFPROBE_PATH`` (settings ``ffmpeg_path`` /
   ``ffprobe_path``) -- an explicit binary or a directory containing it,
2. ``shutil.which`` on PATH.

No developer-specific absolute path is baked in.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Filters the renderer needs. Either subtitle filter is acceptable; both are
# provided by libass and appear/disappear together in practice.
SUBTITLE_FILTERS = ("ass", "subtitles")
REQUIRED_FILTERS = ("crop", "scale", "loudnorm", "sendcmd")
REQUIRED_ENCODERS = ("libx264", "aac")


class FFmpegNotFoundError(RuntimeError):
    """Neither the configured path nor PATH yielded a usable binary."""


class FFmpegCapabilityError(RuntimeError):
    """The resolved FFmpeg build lacks a capability the render requires."""


class SubtitleBurnUnsupportedError(FFmpegCapabilityError):
    """Subtitles were requested but this FFmpeg cannot burn them.

    Raised instead of quietly producing a subtitle-less file. A render that was
    asked for subtitles and cannot deliver them is a failed render.
    """


@dataclass(frozen=True)
class FFmpegCapabilities:
    ffmpeg_path: str
    ffprobe_path: Optional[str]
    version: str
    filters: frozenset[str] = field(default_factory=frozenset)
    encoders: frozenset[str] = field(default_factory=frozenset)

    @property
    def subtitle_filter(self) -> Optional[str]:
        """The filter to use for burning ASS, or None when unsupported."""
        for name in SUBTITLE_FILTERS:
            if name in self.filters:
                return name
        return None

    @property
    def supports_subtitle_burn(self) -> bool:
        return self.subtitle_filter is not None

    def missing_filters(self) -> tuple[str, ...]:
        return tuple(f for f in REQUIRED_FILTERS if f not in self.filters)

    def missing_encoders(self) -> tuple[str, ...]:
        return tuple(e for e in REQUIRED_ENCODERS if e not in self.encoders)


def _resolve_binary(name: str, configured: Optional[str]) -> Optional[str]:
    if configured:
        p = Path(configured).expanduser()
        if p.is_dir():
            cand = p / name
            if cand.is_file():
                return str(cand)
            return None
        if p.is_file():
            return str(p)
        # allow a bare command name to be resolved through PATH
        found = shutil.which(configured)
        if found:
            return found
        return None
    return shutil.which(name)


def resolve_ffmpeg(settings: object | None = None) -> str:
    configured = getattr(settings, "ffmpeg_path", None) if settings is not None else None
    path = _resolve_binary("ffmpeg", configured)
    if not path:
        raise FFmpegNotFoundError(
            "ffmpeg was not found. Install it, put it on PATH, or set "
            "FREECHER_FFMPEG_PATH to the binary (or the directory containing it)."
            + (f" Configured value was: {configured!r}" if configured else "")
        )
    return path


def resolve_ffprobe(settings: object | None = None) -> Optional[str]:
    configured = getattr(settings, "ffprobe_path", None) if settings is not None else None
    path = _resolve_binary("ffprobe", configured)
    if path:
        return path
    # ffprobe usually sits next to ffmpeg; try that before giving up.
    try:
        sibling = Path(resolve_ffmpeg(settings)).parent / "ffprobe"
    except FFmpegNotFoundError:
        return None
    return str(sibling) if sibling.is_file() else None


#: Characters that can appear in the flags column of `-filters` / `-encoders`.
#: Anything else in that position means the line is not a listing entry.
_FLAG_CHARS = frozenset(".|TSCVANXBFDIL")


def _list_section(ffmpeg: str, flag: str, timeout: float = 15.0) -> frozenset[str]:
    """Parse `ffmpeg -filters` / `-encoders` into a set of names.

    Both listings are a header, a legend, and then one entry per line with the
    name in the second whitespace-separated column.

    This used to skip everything before a `------` separator line. FFmpeg 7 and
    later emit one; FFmpeg 6.1 -- which is what Ubuntu 24.04 ships, and so what
    the production ARM host runs -- does not. The result was an empty filter set
    on the deployment target, reported as "this build has no crop/scale/libass"
    when in fact it has all of them. Recognising entries by their own shape
    works on both, and does not depend on a cosmetic line that upstream is free
    to change again.
    """
    try:
        res = subprocess.run(
            [ffmpeg, "-nostdin", "-hide_banner", flag],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("[ffmpeg] could not run %s %s: %s", ffmpeg, flag, exc)
        return frozenset()

    names: set[str] = set()
    for line in (res.stdout or "").splitlines():
        if not line[:1].isspace():
            continue                       # "Filters:" / "Encoders:" section title
        parts = line.split()
        if len(parts) < 3:
            continue                       # the separator, and blank lines
        flags, name = parts[0], parts[1]
        if name == "=":
            continue                       # legend, e.g. "  T.. = Timeline support"
        if set(flags) - _FLAG_CHARS:
            continue
        names.add(name)
    return frozenset(names)


_CACHE: dict[tuple[str, str], FFmpegCapabilities] = {}


def probe_capabilities(settings: object | None = None, *, refresh: bool = False) -> FFmpegCapabilities:
    """Resolve the binaries and probe what this build can do. Cached per path pair."""
    ffmpeg = resolve_ffmpeg(settings)
    ffprobe = resolve_ffprobe(settings)
    key = (ffmpeg, ffprobe or "")
    if not refresh and key in _CACHE:
        return _CACHE[key]

    version = "unknown"
    try:
        res = subprocess.run(
            [ffmpeg, "-nostdin", "-version"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=15.0, check=False,
        )
        if res.stdout:
            version = res.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass

    caps = FFmpegCapabilities(
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
        version=version,
        filters=_list_section(ffmpeg, "-filters"),
        encoders=_list_section(ffmpeg, "-encoders"),
    )
    _CACHE[key] = caps
    return caps


def clear_capability_cache() -> None:
    _CACHE.clear()


def require_subtitle_burn(settings: object | None = None) -> str:
    """Return the usable subtitle filter, or raise with an actionable message."""
    caps = probe_capabilities(settings)
    name = caps.subtitle_filter
    if name:
        return name
    raise SubtitleBurnUnsupportedError(
        "Subtitles were requested but this FFmpeg build cannot burn them: neither "
        f"the 'ass' nor the 'subtitles' filter is available in {caps.ffmpeg_path} "
        f"({caps.version}). This build lacks libass.\n"
        "Fix one of:\n"
        "  * install an FFmpeg built with --enable-libass "
        "(macOS: `brew install ffmpeg-full`), then\n"
        "  * point Freecher at it: FREECHER_FFMPEG_PATH=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg\n"
        "  * or re-run with subtitles disabled (--no-subtitles) if a silent clip is acceptable.\n"
        "Run `freecher-worker doctor` to see the resolved binary and its capabilities."
    )
