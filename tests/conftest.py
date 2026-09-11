"""Shared test setup.

The renderer now FAILS a subtitle render when FFmpeg cannot burn ASS, instead of
silently emitting a subtitle-less MP4. Tests that render with subtitles therefore
need an FFmpeg built with libass.

If one is discoverable we point the whole session at it via FREECHER_FFMPEG_PATH.
If none exists on this machine, the affected tests are SKIPPED with an explicit
reason -- they are never quietly turned into passes.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# Places a libass-capable build commonly lives, checked in order. PATH first so a
# deliberately configured environment always wins.
_CANDIDATE_FFMPEGS = (
    os.environ.get("FREECHER_FFMPEG_PATH"),
    shutil.which("ffmpeg"),
    "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
    "/usr/local/opt/ffmpeg-full/bin/ffmpeg",
    "/opt/homebrew/bin/ffmpeg",
    "/usr/bin/ffmpeg",
)


def _burns_subtitles(binary: str | None) -> bool:
    if not binary:
        return False
    p = Path(binary)
    if p.is_dir():
        p = p / "ffmpeg"
    if not p.is_file():
        return False
    try:
        res = subprocess.run(
            [str(p), "-nostdin", "-h", "filter=ass"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return res.returncode == 0 and "Filter ass" in (res.stdout or "")


def _find_subtitle_capable_ffmpeg() -> str | None:
    for cand in _CANDIDATE_FFMPEGS:
        if _burns_subtitles(cand):
            p = Path(cand)
            return str(p / "ffmpeg" if p.is_dir() else p)
    return None


SUBTITLE_FFMPEG = _find_subtitle_capable_ffmpeg()


def pytest_configure(config):
    """Point the session at a libass build so subtitle renders exercise the real path."""
    if SUBTITLE_FFMPEG:
        os.environ["FREECHER_FFMPEG_PATH"] = SUBTITLE_FFMPEG
    config.addinivalue_line(
        "markers", "needs_libass: requires an FFmpeg built with libass (subtitle burn-in)"
    )


def pytest_report_header(config):
    if SUBTITLE_FFMPEG:
        return f"libass ffmpeg: {SUBTITLE_FFMPEG} (subtitle burn-in tests ENABLED)"
    return ("libass ffmpeg: NOT FOUND — subtitle burn-in tests will be SKIPPED. "
            "Install one (brew install ffmpeg-full) to exercise them.")


@pytest.fixture(scope="session")
def subtitle_ffmpeg() -> str:
    if not SUBTITLE_FFMPEG:
        pytest.skip("no FFmpeg with libass available; cannot burn subtitles")
    return SUBTITLE_FFMPEG


def pytest_collection_modifyitems(config, items):
    """Skip renders that need subtitle burning when no capable FFmpeg exists."""
    if SUBTITLE_FFMPEG:
        return
    skip = pytest.mark.skip(
        reason="needs an FFmpeg built with libass; renderer now fails rather than "
               "silently dropping subtitles"
    )
    for item in items:
        if "needs_libass" in item.keywords:
            item.add_marker(skip)
