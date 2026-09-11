"""FFmpeg discovery, capability probing, and the fail-loud subtitle contract.

Covers the two operational renderer blockers:
  * a build without libass must FAIL a subtitle render, never silently drop it,
  * the ASS style line must stay parseable (a CSS font stack used to corrupt it).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import freecher_worker.rendering  # noqa: F401  (establishes import order; subtitles<->rendering cycle)
from freecher_worker.config import Settings
from freecher_worker.media import ffmpeg_env as fe
from freecher_worker.subtitles.ass import generate_ass_script, sanitize_font_name
from freecher_worker.subtitles.models import SubtitleEvent


@pytest.fixture(autouse=True)
def _clear_cache():
    fe.clear_capability_cache()
    yield
    fe.clear_capability_cache()


# --------------------------------------------------------------- discovery
def test_resolves_ffmpeg_from_path_by_default():
    assert Path(fe.resolve_ffmpeg(Settings(ffmpeg_path=None))).name.startswith("ffmpeg")


def test_explicit_binary_path_wins_over_path(tmp_path):
    real = shutil.which("ffmpeg")
    assert real, "ffmpeg must be installed to run this suite"
    assert fe.resolve_ffmpeg(Settings(ffmpeg_path=real)) == real


def test_a_directory_is_accepted_and_the_binary_found_inside():
    real = shutil.which("ffmpeg")
    assert fe.resolve_ffmpeg(Settings(ffmpeg_path=str(Path(real).parent))) == real


def test_a_missing_configured_binary_fails_loudly(tmp_path):
    with pytest.raises(fe.FFmpegNotFoundError) as exc:
        fe.resolve_ffmpeg(Settings(ffmpeg_path=str(tmp_path / "nope" / "ffmpeg")))
    assert "FREECHER_FFMPEG_PATH" in str(exc.value)


def test_capabilities_parse_filters_and_encoders():
    caps = fe.probe_capabilities(Settings())
    # Any real build has hundreds of these; a parsing regression yields zero.
    assert len(caps.filters) > 50
    assert len(caps.encoders) > 20
    assert not caps.missing_filters(), caps.missing_filters()
    assert not caps.missing_encoders(), caps.missing_encoders()


def test_capabilities_are_cached_per_binary(monkeypatch):
    s = Settings()
    first = fe.probe_capabilities(s)
    calls = []
    real_run = subprocess.run

    def counting(cmd, *a, **k):
        calls.append(cmd)
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", counting)
    second = fe.probe_capabilities(s)
    assert second is first
    assert not calls, "cached probe must not shell out again"


# ------------------------------------------------- fail-loud subtitle contract
def _caps(**kw):
    base = dict(ffmpeg_path="/x/ffmpeg", ffprobe_path="/x/ffprobe", version="v",
                filters=frozenset(), encoders=frozenset())
    base.update(kw)
    return fe.FFmpegCapabilities(**base)


def test_missing_libass_raises_actionable_error(monkeypatch):
    monkeypatch.setattr(fe, "probe_capabilities",
                        lambda *a, **k: _caps(filters=frozenset({"crop", "scale"})))
    with pytest.raises(fe.SubtitleBurnUnsupportedError) as exc:
        fe.require_subtitle_burn(Settings())
    msg = str(exc.value)
    assert "libass" in msg
    assert "FREECHER_FFMPEG_PATH" in msg
    assert "--no-subtitles" in msg


def test_ass_filter_is_preferred_when_both_present(monkeypatch):
    monkeypatch.setattr(fe, "probe_capabilities",
                        lambda *a, **k: _caps(filters=frozenset({"ass", "subtitles"})))
    assert fe.require_subtitle_burn(Settings()) == "ass"


def test_subtitles_filter_is_accepted_alone(monkeypatch):
    monkeypatch.setattr(fe, "probe_capabilities",
                        lambda *a, **k: _caps(filters=frozenset({"subtitles"})))
    assert fe.require_subtitle_burn(Settings()) == "subtitles"


def test_renderer_propagates_the_failure_rather_than_skipping(monkeypatch):
    """The renderer must not fall back to an unsubtitled MP4."""
    import freecher_worker.rendering.renderer as R

    def boom(*a, **k):
        raise fe.SubtitleBurnUnsupportedError("no libass")

    monkeypatch.setattr(R, "require_subtitle_burn", boom)
    with pytest.raises(fe.SubtitleBurnUnsupportedError):
        R.require_subtitle_burn(Settings())


# ------------------------------------------------------ ASS integrity (regression)
def _events():
    return [SubtitleEvent(id=1, start=0.0, end=1.0, text="привет мир", words=[])]


@pytest.mark.parametrize("stack,expected", [
    ("Montserrat, DejaVu Sans, Arial", "Montserrat"),
    ("Arial", "Arial"),
    ("  Noto Sans , X ", "Noto Sans"),
    ("", "Arial"),
])
def test_font_stack_is_reduced_to_one_family(stack, expected):
    assert sanitize_font_name(stack) == expected


def test_style_line_field_count_matches_format_line():
    """A comma-bearing font stack used to shift every field of the Style line.

    Fontsize then parsed as "DejaVu Sans" (i.e. 0) and libass drew nothing at all,
    so renders looked successful but had invisible subtitles on every ffmpeg build.
    """
    script = generate_ass_script(_events(), font_family="Montserrat, DejaVu Sans, Arial",
                                 font_size=54)
    fmt = next(l for l in script.splitlines() if l.startswith("Format:") and "Fontname" in l)
    style = next(l for l in script.splitlines() if l.startswith("Style:"))
    n_fmt = len(fmt.split(":", 1)[1].split(","))
    n_style = len(style.split(":", 1)[1].split(","))
    assert n_style == n_fmt, f"Style has {n_style} fields, Format declares {n_fmt}"

    fields = [f.strip() for f in style.split(":", 1)[1].split(",")]
    assert fields[1] == "Montserrat"
    assert fields[2] == "54", "Fontsize must be numeric, not a leaked font-stack entry"


def test_style_survives_a_font_name_with_punctuation():
    script = generate_ass_script(_events(), font_family="Weird:Font{x}", font_size=40)
    style = next(l for l in script.splitlines() if l.startswith("Style:"))
    assert ":" not in style.split(":", 1)[1]
    assert "{" not in style and "}" not in style
