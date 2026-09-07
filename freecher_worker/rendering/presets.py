"""Rendering presets for short-form social video platforms."""

from __future__ import annotations

from typing import Dict
from pydantic import BaseModel, Field


class RenderPreset(BaseModel):
    """Configuration preset for rendering vertical short-form video."""

    name: str = Field(description="Preset name (e.g. shorts, tiktok, reels, original)")
    width: int = Field(default=1080, description="Target video width in pixels")
    height: int = Field(default=1920, description="Target video height in pixels")
    aspect_ratio: str = Field(default="9:16", description="Target aspect ratio")
    target_lufs: float = Field(default=-16.0, description="Target integrated loudness in LUFS")
    target_lra: float = Field(default=11.0, description="Target loudness range in LU")
    target_tp: float = Field(default=-1.5, description="Maximum true peak in dBFS")
    max_words_per_subtitle: int = Field(default=4, description="Max words visible per subtitle card")
    font_size: int = Field(default=54, description="Font size for ASS subtitles")
    active_word_highlight: bool = Field(default=True, description="Enable active word karaoke highlighting")
    smart_crop: bool = Field(default=True, description="Enable dynamic 9:16 smart crop")


PRESETS: Dict[str, RenderPreset] = {
    "shorts": RenderPreset(
        name="shorts",
        width=1080,
        height=1920,
        aspect_ratio="9:16",
        target_lufs=-16.0,
        font_size=54,
        max_words_per_subtitle=4,
        active_word_highlight=True,
        smart_crop=True,
    ),
    "tiktok": RenderPreset(
        name="tiktok",
        width=1080,
        height=1920,
        aspect_ratio="9:16",
        target_lufs=-16.0,
        font_size=56,
        max_words_per_subtitle=4,
        active_word_highlight=True,
        smart_crop=True,
    ),
    "reels": RenderPreset(
        name="reels",
        width=1080,
        height=1920,
        aspect_ratio="9:16",
        target_lufs=-16.0,
        font_size=54,
        max_words_per_subtitle=4,
        active_word_highlight=True,
        smart_crop=True,
    ),
    "original": RenderPreset(
        name="original",
        width=1920,
        height=1080,
        aspect_ratio="16:9",
        target_lufs=-16.0,
        font_size=48,
        max_words_per_subtitle=6,
        active_word_highlight=True,
        smart_crop=False,
    ),
}


def get_preset(name: str = "shorts") -> RenderPreset:
    """Retrieve render preset by name, defaulting to 'shorts'."""
    return PRESETS.get(name.lower(), PRESETS["shorts"])


AVAILABLE_PRESETS = PRESETS

