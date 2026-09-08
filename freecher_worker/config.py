"""Configuration management for freecher-worker using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings populated from environment variables with FREECHER_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="FREECHER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ASR Settings
    asr_model: str = Field(default="small", description="Whisper model name (small, medium, turbo, etc.)")
    asr_device: str = Field(default="cuda", description="Device for Whisper inference (cuda or cpu)")
    asr_compute_type: str = Field(default="int8_float16", description="Compute type for CTranslate2")
    asr_beam_size: int = Field(default=5, description="Beam size for Whisper decoding")
    asr_vad_filter: bool = Field(default=True, description="Enable Silero VAD filter in Whisper")
    asr_language: Optional[str] = Field(default=None, description="Audio language code (e.g. 'ru'). None for auto")

    # Highlight & Window Settings
    highlight_min_seconds: float = Field(default=30.0, description="Minimum candidate highlight duration in seconds")
    highlight_target_seconds: float = Field(default=60.0, description="Target candidate highlight duration in seconds")
    highlight_max_seconds: float = Field(default=90.0, description="Maximum candidate highlight duration in seconds")
    highlight_overlap_seconds: float = Field(default=15.0, description="Target overlap between candidate windows")
    highlight_top_k: int = Field(default=5, description="Number of top highlights to select and clip")
    dedup_overlap_threshold: float = Field(default=0.60, description="Temporal overlap threshold for deduplication")
    clip_padding_seconds: float = Field(default=2.0, description="Contextual padding before/after highlight in seconds")

    # Scorer Settings
    scorer: str = Field(default="heuristic", description="Scorer implementation ('heuristic' or 'llm' or 'highlight_v2')")

    # Optional OpenAI-compatible LLM Scorer
    llm_base_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("FREECHER_LLM_BASE_URL", "ARNY_LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_BASE_URL"),
        description="Base URL for OpenAI-compatible LLM API",
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("FREECHER_LLM_API_KEY", "ARNY_LLM_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"),
        description="API key for OpenAI-compatible LLM API",
    )
    llm_model: Optional[str] = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("FREECHER_LLM_MODEL", "ARNY_LLM_MODEL", "OPENAI_MODEL", "LLM_MODEL"),
        description="Model name for LLM scoring",
    )

    # Paths
    output_dir: Path = Field(default=Path("runs"), description="Base directory for run outputs")

    # Phase 2 — Boundary Refinement Settings
    boundary_max_shift_seconds: float = Field(default=5.0, description="Max seconds to shift highlight boundaries")
    boundary_context_before: float = Field(default=0.5, description="Context seconds preserved before phrase start")
    boundary_context_after: float = Field(default=0.5, description="Context seconds preserved after phrase end")

    # Phase 2 — Refined ASR with Word Timestamps
    refinement_asr_model: str = Field(default="medium", description="ASR model for refined word timestamping")
    refinement_asr_compute_type: str = Field(default="int8", description="Compute type for refined ASR")

    # Phase 2 — Subtitles & ASS Styling
    subtitle_font: str = Field(default="Montserrat, DejaVu Sans, Arial", description="Font family for subtitles")
    subtitle_font_size: int = Field(default=54, description="Font size for vertical video ASS subtitles")
    subtitle_max_words: int = Field(default=4, description="Maximum words grouped per subtitle event")
    subtitle_active_word: bool = Field(default=True, description="Enable active spoken word karaoke pop effect")

    # Phase 2 — Smart Crop 9:16 Settings
    crop_analysis_fps: float = Field(default=2.0, description="Frame sampling rate (FPS) for subject detection")
    crop_deadzone_ratio: float = Field(default=0.03, description="Dead-zone ratio of width to suppress jitter")
    crop_max_velocity_pixels_per_sec: float = Field(default=200.0, description="Maximum pan velocity in pixels/second")

    # Phase 2 — Audio Loudness Normalization Settings (EBU R128)
    audio_normalize_loudness: bool = Field(default=True, description="Enable two-pass loudness normalization")
    audio_target_i: float = Field(default=-16.0, description="Target integrated loudness in LUFS")
    audio_target_lra: float = Field(default=11.0, description="Target loudness range in LU")
    audio_target_tp: float = Field(default=-1.5, description="Maximum true peak in dBFS")

    # Phase 2 — Render Presets
    render_preset: str = Field(default="shorts", description="Render preset (shorts, tiktok, reels, original)")

    # Optional External Services
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN", description="HuggingFace token if needed")


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
