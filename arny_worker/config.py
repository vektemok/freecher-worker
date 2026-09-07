"""Configuration management for arny-worker using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings populated from environment variables with ARNY_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="ARNY_",
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
    scorer: str = Field(default="heuristic", description="Scorer implementation ('heuristic' or 'llm')")

    # Optional OpenAI-compatible LLM Scorer
    llm_base_url: Optional[str] = Field(default=None, description="Base URL for OpenAI-compatible LLM API")
    llm_api_key: Optional[str] = Field(default=None, description="API key for OpenAI-compatible LLM API")
    llm_model: Optional[str] = Field(default="gpt-4o-mini", description="Model name for LLM scoring")

    # Paths
    output_dir: Path = Field(default=Path("runs"), description="Base directory for run outputs")

    # Optional External Services
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN", description="HuggingFace token if needed")


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
