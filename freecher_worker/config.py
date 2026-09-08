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

    # Multimodal Highlight Reranker Settings
    multimodal_base_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("FREECHER_MULTIMODAL_BASE_URL", "FREECHER_LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_BASE_URL"),
        description="Base URL for Multimodal LLM API",
    )
    multimodal_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("FREECHER_MULTIMODAL_API_KEY", "FREECHER_LLM_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"),
        description="API key for Multimodal LLM API",
    )
    multimodal_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("FREECHER_MULTIMODAL_MODEL", "MULTIMODAL_MODEL", "FREECHER_LLM_MODEL", "OPENAI_MODEL"),
        description="Vision-capable model name for multimodal reranking",
    )
    multimodal_heuristic_top_k: int = Field(default=12, description="Candidates taken from heuristic_v1 for shortlist")
    multimodal_llm_top_k: int = Field(default=12, description="Candidates taken from highlight_v2_1 for shortlist")
    multimodal_max_candidates: int = Field(default=20, description="Maximum capacity for multimodal candidate shortlist")
    multimodal_max_long_edge: int = Field(default=640, description="Max long edge dimension in pixels for extracted frames")
    multimodal_reasoning_effort: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "FREECHER_MULTIMODAL_REASONING_EFFORT",
            "MULTIMODAL_REASONING_EFFORT",
            "REASONING_EFFORT",
        ),
        description="Reasoning effort for reasoning models (none|low|medium|high|xhigh|max)",
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

    # Phase 3 — Dynamic Subclip Refinement (post-ranking production stage)
    subclip_min_duration_sec: float = Field(default=8.0, description="Hard minimum final short duration in seconds")
    subclip_target_min_duration_sec: float = Field(default=15.0, description="Preferred duration band lower bound")
    subclip_target_max_duration_sec: float = Field(default=30.0, description="Preferred duration band upper bound")
    subclip_max_duration_sec: float = Field(default=45.0, description="Hard maximum final short duration in seconds")
    subclip_duration_mode: str = Field(default="auto", description="Subclip duration mode ('auto' or 'full')")
    subclip_hook_window_sec: float = Field(default=3.0, description="Leading window measured for hook strength")
    subclip_tail_window_sec: float = Field(default=3.0, description="Trailing window measured for dead-air penalty")
    subclip_pre_roll_sec: float = Field(default=0.15, description="Lead-in kept before a phrase onset")
    subclip_post_roll_sec: float = Field(default=0.30, description="Tail kept after a phrase completes")
    subclip_boring_threshold: float = Field(default=0.25, description="Normalized activity below which a bin is 'boring'")
    subclip_strict_timestamps: bool = Field(default=True, description="Reject ambiguous advisory regions instead of guessing")
    subclip_use_advisory_region: bool = Field(default=True, description="Use multimodal best_observed_region as a soft prior")

    # Phase 3 — Smart 9:16 Reframing
    reframe_analysis_fps: float = Field(default=5.0, description="Frame sampling rate for detection and tracking")
    reframe_detector: str = Field(default="haar", description="Subject detector implementation name")
    reframe_detect_max_width: int = Field(default=640, description="Frames downscaled to this width before detection")
    reframe_subject_padding_ratio: float = Field(default=0.55, description="Padding around subject box as a ratio of its width")
    reframe_head_position_ratio: float = Field(default=0.38, description="Vertical position of the face inside the crop (0=top)")
    reframe_headroom_ratio: float = Field(default=0.45, description="Minimum headroom above the face as a ratio of face height")
    reframe_edge_margin_ratio: float = Field(default=0.06, description="Minimum margin between subject and crop edge")
    reframe_deadzone_ratio: float = Field(default=0.02, description="Dead-zone as a ratio of source width to suppress micro-jitter")
    reframe_smoothing_alpha: float = Field(default=0.22, description="Per-sample proportional gain toward the target center")
    reframe_max_velocity_px_per_sec: float = Field(default=160.0, description="Maximum crop pan velocity in px/s")
    reframe_max_acceleration_px_per_sec2: float = Field(default=420.0, description="Maximum crop pan acceleration in px/s^2")
    reframe_switch_hold_sec: float = Field(default=0.8, description="Sustained evidence required before switching subject")
    reframe_switch_margin: float = Field(default=0.25, description="Relative score margin required to switch subject")
    reframe_min_switch_interval_sec: float = Field(default=1.5, description="Minimum seconds between subject switches")
    reframe_track_max_misses: int = Field(default=6, description="Consecutive missed detections before a track is dropped")
    reframe_scene_cut_threshold: float = Field(default=0.35, description="Normalized frame difference treated as a scene cut")
    reframe_dual_subject_balance: float = Field(default=0.35, description="Max relative weight gap for dual-subject framing")
    reframe_jitter_epsilon_px: float = Field(default=2.0, description="Emitted crop movements below this are suppressed")

    # Phase 3 — Vertical Short Output (always 9:16, no other aspect ratio is supported)
    shorts_output_width: int = Field(default=1080, description="Final short width in pixels (fixed 9:16 output)")
    shorts_output_height: int = Field(default=1920, description="Final short height in pixels (fixed 9:16 output)")
    shorts_encoder: str = Field(default="auto", description="Encoder selection: auto | libx264 | h264_nvenc")
    shorts_x264_preset: str = Field(default="veryfast", description="libx264 preset for the final encode")
    shorts_x264_crf: int = Field(default=20, description="libx264 CRF quality for the final encode")

    # Optional External Services
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN", description="HuggingFace token if needed")


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
