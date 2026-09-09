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

    # Contextual Highlight Intelligence (contextual_reranker_v1_1)
    contextual_base_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "FREECHER_CONTEXTUAL_BASE_URL", "FREECHER_LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_BASE_URL"
        ),
        description="Base URL for the contextual reranker LLM API",
    )
    contextual_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "FREECHER_CONTEXTUAL_API_KEY", "FREECHER_LLM_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"
        ),
        description="API key for the contextual reranker LLM API",
    )
    contextual_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices(
            "FREECHER_CONTEXTUAL_MODEL", "CONTEXTUAL_MODEL", "FREECHER_MULTIMODAL_MODEL", "FREECHER_LLM_MODEL"
        ),
        description="Model used for contextual reranking stages",
    )
    contextual_reasoning_effort: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "FREECHER_CONTEXTUAL_REASONING_EFFORT",
            "CONTEXTUAL_REASONING_EFFORT",
            "FREECHER_MULTIMODAL_REASONING_EFFORT",
        ),
        description="Reasoning effort for the contextual reranker (none|low|medium|high|xhigh|max)",
    )
    contextual_temperature: float = Field(default=0.1, description="Sampling temperature for contextual stages")
    contextual_input_scorer: str = Field(
        default="multimodal_v1_1",
        description="Upstream scorer whose candidate set the contextual reranker reranks",
    )
    contextual_comparison_mode: str = Field(
        default="full",
        description="Comparative ranking mode: full | swiss | listwise | none",
    )
    contextual_before_seconds: float = Field(default=75.0, description="BEFORE context window (understanding only)")
    contextual_after_seconds: float = Field(default=25.0, description="AFTER context window (understanding only)")
    contextual_chapter_target_seconds: float = Field(default=240.0, description="Target chapter length in seconds")
    contextual_chapter_min_seconds: float = Field(default=150.0, description="Shortest allowed chapter in seconds")
    contextual_chapter_max_seconds: float = Field(default=420.0, description="Longest allowed chapter in seconds")
    contextual_listwise_batch_size: int = Field(default=7, description="Candidates compared per listwise request")
    contextual_final_pairwise_top: int = Field(default=4, description="Top group refined with round-robin pairwise")
    contextual_top: int = Field(default=5, description="Number of top highlights reported after reranking")

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
    reframe_detector: str = Field(
        default="auto",
        description="Subject detector: auto (YuNet, then legacy) | yunet | haar | center",
    )
    reframe_detect_max_width: int = Field(default=960, description="Frames downscaled to this width before detection")
    reframe_face_score_threshold: float = Field(default=0.6, description="Minimum YuNet face confidence")
    reframe_face_model_path: Optional[Path] = Field(
        default=None, description="Explicit path to the YuNet ONNX weights (or their directory)"
    )
    reframe_allow_model_download: bool = Field(
        default=True, description="Allow downloading detector weights into the local model cache"
    )
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
    reframe_track_max_misses: int = Field(default=6, description="Missed samples before the crop stops holding a lost subject's framing")
    reframe_track_max_gap_sec: float = Field(default=1.6, description="How long a subject may go undetected and still recover its identity")
    reframe_track_confirm_hits: int = Field(default=3, description="Detections before a candidate track counts as a real identity")
    reframe_track_max_reassociation_px: float = Field(default=520.0, description="Ceiling on how far a recovered identity may jump")
    reframe_track_appearance: bool = Field(default=True, description="Use a tiny appearance patch to confirm re-attachment after a gap")
    reframe_track_appearance_min_similarity: float = Field(default=0.15, description="Minimum appearance correlation to reattach across a gap")
    reframe_track_appearance_weight: float = Field(default=0.25, description="Share of the association cost driven by appearance")
    reframe_track_scene_cut_reset: bool = Field(default=True, description="A scene cut resets motion prediction and drops already-lost tracks")
    reframe_scene_cut_threshold: float = Field(default=0.35, description="Normalized frame difference treated as a scene cut")
    reframe_dual_subject_balance: float = Field(default=0.35, description="Max relative weight gap for dual-subject framing")
    reframe_jitter_epsilon_px: float = Field(default=2.0, description="Emitted crop movements below this are suppressed")

    # Phase 3 — Adaptive Vertical Layout (presentation layer only; never touches ranking)
    layout_mode: str = Field(
        default="single",
        description="Vertical layout strategy: single | adaptive | full-frame",
    )
    layout_window_sec: float = Field(default=0.75, description="Length of one layout decision window")
    layout_min_duration_sec: float = Field(default=2.5, description="Shortest stretch a layout may hold")
    layout_switch_confirmation_sec: float = Field(default=1.5, description="Evidence required before switching layout")
    layout_subject_missing_grace_sec: float = Field(default=1.2, description="A briefly lost subject still counts as present")
    layout_switch_penalty: float = Field(default=0.08, description="Confidence margin a new layout must win by")
    layout_scene_cut_immediate_switch: bool = Field(default=True, description="A real scene cut may switch layout at once")
    layout_persistent_min_visible_sec: float = Field(default=1.0, description="Screen time before a track is a real subject")
    layout_persistent_min_visibility_ratio: float = Field(default=0.20, description="Share of the clip a persistent track is visible")
    layout_persistent_min_hits: int = Field(default=4, description="Detections required behind a persistent track")
    layout_significant_min_area_ratio: float = Field(default=0.0012, description="Minimum mean subject area as a fraction of the frame")
    layout_dominant_min_visibility: float = Field(default=0.65, description="Window visibility required of a dominant subject")
    layout_secondary_min_visibility: float = Field(default=0.45, description="Window visibility required of a second subject")
    layout_group_min_subjects: int = Field(default=3, description="Significant subjects that make a scene a group scene")
    layout_min_detection_coverage: float = Field(default=0.35, description="Below this, evidence is too thin to crop")
    layout_min_tracking_confidence: float = Field(default=0.45, description="Below this, evidence is too thin to crop")
    layout_min_face_height_ratio: float = Field(default=0.055, description="Smallest readable subject height in the output frame")
    layout_safe_top_ratio: float = Field(default=0.06, description="Top band reserved for platform UI")
    layout_safe_bottom_ratio: float = Field(default=0.24, description="Bottom band reserved for captions and platform UI")
    layout_full_frame_blur: bool = Field(default=True, description="Fill unused vertical space with a blurred copy of the frame")
    layout_full_frame_blur_sigma: float = Field(default=12.0, description="Blur strength behind a fitted full frame")
    layout_full_frame_background_color: str = Field(default="black", description="Background used when blur is disabled")

    # Phase 3 — Vertical Short Output (always 9:16, no other aspect ratio is supported)
    shorts_output_width: int = Field(default=1080, description="Final short width in pixels (fixed 9:16 output)")
    shorts_output_height: int = Field(default=1920, description="Final short height in pixels (fixed 9:16 output)")
    shorts_encoder: str = Field(default="auto", description="Encoder selection: auto | libx264 | h264_nvenc")
    shorts_x264_preset: str = Field(default="veryfast", description="libx264 preset for the final encode")
    shorts_x264_crf: int = Field(default=20, description="libx264 CRF quality for the final encode")

    # Streaming Ingest — Cloudflare R2 (S3-compatible)
    r2_bucket: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("FREECHER_R2_BUCKET", "R2_BUCKET"),
        description="R2 bucket that receives ingested source videos",
    )
    r2_endpoint: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("FREECHER_R2_ENDPOINT", "R2_ENDPOINT", "R2_ENDPOINT_URL"),
        description="R2 S3 API endpoint, e.g. https://<account_id>.r2.cloudflarestorage.com",
    )
    r2_access_key_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("FREECHER_R2_ACCESS_KEY_ID", "R2_ACCESS_KEY_ID"),
        description="R2 API token access key id",
    )
    r2_secret_access_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("FREECHER_R2_SECRET_ACCESS_KEY", "R2_SECRET_ACCESS_KEY"),
        description="R2 API token secret access key",
    )
    r2_public_base_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("FREECHER_R2_PUBLIC_BASE_URL", "R2_PUBLIC_BASE_URL"),
        description="Optional public/custom domain used to build a readable object URL",
    )
    ingest_quality: str = Field(default="best", description="Ingest quality mode: 'best' or 'progressive'")
    ingest_max_height: Optional[int] = Field(default=None, description="Optional cap on source video height in pixels")
    ingest_part_size_mb: int = Field(default=16, description="Multipart chunk size in MiB (minimum 5)")
    ingest_concurrency: int = Field(default=4, description="Parts uploaded to R2 in parallel")
    ingest_key_template: str = Field(
        default="input/{video_id}/source.mp4",
        description="Object key template; supports {video_id}, {project_id}, {date}",
    )
    # The audio artifact key is derived from the resolved video key
    # (input/<id>/source.mp4 -> processing/<id>/audio.m4a), so it has no
    # template of its own; only the encode is configurable.
    ingest_extract_audio: bool = Field(
        default=True,
        description="Also write a transcription-ready processing/{id}/audio.m4a during ingest",
    )
    ingest_audio_codec: str = Field(default="aac", description="Audio codec for the transcription artifact")
    ingest_audio_sample_rate: int = Field(default=16000, description="Audio sample rate in Hz (16 kHz is what ASR uses)")
    ingest_audio_channels: int = Field(default=1, description="Audio channel count; mono for speech")
    ingest_audio_bitrate: str = Field(default="64k", description="Audio bitrate for the transcription artifact")

    # R2-backed transcription (processing/{id}/audio.m4a -> transcript.json).
    # Separate from the asr_* defaults, which describe the local `process`
    # pipeline on modest hardware; this milestone targets a T4 with float16.
    transcribe_model: str = Field(default="large-v3", description="Whisper model for R2 transcription")
    transcribe_device: str = Field(default="cuda", description="Inference device for R2 transcription")
    transcribe_compute_type: str = Field(default="float16", description="CTranslate2 compute type on GPU")
    transcribe_beam_size: int = Field(default=5, description="Beam size for R2 transcription")
    transcribe_vad_filter: bool = Field(default=True, description="Enable Silero VAD for R2 transcription")
    transcribe_word_timestamps: bool = Field(default=True, description="Emit word-level timestamps")

    # Optional External Services
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN", description="HuggingFace token if needed")


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
