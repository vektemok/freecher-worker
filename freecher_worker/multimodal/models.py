"""Data models for Multimodal Highlight Reranker v1."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AudioFeatures(BaseModel):
    """Cheap, locally computed audio energy and activity features for a candidate window."""

    rms_mean: float = Field(description="Mean RMS energy across 50ms windows")
    rms_std: float = Field(description="Standard deviation of windowed RMS energy")
    peak: float = Field(description="Peak absolute amplitude normalized to [0, 1]")
    silence_ratio: float = Field(description="Fraction of 50ms windows below silence threshold")
    speech_coverage: float = Field(description="Fraction of windows with detected speech (1 - silence_ratio)")
    energy_change_rate: float = Field(description="Mean absolute frame-to-frame delta of RMS energy")
    energy_percentile: Optional[float] = Field(
        default=None,
        description="Candidate mean RMS percentile relative to whole-source audio distribution [0, 1]",
    )
    beginning_rms: Optional[float] = Field(default=None, description="RMS of the first third of candidate")
    middle_rms: Optional[float] = Field(default=None, description="RMS of the middle third of candidate")
    ending_rms: Optional[float] = Field(default=None, description="RMS of the final third of candidate")


class VisualFeatures(BaseModel):
    """Cheap, locally computed visual change and presence features for a candidate window."""

    motion_score: float = Field(description="Mean normalized pixel difference between consecutive frames [0, 1]")
    scene_change_count: int = Field(description="Number of detected scene changes between sampled frames")
    face_presence_ratio: float = Field(description="Fraction of frames containing at least one detected face")
    person_presence_ratio: Optional[float] = Field(
        default=None,
        description="Fraction of frames containing a detected person (None if person detector unavailable)",
    )
    decoded_frame_count: int = Field(description="Count of successfully decoded frames")
    requested_frame_count: int = Field(description="Count of requested frames")
    failed_frame_count: int = Field(default=0, description="Count of frame extraction failures")
    decoder_used: str = Field(default="software", description="FFmpeg decoder used for frame extraction")


class ExtractedFrame(BaseModel):
    """Metadata for a single extracted candidate video frame."""

    timestamp_offset: float = Field(description="Timestamp in seconds relative to candidate start")
    absolute_timestamp: float = Field(description="Absolute timestamp in seconds in the source video")
    image_path: str = Field(description="Path to stored JPEG frame")
    width: int = Field(description="Frame width in pixels")
    height: int = Field(description="Frame height in pixels")
    source_type: str = Field(default="uniform", description="Extraction strategy ('uniform' or 'scene_change')")


class SourceTemporalActivityPoint(BaseModel):
    """1-second binned source-wide activity observation."""

    absolute_timestamp: float = Field(description="Timestamp in seconds from video start")
    audio_energy: float = Field(description="Normalized RMS audio energy [0, 1]")
    audio_delta: float = Field(description="Normalized frame-to-frame RMS delta [0, 1]")
    speech_activity: float = Field(description="Fraction of active speech windows [0, 1]")
    visual_motion: float = Field(description="Normalized frame-to-frame visual motion [0, 1]")
    scene_change: bool = Field(description="Whether a visual scene change was detected")
    combined_activity: float = Field(description="Deterministic weighted combined activity score [0, 1]")


class SourceTemporalActivityProfile(BaseModel):
    """Cached whole-source temporal activity profile computed in a single pass."""

    source_fingerprint: str
    formula_version: str = "activity_v1_1_formula_v1"
    bin_size_seconds: float = 1.0
    duration_seconds: float
    timeline: List[SourceTemporalActivityPoint] = Field(default_factory=list)


class ActivityPoint(BaseModel):
    """Candidate-relative 1-second activity curve point."""

    offset: float = Field(description="Timestamp in seconds relative to candidate start")
    absolute_timestamp: float = Field(description="Absolute timestamp in seconds in source video")
    audio_energy: float = Field(description="Normalized audio energy [0, 1]")
    audio_delta: float = Field(description="Normalized audio energy delta [0, 1]")
    speech_activity: float = Field(description="Speech activity ratio [0, 1]")
    visual_motion: float = Field(description="Visual motion difference [0, 1]")
    scene_change: bool = Field(description="Whether a scene cut occurred in this bin")
    combined_activity: float = Field(description="Combined activity signal [0, 1]")


class ActivityCurveSummary(BaseModel):
    """Summarized candidate-local activity curve and top activity peaks."""

    curve: List[ActivityPoint] = Field(default_factory=list)
    top_audio_peaks: List[float] = Field(default_factory=list)
    top_motion_peaks: List[float] = Field(default_factory=list)
    top_combined_activity_peaks: List[float] = Field(default_factory=list)


class TemporalBurst(BaseModel):
    """A localized ~2-second temporal burst with aligned visual frames and transcript."""

    burst_index: int = Field(description="Burst sequence index (1 or 2)")
    center_offset: float = Field(description="Center offset in seconds relative to candidate start")
    start_offset: float = Field(description="Burst start offset in seconds relative to candidate start")
    end_offset: float = Field(description="Burst end offset in seconds relative to candidate start")
    selection_reason: str = Field(description="Provenance reason (e.g. combined_activity_peak, audio_delta_peak)")
    combined_activity: float = Field(description="Combined activity score at burst center")
    activity_rank: int = Field(description="Activity rank among candidate peaks (1=strongest)")
    transcript: str = Field(default="", description="Spoken transcript aligned to [start_offset - 0.75s, end_offset + 0.75s]")
    frames: List[ExtractedFrame] = Field(default_factory=list, description="Extracted frames covering the ~2s burst")
    audio_energy_mean: float = Field(default=0.0, description="Mean audio energy during burst")
    motion_mean: float = Field(default=0.0, description="Mean visual motion during burst")
    has_scene_change: bool = Field(default=False, description="Whether burst contains a scene cut")


class MultimodalCandidatePackage(BaseModel):
    """Self-contained multimodal evidence package for an individual candidate highlight window."""

    candidate_id: str = Field(description="Candidate identifier matching CandidateWindow")
    start: float = Field(description="Candidate start time in seconds")
    end: float = Field(description="Candidate end time in seconds")
    duration: float = Field(description="Candidate duration in seconds")

    candidate_transcript: str = Field(description="Transcribed speech of the candidate")
    previous_context: str = Field(default="", description="Preceding speech context (up to 45s)")
    next_context: str = Field(default="", description="Following speech context (up to 45s)")

    frames: List[ExtractedFrame] = Field(default_factory=list, description="Extracted downscaled JPEG frames")
    audio_features: AudioFeatures = Field(description="Locally computed audio energy signals")
    visual_features: VisualFeatures = Field(description="Locally computed visual activity signals")

    source_fingerprint: str = Field(description="Fingerprint of the source media file")
    candidate_set_id: str = Field(description="Immutable candidate set ID")
    package_hash: str = Field(description="Deterministic hash of extraction inputs and candidate boundaries")
    insufficient_visual_evidence: bool = Field(
        default=False,
        description="True if fewer than 4 frames could be decoded",
    )

    package_version: str = Field(default="multimodal_package_v1", description="Package schema version")
    temporal_bursts: Optional[List[TemporalBurst]] = Field(
        default=None,
        description="Localized 2-second activity bursts with aligned transcript (v1.1)",
    )
    activity_curve: Optional[ActivityCurveSummary] = Field(
        default=None,
        description="Candidate-local 1-second activity curve (v1.1)",
    )


class ObservedRegion(BaseModel):
    """Advisory sub-region identified by multimodal model as the strongest span."""

    start_offset: float = Field(description="Start offset in seconds relative to candidate start")
    end_offset: float = Field(description="End offset in seconds relative to candidate start")
    confidence: Optional[float] = Field(default=None, description="Confidence in best observed region [0, 1]")
    reason: Optional[str] = Field(default=None, description="Reason why this span was selected")


class ObservedEvidenceItem(BaseModel):
    """A concrete visual or audio moment noted by the multimodal model."""

    timestamp_offset: float = Field(description="Timestamp in seconds relative to candidate start")
    description: str = Field(description="Brief observation of visible action or audio event")


class MultimodalModelResult(BaseModel):
    """Structured response from multimodal LLM provider for a candidate package."""

    candidate_id: str

    observable_event: bool = Field(description="Whether a clear, distinct visual/audio event is observable")
    visual_payoff: bool = Field(description="Whether a visual climax, reaction, or resolution is contained")

    visual_event: float = Field(ge=0, le=100, description="Visual action dynamism 0-100")
    reaction: float = Field(ge=0, le=100, description="Facial/emotional expression intensity 0-100")
    emotion: float = Field(ge=0, le=100, description="Emotional resonance 0-100")
    humor: float = Field(ge=0, le=100, description="Comedic value 0-100")
    surprise: float = Field(ge=0, le=100, description="Unpredictability / twist 0-100")
    energy: float = Field(ge=0, le=100, description="Spoken and visual pacing energy 0-100")
    standalone: float = Field(ge=0, le=100, description="Comprehensibility without prior stream lore 0-100")
    retention: float = Field(ge=0, le=100, description="Short-form retention pull 0-100")
    shareability: float = Field(ge=0, le=100, description="Viral shareability 0-100")

    boringness: float = Field(ge=0, le=100, description="Monotony / lack of eventfulness 0-100")
    context_dependency: float = Field(ge=0, le=100, description="Reliance on outside stream lore 0-100")

    outside_payoff: bool = Field(default=False, description="True if resolution occurs outside candidate in NEXT CONTEXT")
    missing_setup: bool = Field(default=False, description="True if candidate begins after key event started")
    insufficient_visual_evidence: bool = Field(default=False, description="True if visual frames were insufficient")

    confidence: float = Field(ge=0.0, le=1.0, description="Model self-assessed confidence [0, 1]")

    best_observed_region: Optional[ObservedRegion] = Field(
        default=None,
        description="Advisory span of peak interest for downstream refinement",
    )
    evidence: List[ObservedEvidenceItem] = Field(
        default_factory=list,
        description="Observable moments tied to frame timestamps",
    )
    reason: str = Field(description="Concise editorial rationale")
    quality_score: float = Field(ge=0, le=100, description="Overall candidate short-form quality rating 0-100")


class ShortlistItem(BaseModel):
    """Detailed metadata for a candidate in the retrieval shortlist."""

    candidate_id: str
    heuristic_rank: Optional[int] = Field(default=None, description="1-based rank in heuristic retrieval, or null if absent")
    llm_rank: Optional[int] = Field(default=None, description="1-based rank in LLM retrieval, or null if absent")
    best_rank: int = Field(description="Minimum rank among retrieval sources")
    sum_rank: int = Field(description="Sum of source ranks (using top_k + 1 for absent sources)")
    retrieval_sources: List[str] = Field(default_factory=list, description="Retrieval sources that nominated this candidate")


class ShortlistDocument(BaseModel):
    """Deterministic candidate shortlist metadata for multimodal reranking."""

    candidate_set_id: str
    strategy: str = Field(default="union", description="Retrieval strategy name")
    heuristic_top_k: int
    llm_top_k: int
    max_candidates: int
    candidate_ids: List[str]
    total_unique: int
    items: Optional[List[ShortlistItem]] = Field(default=None, description="Detailed shortlist items (v1.1)")


class MultimodalUsage(BaseModel):
    """API token consumption and estimated cost tracking."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: Optional[float] = None


class SourceAudioProfile(BaseModel):
    """Cached whole-source audio energy distribution used for relative percentile ranking."""

    source_fingerprint: str
    sample_rate: int = 16000
    duration_seconds: float
    rms_percentiles: Dict[str, float] = Field(
        default_factory=dict,
        description="Precomputed percentiles (e.g. 'p10', 'p25', 'p50', 'p75', 'p90') for RMS",
    )
