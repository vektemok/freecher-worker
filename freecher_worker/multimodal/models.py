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


class ObservedRegion(BaseModel):
    """Advisory sub-region identified by multimodal model as the strongest span."""

    start_offset: float = Field(description="Start offset in seconds relative to candidate start")
    end_offset: float = Field(description="End offset in seconds relative to candidate start")


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


class ShortlistDocument(BaseModel):
    """Deterministic candidate shortlist metadata for multimodal reranking."""

    candidate_set_id: str
    strategy: str = Field(default="union", description="Retrieval strategy name")
    heuristic_top_k: int
    llm_top_k: int
    max_candidates: int
    candidate_ids: List[str]
    total_unique: int


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
