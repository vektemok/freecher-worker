"""Freecher Multimodal Highlight Reranker (multimodal_v1)."""

from .activity import (
    ACTIVITY_V1_1_FORMULA_VERSION,
    compute_combined_activity,
    compute_source_temporal_activity_profile,
    select_temporal_burst_peaks,
    slice_candidate_activity_curve,
)
from .audio_features import compute_source_audio_profile, extract_candidate_audio_features
from .frames import compute_v1_1_sample_timestamps, extract_candidate_frames, probe_software_decoder
from .models import (
    ActivityCurveSummary,
    ActivityPoint,
    AudioFeatures,
    ExtractedFrame,
    MultimodalCandidatePackage,
    MultimodalModelResult,
    MultimodalUsage,
    ObservedEvidenceItem,
    ObservedRegion,
    ShortlistDocument,
    ShortlistItem,
    SourceAudioProfile,
    SourceTemporalActivityPoint,
    SourceTemporalActivityProfile,
    TemporalBurst,
    VisualFeatures,
)
from .openai_provider import OpenAIMultimodalProvider
from .package import PACKAGE_VERSION_V1, PACKAGE_VERSION_V1_1, build_multimodal_package
from .provider import PROMPT_VERSION_MULTIMODAL_V1, PROMPT_VERSION_MULTIMODAL_V1_1, MultimodalProvider
from .scorer import (
    FORMULA_VERSION_MULTIMODAL_V1,
    FORMULA_VERSION_MULTIMODAL_V1_1,
    SCORER_VERSION_MULTIMODAL_V1,
    SCORER_VERSION_MULTIMODAL_V1_1,
    MultimodalReranker,
    extract_canonical_fingerprint,
    multimodal_v1_formula_v1,
    multimodal_v1_1_formula_v1,
    resolve_source_video_path,
)
from .shortlist import generate_shortlist
from .visual_features import extract_candidate_visual_features

__all__ = [
    "ACTIVITY_V1_1_FORMULA_VERSION",
    "ActivityCurveSummary",
    "ActivityPoint",
    "AudioFeatures",
    "ExtractedFrame",
    "FORMULA_VERSION_MULTIMODAL_V1",
    "FORMULA_VERSION_MULTIMODAL_V1_1",
    "MultimodalCandidatePackage",
    "MultimodalModelResult",
    "MultimodalProvider",
    "MultimodalReranker",
    "MultimodalUsage",
    "ObservedEvidenceItem",
    "ObservedRegion",
    "OpenAIMultimodalProvider",
    "PACKAGE_VERSION_V1",
    "PACKAGE_VERSION_V1_1",
    "PROMPT_VERSION_MULTIMODAL_V1",
    "PROMPT_VERSION_MULTIMODAL_V1_1",
    "SCORER_VERSION_MULTIMODAL_V1",
    "SCORER_VERSION_MULTIMODAL_V1_1",
    "ShortlistDocument",
    "ShortlistItem",
    "SourceAudioProfile",
    "SourceTemporalActivityPoint",
    "SourceTemporalActivityProfile",
    "TemporalBurst",
    "VisualFeatures",
    "build_multimodal_package",
    "compute_combined_activity",
    "compute_source_audio_profile",
    "compute_source_temporal_activity_profile",
    "compute_v1_1_sample_timestamps",
    "extract_candidate_audio_features",
    "extract_candidate_frames",
    "extract_candidate_visual_features",
    "extract_canonical_fingerprint",
    "generate_shortlist",
    "multimodal_v1_formula_v1",
    "multimodal_v1_1_formula_v1",
    "probe_software_decoder",
    "resolve_source_video_path",
    "select_temporal_burst_peaks",
    "slice_candidate_activity_curve",
]

