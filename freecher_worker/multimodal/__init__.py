"""Freecher Multimodal Highlight Reranker (multimodal_v1)."""

from .audio_features import compute_source_audio_profile, extract_candidate_audio_features
from .frames import extract_candidate_frames, probe_software_decoder
from .models import (
    AudioFeatures,
    ExtractedFrame,
    MultimodalCandidatePackage,
    MultimodalModelResult,
    MultimodalUsage,
    ObservedEvidenceItem,
    ObservedRegion,
    ShortlistDocument,
    SourceAudioProfile,
    VisualFeatures,
)
from .openai_provider import OpenAIMultimodalProvider
from .package import build_multimodal_package
from .provider import MultimodalProvider
from .scorer import (
    FORMULA_VERSION_MULTIMODAL_V1,
    SCORER_VERSION_MULTIMODAL_V1,
    MultimodalReranker,
    extract_canonical_fingerprint,
    multimodal_v1_formula_v1,
    resolve_source_video_path,
)
from .shortlist import generate_shortlist
from .visual_features import extract_candidate_visual_features

__all__ = [
    "AudioFeatures",
    "ExtractedFrame",
    "MultimodalCandidatePackage",
    "MultimodalModelResult",
    "MultimodalProvider",
    "MultimodalReranker",
    "MultimodalUsage",
    "ObservedEvidenceItem",
    "ObservedRegion",
    "OpenAIMultimodalProvider",
    "ShortlistDocument",
    "SourceAudioProfile",
    "VisualFeatures",
    "build_multimodal_package",
    "compute_source_audio_profile",
    "extract_candidate_audio_features",
    "extract_candidate_frames",
    "extract_candidate_visual_features",
    "extract_canonical_fingerprint",
    "generate_shortlist",
    "multimodal_v1_formula_v1",
    "probe_software_decoder",
    "resolve_source_video_path",
    "FORMULA_VERSION_MULTIMODAL_V1",
    "SCORER_VERSION_MULTIMODAL_V1",
]
