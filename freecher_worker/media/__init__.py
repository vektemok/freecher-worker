"""Media processing module: probe, fingerprint, audio extraction, clipping."""

from .probe import MediaInfo, probe_media, MediaProbeError, NoAudioStreamError
from .fingerprint import SourceFingerprint, compute_source_fingerprint, compute_lightweight_content_hash
from .audio import extract_audio, AudioExtractionError
from .clipper import clip_video, is_nvenc_available, VideoClippingError

__all__ = [
    "MediaInfo",
    "probe_media",
    "MediaProbeError",
    "NoAudioStreamError",
    "SourceFingerprint",
    "compute_source_fingerprint",
    "compute_lightweight_content_hash",
    "extract_audio",
    "AudioExtractionError",
    "clip_video",
    "is_nvenc_available",
    "VideoClippingError",
]
