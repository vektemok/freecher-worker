"""Smart vertical 9:16 crop module with subject detection and tracking."""

from .models import DetectedSubject, CropPoint, CropTrajectory
from .detector import (
    CenterCropDetector,
    CompositeSubjectDetector,
    DetectorUnavailableError,
    HaarCascadeFaceDetector,
    SubjectDetector,
    YuNetFaceDetector,
    get_subject_detector,
)
from .weights import ModelWeightsError, resolve_yunet_weights
from .tracker import calculate_crop_dimensions, select_target_center, generate_crop_trajectory
from .expression import (
    build_ffmpeg_crop_expression,
    fit_points_to_expression_budget,
    build_ffmpeg_crop_x_expression,
    simplify_trajectory_points,
)

__all__ = [
    "DetectedSubject",
    "CropPoint",
    "CropTrajectory",
    "SubjectDetector",
    "YuNetFaceDetector",
    "CompositeSubjectDetector",
    "DetectorUnavailableError",
    "ModelWeightsError",
    "resolve_yunet_weights",
    "HaarCascadeFaceDetector",
    "CenterCropDetector",
    "get_subject_detector",
    "calculate_crop_dimensions",
    "select_target_center",
    "generate_crop_trajectory",
    "build_ffmpeg_crop_expression",
    "fit_points_to_expression_budget",
    "build_ffmpeg_crop_x_expression",
    "simplify_trajectory_points",
]
