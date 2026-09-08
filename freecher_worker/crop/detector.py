"""Subject detectors used by smart 9:16 reframing.

YuNet (OpenCV Zoo, ~230 KB ONNX) is the production baseline: modern, CPU-fast and far more
reliable than Haar cascades. Haar/HOG remain only as a legacy fallback for builds that still
ship them - notably, `opencv-python-headless` 5.x removed `CascadeClassifier` and
`HOGDescriptor` entirely, which made the old baseline silently detect nothing at all.

Detectors report their own availability so a dead detector surfaces as a loud diagnostic
instead of an empty detection list that looks like an empty scene.
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from .models import DetectedSubject
from .weights import ModelWeightsError, resolve_yunet_weights

logger = logging.getLogger("freecher_worker")


class DetectorUnavailableError(RuntimeError):
    """Raised when a detector cannot be constructed on the current OpenCV build."""


class SubjectDetector(abc.ABC):
    """Abstract interface for video frame subject detection."""

    #: Human-readable identifier recorded in reframing diagnostics.
    name: str = "subject_detector"

    @abc.abstractmethod
    def detect(self, frame_bgr: np.ndarray, timestamp: float) -> List[DetectedSubject]:
        """Detect faces or persons in a BGR video frame."""

    @property
    def is_operational(self) -> bool:
        """Whether this detector can actually produce detections on this build."""
        return True

    def describe(self) -> str:
        """Short description used in logs and diagnostics."""
        return self.name


class YuNetFaceDetector(SubjectDetector):
    """OpenCV YuNet face detector - the production face-detection baseline.

    Small (~230 KB), CPU-only, and markedly more reliable than Haar cascades on the angled,
    partially occluded and unevenly lit faces that stream footage is full of.
    """

    name = "yunet"

    def __init__(
        self,
        model_path: Optional[Path] = None,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 50,
        allow_download: bool = True,
    ) -> None:
        import cv2

        if not hasattr(cv2, "FaceDetectorYN_create"):
            raise DetectorUnavailableError(
                f"OpenCV {cv2.__version__} has no FaceDetectorYN; install opencv-python>=4.5.4"
            )

        try:
            weights = resolve_yunet_weights(explicit_path=model_path, allow_download=allow_download)
        except ModelWeightsError as exc:
            raise DetectorUnavailableError(str(exc)) from exc

        self.cv2 = cv2
        self.model_path = weights
        self.score_threshold = score_threshold
        self._input_size = (320, 320)
        try:
            self._detector = cv2.FaceDetectorYN_create(
                str(weights), "", self._input_size, score_threshold, nms_threshold, top_k
            )
        except Exception as exc:  # pragma: no cover - corrupt weights
            raise DetectorUnavailableError(f"Could not initialize YuNet from {weights}: {exc}") from exc

    def detect(self, frame_bgr: np.ndarray, timestamp: float) -> List[DetectedSubject]:
        height, width = frame_bgr.shape[:2]
        if (width, height) != self._input_size:
            self._detector.setInputSize((width, height))
            self._input_size = (width, height)

        _, faces = self._detector.detect(frame_bgr)
        if faces is None:
            return []

        detected: List[DetectedSubject] = []
        for face in faces:
            x, y, w, h = (int(round(v)) for v in face[:4])
            confidence = float(face[-1])
            if w <= 0 or h <= 0:
                continue
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = min(w, width - x)
            h = min(h, height - y)
            detected.append(
                DetectedSubject(
                    box=(x, y, w, h),
                    confidence=confidence,
                    subject_type="face",
                    area=float(w * h),
                    center_x=x + w / 2.0,
                    center_y=y + h / 2.0,
                )
            )
        return detected

    def describe(self) -> str:
        return f"yunet({self.model_path.name}, score>={self.score_threshold})"


class CompositeSubjectDetector(SubjectDetector):
    """Run detectors in priority order, returning the first non-empty result.

    Faces take priority over persons, matching the reframing priority order: an active speaker
    or any visible face outranks a merely present body.
    """

    name = "composite"

    def __init__(self, detectors: List[SubjectDetector]) -> None:
        self.detectors = [d for d in detectors if d is not None]

    def detect(self, frame_bgr: np.ndarray, timestamp: float) -> List[DetectedSubject]:
        for detector in self.detectors:
            try:
                found = detector.detect(frame_bgr, timestamp)
            except Exception as exc:
                logger.warning(f"[detector] {detector.name} failed at t={timestamp:.2f}s: {exc}")
                continue
            if found:
                return found
        return []

    @property
    def is_operational(self) -> bool:
        return any(d.is_operational for d in self.detectors)

    def describe(self) -> str:
        return "+".join(d.describe() for d in self.detectors) or "none"


class HaarCascadeFaceDetector(SubjectDetector):
    """Legacy OpenCV Haar cascade + HOG person detector.

    Kept only as a fallback. It is not a production-quality baseline, and on
    `opencv-python-headless` 5.x neither `CascadeClassifier` nor `HOGDescriptor` exists at all,
    in which case :attr:`is_operational` is False and the caller must say so rather than treat
    an empty detection list as an empty scene.
    """

    name = "haar"

    def __init__(self, min_face_size: int = 30) -> None:
        import cv2

        self.cv2 = cv2
        self.min_face_size = min_face_size

        self.face_cascade = None
        self.profile_cascade = None
        self.hog = None

        # Pre-trained Haar cascades bundled directly with OpenCV
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            try:
                frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
                self.face_cascade = cv2.CascadeClassifier(frontal_path)
                self.profile_cascade = cv2.CascadeClassifier(profile_path)
            except Exception as e:
                logger.warning(f"Failed initializing CascadeClassifier: {e}")

        # Built-in HOG person detector fallback
        if hasattr(cv2, "HOGDescriptor") and hasattr(cv2.HOGDescriptor, "getDefaultPeopleDetector"):
            try:
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
            except Exception as e:
                logger.warning(f"Failed initializing HOGDescriptor: {e}")

        if self.face_cascade is None and self.hog is None:
            logger.error(
                f"[detector] OpenCV {cv2.__version__} provides neither CascadeClassifier nor "
                f"HOGDescriptor; the Haar detector cannot detect anything on this build"
            )

    def detect(self, frame_bgr: np.ndarray, timestamp: float) -> List[DetectedSubject]:
        """Detect faces first; fall back to people if no faces detected."""
        cv2 = self.cv2
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        detected: List[DetectedSubject] = []

        # 1. Detect frontal faces
        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
            )

            for (x, y, fw, fh) in faces:
                cx = float(x + fw / 2.0)
                cy = float(y + fh / 2.0)
                detected.append(
                    DetectedSubject(
                        box=(int(x), int(y), int(fw), int(fh)),
                        confidence=0.90,
                        subject_type="face",
                        area=float(fw * fh),
                        center_x=cx,
                        center_y=cy,
                    )
                )

        # If no frontal faces, check profile faces
        if not detected and self.profile_cascade is not None:
            profiles = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(self.min_face_size, self.min_face_size),
            )
            for (x, y, fw, fh) in profiles:
                cx = float(x + fw / 2.0)
                cy = float(y + fh / 2.0)
                detected.append(
                    DetectedSubject(
                        box=(int(x), int(y), int(fw), int(fh)),
                        confidence=0.80,
                        subject_type="face",
                        area=float(fw * fh),
                        center_x=cx,
                        center_y=cy,
                    )
                )

        # 2. If no faces at all, fall back to detecting standing/seated persons via HOG
        if not detected and self.hog is not None:
            boxes, weights = self.hog.detectMultiScale(
                gray,
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05,
            )
            for (x, y, pw, ph), weight in zip(boxes, weights):
                if weight > 0.2:
                    cx = float(x + pw / 2.0)
                    cy = float(y + ph / 2.0)
                    detected.append(
                        DetectedSubject(
                            box=(int(x), int(y), int(pw), int(ph)),
                            confidence=float(weight),
                            subject_type="person",
                            area=float(pw * ph),
                            center_x=cx,
                            center_y=cy,
                        )
                    )

        return detected


    @property
    def is_operational(self) -> bool:
        return self.face_cascade is not None or self.hog is not None

    def describe(self) -> str:
        parts = []
        if self.face_cascade is not None:
            parts.append("haar-face")
        if self.profile_cascade is not None:
            parts.append("haar-profile")
        if self.hog is not None:
            parts.append("hog-person")
        return f"haar({'+'.join(parts) or 'unavailable'})"


class CenterCropDetector(SubjectDetector):
    """Detector that never reports subjects, forcing a pure center crop."""

    name = "center"

    def detect(self, frame_bgr: np.ndarray, timestamp: float) -> List[DetectedSubject]:
        return []

    @property
    def is_operational(self) -> bool:
        return False

    def describe(self) -> str:
        return "center (detection disabled)"


def get_subject_detector(
    name: str = "auto",
    model_path: Optional[Path] = None,
    allow_download: bool = True,
    score_threshold: float = 0.6,
) -> SubjectDetector:
    """Instantiate a subject detector by name.

    ``auto`` prefers YuNet and only degrades to the legacy Haar/HOG detector when YuNet's
    weights or OpenCV support are unavailable, logging why.
    """
    choice = (name or "auto").lower()

    def _yunet() -> SubjectDetector:
        return YuNetFaceDetector(
            model_path=model_path, allow_download=allow_download, score_threshold=score_threshold
        )

    if choice in ("yunet", "face"):
        return _yunet()
    if choice in ("center", "none"):
        return CenterCropDetector()
    if choice in ("haar", "opencv", "legacy"):
        return HaarCascadeFaceDetector()

    if choice not in ("auto", "baseline", "default"):
        logger.warning(f"[detector] Unknown detector '{name}', using 'auto'")

    detectors: List[SubjectDetector] = []
    try:
        detectors.append(_yunet())
    except DetectorUnavailableError as exc:
        logger.error(f"[detector] YuNet unavailable ({exc}); falling back to Haar/HOG")

    legacy = HaarCascadeFaceDetector()
    if legacy.is_operational:
        # Only useful as a person-detection backstop once YuNet has found no faces.
        detectors.append(legacy)
    elif not detectors:
        logger.error(
            "[detector] No usable subject detector on this OpenCV build; reframing will rely "
            "entirely on the dominant-region fallback"
        )

    if not detectors:
        return CenterCropDetector()
    if len(detectors) == 1:
        return detectors[0]
    return CompositeSubjectDetector(detectors)
