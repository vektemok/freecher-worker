"""Subject detector interfaces and baseline OpenCV implementation.

NOTE: OpenCV Haar/HOG is implemented strictly as a baseline/smoke-test detector.
The architecture is decoupled to allow drop-in replacement with YuNet or YOLO-Face.
"""

from __future__ import annotations

import abc
import logging
from typing import List
import numpy as np

from .models import DetectedSubject

logger = logging.getLogger("freecher_worker")


class SubjectDetector(abc.ABC):
    """Abstract interface for video frame subject detection."""

    @abc.abstractmethod
    def detect(self, frame_bgr: np.ndarray, timestamp: float) -> List[DetectedSubject]:
        """Detect faces or persons in a BGR video frame."""
        pass


class HaarCascadeFaceDetector(SubjectDetector):
    """Baseline smoke-test detector utilizing OpenCV Haar Cascades and HOG Person detector.

    This baseline requires zero external weight downloads and runs completely offline.
    """

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


class CenterCropDetector(SubjectDetector):
    """Fallback detector that returns no subjects, forcing pure center crop."""

    def detect(self, frame_bgr: np.ndarray, timestamp: float) -> List[DetectedSubject]:
        return []


def get_subject_detector(name: str = "haar") -> SubjectDetector:
    """Factory to instantiate subject detector by name."""
    if name.lower() in ("haar", "opencv", "baseline"):
        return HaarCascadeFaceDetector()
    elif name.lower() in ("center", "none"):
        return CenterCropDetector()
    else:
        logger.warning(f"Unknown detector '{name}', falling back to HaarCascadeFaceDetector")
        return HaarCascadeFaceDetector()
