"""Visual activity and presence feature extraction for candidate highlight windows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .models import VisualFeatures

logger = logging.getLogger("freecher_worker")

MOTION_RESIZE_DIMS = (160, 90)  # (width, height) for fast motion diffing
SCENE_CHANGE_MAD_THRESHOLD = 0.35  # Threshold on normalized pixel delta for cut detection


def _get_face_cascade():
    """Attempt to initialize OpenCV Haar face cascade classifier if available."""
    try:
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if cascade_path.is_file():
                clf = cv2.CascadeClassifier(str(cascade_path))
                if not clf.empty():
                    return clf
    except Exception as exc:
        logger.debug(f"[multimodal-visual] Face cascade init error: {exc}")
    return None


def _get_person_detector():
    """Attempt to initialize OpenCV HOG pedestrian detector if available.

    Requirement: face_presence_ratio and person_presence_ratio must represent real,
    separate detectors. If no person detector exists, person_presence_ratio=null.
    """
    try:
        if hasattr(cv2, "HOGDescriptor") and hasattr(cv2, "HOGDescriptor_getDefaultPeopleDetector"):
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            return hog
    except Exception as exc:
        logger.debug(f"[multimodal-visual] HOG person detector init error: {exc}")
    return None


def extract_candidate_visual_features(
    frame_paths: List[str | Path],
    decoder_used: str = "ffmpeg_default_software",
    requested_count: int = 8,
    failed_count: int = 0,
    decoder_mode: Optional[str] = None,
    requested_decoder: Optional[str] = None,
    hardware_acceleration: bool = False,
) -> VisualFeatures:
    """Compute local visual features across candidate extracted frames.

    Args:
        frame_paths: Paths to extracted JPEG frames.
        decoder_used: Description/name of the decoder used.
        requested_count: Number of frames originally requested.
        failed_count: Number of frames that failed to extract.

    Returns:
        VisualFeatures document.
    """
    gray_frames: List[np.ndarray] = []
    bgr_frames: List[np.ndarray] = []

    for fp in frame_paths:
        p = Path(fp)
        if not p.is_file():
            continue
        try:
            bgr = cv2.imread(str(p))
            if bgr is not None and bgr.size > 0:
                bgr_frames.append(bgr)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                # Resize for lightweight motion computation
                gray_small = cv2.resize(gray, MOTION_RESIZE_DIMS)
                gray_frames.append(gray_small)
        except Exception as exc:
            logger.debug(f"[multimodal-visual] Failed loading frame {p}: {exc}")

    resolved_mode = decoder_mode or (decoder_used if decoder_used in ("libdav1d", "ffmpeg_auto") else "ffmpeg_auto")

    decoded_count = len(gray_frames)
    if decoded_count == 0:
        return VisualFeatures(
            motion_score=0.0,
            scene_change_count=0,
            face_presence_ratio=0.0,
            person_presence_ratio=None,
            decoded_frame_count=0,
            requested_frame_count=requested_count,
            failed_frame_count=failed_count or requested_count,
            decoder_used=decoder_used,
            decoder_mode=resolved_mode,
            requested_decoder=requested_decoder,
            hardware_acceleration=hardware_acceleration,
        )

    # 1. Motion Score & Scene Changes
    motion_score = 0.0
    scene_changes = 0
    if decoded_count >= 2:
        mads = []
        for i in range(decoded_count - 1):
            f1 = gray_frames[i].astype(np.float32) / 255.0
            f2 = gray_frames[i + 1].astype(np.float32) / 255.0
            mad = float(np.mean(np.abs(f1 - f2)))
            mads.append(mad)
            if mad >= SCENE_CHANGE_MAD_THRESHOLD:
                scene_changes += 1

        motion_score = round(float(np.mean(mads)), 4) if mads else 0.0

    # 2. Face Presence (Haar cascade)
    face_cascade = _get_face_cascade()
    face_count = 0
    if face_cascade:
        for bgr in bgr_frames:
            try:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                )
                if len(faces) > 0:
                    face_count += 1
            except Exception:
                pass
        face_presence_ratio = round(face_count / float(decoded_count), 4)
    else:
        face_presence_ratio = 0.0

    # 3. Person Presence (HOG detector, separate detector or None if unavailable)
    person_detector = _get_person_detector()
    if person_detector is not None:
        person_count = 0
        for bgr in bgr_frames:
            try:
                boxes, _ = person_detector.detectMultiScale(
                    bgr, winStride=(8, 8), padding=(4, 4), scale=1.05
                )
                if len(boxes) > 0:
                    person_count += 1
            except Exception:
                pass
        person_presence_ratio = round(person_count / float(decoded_count), 4)
    else:
        person_presence_ratio = None

    resolved_mode = decoder_mode or (decoder_used if decoder_used in ("libdav1d", "ffmpeg_auto") else "ffmpeg_auto")

    return VisualFeatures(
        motion_score=motion_score,
        scene_change_count=scene_changes,
        face_presence_ratio=face_presence_ratio,
        person_presence_ratio=person_presence_ratio,
        decoded_frame_count=decoded_count,
        requested_frame_count=requested_count,
        failed_frame_count=failed_count,
        decoder_used=decoder_used,
        decoder_mode=resolved_mode,
        requested_decoder=requested_decoder,
        hardware_acceleration=hardware_acceleration,
    )
