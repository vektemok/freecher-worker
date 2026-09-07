"""Data models for subject detection, crop tracking, and trajectories."""

from __future__ import annotations

from typing import List, Tuple
from pydantic import BaseModel, Field


class DetectedSubject(BaseModel):
    """A detected visual subject (face or person) in a video frame."""

    box: Tuple[int, int, int, int] = Field(description="(x, y, width, height) bounding box in pixels")
    confidence: float = Field(default=1.0, description="Detection confidence score")
    subject_type: str = Field(description="'face' or 'person'")
    area: float = Field(description="Bounding box pixel area (w * h)")
    center_x: float = Field(description="Horizontal center coordinate")
    center_y: float = Field(description="Vertical center coordinate")


class CropPoint(BaseModel):
    """A single temporal keyframe along the crop trajectory."""

    time: float = Field(description="Timestamp in seconds relative to clip start")
    center_x: float = Field(description="Target horizontal center coordinate in source space")
    center_y: float = Field(description="Target vertical center coordinate in source space")
    crop_x: int = Field(description="Even integer top-left x coordinate in source space")
    crop_y: int = Field(description="Even integer top-left y coordinate in source space")
    crop_w: int = Field(description="Even integer crop width in source space")
    crop_h: int = Field(description="Even integer crop height in source space")
    subject_type: str = Field(description="Type of tracked subject at this keyframe")


class CropTrajectory(BaseModel):
    """Complete temporal crop trajectory across a highlight clip."""

    source_width: int
    source_height: int
    crop_w: int
    crop_h: int
    points: List[CropPoint] = Field(default_factory=list)
