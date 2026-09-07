"""Data models for subtitle words, events, and styling."""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class SubtitleWord(BaseModel):
    """A single word with clip-relative start and end timestamps."""

    word: str = Field(description="Word text")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")


class SubtitleEvent(BaseModel):
    """A single subtitle card containing 2 to 5 words."""

    id: int = Field(description="Event sequence identifier")
    start: float = Field(description="Event start time in seconds")
    end: float = Field(description="Event end time in seconds")
    text: str = Field(description="Formatted text of the card")
    words: List[SubtitleWord] = Field(default_factory=list, description="Constituent words with timings")
