"""Subtitles processing, event segmentation, and ASS styling module."""

from .models import SubtitleEvent, SubtitleWord
from .segmenter import segment_words_to_events
from .ass import format_ass_timestamp, generate_ass_script, save_ass_file

__all__ = [
    "SubtitleEvent",
    "SubtitleWord",
    "segment_words_to_events",
    "format_ass_timestamp",
    "generate_ass_script",
    "save_ass_file",
]
