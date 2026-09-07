"""Subtitle event segmentation from word timestamps."""

from __future__ import annotations

from typing import List
from arny_worker.rendering.asr_refinement import WordItem
from .models import SubtitleEvent, SubtitleWord

PUNCTUATION_TERMINAL = {".", "!", "?"}
PUNCTUATION_PAUSE = {",", ":", ";", "—", "-"}


def segment_words_to_events(
    words: List[WordItem],
    max_words: int = 4,
    max_chars: int = 32,
    pause_threshold_seconds: float = 0.35,
    min_event_duration: float = 0.4,
) -> List[SubtitleEvent]:
    """Segment a stream of word timestamps into concise, readable subtitle cards."""
    if not words:
        return []

    events: List[SubtitleEvent] = []
    current_chunk: List[WordItem] = []

    def _flush_chunk(chunk: List[WordItem], event_id: int) -> SubtitleEvent:
        sub_words = [
            SubtitleWord(word=w.word, start=round(w.start, 3), end=round(w.end, 3))
            for w in chunk
        ]
        card_text = " ".join(w.word for w in chunk)
        start = chunk[0].start
        end = max(chunk[-1].end, start + min_event_duration)
        return SubtitleEvent(
            id=event_id,
            start=round(start, 3),
            end=round(end, 3),
            text=card_text,
            words=sub_words,
        )

    for i, w in enumerate(words):
        current_chunk.append(w)
        is_last_word = (i == len(words) - 1)

        if is_last_word:
            break

        next_w = words[i + 1]
        pause_after = next_w.start - w.end

        # Check triggers to break the card
        has_terminal_punct = any(w.word.endswith(p) for p in PUNCTUATION_TERMINAL)
        has_pause_punct = any(w.word.endswith(p) for p in PUNCTUATION_PAUSE)
        long_pause = pause_after >= pause_threshold_seconds
        hit_max_words = len(current_chunk) >= max_words
        hit_max_chars = sum(len(cw.word) for cw in current_chunk) + len(current_chunk) - 1 >= max_chars

        # If terminal punctuation, always break
        if has_terminal_punct:
            events.append(_flush_chunk(current_chunk, len(events) + 1))
            current_chunk = []
        # If long pause or pause punctuation with at least 2 words
        elif long_pause or (has_pause_punct and len(current_chunk) >= 2):
            events.append(_flush_chunk(current_chunk, len(events) + 1))
            current_chunk = []
        # If word count or character limits reached
        elif hit_max_words or hit_max_chars:
            events.append(_flush_chunk(current_chunk, len(events) + 1))
            current_chunk = []

    if current_chunk:
        events.append(_flush_chunk(current_chunk, len(events) + 1))

    # Second pass: smoothly bridge small gaps (< 0.2s) between adjacent events
    for i in range(len(events) - 1):
        gap = events[i + 1].start - events[i].end
        if 0 < gap < 0.20:
            events[i].end = events[i + 1].start

    return events
