"""ASS (Advanced SubStation Alpha) subtitle generator for vertical video with active-word pop."""

from __future__ import annotations

from pathlib import Path
from typing import List
from .models import SubtitleEvent

# Color constants in ASS BGR format (&HAABBGGRR)
COLOR_WHITE = "&H00FFFFFF&"
COLOR_GOLD = "&H00D7FF&"  # Vibrant energetic gold/yellow pop


def format_ass_timestamp(seconds: float) -> str:
    """Format seconds into ASS timestamp format H:MM:SS.cs."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        secs += 1
        cs = 0
    return f"{hours}:{mins:02d}:{secs:02d}.{cs:02d}"


def generate_ass_script(
    events: List[SubtitleEvent],
    font_family: str = "Montserrat, DejaVu Sans, Arial",
    font_size: int = 54,
    active_word_highlight: bool = True,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    margin_v: int = 320,
) -> str:
    """Generate complete ASS subtitle script content with karaoke pop effect."""
    lines: List[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {play_res_x}",
        f"PlayResY: {play_res_y}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font_family},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        f"1,0,0,0,100,100,0,0,1,3.5,2.0,2,80,80,{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for event in events:
        words = event.words
        if not active_word_highlight or not words:
            # Static card event without per-word highlight
            start_str = format_ass_timestamp(event.start)
            end_str = format_ass_timestamp(event.end)
            lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{event.text}")
            continue

        # Active-word karaoke generation:
        # Create non-overlapping, continuous sub-dialogue events covering the card duration.
        num_words = len(words)
        for j, curr_word in enumerate(words):
            # Start at event.start for the first word, else word's start
            t_start = event.start if j == 0 else curr_word.start

            # End when next word starts, or at event.end for the final word
            if j < num_words - 1:
                t_end = words[j + 1].start
            else:
                t_end = event.end

            # Ensure positive duration
            if t_end <= t_start:
                t_end = t_start + 0.10

            # Build card text with word j highlighted in gold
            word_parts = []
            for k, w in enumerate(words):
                if k == j:
                    word_parts.append(f"{{\\c{COLOR_GOLD}}}{w.word}{{\\c{COLOR_WHITE}}}")
                else:
                    word_parts.append(w.word)

            styled_text = " ".join(word_parts)
            start_str = format_ass_timestamp(t_start)
            end_str = format_ass_timestamp(t_end)
            lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{styled_text}")

    return "\n".join(lines) + "\n"


def save_ass_file(
    events: List[SubtitleEvent],
    output_path: Path,
    font_family: str = "Montserrat, DejaVu Sans, Arial",
    font_size: int = 54,
    active_word_highlight: bool = True,
    margin_v: int = 320,
) -> Path:
    """Save ASS subtitle script to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = generate_ass_script(
        events=events,
        font_family=font_family,
        font_size=font_size,
        active_word_highlight=active_word_highlight,
        margin_v=margin_v,
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path
