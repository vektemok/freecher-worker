"""Unit tests for subtitle segmentation and ASS styling."""

from arny_worker.rendering.asr_refinement import WordItem
from arny_worker.subtitles.ass import format_ass_timestamp, generate_ass_script
from arny_worker.subtitles.segmenter import segment_words_to_events


def test_format_ass_timestamp():
    """Verify ASS timestamp formatting H:MM:SS.cs."""
    assert format_ass_timestamp(0.0) == "0:00:00.00"
    assert format_ass_timestamp(1.25) == "0:00:01.25"
    assert format_ass_timestamp(65.5) == "0:01:05.50"
    assert format_ass_timestamp(3661.05) == "1:01:01.05"


def test_segment_words_to_events():
    """Verify grouping word stream into 2-5 word subtitle events."""
    words = [
        WordItem(word="Привет,", start=0.2, end=0.6, probability=0.98),
        WordItem(word="это", start=0.7, end=0.9, probability=0.99),
        WordItem(word="первое", start=1.0, end=1.4, probability=0.97),
        WordItem(word="тестовое", start=1.5, end=1.9, probability=0.95),
        WordItem(word="видео.", start=2.0, end=2.5, probability=0.99),
        # Gap of 0.8s
        WordItem(word="А", start=3.3, end=3.5, probability=0.99),
        WordItem(word="здесь", start=3.6, end=3.9, probability=0.98),
        WordItem(word="вторая", start=4.0, end=4.4, probability=0.95),
        WordItem(word="мысль.", start=4.5, end=4.9, probability=0.99),
    ]

    events = segment_words_to_events(words, max_words=4)
    assert len(events) >= 2

    # First event has punctuation break or max words
    e1 = events[0]
    assert e1.start == 0.2
    assert len(e1.words) >= 2
    assert "Привет" in e1.text

    # Second event starts after the silence gap
    assert any(e.start >= 3.0 for e in events)


def test_generate_ass_script_with_active_word():
    """Verify ASS script generation with active word karaoke coloring."""
    words = [
        WordItem(word="Быстрый", start=0.1, end=0.5, probability=1.0),
        WordItem(word="тест", start=0.6, end=1.0, probability=1.0),
    ]
    events = segment_words_to_events(words, max_words=2)
    ass_content = generate_ass_script(events, active_word_highlight=True)

    assert "[Script Info]" in ass_content
    assert "PlayResX: 1080" in ass_content
    assert "PlayResY: 1920" in ass_content
    assert ",80,80,320,1" in ass_content
    assert "[Events]" in ass_content

    # Check for color tag and Dialogue lines
    assert r"{\c&H00D7FF&}" in ass_content  # Active gold
    assert r"{\c&H00FFFFFF&}" in ass_content  # White reset
    dialogue_lines = [line for line in ass_content.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogue_lines) == 2  # One line per word in the card
