"""Tests for the audio+text blended highlight scorer."""

from __future__ import annotations

import subprocess
import wave

import numpy as np
import pytest

from freecher_worker.highlights.models import CandidateWindow
from freecher_worker.scoring.audio_text import (
    AUDIO_TEXT_SCORER_VERSION,
    DEFAULT_AUDIO_FEATURE,
    DEFAULT_AUDIO_WEIGHT,
    AudioTextScorer,
    AudioTextScoringError,
    ensure_wav,
    rank_fractions,
)
from freecher_worker.scoring.base import HighlightScorer
from freecher_worker.highlights.models import HighlightScore

SAMPLE_RATE = 16_000


def candidate(cid: str, start: float, end: float) -> CandidateWindow:
    return CandidateWindow(
        id=cid, start=start, end=end, duration=end - start, text=f"текст {cid}", segment_ids=[],
    )


@pytest.fixture(scope="module")
def varied_wav(tmp_path_factory):
    """60s of audio whose loudness spread differs sharply between windows.

    0-20s  steady tone      -> low rms_std
    20-40s alternating loud/quiet bursts -> high rms_std
    40-60s near silence     -> low everything
    """
    path = tmp_path_factory.mktemp("audio") / "source.wav"
    t = np.arange(SAMPLE_RATE * 60) / SAMPLE_RATE
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)

    signal = np.zeros_like(tone)
    signal[: SAMPLE_RATE * 20] = tone[: SAMPLE_RATE * 20]
    burst = tone[SAMPLE_RATE * 20 : SAMPLE_RATE * 40].copy()
    envelope = np.tile(
        np.concatenate([np.full(SAMPLE_RATE // 2, 1.0), np.full(SAMPLE_RATE // 2, 0.02)]), 20
    )[: len(burst)]
    signal[SAMPLE_RATE * 20 : SAMPLE_RATE * 40] = burst * envelope
    signal[SAMPLE_RATE * 40 :] = tone[SAMPLE_RATE * 40 :] * 0.005

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes((signal * 32767).astype("<i2").tobytes())
    return path


class StubTextScorer(HighlightScorer):
    """Returns a caller-supplied score per candidate id."""

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def score(self, candidate, context=None):
        return HighlightScore(
            score=self.table[candidate.id], hook_score=0, standalone_score=0, emotion_score=0,
            information_score=0, shareability_score=0, reason="stub",
        )

    def score_batch(self, candidates, transcript=None):
        self.calls += 1
        return [self.score(c) for c in candidates]


# --------------------------------------------------------------------------
# rank normalization
# --------------------------------------------------------------------------


def test_ranks_spread_values_across_zero_to_one():
    # sorted: 1, 3, 5, 9 -> positions 0, 1, 2, 3 -> divided by (n - 1).
    assert rank_fractions([5, 1, 3, 9]) == [2 / 3, 0.0, 1 / 3, 1.0]


def test_tied_values_share_a_rank():
    # Ties must not be broken arbitrarily; both get the average position.
    assert rank_fractions([5, 1, 3, 3, 9]) == [0.75, 0.0, 0.375, 0.375, 1.0]


def test_degenerate_inputs_are_handled():
    assert rank_fractions([]) == []
    assert rank_fractions([7]) == [0.5]
    # Everything tied lands in the middle rather than at an arbitrary end.
    assert rank_fractions([2, 2, 2]) == [0.5, 0.5, 0.5]


def test_rank_space_makes_incompatible_scales_comparable():
    """The whole point: an unbounded feature must not swamp a bounded one."""
    # A bounded 0-100 score piled up at its floor, and a tiny unbounded one
    # spanning three orders of magnitude -- the real shapes of the text and
    # audio sides.
    bounded = [0.0, 0.0, 0.0, 100.0]
    unbounded = [1e-6, 5e-6, 2e-3, 1e-6]

    for values in (bounded, unbounded):
        fractions = rank_fractions(values)
        # Both end up on the same 0..1 scale whatever their raw magnitude, so
        # neither can dominate the blend by numeric range alone.
        assert all(0.0 <= f <= 1.0 for f in fractions)
        assert max(fractions) == 1.0

    # And the ordering each carries is preserved exactly.
    assert rank_fractions(unbounded).index(1.0) == unbounded.index(max(unbounded))
    assert rank_fractions(bounded).index(1.0) == bounded.index(max(bounded))


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_the_defaults_are_the_measured_ones():
    assert DEFAULT_AUDIO_WEIGHT == 0.4
    assert DEFAULT_AUDIO_FEATURE == "rms_std"


@pytest.mark.parametrize("weight", [-0.1, 1.5, 2.0])
def test_an_out_of_range_weight_is_refused(weight, varied_wav):
    with pytest.raises(ValueError, match="audio_weight"):
        AudioTextScorer(varied_wav, text_scores={}, audio_weight=weight)


def test_a_blend_without_a_text_side_is_refused(varied_wav):
    with pytest.raises(ValueError, match="text side is required"):
        AudioTextScorer(varied_wav, audio_weight=0.4)


def test_pure_audio_needs_no_text_side(varied_wav):
    scorer = AudioTextScorer(varied_wav, audio_weight=1.0)
    assert scorer.audio_weight == 1.0


def test_single_candidate_scoring_is_refused_with_an_explanation(varied_wav):
    # A rank blend has no meaning for one candidate, and silently returning
    # something would be worse than saying so.
    scorer = AudioTextScorer(varied_wav, text_scores={"a": 1.0})
    with pytest.raises(NotImplementedError, match="score_batch"):
        scorer.score(candidate("a", 0.0, 10.0))


# --------------------------------------------------------------------------
# audio extraction
# --------------------------------------------------------------------------


def test_the_loud_varied_window_scores_above_the_steady_one(varied_wav):
    """rms_std must separate dynamics from level, which is why it was chosen."""
    scorer = AudioTextScorer(varied_wav, audio_weight=1.0)
    values = scorer.audio_values(
        [candidate("steady", 2.0, 18.0), candidate("bursts", 22.0, 38.0), candidate("quiet", 42.0, 58.0)]
    )
    steady, bursts, quiet = values
    assert bursts > steady
    assert bursts > quiet


def test_an_unknown_audio_feature_is_reported_clearly(varied_wav):
    scorer = AudioTextScorer(varied_wav, audio_weight=1.0, audio_feature="loudness_vibe")
    with pytest.raises(AudioTextScoringError, match="loudness_vibe"):
        scorer.audio_values([candidate("a", 0.0, 10.0)])


# --------------------------------------------------------------------------
# the blend
# --------------------------------------------------------------------------


def test_the_blend_weights_both_sides_as_asked(varied_wav):
    candidates = [candidate("steady", 2.0, 18.0), candidate("bursts", 22.0, 38.0)]
    # Text ranks the opposite way to audio, so the weight decides the winner.
    text = {"steady": 100.0, "bursts": 0.0}

    audio_led = AudioTextScorer(varied_wav, text_scores=text, audio_weight=0.9).score_batch(candidates)
    text_led = AudioTextScorer(varied_wav, text_scores=text, audio_weight=0.1).score_batch(candidates)

    assert audio_led[1].score > audio_led[0].score  # bursts win on audio
    assert text_led[0].score > text_led[1].score    # steady wins on text


def test_a_precomputed_text_side_costs_no_calls(varied_wav):
    stub = StubTextScorer({"a": 50.0, "b": 10.0})
    scorer = AudioTextScorer(
        varied_wav, text_scores={"a": 50.0, "b": 10.0}, text_scorer=stub, audio_weight=0.5
    )
    scorer.score_batch([candidate("a", 2.0, 18.0), candidate("b", 22.0, 38.0)])
    # text_scores wins over text_scorer, so nothing was scored again.
    assert stub.calls == 0


def test_a_text_scorer_is_used_when_no_scores_are_supplied(varied_wav):
    stub = StubTextScorer({"a": 50.0, "b": 10.0})
    scorer = AudioTextScorer(varied_wav, text_scorer=stub, audio_weight=0.5)
    scorer.score_batch([candidate("a", 2.0, 18.0), candidate("b", 22.0, 38.0)])
    assert stub.calls == 1


def test_a_candidate_with_no_text_score_is_reported_not_defaulted(varied_wav):
    # Treating a missing score as zero would silently sink that candidate.
    scorer = AudioTextScorer(varied_wav, text_scores={"a": 1.0}, audio_weight=0.5)
    with pytest.raises(AudioTextScoringError, match="no text score"):
        scorer.score_batch([candidate("a", 2.0, 18.0), candidate("b", 22.0, 38.0)])


def test_pure_audio_ignores_the_text_side_entirely(varied_wav):
    scorer = AudioTextScorer(varied_wav, text_scores={"a": 0.0, "b": 100.0}, audio_weight=1.0)
    scores = scorer.score_batch([candidate("a", 2.0, 18.0), candidate("b", 22.0, 38.0)])
    # b is louder-varied; text says the opposite and must not matter.
    assert scores[1].score > scores[0].score
    assert all(s.subscores["text_rank"] == 0.0 for s in scores)


def test_every_score_stays_in_range_and_records_its_parts(varied_wav):
    candidates = [candidate("a", 2.0, 18.0), candidate("b", 22.0, 38.0), candidate("c", 42.0, 58.0)]
    scores = AudioTextScorer(
        varied_wav, text_scores={"a": 10.0, "b": 20.0, "c": 30.0}, audio_weight=0.4
    ).score_batch(candidates)

    assert len(scores) == 3
    for s in scores:
        assert 0.0 <= s.score <= 100.0
        assert s.scorer_version == AUDIO_TEXT_SCORER_VERSION
        assert s.subscores["audio_weight"] == 0.4
        assert 0.0 <= s.subscores["audio_rank"] <= 1.0
        assert "blend=" in s.reason


def test_an_empty_batch_is_not_an_error(varied_wav):
    assert AudioTextScorer(varied_wav, text_scores={}).score_batch([]) == []


# --------------------------------------------------------------------------
# WAV preparation
# --------------------------------------------------------------------------


def test_a_wav_is_passed_through_untouched(varied_wav):
    assert ensure_wav(varied_wav) == varied_wav


def test_a_missing_artifact_is_reported(tmp_path):
    with pytest.raises(AudioTextScoringError, match="not found"):
        ensure_wav(tmp_path / "nope.m4a")


@pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="ffmpeg is not installed",
)
def test_an_m4a_is_decoded_once_and_then_reused(tmp_path, varied_wav):
    m4a = tmp_path / "audio.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(varied_wav),
         "-ac", "1", "-ar", "16000", "-c:a", "aac", "-f", "ipod", str(m4a)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    target = tmp_path / "decoded.wav"

    first = ensure_wav(m4a, target)
    assert first == target and target.is_file()

    with wave.open(str(target), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == SAMPLE_RATE

    # Second call reuses rather than re-decoding.
    stamp = target.stat().st_mtime_ns
    assert ensure_wav(m4a, target) == target
    assert target.stat().st_mtime_ns == stamp


@pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="ffmpeg is not installed",
)
def test_bytes_that_are_not_audio_are_refused(tmp_path):
    junk = tmp_path / "audio.m4a"
    junk.write_bytes(b"definitely not audio" * 100)
    with pytest.raises(AudioTextScoringError, match="could not decode"):
        ensure_wav(junk, tmp_path / "out.wav")


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------


def test_the_cli_requires_audio_for_this_scorer():
    import inspect

    from freecher_worker.cli import score_run_command

    source = inspect.getsource(score_run_command)
    assert '"audio_text"' in source
    assert "--audio is required" in source
    # A text-scores document from a different candidate set must not be blended in.
    assert "Candidate set mismatch" in source
