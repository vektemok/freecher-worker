"""Tests for the logprob-based highlight scorer."""

from __future__ import annotations

import json

import pytest

from freecher_worker.highlights.models import CandidateWindow
from freecher_worker.scoring.logprob import (
    BINARY_SYSTEM_PROMPT,
    BINARY_VALUES,
    GRADE_VALUES,
    GRADED_SYSTEM_PROMPT,
    MODE_BINARY,
    MODE_GRADED,
    LogprobBinaryScorer,
    LogprobScoringError,
    expected_value,
    normalize_token,
)
from freecher_worker.transcription.models import Transcript, TranscriptSegment


def candidate(cid: str = "cand_001", start: float = 0.0, end: float = 60.0) -> CandidateWindow:
    return CandidateWindow(
        id=cid, start=start, end=end, duration=end - start,
        text="какой-то текст кандидата", segment_ids=[1, 2],
    )


def logprob_response(pairs, *, answer="no", model="gpt-4o-mini-2024-07-18"):
    """An OpenAI chat completion carrying top_logprobs for the first token."""
    return {
        "model": model,
        "choices": [{
            "message": {"content": answer},
            "logprobs": {"content": [{
                "token": answer,
                "top_logprobs": [{"token": t, "logprob": lp} for t, lp in pairs],
            }]},
        }],
        "usage": {"prompt_tokens": 900, "completion_tokens": 1},
    }


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeClient:
    """Stands in for httpx.Client; replays a queue of responses or raises."""

    def __init__(self, queue):
        self.queue = queue

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None):
        FakeClient.last_payload = json
        FakeClient.last_url = url
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def patch_httpx(monkeypatch):
    def install(queue):
        import httpx
        monkeypatch.setattr(httpx, "Client", FakeClient(queue))
    return install


def make_scorer(**kwargs):
    kwargs.setdefault("mode", MODE_BINARY)
    kwargs.setdefault("retry_backoff_seconds", 0.0)
    return LogprobBinaryScorer(base_url="https://api.test/v1", api_key="k", **kwargs)


# --------------------------------------------------------------------------
# token handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(" Yes.", "yes"), ("NO", "no"), ("  STRONG", "strong"), ('"dead"', "dead"), ("...", "")],
)
def test_tokens_are_normalized_before_matching(raw, expected):
    assert normalize_token(raw) == expected


def test_binary_value_is_the_probability_mass_on_yes():
    # ~95% on yes, ~5% on no.
    value = expected_value(
        [{"token": "Yes", "logprob": -0.05}, {"token": "No", "logprob": -3.0}], BINARY_VALUES
    )
    assert value == pytest.approx(95.03, abs=0.1)


def test_unrelated_tokens_do_not_drag_the_score_down():
    # A quote mark or preamble carrying mass must not count as a "no".
    with_noise = expected_value(
        [{"token": "Yes", "logprob": -0.05}, {"token": '"', "logprob": -1.0},
         {"token": "No", "logprob": -3.0}],
        BINARY_VALUES,
    )
    without = expected_value(
        [{"token": "Yes", "logprob": -0.05}, {"token": "No", "logprob": -3.0}], BINARY_VALUES
    )
    assert with_noise == pytest.approx(without)


def test_graded_value_is_the_weighted_average_of_the_grades():
    # All mass on WEAK -> exactly the WEAK value.
    assert expected_value([{"token": "WEAK", "logprob": 0.0}], GRADE_VALUES) == pytest.approx(33.333)
    # Split between WEAK and DECENT lands between them.
    mixed = expected_value(
        [{"token": "WEAK", "logprob": -0.693}, {"token": "DECENT", "logprob": -0.693}],
        GRADE_VALUES,
    )
    assert 33.333 < mixed < 66.667


def test_a_missing_answer_token_is_reported_rather_than_scored_zero():
    # Absent is not the same as improbable; the caller must be able to tell.
    assert expected_value([{"token": "Maybe", "logprob": -0.1}], BINARY_VALUES) is None
    assert expected_value([], GRADE_VALUES) is None


def test_case_and_punctuation_variants_all_count():
    value = expected_value(
        [{"token": " Yes", "logprob": -0.7}, {"token": "yes", "logprob": -0.7},
         {"token": "No", "logprob": -10.0}],
        BINARY_VALUES,
    )
    # Both yes spellings contribute, so this is far above 50.
    assert value > 99.0


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_the_two_modes_are_versioned_apart():
    assert make_scorer(mode=MODE_BINARY).version == "logprob_v1_binary"
    assert make_scorer(mode=MODE_GRADED).version == "logprob_v1_graded"


def test_each_mode_uses_its_own_prompt_and_value_table():
    binary, graded = make_scorer(mode=MODE_BINARY), make_scorer(mode=MODE_GRADED)
    assert binary.system_prompt == BINARY_SYSTEM_PROMPT
    assert graded.system_prompt == GRADED_SYSTEM_PROMPT
    assert binary.values == BINARY_VALUES
    assert graded.values == GRADE_VALUES


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unknown mode"):
        make_scorer(mode="sideways")


def test_the_prompt_states_the_product_goal_not_topical_relevance():
    # The target is a stranger scrolling a short-form feed; a prompt that drifts
    # to "is this interesting" would score ordinary conversation highly.
    for prompt in (BINARY_SYSTEM_PROMPT, GRADED_SYSTEM_PROMPT):
        lowered = prompt.lower()
        assert "tiktok" in lowered and "shorts" in lowered and "reels" in lowered
        assert "stranger" in lowered


def test_the_candidate_and_its_context_reach_the_prompt():
    scorer = make_scorer()
    content = scorer.build_user_content(
        candidate(), {"previous_context": "ПРЕДЫДУЩЕЕ", "next_context": "СЛЕДУЮЩЕЕ"}
    )
    assert "какой-то текст кандидата" in content
    assert "ПРЕДЫДУЩЕЕ" in content and "СЛЕДУЮЩЕЕ" in content
    # Context is labelled as context, so the model does not score it.
    assert "NOT part of the clip" in content


def test_missing_context_is_labelled_rather_than_left_blank():
    content = make_scorer().build_user_content(candidate(), None)
    assert "[nothing - start of stream]" in content
    assert "[nothing - end of stream]" in content


# --------------------------------------------------------------------------
# the request
# --------------------------------------------------------------------------


def test_the_request_asks_for_logprobs(patch_httpx):
    patch_httpx([FakeResponse(logprob_response([("no", -0.01), ("yes", -5.0)]))])
    make_scorer().score(candidate())

    payload = FakeClient.last_payload
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 20
    # The newer parameter name; `max_tokens` is rejected by recent models.
    assert payload["max_completion_tokens"] == 4
    assert "max_tokens" not in payload
    assert payload["temperature"] == 0.0


def test_temperature_can_be_omitted_for_models_that_refuse_it(patch_httpx):
    patch_httpx([FakeResponse(logprob_response([("no", -0.01), ("yes", -5.0)]))])
    make_scorer(temperature=None).score(candidate())
    assert "temperature" not in FakeClient.last_payload


def test_a_rejected_request_fails_loudly_and_is_not_retried(patch_httpx):
    # A model that cannot return logprobs at all must not be hammered.
    rejection = FakeResponse(
        {"error": {"message": "Unsupported parameter: 'logprobs' is not supported"}}, status_code=400
    )
    patch_httpx([rejection])

    with pytest.raises(LogprobScoringError, match="rejected the request"):
        make_scorer(model="gpt-5.6-luna").score(candidate())


def test_a_response_without_logprobs_is_an_error_not_a_zero(patch_httpx):
    payload = logprob_response([("no", -0.01)])
    payload["choices"][0]["logprobs"] = None
    patch_httpx([FakeResponse(payload)])

    with pytest.raises(LogprobScoringError, match="no logprobs"):
        make_scorer().score(candidate())


def test_a_response_with_no_answer_token_is_an_error(patch_httpx):
    patch_httpx([FakeResponse(logprob_response([("Maybe", -0.1), ("Perhaps", -1.0)]))])

    with pytest.raises(LogprobScoringError, match="no answer token"):
        make_scorer().score(candidate())


def test_a_transport_failure_is_retried(patch_httpx):
    patch_httpx([
        ConnectionError("dns went away"),
        ConnectionError("dns still away"),
        FakeResponse(logprob_response([("yes", -0.05), ("no", -3.0)])),
    ])

    result = make_scorer(max_retries=4).score(candidate())
    assert result.score == pytest.approx(95.03, abs=0.1)


def test_retries_are_bounded(patch_httpx):
    patch_httpx([ConnectionError("down")] * 3)

    with pytest.raises(LogprobScoringError, match="after 3 attempts"):
        make_scorer(max_retries=3).score(candidate())


def test_a_missing_api_key_is_reported_before_any_request(patch_httpx):
    patch_httpx([])
    scorer = LogprobBinaryScorer(base_url="https://api.test/v1", api_key=None)
    with pytest.raises(LogprobScoringError, match="no API key"):
        scorer.score(candidate())


# --------------------------------------------------------------------------
# the produced score
# --------------------------------------------------------------------------


def test_the_score_records_how_it_was_produced(patch_httpx):
    patch_httpx([FakeResponse(logprob_response([("yes", -0.05), ("no", -3.0)], answer="yes"))])

    result = make_scorer(model="gpt-4o-mini").score(candidate())

    assert result.requested_model == "gpt-4o-mini"
    assert result.actual_model == "gpt-4o-mini-2024-07-18"
    assert result.scorer_version == "logprob_v1_binary"
    assert result.fallback_used is False
    assert "E[binary]" in result.reason and "yes" in result.reason


def test_the_score_stays_inside_the_zero_to_hundred_range(patch_httpx):
    for pairs in ([("yes", 0.0)], [("no", 0.0)], [("yes", -0.7), ("no", -0.7)]):
        patch_httpx([FakeResponse(logprob_response(pairs))])
        result = make_scorer().score(candidate())
        assert 0.0 <= result.score <= 100.0


def test_a_confident_no_scores_near_zero_and_a_confident_yes_near_hundred(patch_httpx):
    patch_httpx([FakeResponse(logprob_response([("no", -0.0001), ("yes", -12.0)]))])
    assert make_scorer().score(candidate()).score < 1.0

    patch_httpx([FakeResponse(logprob_response([("yes", -0.0001), ("no", -12.0)]))])
    assert make_scorer().score(candidate()).score > 99.0


# --------------------------------------------------------------------------
# batch scoring
# --------------------------------------------------------------------------


def transcript_with(*windows) -> Transcript:
    segments = [
        TranscriptSegment(id=i, start=start, end=end, text=text)
        for i, (start, end, text) in enumerate(windows)
    ]
    return Transcript(
        language="ru", language_probability=0.85, duration=segments[-1].end,
        model="large-v3", compute_type="float16", device="cuda", segments=segments,
    )


def test_batch_scoring_feeds_each_candidate_its_surrounding_context(patch_httpx):
    transcript = transcript_with(
        (0.0, 30.0, "самое начало"), (30.0, 90.0, "сам кандидат"), (90.0, 120.0, "что было после"),
    )
    patch_httpx([FakeResponse(logprob_response([("no", -0.01), ("yes", -5.0)]))] * 2)

    scorer = make_scorer()
    scores = scorer.score_batch([candidate("cand_001", 30.0, 90.0), candidate("cand_002", 30.0, 90.0)], transcript)

    assert len(scores) == 2
    # The last request carried real context pulled from the transcript.
    sent = FakeClient.last_payload["messages"][1]["content"]
    assert "самое начало" in sent and "что было после" in sent


def test_batch_scoring_works_without_a_transcript(patch_httpx):
    patch_httpx([FakeResponse(logprob_response([("no", -0.01), ("yes", -5.0)]))])
    scores = make_scorer().score_batch([candidate()], transcript=None)
    assert len(scores) == 1
    assert "[nothing - start of stream]" in FakeClient.last_payload["messages"][1]["content"]


def test_batch_scoring_returns_one_score_per_candidate_in_order(patch_httpx):
    responses = [
        FakeResponse(logprob_response([("yes", -0.05), ("no", -3.0)])),
        FakeResponse(logprob_response([("no", -0.05), ("yes", -3.0)])),
        FakeResponse(logprob_response([("yes", -0.7), ("no", -0.7)])),
    ]
    patch_httpx(responses)

    scores = make_scorer().score_batch(
        [candidate("a"), candidate("b"), candidate("c")], transcript=None
    )

    assert [round(s.score) for s in scores] == [95, 5, 50]


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------


def test_the_cli_selects_the_mode_from_the_scorer_name():
    """Both arms must be reachable, since neither is settled."""
    import inspect

    from freecher_worker.cli import score_run_command

    source = inspect.getsource(score_run_command)
    assert '"logprob_graded"' in source and '"logprob_binary"' in source
    assert "MODE_GRADED if requested_scorer ==" in source


def test_the_frozen_scorer_keeps_its_default_temperature():
    """--temperature must be additive: unset means the old behaviour."""
    from freecher_worker.scoring.llm import OpenAILLMScorer

    scorer = OpenAILLMScorer(base_url=None, api_key="k", model="gpt-4o-mini")
    assert scorer.temperature == 0.1
