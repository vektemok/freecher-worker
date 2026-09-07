"""Tests for LLM fallback observability."""

from freecher_worker.highlights.models import CandidateWindow
from freecher_worker.scoring.llm import OpenAILLMScorer


def test_llm_fallback_flag_and_reason_on_missing_api_key():
    scorer = OpenAILLMScorer(api_key=None)
    cand = CandidateWindow(
        id="cand_test",
        start=0.0,
        end=45.0,
        duration=45.0,
        text="Почему это важно? Мы открыли невероятный факт!",
        segment_ids=[0],
    )
    score = scorer.score(cand)

    assert score.fallback_used is True
    assert score.fallback_reason is not None
    assert "FREECHER_LLM_API_KEY is not set" in score.fallback_reason
    assert score.score > 0.0


def test_llm_fallback_on_network_error(monkeypatch):
    scorer = OpenAILLMScorer(base_url="http://invalid-unreachable-host:9999", api_key="sk-dummy")
    cand = CandidateWindow(
        id="cand_test2",
        start=0.0,
        end=45.0,
        duration=45.0,
        text="Как заработать первый миллион рублей? Это главный секрет успеха!",
        segment_ids=[0],
    )
    score = scorer.score(cand)

    assert score.fallback_used is True
    assert score.fallback_reason is not None
    assert "ConnectError" in score.fallback_reason or "Error" in score.fallback_reason
    assert score.score > 0.0
