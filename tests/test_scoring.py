"""Tests for heuristic and LLM scoring modules."""

from arny_worker.highlights.models import CandidateWindow
from arny_worker.scoring.heuristic import HeuristicScorer
from arny_worker.scoring.llm import OpenAILLMScorer


def test_heuristic_empty_text():
    scorer = HeuristicScorer()
    cand = CandidateWindow(id="c0", start=0.0, end=30.0, duration=30.0, text="...", segment_ids=[0])
    score = scorer.score(cand)
    assert score.score <= 20.0
    assert "no speech words" in score.reason


def test_heuristic_hook_and_emotion():
    scorer = HeuristicScorer()
    text = (
        "Почему 90% людей совершают эту ошибку? Вау, это просто невероятно! "
        "Представьте, что 15 тысяч рублей в месяц можно сохранить благодаря этому правилу. "
        "Это шок и полная жесть!"
    )
    cand = CandidateWindow(id="c1", start=0.0, end=50.0, duration=50.0, text=text, segment_ids=[0, 1])
    score = scorer.score(cand)

    assert score.hook_score >= 60.0
    assert score.emotion_score >= 60.0
    assert score.information_score >= 50.0
    assert score.score >= 50.0
    assert 0.0 <= score.score <= 100.0
    assert "opening question hook" in score.reason or "emotional keywords" in score.reason


def test_heuristic_pacing_penalty():
    scorer = HeuristicScorer()
    # 6 words in 60 seconds = very slow pacing (6 WPM)
    cand = CandidateWindow(
        id="c_slow",
        start=0.0,
        end=60.0,
        duration=60.0,
        text="Раз два три четыре пять шесть.",
        segment_ids=[0],
    )
    score = scorer.score(cand)
    assert "slow speech density" in score.reason


def test_llm_scorer_fallback_without_api_key():
    llm_scorer = OpenAILLMScorer(api_key=None)
    cand = CandidateWindow(
        id="c_test",
        start=0.0,
        end=45.0,
        duration=45.0,
        text="Как заработать первый миллион рублей? Это главный секрет успеха!",
        segment_ids=[0],
    )
    score = llm_scorer.score(cand)
    assert score.score > 0.0
    assert 0.0 <= score.score <= 100.0
