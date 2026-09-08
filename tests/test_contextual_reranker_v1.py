"""Unit and integration tests for Contextual Highlight Intelligence (contextual_reranker_v1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from freecher_worker.contextual import (
    COMPARISON_MODES,
    CONTEXT_VERSION,
    EDITORIAL_SCHEMA_VERSION,
    RERANKER_VERSION,
    SCORER_VERSION_CONTEXTUAL_V1,
    ChapterContext,
    ChapterContextDocument,
    ContextualCache,
    ContextualProvider,
    ContextualReranker,
    ContextualUsage,
    EditorialAnalysis,
    GlobalContext,
    build_candidate_context_package,
    build_chapter_plan,
    build_context_windows,
    build_global_context,
    build_chapter_contexts,
    build_listwise_batches,
    chapter_plan_hash,
    compute_global_context_hash,
    find_chapter_for_range,
    parse_critic_response,
    parse_editorial_response,
    parse_json_object,
    run_comparative_ranking,
    run_pairwise_comparison,
    seed_order,
)
from freecher_worker.contextual.prompts import get_stage_prompt
from freecher_worker.evaluation.models import (
    BlindEvaluationDocument,
    BlindEvaluationItem,
    ScorerPredictionDocument,
    ScorerPredictionItem,
)
from freecher_worker.evaluation.editorial_metrics import compute_editorial_metrics
from freecher_worker.highlights.models import CandidateDocument, CandidateWindow
from freecher_worker.transcription.models import Transcript, TranscriptSegment
from freecher_worker.utils.json_io import load_json, save_json


# ----------------------------------------------------------------------------------
# Fixtures and fakes
# ----------------------------------------------------------------------------------

CANDIDATE_COUNT = 20
SEGMENT_COUNT = 160
SEGMENT_LENGTH = 10.0
SOURCE_DURATION = SEGMENT_COUNT * SEGMENT_LENGTH


def make_transcript() -> Transcript:
    segments = [
        TranscriptSegment(
            id=i,
            start=i * SEGMENT_LENGTH,
            end=i * SEGMENT_LENGTH + 9.0,
            text=f"Segment {i} spoken line about topic {i % 5}.",
        )
        for i in range(SEGMENT_COUNT)
    ]
    return Transcript(
        source_fingerprint_id="fp_test",
        language="ru",
        duration=SOURCE_DURATION,
        model="small",
        compute_type="int8",
        device="cpu",
        segments=segments,
    )


def make_candidates(count: int = CANDIDATE_COUNT) -> List[CandidateWindow]:
    candidates = []
    for i in range(count):
        start = 100.0 + i * 60.0
        end = start + 50.0
        candidates.append(
            CandidateWindow(
                id=f"cand_{i + 1:03d}",
                start=start,
                end=end,
                duration=end - start,
                text=f"Candidate {i + 1} transcript body with a claim and a reaction.",
                segment_ids=[int(start // SEGMENT_LENGTH)],
            )
        )
    return candidates


class FakeProvider(ContextualProvider):
    """Deterministic provider that records every request and never touches the network."""

    def __init__(
        self,
        reject_ids: Optional[set[str]] = None,
        critic_reject_ids: Optional[set[str]] = None,
        model: str = "fake-model",
        fail_stages: Optional[set[str]] = None,
        bad_json_stages: Optional[set[str]] = None,
    ) -> None:
        self.name = "fake_contextual"
        self.model = model
        self.reasoning_effort = "none"
        self.temperature = 0.1
        self.effective_temperature = 0.1
        self.usage = ContextualUsage()
        self.reject_ids = reject_ids or set()
        self.critic_reject_ids = critic_reject_ids or set()
        self.fail_stages = fail_stages or set()
        self.bad_json_stages = bad_json_stages or set()
        self.calls: List[tuple[str, str]] = []

    def get_usage(self) -> ContextualUsage:
        return self.usage

    @staticmethod
    def _candidate_id(user_content: str) -> str:
        for line in user_content.splitlines():
            if line.startswith("Candidate ID: "):
                return line.split("Candidate ID: ", 1)[1].strip()
        return "unknown"

    @staticmethod
    def _listed_ids(user_content: str) -> List[str]:
        for line in user_content.splitlines():
            if line.startswith("Candidate ids to order"):
                raw = line.split(":", 1)[1]
                return [part.strip() for part in raw.split(",") if part.strip()]
        return []

    def complete_json(self, system_prompt: str, user_content: str, stage: str) -> Dict[str, Any]:
        self.calls.append((stage, user_content))
        self.usage.number_of_api_calls += 1
        self.usage.input_tokens += 100
        self.usage.output_tokens += 20
        self.usage.total_tokens = self.usage.input_tokens + self.usage.output_tokens
        self.usage.usage_reported_by_provider = True
        counter = {
            "chapter": "chapter_calls",
            "global_context": "global_context_calls",
            "editorial": "candidate_analysis_calls",
            "critic": "critic_calls",
            "listwise": "listwise_calls",
            "pairwise": "comparison_calls",
        }.get(stage)
        if counter:
            setattr(self.usage, counter, getattr(self.usage, counter) + 1)

        if stage in self.fail_stages:
            from freecher_worker.contextual.provider import ContextualProviderError

            self.usage.failed_calls += 1
            raise ContextualProviderError(f"synthetic failure in stage {stage}")
        if stage in self.bad_json_stages:
            return parse_json_object("this is definitely not json")  # raises ValueError

        if stage == "chapter":
            return {
                "summary": "A chapter where the participants set something up and then resolve it.",
                "participants": ["host", "guest"],
                "topic": "a running bet",
                "events": ["the host makes a claim"],
                "setups": ["a bet is proposed"],
                "payoffs": ["the bet is settled"],
                "open_loops": [],
            }
        if stage == "global_context":
            return {
                "video_summary": "Two people run a challenge across the whole recording.",
                "content_type": "stream",
                "participants": [
                    {"id": "person_1", "description": "the host", "role": "host"},
                    {"id": "person_2", "description": "the guest", "role": "guest"},
                ],
                "main_topics": ["the challenge"],
                "ongoing_goals": ["finish the challenge"],
                "recurring_jokes": ["the sauna bit"],
                "conflicts": ["host versus guest"],
                "important_context": ["the challenge has a prize"],
            }
        if stage == "editorial":
            cid = self._candidate_id(user_content)
            if cid in self.reject_ids:
                return {
                    "candidate_id": cid,
                    "editorial_class": "REJECT",
                    "scroll_stop": 0.1,
                    "reason_to_watch": None,
                    "reason_to_skip": "Ordinary conversation with no payoff.",
                    "reject_reasons": ["ordinary conversation", "no payoff"],
                    "confidence": 0.8,
                }
            index = int(cid.split("_")[-1])
            return {
                "candidate_id": cid,
                "editorial_class": "STRONG" if index <= 5 else "GOOD",
                "scroll_stop": round(1.0 - index * 0.02, 3),
                "hook": 0.7,
                "payoff": 0.8,
                "surprise": 0.6,
                "humor": 0.5,
                "tension": 0.4,
                "emotion": 0.6,
                "visual_interest": 0.5,
                "novelty": 0.6,
                "self_contained": 0.8,
                "shareability": 0.6,
                "context_dependency": 0.2,
                "dead_air": 0.1,
                "reason_to_watch": (
                    f"A claim in candidate {index} is immediately met with an unexpected reaction."
                ),
                "reason_to_skip": None,
                "reject_reasons": [],
                "confidence": 0.75,
            }
        if stage == "critic":
            cid = self._candidate_id(user_content)
            if cid in self.critic_reject_ids:
                return {
                    "candidate_id": cid,
                    "decision": "REJECT",
                    "reason": "Only positive signal is loudness.",
                    "confidence": 0.7,
                }
            return {
                "candidate_id": cid,
                "decision": "KEEP",
                "reason": "Concrete setup and payoff inside the clip.",
                "confidence": 0.7,
            }
        if stage == "listwise":
            ids = self._listed_ids(user_content)
            return {"ordering": sorted(ids), "reason": "Sorted deterministically for the test."}
        if stage == "pairwise":
            assert "### CANDIDATE A" in user_content and "### CANDIDATE B" in user_content
            # Candidate A always wins, deterministically.
            return {
                "winner": "A",
                "confidence": 0.8,
                "reason": "A has the clearer payoff.",
                "a_strength": "clear payoff",
                "b_strength": "some energy",
            }
        raise AssertionError(f"Unexpected stage {stage}")


def build_run(
    tmp_path: Path,
    candidate_count: int = CANDIDATE_COUNT,
    include_multimodal: bool = True,
) -> Path:
    """Create a minimal but complete run directory."""
    run_dir = tmp_path / "run"
    (run_dir / "scores").mkdir(parents=True, exist_ok=True)

    transcript = make_transcript()
    candidates = make_candidates(candidate_count)
    cand_doc = CandidateDocument(
        transcript_hash=transcript.compute_transcript_hash(),
        min_seconds=30.0,
        target_seconds=60.0,
        max_seconds=90.0,
        overlap_seconds=15.0,
        candidates=candidates,
    )
    save_json(cand_doc, run_dir / "candidates.json")
    save_json(transcript, run_dir / "transcript.json")
    save_json(
        {
            "source": "source.mp4",
            "source_fingerprint": {
                "fingerprint_id": "fp_test",
                "duration_seconds": SOURCE_DURATION,
            },
        },
        run_dir / "manifest.json",
    )

    def prediction_doc(scorer: str, ordered: List[CandidateWindow]) -> ScorerPredictionDocument:
        return ScorerPredictionDocument(
            candidate_set_id=cand_doc.candidate_set_id,
            scorer=scorer,
            scorer_version=scorer,
            model="upstream-model",
            predictions=[
                ScorerPredictionItem(
                    candidate_id=c.id,
                    rank=rank,
                    score=round(90.0 - rank * 2.0, 2),
                    reason=f"{scorer} rationale",
                    observable_event=True,
                    visual_payoff=rank % 2 == 0,
                    confidence=0.6,
                    audio_features={"rms_mean": 0.2, "speech_coverage": 0.8},
                    visual_features={"motion_score": 0.1, "scene_change_count": 1},
                )
                for rank, c in enumerate(ordered, start=1)
            ],
        )

    save_json(prediction_doc("heuristic_v1", candidates), run_dir / "scores" / "heuristic_v1.json")
    save_json(prediction_doc("highlight_v2_1", candidates), run_dir / "scores" / "highlight_v2_1.json")
    if include_multimodal:
        # Reverse order so rank movement is observable.
        save_json(
            prediction_doc("multimodal_v1_1", list(reversed(candidates))),
            run_dir / "scores" / "multimodal_v1_1.json",
        )
        save_json(
            {"candidate_ids": [c.id for c in candidates], "candidate_set_id": cand_doc.candidate_set_id},
            run_dir / "multimodal" / "shortlist_v1_1.json",
        )
    return run_dir


def make_cache(tmp_path: Path, **kwargs) -> ContextualCache:
    defaults = dict(
        run_dir=tmp_path,
        model="fake-model",
        reasoning_effort="none",
        temperature=0.1,
    )
    defaults.update(kwargs)
    return ContextualCache(**defaults)


# ----------------------------------------------------------------------------------
# 1. Global context deterministic cache identity
# ----------------------------------------------------------------------------------


def test_global_context_cache_identity_is_deterministic(tmp_path):
    """Rebuilding the same source produces the same context hash and issues no new calls."""
    transcript = make_transcript()
    plans = build_chapter_plan(transcript)
    cache = make_cache(tmp_path)

    provider_a = FakeProvider()
    chapters_a = build_chapter_contexts(plans, provider_a, cache, "fp_test")
    context_a = build_global_context(chapters_a, provider_a, cache, "fp_test", SOURCE_DURATION)
    first_calls = provider_a.usage.number_of_api_calls
    assert first_calls == len(plans) + 1

    provider_b = FakeProvider()
    chapters_b = build_chapter_contexts(plans, provider_b, cache, "fp_test")
    context_b = build_global_context(chapters_b, provider_b, cache, "fp_test", SOURCE_DURATION)

    assert provider_b.usage.number_of_api_calls == 0, "A second build must be served from cache"
    assert provider_b.usage.cache_hits == len(plans) + 1
    assert context_a.context_hash == context_b.context_hash
    assert context_a.context_hash == compute_global_context_hash(context_a, chapters_a)


def test_global_context_hash_ignores_creation_timestamp(tmp_path):
    """The context hash identifies content, not when it was built."""
    transcript = make_transcript()
    plans = build_chapter_plan(transcript)
    doc = ChapterContextDocument(source_fingerprint="fp_test", chapter_plan_hash=chapter_plan_hash(plans))
    a = GlobalContext(source_fingerprint="fp_test", video_summary="same", created_at="2020-01-01T00:00:00")
    b = GlobalContext(source_fingerprint="fp_test", video_summary="same", created_at="2030-01-01T00:00:00")
    assert compute_global_context_hash(a, doc) == compute_global_context_hash(b, doc)


# ----------------------------------------------------------------------------------
# 2. Chapter boundaries
# ----------------------------------------------------------------------------------


def test_chapter_boundaries_are_contiguous_and_within_bounds():
    transcript = make_transcript()
    plans = build_chapter_plan(transcript, target_seconds=240.0, min_seconds=150.0, max_seconds=420.0)

    assert len(plans) >= 2
    for plan in plans:
        assert plan.start < plan.end
        assert plan.duration <= 420.0 + SEGMENT_LENGTH
        assert plan.segments, "every chapter must own transcript segments"
    for earlier, later in zip(plans, plans[1:]):
        assert earlier.end <= later.start, "chapters must not overlap"

    all_ids = [sid for plan in plans for sid in plan.segment_ids]
    assert sorted(all_ids) == [s.id for s in transcript.segments], "every segment lands in exactly one chapter"
    assert plans[0].start == transcript.segments[0].start
    assert plans[-1].end == transcript.segments[-1].end


def test_chapter_short_tail_is_merged():
    segments = [
        TranscriptSegment(id=i, start=i * 10.0, end=i * 10.0 + 9.0, text=f"line {i}")
        for i in range(28)  # 280s: one full chapter plus a short tail
    ]
    transcript = Transcript(
        language="ru", duration=280.0, model="small", compute_type="int8", device="cpu", segments=segments
    )
    plans = build_chapter_plan(transcript, target_seconds=240.0, min_seconds=150.0, max_seconds=420.0)
    assert len(plans) == 1, "a sub-minimum tail chapter must merge into its predecessor"
    assert plans[0].end == segments[-1].end


def test_empty_transcript_produces_no_chapters():
    assert build_chapter_plan(None) == []
    empty = Transcript(language="ru", duration=0.0, model="small", compute_type="int8", device="cpu", segments=[])
    assert build_chapter_plan(empty) == []


# ----------------------------------------------------------------------------------
# 3 & 4. Candidate context windows
# ----------------------------------------------------------------------------------


def test_before_candidate_after_windows_are_not_swapped():
    transcript = make_transcript()
    candidate = CandidateWindow(
        id="cand_x", start=600.0, end=660.0, duration=60.0, text="the candidate body", segment_ids=[60]
    )
    before, cand, after = build_context_windows(candidate, transcript, SOURCE_DURATION)

    assert before.end == cand.start, "BEFORE must end exactly where the candidate starts"
    assert after.start == cand.end, "AFTER must start exactly where the candidate ends"
    assert before.start < before.end < after.start < after.end
    assert cand.transcript == "the candidate body"
    assert "Segment 59" in before.transcript and "Segment 60" not in before.transcript
    assert "Segment 66" in after.transcript and "Segment 65" not in after.transcript
    assert before.transcript != after.transcript


def test_context_windows_are_clamped_to_source_duration():
    transcript = make_transcript()

    at_start = CandidateWindow(id="c0", start=5.0, end=40.0, duration=35.0, text="opening", segment_ids=[0])
    before, cand, after = build_context_windows(at_start, transcript, SOURCE_DURATION)
    assert before.start == 0.0, "BEFORE must never go negative"
    assert before.duration <= 5.0

    at_end = CandidateWindow(
        id="cN",
        start=SOURCE_DURATION - 30.0,
        end=SOURCE_DURATION - 5.0,
        duration=25.0,
        text="closing",
        segment_ids=[SEGMENT_COUNT - 1],
    )
    before, cand, after = build_context_windows(at_end, transcript, SOURCE_DURATION)
    assert after.end <= SOURCE_DURATION, "AFTER must never exceed the source duration"
    assert cand.end <= SOURCE_DURATION


def test_context_window_lengths_match_configured_baselines():
    transcript = make_transcript()
    candidate = CandidateWindow(id="c", start=600.0, end=660.0, duration=60.0, text="body", segment_ids=[60])
    before, _, after = build_context_windows(
        candidate, transcript, SOURCE_DURATION, before_seconds=75.0, after_seconds=25.0
    )
    assert before.duration == pytest.approx(75.0)
    assert after.duration == pytest.approx(25.0)


def test_candidate_package_carries_context_evidence_and_provenance():
    transcript = make_transcript()
    plans = build_chapter_plan(transcript)
    chapters = [
        ChapterContext(
            chapter_id=plan.chapter_id,
            start=plan.start,
            end=plan.end,
            summary=f"summary of {plan.chapter_id}",
            topic="a topic",
            payoffs=["a payoff lands"],
        )
        for plan in plans
    ]
    global_context = GlobalContext(source_fingerprint="fp_test", video_summary="whole video")
    global_context.context_hash = "ctx_hash"

    candidate = CandidateWindow(
        id="cand_007", start=600.0, end=660.0, duration=60.0, text="body", segment_ids=[60]
    )
    multimodal_item = ScorerPredictionItem(
        candidate_id="cand_007",
        rank=10,
        score=62.0,
        observable_event=True,
        visual_payoff=False,
        outside_payoff=True,
        confidence=0.55,
        audio_features={"rms_mean": 0.3, "speech_coverage": 0.9, "ignored": 1},
        visual_features={"motion_score": 0.2, "ignored": 1},
        reason="upstream note",
    )
    package = build_candidate_context_package(
        candidate=candidate,
        transcript=transcript,
        global_context=global_context,
        chapter_contexts=chapters,
        chapter_plans=plans,
        source_duration=SOURCE_DURATION,
        multimodal_item=multimodal_item,
        retrieval=None,
    )

    expected_chapter = find_chapter_for_range(plans, candidate.start, candidate.end)
    assert package.chapter_id == expected_chapter.chapter_id
    assert package.chapter_context.summary.endswith(expected_chapter.chapter_id)
    assert package.global_context_ref == "ctx_hash"
    assert package.multimodal_evidence["observable_event"] is True
    assert package.multimodal_evidence["outside_payoff"] is True
    assert "ignored" not in package.multimodal_evidence["audio_features"]
    assert package.multimodal_evidence["upstream_reason"] == "upstream note"
    assert package.reaction_signals is not None and not package.reaction_signals.available
    assert package.package_hash

    same = build_candidate_context_package(
        candidate=candidate,
        transcript=transcript,
        global_context=global_context,
        chapter_contexts=chapters,
        chapter_plans=plans,
        source_duration=SOURCE_DURATION,
        multimodal_item=multimodal_item,
        retrieval=None,
    )
    assert same.package_hash == package.package_hash, "package hash must be content-addressed"


def test_seed_order_prefers_stronger_editorial_classes():
    analyses = {
        "c_weak": EditorialAnalysis(candidate_id="c_weak", editorial_class="WEAK", scroll_stop=0.9),
        "c_good": EditorialAnalysis(candidate_id="c_good", editorial_class="GOOD", scroll_stop=0.1),
        "c_strong": EditorialAnalysis(candidate_id="c_strong", editorial_class="STRONG", scroll_stop=0.1),
        "c_strong2": EditorialAnalysis(candidate_id="c_strong2", editorial_class="STRONG", scroll_stop=0.8),
    }
    assert seed_order(list(analyses), analyses) == ["c_strong2", "c_strong", "c_good", "c_weak"]
    # Deterministic regardless of input order.
    assert seed_order(sorted(analyses, reverse=True), analyses) == seed_order(list(analyses), analyses)


# ----------------------------------------------------------------------------------
# 5 & 6. Response parsing
# ----------------------------------------------------------------------------------


def test_reject_result_parses_correctly():
    analysis = parse_editorial_response(
        {
            "candidate_id": "cand_006",
            "editorial_class": "REJECT",
            "scroll_stop": 0.05,
            "reason_to_watch": None,
            "reason_to_skip": "Nothing happens.",
            "reject_reasons": ["ordinary conversation", "no payoff"],
            "confidence": 0.9,
        },
        "cand_006",
        "m",
        "p",
    )
    assert analysis.editorial_class == "REJECT"
    assert analysis.reject_reasons == ["ordinary conversation", "no payoff"]
    assert analysis.reason_to_watch is None
    assert analysis.confidence == pytest.approx(0.9)
    assert not analysis.parse_failed


def test_editorial_scores_accept_both_unit_and_percent_scales():
    unit = parse_editorial_response(
        {"editorial_class": "STRONG", "scroll_stop": 0.86, "reason_to_watch": "A claim draws an unexpected reaction."},
        "c",
        "m",
        "p",
    )
    percent = parse_editorial_response(
        {"editorial_class": "STRONG", "scroll_stop": 86, "reason_to_watch": "A claim draws an unexpected reaction."},
        "c",
        "m",
        "p",
    )
    assert unit.scroll_stop == pytest.approx(0.86)
    assert percent.scroll_stop == pytest.approx(0.86)


def test_invalid_api_json_is_handled_safely(tmp_path):
    """A parse failure never rejects or promotes a candidate; it degrades to WEAK."""
    run_dir = build_run(tmp_path)
    provider = FakeProvider(bad_json_stages={"editorial"})
    reranker = ContextualReranker(provider=provider, comparison_mode="none", critic_enabled=False)
    pred_doc, rerank_doc = reranker.rerank_run(run_dir)

    assert rerank_doc.survivor_count == CANDIDATE_COUNT
    assert all(record.editorial_class == "WEAK" for record in rerank_doc.results)
    assert all(record.editorial_parse_failed for record in rerank_doc.results)
    assert all(record.confidence == 0.0 for record in rerank_doc.results)
    assert len(pred_doc.predictions) == CANDIDATE_COUNT


def test_provider_failure_is_handled_safely(tmp_path):
    """A transport failure in every stage still produces a complete, ranked artifact."""
    run_dir = build_run(tmp_path)
    provider = FakeProvider(fail_stages={"chapter", "global_context", "editorial", "critic", "listwise", "pairwise"})
    reranker = ContextualReranker(provider=provider)
    pred_doc, rerank_doc = reranker.rerank_run(run_dir)

    assert len(pred_doc.predictions) == CANDIDATE_COUNT
    assert rerank_doc.usage.failed_calls > 0
    ranks = [item.rank for item in pred_doc.predictions]
    assert ranks == sorted(ranks)


def test_parse_json_object_rejects_non_objects():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 2}\n```') == {"a": 2}
    with pytest.raises(ValueError):
        parse_json_object("[1, 2, 3]")
    with pytest.raises(ValueError):
        parse_json_object("")


def test_critic_unreadable_verdict_defaults_to_keep():
    result = parse_critic_response({"decision": "maybe?", "confidence": 0.4}, "c", "m", "p")
    assert result.decision == "KEEP", "an unreadable critic verdict must not silently cut a candidate"
    assert result.parse_failed


# ----------------------------------------------------------------------------------
# 7-11. Ranking guarantees
# ----------------------------------------------------------------------------------


def test_rejected_candidates_never_reach_the_comparative_stage(tmp_path):
    run_dir = build_run(tmp_path)
    rejected = {"cand_001", "cand_002", "cand_003"}
    provider = FakeProvider(reject_ids=rejected)
    reranker = ContextualReranker(provider=provider)
    pred_doc, rerank_doc = reranker.rerank_run(run_dir)

    compared_ids = {c.candidate_a for c in rerank_doc.comparisons} | {
        c.candidate_b for c in rerank_doc.comparisons
    }
    assert not (compared_ids & rejected), "a rejected candidate must never be compared"

    listwise_ids = {cid for batch in rerank_doc.listwise_batches for cid in batch.candidate_ids}
    assert not (listwise_ids & rejected)

    ranked_ids = {record.candidate_id for record in rerank_doc.results}
    assert not (ranked_ids & rejected)
    assert {record.candidate_id for record in rerank_doc.rejected} == rejected


def test_rejected_candidates_rank_below_every_survivor(tmp_path):
    run_dir = build_run(tmp_path)
    provider = FakeProvider(reject_ids={"cand_001", "cand_005"}, critic_reject_ids={"cand_009"})
    reranker = ContextualReranker(provider=provider, top=5)
    pred_doc, rerank_doc = reranker.rerank_run(run_dir)

    survivor_ranks = [r.rank for r in rerank_doc.results]
    rejected_ranks = [r.rank for r in rerank_doc.rejected]
    assert max(survivor_ranks) < min(rejected_ranks)

    top_k = [item for item in pred_doc.predictions if item.rank <= 5]
    assert all(item.status == "ranked" for item in top_k), "no rejected candidate may enter Top-K"
    assert all(item.score >= 45.0 for item in top_k)

    critic_rejected = [r for r in rerank_doc.rejected if r.candidate_id == "cand_009"]
    assert critic_rejected and critic_rejected[0].status == "rejected_critic"


def test_comparative_winner_is_deterministic_with_cached_responses(tmp_path):
    run_dir = build_run(tmp_path)
    first = ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)[1]
    # Second run: a provider that would explode if consulted, proving cache determinism.
    second_provider = FakeProvider(fail_stages={"editorial", "critic", "listwise", "pairwise", "chapter", "global_context"})
    second = ContextualReranker(provider=second_provider).rerank_run(run_dir)[1]

    assert second_provider.usage.number_of_api_calls == 0
    assert [r.candidate_id for r in first.results] == [r.candidate_id for r in second.results]
    assert [r.comparison_score for r in first.results] == [r.comparison_score for r in second.results]


def test_candidate_is_never_compared_with_itself(tmp_path):
    run_dir = build_run(tmp_path)
    provider = FakeProvider()
    _, rerank_doc = ContextualReranker(provider=provider).rerank_run(run_dir)

    assert rerank_doc.comparisons, "the full mode must actually run comparisons"
    for comparison in rerank_doc.comparisons:
        assert comparison.candidate_a != comparison.candidate_b

    with pytest.raises(ValueError):
        run_pairwise_comparison("cand_001", "cand_001", {}, None, make_cache(tmp_path), "test")


def test_ranking_contains_no_duplicate_candidate_ids(tmp_path):
    run_dir = build_run(tmp_path)
    pred_doc, rerank_doc = ContextualReranker(provider=FakeProvider(reject_ids={"cand_004"})).rerank_run(run_dir)

    all_ids = [item.candidate_id for item in pred_doc.predictions]
    assert len(all_ids) == len(set(all_ids))
    ranks = [item.rank for item in pred_doc.predictions]
    assert sorted(ranks) == list(range(1, len(ranks) + 1))


def test_ranking_contains_only_input_candidate_set(tmp_path):
    run_dir = build_run(tmp_path)
    pred_doc, _ = ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)

    cand_doc = CandidateDocument.model_validate(load_json(run_dir / "candidates.json"))
    known = {c.id for c in cand_doc.candidates}
    produced = {item.candidate_id for item in pred_doc.predictions}
    assert produced <= known
    assert pred_doc.candidate_set_id == cand_doc.candidate_set_id


def test_comparison_count_is_far_below_full_pairwise(tmp_path):
    """The schedule must stay O(N log N), not O(N^2)."""
    run_dir = build_run(tmp_path, candidate_count=32)
    _, rerank_doc = ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)

    n = rerank_doc.survivor_count
    full_pairwise = n * (n - 1) // 2
    assert len(rerank_doc.comparisons) < full_pairwise / 2
    assert len(rerank_doc.comparisons) <= n * 4


@pytest.mark.parametrize("mode", COMPARISON_MODES)
def test_every_comparison_mode_produces_a_complete_ranking(tmp_path, mode):
    run_dir = build_run(tmp_path)
    pred_doc, rerank_doc = ContextualReranker(provider=FakeProvider(), comparison_mode=mode).rerank_run(run_dir)
    assert len(pred_doc.predictions) == CANDIDATE_COUNT
    assert rerank_doc.comparison_mode == mode
    if mode == "none":
        assert not rerank_doc.comparisons
    if mode in ("swiss", "none"):
        assert not rerank_doc.listwise_batches


def test_comparative_ranking_handles_a_single_survivor(tmp_path):
    analyses = {"only": EditorialAnalysis(candidate_id="only", editorial_class="GOOD")}
    ranking = run_comparative_ranking(["only"], {}, analyses, None, make_cache(tmp_path))
    assert ranking.ordered_ids == ["only"]
    assert not ranking.comparisons


def test_listwise_batches_partition_without_loss():
    ids = [f"c{i}" for i in range(16)]
    batches = build_listwise_batches(ids, 7)
    flattened = [cid for batch in batches for cid in batch]
    assert sorted(flattened) == sorted(ids)
    assert all(2 <= len(batch) <= 7 for batch in batches)


# ----------------------------------------------------------------------------------
# 12. Cache invalidation
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "other-model"},
        {"reasoning_effort": "high"},
        {"temperature": 0.9},
        {"context_version": "contextual_context_v2"},
        {"reranker_version": "contextual_reranker_v2"},
    ],
)
def test_cache_key_changes_with_model_and_version(tmp_path, kwargs):
    base = make_cache(tmp_path)
    changed = make_cache(tmp_path, **kwargs)
    payload = {"candidate_id": "c1"}
    args = ("editorial", "pv", "ph", EDITORIAL_SCHEMA_VERSION, payload)
    assert base.request_hash(*args) != changed.request_hash(*args)


def test_cache_key_changes_with_prompt_and_schema(tmp_path):
    cache = make_cache(tmp_path)
    payload = {"candidate_id": "c1"}
    base = cache.request_hash("editorial", "pv1", "ph1", EDITORIAL_SCHEMA_VERSION, payload)
    assert cache.request_hash("editorial", "pv2", "ph1", EDITORIAL_SCHEMA_VERSION, payload) != base
    assert cache.request_hash("editorial", "pv1", "ph2", EDITORIAL_SCHEMA_VERSION, payload) != base
    assert cache.request_hash("editorial", "pv1", "ph1", "editorial_analysis_v2", payload) != base
    assert cache.request_hash("critic", "pv1", "ph1", EDITORIAL_SCHEMA_VERSION, payload) != base


def test_prompt_edit_invalidates_only_its_own_stage(tmp_path):
    cache = make_cache(tmp_path)
    payload = {"candidate_id": "c1"}
    editorial_pv, editorial_ph, _ = get_stage_prompt("editorial")
    critic_pv, critic_ph, _ = get_stage_prompt("critic")

    editorial_key = cache.request_hash("editorial", editorial_pv, editorial_ph, EDITORIAL_SCHEMA_VERSION, payload)
    critic_key = cache.request_hash("critic", critic_pv, critic_ph, EDITORIAL_SCHEMA_VERSION, payload)
    edited_editorial = cache.request_hash("editorial", editorial_pv, "edited_hash", EDITORIAL_SCHEMA_VERSION, payload)

    assert edited_editorial != editorial_key
    assert critic_key == cache.request_hash("critic", critic_pv, critic_ph, EDITORIAL_SCHEMA_VERSION, payload)


def test_force_bypasses_the_cache(tmp_path):
    cache = make_cache(tmp_path)
    req = cache.request_hash("editorial", "pv", "ph", EDITORIAL_SCHEMA_VERSION, {"a": 1})
    cache.store("editorial", req, {"editorial_class": "GOOD"})
    assert cache.load("editorial", req) == {"editorial_class": "GOOD"}

    forced = make_cache(tmp_path, force=True)
    assert forced.load("editorial", req) is None


def test_corrupt_cache_entry_is_treated_as_a_miss(tmp_path):
    cache = make_cache(tmp_path)
    req = cache.request_hash("editorial", "pv", "ph", EDITORIAL_SCHEMA_VERSION, {"a": 1})
    path = cache.path_for("editorial", req)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not valid json", encoding="utf-8")
    assert cache.load("editorial", req) is None


def test_global_and_chapter_context_are_not_recomputed_on_rerank(tmp_path):
    """A second rerank with --force still reuses nothing, but a normal rerun reuses everything."""
    run_dir = build_run(tmp_path)
    first = FakeProvider()
    ContextualReranker(provider=first).rerank_run(run_dir)
    assert first.usage.chapter_calls > 0 and first.usage.global_context_calls == 1

    second = FakeProvider()
    ContextualReranker(provider=second).rerank_run(run_dir)
    assert second.usage.chapter_calls == 0
    assert second.usage.global_context_calls == 0
    assert second.usage.cache_hits > 0


# ----------------------------------------------------------------------------------
# 13. Backward compatibility of existing scorers
# ----------------------------------------------------------------------------------


def test_existing_scorer_artifacts_are_not_modified(tmp_path):
    run_dir = build_run(tmp_path)
    paths = [
        run_dir / "scores" / "heuristic_v1.json",
        run_dir / "scores" / "highlight_v2_1.json",
        run_dir / "scores" / "multimodal_v1_1.json",
        run_dir / "candidates.json",
        run_dir / "transcript.json",
    ]
    before = {path: path.read_bytes() for path in paths}

    ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)

    for path in paths:
        assert path.read_bytes() == before[path], f"{path.name} must not be rewritten"
    assert (run_dir / "scores" / f"{SCORER_VERSION_CONTEXTUAL_V1}.json").is_file()


def test_existing_prediction_documents_still_validate():
    """Documents written before the contextual fields existed remain valid."""
    legacy = {
        "candidate_set_id": "cset_x",
        "scorer": "highlight_v2_1",
        "scorer_version": "highlight_v2_1",
        "predictions": [{"candidate_id": "cand_001", "rank": 1, "score": 72.5}],
    }
    doc = ScorerPredictionDocument.model_validate(legacy)
    assert doc.predictions[0].editorial_class is None
    assert doc.predictions[0].contextual is None
    assert doc.usage is None
    assert doc.survivor_count is None


def test_missing_input_scorer_fails_explicitly(tmp_path):
    run_dir = build_run(tmp_path, include_multimodal=False)
    with pytest.raises(FileNotFoundError, match="multimodal_v1_1"):
        ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)


def test_legacy_bare_list_candidates_are_supported(tmp_path):
    """Older runs store candidates.json as a bare list; score-run accepts it, so must this."""
    run_dir = build_run(tmp_path)
    cand_doc = CandidateDocument.model_validate(load_json(run_dir / "candidates.json"))
    save_json([c.model_dump(mode="json") for c in cand_doc.candidates], run_dir / "candidates.json")

    pred_doc, rerank_doc = ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)
    assert rerank_doc.candidate_set_id == "legacy_cset"
    assert len(pred_doc.predictions) == CANDIDATE_COUNT


def test_malformed_candidates_file_fails_explicitly(tmp_path):
    run_dir = build_run(tmp_path)
    save_json("not a candidate set", run_dir / "candidates.json")
    with pytest.raises(ValueError, match="Invalid candidates format"):
        ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)


def test_input_scorer_can_be_switched(tmp_path):
    run_dir = build_run(tmp_path, include_multimodal=False)
    pred_doc, rerank_doc = ContextualReranker(
        provider=FakeProvider(), input_scorer="highlight_v2_1"
    ).rerank_run(run_dir)
    assert rerank_doc.input_scorer == "highlight_v2_1"
    assert len(pred_doc.predictions) == CANDIDATE_COUNT


# ----------------------------------------------------------------------------------
# Observability, artifacts, and usage
# ----------------------------------------------------------------------------------


def test_rank_movement_and_provenance_are_observable(tmp_path):
    run_dir = build_run(tmp_path)
    pred_doc, rerank_doc = ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)

    for record in rerank_doc.results:
        assert record.previous_multimodal_rank is not None
        assert record.previous_rank is not None
        assert record.rank_delta == record.previous_rank - record.new_rank
        assert record.editorial_dimensions
        assert record.critic_result in ("KEEP", "REJECT", "NOT_RUN")

    # multimodal ranks were reversed in the fixture, so something must have moved up.
    assert any(record.rank_delta and record.rank_delta > 0 for record in rerank_doc.results)

    item = pred_doc.predictions[0]
    assert item.editorial_class and item.reason_to_watch
    assert item.contextual and item.contextual["candidate_id"] == item.candidate_id


def test_artifact_files_are_written(tmp_path):
    run_dir = build_run(tmp_path)
    ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)

    for relative in (
        f"scores/{SCORER_VERSION_CONTEXTUAL_V1}.json",
        "contextual/global_context_v1.json",
        "contextual/chapter_context_v1.json",
        "contextual/candidate_context_v1.json",
        f"contextual/{SCORER_VERSION_CONTEXTUAL_V1}_run.json",
    ):
        assert (run_dir / relative).is_file(), f"missing artifact {relative}"

    record = load_json(run_dir / "contextual" / f"{SCORER_VERSION_CONTEXTUAL_V1}_run.json")
    assert record["reranker_version"] == RERANKER_VERSION
    assert record["context_version"] == CONTEXT_VERSION
    assert record["global_context_ref"]
    assert record["retrieval_candidate_count"] == CANDIDATE_COUNT
    assert record["survivor_count"] + record["rejected_count"] == CANDIDATE_COUNT


def test_api_and_token_usage_is_observable(tmp_path):
    run_dir = build_run(tmp_path)
    provider = FakeProvider()
    pred_doc, rerank_doc = ContextualReranker(provider=provider).rerank_run(run_dir)

    usage = rerank_doc.usage
    assert usage.number_of_api_calls > 0
    assert usage.input_tokens > 0 and usage.output_tokens > 0
    assert usage.candidate_analysis_calls == CANDIDATE_COUNT
    assert usage.critic_calls == CANDIDATE_COUNT
    assert usage.comparison_calls == len(rerank_doc.comparisons)
    assert usage.listwise_calls == len(rerank_doc.listwise_batches)
    assert usage.global_context_calls == 1
    assert (
        usage.number_of_api_calls
        == usage.chapter_calls
        + usage.global_context_calls
        + usage.candidate_analysis_calls
        + usage.critic_calls
        + usage.listwise_calls
        + usage.comparison_calls
    )
    assert pred_doc.usage["number_of_api_calls"] == usage.number_of_api_calls


def test_critic_can_be_disabled(tmp_path):
    run_dir = build_run(tmp_path)
    provider = FakeProvider(critic_reject_ids={"cand_002"})
    _, rerank_doc = ContextualReranker(provider=provider, critic_enabled=False).rerank_run(run_dir)
    assert provider.usage.critic_calls == 0
    assert all(record.critic_result == "NOT_RUN" for record in rerank_doc.results)
    assert "cand_002" in {r.candidate_id for r in rerank_doc.results}


def test_all_rejected_by_critic_restores_strongest_candidates(tmp_path):
    run_dir = build_run(tmp_path)
    all_ids = {f"cand_{i:03d}" for i in range(1, CANDIDATE_COUNT + 1)}
    reranker = ContextualReranker(provider=FakeProvider(critic_reject_ids=all_ids))
    _, rerank_doc = reranker.rerank_run(run_dir)

    assert rerank_doc.results, "the ranking must never be empty"
    assert reranker.warnings and "restored" in reranker.warnings[0]


def test_human_labels_never_reach_the_provider(tmp_path):
    """Nothing resembling a human label may appear in any prompt."""
    run_dir = build_run(tmp_path)
    eval_doc = BlindEvaluationDocument(
        candidate_set_id="cset",
        total_candidates=1,
        items=[
            BlindEvaluationItem(
                candidate_id="cand_001",
                start=100.0,
                end=150.0,
                duration=50.0,
                text="body",
                human_score=4.0,
                publishable=True,
                human_notes="a perfect clip",
            )
        ],
    )
    save_json(eval_doc, run_dir / "evaluation.json")

    provider = FakeProvider()
    ContextualReranker(provider=provider).rerank_run(run_dir)

    for stage, content in provider.calls:
        lowered = content.lower()
        for forbidden in ("human_score", "human score", "publishable", "human_notes", "a perfect clip"):
            assert forbidden not in lowered, f"'{forbidden}' leaked into the {stage} prompt"


def test_editorial_metrics_report_reject_and_strong_precision(tmp_path):
    run_dir = build_run(tmp_path)
    pred_doc, _ = ContextualReranker(
        provider=FakeProvider(reject_ids={"cand_010", "cand_011"})
    ).rerank_run(run_dir)

    cand_doc = CandidateDocument.model_validate(load_json(run_dir / "candidates.json"))
    items = []
    for candidate in cand_doc.candidates:
        index = int(candidate.id.split("_")[-1])
        # Rejected candidates are genuinely bad; the first few are genuinely strong.
        human = 1.0 if index in (10, 11) else (4.0 if index <= 5 else 2.0)
        items.append(
            BlindEvaluationItem(
                candidate_id=candidate.id,
                start=candidate.start,
                end=candidate.end,
                duration=candidate.duration,
                text=candidate.text,
                human_score=human,
                publishable=human >= 3.0,
            )
        )
    eval_doc = BlindEvaluationDocument(
        candidate_set_id=cand_doc.candidate_set_id,
        total_candidates=len(items),
        items=items,
    )
    eval_doc.update_labeled_count()

    metrics = compute_editorial_metrics(eval_doc, pred_doc)
    assert metrics.reject_precision == pytest.approx(1.0)
    assert metrics.strong_precision == pytest.approx(1.0)
    assert metrics.class_distribution["REJECT"] == 2
    assert metrics.class_distribution["STRONG"] == 5
    assert metrics.mean_human_by_class["STRONG"] == pytest.approx(4.0)


def test_editorial_metrics_reject_mismatched_candidate_set():
    eval_doc = BlindEvaluationDocument(candidate_set_id="a", total_candidates=0, items=[])
    pred_doc = ScorerPredictionDocument(candidate_set_id="b", scorer="s", scorer_version="v")
    with pytest.raises(ValueError, match="Candidate set ID mismatch"):
        compute_editorial_metrics(eval_doc, pred_doc)
