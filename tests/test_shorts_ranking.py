"""Ranking source selection.

Regression coverage for the reported defect: `render-shorts` rendered the heuristic_v1 top-5
even though a multimodal_v1_1 reranking existed, because `highlights.json` was preferred over
`scores/multimodal_v1_1.json`.
"""

import json
from pathlib import Path

import pytest

from freecher_worker.shorts.render import (
    RANKING_SOURCE_AUTO,
    RANKING_SOURCE_HIGHLIGHTS,
    SCORER_PREFERENCE,
    load_run_context,
    resolve_ranking_source,
    resolve_timeframe,
    select_ranked_candidates,
)

# heuristic_v1 and multimodal_v1_1 deliberately disagree, as they did on benchmark_02.
HEURISTIC_TOP = [("cand_033", 81.7), ("cand_001", 78.5), ("cand_010", 77.9),
                 ("cand_070", 77.4), ("cand_025", 77.3)]
MULTIMODAL_TOP = [("cand_080", 74.0), ("cand_010", 71.2), ("cand_033", 68.9),
                  ("cand_004", 66.5), ("cand_055", 64.1)]
ALL_IDS = sorted({cid for cid, _ in HEURISTIC_TOP + MULTIMODAL_TOP})


def build_run(tmp_path: Path, with_multimodal: bool = True) -> Path:
    run_dir = tmp_path / "runs" / "benchmark_02"
    run_dir.mkdir(parents=True)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00" * 32)

    (run_dir / "manifest.json").write_text(json.dumps({
        "source": str(source),
        "source_fingerprint": {"fingerprint_id": "fp", "duration_seconds": 3600.0},
    }))
    (run_dir / "candidates.json").write_text(json.dumps({
        "transcript_hash": "h", "min_seconds": 30.0, "target_seconds": 60.0,
        "max_seconds": 90.0, "overlap_seconds": 15.0,
        "candidates": [
            {"id": cid, "start": 100.0 + i * 70, "end": 160.0 + i * 70,
             "duration": 60.0, "text": cid, "segment_ids": []}
            for i, cid in enumerate(ALL_IDS)
        ],
    }))
    # highlights.json is the heuristic pipeline's own top-K.
    (run_dir / "highlights.json").write_text(json.dumps([
        {"rank": r, "start": 100.0, "end": 160.0, "duration": 60.0, "score": score,
         "reason": "heuristic_v1", "candidate_id": cid, "text": cid}
        for r, (cid, score) in enumerate(HEURISTIC_TOP, 1)
    ]))

    scores = run_dir / "scores"
    scores.mkdir()
    (scores / "heuristic_v1.json").write_text(json.dumps({
        "candidate_set_id": "cset", "scorer": "heuristic", "scorer_version": "heuristic_v1",
        "predictions": [
            {"candidate_id": cid, "rank": r, "score": score}
            for r, (cid, score) in enumerate(HEURISTIC_TOP, 1)
        ],
    }))
    if with_multimodal:
        (scores / "multimodal_v1_1.json").write_text(json.dumps({
            "candidate_set_id": "cset", "scorer": "multimodal_v1_1",
            "scorer_version": "multimodal_v1_1", "model": "gpt-5.6-luna",
            "predictions": [
                {"candidate_id": cid, "rank": r, "score": score,
                 "best_observed_region": {"start_offset": 5.0, "end_offset": 30.0}}
                for r, (cid, score) in enumerate(MULTIMODAL_TOP, 1)
            ],
        }))
    return run_dir


def test_multimodal_ranking_wins_when_it_disagrees_with_heuristic(tmp_path):
    """The reported bug: heuristic top-5 was rendered instead of the multimodal top-5."""
    context = load_run_context(build_run(tmp_path))
    heuristic_top = [cid for cid, _ in HEURISTIC_TOP]
    multimodal_top = [cid for cid, _ in MULTIMODAL_TOP]
    assert heuristic_top[0] != multimodal_top[0], "fixture must actually disagree"

    selected, source = select_ranked_candidates(context, top=5, scorer=RANKING_SOURCE_AUTO)

    assert source.name == "multimodal_v1_1"
    assert selected == multimodal_top
    assert selected[0] == "cand_080"
    assert selected != heuristic_top


def test_explicit_multimodal_scorer_selects_the_multimodal_top(tmp_path):
    context = load_run_context(build_run(tmp_path))
    selected, source = select_ranked_candidates(context, top=5, scorer="multimodal_v1_1")

    assert source.name == "multimodal_v1_1"
    assert source.origin == "scores/multimodal_v1_1.json"
    assert source.model == "gpt-5.6-luna"
    assert selected[0] == "cand_080"
    assert source.rank_of("cand_080") == 1
    assert source.score_of("cand_080") == pytest.approx(74.0)


def test_heuristic_can_still_be_requested_explicitly(tmp_path):
    context = load_run_context(build_run(tmp_path))
    selected, source = select_ranked_candidates(context, top=5, scorer="heuristic_v1")

    assert source.name == "heuristic_v1"
    assert selected == [cid for cid, _ in HEURISTIC_TOP]


def test_legacy_highlights_ranking_can_be_requested(tmp_path):
    context = load_run_context(build_run(tmp_path))
    selected, source = select_ranked_candidates(context, top=3, scorer=RANKING_SOURCE_HIGHLIGHTS)

    assert source.name == RANKING_SOURCE_HIGHLIGHTS
    assert source.origin == "highlights.json"
    assert selected == [cid for cid, _ in HEURISTIC_TOP[:3]]


def test_auto_falls_back_to_heuristic_when_no_multimodal_ranking_exists(tmp_path):
    context = load_run_context(build_run(tmp_path, with_multimodal=False))
    selected, source = select_ranked_candidates(context, top=5, scorer=RANKING_SOURCE_AUTO)

    assert source.name == "heuristic_v1"
    assert selected == [cid for cid, _ in HEURISTIC_TOP]


def test_scorer_preference_puts_multimodal_ahead_of_earlier_stages():
    assert SCORER_PREFERENCE.index("multimodal_v1_1") < SCORER_PREFERENCE.index("highlight_v2_1")
    assert SCORER_PREFERENCE.index("highlight_v2_1") < SCORER_PREFERENCE.index("heuristic_v1")


def test_unknown_scorer_is_reported_with_the_available_options(tmp_path):
    context = load_run_context(build_run(tmp_path))
    with pytest.raises(KeyError, match="multimodal_v1_1"):
        resolve_ranking_source(context, "does_not_exist")


def test_scorer_name_accepts_a_json_suffix(tmp_path):
    context = load_run_context(build_run(tmp_path))
    assert resolve_ranking_source(context, "multimodal_v1_1.json").name == "multimodal_v1_1"


def test_available_rankings_are_listed_in_preference_order(tmp_path):
    context = load_run_context(build_run(tmp_path))
    available = context.available_rankings()

    assert available[0] == "multimodal_v1_1"
    assert available[-1] == RANKING_SOURCE_HIGHLIGHTS
    assert "heuristic_v1" in available


def test_run_without_any_ranking_is_reported_clearly(tmp_path):
    run_dir = build_run(tmp_path)
    (run_dir / "highlights.json").unlink()
    for f in (run_dir / "scores").glob("*.json"):
        f.unlink()

    context = load_run_context(run_dir)
    with pytest.raises(FileNotFoundError, match="No ranking found"):
        resolve_ranking_source(context, RANKING_SOURCE_AUTO)


def test_timeframes_come_from_the_frozen_candidate_set(tmp_path):
    """Candidates ranked only by the multimodal pass are absent from highlights.json."""
    context = load_run_context(build_run(tmp_path))
    assert not any(h.candidate_id == "cand_055" for h in context.highlights)

    timeframe = resolve_timeframe(context, "cand_055")
    assert timeframe.candidate_id == "cand_055"
    assert timeframe.duration_sec == pytest.approx(60.0)


def test_advisory_regions_are_collected_from_the_multimodal_scores(tmp_path):
    context = load_run_context(build_run(tmp_path))
    assert context.advisory_regions["cand_080"]["end_offset"] == 30.0
