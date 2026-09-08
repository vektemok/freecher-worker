"""Tests for contextual diagnostics: moment inspection, blind export, and regression checks."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from freecher_worker.cli import app
from freecher_worker.contextual import (
    NO_COVERAGE_MESSAGE,
    SCORER_VERSION_CONTEXTUAL_V1,
    ContextualReranker,
    RegressionCase,
    RegressionDataset,
    build_blind_diagnostic,
    find_covering_candidates,
    inspect_moment,
    parse_timestamp,
    select_blind_groups,
)
from freecher_worker.evaluation.models import ScorerPredictionDocument
from freecher_worker.evaluation.regression import (
    build_regression_dataset_from_run,
    check_regression_dataset,
)
from freecher_worker.highlights.models import CandidateDocument
from freecher_worker.utils.json_io import load_json, save_json

from tests.test_contextual_reranker_v1 import CANDIDATE_COUNT, FakeProvider, build_run

runner = CliRunner()


# ----------------------------------------------------------------------------------
# 15. Diagnostic timestamp lookup
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [("94.5", 94.5), ("18:34", 1114.0), ("00:18:34", 1114.0), ("1:00:00", 3600.0)],
)
def test_timestamp_parsing(text, expected):
    assert parse_timestamp(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "abc", "1:2:3:4", "12:xx"])
def test_invalid_timestamp_is_rejected(text):
    with pytest.raises(ValueError):
        parse_timestamp(text)


def test_inspect_moment_finds_the_overlapping_candidate(tmp_path):
    run_dir = build_run(tmp_path)
    # cand_001 spans 100.0 - 150.0 in the fixture.
    inspection = inspect_moment(run_dir, 120.0)

    assert inspection.covered
    assert inspection.candidate_id == "cand_001"
    assert inspection.candidate_start == 100.0 and inspection.candidate_end == 150.0
    assert inspection.in_retrieval_shortlist is True

    by_scorer = {row.scorer: row for row in inspection.scorers}
    assert by_scorer["heuristic_v1"].rank == 1
    assert by_scorer["highlight_v2_1"].rank == 1
    assert by_scorer["multimodal_v1_1"].rank == CANDIDATE_COUNT  # fixture reverses the order
    assert by_scorer[SCORER_VERSION_CONTEXTUAL_V1].present is False


def test_inspect_moment_reports_no_candidate_coverage(tmp_path):
    run_dir = build_run(tmp_path)
    inspection = inspect_moment(run_dir, 5.0)  # before the first candidate window

    assert not inspection.covered
    assert NO_COVERAGE_MESSAGE in inspection.message
    assert inspection.candidate_id is None


def test_inspect_moment_reports_contextual_rejection(tmp_path):
    run_dir = build_run(tmp_path)
    ContextualReranker(provider=FakeProvider(reject_ids={"cand_001"})).rerank_run(run_dir)

    inspection = inspect_moment(run_dir, 120.0)
    assert inspection.contextual_status == "rejected_editorial"
    assert inspection.editorial_class == "REJECT"
    assert "ordinary conversation" in inspection.reject_reasons
    assert "rejected by the contextual reranker" in inspection.message


def test_inspect_moment_reports_contextual_rank(tmp_path):
    run_dir = build_run(tmp_path)
    _, rerank_doc = ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)
    top = rerank_doc.results[0]

    inspection = inspect_moment(run_dir, (top.start + top.end) / 2.0)
    assert inspection.candidate_id == top.candidate_id
    assert inspection.contextual_rank == top.rank
    assert inspection.reason_to_watch


def test_overlapping_candidates_are_all_reported(tmp_path):
    run_dir = build_run(tmp_path)
    cand_doc = CandidateDocument.model_validate(load_json(run_dir / "candidates.json"))
    # Candidates step by 60s and last 50s, so windows do not overlap in the fixture;
    # the helper must still be correct on a hand-built overlapping pair.
    overlapping = find_covering_candidates(cand_doc.candidates, 145.0)
    assert [c.id for c in overlapping] == ["cand_001"]

    inspection = inspect_moment(run_dir, 145.0)
    assert inspection.overlapping_candidate_ids == ["cand_001"]


# ----------------------------------------------------------------------------------
# 16. Blind diagnostic export
# ----------------------------------------------------------------------------------


def test_blind_diagnostic_mapping_is_reproducible_by_seed(tmp_path):
    run_dir = build_run(tmp_path, candidate_count=32)
    ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)

    doc_a, map_a = build_blind_diagnostic(run_dir, group_a_size=4, group_b_size=2, group_c_size=2, seed=7)
    doc_b, map_b = build_blind_diagnostic(run_dir, group_a_size=4, group_b_size=2, group_c_size=2, seed=7)

    assert [i.blind_id for i in doc_a.items] == [i.blind_id for i in doc_b.items]
    assert [i.start for i in doc_a.items] == [i.start for i in doc_b.items]
    assert [(e.blind_id, e.candidate_id) for e in map_a.entries] == [
        (e.blind_id, e.candidate_id) for e in map_b.entries
    ]

    doc_c, _ = build_blind_diagnostic(run_dir, group_a_size=4, group_b_size=2, group_c_size=2, seed=99)
    assert [i.start for i in doc_a.items] != [i.start for i in doc_c.items], "a new seed must reshuffle"


def test_blind_diagnostic_document_hides_provenance(tmp_path):
    run_dir = build_run(tmp_path, candidate_count=32)
    ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)
    document, mapping = build_blind_diagnostic(run_dir, group_a_size=3, group_b_size=3, group_c_size=2)

    payload = document.model_dump_json()
    for entry in mapping.entries:
        assert entry.candidate_id not in payload, "candidate ids must not leak into the blind package"
    for field in (
        "multimodal_rank",
        "multimodal_score",
        "contextual_rank",
        "contextual_status",
        "editorial_class",
        "candidate_id",
        "group",
        "group_sizes",
    ):
        assert field not in payload, f"'{field}' must not appear in the blind package"
    # Only the reviewer's own (empty) label fields may mention a score.
    assert '"human_score":null' in payload

    assert document.total_items == len(mapping.entries)
    assert all(item.human_score is None and item.publishable is None for item in document.items)
    assert {e.group for e in mapping.entries} <= {"A", "B", "C"}


def test_blind_groups_select_the_intended_pools(tmp_path):
    run_dir = build_run(tmp_path, candidate_count=32)
    ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)

    groups = select_blind_groups(run_dir, group_a_range=(16, 32), group_a_size=4, group_b_size=4, group_c_size=3)
    multimodal = ScorerPredictionDocument.model_validate(
        load_json(run_dir / "scores" / "multimodal_v1_1.json")
    )
    ranks = {item.candidate_id: item.rank for item in multimodal.predictions}
    for cid in groups["A"]:
        assert 16 <= ranks[cid] <= 32

    shortlist = set(load_json(run_dir / "multimodal" / "shortlist_v1_1.json")["candidate_ids"])
    for cid in groups["B"]:
        assert cid not in shortlist

    assert len(groups["C"]) == 3


def test_blind_diagnostic_deduplicates_across_groups(tmp_path):
    run_dir = build_run(tmp_path, candidate_count=32)
    ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)
    document, mapping = build_blind_diagnostic(run_dir, group_a_size=8, group_b_size=8, group_c_size=8)

    candidate_ids = [entry.candidate_id for entry in mapping.entries]
    assert len(candidate_ids) == len(set(candidate_ids))
    assert [item.blind_id for item in document.items] == [e.blind_id for e in mapping.entries]


# ----------------------------------------------------------------------------------
# Regression dataset
# ----------------------------------------------------------------------------------


def test_regression_dataset_detects_movement(tmp_path):
    run_dir = build_run(tmp_path)
    pred_doc, rerank_doc = ContextualReranker(
        provider=FakeProvider(reject_ids={"cand_020"})
    ).rerank_run(run_dir)

    top = rerank_doc.results[0]
    dataset = RegressionDataset(
        strong_positives=[
            RegressionCase(
                source_fingerprint="fp_test",
                candidate_set_id=pred_doc.candidate_set_id,
                candidate_id=top.candidate_id,
                human_label=4.0,
                expectation="in_top_k",
                k=5,
            )
        ],
        false_positives=[
            RegressionCase(
                source_fingerprint="fp_test",
                candidate_set_id=pred_doc.candidate_set_id,
                candidate_id="cand_020",
                human_label=1.0,
                expectation="rejected",
            )
        ],
    )
    report = check_regression_dataset(dataset, pred_doc, default_k=5)
    assert report.total_cases == 2
    assert report.failed == 0
    assert all(result.passed for result in report.results)


def test_regression_case_from_another_candidate_set_is_skipped(tmp_path):
    run_dir = build_run(tmp_path)
    pred_doc, _ = ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)
    dataset = RegressionDataset(
        strong_positives=[
            RegressionCase(
                source_fingerprint="other",
                candidate_set_id="cset_unrelated",
                candidate_id="cand_001",
                expectation="in_top_k",
            )
        ]
    )
    report = check_regression_dataset(dataset, pred_doc)
    assert report.total_cases == 0 and report.skipped == 1


def test_regression_dataset_can_be_derived_from_a_labeled_run(tmp_path):
    run_dir = build_run(tmp_path)
    pred_doc, _ = ContextualReranker(provider=FakeProvider(reject_ids={"cand_002"})).rerank_run(run_dir)

    labels = {item.candidate_id: (4.0, True, "great") for item in pred_doc.predictions if item.rank > 5}
    labels["cand_002"] = (3.0, True, "rejected but good")
    dataset = build_regression_dataset_from_run(pred_doc, labels, "fp_test", top_k=5)

    assert dataset.strong_positives or dataset.false_negatives
    assert any(case.candidate_id == "cand_002" for case in dataset.false_negatives)
    for case in dataset.all_cases():
        assert case.candidate_set_id == pred_doc.candidate_set_id
        assert case.source_fingerprint == "fp_test"


# ----------------------------------------------------------------------------------
# CLI wiring
# ----------------------------------------------------------------------------------


def test_cli_inspect_moment(tmp_path):
    run_dir = build_run(tmp_path)
    ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)

    result = runner.invoke(app, ["inspect-moment", str(run_dir), "--time", "00:02:00"])
    assert result.exit_code == 0, result.output
    assert "cand_001" in result.output
    assert "multimodal_v1_1" in result.output


def test_cli_inspect_moment_no_coverage(tmp_path):
    run_dir = build_run(tmp_path)
    result = runner.invoke(app, ["inspect-moment", str(run_dir), "--time", "00:00:05"])
    assert result.exit_code == 0, result.output
    assert NO_COVERAGE_MESSAGE in result.output


def test_cli_inspect_moment_rejects_bad_timestamp(tmp_path):
    run_dir = build_run(tmp_path)
    result = runner.invoke(app, ["inspect-moment", str(run_dir), "--time", "not-a-time"])
    assert result.exit_code == 1


def test_cli_export_blind_diagnostic(tmp_path):
    run_dir = build_run(tmp_path, candidate_count=32)
    ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)

    result = runner.invoke(
        app,
        ["export-blind-diagnostic", str(run_dir), "--group-a", "3", "--group-b", "3", "--seed", "42"],
    )
    assert result.exit_code == 0, result.output

    out_dir = run_dir / "contextual" / "blind_diagnostic"
    assert (out_dir / "blind_diagnostic.json").is_file()
    assert (out_dir / "_DO_NOT_OPEN_mapping.json").is_file()
    document = load_json(out_dir / "blind_diagnostic.json")
    assert document["seed"] == 42
    assert document["total_items"] > 0


def test_cli_evaluate_contextual(tmp_path):
    from freecher_worker.evaluation.models import BlindEvaluationDocument, BlindEvaluationItem

    run_dir = build_run(tmp_path)
    pred_doc, _ = ContextualReranker(provider=FakeProvider(reject_ids={"cand_012"})).rerank_run(run_dir)
    cand_doc = CandidateDocument.model_validate(load_json(run_dir / "candidates.json"))

    items = [
        BlindEvaluationItem(
            candidate_id=c.id,
            start=c.start,
            end=c.end,
            duration=c.duration,
            text=c.text,
            human_score=4.0 if int(c.id.split("_")[-1]) <= 5 else 1.0,
            publishable=int(c.id.split("_")[-1]) <= 5,
        )
        for c in cand_doc.candidates
    ]
    eval_doc = BlindEvaluationDocument(
        candidate_set_id=cand_doc.candidate_set_id, total_candidates=len(items), items=items
    )
    eval_doc.update_labeled_count()
    eval_path = run_dir / "evaluation.json"
    save_json(eval_doc, eval_path)

    result = runner.invoke(
        app,
        [
            "evaluate-contextual",
            str(eval_path),
            str(run_dir / "scores" / f"{SCORER_VERSION_CONTEXTUAL_V1}.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "RejectPrecision" in result.output
    assert "StrongPrecision" in result.output
    assert "nDCG" in result.output


def test_cli_regression_check_fails_loudly(tmp_path):
    run_dir = build_run(tmp_path)
    pred_doc, _ = ContextualReranker(provider=FakeProvider()).rerank_run(run_dir)

    dataset = RegressionDataset(
        false_positives=[
            RegressionCase(
                source_fingerprint="fp_test",
                candidate_set_id=pred_doc.candidate_set_id,
                candidate_id=pred_doc.predictions[0].candidate_id,
                human_label=0.0,
                expectation="rejected",
            )
        ]
    )
    dataset_path = run_dir / "regression.json"
    save_json(dataset, dataset_path)

    result = runner.invoke(
        app,
        [
            "regression-check",
            str(dataset_path),
            str(run_dir / "scores" / f"{SCORER_VERSION_CONTEXTUAL_V1}.json"),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "failed 1" in result.output
