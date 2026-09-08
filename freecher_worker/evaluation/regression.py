"""Hard-case regression dataset for highlight ranking.

Stores metadata only (fingerprint, candidate set, candidate id, human label, notes) so
tracking a regression never requires copying video. The check answers two questions:
did a known strong candidate move up, and did known false positives move down?
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from freecher_worker.contextual.models import (
    RegressionCase,
    RegressionCheckReport,
    RegressionCheckResult,
    RegressionDataset,
)
from freecher_worker.utils.json_io import load_json, save_json

from .models import ScorerPredictionDocument

VALID_EXPECTATIONS = ("rank_up", "rank_down", "in_top_k", "out_of_top_k", "rejected")


def load_regression_dataset(path: Path | str) -> RegressionDataset:
    """Load a regression dataset from disk."""
    return RegressionDataset.model_validate(load_json(path))


def save_regression_dataset(dataset: RegressionDataset, path: Path | str) -> Path:
    """Persist a regression dataset."""
    save_json(dataset, path)
    return Path(path)


def _check_case(
    case: RegressionCase,
    group: str,
    item,
    default_k: int,
) -> RegressionCheckResult:
    """Evaluate one case against its prediction item."""
    if item is None:
        return RegressionCheckResult(
            candidate_id=case.candidate_id,
            group=group,
            expectation=case.expectation,
            passed=False,
            message="Candidate is absent from the prediction document (not scored).",
        )

    previous = item.previous_rank
    new_rank = item.rank
    status = item.status
    k = case.k or default_k
    expectation = case.expectation

    if expectation == "rank_up":
        if previous is None:
            return RegressionCheckResult(
                candidate_id=case.candidate_id,
                group=group,
                expectation=expectation,
                passed=False,
                new_rank=new_rank,
                status=status,
                message="No previous rank recorded; cannot verify upward movement.",
            )
        passed = new_rank < previous
        message = f"Moved from #{previous} to #{new_rank}."
    elif expectation == "rank_down":
        if previous is None:
            return RegressionCheckResult(
                candidate_id=case.candidate_id,
                group=group,
                expectation=expectation,
                passed=False,
                new_rank=new_rank,
                status=status,
                message="No previous rank recorded; cannot verify downward movement.",
            )
        passed = new_rank > previous
        message = f"Moved from #{previous} to #{new_rank}."
    elif expectation == "in_top_k":
        passed = new_rank <= k and status in (None, "ranked")
        message = f"Rank #{new_rank} against K={k} (status={status})."
    elif expectation == "out_of_top_k":
        passed = new_rank > k or (status is not None and status != "ranked")
        message = f"Rank #{new_rank} against K={k} (status={status})."
    elif expectation == "rejected":
        passed = status is not None and status.startswith("rejected")
        message = f"Status is {status}."
    else:
        return RegressionCheckResult(
            candidate_id=case.candidate_id,
            group=group,
            expectation=expectation,
            passed=False,
            new_rank=new_rank,
            status=status,
            message=f"Unknown expectation '{expectation}'. Valid: {list(VALID_EXPECTATIONS)}.",
        )

    return RegressionCheckResult(
        candidate_id=case.candidate_id,
        group=group,
        expectation=expectation,
        passed=passed,
        previous_rank=previous,
        new_rank=new_rank,
        status=status,
        message=message,
    )


def check_regression_dataset(
    dataset: RegressionDataset,
    prediction_doc: ScorerPredictionDocument,
    default_k: int = 5,
) -> RegressionCheckReport:
    """Check every case whose candidate_set_id matches the prediction document."""
    items = {item.candidate_id: item for item in prediction_doc.predictions}
    results: List[RegressionCheckResult] = []
    skipped = 0

    groups: Dict[str, List[RegressionCase]] = {
        "strong_positives": dataset.strong_positives,
        "false_positives": dataset.false_positives,
        "false_negatives": dataset.false_negatives,
    }

    for group, cases in groups.items():
        for case in cases:
            if case.candidate_set_id != prediction_doc.candidate_set_id:
                skipped += 1
                continue
            results.append(_check_case(case, group, items.get(case.candidate_id), default_k))

    passed = sum(1 for r in results if r.passed)
    return RegressionCheckReport(
        candidate_set_id=prediction_doc.candidate_set_id,
        scorer_version=prediction_doc.scorer_version,
        total_cases=len(results),
        passed=passed,
        failed=len(results) - passed,
        skipped=skipped,
        results=results,
    )


def build_regression_dataset_from_run(
    prediction_doc: ScorerPredictionDocument,
    eval_items: Dict[str, tuple[Optional[float], Optional[bool], Optional[str]]],
    source_fingerprint: str,
    top_k: int = 5,
) -> RegressionDataset:
    """Derive a starting regression dataset from a labeled run.

    strong_positives: human_score >= 4 that the scorer did not place in the top K.
    false_positives:  human_score <= 2 that the scorer did place in the top K.
    false_negatives:  human_score >= 3 that the scorer rejected outright.
    """
    dataset = RegressionDataset(
        description=(
            "Auto-derived hard cases. Human labels are recorded for evaluation only and are "
            "never provided to the scorer."
        )
    )
    for item in prediction_doc.predictions:
        label = eval_items.get(item.candidate_id)
        if label is None:
            continue
        human, publishable, notes = label
        if human is None:
            continue
        base = dict(
            source_fingerprint=source_fingerprint,
            candidate_set_id=prediction_doc.candidate_set_id,
            candidate_id=item.candidate_id,
            human_label=human,
            publishable=publishable,
            notes=notes or "",
        )
        rejected = (item.status or "ranked") != "ranked"
        if human >= 4.0 and (item.rank > top_k or rejected):
            dataset.strong_positives.append(
                RegressionCase(**base, expectation="in_top_k", k=top_k)
            )
        elif human <= 2.0 and item.rank <= top_k and not rejected:
            dataset.false_positives.append(
                RegressionCase(**base, expectation="out_of_top_k", k=top_k)
            )
        elif human >= 3.0 and rejected:
            dataset.false_negatives.append(
                RegressionCase(**base, expectation="in_top_k", k=top_k)
            )
    return dataset
