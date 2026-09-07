"""Unit tests for evaluation metrics mathematics and edge cases."""

import pytest
import math
from arny_worker.evaluation.models import (
    BlindEvaluationDocument,
    BlindEvaluationItem,
    ScorerPredictionDocument,
    ScorerPredictionItem,
)
from arny_worker.evaluation.metrics import (
    calculate_dcg,
    compute_evaluation_metrics,
    RELEVANCE_THRESHOLD,
)


def _make_eval_item(cid: str, score: int | None, pub: bool = False) -> BlindEvaluationItem:
    return BlindEvaluationItem(
        candidate_id=cid,
        start=0.0,
        end=30.0,
        duration=30.0,
        text=f"Text for {cid}",
        human_score=score,
        publishable=pub,
    )


def _make_pred_item(cid: str, rank: int, score: float = 80.0) -> ScorerPredictionItem:
    return ScorerPredictionItem(
        candidate_id=cid,
        rank=rank,
        score=score,
    )


def test_dcg_calculation():
    """Verify DCG calculation using formula: gain = 2^rel - 1, discount = log2(rank + 1)."""
    # rel = [4, 2, 3]
    # rank 1: (2^4 - 1) / log2(2) = 15.0 / 1.0 = 15.0
    # rank 2: (2^2 - 1) / log2(3) = 3.0 / 1.5849625 = 1.892789
    # rank 3: (2^3 - 1) / log2(4) = 7.0 / 2.0 = 3.5
    # sum = 20.392789
    relevances = [4, 2, 3]
    dcg = calculate_dcg(relevances, k=3)
    expected = 15.0 + (3.0 / math.log2(3)) + (7.0 / 2.0)
    assert abs(dcg - expected) < 1e-5


def test_metrics_precision_ndcg_hitrate():
    """Verify precision@K, nDCG@K, HitRate@K, and PublishableRate@K."""
    cset_id = "cset_test_123"

    eval_items = [
        _make_eval_item("c1", score=4, pub=True),
        _make_eval_item("c2", score=2, pub=False),
        _make_eval_item("c3", score=3, pub=True),
        _make_eval_item("c4", score=1, pub=False),
        _make_eval_item("c5", score=0, pub=False),
    ]
    eval_doc = BlindEvaluationDocument(
        candidate_set_id=cset_id,
        total_candidates=5,
        items=eval_items,
    )

    # Scorer ranked order: c1 (rel=4), c2 (rel=2), c3 (rel=3), c4 (rel=1), c5 (rel=0)
    pred_items = [
        _make_pred_item("c1", rank=1),
        _make_pred_item("c2", rank=2),
        _make_pred_item("c3", rank=3),
        _make_pred_item("c4", rank=4),
        _make_pred_item("c5", rank=5),
    ]
    pred_doc = ScorerPredictionDocument(
        candidate_set_id=cset_id,
        scorer="test_scorer",
        scorer_version="v1",
        predictions=pred_items,
    )

    metrics = compute_evaluation_metrics(
        eval_doc,
        pred_doc,
        k_values=[3, 5],
        hit_rate_k_values=[1, 2, 3],
    )

    # Precision@3: top 3 are c1(4), c2(2), c3(3). Relevant (>=3) are c1, c3 -> 2/3 = 0.6667
    assert metrics.precision_at_k[3] == 0.6667
    # Precision@5: top 5 have 2 relevant -> 2/5 = 0.40
    assert metrics.precision_at_k[5] == 0.40

    # PublishableRate@3: c1(True), c2(False), c3(True) -> 2/3 = 0.6667
    assert metrics.publishable_rate_at_k[3] == 0.6667

    # MeanHumanScore@3: (4 + 2 + 3) / 3 = 3.0
    assert metrics.mean_human_score_at_k[3] == 3.0
    # MeanHumanScore@5: (4 + 2 + 3 + 1 + 0) / 5 = 2.0
    assert metrics.mean_human_score_at_k[5] == 2.0

    # HitRate:
    # @1: c1 has score 4 >= 3 -> 1.0
    # @2: has hit -> 1.0
    # @3: has hit -> 1.0
    assert metrics.hit_rate_at_k[1] == 1.0
    assert metrics.hit_rate_at_k[2] == 1.0
    assert metrics.hit_rate_at_k[3] == 1.0

    # nDCG@3:
    # dcg@3 = 15.0/1 + 3.0/log2(3) + 7.0/2 = 20.392789
    # ideal scores: [4, 3, 2, 1, 0]
    # idcg@3 = 15.0/1 + 7.0/log2(3) + 3.0/2 = 15.0 + 4.416508 + 1.5 = 20.916508
    # ndcg@3 = 20.392789 / 20.916508 = 0.97496... -> 0.9750
    expected_ndcg_3 = round(
        (15.0 + (3.0 / math.log2(3)) + 3.5) / (15.0 + (7.0 / math.log2(3)) + 1.5),
        4,
    )
    assert metrics.ndcg_at_k[3] == expected_ndcg_3


def test_recall_guard_when_partially_labeled():
    """Verify Recall@K is blocked and returns None when candidate pool is not 100% labeled."""
    cset_id = "cset_partial"

    eval_items = [
        _make_eval_item("c1", score=4),
        _make_eval_item("c2", score=None),  # Unlabeled
        _make_eval_item("c3", score=3),
    ]
    eval_doc = BlindEvaluationDocument(
        candidate_set_id=cset_id,
        total_candidates=3,
        items=eval_items,
    )

    pred_items = [
        _make_pred_item("c1", rank=1),
        _make_pred_item("c2", rank=2),
        _make_pred_item("c3", rank=3),
    ]
    pred_doc = ScorerPredictionDocument(
        candidate_set_id=cset_id,
        scorer="test_scorer",
        scorer_version="v1",
        predictions=pred_items,
    )

    metrics = compute_evaluation_metrics(eval_doc, pred_doc, k_values=[2])

    assert metrics.recall_at_k is None
    assert "Recall requires 100% labeled candidate pool (currently 2/3 labeled)" in metrics.recall_message


def test_recall_when_fully_labeled():
    """Verify Recall@K calculation when 100% of candidate pool is labeled."""
    cset_id = "cset_full"

    eval_items = [
        _make_eval_item("c1", score=4),
        _make_eval_item("c2", score=1),
        _make_eval_item("c3", score=3),
        _make_eval_item("c4", score=3),
    ]
    eval_doc = BlindEvaluationDocument(
        candidate_set_id=cset_id,
        total_candidates=4,
        items=eval_items,
    )

    # Scorer ranked: c1 (rel=4), c2 (rel=1), c3 (rel=3), c4 (rel=3)
    pred_items = [
        _make_pred_item("c1", rank=1),
        _make_pred_item("c2", rank=2),
        _make_pred_item("c3", rank=3),
        _make_pred_item("c4", rank=4),
    ]
    pred_doc = ScorerPredictionDocument(
        candidate_set_id=cset_id,
        scorer="test_scorer",
        scorer_version="v1",
        predictions=pred_items,
    )

    # Total relevant in pool: c1(4), c3(3), c4(3) -> 3 relevant
    # Top 2: c1(4), c2(1) -> 1 relevant -> Recall@2 = 1/3 = 0.3333
    # Top 3: c1(4), c2(1), c3(3) -> 2 relevant -> Recall@3 = 2/3 = 0.6667
    # Top 4: c1, c2, c3, c4 -> 3 relevant -> Recall@4 = 3/3 = 1.0000
    metrics = compute_evaluation_metrics(eval_doc, pred_doc, k_values=[2, 3, 4])

    assert metrics.recall_at_k is not None
    assert metrics.recall_at_k[2] == 0.3333
    assert metrics.recall_at_k[3] == 0.6667
    assert metrics.recall_at_k[4] == 1.0


def test_candidate_set_id_mismatch_raises_error():
    """Verify ValueError is raised if candidate_set_id does not match."""
    eval_doc = BlindEvaluationDocument(
        candidate_set_id="cset_A",
        total_candidates=1,
        items=[_make_eval_item("c1", score=3)],
    )
    pred_doc = ScorerPredictionDocument(
        candidate_set_id="cset_B",
        scorer="test",
        scorer_version="v1",
        predictions=[_make_pred_item("c1", rank=1)],
    )

    with pytest.raises(ValueError, match="Candidate set ID mismatch"):
        compute_evaluation_metrics(eval_doc, pred_doc)
