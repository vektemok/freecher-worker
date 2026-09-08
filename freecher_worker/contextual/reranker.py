"""Orchestrator for Contextual Highlight Intelligence (contextual_reranker_v1_1).

Sits between the existing retrieval/multimodal stages and the final Top-K selection.
Never modifies an upstream scorer, and never sees human labels.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from freecher_worker.evaluation.models import (
    ScorerPredictionDocument,
    ScorerPredictionItem,
)
from freecher_worker.highlights.models import CandidateDocument, CandidateWindow
from freecher_worker.transcription.models import Transcript
from freecher_worker.utils.json_io import load_json, save_json

from .cache import ContextualCache
from .candidate_context import (
    DEFAULT_AFTER_SECONDS,
    DEFAULT_BEFORE_SECONDS,
    build_candidate_context_package,
)
from .chapters import (
    DEFAULT_CHAPTER_MAX_SECONDS,
    DEFAULT_CHAPTER_MIN_SECONDS,
    DEFAULT_CHAPTER_TARGET_SECONDS,
)
from .comparative import (
    DEFAULT_FINAL_PAIRWISE_TOP,
    DEFAULT_LISTWISE_BATCH_SIZE,
    ComparativeRanking,
    run_comparative_ranking,
)
from .context import build_source_context
from .critic import run_critic_pass
from .editorial import analyze_candidate, recover_pathological_rejection_distribution
from .models import (
    CandidateContextPackage,
    ContextualRerankDocument,
    ContextualRerankItem,
    ContextualUsage,
    CriticResult,
    EditorialAnalysis,
    GlobalContext,
    RetrievalProvenance,
)
from .prompts import PROMPT_BUNDLE_VERSION
from .provider import ContextualProvider
from .signals import load_source_activity_profile
from .versions import (
    CONTEXT_VERSION,
    DEFAULT_INPUT_SCORER,
    EDITORIAL_SCHEMA_VERSION,
    RANKING_ALGORITHM_VERSION,
    RERANKER_VERSION,
    SCORER_VERSION_CONTEXTUAL_V1,
)

logger = logging.getLogger("freecher_worker")

#: Display score bands. Survivors always outrank rejected candidates.
SURVIVOR_SCORE_HIGH = 100.0
SURVIVOR_SCORE_LOW = 45.0
REJECTED_SCORE_HIGH = 35.0
REJECTED_SCORE_LOW = 5.0

KNOWN_SCORER_FILES = {
    "heuristic_v1": "heuristic_v1.json",
    "highlight_v2": "highlight_v2.json",
    "highlight_v2_1": "highlight_v2_1.json",
    "multimodal_v1": "multimodal_v1.json",
    "multimodal_v1_1": "multimodal_v1_1.json",
}


def load_scorer_document(
    run_dir: Path | str,
    scorer: str,
) -> Optional[ScorerPredictionDocument]:
    """Load an existing scorer artifact, or None when it does not exist."""
    filename = KNOWN_SCORER_FILES.get(scorer, f"{scorer}.json")
    path = Path(run_dir).resolve() / "scores" / filename
    if not path.is_file():
        return None
    try:
        return ScorerPredictionDocument.model_validate(load_json(path))
    except Exception as exc:  # noqa: BLE001 - a malformed upstream artifact is reported, not fatal
        logger.warning(f"[contextual-rerank] Could not read {path}: {exc}")
        return None


LEGACY_CANDIDATE_SET_ID = "legacy_cset"


def load_candidate_document(run_dir: Path | str) -> CandidateDocument:
    """Load candidates.json, accepting the legacy bare-list format score-run also accepts."""
    cand_file = Path(run_dir).resolve() / "candidates.json"
    if not cand_file.is_file():
        raise FileNotFoundError(f"Missing candidates file: {cand_file}")

    data = load_json(cand_file)
    if isinstance(data, dict):
        return CandidateDocument.model_validate(data)
    if isinstance(data, list):
        candidates = [CandidateWindow.model_validate(entry) for entry in data]
        return CandidateDocument(
            candidate_set_id=LEGACY_CANDIDATE_SET_ID,
            transcript_hash="",
            min_seconds=0.0,
            target_seconds=0.0,
            max_seconds=0.0,
            overlap_seconds=0.0,
            candidates=candidates,
        )
    raise ValueError(f"Invalid candidates format in {cand_file}: expected an object or a list.")


def load_shortlist_ids(run_dir: Path | str) -> Optional[List[str]]:
    """Read the multimodal retrieval shortlist ids, if the run has one."""
    base = Path(run_dir).resolve() / "multimodal"
    for name in ("shortlist_v1_1.json", "shortlist_v1.json"):
        path = base / name
        if path.is_file():
            try:
                data = load_json(path)
                ids = data.get("candidate_ids") if isinstance(data, dict) else None
                if isinstance(ids, list):
                    return [str(cid) for cid in ids]
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[contextual-rerank] Could not read {path}: {exc}")
    return None


def read_manifest_source(run_dir: Path | str) -> Tuple[str, float]:
    """Return (source_fingerprint_id, duration_seconds) from the run manifest."""
    manifest_file = Path(run_dir).resolve() / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"Canonical run manifest not found at {manifest_file}. "
            f"{RERANKER_VERSION} requires manifest.json with source_fingerprint.fingerprint_id."
        )
    data = load_json(manifest_file)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest at {manifest_file}: expected a JSON object.")
    fingerprint = data.get("source_fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError(f"Invalid manifest at {manifest_file}: missing 'source_fingerprint'.")
    fp_id = fingerprint.get("fingerprint_id")
    if not fp_id or not str(fp_id).strip():
        raise ValueError(
            f"Invalid manifest at {manifest_file}: 'source_fingerprint.fingerprint_id' is empty."
        )
    duration = float(fingerprint.get("duration_seconds") or 0.0)
    return str(fp_id).strip(), duration


def _linear_band(index: int, count: int, high: float, low: float) -> float:
    """Map a 0-based position within ``count`` items onto [low, high], high first."""
    if count <= 1:
        return round(high, 2)
    span = high - low
    return round(high - span * (index / (count - 1)), 2)


class ContextualReranker:
    """Runs the contextual reranking pipeline over an existing run directory."""

    def __init__(
        self,
        provider: Optional[ContextualProvider] = None,
        input_scorer: str = DEFAULT_INPUT_SCORER,
        comparison_mode: str = "full",
        critic_enabled: bool = True,
        force: bool = False,
        before_seconds: float = DEFAULT_BEFORE_SECONDS,
        after_seconds: float = DEFAULT_AFTER_SECONDS,
        chapter_target_seconds: float = DEFAULT_CHAPTER_TARGET_SECONDS,
        chapter_min_seconds: float = DEFAULT_CHAPTER_MIN_SECONDS,
        chapter_max_seconds: float = DEFAULT_CHAPTER_MAX_SECONDS,
        listwise_batch_size: int = DEFAULT_LISTWISE_BATCH_SIZE,
        final_pairwise_top: int = DEFAULT_FINAL_PAIRWISE_TOP,
        top: Optional[int] = None,
    ) -> None:
        self.provider = provider
        self.input_scorer = input_scorer
        self.comparison_mode = comparison_mode
        self.critic_enabled = critic_enabled
        self.force = force
        self.before_seconds = before_seconds
        self.after_seconds = after_seconds
        self.chapter_target_seconds = chapter_target_seconds
        self.chapter_min_seconds = chapter_min_seconds
        self.chapter_max_seconds = chapter_max_seconds
        self.listwise_batch_size = listwise_batch_size
        self.final_pairwise_top = final_pairwise_top
        self.top = top

        self.scorer_version = SCORER_VERSION_CONTEXTUAL_V1
        self.reranker_version = RERANKER_VERSION
        self.context_version = CONTEXT_VERSION
        self.prompt_version = PROMPT_BUNDLE_VERSION
        self.warnings: List[str] = []

    # ---------------------------------------------------------------- inputs

    def _resolve_retrieval_candidates(
        self,
        run_dir: Path,
        cand_map: Dict[str, CandidateWindow],
    ) -> Tuple[List[str], Dict[str, RetrievalProvenance], Dict[str, ScorerPredictionItem]]:
        """Determine which candidates enter the reranker and where they came from."""
        input_doc = load_scorer_document(run_dir, self.input_scorer)
        if input_doc is None:
            raise FileNotFoundError(
                f"Missing input scorer predictions for '{self.input_scorer}' in {run_dir / 'scores'}. "
                f"{RERANKER_VERSION} reranks an existing candidate set and does not perform retrieval."
            )

        heuristic_doc = load_scorer_document(run_dir, "heuristic_v1")
        v2_1_doc = load_scorer_document(run_dir, "highlight_v2_1")
        multimodal_doc = (
            input_doc
            if self.input_scorer.startswith("multimodal")
            else load_scorer_document(run_dir, "multimodal_v1_1")
        )
        shortlist_ids = load_shortlist_ids(run_dir)

        def rank_map(doc: Optional[ScorerPredictionDocument]) -> Dict[str, ScorerPredictionItem]:
            if doc is None:
                return {}
            return {item.candidate_id: item for item in doc.predictions}

        heur = rank_map(heuristic_doc)
        v2_1 = rank_map(v2_1_doc)
        multimodal = rank_map(multimodal_doc)
        input_items = rank_map(input_doc)

        ordered_ids = [
            item.candidate_id
            for item in sorted(input_doc.predictions, key=lambda p: p.rank)
            if item.candidate_id in cand_map
        ]

        provenance: Dict[str, RetrievalProvenance] = {}
        for cid in ordered_ids:
            provenance[cid] = RetrievalProvenance(
                heuristic_rank=heur[cid].rank if cid in heur else None,
                heuristic_score=heur[cid].score if cid in heur else None,
                highlight_v2_1_rank=v2_1[cid].rank if cid in v2_1 else None,
                highlight_v2_1_score=v2_1[cid].score if cid in v2_1 else None,
                multimodal_rank=multimodal[cid].rank if cid in multimodal else None,
                multimodal_score=multimodal[cid].score if cid in multimodal else None,
                input_scorer=self.input_scorer,
                input_rank=input_items[cid].rank if cid in input_items else None,
                input_score=input_items[cid].score if cid in input_items else None,
                in_retrieval_shortlist=(cid in shortlist_ids) if shortlist_ids is not None else None,
            )

        return ordered_ids, provenance, multimodal

    # ---------------------------------------------------------------- pipeline

    def rerank_run(
        self,
        run_dir: Path | str,
        output_file: Optional[Path | str] = None,
    ) -> Tuple[ScorerPredictionDocument, ContextualRerankDocument]:
        """Execute the full contextual reranking pipeline on a run directory."""
        r_dir = Path(run_dir).resolve()
        self.warnings = []

        cand_doc = load_candidate_document(r_dir)
        cand_map: Dict[str, CandidateWindow] = {c.id: c for c in cand_doc.candidates}

        source_fingerprint, manifest_duration = read_manifest_source(r_dir)

        transcript: Optional[Transcript] = None
        transcript_file = r_dir / "transcript.json"
        if transcript_file.is_file():
            try:
                transcript = Transcript.model_validate(load_json(transcript_file))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[contextual-rerank] Could not read transcript: {exc}")
        source_duration = manifest_duration or (transcript.duration if transcript else 0.0)

        retrieval_ids, provenance, multimodal_items = self._resolve_retrieval_candidates(
            r_dir, cand_map
        )
        logger.info(
            f"[contextual-rerank] {len(retrieval_ids)} retrieval candidates from "
            f"'{self.input_scorer}'."
        )

        cache = ContextualCache(
            run_dir=r_dir,
            model=getattr(self.provider, "model", None),
            reasoning_effort=getattr(self.provider, "reasoning_effort", None),
            temperature=getattr(self.provider, "effective_temperature", None),
            context_version=self.context_version,
            reranker_version=self.reranker_version,
            force=self.force,
        )

        # 1. Global + chapter context (cached once per source).
        plans, chapter_doc, global_context = build_source_context(
            transcript=transcript,
            provider=self.provider,
            cache=cache,
            source_fingerprint=source_fingerprint,
            run_dir=r_dir,
            duration_seconds=source_duration,
            target_seconds=self.chapter_target_seconds,
            min_seconds=self.chapter_min_seconds,
            max_seconds=self.chapter_max_seconds,
        )
        if global_context.degraded:
            self.warnings.append(
                f"Global context degraded: {global_context.degraded_reason or 'unknown reason'}"
            )

        context_dir = r_dir / "contextual"
        global_file = context_dir / "global_context_v1.json"
        chapter_file = context_dir / "chapter_context_v1.json"
        save_json(global_context, global_file)
        save_json(chapter_doc, chapter_file)

        # 2. Candidate context packages.
        activity_profile = load_source_activity_profile(r_dir)
        packages: Dict[str, CandidateContextPackage] = {}
        for cid in retrieval_ids:
            packages[cid] = build_candidate_context_package(
                candidate=cand_map[cid],
                transcript=transcript,
                global_context=global_context,
                chapter_contexts=chapter_doc.chapters,
                chapter_plans=plans,
                source_duration=source_duration,
                multimodal_item=multimodal_items.get(cid),
                activity_profile=activity_profile,
                retrieval=provenance.get(cid),
                before_seconds=self.before_seconds,
                after_seconds=self.after_seconds,
            )
        save_json(
            [packages[cid].model_dump(mode="json") for cid in retrieval_ids],
            context_dir / "candidate_context_v1.json",
        )

        # 3. Salvage-aware editorial assessment. Only fatal-quality windows disappear.
        analyses: Dict[str, EditorialAnalysis] = {}
        for cid in retrieval_ids:
            analysis = analyze_candidate(packages[cid], global_context, self.provider, cache)
            analyses[cid] = analysis

        survivors, rejection_distribution_warning = recover_pathological_rejection_distribution(
            retrieval_ids, analyses
        )
        if rejection_distribution_warning:
            self.warnings.append(rejection_distribution_warning)
            logger.warning(f"[contextual-rerank] {rejection_distribution_warning}")
        logger.info(
            f"[contextual-rerank] Reject filter kept {len(survivors)}/{len(retrieval_ids)} candidates."
        )

        # 4. Penalty-oriented critic. Deletion requires explicit, strongly evidenced fatality.
        kept, critic_verdicts = run_critic_pass(
            survivors, packages, global_context, self.provider, cache, enabled=self.critic_enabled
        )
        logger.info(f"[contextual-rerank] Critic kept {len(kept)}/{len(survivors)} candidates.")

        # 5. Comparative ranking over survivors only.
        ranking = run_comparative_ranking(
            survivors=kept,
            packages=packages,
            analyses=analyses,
            provider=self.provider,
            cache=cache,
            critic_results=critic_verdicts,
            mode=self.comparison_mode,
            listwise_batch_size=self.listwise_batch_size,
            final_pairwise_top=self.final_pairwise_top,
        )

        rejected_ids = [cid for cid in retrieval_ids if cid not in set(ranking.ordered_ids)]
        rerank_doc = self._build_documents(
            run_dir=r_dir,
            cand_map=cand_map,
            candidate_set_id=cand_doc.candidate_set_id,
            source_fingerprint=source_fingerprint,
            global_context=global_context,
            chapter_doc_count=len(chapter_doc.chapters),
            global_file=global_file,
            chapter_file=chapter_file,
            retrieval_ids=retrieval_ids,
            ordered_ids=ranking.ordered_ids,
            rejected_ids=rejected_ids,
            analyses=analyses,
            critic_verdicts=critic_verdicts,
            ranking=ranking,
            provenance=provenance,
            rejection_distribution_warning=rejection_distribution_warning,
        )

        pred_doc = self._build_prediction_document(rerank_doc, cand_map)

        target = Path(output_file) if output_file else (r_dir / "scores" / f"{self.scorer_version}.json")
        save_json(pred_doc, target)
        save_json(rerank_doc, context_dir / f"{self.scorer_version}_run.json")
        logger.info(f"[contextual-rerank] Saved predictions to {target}")

        return pred_doc, rerank_doc

    # ---------------------------------------------------------------- artifacts

    def _build_documents(
        self,
        run_dir: Path,
        cand_map: Dict[str, CandidateWindow],
        candidate_set_id: str,
        source_fingerprint: str,
        global_context: GlobalContext,
        chapter_doc_count: int,
        global_file: Path,
        chapter_file: Path,
        retrieval_ids: Sequence[str],
        ordered_ids: Sequence[str],
        rejected_ids: Sequence[str],
        analyses: Dict[str, EditorialAnalysis],
        critic_verdicts: Dict[str, CriticResult],
        ranking: ComparativeRanking,
        provenance: Dict[str, RetrievalProvenance],
        rejection_distribution_warning: Optional[str],
    ) -> ContextualRerankDocument:
        """Assemble the full contextual record, survivors first then rejected."""
        survivor_count = len(ordered_ids)

        def make_item(cid: str, rank: int, status: str, display_score: float) -> ContextualRerankItem:
            analysis = analyses.get(cid)
            verdict = critic_verdicts.get(cid)
            prov = provenance.get(cid) or RetrievalProvenance()
            previous_rank = prov.input_rank
            candidate = cand_map.get(cid)
            return ContextualRerankItem(
                rank=rank,
                final_rank=rank,
                new_rank=rank,
                candidate_id=cid,
                start=candidate.start if candidate else 0.0,
                end=candidate.end if candidate else 0.0,
                duration=candidate.duration if candidate else 0.0,
                status=status,
                previous_rank=previous_rank,
                previous_score=prov.input_score,
                previous_multimodal_rank=prov.multimodal_rank,
                previous_multimodal_score=prov.multimodal_score,
                previous_heuristic_rank=prov.heuristic_rank,
                previous_highlight_v2_1_rank=prov.highlight_v2_1_rank,
                rank_delta=(previous_rank - rank) if previous_rank is not None else None,
                editorial_class=analysis.editorial_class if analysis else "WEAK",
                scroll_stop=analysis.scroll_stop if analysis else 0.0,
                editorial_dimensions=analysis.dimensions() if analysis else {},
                reason_to_watch=analysis.reason_to_watch if analysis else None,
                reason_to_skip=analysis.reason_to_skip if analysis else None,
                reject_reasons=list(analysis.reject_reasons) if analysis else [],
                editorial_confidence=analysis.confidence if analysis else 0.0,
                editorial_parse_failed=analysis.parse_failed if analysis else False,
                salvageable=analysis.salvageable if analysis else True,
                best_internal_moment_present=(
                    analysis.best_internal_moment_present if analysis else False
                ),
                needs_more_setup=analysis.needs_more_setup if analysis else False,
                needs_boundary_refinement=(
                    analysis.needs_boundary_refinement if analysis else False
                ),
                required_setup_seconds_estimate=(
                    analysis.required_setup_seconds_estimate if analysis else 0.0
                ),
                payoff_inside_candidate=analysis.payoff_inside_candidate if analysis else False,
                standalone_after_refinement_probability=(
                    analysis.standalone_after_refinement_probability if analysis else 0.0
                ),
                editorial_penalty=analysis.quality_penalty if analysis else 0.0,
                recovered_for_comparison=(
                    analysis.recovered_for_comparison if analysis else False
                ),
                critic_result=verdict.decision if verdict else "NOT_RUN",
                critic_reason=verdict.reason if verdict else None,
                critic_confidence=verdict.confidence if verdict else None,
                critic_penalty=verdict.penalty if verdict else 0.0,
                critic_failure_modes=list(verdict.failure_modes) if verdict else [],
                keep_for_comparison=verdict.keep_for_comparison if verdict else True,
                comparison_score=round(ranking.points.get(cid, 0.0), 3),
                adjusted_comparison_score=round(ranking.adjusted_points.get(cid, 0.0), 3),
                comparison_wins=ranking.wins.get(cid, 0),
                comparison_losses=ranking.losses.get(cid, 0),
                comparison_ties=ranking.ties.get(cid, 0),
                listwise_points=round(ranking.listwise_points.get(cid, 0.0), 4),
                head_to_head=dict(ranking.head_to_head.get(cid, {})),
                display_score=display_score,
                confidence=round(
                    min(
                        1.0,
                        max(
                            analysis.confidence if analysis else 0.0,
                            verdict.confidence if verdict else 0.0,
                        ),
                    ),
                    3,
                ),
            )

        results = [
            make_item(
                cid,
                rank=index + 1,
                status="ranked",
                display_score=_linear_band(
                    index, survivor_count, SURVIVOR_SCORE_HIGH, SURVIVOR_SCORE_LOW
                ),
            )
            for index, cid in enumerate(ordered_ids)
        ]

        # Rejected candidates keep a deterministic order but always rank below survivors.
        def reject_key(cid: str) -> Tuple[float, float, str]:
            analysis = analyses.get(cid)
            return (
                -(analysis.scroll_stop if analysis else 0.0),
                -(analysis.confidence if analysis else 0.0),
                cid,
            )

        ordered_rejected = sorted(rejected_ids, key=reject_key)
        rejected_items = [
            make_item(
                cid,
                rank=survivor_count + index + 1,
                status=(
                    "rejected_critic"
                    if critic_verdicts.get(cid) is not None
                    and critic_verdicts[cid].decision == "REJECT"
                    else "rejected_editorial"
                ),
                display_score=_linear_band(
                    index, len(ordered_rejected), REJECTED_SCORE_HIGH, REJECTED_SCORE_LOW
                ),
            )
            for index, cid in enumerate(ordered_rejected)
        ]

        usage = (
            self.provider.get_usage() if self.provider is not None else ContextualUsage()
        )

        return ContextualRerankDocument(
            scorer_version=self.scorer_version,
            reranker_version=self.reranker_version,
            model=getattr(self.provider, "model", None),
            candidate_set_id=candidate_set_id,
            source_fingerprint=source_fingerprint,
            context_version=self.context_version,
            prompt_version=self.prompt_version,
            schema_version=EDITORIAL_SCHEMA_VERSION,
            ranking_algorithm_version=RANKING_ALGORITHM_VERSION,
            input_scorer=self.input_scorer,
            retrieval_candidate_count=len(retrieval_ids),
            survivor_count=survivor_count,
            rejected_count=len(rejected_items),
            comparative_pool_count=survivor_count,
            recovered_for_comparison_count=sum(
                1 for analysis in analyses.values() if analysis.recovered_for_comparison
            ),
            rejection_distribution_warning=rejection_distribution_warning,
            global_context_ref=global_context.context_hash,
            global_context_file=str(global_file.relative_to(run_dir)),
            chapter_context_file=str(chapter_file.relative_to(run_dir)),
            chapter_count=chapter_doc_count,
            reasoning_effort=getattr(self.provider, "reasoning_effort", None),
            temperature=getattr(self.provider, "effective_temperature", None),
            comparison_mode=self.comparison_mode,
            critic_enabled=self.critic_enabled,
            top=self.top,
            results=results,
            rejected=rejected_items,
            comparisons=ranking.comparisons,
            listwise_batches=ranking.listwise_batches,
            usage=usage,
        )

    def _build_prediction_document(
        self,
        rerank_doc: ContextualRerankDocument,
        cand_map: Dict[str, CandidateWindow],
    ) -> ScorerPredictionDocument:
        """Emit a ScorerPredictionDocument so the existing evaluation harness works unchanged."""
        from freecher_worker.scoring.llm import compute_score_distribution

        items: List[ScorerPredictionItem] = []
        for record in [*rerank_doc.results, *rerank_doc.rejected]:
            reason_bits: List[str] = [f"{self.scorer_version} ({rerank_doc.model}): {record.editorial_class}"]
            if record.reason_to_watch:
                reason_bits.append(record.reason_to_watch)
            elif record.reject_reasons:
                reason_bits.append("; ".join(record.reject_reasons[:3]))
            items.append(
                ScorerPredictionItem(
                    candidate_id=record.candidate_id,
                    rank=record.rank,
                    score=record.display_score,
                    final_score=record.display_score,
                    reason=" — ".join(reason_bits),
                    subscores={
                        key: round(value, 4)
                        for key, value in record.editorial_dimensions.items()
                    },
                    scorer=self.scorer_version,
                    scorer_version=self.scorer_version,
                    requested_model=rerank_doc.model,
                    actual_model=rerank_doc.model,
                    confidence=record.confidence,
                    status=record.status,
                    editorial_class=record.editorial_class,
                    scroll_stop=record.scroll_stop,
                    reason_to_watch=record.reason_to_watch,
                    reason_to_skip=record.reason_to_skip,
                    reject_reasons=record.reject_reasons or None,
                    critic_result=record.critic_result,
                    critic_reason=record.critic_reason,
                    salvageable=record.salvageable,
                    best_internal_moment_present=record.best_internal_moment_present,
                    needs_more_setup=record.needs_more_setup,
                    needs_boundary_refinement=record.needs_boundary_refinement,
                    required_setup_seconds_estimate=record.required_setup_seconds_estimate,
                    payoff_inside_candidate=record.payoff_inside_candidate,
                    standalone_after_refinement_probability=(
                        record.standalone_after_refinement_probability
                    ),
                    editorial_penalty=record.editorial_penalty,
                    critic_penalty=record.critic_penalty,
                    critic_failure_modes=record.critic_failure_modes or None,
                    keep_for_comparison=record.keep_for_comparison,
                    recovered_for_comparison=record.recovered_for_comparison,
                    comparison_score=record.comparison_score,
                    previous_rank=record.previous_rank,
                    previous_score=record.previous_score,
                    previous_multimodal_rank=record.previous_multimodal_rank,
                    previous_multimodal_score=record.previous_multimodal_score,
                    new_rank=record.new_rank,
                    final_rank=record.final_rank,
                    rank_delta=record.rank_delta,
                    contextual=record.model_dump(mode="json"),
                )
            )

        scores = [item.score for item in items]
        return ScorerPredictionDocument(
            candidate_set_id=rerank_doc.candidate_set_id,
            scorer=self.scorer_version,
            scorer_version=self.scorer_version,
            requested_scorer=self.scorer_version,
            actual_scorer=self.scorer_version,
            model=rerank_doc.model,
            prompt_version=rerank_doc.prompt_version,
            score_formula_version=RANKING_ALGORITHM_VERSION,
            created_at=datetime.now().isoformat(),
            predictions=items,
            distribution_diagnostics=compute_score_distribution(scores) if scores else None,
            source_fingerprint=rerank_doc.source_fingerprint,
            context_version=rerank_doc.context_version,
            reranker_version=rerank_doc.reranker_version,
            schema_version=rerank_doc.schema_version,
            ranking_algorithm_version=rerank_doc.ranking_algorithm_version,
            input_scorer=rerank_doc.input_scorer,
            retrieval_candidate_count=rerank_doc.retrieval_candidate_count,
            survivor_count=rerank_doc.survivor_count,
            rejected_count=rerank_doc.rejected_count,
            comparative_pool_count=rerank_doc.comparative_pool_count,
            recovered_for_comparison_count=rerank_doc.recovered_for_comparison_count,
            rejection_distribution_warning=rerank_doc.rejection_distribution_warning,
            global_context_ref=rerank_doc.global_context_ref,
            reasoning_effort=rerank_doc.reasoning_effort,
            temperature=rerank_doc.temperature,
            usage=rerank_doc.usage.model_dump(),
        )
