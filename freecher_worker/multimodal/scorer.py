"""Multimodal Highlight Reranker (multimodal_v1) scoring formula and pipeline orchestrator."""

from __future__ import annotations

from datetime import datetime
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from freecher_worker.evaluation.models import (
    ScorerPredictionDocument,
    ScorerPredictionItem,
)
from freecher_worker.highlights.models import CandidateDocument, CandidateWindow
from freecher_worker.media.probe import probe_media
from freecher_worker.scoring.llm import compute_score_distribution
from freecher_worker.transcription.models import Transcript

TranscriptDocument = Transcript
from freecher_worker.utils.json_io import load_json, save_json

from .audio_features import compute_source_audio_profile
from .models import MultimodalCandidatePackage, MultimodalModelResult
from .package import build_multimodal_package
from .provider import MultimodalProvider
from .shortlist import generate_shortlist

logger = logging.getLogger("freecher_worker")

SCORER_VERSION_MULTIMODAL_V1 = "multimodal_v1"
FORMULA_VERSION_MULTIMODAL_V1 = "multimodal_v1_formula_v1"


def compute_api_request_hash(
    package_hash: str,
    provider_name: str,
    model_name: str,
    prompt_version: str,
) -> str:
    """Deterministic hash identifying an API evaluation request."""
    key = f"{package_hash}:{provider_name}:{model_name}:{prompt_version}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def multimodal_v1_formula_v1(
    result: MultimodalModelResult,
    package: MultimodalCandidatePackage,
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    """Versioned scoring formula for Multimodal Highlight Reranker v1.

    Args:
        result: Evaluated result from multimodal provider.
        package: Multimodal candidate package.

    Returns:
        (final_score, subscores, diagnostics)
    """
    raw_score = float(result.quality_score)
    score = raw_score
    applied_caps: List[str] = []

    # 1. Boringness & low retention cap
    if result.boringness >= 85.0 and result.retention <= 25.0:
        if score > 35.0:
            score = 35.0
            applied_caps.append("boring_low_retention_cap:35.0")

    # 2. Outside payoff with no observable event
    if result.outside_payoff and not result.observable_event:
        if score > 40.0:
            score = 40.0
            applied_caps.append("outside_payoff_unobservable_cap:40.0")

    # 3. Missing setup with extreme context dependency
    if result.missing_setup and result.context_dependency >= 80.0:
        if score > 45.0:
            score = 45.0
            applied_caps.append("missing_setup_context_cap:45.0")

    # 4. Insufficient visual evidence: Lower confidence rather than penalizing score
    if package.insufficient_visual_evidence or result.insufficient_visual_evidence:
        effective_conf = max(0.2, min(result.confidence, 0.4))
        applied_caps.append(f"insufficient_visual_evidence_confidence_lowered:{effective_conf:.2f}")

    final_score = round(float(np.clip(score, 0.0, 100.0)), 2)

    subscores: Dict[str, Any] = {
        "quality_score": result.quality_score,
        "visual_event": result.visual_event,
        "reaction": result.reaction,
        "emotion": result.emotion,
        "humor": result.humor,
        "surprise": result.surprise,
        "energy": result.energy,
        "standalone": result.standalone,
        "retention": result.retention,
        "shareability": result.shareability,
        "boringness": result.boringness,
        "context_dependency": result.context_dependency,
        "confidence": result.confidence,
    }

    diagnostics: Dict[str, Any] = {
        "raw_score": raw_score,
        "final_score": final_score,
        "applied_caps": applied_caps,
    }

    return final_score, subscores, diagnostics


class MultimodalReranker:
    """Orchestrator for candidate shortlisting, multimodal packaging, evaluation, and reranking."""

    def __init__(
        self,
        provider: MultimodalProvider,
        heuristic_top_k: int = 12,
        llm_top_k: int = 12,
        max_candidates: int = 20,
        allow_missing_llm: bool = False,
        force_rescore: bool = False,
        force_repackage: bool = False,
    ) -> None:
        self.provider = provider
        self.heuristic_top_k = heuristic_top_k
        self.llm_top_k = llm_top_k
        self.max_candidates = max_candidates
        self.allow_missing_llm = allow_missing_llm
        self.force_rescore = force_rescore
        self.force_repackage = force_repackage

        self.scorer_version = SCORER_VERSION_MULTIMODAL_V1
        self.prompt_version = provider.prompt_version
        self.score_formula_version = FORMULA_VERSION_MULTIMODAL_V1

    def rerank_run(
        self,
        run_dir: Path | str,
        output_file: Optional[Path | str] = None,
        source_video_override: Optional[Path | str] = None,
    ) -> ScorerPredictionDocument:
        """Execute multimodal reranking on the run directory.

        Produces scores/multimodal_v1.json.
        """
        r_dir = Path(run_dir).resolve()
        cand_file = r_dir / "candidates.json"
        if not cand_file.is_file():
            raise FileNotFoundError(f"Missing candidates file: {cand_file}")

        cand_doc = CandidateDocument.model_validate(load_json(cand_file))
        cand_map: Dict[str, CandidateWindow] = {c.id: c for c in cand_doc.candidates}

        # 1. Transcript document
        transcript_file = r_dir / "transcript.json"
        transcript_doc: Optional[TranscriptDocument] = None
        if transcript_file.is_file():
            try:
                transcript_doc = TranscriptDocument.model_validate(load_json(transcript_file))
            except Exception as exc:
                logger.warning(f"[multimodal-rerank] Failed loading transcript: {exc}")

        # 2. Locate source video
        if source_video_override and Path(source_video_override).is_file():
            video_path = Path(source_video_override).resolve()
        else:
            cand_video = getattr(cand_doc, "source_video", None)
            if cand_video and Path(cand_video).is_file():
                video_path = Path(cand_video).resolve()
            elif (r_dir / "source.mp4").is_file():
                video_path = r_dir / "source.mp4"
            else:
                # Check for any .mp4 or .mkv in run_dir
                vids = list(r_dir.glob("*.mp4")) + list(r_dir.glob("*.mkv"))
                if vids:
                    video_path = vids[0].resolve()
                else:
                    raise FileNotFoundError(f"No source video found for run at {r_dir}")

        # Probe video fingerprint
        media_info = probe_media(video_path)
        source_fingerprint = f"{video_path.name}_{media_info.duration:.1f}s"

        # 3. Locate source audio WAV
        wav_path = r_dir / "audio.wav"
        if not wav_path.is_file():
            from freecher_worker.media.audio import extract_audio
            logger.info(f"[multimodal-rerank] Extracting audio to {wav_path}...")
            extract_audio(video_path, wav_path)

        # 4. Source-wide audio profile (cached once)
        audio_cache_file = r_dir / "multimodal" / "cache" / "source_audio_profile.json"
        source_profile = compute_source_audio_profile(
            wav_path=wav_path,
            source_fingerprint=source_fingerprint,
            cache_file=audio_cache_file,
        )

        # 5. Deterministic shortlist generation
        shortlist = generate_shortlist(
            run_dir=r_dir,
            heuristic_top_k=self.heuristic_top_k,
            llm_top_k=self.llm_top_k,
            max_candidates=self.max_candidates,
            allow_missing_llm=self.allow_missing_llm,
        )

        # 6. Evaluate each candidate in shortlist
        prediction_items: List[ScorerPredictionItem] = []
        api_cache_dir = r_dir / "multimodal" / "api_cache"
        api_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[multimodal-rerank] Evaluating {len(shortlist.candidate_ids)} shortlist candidates "
            f"using {self.provider.name} ({self.provider.model})..."
        )

        for cid in shortlist.candidate_ids:
            if cid not in cand_map:
                logger.warning(f"[multimodal-rerank] Shortlist candidate {cid} not in candidates.json")
                continue

            candidate = cand_map[cid]

            # Build package
            package = build_multimodal_package(
                candidate=candidate,
                transcript_doc=transcript_doc,
                source_video_path=video_path,
                source_wav_path=wav_path,
                source_fingerprint=source_fingerprint,
                candidate_set_id=cand_doc.candidate_set_id,
                run_dir=r_dir,
                source_audio_profile=source_profile,
                force_rebuild=self.force_repackage,
            )

            # API Request caching
            req_hash = compute_api_request_hash(
                package_hash=package.package_hash,
                provider_name=self.provider.name,
                model_name=self.provider.model,
                prompt_version=self.provider.prompt_version,
            )
            api_cache_file = api_cache_dir / f"{req_hash}.json"

            model_result: Optional[MultimodalModelResult] = None
            if api_cache_file.is_file() and not self.force_rescore:
                try:
                    cached_res = load_json(api_cache_file)
                    model_result = MultimodalModelResult.model_validate(cached_res)
                    logger.debug(f"[multimodal-rerank] Using cached API result for {cid} ({req_hash})")
                except Exception as exc:
                    logger.warning(f"[multimodal-rerank] Cached API result corrupt for {cid}: {exc}")

            if model_result is None:
                model_result = self.provider.score_candidate(package)
                save_json(model_result, api_cache_file)

            # Formula evaluation
            final_score, subscores, diagnostics = multimodal_v1_formula_v1(model_result, package)

            flags = {
                "observable_event": model_result.observable_event,
                "visual_payoff": model_result.visual_payoff,
                "outside_payoff": model_result.outside_payoff,
                "missing_setup": model_result.missing_setup,
                "insufficient_visual_evidence": (
                    package.insufficient_visual_evidence or model_result.insufficient_visual_evidence
                ),
            }

            pred_item = ScorerPredictionItem(
                candidate_id=cid,
                rank=1,  # Ranks will be assigned after sorting
                score=final_score,
                final_score=final_score,
                llm_quality_score=model_result.quality_score,
                subscores=subscores,
                flags=flags,
                applied_caps=diagnostics["applied_caps"],
                reason=f"{self.scorer_version} ({self.provider.model}): {model_result.reason}",
                scorer=self.scorer_version,
                scorer_version=self.scorer_version,
                requested_model=self.provider.model,
                actual_model=self.provider.model,
                fallback_used=False,
                fallback_reason=None,
            )
            prediction_items.append(pred_item)

        # 7. Sort predictions deterministically: score desc, candidate.start asc
        def _sort_key(item: ScorerPredictionItem) -> Tuple[float, float]:
            cand = cand_map.get(item.candidate_id)
            c_start = cand.start if cand else 0.0
            return (-item.score, c_start)

        prediction_items.sort(key=_sort_key)

        # 8. Assign 1-based ranks
        for idx, item in enumerate(prediction_items, start=1):
            item.rank = idx

        # 9. Compute score distribution diagnostics
        scores_list = [p.score for p in prediction_items]
        dist_diagnostics = compute_score_distribution(scores_list)

        pred_doc = ScorerPredictionDocument(
            candidate_set_id=cand_doc.candidate_set_id,
            scorer=self.scorer_version,
            scorer_version=self.scorer_version,
            requested_scorer=self.scorer_version,
            actual_scorer=self.scorer_version,
            model=self.provider.model,
            prompt_version=self.prompt_version,
            score_formula_version=self.score_formula_version,
            predictions=prediction_items,
            distribution_diagnostics=dist_diagnostics,
        )

        # 10. Persist predictions to scores/multimodal_v1.json
        target_output = Path(output_file) if output_file else (r_dir / "scores" / "multimodal_v1.json")
        target_output.parent.mkdir(parents=True, exist_ok=True)
        save_json(pred_doc, target_output)
        logger.info(f"[multimodal-rerank] Saved predictions to {target_output}")

        # 11. Persist detailed reranking run record
        usage = self.provider.get_usage()
        rerank_run_doc = {
            "candidate_set_id": cand_doc.candidate_set_id,
            "scorer_version": self.scorer_version,
            "prompt_version": self.prompt_version,
            "score_formula_version": self.score_formula_version,
            "created_at": datetime.now().isoformat(),
            "shortlist_strategy": shortlist.strategy,
            "shortlist_candidates_count": len(shortlist.candidate_ids),
            "scored_candidates_count": len(prediction_items),
            "usage": usage.model_dump(),
        }
        rerank_record_file = r_dir / "multimodal" / "rerank_v1.json"
        save_json(rerank_run_doc, rerank_record_file)

        return pred_doc
