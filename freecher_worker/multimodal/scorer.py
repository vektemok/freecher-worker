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

from .activity import compute_source_temporal_activity_profile
from .audio_features import compute_source_audio_profile
from .models import MultimodalCandidatePackage, MultimodalModelResult
from .package import PACKAGE_VERSION_V1, PACKAGE_VERSION_V1_1, build_multimodal_package
from .provider import MultimodalProvider, PROMPT_VERSION_MULTIMODAL_V1_1
from .shortlist import generate_shortlist

logger = logging.getLogger("freecher_worker")

SCORER_VERSION_MULTIMODAL_V1 = "multimodal_v1"
SCORER_VERSION_MULTIMODAL_V1_1 = "multimodal_v1_1"
FORMULA_VERSION_MULTIMODAL_V1 = "multimodal_v1_formula_v1"
FORMULA_VERSION_MULTIMODAL_V1_1 = "multimodal_v1_1_formula_v1"


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
    """Versioned scoring formula for Multimodal Highlight Reranker v1 and v1.1.

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


multimodal_v1_1_formula_v1 = multimodal_v1_formula_v1


def resolve_source_video_path(
    run_dir: Path | str,
    source_video_override: Optional[Path | str] = None,
) -> Path:
    """Resolve the source video file for a run directory.

    Resolution order:
    1. Explicit override (CLI --source-video / API argument)
    2. media.json in run_dir ("path")
    3. manifest.json in run_dir ("source" or "source_fingerprint.path")
    4. Direct video files inside run_dir (*.mp4, *.mkv, *.webm, *.mov, *.avi)
    5. Basename matches in current working directory or run_dir parent
    """
    r_dir = Path(run_dir).resolve()

    # 1. Explicit override
    if source_video_override:
        ov = Path(source_video_override).resolve()
        if ov.is_file():
            return ov
        if (r_dir / Path(source_video_override).name).is_file():
            return (r_dir / Path(source_video_override).name).resolve()
        if (Path.cwd() / Path(source_video_override).name).is_file():
            return (Path.cwd() / Path(source_video_override).name).resolve()
        raise FileNotFoundError(
            f"Explicit source video override not found: {source_video_override}"
        )

    possible_paths: list[str] = []

    # 2. media.json in run_dir
    media_file = r_dir / "media.json"
    if media_file.is_file():
        try:
            m_data = load_json(media_file)
            if isinstance(m_data, dict) and m_data.get("path"):
                possible_paths.append(str(m_data["path"]))
        except Exception:
            pass

    # 3. manifest.json in run_dir
    manifest_file = r_dir / "manifest.json"
    if manifest_file.is_file():
        try:
            man_data = load_json(manifest_file)
            if isinstance(man_data, dict):
                if man_data.get("source"):
                    possible_paths.append(str(man_data["source"]))
                fp_path = man_data.get("source_fingerprint", {}).get("path")
                if fp_path:
                    possible_paths.append(str(fp_path))
        except Exception:
            pass

    # Check all extracted candidate paths
    for p_str in possible_paths:
        p = Path(p_str)
        if p.is_file():
            return p.resolve()
        if (r_dir / p.name).is_file():
            return (r_dir / p.name).resolve()
        if (Path.cwd() / p.name).is_file():
            return (Path.cwd() / p.name).resolve()
        if (r_dir.parent / p.name).is_file():
            return (r_dir.parent / p.name).resolve()

    # 4. Check direct video files in run_dir
    direct_vids = (
        list(r_dir.glob("*.mp4"))
        + list(r_dir.glob("*.mkv"))
        + list(r_dir.glob("*.webm"))
        + list(r_dir.glob("*.mov"))
        + list(r_dir.glob("*.avi"))
    )
    if direct_vids:
        for pref in ("source.mp4", "input.mp4", "video.mp4"):
            if (r_dir / pref).is_file():
                return (r_dir / pref).resolve()
        return direct_vids[0].resolve()

    checked_msg = ", ".join(f"'{p}'" for p in possible_paths) if possible_paths else "none found in manifest/media.json"
    raise FileNotFoundError(
        f"No source video found for run at {r_dir}. "
        f"Checked paths recorded in run: [{checked_msg}]. "
        f"Please provide the source video path explicitly via --source-video / -s."
    )


def extract_canonical_fingerprint(run_dir: Path | str) -> str:
    """Extract canonical source fingerprint ID from manifest.json in run_dir.

    Fails explicitly if manifest.json is missing or if
    manifest['source_fingerprint']['fingerprint_id'] is missing or empty.
    """
    r_dir = Path(run_dir).resolve()
    manifest_file = r_dir / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"Canonical run manifest not found at {manifest_file}. "
            f"multimodal_v1 requires manifest.json with source_fingerprint.fingerprint_id."
        )

    try:
        manifest_data = load_json(manifest_file)
    except Exception as exc:
        raise ValueError(f"Failed to read manifest at {manifest_file}: {exc}") from exc

    if not isinstance(manifest_data, dict):
        raise ValueError(f"Invalid manifest at {manifest_file}: expected JSON object.")

    source_fp = manifest_data.get("source_fingerprint")
    if not isinstance(source_fp, dict):
        raise ValueError(
            f"Invalid manifest at {manifest_file}: missing 'source_fingerprint' object."
        )

    fp_id = source_fp.get("fingerprint_id")
    if not fp_id or not isinstance(fp_id, str) or not fp_id.strip():
        raise ValueError(
            f"Invalid manifest at {manifest_file}: 'source_fingerprint.fingerprint_id' is missing or empty."
        )

    return fp_id.strip()


class MultimodalReranker:
    """Orchestrator for candidate shortlisting, multimodal packaging, evaluation, and reranking."""

    def __init__(
        self,
        provider: MultimodalProvider,
        scorer_version: str = SCORER_VERSION_MULTIMODAL_V1,
        heuristic_top_k: Optional[int] = None,
        llm_top_k: Optional[int] = None,
        max_candidates: Optional[int] = None,
        allow_missing_llm: bool = False,
        force_rescore: bool = False,
        force_repackage: bool = False,
        source_video: Optional[Path | str] = None,
    ) -> None:
        self.provider = provider
        self.scorer_version = scorer_version
        is_v1_1 = scorer_version == SCORER_VERSION_MULTIMODAL_V1_1

        self.heuristic_top_k = heuristic_top_k if heuristic_top_k is not None else (20 if is_v1_1 else 12)
        self.llm_top_k = llm_top_k if llm_top_k is not None else (20 if is_v1_1 else 12)
        self.max_candidates = max_candidates if max_candidates is not None else (32 if is_v1_1 else 20)
        self.allow_missing_llm = allow_missing_llm
        self.force_rescore = force_rescore
        self.force_repackage = force_repackage
        self.source_video_override = source_video

        self.prompt_version = provider.prompt_version
        self.score_formula_version = (
            FORMULA_VERSION_MULTIMODAL_V1_1 if is_v1_1 else FORMULA_VERSION_MULTIMODAL_V1
        )
        self.package_version = (
            PACKAGE_VERSION_V1_1 if is_v1_1 else PACKAGE_VERSION_V1
        )

    def rerank_run(
        self,
        run_dir: Path | str,
        output_file: Optional[Path | str] = None,
        source_video_override: Optional[Path | str] = None,
    ) -> ScorerPredictionDocument:
        """Execute multimodal reranking on the run directory."""
        is_v1_1 = self.scorer_version == SCORER_VERSION_MULTIMODAL_V1_1
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
        video_path = resolve_source_video_path(
            run_dir=r_dir,
            source_video_override=source_video_override or self.source_video_override,
        )

        # Canonical source fingerprint from manifest.json
        source_fingerprint = extract_canonical_fingerprint(r_dir)

        # Probe video metadata for diagnostics / codec validation
        media_info = probe_media(video_path)
        dur = getattr(media_info, "duration_seconds", getattr(media_info, "duration", 0.0))
        fps = getattr(media_info, "fps", 0.0)
        width = getattr(media_info, "width", 0)
        height = getattr(media_info, "height", 0)
        codec = getattr(media_info, "video_codec", "unknown")
        logger.info(
            f"[multimodal-rerank] Source video: {video_path.name} "
            f"({width}x{height} @ {fps:.1f}fps, {dur:.1f}s, codec={codec}), "
            f"canonical fingerprint={source_fingerprint}"
        )

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

        # 5. Source-wide temporal activity profile for v1.1 (cached once)
        source_activity_profile = None
        if is_v1_1:
            activity_cache_file = (
                r_dir / "multimodal" / "cache" / "source_temporal_activity_profile_v1_1.json"
            )
            source_activity_profile = compute_source_temporal_activity_profile(
                wav_path=wav_path,
                video_path=video_path,
                source_fingerprint=source_fingerprint,
                cache_file=activity_cache_file,
                decoder_name=codec,
            )

        # 6. Deterministic shortlist generation
        shortlist_ver = "v1_1" if is_v1_1 else "v1"
        shortlist = generate_shortlist(
            run_dir=r_dir,
            heuristic_top_k=self.heuristic_top_k,
            llm_top_k=self.llm_top_k,
            max_candidates=self.max_candidates,
            allow_missing_llm=self.allow_missing_llm,
            shortlist_version=shortlist_ver,
        )

        # 7. Evaluate each candidate in shortlist
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
                source_activity_profile=source_activity_profile,
                package_version=self.package_version,
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

            best_reg_dict = (
                model_result.best_observed_region.model_dump()
                if model_result.best_observed_region
                else None
            )
            evidence_dicts = [ev.model_dump() for ev in model_result.evidence] if model_result.evidence else []
            audio_feat_dict = package.audio_features.model_dump() if package.audio_features else None
            visual_feat_dict = package.visual_features.model_dump() if package.visual_features else None

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
                # Structured observability fields
                observable_event=model_result.observable_event,
                visual_payoff=model_result.visual_payoff,
                outside_payoff=model_result.outside_payoff,
                missing_setup=model_result.missing_setup,
                insufficient_visual_evidence=(
                    package.insufficient_visual_evidence or model_result.insufficient_visual_evidence
                ),
                confidence=model_result.confidence,
                best_observed_region=best_reg_dict,
                evidence=evidence_dicts,
                audio_features=audio_feat_dict,
                visual_features=visual_feat_dict,
                frame_count=len(package.frames),
                actual_decoder=package.visual_features.decoder_used,
                package_hash=package.package_hash,
                request_hash=req_hash,
            )
            prediction_items.append(pred_item)

        # 8. Sort predictions deterministically: score desc, candidate.start asc
        def _sort_key(item: ScorerPredictionItem) -> Tuple[float, float]:
            cand = cand_map.get(item.candidate_id)
            c_start = cand.start if cand else 0.0
            return (-item.score, c_start)

        prediction_items.sort(key=_sort_key)

        # 9. Assign 1-based ranks
        for idx, item in enumerate(prediction_items, start=1):
            item.rank = idx

        # 10. Compute score distribution diagnostics
        scores_list = [p.score for p in prediction_items]
        dist_diagnostics = compute_score_distribution(scores_list)

        # Diagnostic warning if unique scores / scored candidates < 0.60
        if len(prediction_items) > 0:
            uniq_count = dist_diagnostics.unique_score_count_rounded
            total_count = len(prediction_items)
            uniq_ratio = uniq_count / float(total_count)
            if uniq_ratio < 0.60:
                res_msg = (
                    f"Low score resolution warning: unique score ratio "
                    f"{uniq_count}/{total_count} ({uniq_ratio:.1%}) < 60%"
                )
                logger.warning(f"[multimodal-diagnostics] {res_msg}")
                if dist_diagnostics.warning:
                    dist_diagnostics.warning += f"; {res_msg}"
                else:
                    dist_diagnostics.warning = res_msg

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

        # 11. Persist predictions to scores/multimodal_v1_1.json (or scores/multimodal_v1.json)
        default_output_name = "multimodal_v1_1.json" if is_v1_1 else "multimodal_v1.json"
        target_output = Path(output_file) if output_file else (r_dir / "scores" / default_output_name)
        target_output.parent.mkdir(parents=True, exist_ok=True)
        save_json(pred_doc, target_output)
        logger.info(f"[multimodal-rerank] Saved predictions to {target_output}")

        # 12. Persist detailed reranking run record
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
        default_record_name = "rerank_v1_1.json" if is_v1_1 else "rerank_v1.json"
        rerank_record_file = r_dir / "multimodal" / default_record_name
        save_json(rerank_run_doc, rerank_record_file)

        return pred_doc
