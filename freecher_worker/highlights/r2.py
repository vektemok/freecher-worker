"""Highlight discovery driven by an R2 transcript instead of a local video.

The third milestone: read processing/{source_id}/transcript.json, run the
existing candidate generator, scorer and ranker over it unchanged, and write
candidates.json, highlights.json and a compatible manifest.json back beside it.

No video is involved at any point. Candidate generation, scoring with
heuristic_v1, deduplication and top-K ranking are all pure functions of the
transcript, so this module is orchestration only — it does not reimplement or
fork any part of the highlight algorithms.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from freecher_worker.highlights.models import (
    CandidateDocument,
    CandidateWindow,
    Highlight,
    HighlightScore,
)
from freecher_worker.highlights.ranker import rank_and_deduplicate
from freecher_worker.highlights.segmenter import generate_candidate_windows
from freecher_worker.ingest.service import object_exists
from freecher_worker.media.fingerprint import (
    SourceFingerprint,
    compute_r2_transcript_fingerprint,
)
from freecher_worker.pipeline.processor import (
    PIPELINE_VERSION,
    R2_TRANSCRIPT_RUN,
    AsrManifestInfo,
    CandidateConfigInfo,
    EnvironmentInfo,
    HighlightManifestItem,
    Manifest,
    PipelineStatistics,
    PipelineTimings,
    R2ArtifactInfo,
    RankingManifestInfo,
    ScoringManifestInfo,
)
from freecher_worker.scoring.base import HighlightScorer
from freecher_worker.scoring.heuristic import HEURISTIC_SCORER_VERSION, HeuristicScorer
from freecher_worker.transcription.models import Transcript
from freecher_worker.transcription.r2 import transcript_key_for
from freecher_worker.utils.json_io import save_json

logger = logging.getLogger("freecher_worker")

PROCESSING_PREFIX = "processing"
CANDIDATES_FILENAME = "candidates.json"
HIGHLIGHTS_FILENAME = "highlights.json"
MANIFEST_FILENAME = "manifest.json"

JSON_CONTENT_TYPE = "application/json"


class HighlightDiscoveryError(Exception):
    """Raised when the R2-backed discovery workflow cannot complete."""


@dataclass
class DiscoveryResult:
    """Outcome of a completed (or deliberately skipped) discovery run."""

    bucket: str
    source_id: str
    transcript_key: str
    candidates_key: str
    highlights_key: str
    manifest_key: str
    candidates: list[CandidateWindow] = field(default_factory=list)
    highlights: list[Highlight] = field(default_factory=list)
    manifest: Optional[Manifest] = None
    candidate_set_id: str = ""
    scorer_version: str = HEURISTIC_SCORER_VERSION
    skipped: bool = False
    uploaded_bytes: int = 0
    load_seconds: float = 0.0
    candidate_seconds: float = 0.0
    scoring_seconds: float = 0.0
    ranking_seconds: float = 0.0
    total_seconds: float = 0.0
    mirrored_to: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def candidate_durations(self) -> tuple[float, float, float]:
        """(min, mean, max) candidate duration in seconds; zeros when empty."""
        if not self.candidates:
            return (0.0, 0.0, 0.0)
        durations = [candidate.duration for candidate in self.candidates]
        return (min(durations), sum(durations) / len(durations), max(durations))

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"


def _clean_source_id(source_id: str) -> str:
    cleaned = source_id.strip().strip("/")
    if not cleaned or "/" in cleaned:
        raise ValueError(f"source id must be a single path segment, got '{source_id}'")
    return cleaned


def candidates_key_for(source_id: str) -> str:
    return f"{PROCESSING_PREFIX}/{_clean_source_id(source_id)}/{CANDIDATES_FILENAME}"


def highlights_key_for(source_id: str) -> str:
    return f"{PROCESSING_PREFIX}/{_clean_source_id(source_id)}/{HIGHLIGHTS_FILENAME}"


def manifest_key_for(source_id: str) -> str:
    return f"{PROCESSING_PREFIX}/{_clean_source_id(source_id)}/{MANIFEST_FILENAME}"


def serialize_document(payload: Any) -> bytes:
    """Render an artifact deterministically, the way transcript.json is rendered."""
    if isinstance(payload, BaseModel):
        dumpable: Any = payload.model_dump(mode="json")
    elif isinstance(payload, list):
        dumpable = [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item
            for item in payload
        ]
    else:
        dumpable = payload
    return json.dumps(dumpable, ensure_ascii=False, indent=2, sort_keys=False).encode("utf-8")


def _get_json(client: Any, bucket: str, key: str) -> Optional[Any]:
    """Read and parse one JSON object, or None if it is absent or unreadable."""
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except Exception:
        return None
    try:
        return json.loads(response["Body"].read().decode("utf-8"))
    except Exception as exc:
        logger.warning("object %s is not readable JSON (%s); treating as absent", key, exc)
        return None


def load_transcript_from_r2(client: Any, bucket: str, key: str) -> tuple[Transcript, int]:
    """Load and validate the transcript this run reads, or fail loudly.

    Unlike the idempotency probes, a transcript that will not parse is a hard
    stop: it is the input, and there is nothing to discover without it.
    """
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
    except Exception as exc:
        raise HighlightDiscoveryError(
            f"could not read the transcript at s3://{bucket}/{key}: {exc}"
        ) from exc

    try:
        transcript = Transcript.model_validate(json.loads(body.decode("utf-8")))
    except Exception as exc:
        raise HighlightDiscoveryError(
            f"the transcript at s3://{bucket}/{key} is not a valid transcript: {exc}"
        ) from exc

    if not transcript.segments:
        raise HighlightDiscoveryError(
            f"the transcript at s3://{bucket}/{key} has no segments; nothing to discover"
        )

    return transcript, len(body)


def build_manifest(
    *,
    transcript: Transcript,
    fingerprint: SourceFingerprint,
    candidate_document: CandidateDocument,
    highlights: list[Highlight],
    scorer_name: str,
    scorer_version: str,
    top_k: int,
    dedup_threshold: float,
    timings: PipelineTimings,
    artifacts: R2ArtifactInfo,
) -> Manifest:
    """Synthesize a manifest an R2-backed run can honestly claim.

    Shaped so `inspect` and `export-eval` read it unchanged, but explicit that
    no video exists here: `run_kind` is r2_transcript and
    `source_video_available` is False, so a later stage that needs frames finds
    out from the manifest rather than from a missing file. The probe, audio and
    clipping timings stay zero because those stages genuinely did not run;
    transcription time is carried over from what the transcript itself records.
    """
    return Manifest(
        pipeline_version=PIPELINE_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=fingerprint.path,
        source_fingerprint=fingerprint,
        environment=EnvironmentInfo(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform=platform.platform(),
        ),
        asr=AsrManifestInfo(
            model=transcript.model,
            device=transcript.device,
            compute_type=transcript.compute_type,
            language=transcript.language,
            beam_size=transcript.beam_size,
            vad_filter=transcript.vad_filter,
        ),
        candidate_config=CandidateConfigInfo(
            candidate_set_id=candidate_document.candidate_set_id,
            min_seconds=candidate_document.min_seconds,
            target_seconds=candidate_document.target_seconds,
            max_seconds=candidate_document.max_seconds,
            overlap=candidate_document.overlap_seconds,
        ),
        # No LLM is involved in this milestone, so there is nothing to fall
        # back from and no model to record.
        scoring=ScoringManifestInfo(
            scorer=scorer_name,
            scorer_version=scorer_version,
            llm_model=None,
            fallback_used=False,
            fallback_reason=None,
        ),
        ranking=RankingManifestInfo(top_k=top_k, dedup_threshold=dedup_threshold),
        timings=timings,
        statistics=PipelineStatistics(
            transcript_segment_count=len(transcript.segments),
            candidate_count=len(candidate_document.candidates),
            selected_highlight_count=len(highlights),
        ),
        highlights=[
            HighlightManifestItem(
                rank=highlight.rank,
                start=highlight.start,
                end=highlight.end,
                duration=highlight.duration,
                score=highlight.score,
                reason=highlight.reason,
                file=None,  # nothing is clipped without a video
                candidate_id=highlight.candidate_id,
                score_breakdown=highlight.score_breakdown,
            )
            for highlight in highlights
        ],
        run_kind=R2_TRANSCRIPT_RUN,
        source_video_available=False,
        r2=artifacts,
    )


def discover_from_r2(
    source_id: str,
    *,
    client: Any,
    bucket: str,
    transcript_key: Optional[str] = None,
    scorer: Optional[HighlightScorer] = None,
    min_seconds: float = 30.0,
    target_seconds: float = 60.0,
    max_seconds: float = 90.0,
    overlap_seconds: float = 15.0,
    top_k: int = 5,
    dedup_threshold: float = 0.60,
    overwrite: bool = False,
    local_dir: Optional[Path | str] = None,
) -> DiscoveryResult:
    """Turn an R2 transcript into ranked highlights and publish the artifacts.

    Idempotent: a complete, readable artifact set is left alone unless
    `overwrite` is set. The manifest is written last, so its presence is what
    marks a run finished — a failure part-way leaves no manifest and the next
    run rebuilds the set rather than trusting half of it.
    """
    started_at = time.perf_counter()
    warnings: list[str] = []
    cleaned_id = _clean_source_id(source_id)

    resolved_transcript_key = transcript_key or transcript_key_for(cleaned_id)
    artifacts = R2ArtifactInfo(
        bucket=bucket,
        source_id=cleaned_id,
        transcript_key=resolved_transcript_key,
        candidates_key=candidates_key_for(cleaned_id),
        highlights_key=highlights_key_for(cleaned_id),
        manifest_key=manifest_key_for(cleaned_id),
    )

    if not overwrite:
        existing = _existing_run(client, bucket, artifacts)
        if existing is not None:
            logger.info("discovery artifacts already present, skipping: %s", artifacts.manifest_key)
            if local_dir is not None and existing.manifest is not None:
                # Skipping the work must not skip the mirror: the caller asked
                # for a local run directory, and whether R2 already held the
                # artifacts is beside the point. The transcript is fetched here
                # because the skip path never loaded it.
                transcript, _ = load_transcript_from_r2(client, bucket, resolved_transcript_key)
                _mirror_locally(
                    Path(local_dir),
                    CandidateDocument(
                        transcript_hash=transcript.compute_transcript_hash(),
                        min_seconds=existing.manifest.candidate_config.min_seconds,
                        target_seconds=existing.manifest.candidate_config.target_seconds,
                        max_seconds=existing.manifest.candidate_config.max_seconds,
                        overlap_seconds=existing.manifest.candidate_config.overlap,
                        candidates=existing.candidates,
                        candidate_set_id=existing.candidate_set_id,
                    ),
                    existing.highlights,
                    existing.manifest,
                    transcript,
                )
                existing.mirrored_to = str(local_dir)
            return existing

    if not object_exists(client, bucket, resolved_transcript_key):
        raise HighlightDiscoveryError(
            f"no transcript at s3://{bucket}/{resolved_transcript_key}; run transcribe first"
        )

    load_start = time.perf_counter()
    transcript, transcript_bytes = load_transcript_from_r2(client, bucket, resolved_transcript_key)
    load_seconds = time.perf_counter() - load_start
    logger.info(
        "loaded transcript: %d segments, %.1fs, lang=%s",
        len(transcript.segments),
        transcript.duration,
        transcript.language,
    )

    transcript_hash = transcript.compute_transcript_hash()
    fingerprint = compute_r2_transcript_fingerprint(
        source_id=cleaned_id,
        bucket=bucket,
        transcript_key=resolved_transcript_key,
        transcript_hash=transcript_hash,
        duration_seconds=transcript.duration,
        transcript_bytes=transcript_bytes,
    )

    # --- candidates: the existing generator, unchanged -------------------
    candidate_start = time.perf_counter()
    candidates = generate_candidate_windows(
        transcript,
        min_seconds=min_seconds,
        target_seconds=target_seconds,
        max_seconds=max_seconds,
        overlap_seconds=overlap_seconds,
    )
    candidate_seconds = time.perf_counter() - candidate_start

    if not candidates:
        raise HighlightDiscoveryError(
            "the transcript produced no candidate windows; nothing to rank"
        )

    candidate_document = CandidateDocument(
        transcript_hash=transcript_hash,
        min_seconds=min_seconds,
        target_seconds=target_seconds,
        max_seconds=max_seconds,
        overlap_seconds=overlap_seconds,
        candidates=candidates,
    )
    logger.info(
        "generated %d candidate windows (id=%s)",
        len(candidates),
        candidate_document.candidate_set_id,
    )

    # --- scoring: heuristic_v1, no network, no video ---------------------
    active_scorer = scorer or HeuristicScorer()
    scorer_name = getattr(active_scorer, "name", type(active_scorer).__name__)
    scorer_version = getattr(active_scorer, "version", HEURISTIC_SCORER_VERSION)

    scoring_start = time.perf_counter()
    scores: list[HighlightScore] = active_scorer.score_batch(candidates, transcript=transcript)
    scoring_seconds = time.perf_counter() - scoring_start

    # --- ranking: the existing NMS + top-K, unchanged --------------------
    ranking_start = time.perf_counter()
    highlights = rank_and_deduplicate(
        candidates, scores, top_k=top_k, overlap_threshold=dedup_threshold
    )
    ranking_seconds = time.perf_counter() - ranking_start

    if len(highlights) < top_k:
        warnings.append(
            f"only {len(highlights)} highlights survived deduplication out of "
            f"{len(candidates)} candidates (top_k={top_k})"
        )

    timings = PipelineTimings(
        transcription_seconds=transcript.processing_seconds or 0.0,
        candidate_seconds=round(candidate_seconds, 3),
        scoring_seconds=round(scoring_seconds, 3),
        ranking_seconds=round(ranking_seconds, 3),
        total_seconds=round(time.perf_counter() - started_at, 3),
    )

    manifest = build_manifest(
        transcript=transcript,
        fingerprint=fingerprint,
        candidate_document=candidate_document,
        highlights=highlights,
        scorer_name=scorer_name,
        scorer_version=scorer_version,
        top_k=top_k,
        dedup_threshold=dedup_threshold,
        timings=timings,
        artifacts=artifacts,
    )

    # --- publish: manifest last, so it marks the run complete -------------
    uploaded = 0
    for key, payload in (
        (artifacts.candidates_key, candidate_document),
        (artifacts.highlights_key, highlights),
        (artifacts.manifest_key, manifest),
    ):
        body = serialize_document(payload)
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType=JSON_CONTENT_TYPE,
            Metadata=_artifact_metadata(key, cleaned_id, candidate_document, scorer_version),
        )
        uploaded += len(body)

    logger.info(
        "discovery complete: %d candidates -> %d highlights (%s)",
        len(candidates),
        len(highlights),
        artifacts.manifest_key,
    )

    mirrored_to: Optional[str] = None
    if local_dir is not None:
        _mirror_locally(Path(local_dir), candidate_document, highlights, manifest, transcript)
        mirrored_to = str(local_dir)

    return DiscoveryResult(
        bucket=bucket,
        source_id=cleaned_id,
        transcript_key=resolved_transcript_key,
        candidates_key=artifacts.candidates_key,
        highlights_key=artifacts.highlights_key,
        manifest_key=artifacts.manifest_key,
        candidates=candidates,
        highlights=highlights,
        manifest=manifest,
        candidate_set_id=candidate_document.candidate_set_id,
        scorer_version=scorer_version,
        uploaded_bytes=uploaded,
        load_seconds=round(load_seconds, 3),
        candidate_seconds=round(candidate_seconds, 3),
        scoring_seconds=round(scoring_seconds, 3),
        ranking_seconds=round(ranking_seconds, 3),
        total_seconds=round(time.perf_counter() - started_at, 3),
        mirrored_to=mirrored_to,
        warnings=warnings,
    )


def _artifact_metadata(
    key: str,
    source_id: str,
    candidate_document: CandidateDocument,
    scorer_version: str,
) -> dict[str, str]:
    """ASCII-safe user metadata shared by the three discovery artifacts."""
    return {
        "artifact": key.rsplit("/", 1)[-1].removesuffix(".json"),
        "source-id": source_id,
        "run-kind": R2_TRANSCRIPT_RUN,
        "candidate-set-id": candidate_document.candidate_set_id,
        "transcript-hash": candidate_document.transcript_hash,
        "scorer-version": scorer_version,
    }


def _existing_run(client: Any, bucket: str, artifacts: R2ArtifactInfo) -> Optional[DiscoveryResult]:
    """Return the completed run already in R2, or None if it is absent.

    All three artifacts have to be present and parseable. A partial set is not
    a run worth protecting, so it is rebuilt rather than left half-written.
    """
    manifest_payload = _get_json(client, bucket, artifacts.manifest_key)
    candidates_payload = _get_json(client, bucket, artifacts.candidates_key)
    highlights_payload = _get_json(client, bucket, artifacts.highlights_key)
    if manifest_payload is None or candidates_payload is None or highlights_payload is None:
        return None

    try:
        manifest = Manifest.model_validate(manifest_payload)
        candidate_document = CandidateDocument.model_validate(candidates_payload)
        highlights = [Highlight.model_validate(item) for item in highlights_payload]
    except Exception as exc:
        logger.warning("existing discovery artifacts are unreadable (%s); rebuilding", exc)
        return None

    return DiscoveryResult(
        bucket=bucket,
        source_id=artifacts.source_id,
        transcript_key=artifacts.transcript_key,
        candidates_key=artifacts.candidates_key,
        highlights_key=artifacts.highlights_key,
        manifest_key=artifacts.manifest_key,
        candidates=candidate_document.candidates,
        highlights=highlights,
        manifest=manifest,
        candidate_set_id=candidate_document.candidate_set_id,
        scorer_version=manifest.scoring.scorer_version,
        skipped=True,
    )


def _mirror_locally(
    directory: Path,
    candidate_document: CandidateDocument,
    highlights: list[Highlight],
    manifest: Manifest,
    transcript: Transcript,
) -> None:
    """Write the same artifacts into a run directory.

    `inspect` and `export-eval` read a local run directory, so mirroring makes
    them work against an R2-backed run without either command changing. The
    layout is the one `process` produces, minus the video-derived files.
    """
    directory.mkdir(parents=True, exist_ok=True)
    save_json(transcript, directory / "transcript.json")
    save_json(candidate_document, directory / CANDIDATES_FILENAME)
    save_json(highlights, directory / HIGHLIGHTS_FILENAME)
    # Last again: _resolve_run_path treats manifest.json as the marker that a
    # run directory is real.
    save_json(manifest, directory / MANIFEST_FILENAME)
    logger.info("mirrored discovery artifacts to %s", directory)
