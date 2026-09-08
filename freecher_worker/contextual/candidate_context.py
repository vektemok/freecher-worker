"""Per-candidate context package construction for contextual_reranker_v1.

BEFORE and AFTER windows exist purely so the model understands the moment. They are
never added to the clip: only ``package.candidate`` describes the segment being judged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from freecher_worker.evaluation.models import ScorerPredictionItem
from freecher_worker.highlights.models import CandidateWindow
from freecher_worker.multimodal.models import SourceTemporalActivityProfile
from freecher_worker.transcription.models import Transcript

from .cache import stable_hash
from .chapters import ChapterPlan, find_chapter_for_range
from .models import (
    CandidateContextPackage,
    ChapterContext,
    GlobalContext,
    ReactionSignals,
    RetrievalProvenance,
    TranscriptWindow,
)
from .signals import extract_reaction_signals
from .versions import CANDIDATE_CONTEXT_SCHEMA_VERSION, CONTEXT_VERSION

DEFAULT_BEFORE_SECONDS = 75.0
DEFAULT_AFTER_SECONDS = 25.0

#: Multimodal fields carried forward as evidence. Scores stay out of the reranker's
#: reasoning on purpose: only observations and flags are forwarded.
MULTIMODAL_EVIDENCE_FIELDS = (
    "observable_event",
    "visual_payoff",
    "outside_payoff",
    "missing_setup",
    "insufficient_visual_evidence",
    "confidence",
    "best_observed_region",
    "evidence",
)


def extract_window_transcript(
    transcript: Optional[Transcript],
    start: float,
    end: float,
) -> str:
    """Join transcript segments overlapping [start, end) into one string."""
    if transcript is None or not transcript.segments or end <= start:
        return ""
    parts: List[str] = []
    for seg in sorted(transcript.segments, key=lambda s: (s.start, s.id)):
        if seg.end <= start or seg.start >= end:
            continue
        text = seg.text.strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def build_context_windows(
    candidate: CandidateWindow,
    transcript: Optional[Transcript],
    source_duration: float,
    before_seconds: float = DEFAULT_BEFORE_SECONDS,
    after_seconds: float = DEFAULT_AFTER_SECONDS,
) -> tuple[TranscriptWindow, TranscriptWindow, TranscriptWindow]:
    """Build the BEFORE / CANDIDATE / AFTER windows, clamped to the source duration.

    Returns them in that order; they never overlap and never leave [0, source_duration].
    """
    upper = max(0.0, float(source_duration)) if source_duration else max(candidate.end, 0.0)
    upper = max(upper, candidate.end)

    cand_start = max(0.0, min(candidate.start, upper))
    cand_end = max(cand_start, min(candidate.end, upper))

    before_start = max(0.0, cand_start - max(0.0, before_seconds))
    before_end = cand_start
    after_start = cand_end
    after_end = min(upper, cand_end + max(0.0, after_seconds))

    before = TranscriptWindow(
        start=round(before_start, 3),
        end=round(before_end, 3),
        transcript=extract_window_transcript(transcript, before_start, before_end),
    )
    cand_window = TranscriptWindow(
        start=round(cand_start, 3),
        end=round(cand_end, 3),
        transcript=candidate.text.strip()
        or extract_window_transcript(transcript, cand_start, cand_end),
    )
    after = TranscriptWindow(
        start=round(after_start, 3),
        end=round(after_end, 3),
        transcript=extract_window_transcript(transcript, after_start, after_end),
    )
    return before, cand_window, after


def extract_multimodal_evidence(
    item: Optional[ScorerPredictionItem],
) -> Optional[Dict[str, Any]]:
    """Copy the reusable evidence fields out of a multimodal prediction item."""
    if item is None:
        return None
    evidence: Dict[str, Any] = {}
    for field in MULTIMODAL_EVIDENCE_FIELDS:
        value = getattr(item, field, None)
        if value is not None:
            evidence[field] = value

    audio = item.audio_features or {}
    if audio:
        evidence["audio_features"] = {
            key: audio[key]
            for key in (
                "rms_mean",
                "speech_coverage",
                "silence_ratio",
                "energy_percentile",
                "energy_change_rate",
            )
            if key in audio
        }
    visual = item.visual_features or {}
    if visual:
        evidence["visual_features"] = {
            key: visual[key]
            for key in (
                "motion_score",
                "scene_change_count",
                "face_presence_ratio",
                "person_presence_ratio",
            )
            if key in visual
        }
    if item.reason:
        evidence["upstream_reason"] = item.reason
    return evidence or None


def build_candidate_context_package(
    candidate: CandidateWindow,
    transcript: Optional[Transcript],
    global_context: GlobalContext,
    chapter_contexts: Sequence[ChapterContext],
    chapter_plans: Sequence[ChapterPlan],
    source_duration: float,
    multimodal_item: Optional[ScorerPredictionItem] = None,
    activity_profile: Optional[SourceTemporalActivityProfile] = None,
    retrieval: Optional[RetrievalProvenance] = None,
    before_seconds: float = DEFAULT_BEFORE_SECONDS,
    after_seconds: float = DEFAULT_AFTER_SECONDS,
) -> CandidateContextPackage:
    """Assemble everything the contextual scorer sees for one candidate."""
    before, cand_window, after = build_context_windows(
        candidate,
        transcript,
        source_duration,
        before_seconds=before_seconds,
        after_seconds=after_seconds,
    )

    plan = find_chapter_for_range(chapter_plans, candidate.start, candidate.end)
    chapter_ctx: Optional[ChapterContext] = None
    if plan is not None:
        by_id = {ch.chapter_id: ch for ch in chapter_contexts}
        chapter_ctx = by_id.get(plan.chapter_id)

    signals: ReactionSignals = extract_reaction_signals(
        activity_profile, cand_window.start, cand_window.end
    )

    package = CandidateContextPackage(
        schema_version=CANDIDATE_CONTEXT_SCHEMA_VERSION,
        context_version=CONTEXT_VERSION,
        candidate_id=candidate.id,
        global_context_ref=global_context.context_hash,
        chapter_id=chapter_ctx.chapter_id if chapter_ctx else None,
        chapter_context=chapter_ctx,
        before=before,
        candidate=cand_window,
        after=after,
        multimodal_evidence=extract_multimodal_evidence(multimodal_item),
        reaction_signals=signals,
        retrieval=retrieval or RetrievalProvenance(),
    )
    package.package_hash = compute_package_hash(package)
    return package


def compute_package_hash(package: CandidateContextPackage) -> str:
    """Deterministic identity of a candidate context package (cache key input)."""
    return stable_hash(
        {
            "schema": package.schema_version,
            "context_version": package.context_version,
            "candidate_id": package.candidate_id,
            "global_context_ref": package.global_context_ref,
            "chapter_id": package.chapter_id,
            "chapter_hash": package.chapter_context.transcript_hash
            if package.chapter_context
            else None,
            "before": package.before.model_dump(),
            "candidate": package.candidate.model_dump(),
            "after": package.after.model_dump(),
            "multimodal_evidence": package.multimodal_evidence,
            "reaction_signals": package.reaction_signals.model_dump()
            if package.reaction_signals
            else None,
        }
    )


def render_global_context_block(context: GlobalContext) -> str:
    """Human-readable global context block for prompts."""
    participants = (
        "; ".join(
            f"{p.id} ({p.role}): {p.description}".strip()
            for p in context.participants
        )
        or "unknown"
    )
    return "\n".join(
        [
            "--- GLOBAL VIDEO CONTEXT ---",
            f"Content type: {context.content_type}",
            f"Summary: {context.video_summary or '[unavailable]'}",
            f"Participants: {participants}",
            f"Main topics: {'; '.join(context.main_topics) or 'none recorded'}",
            f"Ongoing goals: {'; '.join(context.ongoing_goals) or 'none recorded'}",
            f"Recurring jokes: {'; '.join(context.recurring_jokes) or 'none recorded'}",
            f"Conflicts: {'; '.join(context.conflicts) or 'none recorded'}",
            f"Important context: {'; '.join(context.important_context) or 'none recorded'}",
        ]
    )


def render_chapter_context_block(chapter: Optional[ChapterContext]) -> str:
    """Human-readable chapter context block for prompts."""
    if chapter is None:
        return "--- CHAPTER CONTEXT ---\n[No chapter context available]"
    return "\n".join(
        [
            "--- CHAPTER CONTEXT ---",
            f"Chapter: {chapter.chapter_id} ({chapter.start:.0f}s - {chapter.end:.0f}s)",
            f"Topic: {chapter.topic or 'unknown'}",
            f"Summary: {chapter.summary or '[unavailable]'}",
            f"Events: {'; '.join(chapter.events) or 'none recorded'}",
            f"Setups: {'; '.join(chapter.setups) or 'none recorded'}",
            f"Payoffs: {'; '.join(chapter.payoffs) or 'none recorded'}",
            f"Open loops: {'; '.join(chapter.open_loops) or 'none recorded'}",
        ]
    )


def render_evidence_block(package: CandidateContextPackage) -> str:
    """Human-readable multimodal + activity evidence block for prompts."""
    lines = ["--- UPSTREAM EVIDENCE (evidence only, never a verdict) ---"]
    mm = package.multimodal_evidence
    if mm:
        flags = []
        for key in (
            "observable_event",
            "visual_payoff",
            "outside_payoff",
            "missing_setup",
            "insufficient_visual_evidence",
        ):
            if key in mm:
                flags.append(f"{key}={mm[key]}")
        if flags:
            lines.append("Multimodal flags: " + ", ".join(flags))
        if mm.get("confidence") is not None:
            lines.append(f"Multimodal confidence: {mm['confidence']}")
        region = mm.get("best_observed_region")
        if isinstance(region, dict):
            lines.append(
                f"Strongest observed span: +{region.get('start_offset')}s -> "
                f"+{region.get('end_offset')}s ({region.get('reason') or 'no reason given'})"
            )
        observations = mm.get("evidence")
        if isinstance(observations, list) and observations:
            for obs in observations[:6]:
                if isinstance(obs, dict):
                    lines.append(
                        f"Observed @ +{obs.get('timestamp_offset')}s: {obs.get('description')}"
                    )
        audio = mm.get("audio_features")
        if isinstance(audio, dict) and audio:
            lines.append(
                "Audio: "
                + ", ".join(f"{k}={v}" for k, v in audio.items() if v is not None)
            )
        visual = mm.get("visual_features")
        if isinstance(visual, dict) and visual:
            lines.append(
                "Visual: "
                + ", ".join(f"{k}={v}" for k, v in visual.items() if v is not None)
            )
        if mm.get("upstream_reason"):
            lines.append(f"Upstream editorial note: {mm['upstream_reason']}")
    else:
        lines.append("[No multimodal evidence available for this candidate]")

    signals = package.reaction_signals
    if signals and signals.available:
        lines.append(
            "Reaction signals: "
            f"laughter_like={signals.laughter_like_activity}, "
            f"sudden_energy_increase={signals.sudden_energy_increase}, "
            f"excited_speech_ratio={signals.excited_speech_ratio}, "
            f"pause_then_reaction={signals.pause_then_reaction}, "
            f"rapid_reaction_chain={signals.rapid_reaction_chain}, "
            f"dead_air_ratio={signals.dead_air_ratio}"
        )
        lines.append(
            "Reminder: high activity alone is not a reason to keep a clip, and low "
            "activity alone is not a reason to reject one."
        )
    else:
        lines.append("[No activity-derived reaction signals available]")
    return "\n".join(lines)


def render_candidate_block(package: CandidateContextPackage) -> str:
    """The BEFORE / CANDIDATE / AFTER block, with explicit boundary rules."""
    return "\n".join(
        [
            f"--- BEFORE ({package.before.start:.1f}s - {package.before.end:.1f}s) — "
            "context only, NOT part of the clip ---",
            package.before.transcript or "[No speech before this candidate]",
            "",
            f"--- CANDIDATE ({package.candidate.start:.1f}s - {package.candidate.end:.1f}s, "
            f"{package.candidate.duration:.1f}s) — THIS IS THE CLIP ---",
            f'"{package.candidate.transcript}"'
            if package.candidate.transcript
            else "[No speech inside this candidate]",
            "",
            f"--- AFTER ({package.after.start:.1f}s - {package.after.end:.1f}s) — "
            "context only, NOT part of the clip ---",
            package.after.transcript or "[No speech after this candidate]",
        ]
    )


def render_candidate_prompt(
    package: CandidateContextPackage,
    global_context: GlobalContext,
) -> str:
    """Full user message for the editorial and critic stages."""
    return "\n\n".join(
        [
            f"Candidate ID: {package.candidate_id}",
            render_global_context_block(global_context),
            render_chapter_context_block(package.chapter_context),
            render_candidate_block(package),
            render_evidence_block(package),
        ]
    )


def render_candidate_digest(package: CandidateContextPackage, label: str) -> str:
    """Compact evidence digest used by the comparative (listwise/pairwise) stages."""
    signals = package.reaction_signals
    signal_line = "[no activity signals]"
    if signals and signals.available:
        signal_line = (
            f"dead_air_ratio={signals.dead_air_ratio}, "
            f"sudden_energy_increase={signals.sudden_energy_increase}, "
            f"laughter_like={signals.laughter_like_activity}"
        )
    chapter = package.chapter_context
    return "\n".join(
        [
            f"### {label}",
            f"Duration: {package.candidate.duration:.1f}s",
            f"Chapter topic: {chapter.topic if chapter else 'unknown'}",
            "Immediately before (context only, not in the clip): "
            + (package.before.transcript[-400:] or "[none]"),
            "CLIP TRANSCRIPT: "
            + (f'"{package.candidate.transcript}"' if package.candidate.transcript else "[no speech]"),
            "Immediately after (context only, not in the clip): "
            + (package.after.transcript[:300] or "[none]"),
            f"Activity signals: {signal_line}",
        ]
    )
