"""Global and chapter context construction for contextual_reranker_v1.

The long transcript is never sent in a single request: chapter summaries are built
first (one call per chapter, cached), and the global context is derived from those
summaries (one call per source, cached).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from freecher_worker.transcription.models import Transcript

from .cache import ContextualCache, stable_hash
from .chapters import (
    ChapterPlan,
    build_chapter_plan,
    chapter_plan_hash,
    load_scene_change_times,
)
from .models import (
    ChapterContext,
    ChapterContextDocument,
    GlobalContext,
    Participant,
)
from .prompts import PROMPT_BUNDLE_VERSION, get_stage_prompt
from .provider import ContextualProvider, ContextualProviderError
from .versions import (
    CHAPTER_CONTEXT_SCHEMA_VERSION,
    GLOBAL_CONTEXT_SCHEMA_VERSION,
)

logger = logging.getLogger("freecher_worker")

#: Chapter transcripts are truncated before being sent, keeping requests bounded.
MAX_CHAPTER_TRANSCRIPT_CHARS = 12000


def _as_str_list(value: Any, limit: int = 12) -> List[str]:
    """Coerce a model-provided field into a clean list of short strings."""
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return []
    out: List[str] = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("description") or item.get("text") or item.get("name")
        else:
            text = item
        if text is None:
            continue
        text = str(text).strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _truncate(text: str, limit: int = MAX_CHAPTER_TRANSCRIPT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[... chapter transcript truncated ...]"


def build_chapter_contexts(
    plans: Sequence[ChapterPlan],
    provider: Optional[ContextualProvider],
    cache: ContextualCache,
    source_fingerprint: str,
) -> ChapterContextDocument:
    """Summarize every chapter, one cached API call each."""
    prompt_version, prompt_hash, system_prompt = get_stage_prompt("chapter")
    chapters: List[ChapterContext] = []

    for plan in plans:
        payload = {
            "source_fingerprint": source_fingerprint,
            "chapter_id": plan.chapter_id,
            "transcript_hash": plan.transcript_hash(),
        }
        req_hash = cache.request_hash(
            stage="chapter",
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            schema_version=CHAPTER_CONTEXT_SCHEMA_VERSION,
            payload=payload,
        )

        cached = cache.load("chapter", req_hash)
        parsed: Optional[Dict[str, Any]] = cached
        degraded = False
        degraded_reason: Optional[str] = None

        if cached is not None and provider is not None:
            provider.note_cache_hit("chapter")
        elif cached is None:
            if provider is None:
                parsed = None
                degraded = True
                degraded_reason = "no provider configured"
            else:
                user_content = "\n".join(
                    [
                        f"Chapter: {plan.chapter_id}",
                        f"Absolute time range: {plan.start:.1f}s - {plan.end:.1f}s "
                        f"({plan.duration:.0f}s)",
                        "",
                        "--- CHAPTER TRANSCRIPT ---",
                        _truncate(plan.timestamped_text()) or "[No speech in this chapter]",
                    ]
                )
                try:
                    parsed = provider.complete_json(system_prompt, user_content, stage="chapter")
                    cache.store("chapter", req_hash, parsed)
                except (ContextualProviderError, ValueError) as exc:
                    logger.warning(
                        f"[contextual-context] Chapter {plan.chapter_id} summary failed: {exc}"
                    )
                    parsed = None
                    degraded = True
                    degraded_reason = str(exc)[:300]

        if parsed is None:
            parsed = {}

        chapters.append(
            ChapterContext(
                chapter_id=plan.chapter_id,
                start=plan.start,
                end=plan.end,
                summary=str(parsed.get("summary") or "").strip(),
                participants=_as_str_list(parsed.get("participants")),
                topic=str(parsed.get("topic") or "").strip(),
                events=_as_str_list(parsed.get("events")),
                setups=_as_str_list(parsed.get("setups")),
                payoffs=_as_str_list(parsed.get("payoffs")),
                open_loops=_as_str_list(parsed.get("open_loops")),
                segment_ids=plan.segment_ids,
                transcript_hash=plan.transcript_hash(),
                boundary_source=plan.boundary_source,
                degraded=degraded,
                degraded_reason=degraded_reason,
            )
        )

    return ChapterContextDocument(
        source_fingerprint=source_fingerprint,
        model=getattr(provider, "model", None),
        prompt_version=prompt_version,
        chapter_plan_hash=chapter_plan_hash(plans),
        chapters=chapters,
    )


def build_global_context(
    chapter_doc: ChapterContextDocument,
    provider: Optional[ContextualProvider],
    cache: ContextualCache,
    source_fingerprint: str,
    duration_seconds: float,
) -> GlobalContext:
    """Derive one compact whole-video understanding from the chapter summaries."""
    prompt_version, prompt_hash, system_prompt = get_stage_prompt("global_context")

    chapter_digest = [
        {
            "chapter_id": ch.chapter_id,
            "start": round(ch.start, 1),
            "end": round(ch.end, 1),
            "summary": ch.summary,
            "topic": ch.topic,
            "participants": ch.participants,
            "events": ch.events,
            "setups": ch.setups,
            "payoffs": ch.payoffs,
            "open_loops": ch.open_loops,
        }
        for ch in chapter_doc.chapters
    ]
    payload = {
        "source_fingerprint": source_fingerprint,
        "chapter_plan_hash": chapter_doc.chapter_plan_hash,
        "chapters": chapter_digest,
    }
    req_hash = cache.request_hash(
        stage="global_context",
        prompt_version=prompt_version,
        prompt_hash=prompt_hash,
        schema_version=GLOBAL_CONTEXT_SCHEMA_VERSION,
        payload=payload,
    )

    cached = cache.load("global_context", req_hash)
    parsed: Optional[Dict[str, Any]] = cached
    degraded = False
    degraded_reason: Optional[str] = None

    if cached is not None and provider is not None:
        provider.note_cache_hit("global_context")
    elif cached is None:
        if provider is None:
            parsed = None
            degraded = True
            degraded_reason = "no provider configured"
        else:
            lines = [
                f"Video duration: {duration_seconds:.0f}s across {len(chapter_digest)} chapters.",
                "",
                "--- CHAPTER SUMMARIES ---",
            ]
            for ch in chapter_digest:
                lines.extend(
                    [
                        f"\n[{ch['chapter_id']}] {ch['start']:.0f}s - {ch['end']:.0f}s",
                        f"Topic: {ch['topic'] or 'unknown'}",
                        f"Summary: {ch['summary'] or '[unavailable]'}",
                        f"Participants: {', '.join(ch['participants']) or 'unknown'}",
                        f"Events: {'; '.join(ch['events']) or 'none recorded'}",
                        f"Setups: {'; '.join(ch['setups']) or 'none recorded'}",
                        f"Payoffs: {'; '.join(ch['payoffs']) or 'none recorded'}",
                        f"Open loops: {'; '.join(ch['open_loops']) or 'none recorded'}",
                    ]
                )
            try:
                parsed = provider.complete_json(
                    system_prompt, "\n".join(lines), stage="global_context"
                )
                cache.store("global_context", req_hash, parsed)
            except (ContextualProviderError, ValueError) as exc:
                logger.warning(f"[contextual-context] Global context failed: {exc}")
                parsed = None
                degraded = True
                degraded_reason = str(exc)[:300]

    if parsed is None:
        parsed = {}

    participants: List[Participant] = []
    raw_participants = parsed.get("participants")
    if isinstance(raw_participants, list):
        for idx, item in enumerate(raw_participants[:12], start=1):
            if isinstance(item, dict):
                participants.append(
                    Participant(
                        id=str(item.get("id") or f"person_{idx}"),
                        description=str(item.get("description") or "").strip(),
                        role=str(item.get("role") or "unknown").strip() or "unknown",
                    )
                )
            elif isinstance(item, str) and item.strip():
                participants.append(Participant(id=f"person_{idx}", description=item.strip()))

    content_type = str(parsed.get("content_type") or "other").strip().lower() or "other"
    if content_type not in ("stream", "interview", "vlog", "game", "challenge", "other"):
        content_type = "other"

    context = GlobalContext(
        source_fingerprint=source_fingerprint,
        video_summary=str(parsed.get("video_summary") or "").strip(),
        content_type=content_type,
        participants=participants,
        main_topics=_as_str_list(parsed.get("main_topics")),
        ongoing_goals=_as_str_list(parsed.get("ongoing_goals")),
        recurring_jokes=_as_str_list(parsed.get("recurring_jokes")),
        conflicts=_as_str_list(parsed.get("conflicts")),
        important_context=_as_str_list(parsed.get("important_context")),
        chapter_count=len(chapter_doc.chapters),
        duration_seconds=round(duration_seconds, 3),
        model=getattr(provider, "model", None),
        prompt_version=prompt_version,
        degraded=degraded,
        degraded_reason=degraded_reason,
    )
    context.context_hash = compute_global_context_hash(context, chapter_doc)
    return context


def compute_global_context_hash(
    context: GlobalContext,
    chapter_doc: ChapterContextDocument,
) -> str:
    """Deterministic identity of a global context, independent of creation timestamps."""
    return stable_hash(
        {
            "schema": GLOBAL_CONTEXT_SCHEMA_VERSION,
            "prompt_bundle": PROMPT_BUNDLE_VERSION,
            "source_fingerprint": context.source_fingerprint,
            "chapter_plan_hash": chapter_doc.chapter_plan_hash,
            "model": context.model,
            "prompt_version": context.prompt_version,
            "video_summary": context.video_summary,
            "content_type": context.content_type,
            "participants": [p.model_dump() for p in context.participants],
            "main_topics": context.main_topics,
            "ongoing_goals": context.ongoing_goals,
            "recurring_jokes": context.recurring_jokes,
            "conflicts": context.conflicts,
            "important_context": context.important_context,
        }
    )


def build_source_context(
    transcript: Optional[Transcript],
    provider: Optional[ContextualProvider],
    cache: ContextualCache,
    source_fingerprint: str,
    run_dir: Path | str,
    duration_seconds: float,
    target_seconds: float,
    min_seconds: float,
    max_seconds: float,
    use_scene_boundaries: bool = True,
) -> tuple[List[ChapterPlan], ChapterContextDocument, GlobalContext]:
    """Build the chapter plan, chapter summaries, and global context for one source."""
    scene_times = load_scene_change_times(run_dir) if use_scene_boundaries else []
    if scene_times:
        logger.info(
            f"[contextual-context] Using {len(scene_times)} cached scene changes for chapter boundaries."
        )

    plans = build_chapter_plan(
        transcript,
        target_seconds=target_seconds,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        scene_change_times=scene_times,
    )
    logger.info(f"[contextual-context] Planned {len(plans)} chapters.")

    chapter_doc = build_chapter_contexts(plans, provider, cache, source_fingerprint)
    global_context = build_global_context(
        chapter_doc, provider, cache, source_fingerprint, duration_seconds
    )
    return plans, chapter_doc, global_context
