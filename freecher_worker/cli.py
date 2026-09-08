"""Command-line interface for freecher-worker using Typer."""

from __future__ import annotations

import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from freecher_worker.config import Settings, get_settings
from freecher_worker.highlights.models import (
    CandidateDocument,
    CandidateWindow,
    Highlight,
    compute_candidate_set_id,
)
from freecher_worker.evaluation.models import (
    BlindEvaluationDocument,
    BlindEvaluationItem,
    ScorerPredictionDocument,
    ScorerPredictionItem,
    EvaluationMetrics,
    DisagreementReport,
)
from freecher_worker.evaluation.metrics import compute_evaluation_metrics
from freecher_worker.evaluation.annotator import run_terminal_annotator, preview_clip
from freecher_worker.evaluation.disagreements import extract_disagreements
from freecher_worker.scoring.heuristic import HeuristicScorer
from freecher_worker.scoring.llm import OpenAILLMScorer, compute_score_distribution
from freecher_worker.media.clipper import is_nvenc_available
from freecher_worker.pipeline.processor import Manifest, run_pipeline
from freecher_worker.rendering import (
    AVAILABLE_PRESETS,
    get_preset,
    render_highlights_for_run,
    render_single_short,
)
from freecher_worker.multimodal import (
    MultimodalReranker,
    OpenAIMultimodalProvider,
    PROMPT_VERSION_MULTIMODAL_V1,
    PROMPT_VERSION_MULTIMODAL_V1_1,
    SCORER_VERSION_MULTIMODAL_V1,
    SCORER_VERSION_MULTIMODAL_V1_1,
    generate_shortlist,
)
from freecher_worker.transcription.models import Transcript
from freecher_worker.utils.json_io import load_json, save_json

app = typer.Typer(
    name="freecher-worker",
    help="freecher-worker: Automated video highlight extraction and clipping worker.",
    add_completion=False,
)
console = Console()


def _format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    total_secs = int(seconds)
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _resolve_run_path(run_arg: Path | str, base_dir: Path = Path("runs")) -> Path:
    """Resolve a run argument to an existing run directory."""
    p = Path(run_arg)
    if p.is_dir() and (p / "manifest.json").is_file():
        return p
    if p.is_file() and p.name == "manifest.json":
        return p.parent

    candidate = base_dir / run_arg
    if candidate.is_dir() and (candidate / "manifest.json").is_file():
        return candidate

    # Search in base_dir for prefix or substring
    if base_dir.is_dir():
        matches = [d for d in base_dir.iterdir() if d.is_dir() and str(run_arg) in d.name]
        if matches:
            return sorted(matches, key=lambda d: d.name, reverse=True)[0]

    raise FileNotFoundError(f"Run directory or manifest not found for: {run_arg}")


@app.command("process")
def process_command(
    video_path: Path = typer.Argument(
        ...,
        help="Path to input video file (e.g. test.mp4)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Base output directory for runs (default: 'runs')",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Spoken language code (e.g. 'ru', 'en'). None for auto-detection.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Whisper model name (tiny, base, small, medium, turbo, large-v3)",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help="Inference device: 'cuda' or 'cpu'",
    ),
    compute_type: Optional[str] = typer.Option(
        None,
        "--compute-type",
        "-c",
        help="CTranslate2 compute type (int8, int8_float16, float16, float32, default)",
    ),
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        "-k",
        help="Number of top highlight clips to generate (default: 5)",
    ),
    scorer: Optional[str] = typer.Option(
        None,
        "--scorer",
        "-s",
        help="Highlight scoring method: 'heuristic' or 'llm'",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force recomputation of all stages, ignoring existing cache.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Specific run directory name to create or resume.",
    ),
    analysis_only: bool = typer.Option(
        False,
        "--analysis-only",
        help="Run analysis, scoring, and ranking without rendering video clips.",
    ),
) -> None:
    """Process video file: transcribe, discover highlight moments, and cut clips."""
    base_settings = get_settings()

    overrides = {}
    if model is not None:
        overrides["asr_model"] = model
    if device is not None:
        overrides["asr_device"] = device
    if compute_type is not None:
        overrides["asr_compute_type"] = compute_type
    if language is not None:
        overrides["asr_language"] = language
    if top_k is not None:
        overrides["highlight_top_k"] = top_k
    if scorer is not None:
        overrides["scorer"] = scorer
    if output is not None:
        overrides["output_dir"] = output

    cfg = base_settings.model_copy(update=overrides)

    try:
        manifest = run_pipeline(
            video_path=video_path,
            output_dir=cfg.output_dir,
            config=cfg,
            force=force,
            run_id=run_id,
            analysis_only=analysis_only,
        )

        console.print(f"\n[bold green]Processing completed successfully in {manifest.timings.total_seconds:.1f}s![/bold green]")
        if manifest.highlights:
            table = Table(title="Top Highlights")
            table.add_column("Rank", justify="center", style="cyan")
            table.add_column("Time Range", justify="center", style="magenta")
            table.add_column("Duration", justify="center")
            table.add_column("Score", justify="center", style="bold yellow")
            table.add_column("Clip File", style="green")
            table.add_column("Reason")

            for hl in manifest.highlights:
                time_range = f"{_format_timestamp(hl.start)} - {_format_timestamp(hl.end)}"
                table.add_row(
                    str(hl.rank),
                    time_range,
                    f"{hl.duration:.1f}s",
                    f"{hl.score:.1f}",
                    hl.file or "[dim]skipped[/dim]",
                    hl.reason,
                )
            console.print(table)
        else:
            console.print("[yellow]No highlights were found in the video.[/yellow]")

    except Exception as exc:
        console.print(f"\n[bold red]Error during processing:[/bold red] {exc}")
        sys.exit(1)


@app.command("inspect")
def inspect_command(
    run_dir: Path = typer.Argument(
        ...,
        help="Path or name of run directory (e.g. runs/20260907_082719_test)",
    ),
) -> None:
    """Inspect and display ranked highlights and detailed scores from a previous run."""
    try:
        resolved_dir = _resolve_run_path(run_dir)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    manifest_file = resolved_dir / "manifest.json"
    data = load_json(manifest_file)
    manifest = Manifest.model_validate(data)

    console.print(f"\n[bold cyan]Run Inspection: {resolved_dir.name}[/bold cyan]")
    console.print(f"Source: [bold]{manifest.source}[/bold] (Duration: {_format_timestamp(manifest.source_fingerprint.duration_seconds)})")
    console.print(
        f"ASR: model={manifest.asr.model} | device={manifest.asr.device} | compute={manifest.asr.compute_type} | lang={manifest.asr.language}"
    )
    console.print(
        f"Scorer: {manifest.scoring.scorer} (v{manifest.scoring.scorer_version}) "
        + (f"[yellow][Fallback used: {manifest.scoring.fallback_reason}][/yellow]" if manifest.scoring.fallback_used else "")
    )
    console.print(
        f"Timings: total={manifest.timings.total_seconds:.1f}s (ASR={manifest.timings.transcription_seconds:.1f}s, Clipping={manifest.timings.clipping_seconds:.1f}s)\n"
    )

    if not manifest.highlights:
        console.print("[yellow]No highlights recorded in this run.[/yellow]")
        return

    # Load candidates if available to show full text preview
    cand_text_map: dict[str, str] = {}
    candidates_file = resolved_dir / "candidates.json"
    if candidates_file.is_file():
        try:
            cand_data = load_json(candidates_file)
            if isinstance(cand_data, dict) and "candidates" in cand_data:
                cand_list = cand_data["candidates"]
            else:
                cand_list = cand_data
            for c in cand_list:
                cand_text_map[c["id"]] = c.get("text", "")
        except Exception:
            pass

    for hl in manifest.highlights:
        time_range = f"{_format_timestamp(hl.start)} - {_format_timestamp(hl.end)}"
        breakdown = hl.score_breakdown
        subscores_str = ""
        if breakdown:
            subscores_str = (
                f"[dim]| Hook: {breakdown.hook_score:.0f} | Standalone: {breakdown.standalone_score:.0f} "
                f"| Emotion: {breakdown.emotion_score:.0f} | Info: {breakdown.information_score:.0f} "
                f"| Share: {breakdown.shareability_score:.0f}[/dim]"
            )

        header = f"[bold cyan]#{hl.rank}[/bold cyan]  {time_range}  ({hl.duration:.1f}s)  [bold yellow]Score: {hl.score:.1f}[/bold yellow] {subscores_str}"
        preview_text = cand_text_map.get(hl.candidate_id, "")
        if not preview_text and hl.reason:
            preview_text = ""

        content = ""
        if preview_text:
            content += f"\"{preview_text}\"\n\n"
        content += f"[bold]Reason:[/bold] {hl.reason}\n"
        if hl.file:
            content += f"[bold]Clip:[/bold] {resolved_dir / hl.file}"

        console.print(Panel(content.strip(), title=header, title_align="left", expand=False))

    render_manifest_file = resolved_dir / "render_manifest.json"
    if render_manifest_file.is_file():
        try:
            rm_data = load_json(render_manifest_file)
            rm_shorts = rm_data.get("shorts", [])
            if rm_shorts:
                console.print(f"\n[bold green]Rendered Vertical Shorts ({len(rm_shorts)}):[/bold green]")
                r_table = Table(show_header=True, header_style="bold green")
                r_table.add_column("Rank", width=6, justify="center")
                r_table.add_column("Candidate", style="dim", width=14)
                r_table.add_column("Refined Window", width=18)
                r_table.add_column("Dur", width=8, justify="right")
                r_table.add_column("Resolution", width=11)
                r_table.add_column("Encoder", width=12)
                r_table.add_column("Short File")
                for s in rm_shorts:
                    ref_win = f"{_format_timestamp(s['refined_start'])} -> {_format_timestamp(s['refined_end'])}"
                    r_table.add_row(
                        f"#{s['rank']}",
                        s.get("candidate_id", ""),
                        ref_win,
                        f"{s.get('duration', 0.0):.1f}s",
                        s.get("resolution", "1080x1920"),
                        s.get("encoder", "unknown"),
                        s.get("file", ""),
                    )
                console.print(r_table)
        except Exception:
            pass


@app.command("export-eval")
def export_eval_command(
    run_dir: str = typer.Argument(
        ...,
        help="Path or name of run directory to export evaluation set from",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination path for evaluation file (default: evaluation_blind.json or evaluation.json)",
    ),
    blind: bool = typer.Option(
        False,
        "--blind",
        help="Export blind evaluation set: stripped of model scores/ranks, deterministically shuffled",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for deterministic candidate shuffling in blind mode (default: 42)",
    ),
) -> None:
    """Export candidate highlights for manual human evaluation."""
    try:
        resolved_dir = _resolve_run_path(run_dir)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    candidates_file = resolved_dir / "candidates.json"
    highlights_file = resolved_dir / "highlights.json"
    manifest_file = resolved_dir / "manifest.json"

    if not candidates_file.is_file():
        console.print(f"[bold red]Error:[/bold red] candidates.json not found in {resolved_dir}")
        sys.exit(1)

    cand_data = load_json(candidates_file)
    candidate_set_id = ""
    if isinstance(cand_data, dict) and "candidates" in cand_data:
        raw_candidates = cand_data["candidates"]
        candidate_set_id = cand_data.get("candidate_set_id", "")
    else:
        raw_candidates = cand_data

    source_video = None
    if manifest_file.is_file():
        try:
            m = load_json(manifest_file)
            source_video = m.get("source")
        except Exception:
            pass

    if not candidate_set_id:
        # Compute fallback candidate_set_id if not present
        candidate_set_id = compute_candidate_set_id(
            transcript_hash=cand_data.get("transcript_hash", "") if isinstance(cand_data, dict) else "",
            min_seconds=cand_data.get("min_seconds", 30.0) if isinstance(cand_data, dict) else 30.0,
            target_seconds=cand_data.get("target_seconds", 60.0) if isinstance(cand_data, dict) else 60.0,
            max_seconds=cand_data.get("max_seconds", 90.0) if isinstance(cand_data, dict) else 90.0,
            overlap_seconds=cand_data.get("overlap_seconds", 15.0) if isinstance(cand_data, dict) else 15.0,
        )

    # 1. BLIND MODE: strictly omit any model prediction, score, rank, or subscores
    if blind:
        blind_items = []
        for cand in raw_candidates:
            blind_items.append(
                BlindEvaluationItem(
                    candidate_id=cand["id"],
                    start=cand["start"],
                    end=cand["end"],
                    duration=cand["duration"],
                    text=cand["text"],
                    segment_ids=cand.get("segment_ids", []),
                    human_score=None,
                    publishable=None,
                    human_notes=None,
                )
            )

        # Deterministic shuffle using seed
        rnd = random.Random(seed)
        rnd.shuffle(blind_items)

        blind_doc = BlindEvaluationDocument(
            candidate_set_id=candidate_set_id,
            source_video=source_video,
            total_candidates=len(blind_items),
            labeled_candidates=0,
            seed=seed,
            items=blind_items,
        )

        target_path = output or (resolved_dir / "evaluation_blind.json")
        save_json(blind_doc, target_path)

        console.print(f"\n[bold green]Exported {len(blind_items)} BLIND evaluation candidates to:[/bold green] {target_path}")
        console.print(f"Candidate Set ID: [bold]{candidate_set_id}[/bold]")
        console.print(f"Random Seed:      {seed} (deterministic order)")
        console.print("All model scores, ranks, and reasons have been completely removed.\n")
        return

    # 2. LEGACY NON-BLIND MODE: includes model scores and ranks for debugging
    highlight_map: dict[str, Any] = {}
    if highlights_file.is_file():
        try:
            for item in load_json(highlights_file):
                highlight_map[item["candidate_id"]] = item
        except Exception:
            pass
    elif manifest_file.is_file():
        try:
            man = load_json(manifest_file)
            for item in man.get("highlights", []):
                highlight_map[item["candidate_id"]] = item
        except Exception:
            pass

    eval_items = []
    for cand in raw_candidates:
        cid = cand["id"]
        hl_info = highlight_map.get(cid, {})

        score_val = hl_info.get("score")
        rank_val = hl_info.get("rank")
        reason_val = hl_info.get("reason", "")
        sb = hl_info.get("score_breakdown") or {}

        subscores = {
            "hook_score": sb.get("hook_score"),
            "standalone_score": sb.get("standalone_score"),
            "emotion_score": sb.get("emotion_score"),
            "information_score": sb.get("information_score"),
            "shareability_score": sb.get("shareability_score"),
        }

        eval_items.append({
            "candidate_id": cid,
            "rank": rank_val,
            "start": cand["start"],
            "end": cand["end"],
            "duration": cand["duration"],
            "text": cand["text"],
            "score": score_val,
            "subscores": subscores,
            "reason": reason_val,
            "human_label": None,
            "human_score": None,
            "human_notes": None,
        })

    # Sort: ranked items first by rank, then unranked by score descending
    eval_items.sort(key=lambda x: (0 if x["rank"] is not None else 1, x["rank"] if x["rank"] is not None else - (x["score"] or 0)))

    target_path = output or (resolved_dir / "evaluation.json")
    save_json(eval_items, target_path)

    console.print(f"\n[bold green]Exported {len(eval_items)} evaluation candidates to:[/bold green] {target_path}")
    console.print("Fields for manual human evaluation ready: human_label, human_score (0-4), human_notes\n")


@app.command("label-eval")
def label_eval_command(
    eval_file: Path = typer.Argument(
        ...,
        help="Path to evaluation file (e.g. runs/<run_id>/evaluation_blind.json)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    video: Optional[Path] = typer.Option(
        None,
        "--video",
        "-v",
        help="Optional path to source video file for [p]review clipping during annotation",
    ),
    all_items: bool = typer.Option(
        False,
        "--all",
        help="Re-label all candidates from start, ignoring existing human scores",
    ),
) -> None:
    """Interactive terminal annotation workflow for blind evaluation."""
    data = load_json(eval_file)
    if isinstance(data, dict) and "items" in data:
        eval_doc = BlindEvaluationDocument.model_validate(data)
    elif isinstance(data, list):
        items = [
            BlindEvaluationItem(
                candidate_id=d.get("candidate_id", d.get("id", "")),
                start=d.get("start", 0.0),
                end=d.get("end", 0.0),
                duration=d.get("duration", 0.0),
                text=d.get("text", ""),
                segment_ids=d.get("segment_ids", []),
                human_score=d.get("human_score"),
                publishable=d.get("publishable"),
                human_notes=d.get("human_notes"),
            )
            for d in data
        ]
        eval_doc = BlindEvaluationDocument(
            candidate_set_id="legacy_cset",
            total_candidates=len(items),
            items=items,
        )
    else:
        console.print(f"[bold red]Error:[/bold red] Unrecognized evaluation format in {eval_file}")
        raise typer.Exit(code=1)

    video_path = video
    if video_path is None and eval_doc.source_video:
        cand_v = Path(eval_doc.source_video)
        if cand_v.is_file():
            video_path = cand_v

    run_terminal_annotator(
        eval_doc=eval_doc,
        eval_file_path=eval_file,
        video_path=video_path,
        re_label_all=all_items,
    )


@app.command("preview-candidate")
def preview_candidate_command(
    video_path: Path = typer.Argument(
        ...,
        help="Path to source video file",
        exists=True,
        file_okay=True,
        readable=True,
    ),
    start: float = typer.Option(..., "--start", "-s", help="Start time in seconds"),
    end: float = typer.Option(..., "--end", "-e", help="End time in seconds"),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for preview MP4 clip",
    ),
    open_clip: bool = typer.Option(
        True,
        "--open/--no-open",
        help="Open clip in system default media player",
    ),
) -> None:
    """Extract and optionally preview a candidate video window via FFmpeg."""
    try:
        clip_path = preview_clip(
            video_path=video_path,
            start=start,
            end=end,
            output_path=output,
            open_player=open_clip,
        )
        console.print(f"[bold green]Preview clip generated:[/bold green] {clip_path}")
    except Exception as exc:
        console.print(f"[bold red]Preview generation failed:[/bold red] {exc}")
        raise typer.Exit(code=1)


@app.command("score-run")
def score_run_command(
    run_dir: str = typer.Argument(
        ...,
        help="Path or ID of existing run directory (e.g. 'runs/20260907_sample_video')",
    ),
    scorer: str = typer.Option(
        "heuristic",
        "--scorer",
        "-s",
        help="Highlight scoring method ('heuristic', 'highlight_v2', or 'llm')",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name for LLM scorer (e.g. 'gpt-4o-mini')",
    ),
    allow_fallback: bool = typer.Option(
        False,
        "--allow-fallback",
        help="Allow fallback to heuristic if LLM fails (default: False for benchmark purity)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Custom destination for scores JSON file (default: <run_dir>/scores/<scorer>_<version>.json)",
    ),
    source_video: Optional[Path] = typer.Option(
        None,
        "--source-video",
        "-v",
        help="Path to source video file (used by multimodal reranker if moved)",
    ),
) -> None:
    """Score a frozen candidate set in an existing run without re-transcribing or clipping."""
    try:
        resolved_dir = _resolve_run_path(run_dir)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    cand_file = resolved_dir / "candidates.json"
    if not cand_file.is_file():
        console.print(f"[bold red]Error:[/bold red] candidates.json not found in {resolved_dir}")
        raise typer.Exit(code=1)

    cand_data = load_json(cand_file)
    if isinstance(cand_data, dict) and "candidates" in cand_data:
        cand_doc = CandidateDocument.model_validate(cand_data)
        candidates = cand_doc.candidates
        candidate_set_id = cand_doc.candidate_set_id
        # Integrity verification
        expected_id = compute_candidate_set_id(
            transcript_hash=cand_doc.transcript_hash,
            min_seconds=cand_doc.min_seconds,
            target_seconds=cand_doc.target_seconds,
            max_seconds=cand_doc.max_seconds,
            overlap_seconds=cand_doc.overlap_seconds,
            segmentation_version=cand_doc.segmentation_version,
        )
        if candidate_set_id and candidate_set_id != expected_id:
            console.print(
                f"[bold red]Error:[/bold red] candidate_set_id integrity check failed! "
                f"Document has '{candidate_set_id}', but computed '{expected_id}'."
            )
            raise typer.Exit(code=1)
    elif isinstance(cand_data, list):
        candidates = [CandidateWindow.model_validate(c) for c in cand_data]
        candidate_set_id = "legacy_cset"
    else:
        console.print(f"[bold red]Error:[/bold red] Invalid candidates format in {cand_file}")
        raise typer.Exit(code=1)

    if not candidates:
        console.print("[yellow]No candidates found in candidates.json.[/yellow]")
        raise typer.Exit(code=0)

    # Instantiate scorer
    requested_scorer = scorer.lower()
    if requested_scorer in ("multimodal", "multimodal_v1"):
        settings = get_settings()
        actual_model = model or settings.multimodal_model or "gpt-4o-mini"
        provider = OpenAIMultimodalProvider(
            base_url=settings.multimodal_base_url,
            api_key=settings.multimodal_api_key,
            model=actual_model,
        )
        reranker = MultimodalReranker(
            provider=provider,
            heuristic_top_k=settings.multimodal_heuristic_top_k,
            llm_top_k=settings.multimodal_llm_top_k,
            max_candidates=settings.multimodal_max_candidates,
            allow_missing_llm=allow_fallback,
            source_video=source_video,
        )
        console.print(f"Executing [bold]multimodal_v1[/bold] reranker on {resolved_dir}...")
        pred_doc = reranker.rerank_run(
            resolved_dir,
            output_file=output,
            source_video_override=source_video,
        )
        target_path = output or (resolved_dir / "scores" / "multimodal_v1.json")
        console.print(f"[bold green]Multimodal predictions saved to:[/bold green] {target_path}")
        console.print(f"Candidate Set ID: {pred_doc.candidate_set_id}")
        if pred_doc.predictions:
            console.print(f"Top candidate: #{pred_doc.predictions[0].candidate_id} (Score: {pred_doc.predictions[0].score:.2f})")
        if pred_doc.distribution_diagnostics and pred_doc.predictions:
            dist_diag = pred_doc.distribution_diagnostics
            console.print("\n[bold]Score Distribution Diagnostics:[/bold]")
            dist_table = Table(box=box.SIMPLE)
            dist_table.add_column("Metric", style="cyan")
            dist_table.add_column("Value", style="bold")
            dist_table.add_row("Min", f"{dist_diag.min:.2f}")
            dist_table.add_row("Median", f"{dist_diag.median:.2f}")
            dist_table.add_row("Max", f"{dist_diag.max:.2f}")
            dist_table.add_row("Unique Scores", f"{dist_diag.unique_score_count_raw} / {dist_diag.unique_score_count_rounded}")
            dist_table.add_row("Std Dev", f"{dist_diag.standard_deviation:.2f}")
            console.print(dist_table)
        return
    elif requested_scorer in ("llm", "highlight_v2_1"):
        actual_scorer = "highlight_v2_1"
        scorer_ver = "highlight_v2_1"
        settings = get_settings()
        actual_model = model or settings.llm_model or "gpt-4o-mini"
        active_scorer = OpenAILLMScorer(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=actual_model,
            allow_fallback=allow_fallback,
            scorer_version="highlight_v2_1",
        )
    elif requested_scorer == "highlight_v2":
        actual_scorer = "highlight_v2"
        scorer_ver = "highlight_v2"
        settings = get_settings()
        actual_model = model or settings.llm_model or "gpt-4o-mini"
        active_scorer = OpenAILLMScorer(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=actual_model,
            allow_fallback=allow_fallback,
            scorer_version="highlight_v2",
        )
    else:
        requested_scorer = "heuristic"
        actual_scorer = "heuristic"
        scorer_ver = "heuristic_v1"
        actual_model = None
        active_scorer = HeuristicScorer()

    # Load transcript if present for context extraction
    transcript_file = resolved_dir / "transcript.json"
    transcript_obj = None
    if transcript_file.is_file():
        try:
            transcript_obj = Transcript.model_validate(load_json(transcript_file))
        except Exception as exc:
            console.print(f"[dim]Note: Could not load transcript.json for context: {exc}[/dim]")

    console.print(
        f"Scoring {len(candidates)} candidates using [bold]{actual_scorer}[/bold] (v{scorer_ver})..."
    )
    scores = active_scorer.score_batch(candidates, transcript=transcript_obj)

    # Sort descending by score
    cand_score_pairs = list(zip(candidates, scores))
    cand_score_pairs.sort(key=lambda cs: cs[1].score, reverse=True)

    pred_items = [
        ScorerPredictionItem(
            candidate_id=c.id,
            rank=r_idx,
            score=sc_item.score,
            reason=sc_item.reason,
            subscores=getattr(sc_item, "subscores", None) or {
                "hook_score": sc_item.hook_score,
                "standalone_score": sc_item.standalone_score,
                "emotion_score": sc_item.emotion_score,
                "information_score": sc_item.information_score,
                "shareability_score": getattr(sc_item, "shareability_score", 0.0),
            },
            scorer=actual_scorer,
            scorer_version=scorer_ver,
            requested_model=model if requested_scorer in ("llm", "highlight_v2", "highlight_v2_1") else None,
            actual_model=getattr(sc_item, "actual_model", actual_model),
            fallback_used=getattr(sc_item, "fallback_used", False),
            fallback_reason=getattr(sc_item, "fallback_reason", None),
            llm_quality_score=getattr(sc_item, "llm_quality_score", None),
            final_score=getattr(sc_item, "final_score", sc_item.score),
            positive_score=getattr(sc_item, "positive_score", None),
            total_penalty=getattr(sc_item, "total_penalty", None),
            applied_caps=getattr(sc_item, "applied_caps", None),
            raw_positive_dimensions=getattr(sc_item, "raw_positive_dimensions", None),
            raw_negative_dimensions=getattr(sc_item, "raw_negative_dimensions", None),
            flags=getattr(sc_item, "flags", None),
        )
        for r_idx, (c, sc_item) in enumerate(cand_score_pairs, start=1)
    ]

    # Compute score distribution diagnostics
    scores_list = [p.score for p in pred_items]
    unrounded_list = [getattr(p, "final_score", p.score) for p in pred_items]
    dist_diag = compute_score_distribution(scores_list, unrounded_list) if scores_list else None

    pred_doc = ScorerPredictionDocument(
        candidate_set_id=candidate_set_id,
        scorer=actual_scorer,
        scorer_version=scorer_ver,
        model=actual_model,
        predictions=pred_items,
        requested_scorer=requested_scorer,
        actual_scorer=actual_scorer,
        prompt_version=getattr(active_scorer, "prompt_version", None),
        prompt_hash=getattr(active_scorer, "prompt_hash", None),
        score_formula_version=getattr(active_scorer, "score_formula_version", None),
        context_window_seconds=getattr(active_scorer, "context_window_seconds", None),
        temperature=getattr(active_scorer, "temperature", None),
        distribution_diagnostics=dist_diag,
    )

    scores_dir = resolved_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    if scorer_ver in ("highlight_v2_1", "highlight_v2"):
        file_base = scorer_ver
    elif scorer_ver.startswith(f"{actual_scorer}_"):
        file_base = scorer_ver
    else:
        file_base = f"{actual_scorer}_{scorer_ver}"
    target_path = output or (scores_dir / f"{file_base}.json")
    save_json(pred_doc, target_path)

    console.print(f"[bold green]Predictions saved to:[/bold green] {target_path}")
    console.print(f"Candidate Set ID: {candidate_set_id}")
    if pred_items:
        console.print(f"Top candidate: #{pred_items[0].candidate_id} (Score: {pred_items[0].score:.1f})")

    # Display distribution summary table
    if dist_diag and pred_items:
        console.print("\n[bold]Score Distribution Diagnostics:[/bold]")
        dist_table = Table(box=box.SIMPLE)
        dist_table.add_column("Metric", style="cyan")
        dist_table.add_column("Value", style="bold")
        dist_table.add_row("Min", f"{dist_diag.min:.2f}")
        dist_table.add_row("P10", f"{dist_diag.p10:.2f}")
        dist_table.add_row("P25", f"{dist_diag.p25:.2f}")
        dist_table.add_row("Median", f"{dist_diag.median:.2f}")
        dist_table.add_row("P75", f"{dist_diag.p75:.2f}")
        dist_table.add_row("P90", f"{dist_diag.p90:.2f}")
        dist_table.add_row("Max", f"{dist_diag.max:.2f}")
        dist_table.add_row(
            "Unique Scores (raw / rounded)",
            f"{dist_diag.unique_score_count_raw} / {dist_diag.unique_score_count_rounded}",
        )
        dist_table.add_row(
            "Zero Score Count",
            f"{dist_diag.zero_score_count} ({dist_diag.zero_score_count / len(pred_items):.1%})",
        )
        dist_table.add_row("Std Dev", f"{dist_diag.standard_deviation:.2f}")
        console.print(dist_table)
        if dist_diag.warning:
            console.print(f"[bold red]WARNING:[/bold red] {dist_diag.warning}\n")


@app.command("multimodal-score")
def multimodal_score_command(
    run_dir: Path = typer.Argument(
        ...,
        help="Path to run directory containing candidates.json",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Custom output path for predictions JSON (default: scores/multimodal_v1_1.json or scores/multimodal_v1.json)",
    ),
    scorer_version: str = typer.Option(
        SCORER_VERSION_MULTIMODAL_V1_1,
        "--scorer-version",
        help=f"Scorer version to run ({SCORER_VERSION_MULTIMODAL_V1_1} or {SCORER_VERSION_MULTIMODAL_V1})",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Vision-capable model name (e.g. gpt-4o-mini, gpt-4o)",
    ),
    heuristic_top_k: Optional[int] = typer.Option(
        None,
        "--heuristic-top-k",
        help="Number of candidates taken from heuristic_v1 for shortlist (default: 20 for v1.1, 12 for v1)",
    ),
    llm_top_k: Optional[int] = typer.Option(
        None,
        "--llm-top-k",
        help="Number of candidates taken from highlight_v2_1 for shortlist (default: 20 for v1.1, 12 for v1)",
    ),
    max_candidates: Optional[int] = typer.Option(
        None,
        "--max-candidates",
        help="Maximum capacity of candidate shortlist (default: 32 for v1.1, 20 for v1)",
    ),
    allow_missing_llm: bool = typer.Option(
        False,
        "--allow-missing-llm",
        help="Allow heuristic-only retrieval if scores/highlight_v2_1.json is missing (disabled for benchmark)",
    ),
    force_rescore: bool = typer.Option(
        False,
        "--force-rescore",
        help="Force re-scoring via API even if cached response exists",
    ),
    force_repackage: bool = typer.Option(
        False,
        "--force-repackage",
        help="Force re-extracting frames and audio features even if cached package exists",
    ),
    source_video: Optional[Path] = typer.Option(
        None,
        "--source-video",
        "-v",
        help="Path to source video file if moved or not found in manifest/media.json",
    ),
) -> None:
    """Run Multimodal Highlight Reranker (v1.1 default or v1) on a deterministic candidate shortlist."""
    resolved_dir = run_dir.resolve()
    settings = get_settings()
    actual_model = model or settings.multimodal_model or "gpt-4o-mini"

    is_v1_1 = scorer_version == SCORER_VERSION_MULTIMODAL_V1_1
    prompt_ver = (
        PROMPT_VERSION_MULTIMODAL_V1_1 if is_v1_1 else PROMPT_VERSION_MULTIMODAL_V1
    )

    provider = OpenAIMultimodalProvider(
        base_url=settings.multimodal_base_url,
        api_key=settings.multimodal_api_key,
        model=actual_model,
        prompt_version=prompt_ver,
    )

    reranker = MultimodalReranker(
        provider=provider,
        scorer_version=scorer_version,
        heuristic_top_k=heuristic_top_k,
        llm_top_k=llm_top_k,
        max_candidates=max_candidates,
        allow_missing_llm=allow_missing_llm,
        force_rescore=force_rescore,
        force_repackage=force_repackage,
        source_video=source_video,
    )

    console.print(f"Executing [bold]{scorer_version}[/bold] reranker on {resolved_dir}...")
    pred_doc = reranker.rerank_run(
        resolved_dir,
        output_file=output,
        source_video_override=source_video,
    )

    target_path = output or (resolved_dir / "scores" / f"{scorer_version}.json")
    console.print(f"[bold green]Multimodal predictions saved to:[/bold green] {target_path}")
    console.print(f"Candidate Set ID: {pred_doc.candidate_set_id}")
    console.print(f"Shortlist candidates evaluated: {len(pred_doc.predictions)}")
    if pred_doc.predictions:
        console.print(f"Top candidate: #{pred_doc.predictions[0].candidate_id} (Score: {pred_doc.predictions[0].score:.2f})")

    if pred_doc.distribution_diagnostics and pred_doc.predictions:
        dist_diag = pred_doc.distribution_diagnostics
        console.print("\n[bold]Score Distribution Diagnostics:[/bold]")
        dist_table = Table(box=box.SIMPLE)
        dist_table.add_column("Metric", style="cyan")
        dist_table.add_column("Value", style="bold")
        dist_table.add_row("Min", f"{dist_diag.min:.2f}")
        dist_table.add_row("P10", f"{dist_diag.p10:.2f}")
        dist_table.add_row("P25", f"{dist_diag.p25:.2f}")
        dist_table.add_row("Median", f"{dist_diag.median:.2f}")
        dist_table.add_row("P75", f"{dist_diag.p75:.2f}")
        dist_table.add_row("P90", f"{dist_diag.p90:.2f}")
        dist_table.add_row("Max", f"{dist_diag.max:.2f}")
        dist_table.add_row(
            "Unique Scores (raw / rounded)",
            f"{dist_diag.unique_score_count_raw} / {dist_diag.unique_score_count_rounded}",
        )
        dist_table.add_row(
            "Zero Score Count",
            f"{dist_diag.zero_score_count} ({dist_diag.zero_score_count / len(pred_doc.predictions):.1%})",
        )
        dist_table.add_row("Std Dev", f"{dist_diag.standard_deviation:.2f}")
        console.print(dist_table)


@app.command("evaluate")
def evaluate_command(
    eval_file: Path = typer.Argument(
        ...,
        help="Path to evaluation JSON (e.g. evaluation_blind.json)",
        exists=True,
        file_okay=True,
        readable=True,
    ),
    scores_file: Path = typer.Argument(
        ...,
        help="Path to model predictions JSON (e.g. scores/heuristic_v1.json)",
        exists=True,
        file_okay=True,
        readable=True,
    ),
    k: str = typer.Option(
        "5,10",
        "--k",
        help="Comma-separated K values for metrics (e.g. 5,10)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output raw JSON metrics instead of formatted table",
    ),
) -> None:
    """Evaluate model highlight predictions against human judgments."""
    try:
        k_values = [int(x.strip()) for x in k.split(",") if x.strip()]
    except Exception:
        console.print(f"[bold red]Error:[/bold red] Invalid K values: '{k}'. Must be comma-separated integers.")
        raise typer.Exit(code=1)

    eval_data = load_json(eval_file)
    if isinstance(eval_data, dict) and "items" in eval_data:
        eval_doc = BlindEvaluationDocument.model_validate(eval_data)
    elif isinstance(eval_data, list):
        items = [BlindEvaluationItem.model_validate(x) for x in eval_data]
        eval_doc = BlindEvaluationDocument(
            candidate_set_id="legacy_cset",
            total_candidates=len(items),
            items=items,
        )
    else:
        console.print(f"[bold red]Error:[/bold red] Invalid evaluation format in {eval_file}")
        raise typer.Exit(code=1)

    scores_data = load_json(scores_file)
    scores_doc = ScorerPredictionDocument.model_validate(scores_data)

    if eval_doc.candidate_set_id != scores_doc.candidate_set_id:
        console.print("[bold red]Candidate Set ID Mismatch![/bold red]")
        console.print(f"Evaluation Document Candidate Set: {eval_doc.candidate_set_id}")
        console.print(f"Scores Document Candidate Set:     {scores_doc.candidate_set_id}")
        console.print("Models must be evaluated on the identical frozen candidate set.")
        raise typer.Exit(code=1)

    metrics = compute_evaluation_metrics(eval_doc, scores_doc, k_values=k_values)

    if json_output:
        console.print(metrics.model_dump_json(indent=2))
        return

    # Render formatted evaluation table
    console.print(f"\n[bold cyan]=== Evaluation Report: {metrics.scorer} (v{metrics.scorer_version}) ===[/bold cyan]")
    console.print(f"Candidate Set ID: [bold]{metrics.candidate_set_id}[/bold]")
    console.print(f"Coverage:         [bold]{metrics.labeled_candidates}/{metrics.total_candidates}[/bold] candidates labeled")
    is_reranker = (
        metrics.candidate_coverage_ratio is not None
        and metrics.candidate_coverage_ratio < 1.0
    )

    if is_reranker:
        console.print("\n[bold yellow]─── RETRIEVAL / SHORTLIST QUALITY ───[/bold yellow]")
        retrieval_table = Table(box=box.SIMPLE)
        retrieval_table.add_column("Retrieval Metric", style="cyan")
        retrieval_table.add_column("Value", style="bold")
        retrieval_table.add_row(
            "Shortlist Coverage",
            f"{metrics.scored_candidates}/{metrics.total_candidates} ({metrics.candidate_coverage_ratio:.1%})",
        )
        if metrics.perfect_candidate_recall_in_shortlist is not None:
            retrieval_table.add_row(
                "Shortlist Perfect Recall",
                f"{metrics.perfect_candidate_recall_in_shortlist:.1%}",
            )
        if metrics.publishable_candidate_recall_in_shortlist is not None:
            retrieval_table.add_row(
                "Shortlist Publishable Recall",
                f"{metrics.publishable_candidate_recall_in_shortlist:.1%}",
            )
        console.print(retrieval_table)
        console.print("[bold magenta]─── RERANKING QUALITY WITHIN SHORTLIST ───[/bold magenta]")
    else:
        console.print()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=22)
    for kv in metrics.k_values:
        table.add_column(f"@ {kv}", justify="right", width=12)

    # Precision@K
    table.add_row("Precision (rel>=3)", *[f"{metrics.precision_at_k.get(kv, 0.0):.1%}" for kv in metrics.k_values])
    # nDCG@K
    table.add_row("nDCG", *[f"{metrics.ndcg_at_k.get(kv, 0.0):.4f}" for kv in metrics.k_values])
    # MeanHumanScore@K
    table.add_row("Mean Human Score", *[f"{metrics.mean_human_score_at_k.get(kv, 0.0):.2f}/4" for kv in metrics.k_values])
    # PerfectRate@K (human_score >= 4)
    table.add_row("Perfect Rate (>=4)", *[f"{metrics.perfect_rate_at_k.get(kv, 0.0):.1%}" for kv in metrics.k_values])
    # BadRate@K (human_score <= 2)
    table.add_row("Bad Rate (<=2)", *[f"{metrics.bad_rate_at_k.get(kv, 0.0):.1%}" for kv in metrics.k_values])
    # PublishableRate@K
    table.add_row("Publishable Rate", *[f"{metrics.publishable_rate_at_k.get(kv, 0.0):.1%}" for kv in metrics.k_values])
    # HitRate@K
    hit_vals = []
    for kv in metrics.k_values:
        val = metrics.hit_rate_at_k.get(kv)
        hit_vals.append(f"{val:.0%}" if val is not None else "-")
    table.add_row("Hit Rate", *hit_vals)

    # Recall@K
    if metrics.recall_at_k is not None:
        table.add_row("Recall", *[f"{metrics.recall_at_k.get(kv, 0.0):.1%}" for kv in metrics.k_values])
    else:
        table.add_row("Recall", *["N/A" for _ in metrics.k_values])

    console.print(table)
    if metrics.recall_message:
        console.print(f"[dim]Note: {metrics.recall_message}[/dim]")
    console.print()


@app.command("compare-scorers")
def compare_scorers_command(
    eval_file: Path = typer.Argument(
        ...,
        help="Path to evaluation JSON",
        exists=True,
        file_okay=True,
        readable=True,
    ),
    scores_files: list[Path] = typer.Argument(
        ...,
        help="Two or more paths to model prediction JSON files",
    ),
    k: str = typer.Option(
        "5,10",
        "--k",
        help="Comma-separated K values for metrics (e.g. 5,10)",
    ),
    disagreements: Optional[Path] = typer.Option(
        None,
        "--disagreements",
        "-d",
        help="Optional path to output disagreement report JSON",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output raw JSON comparison",
    ),
) -> None:
    """Compare multiple highlight scorers side-by-side against the same human evaluation."""
    if len(scores_files) < 2:
        console.print("[bold red]Error:[/bold red] At least 2 score files are required for comparison.")
        raise typer.Exit(code=1)

    try:
        k_values = [int(x.strip()) for x in k.split(",") if x.strip()]
    except Exception:
        console.print(f"[bold red]Error:[/bold red] Invalid K values: '{k}'")
        raise typer.Exit(code=1)

    eval_data = load_json(eval_file)
    eval_doc = BlindEvaluationDocument.model_validate(eval_data)

    pred_docs = []
    for s_path in scores_files:
        if not s_path.is_file():
            console.print(f"[bold red]Error:[/bold red] File not found: {s_path}")
            raise typer.Exit(code=1)
        doc = ScorerPredictionDocument.model_validate(load_json(s_path))
        if doc.candidate_set_id != eval_doc.candidate_set_id:
            console.print(f"[bold red]Candidate Set ID Mismatch![/bold red]")
            console.print(f"Evaluation Candidate Set: {eval_doc.candidate_set_id}")
            console.print(f"Scores '{s_path.name}' Candidate Set: {doc.candidate_set_id}")
            raise typer.Exit(code=1)
        pred_docs.append(doc)

    all_metrics = [compute_evaluation_metrics(eval_doc, p_doc, k_values=k_values) for p_doc in pred_docs]

    # If disagreements output requested
    if disagreements is not None:
        report = extract_disagreements(
            eval_doc,
            pred_docs[0],
            pred_docs[1] if len(pred_docs) > 1 else None,
        )
        save_json(report, disagreements)
        console.print(f"\n[bold green]Disagreement report exported to:[/bold green] {disagreements}")
        console.print(f"  False positives:   {len(report.false_positives)}")
        console.print(f"  False negatives:   {len(report.false_negatives)}")
        console.print(f"  Scorer divergences: {len(report.scorer_divergences)}")

    if json_output:
        import json
        console.print(json.dumps([m.model_dump() for m in all_metrics], indent=2))
        return

    console.print(f"\n[bold cyan]=== Scorer Comparison Report ===[/bold cyan]")
    console.print(f"Candidate Set ID: [bold]{eval_doc.candidate_set_id}[/bold]")
    console.print(f"Coverage:         [bold]{eval_doc.labeled_candidates}/{eval_doc.total_candidates}[/bold] candidates labeled\n")

    has_reranker = any(
        (m.candidate_coverage_ratio is not None and m.candidate_coverage_ratio < 1.0)
        or (m.perfect_candidate_recall_in_shortlist is not None)
        for m in all_metrics
    )

    if has_reranker:
        console.print("[bold yellow]─── RETRIEVAL / SHORTLIST QUALITY ───[/bold yellow]")
        retrieval_table = Table(show_header=True, header_style="bold yellow", box=box.SIMPLE)
        retrieval_table.add_column("Retrieval Metric", style="dim", width=26)
        for m in all_metrics:
            retrieval_table.add_column(f"{m.scorer}\n({m.scorer_version})", justify="right", width=18)

        cov_row = [
            f"{m.scored_candidates or m.total_candidates}/{m.total_candidates} ({m.candidate_coverage_ratio or 1.0:.0%})"
            for m in all_metrics
        ]
        retrieval_table.add_row("Scored Pool / Coverage", *cov_row)

        perf_rec_row = [
            f"{m.perfect_candidate_recall_in_shortlist:.1%}"
            if m.perfect_candidate_recall_in_shortlist is not None
            else "100.0%"
            for m in all_metrics
        ]
        retrieval_table.add_row("Shortlist Perfect Recall", *perf_rec_row)

        pub_rec_row = [
            f"{m.publishable_candidate_recall_in_shortlist:.1%}"
            if m.publishable_candidate_recall_in_shortlist is not None
            else "100.0%"
            for m in all_metrics
        ]
        retrieval_table.add_row("Shortlist Publish Recall", *pub_rec_row)

        console.print(retrieval_table)
        console.print("\n[bold magenta]─── RERANKING / OVERALL RANKING QUALITY ───[/bold magenta]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=22)
    for m in all_metrics:
        table.add_column(f"{m.scorer}\n({m.scorer_version})", justify="right", width=18)

    for kv in k_values:
        p_row = [f"{m.precision_at_k.get(kv, 0.0):.1%}" for m in all_metrics]
        table.add_row(f"Precision @ {kv}", *p_row)

        ndcg_row = [f"{m.ndcg_at_k.get(kv, 0.0):.4f}" for m in all_metrics]
        table.add_row(f"nDCG @ {kv}", *ndcg_row)

        mhs_row = [f"{m.mean_human_score_at_k.get(kv, 0.0):.2f}/4" for m in all_metrics]
        table.add_row(f"Mean Score @ {kv}", *mhs_row)

        perf_row = [f"{m.perfect_rate_at_k.get(kv, 0.0):.1%}" for m in all_metrics]
        table.add_row(f"Perfect Rate @ {kv}", *perf_row)

        bad_row = [f"{m.bad_rate_at_k.get(kv, 0.0):.1%}" for m in all_metrics]
        table.add_row(f"Bad Rate @ {kv}", *bad_row)

        pub_row = [f"{m.publishable_rate_at_k.get(kv, 0.0):.1%}" for m in all_metrics]
        table.add_row(f"Publishable @ {kv}", *pub_row)

        rec_row = []
        for m in all_metrics:
            if m.recall_at_k is not None and kv in m.recall_at_k:
                rec_row.append(f"{m.recall_at_k[kv]:.1%}")
            else:
                rec_row.append("N/A")
        table.add_row(f"Recall @ {kv}", *rec_row)

    console.print(table)
    console.print()


@app.command("render")
def render_command(
    run_arg: str = typer.Argument(
        ...,
        help="Path to run directory or run directory name in runs/ (e.g. 'runs/20260907_test')",
    ),
    preset: str = typer.Option(
        "shorts",
        "--preset",
        "-p",
        help=f"Rendering preset to use: {', '.join(AVAILABLE_PRESETS.keys())}",
    ),
    top_k: int = typer.Option(
        3,
        "--top-k",
        "-k",
        help="Number of top highlights to render as vertical short-form videos",
    ),
    no_crop: bool = typer.Option(
        False,
        "--no-crop",
        help="Disable smart vertical crop (uses static 9:16 center crop)",
    ),
    no_subtitles: bool = typer.Option(
        False,
        "--no-subtitles",
        help="Disable ASS karaoke subtitles overlay",
    ),
    no_loudnorm: bool = typer.Option(
        False,
        "--no-loudnorm",
        help="Disable two-stage EBU R128 audio loudness normalization",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-rendering even if output short files already exist",
    ),
) -> None:
    """Render top highlights from an existing run into 9:16 publication-ready vertical short-form videos."""
    try:
        run_dir = _resolve_run_path(run_arg)
    except FileNotFoundError as err:
        console.print(f"[bold red]Error:[/bold red] {err}")
        raise typer.Exit(code=1)

    if preset not in AVAILABLE_PRESETS:
        console.print(f"[bold red]Error:[/bold red] Unknown preset '{preset}'. Available: {', '.join(AVAILABLE_PRESETS.keys())}")
        raise typer.Exit(code=1)

    console.print("\n[bold cyan]=== Starting Short-Form Rendering MVP ===[/bold cyan]")
    console.print(f"Run Directory:   [bold]{run_dir}[/bold]")
    console.print(f"Preset:          [bold]{preset}[/bold]")
    console.print(f"Top K:           [bold]{top_k}[/bold]")
    console.print(f"Smart Crop:      [bold]{'Disabled' if no_crop else 'Enabled (Subject Tracking)'}[/bold]")
    console.print(f"Karaoke Subs:    [bold]{'Disabled' if no_subtitles else 'Enabled (ASS)'}[/bold]")
    console.print(f"Audio Loudnorm:  [bold]{'Disabled' if no_loudnorm else 'Enabled (EBU R128 Two-Stage)'}[/bold]\n")

    try:
        manifest = render_highlights_for_run(
            run_dir=run_dir,
            preset_name=preset,
            top_k=top_k,
            enable_smart_crop=not no_crop,
            enable_subtitles=not no_subtitles,
            enable_audio_normalization=not no_loudnorm,
            force=force,
        )
    except Exception as exc:
        console.print(f"[bold red]Rendering Failed:[/bold red] {exc}")
        raise typer.Exit(code=1)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", width=6, justify="center")
    table.add_column("Candidate ID", style="dim", width=14)
    table.add_column("Refined Window", width=18)
    table.add_column("Dur", width=8, justify="right")
    table.add_column("Resolution", width=11)
    table.add_column("Encoder", width=12)
    table.add_column("Validation", width=12)
    table.add_column("Output File")

    for item in manifest.shorts:
        ref = f"{_format_timestamp(item.refined_start)} -> {_format_timestamp(item.refined_end)}"
        dur = f"{item.duration:.1f}s"
        val_status = "[green]PASSED[/green]" if item.validation and item.validation.passed else "[red]FAILED[/red]"
        table.add_row(
            f"#{item.rank}",
            item.candidate_id,
            ref,
            dur,
            item.resolution,
            item.encoder,
            val_status,
            item.file,
        )

    console.print(table)
    console.print(f"\n[bold green]✓ Successfully rendered {len(manifest.shorts)} vertical video(s).[/bold green]")
    console.print(f"Manifest: [bold]{run_dir / 'render_manifest.json'}[/bold]\n")


@app.command("render-highlight")
def render_highlight_command(
    run_arg: str = typer.Argument(
        ...,
        help="Path to run directory or run directory name in runs/",
    ),
    rank: Optional[int] = typer.Option(
        None,
        "--rank",
        "-r",
        help="Rank of the highlight to render (e.g. 1)",
    ),
    candidate_id: Optional[str] = typer.Option(
        None,
        "--candidate-id",
        "-c",
        help="Candidate ID of the highlight to render (e.g. 'cand_0003')",
    ),
    preset: str = typer.Option(
        "shorts",
        "--preset",
        "-p",
        help=f"Rendering preset to use: {', '.join(AVAILABLE_PRESETS.keys())}",
    ),
    no_crop: bool = typer.Option(
        False,
        "--no-crop",
        help="Disable smart crop face/subject tracking (uses static 9:16 center crop)",
    ),
    no_subtitles: bool = typer.Option(
        False,
        "--no-subtitles",
        help="Disable ASS karaoke subtitles overlay",
    ),
    no_loudnorm: bool = typer.Option(
        False,
        "--no-loudnorm",
        help="Disable two-stage EBU R128 audio loudness normalization",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-rendering even if output short file already exists",
    ),
) -> None:
    """Render a specific highlight from a run into a 9:16 publication-ready vertical video."""
    if rank is None and candidate_id is None:
        console.print("[bold red]Error:[/bold red] Either --rank or --candidate-id must be specified.")
        raise typer.Exit(code=1)

    try:
        run_dir = _resolve_run_path(run_arg)
    except FileNotFoundError as err:
        console.print(f"[bold red]Error:[/bold red] {err}")
        raise typer.Exit(code=1)

    if preset not in AVAILABLE_PRESETS:
        console.print(f"[bold red]Error:[/bold red] Unknown preset '{preset}'. Available: {', '.join(AVAILABLE_PRESETS.keys())}")
        raise typer.Exit(code=1)

    highlights_file = run_dir / "highlights.json"
    manifest_file = run_dir / "manifest.json"
    transcript_file = run_dir / "transcript.json"

    if not highlights_file.is_file():
        console.print(f"[bold red]Error:[/bold red] {highlights_file} not found.")
        raise typer.Exit(code=1)
    if not manifest_file.is_file():
        console.print(f"[bold red]Error:[/bold red] {manifest_file} not found.")
        raise typer.Exit(code=1)
    if not transcript_file.is_file():
        console.print(f"[bold red]Error:[/bold red] {transcript_file} not found.")
        raise typer.Exit(code=1)

    man = load_json(manifest_file)
    source_video = Path(man["source"])
    if not source_video.is_file():
        console.print(f"[bold red]Error:[/bold red] Source video does not exist: {source_video}")
        raise typer.Exit(code=1)

    transcript = Transcript.model_validate(load_json(transcript_file))
    highlights_data = load_json(highlights_file)
    highlights = [Highlight.model_validate(h) for h in highlights_data]

    target: Optional[Highlight] = None
    for h in highlights:
        if rank is not None and h.rank == rank:
            target = h
            break
        if candidate_id is not None and h.candidate_id == candidate_id:
            target = h
            break

    if target is None:
        identifier = f"rank={rank}" if rank is not None else f"candidate_id={candidate_id}"
        console.print(f"[bold red]Error:[/bold red] Highlight with {identifier} not found in {highlights_file}.")
        raise typer.Exit(code=1)

    source_fp_id = man.get("source_fingerprint", {}).get("fingerprint_id", "unknown_fp")
    video_duration = float(man.get("source_fingerprint", {}).get("duration_seconds", 0.0))
    if video_duration <= 0.0:
        info = probe_media(source_video)
        video_duration = info.duration or 300.0

    selected_preset = get_preset(preset)

    console.print(f"\n[bold cyan]=== Rendering Highlight #{target.rank} ({target.candidate_id}) ===[/bold cyan]")
    console.print(f"Source Video:    [bold]{source_video}[/bold]")
    console.print(f"Original Window: [bold]{_format_timestamp(target.start)} -> {_format_timestamp(target.end)}[/bold] ({target.duration:.1f}s)")
    console.print(f"Preset:          [bold]{preset}[/bold]\n")

    try:
        item = render_single_short(
            source_video=source_video,
            highlight=target,
            transcript=transcript,
            source_fingerprint_id=source_fp_id,
            video_duration=video_duration,
            run_dir=run_dir,
            preset=selected_preset,
            enable_smart_crop=not no_crop,
            enable_subtitles=not no_subtitles,
            enable_audio_normalization=not no_loudnorm,
            force=force,
        )
    except Exception as exc:
        console.print(f"[bold red]Rendering Failed:[/bold red] {exc}")
        raise typer.Exit(code=1)

    val_str = "[bold green]PASSED[/bold green]" if item.validation and item.validation.passed else "[bold red]FAILED[/bold red]"
    console.print(Panel(
        f"[bold]Output File:[/bold] {item.file}\n"
        f"[bold]Resolution:[/bold]  {item.resolution}\n"
        f"[bold]Refined Window:[/bold] {_format_timestamp(item.refined_start)} -> {_format_timestamp(item.refined_end)} ({item.duration:.1f}s)\n"
        f"[bold]Reason:[/bold]      {item.refinement_reason}\n"
        f"[bold]Encoder:[/bold]     {item.encoder}\n"
        f"[bold]Validation:[/bold]  {val_str}\n"
        f"[bold]Crop Mode:[/bold]   {item.crop_mode}\n"
        f"[bold]Subtitles:[/bold]   {item.subtitle_style}\n"
        f"[bold]Audio Norm:[/bold]  {'Yes' if item.audio_normalized else 'No'}",
        title=f"Rendered Short #{item.rank}",
        border_style="green" if (item.validation and item.validation.passed) else "red",
    ))


@app.command("doctor")
def doctor_command() -> None:
    """Run diagnostics to inspect environment, FFmpeg, CUDA, and faster-whisper availability."""
    console.print("\n[bold cyan]=== freecher-worker System Diagnostics ===[/bold cyan]\n")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="dim", width=24)
    table.add_column("Status", width=14)
    table.add_column("Details")

    # 1. Python Environment
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    table.add_row("Python", "[green]OK[/green]", f"{py_ver} ({sys.executable})")

    # 2. FFmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            res = subprocess.run(
                ["ffmpeg", "-nostdin", "-version"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5.0,
                check=False,
            )
            first_line = res.stdout.split("\n")[0] if res.stdout else "Available"
            table.add_row("FFmpeg", "[green]OK[/green]", f"{first_line} ({ffmpeg_path})")
        except Exception as exc:
            table.add_row("FFmpeg", "[yellow]WARN[/yellow]", f"Found at {ffmpeg_path} but error checking version: {exc}")
    else:
        table.add_row("FFmpeg", "[red]MISSING[/red]", "ffmpeg not found in PATH")

    # 3. ffprobe
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        table.add_row("ffprobe", "[green]OK[/green]", f"Available at {ffprobe_path}")
    else:
        table.add_row("ffprobe", "[red]MISSING[/red]", "ffprobe not found in PATH")

    # 4. NVENC hardware encoding
    nvenc_ok = is_nvenc_available()
    if nvenc_ok:
        table.add_row("FFmpeg NVENC", "[green]OK[/green]", "h264_nvenc hardware encoder available and functional")
    else:
        table.add_row("FFmpeg NVENC", "[yellow]NOT AVAILABLE[/yellow]", "h264_nvenc unavailable, worker will use libx264 fallback")

    # 4b. FFmpeg libass subtitle burning
    from freecher_worker.rendering.renderer import is_ffmpeg_filter_supported
    ass_ok = is_ffmpeg_filter_supported("ass") or is_ffmpeg_filter_supported("subtitles")
    if ass_ok:
        table.add_row("FFmpeg libass", "[green]OK[/green]", "Subtitle filtering (ass/subtitles) supported for burning subtitles")
    else:
        table.add_row("FFmpeg libass", "[yellow]NOT AVAILABLE[/yellow]", "FFmpeg build lacks libass; ASS files saved to disk, burn-in skipped")


    # 5. faster-whisper
    try:
        import faster_whisper
        fw_ver = getattr(faster_whisper, "__version__", "installed")
        table.add_row("faster-whisper", "[green]OK[/green]", f"Version {fw_ver}")
    except ImportError:
        table.add_row("faster-whisper", "[red]MISSING[/red]", "Install via 'pip install faster-whisper'")

    # 6. CTranslate2 & CUDA
    try:
        import ctranslate2
        ct2_ver = getattr(ctranslate2, "__version__", "installed")
        try:
            cuda_count = ctranslate2.get_cuda_device_count()
        except Exception:
            cuda_count = 0
        try:
            cpu_computes = ctranslate2.get_supported_compute_types("cpu")
        except Exception:
            cpu_computes = []

        if cuda_count > 0:
            try:
                cuda_computes = ctranslate2.get_supported_compute_types("cuda")
            except Exception:
                cuda_computes = []
            table.add_row(
                "CTranslate2",
                "[green]OK[/green]",
                f"Version {ct2_ver} | GPU Count: {cuda_count} | CUDA compute types: {', '.join(cuda_computes)}",
            )
            table.add_row("GPU / CUDA", "[green]AVAILABLE[/green]", f"{cuda_count} NVIDIA device(s) detected by CTranslate2")
        else:
            table.add_row(
                "CTranslate2",
                "[yellow]CPU ONLY[/yellow]",
                f"Version {ct2_ver} | No CUDA GPU detected (CPU compute types: {', '.join(cpu_computes)})",
            )
            table.add_row("GPU / CUDA", "[yellow]NOT DETECTED[/yellow]", "No CUDA devices reported. Source scripts/cuda_env.sh if on WSL/Linux.")
    except ImportError:
        table.add_row("CTranslate2", "[red]MISSING[/red]", "Install via 'pip install ctranslate2'")

    console.print(table)
    console.print()


def main() -> None:
    app()


if __name__ == "__main__":
    main()

