"""Command-line interface for arny-worker using Typer."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from arny_worker.config import Settings, get_settings
from arny_worker.highlights.models import CandidateDocument, CandidateWindow, Highlight
from arny_worker.media.clipper import is_nvenc_available
from arny_worker.pipeline.processor import Manifest, run_pipeline
from arny_worker.utils.json_io import load_json, save_json

app = typer.Typer(
    name="arny-worker",
    help="arny-worker: Automated video highlight extraction and clipping worker.",
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


@app.command("export-eval")
def export_eval_command(
    run_dir: Path = typer.Argument(
        ...,
        help="Path or name of run directory to export evaluation set from",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination path for evaluation.json (default: <run_dir>/evaluation.json)",
    ),
) -> None:
    """Export candidate highlights to evaluation.json with blank fields for manual human rating."""
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
    if isinstance(cand_data, dict) and "candidates" in cand_data:
        raw_candidates = cand_data["candidates"]
    else:
        raw_candidates = cand_data

    # Map rank and scores from highlights or manifest
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

    # Sort: ranked items first by rank, then unranked by score descending or start time
    eval_items.sort(key=lambda x: (0 if x["rank"] is not None else 1, x["rank"] if x["rank"] is not None else - (x["score"] or 0)))

    target_path = output or (resolved_dir / "evaluation.json")
    save_json(eval_items, target_path)

    console.print(f"\n[bold green]Exported {len(eval_items)} evaluation candidates to:[/bold green] {target_path}")
    console.print("Fields for manual human evaluation ready: human_label, human_score (0-4), human_notes\n")


@app.command("doctor")
def doctor_command() -> None:
    """Run diagnostics to inspect environment, FFmpeg, CUDA, and faster-whisper availability."""
    console.print("\n[bold cyan]=== arny-worker System Diagnostics ===[/bold cyan]\n")
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
