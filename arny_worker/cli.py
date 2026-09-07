"""Command-line interface for arny-worker using Typer."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from arny_worker.config import Settings, get_settings
from arny_worker.media.clipper import is_nvenc_available
from arny_worker.pipeline.processor import run_pipeline

app = typer.Typer(
    name="arny-worker",
    help="arny-worker: Automated video highlight extraction and clipping worker.",
    add_completion=False,
)
console = Console()


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
        help="Base output directory for runs (default from config: 'runs')",
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
) -> None:
    """Process video file: transcribe, discover highlight moments, and cut clips."""
    base_settings = get_settings()

    # Create config overrides from CLI options
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
        )

        console.print(f"\n[bold green]Processing completed successfully in {manifest.processing_time:.1f}s![/bold green]")
        if manifest.highlights:
            table = Table(title="Top Highlights")
            table.add_column("Rank", justify="center", style="cyan")
            table.add_column("Time Range", justify="center", style="magenta")
            table.add_column("Duration", justify="center")
            table.add_column("Score", justify="center", style="bold yellow")
            table.add_column("Clip File", style="green")
            table.add_column("Reason")

            for hl in manifest.highlights:
                table.add_row(
                    str(hl.rank),
                    f"{hl.start:.1f}s - {hl.end:.1f}s",
                    f"{hl.duration:.1f}s",
                    f"{hl.score:.1f}",
                    hl.file,
                    hl.reason,
                )
            console.print(table)
        else:
            console.print("[yellow]No highlights were found in the video.[/yellow]")

    except Exception as exc:
        console.print(f"\n[bold red]Error during processing:[/bold red] {exc}")
        sys.exit(1)


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
