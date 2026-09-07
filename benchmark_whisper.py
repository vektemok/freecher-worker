#!/usr/bin/env python3
"""benchmark_whisper.py - Benchmarking script for faster-whisper on GPU / CPU.

Measures transcription duration, realtime factor (RTF), and output segment count across
different Whisper models (small, medium, turbo) and compute types (int8, int8_float16, float16).

Usage:
    python benchmark_whisper.py [video_or_audio_path]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def extract_audio_for_benchmark(input_path: Path) -> Path:
    """Extract 16kHz mono WAV if input is video, or return input path if already WAV."""
    if input_path.suffix.lower() == ".wav":
        return input_path

    temp_wav = Path(tempfile.gettempdir()) / f"benchmark_{input_path.stem}.wav"
    print(f"Extracting benchmark audio to: {temp_wav}")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(temp_wav),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_wav


def run_benchmark(
    audio_path: Path,
    configs: list[tuple[str, str, str]],
    language: str = "ru",
) -> None:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("ERROR: faster-whisper is not installed. Run: pip install faster-whisper")
        sys.exit(1)

    print("=" * 70)
    print(f"BENCHMARKING FASTER-WHISPER: {audio_path.name}")
    print("=" * 70)

    results = []

    for model_name, device, compute_type in configs:
        print(f"\n---> Testing model={model_name}, device={device}, compute_type={compute_type}...")
        try:
            load_start = time.perf_counter()
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            load_time = time.perf_counter() - load_start
            print(f"     Model loaded in {load_time:.2f}s")

            transcribe_start = time.perf_counter()
            segments, info = model.transcribe(str(audio_path), language=language, vad_filter=True, beam_size=5)
            # Fully materialize lazy iterator
            segments_list = list(segments)
            elapsed = time.perf_counter() - transcribe_start

            duration = getattr(info, "duration", 0.0)
            if duration <= 0.0 and segments_list:
                duration = segments_list[-1].end

            rtf = duration / elapsed if elapsed > 0 else 0.0
            print(f"     Audio duration: {duration:.1f}s")
            print(f"     Transcription time: {elapsed:.2f}s (~{rtf:.2f}x realtime)")
            print(f"     Segments detected: {len(segments_list)}")

            results.append({
                "model": model_name,
                "compute_type": compute_type,
                "device": device,
                "duration": duration,
                "elapsed": elapsed,
                "rtf": rtf,
                "segments": len(segments_list),
            })
        except Exception as exc:
            print(f"     FAILED: {exc}")
            results.append({
                "model": model_name,
                "compute_type": compute_type,
                "device": device,
                "error": str(exc),
            })

    print("\n" + "=" * 70)
    print(f"{'Model':<10} {'Compute Type':<15} {'Device':<8} {'Time (s)':<10} {'Speedup':<12} {'Segments':<10}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<10} {r['compute_type']:<15} {r['device']:<8} {'FAILED':<10} {'-':<12} {'-'}")
        else:
            print(
                f"{r['model']:<10} {r['compute_type']:<15} {r['device']:<8} "
                f"{r['elapsed']:<10.1f} {f'{r[\"rtf\"]:.2f}x':<12} {r['segments']:<10}"
            )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark faster-whisper on GPU/CPU")
    parser.add_argument("input", nargs="?", default="test.mp4", help="Video or audio file to benchmark")
    parser.add_argument("--device", default="cuda", help="Inference device (cuda or cpu)")
    parser.add_argument("--language", default="ru", help="Language code (default: ru)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    wav_path = extract_audio_for_benchmark(input_path)

    configs = [
        ("small", args.device, "int8"),
        ("small", args.device, "int8_float16"),
        ("medium", args.device, "int8"),
        ("turbo", args.device, "int8"),
    ]

    run_benchmark(wav_path, configs, language=args.language)


if __name__ == "__main__":
    main()
