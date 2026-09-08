# freecher-worker

Local worker for automated highlight discovery in long videos and short clip generation (Phase 1.1 MVP of an OpusClip alternative).

---

## Features (Phase 1.1)

- **End-to-End Pipeline**: From input MP4 to extracted audio, Whisper transcription, temporal candidate window generation, highlight scoring, deduplication, ranking, and final cut MP4 clips.
- **Configuration-Aware Cache Invalidation**: Caches artifacts deterministically. Automatically invalidates transcription cache if ASR model, device, compute type, language, beam size, or VAD settings change. Automatically invalidates candidate cache if window parameters or transcript change.
- **Source Fingerprinting**: Media files are tracked by resolved path, file size, modification nanoseconds, duration, and a lightweight chunk-based content hash (fast even on multi-gigabyte videos).
- **Observable Cache Logging**: Stage logs clearly state `cache HIT`, `cache MISS` with exact reason, and differentiate between `reused cached clip` and `created clip`.
- **Reproducible Manifest**: Expanded `manifest.json` recording environment, full ASR and candidate configs, scorer version, LLM fallback tracking, per-stage timings, and statistics.
- **Analysis-Only Mode (`--analysis-only`)**: Runs full discovery, scoring, and ranking without invoking FFmpeg video clip re-encoding.
- **Inspection CLI (`inspect`)**: Terminal inspection command displaying top highlights with human-readable timestamps (`MM:SS`), duration, text preview, reasons, and subscores.
- **Evaluation Export CLI (`export-eval`)**: Exports candidates to `evaluation.json` with empty human rating fields (`human_label: null`, `human_score: null` [0-4], `human_notes: null`) for scientific benchmarking.
- **Standalone Heuristic Scorer**: Evaluates hooks, questions, emotions, concrete facts, pacing, and repetition penalties with zero external API dependencies.
- **Observable LLM Scorer**: Pluggable OpenAI-compatible LLM scoring with explicit tracking of fallback reason if the API fails.
- **Temporal Candidate Windowing**: Segment aggregation bounded strictly by natural phrase boundaries (~30-90s, target 60s, ~15s overlap).
- **Temporal Deduplication**: Non-Maximum Suppression (NMS) suppressing overlapping candidates (>60% temporal overlap).
- **Hardware-Accelerated Clipping**: Auto-detects `h264_nvenc` with automatic fallback to `libx264`. Includes contextual padding (±2.0s).
- **Environment Doctor**: Built-in `python -m freecher_worker doctor` command for environment diagnostics.

---

## Directory Structure

```
freecher-worker/
    freecher_worker/
        __init__.py
        __main__.py
        cli.py
        config.py

        media/
            __init__.py
            probe.py
            fingerprint.py
            audio.py
            clipper.py

        transcription/
            __init__.py
            whisper.py
            models.py

        highlights/
            __init__.py
            segmenter.py
            ranker.py
            dedup.py
            models.py

        scoring/
            __init__.py
            base.py
            heuristic.py
            llm.py

        pipeline/
            __init__.py
            processor.py

        utils/
            __init__.py
            logging.py
            json_io.py

    scripts/
        cuda_env.sh

    tests/
        test_config.py
        test_media.py
        test_segmenter.py
        test_dedup.py
        test_ranker.py
        test_scoring.py
        test_json_io.py
        test_pipeline.py
        test_cache_invalidation.py
        test_analysis_only.py
        test_inspect_and_eval.py
        test_llm_observability.py

    benchmark_whisper.py
    requirements.txt
    .env.example
    .gitignore
    README.md
```

---

## Installation & Setup

### 1. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure CUDA environment (Linux / WSL2)

On WSL2 / Ubuntu with pip-installed NVIDIA libraries, load the required cuBLAS/cuDNN dynamic libraries into `LD_LIBRARY_PATH`:

```bash
source scripts/cuda_env.sh
```

### 3. Verify environment

Run system diagnostics to check Python, FFmpeg, ffprobe, CUDA, and faster-whisper:

```bash
python -m freecher_worker doctor
```

### 4. Updating an existing environment

When pulling updates or running workers on an existing setup (Linux / WSL2):

```bash
cd ~/freecher-worker
git pull origin master
source .venv/bin/activate
```

---

## CLI Usage

### Process a Video

```bash
# Full process with MP4 clipping
python -m freecher_worker process test.mp4 \
  --language ru \
  --model small \
  --compute-type int8_float16 \
  --top-k 5

# Analysis-only (skip video rendering for rapid iteration & evaluation)
python -m freecher_worker process test.mp4 \
  --language ru \
  --analysis-only
```

### Inspect a Run

```bash
python -m freecher_worker inspect runs/20260907_082719_test
```

### Export Evaluation Dataset

```bash
python -m freecher_worker export-eval runs/20260907_082719_test
```

Generates `evaluation.json` with candidate highlights and blank human feedback fields:

```json
[
  {
    "candidate_id": "cand_002",
    "rank": 1,
    "start": 45.44,
    "end": 71.96,
    "duration": 26.52,
    "text": "...",
    "score": 87.9,
    "subscores": {
      "hook_score": 100.0,
      "standalone_score": 100.0,
      "emotion_score": 64.0,
      "information_score": 88.0,
      "shareability_score": 84.4
    },
    "reason": "...",
    "human_label": null,
    "human_score": null,
    "human_notes": null
  }
]
```

### CLI Parameters for `process`

- `video_path`: Path to input video (required).
- `-o, --output`: Base output directory (default: `runs/`).
- `-l, --language`: Language code (e.g. `ru`, `en`). Defaults to auto-detection.
- `-m, --model`: Whisper model name (`small`, `medium`, `turbo`, etc.). Default: `small`.
- `-d, --device`: Inference device (`cuda` or `cpu`). Default: `cuda`.
- `-c, --compute-type`: CTranslate2 compute type (`int8_float16`, `int8`, `float16`). Default: `int8_float16`.
- `-k, --top-k`: Number of top highlight clips to generate. Default: `5`.
- `-s, --scorer`: Scorer to use (`heuristic` or `llm`). Default: `heuristic`.
- `-f, --force`: Ignore cached stage artifacts and force recalculation.
- `--analysis-only`: Skip video clip cutting, only produce transcript, candidates, scores, and manifest.
- `--run-id`: Resume or target a specific run directory name.

---

## Running Tests

All unit tests run without requiring a GPU:

```bash
pytest -q
```
