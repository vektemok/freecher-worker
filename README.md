# arny-worker

Local worker for automated highlight discovery in long videos and short clip generation (Phase 1 MVP of an OpusClip alternative).

---

## Features (Phase 1 MVP)

- **End-to-End Pipeline**: From input MP4 to extracted audio, Whisper transcription, candidate window generation, highlight scoring, deduplication, ranking, and final cut MP4 clips.
- **Hardware-Accelerated ASR**: Configurable `faster-whisper` on GPU (`cuda`) or `cpu` with custom compute types (`int8_float16`, `int8`, `float16`).
- **Configurable Models**: Seamlessly switch between `small`, `medium`, `turbo`, and other Whisper models via environment variables or CLI flags without changing code.
- **Standalone Heuristic Scorer**: Evaluates hooks, questions, emotions, concrete facts, pacing, and lexical density with zero external API dependencies.
- **Optional OpenAI-Compatible LLM Scorer**: Pluggable LLM scoring with graceful fallback to heuristic scoring.
- **Smart Sliding Window**: Segment aggregation bounded strictly by natural phrase boundaries (~30-90s, target 60s, ~15s overlap).
- **Temporal Deduplication**: Non-Maximum Suppression (NMS) suppressing overlapping candidates (>60% temporal overlap) to guarantee varied highlights.
- **Hardware-Accelerated Clipping**: Auto-detects `h264_nvenc` and transparently falls back to `libx264` if unavailable. Includes contextual padding (±2.0s).
- **Stage Caching**: Reuses `media.json`, `audio.wav`, `transcript.json`, and `candidates.json` across reruns to avoid re-transcribing long videos.
- **Environment Doctor**: Built-in `python -m arny_worker doctor` command for environment diagnostics.

---

## Directory Structure

```
arny-worker/
    arny_worker/
        __init__.py
        __main__.py
        cli.py
        config.py

        media/
            __init__.py
            probe.py
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
python -m arny_worker doctor
```

---

## Usage

### Process a Video

```bash
python -m arny_worker process test.mp4 \
  --language ru \
  --model small \
  --compute-type int8_float16 \
  --top-k 5
```

### CLI Parameters

- `video_path`: Path to input video (required).
- `-o, --output`: Base output directory (default: `runs/`).
- `-l, --language`: Language code (e.g. `ru`, `en`). Defaults to auto-detection.
- `-m, --model`: Whisper model name (`small`, `medium`, `turbo`, etc.). Default: `small`.
- `-d, --device`: Inference device (`cuda` or `cpu`). Default: `cuda`.
- `-c, --compute-type`: CTranslate2 compute type (`int8_float16`, `int8`, `float16`). Default: `int8_float16`.
- `-k, --top-k`: Number of top highlight clips to generate. Default: `5`.
- `-s, --scorer`: Scorer to use (`heuristic` or `llm`). Default: `heuristic`.
- `-f, --force`: Ignore cached stage artifacts and force recalculation.
- `--run-id`: Resume or target a specific run directory name.

---

## Output Artifacts

Each processing run creates a structured directory in `runs/<run_id>/` (e.g., `runs/20260907_021500_test/`):

```
runs/20260907_021500_test/
    media.json          # Container and stream metadata
    audio.wav           # Extracted 16kHz mono PCM WAV
    transcript.json     # Materialized Whisper transcript segments
    candidates.json     # Generated candidate windows
    highlights.json     # Ranked top-K highlights
    manifest.json       # Final run manifest and clip metadata
    clips/
        clip_01.mp4     # Generated highlight video clips
        clip_02.mp4
        ...
    logs/
        worker.log      # Complete stage timing log
```

### Sample `manifest.json`:

```json
{
  "source": "/path/to/test.mp4",
  "duration": 2038.8,
  "processing_time": 231.7,
  "created_at": "2026-09-07T02:18:51",
  "asr": {
    "model": "small",
    "compute_type": "int8_float16",
    "device": "cuda"
  },
  "highlights": [
    {
      "rank": 1,
      "start": 120.5,
      "end": 181.2,
      "duration": 60.7,
      "score": 88.5,
      "reason": "Heuristic (88.5/100): opening question hook; emotional keywords (ого, жесть); concrete numbers / data points",
      "file": "clips/clip_01.mp4",
      "candidate_id": "cand_003",
      "padded_start": 118.5,
      "padded_end": 183.2
    }
  ]
}
```

---

## Configuration via Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Available variables:
- `ARNY_ASR_MODEL`: `small`, `medium`, `turbo`
- `ARNY_ASR_DEVICE`: `cuda`, `cpu`
- `ARNY_ASR_COMPUTE_TYPE`: `int8_float16`, `int8`, `float16`
- `ARNY_HIGHLIGHT_MIN_SECONDS`: `30`
- `ARNY_HIGHLIGHT_TARGET_SECONDS`: `60`
- `ARNY_HIGHLIGHT_MAX_SECONDS`: `90`
- `ARNY_HIGHLIGHT_TOP_K`: `5`
- `ARNY_SCORER`: `heuristic` or `llm`
- `ARNY_LLM_BASE_URL`: OpenAI-compatible endpoint URL
- `ARNY_LLM_API_KEY`: API Key
- `ARNY_LLM_MODEL`: e.g. `gpt-4o-mini`

---

## Running Tests

All unit tests run without requiring a GPU:

```bash
pytest -q
```
