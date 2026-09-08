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

## Vertical Short Production (9:16 only)

Freecher produces exactly one output format: **9:16, 1080x1920, MP4, H.264 + AAC**. There is no
landscape, square, 4:5 or generic aspect-ratio support, and no CLI flag to request one.

Two production stages run *after* ranking. Neither reads nor modifies `heuristic_v1`,
`highlight_v2_1`, `multimodal_v1_1` or any scoring weight.

```
Video -> Whisper -> candidates -> heuristic_v1 + highlight_v2_1 -> shortlist
      -> multimodal_v1_1 / Luna -> ranked candidate
      -> [1] Dynamic Subclip Refinement
      -> [2] Smart 9:16 Reframing
      -> 1080x1920 MP4 + metadata JSON
```

### 1. Dynamic Subclip Refinement

A ~60 s candidate is an *analysis* window, not the published short. Inside it, the strongest
self-contained fragment is selected by `subclip_v1_formula_v1`, scoring hook strength, payoff
coverage, mean activity, phrase completeness, setup preservation, a soft duration prior and the
multimodal advisory region, minus dull lead-in and dead-air-tail penalties.

Boundaries snap to Whisper phrase boundaries, so a short never opens or closes mid-sentence.
Durations are **never quantized**: 11.4 s, 18.7 s, 24.2 s, 31.6 s and 42.0 s are all normal.

Configurable via `FREECHER_SUBCLIP_*` (defaults: min 8 s, target 15-30 s, max 45 s,
`duration_mode=auto`).

Activity signals are taken from the cached multimodal activity profile when the run has one,
otherwise from the run's `audio.wav`, otherwise from the transcript alone.

### 2. Timestamp Semantics

Absolute source timestamps and candidate-relative offsets are kept strictly apart:

| Field | Meaning |
| --- | --- |
| `source_start_sec` / `source_end_sec` | absolute, measured from the start of the source video |
| `short_start_offset_sec` / `short_end_offset_sec` | relative to the candidate start, `0 <= offset <= candidate_duration` |
| `short_source_start_sec` / `short_source_end_sec` | absolute timestamps of the produced short |

All conversion goes through `CandidateTimeframe`, which validates every offset. An incoming
`best_observed_region` is classified before use: plausible only as offsets is accepted, plausible
only as absolute timestamps is converted, and anything plausible as **both** is rejected as
ambiguous (refinement never runs on ambiguous timestamps; the advisory is simply dropped).

### 3. Smart 9:16 Reframing

```
semantic info / Luna -> face/person detection -> active subject selection
    -> tracking -> crop trajectory -> smoothing -> FFmpeg render 1080x1920
```

All local — no per-frame LLM calls. Frames are sampled at `reframe_analysis_fps` (default 5),
downscaled for detection, and detections are associated into persistent tracks by IoU and centre
distance. The active subject is chosen from a weighted blend of a mouth-motion speaking proxy,
box area, centrality and track persistence, with hysteresis (`switch_margin`, `switch_hold_sec`,
`min_switch_interval_sec`) so the crop never ping-pongs between two faces. Two faces that fit
inside the vertical window are framed together.

Safe framing keeps configurable padding around the subject, places the face near the upper third
and preserves headroom so eyes and the top of the head are not cut. The raw target is then
stabilized with a dead-zone, proportional tracking, and velocity/acceleration clamps; a detected
scene cut is allowed to re-anchor instantly.

Fallback ladder, so a valid 9:16 MP4 is always produced: last stable crop -> dominant visual
region (column gradient energy) -> static center crop.

### Ranking source

`render-shorts` renders from the **final** ranking in the run. The multimodal reranker is the last
stage of the ranking pipeline, so `--scorer auto` (the default) prefers it:

```
multimodal_v1_1 > multimodal_v1 > highlight_v2_1 > highlight_v2 > heuristic_v1 > highlights.json
```

Note that `highlights.json` holds the *heuristic* pipeline's top-K. Preferring it silently
rendered the wrong candidates whenever a multimodal pass had reordered them.

Pick a ranking explicitly with `--scorer`:

```bash
python -m freecher_worker render-shorts RUN --scorer multimodal_v1_1 --top 5
```

The chosen source, its file, its model and the selected candidates with their model ranks and
scores are printed before anything is rendered, and recorded in the manifest as
`ranking_source` / `ranking_origin` / `ranking_model`.

Candidate time windows always come from `candidates.json`, the frozen candidate set, so a
candidate ranked only by the multimodal pass can still be rendered.

### Subject detection

The production baseline is **YuNet** (OpenCV Zoo, ~230 KB ONNX): CPU-fast and far more reliable
than Haar cascades, including on the small facecam-sized faces that Haar misses entirely.

Weights resolve in this order, and every download is SHA-256 verified:

1. `FREECHER_REFRAME_FACE_MODEL_PATH` (honoured strictly - never silently substituted)
2. `$FREECHER_MODELS_DIR` or `~/.cache/freecher-worker/models/`
3. `<repo>/models/` or `./models/`
4. download into (2), unless `FREECHER_REFRAME_ALLOW_MODEL_DOWNLOAD=false`

To run fully offline, pre-place `face_detection_yunet_2023mar.onnx` in the models directory.

`--scorer`-style selection also exists for detectors via `FREECHER_REFRAME_DETECTOR`:
`auto` (YuNet, then legacy), `yunet`, `haar`, `center`.

Haar/HOG remain only as a legacy fallback. They are **not** a production baseline, and on
`opencv-python-headless` 5.x neither `CascadeClassifier` nor `HOGDescriptor` exists at all - which
previously made detection silently return nothing on every frame. A detector that cannot work now
reports `detector_operational: false` and logs an error instead of looking like an empty scene.

### Tracking vs render fallbacks

These are independent and are reported separately:

| Field | Meaning |
| --- | --- |
| `render_fallback_used` | the dynamic crop render failed, so a static crop was rendered instead |
| `render_failure_reason` | why, if it did |
| `tracking_mode` | `subject` / `mixed` / `dominant_region` / `center` |
| `tracking_fallback_rate` | fraction of analyzed frames framed by a tracking fallback |

A clip can legitimately be framed entirely from the dominant-region fallback (no people on screen)
and still be rendered perfectly by the dynamic crop driver: `render_fallback_used=false` with
`tracking_fallback_rate=1.0`.

Per-short detection diagnostics: `analyzed_frames`, `frames_with_face`, `frames_with_person`,
`detection_coverage`, `tracking_coverage`, `track_count`, `active_subject_switches`,
`dominant_fallback_rate`, `center_fallback_rate`.

Crop motion is reported twice: `peak_crop_velocity` includes scene-cut re-anchors, which are
deliberately allowed to jump, while `peak_velocity_non_scene_cut` is the number that must respect
the configured velocity limit.

### Dynamic crop delivery

FFmpeg's expression evaluator caps a single expression at roughly one hundred parsed nodes
(`av_expr_parse()` starts with `p.stack_index = 100`). A 31-second short sampled at 5 fps produces
~155 crop keyframes, which exceeds that budget in either a nested `if()` chain or a flat sum, and
`crop` then fails with `Failed to configure input pad ... Invalid argument`.

The trajectory is therefore delivered with `sendcmd`: one short, independently parsed linear
expression per keyframe interval. There is no practical keyframe limit and no loss of tracking
resolution. Two fallbacks follow, in order:

| Driver | When | Keyframes |
| --- | --- | --- |
| `sendcmd` | default | all of them |
| `expression` | build has no `sendcmd` filter | thinned to 45, to stay under the node budget |
| `static` | trajectory invalid, or a dynamic render fails | 1 |

Before any render, the trajectory is validated and repaired: non-finite points are dropped,
coordinates are clamped into `[0, source - crop]`, odd offsets are snapped to even (H.264 yuv420p),
and non-monotonic timestamps are fixed. Geometry that cannot be repaired (crop larger than the
source) drops straight to the static driver. The full report is logged before FFmpeg is invoked and
stored in the short's metadata as `trajectory_report`.

### Output files and failure handling

Every short is rendered to `<name>.tmp.mp4`, validated with ffprobe, and only then moved into place
with an atomic rename, so a failed render never leaves a zero-byte MP4 behind.

Filenames always carry the candidate id, so a single-candidate render can never overwrite a batch
result:

```
shorts/short_01_cand_010.mp4          # batch, ranked position 1
shorts/short_cand_010.mp4             # single --candidate render
shorts/short_01_cand_010.json
shorts/short_01_cand_010_crop_trajectory.json
shorts/short_01_cand_010_crop_commands.txt
shorts/short_01_cand_010_debug.mp4    # --debug-overlay
shorts/shorts_manifest.json
```

A render that writes over an output belonging to a *different* candidate is refused outright.

One failing candidate never aborts a batch. A failed smart render is retried with a static center
crop; if that also fails the candidate is recorded as failed and the batch continues. The manifest
carries `success_count`, `fallback_count`, `failure_count` and a per-candidate `results` list with
`status` (`success` / `fallback` / `failed`) and a reason.

### CLI

```bash
# One ranked candidate
python -m freecher_worker render-short RUN \
  --candidate cand_037 \
  --duration-mode auto

# Top N ranked candidates -> short_01.mp4, short_02.mp4, ...
python -m freecher_worker render-shorts RUN \
  --top 5 \
  --duration-mode auto
```

Useful flags: `--encoder libx264|h264_nvenc|auto`, `--no-reframe` (static center crop),
`--no-loudnorm`, `--debug-overlay` (writes a diagnostic video with detection boxes, crop
rectangle, crop centre and active subject).

Output lands in `RUN/shorts/`:

```
shorts/short_01.mp4
shorts/short_01.json
shorts/short_01_crop_trajectory.json
shorts/shorts_manifest.json
```

```json
{
  "candidate_id": "cand_037",
  "source_start_sec": 1234.2,
  "source_end_sec": 1294.2,

  "short_start_offset_sec": 8.4,
  "short_end_offset_sec": 31.8,

  "short_source_start_sec": 1242.6,
  "short_source_end_sec": 1266.0,

  "duration_sec": 23.4,

  "aspect_ratio": "9:16",
  "width": 1080,
  "height": 1920,

  "reframing_mode": "smart",
  "duration_mode": "auto"
}
```

The same file also carries diagnostics: original candidate duration, selected duration and source
range, detected subjects and tracks, dominant-subject switches, scene cuts, crop trajectory
statistics, fallback usage counters, encoder used and per-stage timings.

### Hardware

Encoding auto-detects `h264_nvenc` and falls back to `libx264`, and retries with `libx264` if an
NVENC encode fails mid-run. NVENC is never required: on the GTX 1650 / WSL2 target it is
typically unavailable and the pipeline runs entirely on CPU x264. Detection and tracking run on
downscaled frames through OpenCV on the CPU, so this stage consumes no VRAM.

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
