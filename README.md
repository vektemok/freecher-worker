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

Three production stages run *after* ranking. None of them reads or modifies `heuristic_v1`,
`highlight_v2_1`, `multimodal_v1_1` or any scoring weight.

```
Video -> Whisper -> candidates -> heuristic_v1 + highlight_v2_1 -> shortlist
      -> multimodal_v1_1 / Luna -> ranked candidate
      -> [1] Dynamic Subclip Refinement
      -> [2] Smart 9:16 Reframing
      -> [3] Adaptive Vertical Layout   (--layout-mode adaptive)
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

### Subject identity and track continuity

Every layout decision downstream rests on one question: *is this the same person as a moment ago?*
Getting that wrong does not look like a tracking bug. It looks like an empty room — a clip with
72% detection coverage, 37 tracks and not one subject that lasts, which the planner can only read
as "there is nobody to frame here".

Four things keep a physical person on one track id:

**Confirmation.** A new detection starts a *tentative* track. It becomes an identity only after
`reframe_track_confirm_hits` sightings, and confirmed identities are matched against detections
before any tentative track is allowed to compete. Real detectors emit single-frame ghosts, and
without this a ghost that lands near a real subject can capture the detection, drag the identity
away from the person, and leave the real face to start a new track on the next sample. Only
confirmed identities are counted, reported, or handed to the layout planner.

**Gap-aware re-attachment.** The elapsed time that governs how far a subject may have moved is
the gap since *that track* was last seen, not the interval between the last two samples. A subject
lost for a second is allowed to have moved for a second. Identity lifetime is
`reframe_track_max_gap_sec` (default 1.6 s) and is expressed in seconds, so it means the same
thing at any analysis frame rate.

**Refusal to merge.** Recovery is bounded so it never becomes invention: a hard ceiling on the
re-association distance (`reframe_track_max_reassociation_px`), a box-size ratio gate, and a tiny
contrast-normalized appearance patch (16x16, CPU-only, no re-identification model and no GPU) that
has to agree before an identity is allowed to survive a gap. Two people at opposite ends of the
frame, a close facecam and a distant face, and two subjects crossing each other all stay distinct.

**Scene awareness.** Across a cut the camera moved, not the subject, so motion prediction is
discarded and a track that was already missing when the shot changed is dropped rather than
carried into a scene it does not belong to. Disable with `FREECHER_REFRAME_TRACK_SCENE_CUT_RESET=false`.

Persistence is judged **locally**: a subject continuously on screen for
`layout_persistent_min_continuous_sec` (default 2 s) is a real subject regardless of how long the
clip is. Requiring a fixed share of the whole clip erased anyone who was not present for most of
it, which is how a busy 30-second scene reported zero persistent subjects.

Every short carries a continuity report under `reframe.fragmentation`:

| Field | Meaning |
| --- | --- |
| `identities` / `raw_candidates` | confirmed subjects vs every track id minted, ghosts included |
| `fragmentation_ratio` | candidates per identity; `1.0` is perfect continuity |
| `mean_lifetime_sec` / `median_lifetime_sec` / `longest_lifetime_sec` | how long identities survive |
| `tracks_under_half_second` / `tracks_under_one_second` | how much of the population is churn |
| `detections_total` / `detections_per_identity` | roughly, frames per person |
| `reattachments` | identities recovered after a detection gap |
| `discarded_tentative` | candidates that never earned an identity |
| `warning` | set when plentiful detections still produced no stable identity |

The layout plan adds `detections_per_persistent_identity` and a `fragmentation_warning` for the
same failure one level up: many tracked subjects, none of them usable. A large
`detections_per_persistent_identity` next to a small `persistent_track_count` means tracking, not
the scene, is the problem.

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

### 4. Adaptive Vertical Layout (`adaptive_layout_v1`)

Smart reframing answers *where do I point one vertical crop*. That is the wrong question for a
two-person exchange or a group shot, where a single crop silently deletes a participant. The
adaptive layout stage answers a different question first — *what shape does this moment need* —
and only then hands the geometry to FFmpeg.

```
final subclip -> visual analysis -> subject tracks -> layout planner -> layout plan -> renderer
```

The planner renders nothing. It emits a declarative plan that is stored in the short's metadata,
so what the renderer did is always inspectable after the fact.

| Layout | When | How it is built |
| --- | --- | --- |
| `single_subject` | one dominant persistent subject | the existing smart crop, unchanged |
| `dual_stack` | two persistent subjects that cannot share one safe crop | two independently tracked 1080x960 viewports, stacked |
| `full_frame_context` | 3+ significant subjects, weak evidence, or ambiguous composition | the whole source frame fitted into 9:16 over a blurred copy of itself |

The product rule behind every tie-break:

> **Loss of context is worse than a smaller subject.**

A slightly smaller person is recoverable. An important participant cropped out of frame is not.
So when the planner is unsure, it chooses `full_frame_context` rather than an aggressive crop.

**Decision inputs** are temporal, never single-frame. Detections are aggregated into per-subject
tracks (`visible_duration`, `visibility_ratio`, `continuity_score`, `mean_bbox`,
`position_variance`, `detection_confidence`, `scene_ids`, ...), and a track only becomes a
*persistent* subject after `FREECHER_LAYOUT_PERSISTENT_MIN_VISIBLE_SEC` of screen time. A face
that flickers for two frames can never open a second viewport.

`can_fit_subjects_in_single_vertical_crop()` decides `single` vs `dual` on three tests at once:
the padded union of the subjects fits horizontally; no subject shrinks below
`layout_min_face_height_ratio` of the output height; and every face lands inside the safe band,
clear of the caption and platform-UI zones (`layout_safe_top_ratio`, `layout_safe_bottom_ratio`).

**Temporal stability.** A layout change reads as an edit, so the planner analyses ~0.75 s windows
and then applies hysteresis: a challenger must win a *share* of the evidence over the last
`layout_switch_confirmation_sec`, beating the incumbent's share by `layout_switch_penalty`, and
the incumbent must first have held for `layout_min_duration_sec`. Comparing shares rather than
single windows is what makes one dissenting window harmless in both directions. A subject the
detector drops for less than `layout_subject_missing_grace_sec` still counts as present. A real
scene cut is allowed to switch immediately — the viewer is already being shown something new.
A 30 s short typically ends up with 0-3 transitions.

**Rendering** is a single FFmpeg pass. Each layout is one branch of one filtergraph, all producing
full 1080x1920 frames, and the plan's segments select which branch is on screen:

```
[0:v] split -> single : sendcmd + crop@single                    -> scale/crop 1080x1920
            -> dual   : sendcmd + crop@dual_top / crop@dual_bottom -> 2x 1080x960 -> vstack
            -> full   : blurred cover + fitted contain            -> overlay centred
      -> overlay ... enable='gte(t,s)*lt(t,e)' -> [v]
```

Two details make that work rather than merely look plausible. Each moving viewport gets its **own**
`sendcmd` script targeting its **own** named crop instance, because three viewports panning
independently cannot share one command stream. And the switch predicate is `gte(t,s)*lt(t,e)`
rather than `between()`, which is inclusive at both ends and would let two layouts claim the frame
sitting exactly on a segment boundary.

Branches are built only when the plan uses them, so a clip that never leaves `single_subject`
costs what it cost before. Layout switches are hard cuts; the planner prefers to place them on
scene cuts, where a hard cut is what the footage is already doing.

**Fallbacks**, in order, so a valid 9:16 MP4 is always produced: an FFmpeg build without named
filter instances -> `single_subject`; a `dual_stack` segment without two distinct tracks ->
`full_frame_context`; an adaptive filtergraph that fails to render -> the single-subject crop ->
static center crop. Every fallback is recorded in `layout_fallback_reason`.

**`--layout-mode`** selects the strategy. The default is `single`, so nothing changes until you
ask for it:

```bash
python -m freecher_worker render-shorts RUN \
  --scorer multimodal_v1_1 \
  --top 5 \
  --duration-mode auto \
  --layout-mode adaptive
```

`--layout-mode full-frame` forces the context-safe layout for the whole short, which is useful for
side-by-side review. `--debug-overlay` additionally draws the active layout, its subjects,
confidence, reason and the safe area onto the diagnostic video.

The layout stage is presentation only. It never reads or changes highlight scoring, retrieval,
ranking or Dynamic Subclip Refinement: the same candidate publishes exactly the same
`short_source_start_sec` / `short_source_end_sec` in every layout mode.

Configurable via `FREECHER_LAYOUT_*` (window length, persistence and significance thresholds,
dominant/secondary visibility, group size, minimum readable face height, safe-area margins,
hysteresis timings, blur strength and background colour).

The plan is stored per short and summarized in `shorts_manifest.json`:

```json
"layout": {
  "version": "adaptive_layout_v1",
  "mode_requested": "adaptive",
  "segments": [
    {"start": 0.0, "end": 9.8, "layout": "single_subject", "subjects": ["track_1"],
     "confidence": 0.85, "reason": "one dominant persistent subject"},
    {"start": 9.8, "end": 19.5, "layout": "dual_stack", "subjects": ["track_1", "track_2"],
     "confidence": 0.82, "reason": "two persistent subjects cannot fit one safe crop"},
    {"start": 19.5, "end": 30.0, "layout": "full_frame_context", "subjects": [],
     "confidence": 0.75, "reason": "group scene: 4 significant subjects"}
  ],
  "switch_count": 2,
  "duration_by_mode": {"single_subject": 9.8, "dual_stack": 9.7, "full_frame_context": 10.5},
  "track_count": 4,
  "persistent_track_count": 4,
  "dominant_track_id": 1,
  "mean_tracking_confidence": 0.92
}
```

**Known limitation.** The planner reasons about people, not objects. When a moment is *about* a
visual object — a bag, a phone, something on the table — a face-only planner has no way to know
which person must stay in frame. `adaptive_layout_v1` handles this only by preferring
`full_frame_context` whenever the composition is ambiguous. Understanding what the scene is about
is `semantic framing v1.1`; the `semantic_context` hook (`important_subject_ids`,
`important_object`, `speaker_hint`, `prefer_full_frame`) is already wired into the planner and
deliberately not connected to `multimodal_v1_1` — highlight intelligence and framing intelligence
stay separate bounded components.

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
shorts/short_01_cand_010_crop_commands_dual_top.txt     # --layout-mode adaptive
shorts/short_01_cand_010_crop_commands_dual_bottom.txt  # --layout-mode adaptive
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

Useful flags: `--encoder libx264|h264_nvenc|auto`, `--layout-mode single|adaptive|full-frame`
(default `single`), `--no-reframe` (static center crop), `--no-loudnorm`, `--debug-overlay`
(writes a diagnostic video with detection boxes, crop rectangle, crop centre, active subject and -
in adaptive mode - the active layout, its subjects, confidence, reason and safe area).

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
  "duration_mode": "auto",
  "layout_mode_requested": "single",
  "layout_version": "adaptive_layout_v1"
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

        contextual/
            __init__.py
            versions.py
            models.py
            prompts.py
            provider.py
            cache.py
            chapters.py
            context.py
            candidate_context.py
            signals.py
            editorial.py
            critic.py
            comparative.py
            reranker.py
            diagnostics.py

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

## Contextual Highlight Intelligence (`contextual_reranker_v1_1`)

A bounded reranking layer that runs **after** retrieval and multimodal evidence and
**before** the final Top-K selection. It does not replace `multimodal_v1_1`; it consumes
its output as evidence.

```
existing ASR -> candidates -> heuristic / highlight_v2_1 retrieval -> multimodal_v1_1 evidence
   -> GLOBAL + CHAPTER CONTEXT
   -> SALVAGE-AWARE EDITORIAL ASSESSMENT
   -> FATAL-QUALITY FILTER + SOFT PENALTIES
   -> PENALTY-ORIENTED CRITIC
   -> COMPARATIVE RERANKER
   -> Top highlights -> existing Dynamic Subclip Refinement -> existing renderer
```

### Why it exists

Candidates were being judged in isolation, so ordinary conversation that *sounded*
energetic outranked moments with a real payoff. This layer gives the model the whole
video's context, then forces an editorial decision rather than a score.

### Stages

1. **Chapters** — the transcript is split on real speech gaps (and cached scene changes
   when a multimodal activity profile exists) into ~3–5 minute chapters. One summary
   request per chapter; the 60-minute transcript is never sent in one request.
2. **Global context** — one request derives the whole-video understanding
   (`video_summary`, `content_type`, participants, goals, running jokes, conflicts) from
   the chapter summaries alone.
3. **Candidate context package** — per candidate: global context, its chapter context,
   a 75 s BEFORE window, the candidate, a 25 s AFTER window, multimodal evidence, and
   cheap activity-derived reaction signals. BEFORE/AFTER provide understanding and setup
   options; later refinement, not the reranker, chooses final clip boundaries.
4. **Editorial assessment** — the model returns `FATAL_REJECT`, `WEAK`, `MAYBE`, `GOOD`,
   or `STRONG`, plus explicit salvageability/setup/boundary fields and a soft quality penalty.
   Missing setup, messy ASR, context dependency, and imperfect boundaries do not delete a
   candidate. Only strongly evidenced, fundamentally unusable material is fatal.
5. **Critic** — a separate pass primarily emits a penalty, confidence, failure modes, and
   `keep_for_comparison`. Hard rejection requires explicit fatal evidence and high confidence.
6. **Comparative reranking** — every non-fatal candidate is ordered against the others,
   not by summing numbers: listwise batches, then a Swiss tournament, then round-robin
   across the top group. For 16 survivors that is ~39 comparisons instead of the 120 a
   full pairwise matrix needs.

The numeric dimensions (`scroll_stop`, `hook`, `payoff`, ...) are recorded for
observability. They are **not** a ranking formula.

### Commands

```bash
# Rerank an existing run (requires scores/multimodal_v1_1.json)
python -m freecher_worker contextual-rerank runs/benchmark_02 \
  --input-scorer multimodal_v1_1 \
  --model gpt-5.6-luna
```

Options: `--top`, `--model`, `--reasoning-effort`, `--temperature`, `--force`,
`--no-critic`, `--comparison-mode` (`full | swiss | listwise | none`), `--input-scorer`,
`--output`.

```bash
# Where did the moment a human liked actually go?
python -m freecher_worker inspect-moment runs/benchmark_02 --time 00:18:34

# Blind diagnostic package: candidate generation vs retrieval vs reranking
python -m freecher_worker export-blind-diagnostic runs/benchmark_02 \
  --group-a 4 --group-b 4 --seed 1337

# Ranking metrics plus RejectPrecision / StrongPrecision
python -m freecher_worker evaluate-contextual \
  runs/benchmark_02/evaluation.json \
  runs/benchmark_02/scores/contextual_reranker_v1_1.json

# Known hard cases (strong positives, false positives, false negatives)
python -m freecher_worker regression-check regression_dataset.json \
  runs/benchmark_02/scores/contextual_reranker_v1_1.json
```

`inspect-moment` prints `NO CANDIDATE COVERAGE` when no candidate window contains the
timestamp — that is a candidate-generation gap, not a ranking problem.

`export-blind-diagnostic` writes `blind_diagnostic.json` (no candidate ids, no ranks, no
scores) next to `_DO_NOT_OPEN_mapping.json`. Both are reproducible from `--seed`.

### Artifacts

| Path | Contents |
| --- | --- |
| `scores/contextual_reranker_v1_1.json` | Scorer artifact; comparative pool ranked first, then fatal rejects with reasons |
| `contextual/contextual_reranker_v1_1_run.json` | Full record: salvageability, penalties, comparisons, diagnostics, and usage |
| `contextual/global_context_v1.json` | Whole-video understanding |
| `contextual/chapter_context_v1.json` | Chapter summaries, setups, payoffs, open loops |
| `contextual/candidate_context_v1.json` | Per-candidate context packages |
| `contextual/cache/<stage>/<hash>.json` | Per-stage response cache |

### Caching and cost

Each stage caches separately, so a rerank never recomputes global or chapter context.
Cache keys carry `model`, `reasoning_effort`, `temperature`, `prompt_version`,
`prompt_hash`, `schema_version`, `context_version`, and `reranker_version`; changing any
one invalidates exactly that stage. `--force` bypasses the cache.

Every run reports `number_of_api_calls`, per-stage call counts, cache hits, and input /
output tokens when the provider returns a usage block.

### Human labels

Human ratings are **never** used during inference — not in a prompt, not in a cache key,
not in the reject filter. They are only read afterwards by `evaluate-contextual` and
`regression-check`.

### Configuration

`FREECHER_CONTEXTUAL_MODEL`, `FREECHER_CONTEXTUAL_BASE_URL`, `FREECHER_CONTEXTUAL_API_KEY`,
`FREECHER_CONTEXTUAL_REASONING_EFFORT`, `FREECHER_CONTEXTUAL_TEMPERATURE`,
`FREECHER_CONTEXTUAL_INPUT_SCORER`, `FREECHER_CONTEXTUAL_COMPARISON_MODE`,
`FREECHER_CONTEXTUAL_BEFORE_SECONDS`, `FREECHER_CONTEXTUAL_AFTER_SECONDS`,
`FREECHER_CONTEXTUAL_CHAPTER_TARGET_SECONDS`, `FREECHER_CONTEXTUAL_LISTWISE_BATCH_SIZE`,
`FREECHER_CONTEXTUAL_FINAL_PAIRWISE_TOP`, `FREECHER_CONTEXTUAL_TOP`.
They fall back to the existing `FREECHER_LLM_*` / `FREECHER_MULTIMODAL_*` variables.

---

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
