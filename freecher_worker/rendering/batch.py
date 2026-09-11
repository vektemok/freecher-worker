"""Batch render Top-N highlights and publish the finished clips to R2.

Phase 2 (render) and Phase 3 (publish) of the product path. Deliberately small:
it orchestrates the existing `render_single_short`, the existing ffprobe
validation, and the existing R2 client. Nothing here re-implements those.

Contracts that matter in production:

* one clip's failure never marks the batch successful -- each clip carries its
  own status, and the batch status is DONE only when every requested clip is DONE,
* every finished MP4 is validated with ffprobe before it is offered to anyone,
* uploads are idempotent: an object already present with the same size and sha256
  is not re-sent, and the local file is never removed here,
* re-running is cheap: a local artifact that still validates is not re-rendered.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from freecher_worker.config import Settings, get_settings
from freecher_worker.highlights.models import Highlight
from freecher_worker.media.ffmpeg_env import resolve_ffprobe
from freecher_worker.transcription.models import Transcript

logger = logging.getLogger("freecher_worker")

CLIP_KEY_TEMPLATE = "output/{source_id}/clips/{clip_id}.mp4"
MANIFEST_KEY_TEMPLATE = "output/{source_id}/clips.json"
DEFAULT_TOP_N = 5


class ClipStatus(str, Enum):
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ClipRecord(BaseModel):
    """Structured metadata for one finished clip."""

    clip_id: str
    candidate_id: str
    rank: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    width: Optional[int] = None
    height: Optional[int] = None
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    subtitles_burned: bool = False
    crop_mode: Optional[str] = None
    crop_summary: Optional[str] = None
    local_path: Optional[str] = None
    sha256: Optional[str] = None
    bytes: Optional[int] = None
    r2_key: Optional[str] = None
    r2_url: Optional[str] = None
    status: ClipStatus = ClipStatus.FAILED
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    render_seconds: Optional[float] = None
    upload_seconds: Optional[float] = None


class ClipsManifest(BaseModel):
    source_id: str
    status: str = "FAILED"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    requested: int = 0
    succeeded: int = 0
    failed: int = 0
    clips: list[ClipRecord] = Field(default_factory=list)


# ------------------------------------------------------------------ validation
class ClipValidationError(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def validate_clip(
    path: Path,
    *,
    expected_width: int,
    expected_height: int,
    expected_duration: float,
    duration_tolerance: float = 1.5,
    settings: Optional[Settings] = None,
) -> dict[str, Any]:
    """ffprobe the finished MP4. Raises ClipValidationError with a specific reason."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ClipValidationError(f"{path.name}: missing or empty")

    ffprobe = resolve_ffprobe(settings or get_settings())
    if not ffprobe:
        raise ClipValidationError("ffprobe not available; cannot validate output")

    res = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height", "-show_entries",
         "format=duration,size", "-of", "json", str(path)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=120, check=False,
    )
    if res.returncode != 0:
        raise ClipValidationError(f"{path.name}: unreadable ({res.stderr.strip()[:160]})")
    try:
        info = json.loads(res.stdout)
    except json.JSONDecodeError as exc:
        raise ClipValidationError(f"{path.name}: ffprobe returned invalid JSON") from exc

    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ClipValidationError(f"{path.name}: no video stream")
    if audio is None:
        raise ClipValidationError(f"{path.name}: no audio stream")

    w, h = int(video.get("width") or 0), int(video.get("height") or 0)
    if (w, h) != (expected_width, expected_height):
        raise ClipValidationError(
            f"{path.name}: expected {expected_width}x{expected_height}, got {w}x{h}")

    duration = float(info.get("format", {}).get("duration") or 0.0)
    if duration <= 0:
        raise ClipValidationError(f"{path.name}: non-positive duration")
    if abs(duration - expected_duration) > duration_tolerance:
        raise ClipValidationError(
            f"{path.name}: duration {duration:.2f}s differs from expected "
            f"{expected_duration:.2f}s by more than {duration_tolerance}s")

    return {"width": w, "height": h, "video_codec": video.get("codec_name"),
            "audio_codec": audio.get("codec_name"), "duration": duration,
            "bytes": int(info.get("format", {}).get("size") or path.stat().st_size)}


# --------------------------------------------------------------------- publish
def upload_clip(
    client: Any,
    bucket: str,
    key: str,
    path: Path,
    sha256: str,
    *,
    content_type: str = "video/mp4",
) -> bool:
    """Upload unless an identical object is already there. Returns True if sent.

    The local file is never deleted here -- callers keep it until they have
    verified the remote copy themselves.
    """
    try:
        head = client.head_object(Bucket=bucket, Key=key)
        remote_size = int(head.get("ContentLength") or 0)
        remote_sha = (head.get("Metadata") or {}).get("sha256")
        if remote_size == path.stat().st_size and remote_sha == sha256:
            logger.info("[publish] %s already present and identical; skipping upload", key)
            return False
    except Exception:  # noqa: BLE001 - any miss/error means "upload it"
        pass

    with open(path, "rb") as fh:
        client.put_object(
            Bucket=bucket, Key=key, Body=fh, ContentType=content_type,
            Metadata={"sha256": sha256},
        )
    return True


def verify_remote(client: Any, bucket: str, key: str, expected_bytes: int) -> bool:
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception:  # noqa: BLE001
        return False
    return int(head.get("ContentLength") or -1) == expected_bytes


# ----------------------------------------------------------------------- batch
def select_highlights(highlights: Iterable[dict], top_n: int) -> list[dict]:
    """Top-N by rank, deterministically. Ties broken by candidate_id."""
    items = sorted(highlights, key=lambda h: (int(h.get("rank", 10**6)),
                                              str(h.get("candidate_id", ""))))
    return items[: max(0, top_n)]


def clip_id_for(source_id: str, candidate_id: str, rank: int) -> str:
    """Stable, deterministic clip identifier."""
    return f"{source_id[:12]}_r{rank:02d}_{candidate_id}"


def render_top_n(
    run_dir: Path,
    source_video: Path,
    source_id: str,
    *,
    top_n: int = DEFAULT_TOP_N,
    preset_name: str = "shorts",
    enable_smart_crop: bool = True,
    enable_subtitles: bool = True,
    enable_audio_normalization: bool = True,
    refine_boundaries: bool = True,
    force: bool = False,
    publish: bool = True,
    settings: Optional[Settings] = None,
) -> ClipsManifest:
    """Render the Top-N ranked highlights, validate each, and publish to R2.

    Each clip is independent: one failure is recorded on that clip and the batch
    continues. The batch is DONE only if every requested clip reached DONE.
    """
    from freecher_worker.rendering.presets import get_preset
    from freecher_worker.rendering.renderer import render_single_short

    cfg = settings or get_settings()
    preset = get_preset(preset_name)
    highlights_raw = json.loads((run_dir / "highlights.json").read_text())
    if isinstance(highlights_raw, dict):
        highlights_raw = highlights_raw.get("highlights", [])
    selected = select_highlights(highlights_raw, top_n)

    transcript = Transcript.model_validate(json.loads((run_dir / "transcript.json").read_text()))
    manifest_data = json.loads((run_dir / "manifest.json").read_text())
    fp_id = (manifest_data.get("source_fingerprint") or {}).get("fingerprint_id", "unknown_fp")
    video_duration = float((manifest_data.get("source_fingerprint") or {}).get("duration_seconds", 0.0))

    client = bucket = None
    if publish:
        from freecher_worker.ingest.r2 import build_r2_client
        client = build_r2_client(cfg.r2_endpoint, cfg.r2_access_key_id, cfg.r2_secret_access_key)
        bucket = cfg.r2_bucket

    out = ClipsManifest(source_id=source_id, requested=len(selected))
    clips_dir = run_dir / "final"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # One refinement model for the whole batch. Construction is free -- the model
    # loads lazily on the first clip that actually needs rendering -- so a batch
    # in which every clip is reused still never loads it. Released in the finally
    # below, so a worker that rendered one job does not hold 2+ GiB afterwards.
    transcriber = None
    if enable_subtitles:
        from freecher_worker.rendering.asr_refinement import HighlightWordTranscriber

        transcriber = HighlightWordTranscriber(
            model_name=cfg.refinement_asr_model,
            device=cfg.asr_device,
            compute_type=cfg.refinement_asr_compute_type,
        )

    try:
        for raw in selected:
            rank = int(raw["rank"])
            cand = str(raw["candidate_id"])
            cid = clip_id_for(source_id, cand, rank)
            rec = ClipRecord(
                clip_id=cid, candidate_id=cand, rank=rank,
                start_seconds=float(raw["start"]), end_seconds=float(raw["end"]),
                duration_seconds=float(raw["end"]) - float(raw["start"]),
            )
            target = clips_dir / f"{cid}.mp4"
            t0 = time.perf_counter()
            try:
                expected_dur = rec.duration_seconds
                reuse = False
                if target.is_file() and not force:
                    # Read the sidecar BEFORE validating. Boundary refinement moves the
                    # window, so the highlight's raw duration is not what was rendered;
                    # validating against it fails and the clip is re-rendered on every
                    # run, quietly defeating idempotency.
                    sidecar = target.with_suffix(".json")
                    prior = None
                    if sidecar.is_file():
                        try:
                            prior = json.loads(sidecar.read_text())
                            expected_dur = float(prior.get("duration_seconds", expected_dur))
                        except (json.JSONDecodeError, TypeError, ValueError):
                            prior = None
                    try:
                        probe = validate_clip(target, expected_width=preset.width,
                                              expected_height=preset.height,
                                              expected_duration=expected_dur, settings=cfg)
                        reuse = True
                        logger.info("[batch] %s already rendered and valid; reusing", cid)
                        if prior:
                            rec.subtitles_burned = bool(prior.get("subtitles_burned", False))
                            rec.crop_mode = prior.get("crop_mode")
                            rec.crop_summary = prior.get("crop_summary")
                            rec.start_seconds = float(prior.get("start_seconds", rec.start_seconds))
                            rec.end_seconds = float(prior.get("end_seconds", rec.end_seconds))
                            rec.duration_seconds = float(prior.get("duration_seconds",
                                                                  rec.duration_seconds))
                    except ClipValidationError as exc:
                        logger.info("[batch] %s exists but did not validate (%s); re-rendering",
                                    cid, exc)
                        reuse = False

                if not reuse:
                    item = render_single_short(
                        highlight=Highlight.model_validate(raw), transcript=transcript,
                        preset=preset, source_video=source_video, run_dir=run_dir,
                        source_fingerprint_id=fp_id, video_duration=video_duration,
                        config=cfg, enable_smart_crop=enable_smart_crop,
                        enable_subtitles=enable_subtitles,
                        enable_audio_normalization=enable_audio_normalization,
                        # `force` here governs the refinement-ASR word cache, not
                        # whether we re-render: reuse was already decided above.
                        # Passing True discarded a cache whose key covers the
                        # fingerprint, the exact boundaries, the model, the
                        # compute type and the language -- so it re-ran ~100 s of
                        # Whisper per clip to produce byte-identical words. Honour
                        # the caller's force instead, which is what a user asking
                        # to redo the work actually means.
                        force=force, refine_boundaries=refine_boundaries,
                        transcriber=transcriber,
                    )
                    produced = run_dir / item.file
                    produced.replace(target)
                    rec.subtitles_burned = bool(item.subtitles_burned)
                    rec.crop_mode = item.crop_mode
                    rec.start_seconds, rec.end_seconds = item.refined_start, item.refined_end
                    rec.duration_seconds = item.duration
                    expected_dur = item.duration
                    crop_json = run_dir / "crop_paths" / f"highlight_{rank:02d}.json"
                    if crop_json.is_file():
                        diag = (json.loads(crop_json.read_text()).get("diagnostics") or {})
                        rec.crop_summary = diag.get("summary")
                    probe = validate_clip(target, expected_width=preset.width,
                                          expected_height=preset.height,
                                          expected_duration=expected_dur, settings=cfg)

                rec.render_seconds = round(time.perf_counter() - t0, 2)
                rec.width, rec.height = probe["width"], probe["height"]
                rec.video_codec, rec.audio_codec = probe["video_codec"], probe["audio_codec"]
                rec.bytes = probe["bytes"]
                rec.local_path = str(target)
                rec.sha256 = sha256_file(target)

                if publish:
                    key = CLIP_KEY_TEMPLATE.format(source_id=source_id, clip_id=cid)
                    t1 = time.perf_counter()
                    upload_clip(client, bucket, key, target, rec.sha256)
                    if not verify_remote(client, bucket, key, rec.bytes):
                        raise RuntimeError(f"remote object {key} did not verify after upload")
                    rec.upload_seconds = round(time.perf_counter() - t1, 2)
                    rec.r2_key = key
                    if cfg.r2_public_base_url:
                        rec.r2_url = f"{cfg.r2_public_base_url.rstrip('/')}/{key}"

                rec.status = ClipStatus.DONE
                # Per-clip sidecar: the durable record for THIS artifact, independent
                # of whichever batch selection last wrote clips.json.
                target.with_suffix(".json").write_text(rec.model_dump_json(indent=2))
                out.succeeded += 1
                logger.info("[batch] %s DONE (%s bytes)", cid, rec.bytes)
            except Exception as exc:  # noqa: BLE001 - per-clip isolation is the point
                rec.status = ClipStatus.FAILED
                rec.error_type = type(exc).__name__
                rec.error_message = str(exc)[:400]
                out.failed += 1
                logger.error("[batch] %s FAILED: %s: %s", cid, type(exc).__name__, exc)
            out.clips.append(rec)
    finally:
        # Per-clip failures are already isolated inside the loop, so reaching
        # here means the batch is over one way or another and the model is no
        # longer needed. A clip failure never invalidates the model itself.
        if transcriber is not None:
            transcriber.release()

    out.status = "DONE" if (out.failed == 0 and out.succeeded == out.requested
                            and out.requested > 0) else "FAILED"

    local_manifest = run_dir / "clips.json"
    local_manifest.write_text(out.model_dump_json(indent=2))
    if publish and client is not None:
        client.put_object(
            Bucket=bucket, Key=MANIFEST_KEY_TEMPLATE.format(source_id=source_id),
            Body=out.model_dump_json(indent=2).encode("utf-8"),
            ContentType="application/json",
        )
    return out
