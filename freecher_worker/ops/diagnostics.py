"""Deployment diagnostics: is this host actually able to do its job?

One implementation feeds both `GET /health` and `freecher-worker preflight`, so
what an operator reads over SSH and what a monitor scrapes cannot drift apart.

Every check answers a question that has a distinct remedy:

    api        the process is serving                     -> nothing to fix
    jobs       the job directory exists and is writable   -> path/permissions
    r2         credentials present and the bucket answers -> env/network
    ffmpeg     a usable binary with the required filters  -> FREECHER_FFMPEG_PATH
    libass     that binary can burn subtitles             -> install a full build
    transcribe placement resolves and is coherent         -> FREECHER_TRANSCRIBE_*

No check returns a credential. R2 is reported by bucket name and reachability;
the access key, secret and account-bearing endpoint never appear.
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

#: A network round-trip per scrape would let a monitor's polling interval drive
#: R2 request volume. One probe per window is plenty for a liveness signal.
R2_CACHE_SECONDS = 15.0
_R2_CACHE: dict[str, tuple[float, "Check"]] = {}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    #: Extra structured facts. Must never carry secrets.
    info: dict[str, Any] = field(default_factory=dict)
    #: False for checks a host may legitimately fail (a worker-only concern on
    #: an API-only host, say). Only required failures make the host degraded.
    required: bool = True


@dataclass
class Diagnostics:
    status: str
    checks: list[Check]

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "checks": [asdict(c) for c in self.checks]}

    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.required and not c.ok]


def check_jobs_dir(jobs_dir: Optional[Path] = None) -> Check:
    from freecher_worker.jobs.store import DEFAULT_ROOT

    root = Path(jobs_dir or os.environ.get("FREECHER_JOBS_DIR") or DEFAULT_ROOT)
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".healthcheck"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        return Check("jobs", False, f"{root} is not writable: {exc}", {"path": str(root)})
    count = len(list(root.glob("*.json")))
    return Check("jobs", True, f"{root} writable", {"path": str(root), "jobs": count})


def check_r2(settings: Any, *, use_cache: bool = True) -> Check:
    """Credentials configured, and the bucket answers a HEAD.

    Reports the bucket name only. The endpoint embeds the Cloudflare account id,
    so it stays out of the response.
    """
    missing = [n for n in ("r2_bucket", "r2_endpoint", "r2_access_key_id",
                           "r2_secret_access_key") if not getattr(settings, n, None)]
    if missing:
        return Check("r2", False, "not configured: " + ", ".join(
            f"FREECHER_{n.upper()}" for n in missing))

    bucket = settings.r2_bucket
    cached = _R2_CACHE.get(bucket)
    if use_cache and cached and (time.monotonic() - cached[0]) < R2_CACHE_SECONDS:
        return cached[1]

    try:
        from freecher_worker.ingest.r2 import build_r2_client

        client = build_r2_client(settings.r2_endpoint, settings.r2_access_key_id,
                                 settings.r2_secret_access_key)
        client.head_bucket(Bucket=bucket)
        result = Check("r2", True, f"bucket '{bucket}' reachable", {"bucket": bucket})
    except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
        result = Check("r2", False, f"bucket '{bucket}' unreachable: {type(exc).__name__}",
                       {"bucket": bucket})
    _R2_CACHE[bucket] = (time.monotonic(), result)
    return result


def check_ffmpeg(settings: Any) -> list[Check]:
    """Binary resolution, required filters/encoders, and libass, as three checks.

    Split because the remedies differ: a missing binary is a PATH problem, a
    missing encoder means a crippled build, and missing libass means the right
    build was never installed -- and only the last one is what silently produced
    subtitle-less clips before renders were made to fail loudly.
    """
    from freecher_worker.media.ffmpeg_env import FFmpegNotFoundError, probe_capabilities

    try:
        caps = probe_capabilities(settings)
    except FFmpegNotFoundError as exc:
        fail = Check("ffmpeg", False, str(exc).splitlines()[0])
        return [fail,
                Check("encoders", False, "not probed: no ffmpeg"),
                Check("libass", False, "not probed: no ffmpeg")]

    missing_f, missing_e = caps.missing_filters(), caps.missing_encoders()
    binary = Check(
        "ffmpeg", not missing_f,
        caps.version if not missing_f else f"missing filters: {', '.join(missing_f)}",
        {"ffmpeg": caps.ffmpeg_path, "ffprobe": caps.ffprobe_path or "",
         "filters": len(caps.filters), "encoders": len(caps.encoders)},
    )
    encoders = Check(
        "encoders", not missing_e,
        "libx264 + aac present" if not missing_e else f"missing: {', '.join(missing_e)}",
    )
    libass = Check(
        "libass", caps.supports_subtitle_burn,
        f"subtitle filter '{caps.subtitle_filter}'" if caps.supports_subtitle_burn
        else "this ffmpeg cannot burn subtitles; set FREECHER_FFMPEG_PATH to a libass build",
    )
    return [binary, encoders, libass]


def check_transcription(settings: Any) -> Check:
    """Placement resolves, and the resolved placement is actually usable."""
    from freecher_worker.transcription.handoff import (
        BACKEND_LOCAL, cuda_available, resolve_backend,
    )

    try:
        backend = resolve_backend(settings)
    except ValueError as exc:
        return Check("transcription", False, str(exc))

    info = {
        "backend": backend,
        "configured": str(getattr(settings, "transcribe_backend", "auto")),
        "cuda": cuda_available(),
        "cpu_limit_seconds": float(getattr(settings, "cpu_transcription_max_seconds", 900.0)),
        "allow_cpu": bool(getattr(settings, "allow_cpu_transcription", False)),
    }
    limit = info["cpu_limit_seconds"]
    if info["cuda"]:
        return Check("transcription", True, "local, CUDA", info)
    if info["allow_cpu"]:
        return Check("transcription", True, "local, CPU (explicitly allowed, any length)", info)
    if info["configured"].lower() == "auto":
        # resolve_backend answers the hardware question; the length guard in
        # plan_transcription is what actually routes a given source, so report
        # both halves rather than the misleadingly absolute "handoff".
        return Check("transcription", True,
                     f"auto: local CPU up to {limit:.0f}s of audio, GPU handoff beyond", info)
    if backend == BACKEND_LOCAL:
        return Check("transcription", True,
                     f"local CPU, refusing audio over {limit:.0f}s", info)
    return Check("transcription", True, "handoff to the GPU host via R2", info)


def check_ytdlp() -> Check:
    """Ingest needs yt-dlp on PATH or beside the interpreter."""
    from freecher_worker.ingest.source import SourceStreamError, resolve_ytdlp_path

    try:
        path = resolve_ytdlp_path()
    except SourceStreamError as exc:
        return Check("yt-dlp", False, str(exc))
    return Check("yt-dlp", True, path, {"path": path})


def check_disk(path: Optional[Path] = None, min_free_gib: float = 5.0) -> Check:
    """Rendering writes multi-hundred-megabyte intermediates; a full disk fails late."""
    target = Path(path or os.environ.get("FREECHER_JOBS_DIR") or ".").resolve()
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    free_gib = usage.free / (1 << 30)
    return Check("disk", free_gib >= min_free_gib,
                 f"{free_gib:.1f} GiB free at {target}",
                 {"free_gib": round(free_gib, 1), "min_gib": min_free_gib})


def collect(settings: Any = None, *, jobs_dir: Optional[Path] = None,
            include_network: bool = True) -> Diagnostics:
    """Every check, in the order an operator would want to read them."""
    from freecher_worker.config import get_settings

    cfg = settings or get_settings()
    checks: list[Check] = [Check("api", True, "serving")]
    checks.append(check_jobs_dir(jobs_dir))
    checks.append(check_r2(cfg) if include_network
                  else Check("r2", True, "not probed", required=False))
    checks.extend(check_ffmpeg(cfg))
    checks.append(check_transcription(cfg))
    checks.append(check_ytdlp())
    checks.append(check_disk(jobs_dir))
    status = "ok" if all(c.ok for c in checks if c.required) else "degraded"
    return Diagnostics(status=status, checks=checks)
