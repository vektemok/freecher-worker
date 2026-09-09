"""Audio preview for blind annotation, backed by the ingest audio artifact.

Labeling an R2-backed run has no local video, but the transcription artifact
at processing/{source_id}/audio.m4a is already mono speech on the source
timeline — which is exactly what a rater needs to judge delivery, a bad start
or a missing payoff.

The artifact is fetched once per session and cached; playback then seeks into
that one local file per candidate. `source.mp4` is never touched. Nothing here
writes to candidate data or scorer output: preview is annotation assistance and
has no effect on any recorded value.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from freecher_worker.transcription.r2 import (
    AudioArtifactInfo,
    TranscriptionWorkflowError,
    audio_key_for,
    download_audio_artifact,
    validate_audio_artifact,
)

logger = logging.getLogger("freecher_worker")

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "freecher-worker" / "audio"

AUDIO_FILENAME = "audio.m4a"


class AudioPreviewError(Exception):
    """Raised when the audio artifact cannot be fetched or played."""


@dataclass
class CachedAudio:
    """A locally cached audio artifact ready for range playback."""

    path: Path
    source_id: str
    bucket: str
    key: str
    duration_seconds: Optional[float] = None
    reused: bool = False

    @property
    def megabytes(self) -> float:
        return self.path.stat().st_size / 1024 / 1024


def cache_path_for(source_id: str, cache_dir: Optional[Path | str] = None) -> Path:
    """Where this source's artifact lives locally."""
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    return root / source_id / AUDIO_FILENAME


def ensure_audio_artifact(
    client: Any,
    bucket: str,
    source_id: str,
    *,
    cache_dir: Optional[Path | str] = None,
    key: Optional[str] = None,
    force: bool = False,
    validate: bool = True,
    ffprobe_path: str = "ffprobe",
    on_progress: Optional[Any] = None,
) -> CachedAudio:
    """Fetch processing/{source_id}/audio.m4a once, reusing the cache after that.

    The download is deliberately per-session, not per-candidate: 146 candidates
    over one two-hour artifact is one ~50 MB fetch, not 146 of them.
    """
    resolved_key = key or audio_key_for(source_id)
    destination = cache_path_for(source_id, cache_dir)

    if destination.is_file() and destination.stat().st_size > 0 and not force:
        logger.info("reusing cached audio artifact at %s", destination)
        duration = None
        if validate:
            try:
                duration = validate_audio_artifact(destination, ffprobe_path=ffprobe_path).duration_seconds
            except TranscriptionWorkflowError as exc:
                # A damaged cache entry is replaced rather than played.
                logger.warning("cached audio is unusable (%s); re-downloading", exc)
                destination.unlink(missing_ok=True)
                return ensure_audio_artifact(
                    client, bucket, source_id, cache_dir=cache_dir, key=resolved_key,
                    force=True, validate=validate, ffprobe_path=ffprobe_path,
                    on_progress=on_progress,
                )
        return CachedAudio(
            path=destination, source_id=source_id, bucket=bucket, key=resolved_key,
            duration_seconds=duration, reused=True,
        )

    try:
        _, metadata = download_audio_artifact(
            client, bucket, resolved_key, destination, on_progress=on_progress
        )
    except TranscriptionWorkflowError as exc:
        raise AudioPreviewError(
            f"could not fetch the audio artifact for '{source_id}': {exc}"
        ) from exc

    info: Optional[AudioArtifactInfo] = None
    if validate:
        try:
            info = validate_audio_artifact(
                destination, metadata=metadata, ffprobe_path=ffprobe_path
            )
        except TranscriptionWorkflowError as exc:
            destination.unlink(missing_ok=True)
            raise AudioPreviewError(f"the downloaded audio is unusable: {exc}") from exc

    return CachedAudio(
        path=destination, source_id=source_id, bucket=bucket, key=resolved_key,
        duration_seconds=info.duration_seconds if info else None, reused=False,
    )


def build_playback_command(
    audio_path: Path | str,
    start: float,
    end: float,
    *,
    ffplay_path: str = "ffplay",
) -> list[str]:
    """Play exactly [start, end) of the cached artifact, nothing else.

    `-ss` before `-i` seeks without decoding the preceding audio, so a window
    at 1:22:47 starts as fast as one at the beginning.
    """
    duration = max(0.0, end - start)
    return [
        ffplay_path,
        "-nodisp",
        "-autoexit",
        "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        str(audio_path),
    ]


class AudioPreviewer:
    """Plays candidate ranges from one cached artifact, one at a time.

    Playback runs detached so the annotator is never blocked for the length of
    a 60-second window; starting a new preview or moving on stops the old one.
    """

    def __init__(self, audio: CachedAudio, *, ffplay_path: str = "ffplay") -> None:
        self.audio = audio
        self.ffplay_path = ffplay_path
        self._process: Optional[subprocess.Popen] = None

    @property
    def available(self) -> bool:
        return shutil.which(self.ffplay_path) is not None

    def play(self, start: float, end: float) -> None:
        """Start playing one candidate's range, replacing anything already playing."""
        if not self.available:
            raise AudioPreviewError(
                f"'{self.ffplay_path}' was not found; install ffmpeg to preview audio."
            )
        self.stop()
        command = build_playback_command(
            self.audio.path, start, end, ffplay_path=self.ffplay_path
        )
        logger.debug("audio preview: %s", " ".join(command))
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        """Silence any preview in progress; safe to call when nothing is playing."""
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=5.0)
        except Exception:  # pragma: no cover - best-effort cleanup
            try:
                process.kill()
            except Exception:
                pass

    @property
    def is_playing(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def __enter__(self) -> "AudioPreviewer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()
