"""Transcription-ready audio, split off the ingest stream as it flies past.

The video bytes yt-dlp produces are already passing through this process on
their way to R2, so the audio artifact is cut from that same stream rather than
fetched a second time: a `TeeReader` mirrors every chunk the uploader consumes
into an ffmpeg process that keeps only the audio track.

Only the audio is staged on disk (roughly 0.5 MB per minute at the defaults),
never the video. Staging it — instead of piping ffmpeg straight into a second
multipart upload — buys a seekable, non-fragmented m4a whose real codec,
sample rate, channels and duration can be read back with ffprobe before the
object is written.
"""

from __future__ import annotations

import collections
import io
import json
import logging
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Optional

logger = logging.getLogger("freecher_worker")

# Speech, not music: mono at 16 kHz is what every ASR frontend resamples to
# anyway, and AAC-LC at 64 kb/s is transparent for voice at that rate.
DEFAULT_AUDIO_CODEC = "aac"
DEFAULT_AUDIO_SAMPLE_RATE = 16_000
DEFAULT_AUDIO_CHANNELS = 1
DEFAULT_AUDIO_BITRATE = "64k"

# ffmpeg exits 0 even after "Error during demuxing" or a truncated input, so
# the length of the result is the only trustworthy completeness check.
DURATION_TOLERANCE_SECONDS = 2.0
DURATION_TOLERANCE_RATIO = 0.01

# How far the audio branch may fall behind the uploader before the reader is
# made to wait for it. Bounds memory; the tee never drops a byte.
DEFAULT_TEE_BUFFER_BYTES = 128 * 1024 * 1024

STDERR_TAIL_LINES = 40


class AudioArtifactError(Exception):
    """Raised when the audio artifact cannot be produced or verified."""


@dataclass(frozen=True)
class AudioEncodeSettings:
    """Encoder settings for the transcription copy of the source audio."""

    codec: str = DEFAULT_AUDIO_CODEC
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
    channels: int = DEFAULT_AUDIO_CHANNELS
    bitrate: str = DEFAULT_AUDIO_BITRATE

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels < 1:
            raise ValueError(f"channels must be at least 1, got {self.channels}")

    @property
    def container_format(self) -> str:
        """ffmpeg muxer name; 'ipod' is the mp4 muxer's audio-only .m4a profile."""
        return "ipod"

    @property
    def content_type(self) -> str:
        return "audio/mp4"


@dataclass
class AudioArtifact:
    """A produced (or deliberately skipped) audio object in R2."""

    bucket: str
    key: str
    source_key: str
    size_bytes: int
    codec: str
    sample_rate: int
    channels: int
    source_duration_seconds: Optional[float] = None
    audio_duration_seconds: Optional[float] = None
    part_count: int = 0
    etag: Optional[str] = None
    skipped: bool = False

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    @property
    def megabytes(self) -> float:
        return self.size_bytes / 1024 / 1024


def build_audio_command(
    destination: Path | str,
    settings: AudioEncodeSettings,
    *,
    ffmpeg_path: str = "ffmpeg",
) -> list[str]:
    """Build the ffmpeg command that turns piped video into a speech m4a.

    `aresample=async=1:first_pts=0` is what keeps the artifact on the source
    timeline: a gap in the source is filled with silence instead of pulling the
    rest of the audio earlier, and an audio track that starts late is padded to
    zero rather than shifted, so a timestamp in the artifact is the same
    timestamp in `source.mp4`.
    """
    return [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        # Demux faults on a pipe are silent otherwise; this turns them into a
        # non-zero exit instead of a quietly truncated artifact.
        "-xerror",
        "-y",
        "-i", "pipe:0",
        # Video, subtitles and data tracks are dropped without being decoded.
        "-vn", "-sn", "-dn",
        "-map", "0:a:0",
        "-af", "aresample=async=1:first_pts=0",
        "-ac", str(settings.channels),
        "-ar", str(settings.sample_rate),
        "-c:a", settings.codec,
        "-b:a", settings.bitrate,
        "-f", settings.container_format,
        str(destination),
    ]


def probe_audio_file(
    path: Path | str,
    *,
    ffprobe_path: str = "ffprobe",
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Read back codec/sample rate/channels/duration from a produced artifact."""
    command = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "format=duration:stream=codec_name,sample_rate,channels",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioArtifactError(f"ffprobe timed out after {timeout:.0f}s") from exc
    except FileNotFoundError as exc:
        raise AudioArtifactError(
            f"ffprobe executable not found at '{ffprobe_path}'; install ffmpeg."
        ) from exc

    if result.returncode != 0:
        raise AudioArtifactError(
            "ffprobe could not read the extracted audio:\n"
            + result.stderr.decode("utf-8", errors="ignore").strip()
        )

    try:
        payload = json.loads(result.stdout.decode("utf-8", errors="ignore"))
    except json.JSONDecodeError as exc:
        raise AudioArtifactError("ffprobe returned malformed JSON") from exc

    streams = payload.get("streams") or []
    if not streams:
        raise AudioArtifactError("the extracted artifact has no audio stream")
    stream = streams[0]

    duration: Optional[float] = None
    raw_duration = (payload.get("format") or {}).get("duration")
    if raw_duration is not None:
        try:
            duration = float(raw_duration)
        except (TypeError, ValueError):
            duration = None

    return {
        "codec": str(stream.get("codec_name") or "unknown"),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "duration_seconds": duration,
    }


def duration_mismatch(
    source_seconds: Optional[float],
    audio_seconds: Optional[float],
) -> Optional[str]:
    """Return a complaint when the artifact does not span the whole source."""
    if not source_seconds or source_seconds <= 0:
        return None
    if audio_seconds is None:
        return "the extracted audio has no readable duration"
    tolerance = max(DURATION_TOLERANCE_SECONDS, source_seconds * DURATION_TOLERANCE_RATIO)
    drift = abs(audio_seconds - source_seconds)
    if drift <= tolerance:
        return None
    return (
        f"extracted audio is {audio_seconds:.1f}s but the source is "
        f"{source_seconds:.1f}s ({drift:.1f}s off, tolerance {tolerance:.1f}s); "
        "the artifact would not line up with the source timeline"
    )


class AudioSidecar:
    """An ffmpeg process fed video bytes, writing a staged audio file."""

    def __init__(
        self,
        destination: Path | str,
        settings: AudioEncodeSettings,
        *,
        ffmpeg_path: str = "ffmpeg",
        pipe_buffer_size: int = 1024 * 1024,
    ) -> None:
        self.destination = Path(destination)
        self.settings = settings
        self.command = build_audio_command(
            self.destination, settings, ffmpeg_path=ffmpeg_path
        )
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=STDERR_TAIL_LINES)

        logger.info("starting audio sidecar: %s", " ".join(self.command))
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=pipe_buffer_size,
            )
        except FileNotFoundError as exc:
            raise AudioArtifactError(
                f"ffmpeg executable not found at '{self.command[0]}'; install ffmpeg."
            ) from exc

        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="audio-stderr", daemon=True
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        """Keep the stderr pipe empty so a chatty ffmpeg can never deadlock."""
        stream = self.process.stderr
        if stream is None:
            return
        for raw in stream:
            line = raw.decode("utf-8", errors="ignore").rstrip()
            if line:
                self._stderr_tail.append(line)
                logger.debug("audio ffmpeg: %s", line)

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    def write(self, chunk: bytes) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(chunk)

    def close_input(self) -> None:
        """Signal end of stream; ffmpeg then finalises the container.

        Idempotent, and deliberately deaf to every failure: an abort closes
        this from one thread while the mirror may still be writing from
        another, and the real error is always the one that caused the abort.
        """
        stdin = self.process.stdin
        if stdin is None or stdin.closed:
            return
        for step in (stdin.flush, stdin.close):
            try:
                step()
            except Exception:
                pass

    def wait(self, timeout: float = 600.0) -> None:
        """Wait for ffmpeg to finish and raise if it failed."""
        self.close_input()
        try:
            return_code = self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            self.terminate()
            raise AudioArtifactError(
                f"audio extraction did not finish within {timeout:.0f}s"
            ) from exc
        self._stderr_thread.join(timeout=5.0)
        if return_code != 0:
            raise AudioArtifactError(
                f"ffmpeg audio extraction exited with code {return_code}:\n{self.stderr_tail}"
            )
        if not self.destination.is_file() or self.destination.stat().st_size == 0:
            raise AudioArtifactError(
                "ffmpeg reported success but produced no audio data"
            )

    def terminate(self) -> None:
        """Kill ffmpeg and close its pipes; safe to call more than once."""
        self.close_input()
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        self._stderr_thread.join(timeout=5.0)
        if self.process.stderr is not None:
            try:
                self.process.stderr.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass


class TeeReader(io.RawIOBase):
    """A read-through wrapper that mirrors every byte into a sink.

    The mirror runs on its own thread behind a byte-bounded queue, so a slow
    sink throttles the reader instead of buffering without limit — and a sink
    that dies is dropped silently: the primary read path keeps working, and the
    failure is surfaced afterwards through `error`.
    """

    def __init__(
        self,
        source: BinaryIO,
        sink: AudioSidecar,
        *,
        buffer_bytes: int = DEFAULT_TEE_BUFFER_BYTES,
    ) -> None:
        self._source = source
        self._sink = sink
        self._buffer_bytes = max(1, buffer_bytes)
        self._pending: collections.deque[bytes] = collections.deque()
        self._pending_bytes = 0
        self._input_closed = False
        self._condition = threading.Condition()
        self.error: Optional[BaseException] = None
        self.mirrored_bytes = 0
        self._thread = threading.Thread(target=self._drain, name="audio-tee", daemon=True)
        self._thread.start()

    # -- reader side -----------------------------------------------------
    def readable(self) -> bool:  # pragma: no cover - trivial
        return True

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        chunk = self._source.read(size)
        if chunk:
            self._offer(chunk)
        else:
            # EOF: let ffmpeg start finalising while the last parts upload.
            self.finish_input()
        return chunk

    def _offer(self, chunk: bytes) -> None:
        with self._condition:
            if self.error is not None:
                return
            # A chunk larger than the whole budget still goes through, as long
            # as it is the only one in flight; otherwise this would deadlock.
            while (
                self._pending
                and self._pending_bytes + len(chunk) > self._buffer_bytes
                and self.error is None
            ):
                self._condition.wait()
            if self.error is not None:
                return
            self._pending.append(chunk)
            self._pending_bytes += len(chunk)
            self._condition.notify_all()

    def finish_input(self) -> None:
        """Tell the mirror no more bytes are coming."""
        with self._condition:
            self._input_closed = True
            self._condition.notify_all()

    # -- mirror side -----------------------------------------------------
    def _drain(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._input_closed and self.error is None:
                    self._condition.wait()
                if self.error is not None:
                    return
                if not self._pending:
                    # Input closed and everything written.
                    self._sink.close_input()
                    return
                chunk = self._pending.popleft()
                self._pending_bytes -= len(chunk)
                self._condition.notify_all()
            try:
                self._sink.write(chunk)
            except BaseException as exc:  # ffmpeg died: never take the video down with it
                logger.warning("audio sidecar stopped accepting data: %s", exc)
                with self._condition:
                    self.error = exc
                    self._pending.clear()
                    self._pending_bytes = 0
                    self._condition.notify_all()
                return
            self.mirrored_bytes += len(chunk)

    def wait_for_mirror(self, timeout: float = 300.0) -> None:
        """Block until every mirrored byte has reached the sink."""
        self.finish_input()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise AudioArtifactError(
                f"the audio mirror did not drain within {timeout:.0f}s"
            )
        if self.error is not None:
            raise AudioArtifactError(
                f"audio extraction stopped consuming the stream: {self.error}"
            ) from self.error

    def abandon(self) -> None:
        """Drop the mirror without waiting; used when the transfer failed."""
        with self._condition:
            self.error = self.error or RuntimeError("ingest aborted")
            self._pending.clear()
            self._pending_bytes = 0
            self._input_closed = True
            self._condition.notify_all()
