"""yt-dlp source streams: probe metadata and expose video bytes on a pipe.

Nothing here is YouTube-specific — any site yt-dlp supports (Twitch, Vimeo,
RuTube, VK, a bare .mp4 URL) goes through the same two calls.
"""

from __future__ import annotations

import collections
import json
import logging
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from freecher_worker.ingest.models import (
    AVAILABLE_QUALITY_MODES,
    QUALITY_BEST,
    QUALITY_PROGRESSIVE,
    VideoInfo,
)

logger = logging.getLogger("freecher_worker")

# Remuxed output is assembled by ffmpeg straight into the pipe. mp4 normally
# needs a seekable target for its moov atom, so the fragmented flags are what
# make an unseekable stdout target legal. The trailing -f mp4 also overrides the
# MPEG-TS default yt-dlp picks for a stdout target.
FRAGMENTED_MP4_ARGS = "-f mp4 -movflags frag_keyframe+empty_moov+default_base_moof"

# HLS hands over AAC as ADTS frames; the mp4 muxer only accepts ASC.
ADTS_TO_ASC_ARGS = "-bsf:a aac_adtstoasc"

STDERR_TAIL_LINES = 40


class SourceStreamError(Exception):
    """Raised when yt-dlp cannot resolve or deliver the requested video."""


def resolve_ytdlp_path(explicit: Optional[str] = None) -> str:
    """Locate the yt-dlp executable, preferring the current interpreter's venv."""
    if explicit:
        return explicit
    candidate = Path(sys.executable).with_name("yt-dlp")
    if candidate.exists():
        return str(candidate)
    found = shutil.which("yt-dlp")
    if not found:
        raise SourceStreamError(
            "yt-dlp executable not found. Install it with: pip install yt-dlp"
        )
    return found


def build_format_selector(quality: str, max_height: Optional[int] = None) -> str:
    """Return the yt-dlp -f selector for a quality mode."""
    if quality not in AVAILABLE_QUALITY_MODES:
        raise ValueError(
            f"unknown quality mode '{quality}'; available: {', '.join(AVAILABLE_QUALITY_MODES)}"
        )
    height = f"[height<={max_height}]" if max_height else ""

    if quality == QUALITY_PROGRESSIVE:
        # A single already-muxed stream. On sites that only serve HLS (Twitch)
        # this still resolves, and the probe then flags it for remuxing.
        candidates = [f"b[ext=mp4]{height}", f"b{height}", "b"]
    else:
        # Prefer H.264 + AAC so the object stays friendly to the ffmpeg/OpenCV
        # stages downstream, falling back to whatever the best pair is.
        candidates = [
            f"bv*[ext=mp4][vcodec^=avc1]{height}+ba[ext=m4a]",
            f"bv*[ext=mp4]{height}+ba",
            f"bv*{height}+ba",
            f"b[ext=mp4]{height}",
            f"b{height}",
            "b",
        ]

    return "/".join(dict.fromkeys(candidates))


def _auth_args(
    cookies_from_browser: Optional[str],
    cookies_file: Optional[str],
) -> list[str]:
    args: list[str] = []
    if cookies_from_browser:
        args += ["--cookies-from-browser", cookies_from_browser]
    if cookies_file:
        args += ["--cookies", cookies_file]
    return args


def build_stream_command(
    url: str,
    *,
    format_selector: str,
    remux: bool = True,
    adts_to_asc: bool = False,
    ytdlp_path: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> list[str]:
    """Build the yt-dlp command that writes the chosen video to stdout."""
    command = [
        resolve_ytdlp_path(ytdlp_path),
        "--no-playlist",
        "--no-progress",
        "--no-warnings",
        "-f",
        format_selector,
    ]

    if remux:
        # Let ffmpeg pull every track and assemble them into the pipe. Left to
        # itself yt-dlp would concatenate the tracks back to back, or emit raw
        # MPEG-TS segments, neither of which is a usable mp4.
        output_args = FRAGMENTED_MP4_ARGS
        if adts_to_asc:
            output_args = f"{output_args} {ADTS_TO_ASC_ARGS}"
        command += [
            "--downloader",
            "ffmpeg",
            "--downloader-args",
            f"ffmpeg_o:{output_args}",
        ]

    command += _auth_args(cookies_from_browser, cookies_file)
    if extra_args:
        command += list(extra_args)

    command += ["-o", "-", url]
    return command


def probe_source(
    url: str,
    *,
    format_selector: Optional[str] = None,
    ytdlp_path: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    timeout: float = 120.0,
) -> VideoInfo:
    """Resolve title/duration/protocol for a video without downloading it.

    Passing the same selector the transfer will use makes the reported protocol
    and size describe the formats actually chosen, which is what decides whether
    ffmpeg has to sit in the middle.
    """
    command = [resolve_ytdlp_path(ytdlp_path), "-J", "--no-playlist", "--no-warnings"]
    if format_selector:
        command += ["-f", format_selector]
    command += _auth_args(cookies_from_browser, cookies_file)
    if extra_args:
        command += list(extra_args)
    command.append(url)

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
        raise SourceStreamError(f"yt-dlp metadata probe timed out after {timeout:.0f}s") from exc

    if result.returncode != 0:
        raise SourceStreamError(
            "yt-dlp could not read the video metadata:\n"
            + result.stderr.decode("utf-8", errors="ignore").strip()
        )

    try:
        payload: dict[str, Any] = json.loads(result.stdout.decode("utf-8", errors="ignore"))
    except json.JSONDecodeError as exc:
        raise SourceStreamError("yt-dlp returned malformed metadata JSON") from exc

    return video_info_from_payload(payload)


def video_info_from_payload(payload: dict[str, Any]) -> VideoInfo:
    """Map a yt-dlp info dict onto VideoInfo, including the selected formats."""
    requested = payload.get("requested_formats") or []
    if requested:
        format_ids = [str(fmt.get("format_id")) for fmt in requested]
        protocol = "+".join(str(fmt.get("protocol") or "") for fmt in requested)
        filesize = sum(
            fmt.get("filesize") or fmt.get("filesize_approx") or 0 for fmt in requested
        ) or None
        acodec = next(
            (fmt.get("acodec") for fmt in requested if fmt.get("acodec") not in (None, "none")),
            None,
        )
    else:
        format_ids = [str(payload["format_id"])] if payload.get("format_id") else []
        protocol = payload.get("protocol")
        filesize = payload.get("filesize") or payload.get("filesize_approx")
        acodec = payload.get("acodec")

    return VideoInfo(
        video_id=payload.get("id") or "unknown",
        title=payload.get("title") or "unknown",
        duration_seconds=payload.get("duration"),
        uploader=payload.get("uploader") or payload.get("channel"),
        width=payload.get("width"),
        height=payload.get("height"),
        filesize_approx=filesize,
        webpage_url=payload.get("webpage_url"),
        extractor=payload.get("extractor_key") or payload.get("extractor"),
        protocol=protocol,
        acodec=acodec,
        requested_format_ids=format_ids,
        is_live=bool(payload.get("is_live")),
    )


class SourceStream:
    """A running yt-dlp process whose stdout carries the video bytes."""

    def __init__(self, process: subprocess.Popen, command: Sequence[str]) -> None:
        self.process = process
        self.command = list(command)
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=STDERR_TAIL_LINES)
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="yt-dlp-stderr", daemon=True
        )
        self._stderr_thread.start()

    @property
    def stdout(self):
        assert self.process.stdout is not None
        return self.process.stdout

    def _drain_stderr(self) -> None:
        """Keep the stderr pipe empty so a chatty yt-dlp can never deadlock."""
        stream = self.process.stderr
        if stream is None:
            return
        for raw in stream:
            line = raw.decode("utf-8", errors="ignore").rstrip()
            if line:
                self._stderr_tail.append(line)
                logger.debug("yt-dlp: %s", line)

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    def wait_and_check(self, timeout: float = 60.0) -> None:
        """Wait for yt-dlp to exit and raise if it failed."""
        return_code = self.process.wait(timeout=timeout)
        self._stderr_thread.join(timeout=5.0)
        if return_code != 0:
            raise SourceStreamError(
                f"yt-dlp exited with code {return_code}:\n{self.stderr_tail}"
            )

    def terminate(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        # The drain thread is still iterating the stderr pipe; closing it under
        # the thread raises there and can swallow the real failure message.
        self._stderr_thread.join(timeout=5.0)
        for stream in (self.process.stdout, self.process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass


@contextmanager
def open_source_stream(
    url: str,
    *,
    format_selector: str,
    remux: bool = True,
    adts_to_asc: bool = False,
    ytdlp_path: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    pipe_buffer_size: int = 1024 * 1024,
) -> Iterator[SourceStream]:
    """Start yt-dlp and yield its output stream, killing it on any failure."""
    command = build_stream_command(
        url,
        format_selector=format_selector,
        remux=remux,
        adts_to_asc=adts_to_asc,
        ytdlp_path=ytdlp_path,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
        extra_args=extra_args,
    )
    logger.info("starting yt-dlp: %s", " ".join(command))

    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=pipe_buffer_size,
    )
    stream = SourceStream(process, command)
    try:
        yield stream
    finally:
        # Kills yt-dlp if it is still alive (an aborted upload) and closes the
        # pipes either way.
        stream.terminate()
