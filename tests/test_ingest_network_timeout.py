"""Regression tests for bounded network I/O and stream watchdog.

Verifies:
1. When an upstream source stalls indefinitely, the process-level watchdog triggers
   and terminates the process group within the configured stall_timeout.
2. Normal EOF terminates cleanly without infinite reconnect loops.
"""
from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from freecher_worker.ingest.source import (
    SourceStream,
    SourceStreamError,
    open_source_stream,
)


def test_watchdog_terminates_stalled_stream(tmp_path):
    """When reader waits on a hung process, watchdog terminates it within stall_timeout."""
    # Run a python child process that prints 10 bytes and then hangs (sleeps forever)
    script = (
        "import sys, time; "
        "sys.stdout.buffer.write(b'1234567890'); "
        "sys.stdout.buffer.flush(); "
        "time.sleep(300)"
    )
    cmd = [sys.executable, "-c", script]

    import subprocess
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    # Configure small stall_timeout: 0.8s
    stream = SourceStream(proc, cmd, stall_timeout=0.8)
    try:
        # First read gets initial 10 bytes immediately
        first_chunk = stream.stdout.read(10)
        assert first_chunk == b"1234567890"

        # Second read blocks waiting for more bytes from the hanging process.
        # Watchdog should kill it after 0.8s.
        t0 = time.perf_counter()
        second_chunk = stream.stdout.read(1024)
        elapsed = time.perf_counter() - t0

        assert elapsed < 3.0, f"Expected watchdog termination under 3.0s, took {elapsed:.2f}s"
        assert stream.stalled is True

        with pytest.raises(SourceStreamError, match="stalled"):
            stream.wait_and_check(timeout=2.0)
    finally:
        stream.terminate()


def test_normal_eof_terminates_cleanly_without_reconnect_loop():
    """Normal source EOF terminates cleanly without hanging or reconnect looping."""
    script = (
        "import sys; "
        "sys.stdout.buffer.write(b'hello world'); "
        "sys.stdout.buffer.flush(); "
        "sys.exit(0)"
    )
    cmd = [sys.executable, "-c", script]

    import subprocess
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    stream = SourceStream(proc, cmd, stall_timeout=2.0)
    try:
        data = stream.stdout.read(1024)
        assert data == b"hello world"
        eof = stream.stdout.read(1024)
        assert eof == b""
        # Must exit 0 and not be marked stalled
        stream.wait_and_check(timeout=2.0)
        assert stream.stalled is False
    finally:
        stream.terminate()
