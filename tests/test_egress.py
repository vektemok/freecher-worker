"""Egress selection and the difference between "no such video" and "not you".

A site that refuses this host's address produces a failure that looks like any
other ingest error, but has a different cause, a different remedy and different
retry semantics. Getting that distinction wrong means a job churns the queue
forever against a wall, and an operator reads "could not read metadata" and goes
looking for a bug in the URL handling.
"""
from __future__ import annotations

import pytest

from freecher_worker.config import Settings
from freecher_worker.ingest import source as S
from freecher_worker.jobs.orchestrator import is_retryable

URL = "https://www.youtube.com/watch?v=aqz-KE-bpKQ"


# ------------------------------------------------------------ egress selection
def test_no_proxy_is_configured_by_default():
    assert S.resolve_proxy(settings=Settings()) is None


def test_the_configured_proxy_is_used():
    assert S.resolve_proxy(settings=Settings(
        ingest_proxy="socks5://10.0.0.1:1080")) == "socks5://10.0.0.1:1080"


def test_an_explicit_proxy_beats_the_configuration():
    assert S.resolve_proxy("http://explicit:3128",
                           settings=Settings(ingest_proxy="socks5://configured:1080")) \
        == "http://explicit:3128"


def test_an_empty_configured_proxy_means_direct():
    assert S.resolve_proxy(settings=Settings(ingest_proxy="")) is None


def test_the_transfer_uses_the_proxy():
    cmd = S.build_stream_command(URL, format_selector="b",
                                 proxy="socks5://10.0.0.1:1080")
    assert "--proxy" in cmd and cmd[cmd.index("--proxy") + 1] == "socks5://10.0.0.1:1080"


def test_the_transfer_is_direct_when_nothing_is_configured(monkeypatch):
    monkeypatch.setattr(S, "resolve_proxy", lambda *a, **k: None)
    assert "--proxy" not in S.build_stream_command(URL, format_selector="b")


def test_probe_and_transfer_share_one_egress(monkeypatch):
    """Resolving metadata from one address and fetching media from another is
    both unreproducible and exactly what makes a site distrust the session."""
    seen: list[list[str]] = []

    class Result:
        returncode = 1
        stdout = b""
        stderr = b"ERROR: nope"

    monkeypatch.setattr(S.subprocess, "run", lambda cmd, **kw: (seen.append(cmd), Result())[1])
    with pytest.raises(S.SourceStreamError):
        S.probe_source(URL, proxy="http://egress:3128")
    transfer = S.build_stream_command(URL, format_selector="b", proxy="http://egress:3128")

    probe = seen[0]
    assert probe[probe.index("--proxy") + 1] == transfer[transfer.index("--proxy") + 1]


# -------------------------------------------------------- failure classification
@pytest.mark.parametrize("stderr", [
    "ERROR: [youtube] xyz: Sign in to confirm you're not a bot. Use --cookies",
    "ERROR: [youtube] xyz: Sign in to confirm you’re not a bot.",
    "ERROR: Unable to download: HTTP Error 429: Too Many Requests",
    "ERROR: The uploader has not made this video available in your country",
])
def test_a_refusal_is_reported_as_blocked(stderr):
    error = S._classify_failure(URL, stderr)
    assert isinstance(error, S.SourceBlockedError)
    assert "egress" in str(error).lower()


@pytest.mark.parametrize("stderr", [
    "ERROR: [generic] Unable to extract video data",
    "ERROR: Video unavailable. This video has been removed by the uploader",
    "ERROR: Unsupported URL: https://example.com/",
])
def test_an_ordinary_failure_is_not_reported_as_blocked(stderr):
    error = S._classify_failure(URL, stderr)
    assert isinstance(error, S.SourceStreamError)
    assert not isinstance(error, S.SourceBlockedError)


def test_a_blocked_source_is_still_a_source_stream_error():
    """Existing handlers catch SourceStreamError; they must keep working."""
    assert issubclass(S.SourceBlockedError, S.SourceStreamError)


def test_a_blocked_source_is_never_retried_automatically():
    assert is_retryable(S.SourceBlockedError("refused")) is False


def test_the_blocked_message_names_the_setting_that_fixes_it():
    message = str(S._classify_failure(URL, "Sign in to confirm you're not a bot"))
    assert "FREECHER_INGEST_PROXY" in message
    assert URL in message
