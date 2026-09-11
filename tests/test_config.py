"""Tests for configuration settings."""

import pytest
from freecher_worker.config import Settings


def test_default_settings(monkeypatch):
    import os
    for k in list(os.environ):
        if k.startswith("FREECHER_"):
            monkeypatch.delenv(k, raising=False)
    settings = Settings(_env_file=None)
    assert settings.asr_model == "small"
    assert settings.asr_device == "cuda"
    assert settings.asr_compute_type == "int8_float16"
    assert settings.highlight_min_seconds == 30.0
    assert settings.highlight_target_seconds == 60.0
    assert settings.highlight_max_seconds == 90.0
    assert settings.highlight_top_k == 5
    assert settings.scorer == "heuristic"


def test_env_override(monkeypatch):
    monkeypatch.setenv("FREECHER_ASR_MODEL", "medium")
    monkeypatch.setenv("FREECHER_ASR_DEVICE", "cpu")
    monkeypatch.setenv("FREECHER_ASR_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("FREECHER_HIGHLIGHT_TOP_K", "10")
    monkeypatch.setenv("FREECHER_SCORER", "llm")

    settings = Settings()
    assert settings.asr_model == "medium"
    assert settings.asr_device == "cpu"
    assert settings.asr_compute_type == "int8"
    assert settings.highlight_top_k == 10
    assert settings.scorer == "llm"
