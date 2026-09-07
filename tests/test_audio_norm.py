"""Unit tests for audio loudness normalization filter generation."""

from freecher_worker.rendering.audio import build_loudnorm_filter


def test_build_loudnorm_filter_with_measured_params():
    """Verify exact linear normalization filter string construction."""
    measured = {
        "input_i": "-22.5",
        "input_lra": "8.4",
        "input_tp": "-3.1",
        "input_thresh": "-32.8",
        "target_offset": "1.2",
    }
    filter_str = build_loudnorm_filter(measured, target_i=-16.0, target_lra=11.0, target_tp=-1.5)

    assert "I=-16.0" in filter_str
    assert "LRA=11.0" in filter_str
    assert "TP=-1.5" in filter_str
    assert "measured_I=-22.5" in filter_str
    assert "measured_LRA=8.4" in filter_str
    assert "measured_TP=-3.1" in filter_str
    assert "linear=true" in filter_str


def test_build_loudnorm_filter_fallback_without_measured():
    """Verify fallback loudnorm filter without measured values."""
    filter_str = build_loudnorm_filter(None, target_i=-16.0, target_lra=11.0, target_tp=-1.5)
    assert filter_str == "loudnorm=I=-16.0:LRA=11.0:TP=-1.5"
