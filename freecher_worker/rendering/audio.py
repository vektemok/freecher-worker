"""Audio loudness analysis and EBU R128 two-stage normalization."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("freecher_worker")


def measure_loudness(
    source_media: Path,
    start: float,
    duration: float,
    target_i: float = -16.0,
    target_lra: float = 11.0,
    target_tp: float = -1.5,
) -> Optional[Dict[str, str]]:
    """Run an audio-only analysis pass to extract measured EBU R128 loudness parameters."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(source_media),
        "-vn",
        "-af", f"loudnorm=I={target_i}:LRA={target_lra}:TP={target_tp}:print_format=json",
        "-f", "null",
        "-",
    ]

    try:
        res = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60.0,
            check=False,
        )

        # Parse JSON block from stderr
        match = re.search(r"\{[\s\S]*?\"input_i\"[\s\S]*?\}", res.stderr)
        if match:
            data = json.loads(match.group(0))
            return {
                "input_i": str(data.get("input_i", target_i)),
                "input_lra": str(data.get("input_lra", target_lra)),
                "input_tp": str(data.get("input_tp", target_tp)),
                "input_thresh": str(data.get("input_thresh", "-26.0")),
                "target_offset": str(data.get("target_offset", "0.0")),
            }
        logger.warning("[audio] No loudnorm JSON block found in ffmpeg stderr")
        return None
    except Exception as exc:
        logger.warning(f"[audio] Loudness analysis failed: {exc}")
        return None


def build_loudnorm_filter(
    measured: Optional[Dict[str, str]] = None,
    target_i: float = -16.0,
    target_lra: float = 11.0,
    target_tp: float = -1.5,
) -> str:
    """Build FFmpeg loudnorm audio filter string.

    Uses measured parameters for accurate, single-pass linear normalization if available.
    """
    if measured:
        return (
            f"loudnorm=I={target_i}:LRA={target_lra}:TP={target_tp}:"
            f"measured_I={measured['input_i']}:"
            f"measured_LRA={measured['input_lra']}:"
            f"measured_TP={measured['input_tp']}:"
            f"measured_thresh={measured['input_thresh']}:"
            f"offset={measured['target_offset']}:"
            "linear=true"
        )
    return f"loudnorm=I={target_i}:LRA={target_lra}:TP={target_tp}"
