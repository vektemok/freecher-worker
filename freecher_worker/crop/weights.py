"""Resolution and caching of local CV model weights.

Weights are looked up on disk first and only downloaded when nothing is found, into a cache
directory the user can pre-populate offline. Every download is checksum-verified.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger("freecher_worker")

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_SIZE_BYTES = 232589


class ModelWeightsError(RuntimeError):
    """Raised when required model weights cannot be resolved."""


def default_cache_dir() -> Path:
    """Directory used for downloaded weights, overridable with FREECHER_MODELS_DIR."""
    env = os.environ.get("FREECHER_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "freecher-worker" / "models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _search_paths(filename: str) -> list[Path]:
    paths: list[Path] = [default_cache_dir() / filename]
    paths.append(Path(__file__).resolve().parents[2] / "models" / filename)
    paths.append(Path.cwd() / "models" / filename)
    return paths


def resolve_model_weights(
    filename: str,
    url: str,
    sha256: str,
    explicit_path: Optional[Path] = None,
    allow_download: bool = True,
    timeout: float = 120.0,
) -> Path:
    """Find model weights on disk, downloading and verifying them only if necessary.

    Raises:
        ModelWeightsError: if the weights are absent and cannot be fetched or verified.
    """
    if explicit_path is not None:
        # An explicit path is a directive, not a hint: honour it or fail, never silently
        # substitute a different copy of the weights.
        explicit = Path(explicit_path).expanduser()
        for candidate in (explicit, explicit / filename):
            if candidate.is_file():
                logger.debug(f"[weights] Using {filename} from configured path {candidate}")
                return candidate
        raise ModelWeightsError(
            f"{filename} not found at the configured path {explicit}"
        )

    for candidate in _search_paths(filename):
        if candidate.is_file():
            logger.debug(f"[weights] Using {filename} from {candidate}")
            return candidate

    if not allow_download:
        raise ModelWeightsError(
            f"{filename} not found and downloads are disabled. Place it in "
            f"{default_cache_dir()} or set FREECHER_MODELS_DIR."
        )

    target_dir = default_cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    logger.info(f"[weights] Downloading {filename} to {target}")

    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(target_dir), suffix=".part")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp_path, "wb") as out:
            shutil.copyfileobj(response, out)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise ModelWeightsError(
            f"Could not download {filename} from {url}: {exc}. Download it manually into "
            f"{target_dir} (or set FREECHER_MODELS_DIR) to run fully offline."
        ) from exc

    actual = _sha256(tmp_path)
    if actual != sha256:
        tmp_path.unlink(missing_ok=True)
        raise ModelWeightsError(
            f"Checksum mismatch for {filename}: expected {sha256}, got {actual}"
        )

    tmp_path.replace(target)
    logger.info(f"[weights] Verified and cached {filename} ({target.stat().st_size} bytes)")
    return target


def resolve_yunet_weights(
    explicit_path: Optional[Path] = None,
    allow_download: bool = True,
) -> Path:
    """Resolve the YuNet face detection ONNX weights."""
    return resolve_model_weights(
        filename=YUNET_FILENAME,
        url=YUNET_URL,
        sha256=YUNET_SHA256,
        explicit_path=explicit_path,
        allow_download=allow_download,
    )
