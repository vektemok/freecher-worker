"""Versioned per-stage disk cache for contextual_reranker_v1.

Each stage caches independently so a rerank never recomputes global or chapter context.
Every cache key carries the full identity of the request:

    model, reasoning_effort, temperature, prompt_version, prompt_hash,
    schema_version, context_version, reranker_version, stage payload hash
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from freecher_worker.utils.json_io import load_json, save_json

from .versions import CONTEXT_VERSION, RERANKER_VERSION

logger = logging.getLogger("freecher_worker")


def stable_hash(payload: Any, length: int = 16) -> str:
    """Deterministic hash of any JSON-serializable payload (key order independent)."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


class ContextualCache:
    """Content-addressed cache rooted at ``<run_dir>/contextual/cache``."""

    def __init__(
        self,
        run_dir: Path | str,
        model: Optional[str],
        reasoning_effort: Optional[str],
        temperature: Optional[float],
        context_version: str = CONTEXT_VERSION,
        reranker_version: str = RERANKER_VERSION,
        force: bool = False,
    ) -> None:
        self.root = Path(run_dir).resolve() / "contextual" / "cache"
        self.model = model or "unknown"
        self.reasoning_effort = (
            str(reasoning_effort).lower() if reasoning_effort is not None else "none"
        )
        self.temperature = temperature
        self.context_version = context_version
        self.reranker_version = reranker_version
        self.force = force

    def request_hash(
        self,
        stage: str,
        prompt_version: str,
        prompt_hash: str,
        schema_version: str,
        payload: Any,
    ) -> str:
        """Deterministic identity of one cached stage request."""
        temp_str = f"{self.temperature:.4f}" if self.temperature is not None else "none"
        key = {
            "stage": stage,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "temperature": temp_str,
            "prompt_version": prompt_version,
            "prompt_hash": prompt_hash,
            "schema_version": schema_version,
            "context_version": self.context_version,
            "reranker_version": self.reranker_version,
            "payload": payload,
        }
        return stable_hash(key)

    def path_for(self, stage: str, request_hash: str) -> Path:
        return self.root / stage / f"{request_hash}.json"

    def load(self, stage: str, request_hash: str) -> Optional[Dict[str, Any]]:
        """Return the cached payload, or None on a miss, corrupt entry, or ``force``."""
        if self.force:
            return None
        path = self.path_for(stage, request_hash)
        if not path.is_file():
            return None
        try:
            data = load_json(path)
        except Exception as exc:  # noqa: BLE001 - a corrupt cache entry is a miss, never fatal
            logger.warning(f"[contextual-cache] Corrupt cache entry {path}: {exc}")
            return None
        if not isinstance(data, dict):
            logger.warning(f"[contextual-cache] Unexpected cache payload type in {path}")
            return None
        return data

    def store(self, stage: str, request_hash: str, payload: Any) -> Path:
        path = self.path_for(stage, request_hash)
        save_json(payload, path)
        return path
