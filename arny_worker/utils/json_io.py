"""JSON input/output helper functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel


def _json_serial_fallback(obj: Any) -> Any:
    """Fallback serializer for paths and other non-standard types."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data: Any, file_path: Path | str, indent: int = 2) -> None:
    """Save data to a JSON file atomically, creating parent directories if needed.

    Supports dicts, lists, Pydantic models, Paths.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dumpable: Any
    if isinstance(data, BaseModel):
        dumpable = data.model_dump(mode="json")
    elif isinstance(data, list):
        dumpable = [item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in data]
    else:
        dumpable = data

    # Atomic write via temp file in same directory
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(dumpable, f, indent=indent, ensure_ascii=False, default=_json_serial_fallback)
        f.flush()
    temp_path.replace(path)


def load_json(file_path: Path | str) -> Any:
    """Load JSON data from a file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
