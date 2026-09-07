"""Tests for JSON I/O utilities."""

import pytest
from pydantic import BaseModel
from freecher_worker.utils.json_io import load_json, save_json


class SampleModel(BaseModel):
    name: str
    count: int
    score: float


def test_save_and_load_pydantic_model(tmp_path):
    obj = SampleModel(name="test", count=42, score=9.5)
    file_path = tmp_path / "model.json"

    save_json(obj, file_path)
    assert file_path.is_file()

    loaded = load_json(file_path)
    assert loaded["name"] == "test"
    assert loaded["count"] == 42
    assert loaded["score"] == 9.5


def test_save_and_load_list(tmp_path):
    items = [
        SampleModel(name="first", count=1, score=1.0),
        SampleModel(name="second", count=2, score=2.0),
    ]
    file_path = tmp_path / "list.json"
    save_json(items, file_path)

    loaded = load_json(file_path)
    assert len(loaded) == 2
    assert loaded[0]["name"] == "first"
    assert loaded[1]["count"] == 2


def test_load_nonexistent_file(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_json(missing)
