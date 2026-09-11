"""The declared dependencies must match what the package actually needs.

boto3 and yt-dlp once lived in requirements.txt only, so `pip install -e .`
produced an environment where ingest failed at import time. These tests pin
the invariant that made that possible.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "freecher_worker"

# Import name -> distribution name, where the two differ.
DISTRIBUTION_FOR_MODULE = {
    "cv2": "opencv-python-headless",
    "faster_whisper": "faster-whisper",
    "pydantic_settings": "pydantic-settings",
    "dotenv": "python-dotenv",
    "jwt": "pyjwt",
}

# Modules that arrive as a hard dependency of a declared distribution and are
# only ever reached through it.
VENDORED_BY_A_DECLARED_DEPENDENCY = {
    "botocore",     # pinned by boto3
    "ctranslate2",  # pinned by faster-whisper
}


@pytest.fixture(scope="module")
def declared() -> set[str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    names = set()
    for requirement in pyproject["project"]["dependencies"]:
        names.add(re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].lower())
    return names


def _imported_top_level_modules() -> set[str]:
    """Every top-level module the package imports, read from the AST."""
    modules: set[str] = set()
    for source in PACKAGE.rglob("*.py"):
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def test_ingest_dependencies_are_declared_for_pip_install_e(declared):
    # The two that were missing: boto3 is imported, yt-dlp is executed from the
    # interpreter's own environment, so both have to be installed by the project.
    assert "boto3" in declared
    assert "yt-dlp" in declared


def test_every_directly_imported_third_party_module_is_declared(declared):
    third_party = {
        module
        for module in _imported_top_level_modules()
        if module not in sys.stdlib_module_names
        and module not in {"freecher_worker", "__future__"}
        and module not in VENDORED_BY_A_DECLARED_DEPENDENCY
    }

    undeclared = {
        module
        for module in third_party
        if DISTRIBUTION_FOR_MODULE.get(module, module).lower() not in declared
    }
    assert undeclared == set(), (
        f"imported but not declared in pyproject.toml: {sorted(undeclared)}"
    )


def test_requirements_txt_installs_the_project_rather_than_repeating_it():
    # Two hand-maintained lists are what let the environments drift apart.
    lines = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines == ["-e .[dev]"]
