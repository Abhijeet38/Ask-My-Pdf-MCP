"""Shared pytest configuration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def small_pdf(data_dir: Path) -> Path:
    """A small PDF that ships with the repo (fast to parse)."""
    candidates = [
        data_dir / "P19-1598.pdf",
        data_dir / "USCOURTS-laed-2_16-md-02740-84.pdf",
    ]
    for c in candidates:
        if c.exists():
            return c
    pytest.skip(f"No sample PDF available in {data_dir}")
