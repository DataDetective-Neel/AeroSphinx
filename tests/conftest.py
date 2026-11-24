"""Shared fixtures for the AeroSphinx test suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aerosphinx.ingestion.cpcb import CPCBRules


@pytest.fixture
def rules() -> CPCBRules:
    """Default quality rules."""
    return CPCBRules()


def write_cpcb_csv(path: Path, rows: list[tuple[str, float | None]]) -> Path:
    """Write a minimal CPCB-style CSV.

    Uses the real export header so that column detection is exercised
    against the production format rather than a simplified one.
    """
    frame = pd.DataFrame(rows, columns=["Timestamp", "PM2.5 (µg/m³)"])
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def cpcb_factory(tmp_path: Path):
    """Factory that writes CPCB CSVs into a temporary directory."""

    def _make(name: str, rows: list[tuple[str, float | None]]) -> Path:
        directory = tmp_path / "cpcb"
        directory.mkdir(exist_ok=True)
        return write_cpcb_csv(directory / name, rows)

    return _make


@pytest.fixture
def hourly_day() -> list[tuple[str, float]]:
    """A complete 24-hour local day of valid PM2.5 readings."""
    return [(f"2024-01-01 {hour:02d}:00:00", 50.0 + hour) for hour in range(24)]
