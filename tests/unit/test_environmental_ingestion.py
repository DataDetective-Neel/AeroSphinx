"""Tests for the station-local meteorological and MERRA-2 ingestion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerosphinx.ingestion.merra import COLUMN_NAMES as MERRA_COLUMNS
from aerosphinx.ingestion.merra import load_merra_daily
from aerosphinx.ingestion.station_met import COLUMN_NAMES, load_station_met_daily

pytestmark = pytest.mark.unit


@pytest.fixture
def synthetic_cpcb(tmp_path):
    """A synthetic CPCB export with hourly meteorological columns."""
    directory = tmp_path / "cpcb"
    directory.mkdir()
    dates = pd.date_range("2024-01-01", "2024-06-30", freq="h")
    hours = dates.hour.to_numpy()
    rows = pd.DataFrame(
        {
            "Timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "PM2.5 (µg/m³)": 80.0,
            "AT (°C)": 22 + 8 * np.sin((hours - 9) * 2 * np.pi / 24),
            "RH (%)": 60 - 15 * np.sin((hours - 9) * 2 * np.pi / 24),
            "WS (m/s)": 1.0 + 0.5 * np.abs(np.sin(hours * 2 * np.pi / 24)),
            "WD (deg)": 90.0,
            "RF (mm)": 0.0,
            "TOT-RF (mm)": 0.0,
            "SR (W/mt2)": np.clip(400 * np.sin((hours - 6) * 2 * np.pi / 24), 0, None),
            "BP (mmHg)": 750.0,
            "VWS (m/s)": -1.0,
        }
    )
    rows.to_csv(
        directory / "City_wise_raw_data_1Hr_2024_Faridabad_1Hr.csv", index=False
    )
    return directory


def test_station_met_produces_a_daily_frame(synthetic_cpcb):
    """The hourly columns must be aggregated to a daily frame."""
    result = load_station_met_daily(synthetic_cpcb)
    assert not result.frame.empty
    for name in COLUMN_NAMES.values():
        if name == "TOT_RF":
            continue
        assert name in result.frame.columns


def test_station_met_is_retrospective_only(synthetic_cpcb):
    """The availability timestamp must remain UNKNOWN."""
    result = load_station_met_daily(synthetic_cpcb)
    assert result.availability_timestamp == "UNKNOWN"
    assert result.forecast_usable is False
    assert "No publication lag is assumed" in result.notes


def test_station_met_preserves_the_measurement_date(synthetic_cpcb):
    """The measurement date must be preserved on every row."""
    result = load_station_met_daily(synthetic_cpcb)
    assert "Date" in result.frame.columns
    assert result.frame["Date"].is_monotonic_increasing


def test_station_met_reports_per_column_coverage(synthetic_cpcb):
    """Coverage must be reported per column, not only in aggregate."""
    result = load_station_met_daily(synthetic_cpcb)
    coverage = result.coverage
    assert "per_column_accepted_days" in coverage
    assert "per_column_missing_fraction" in coverage
    assert set(coverage["per_column_accepted_days"]) == set(COLUMN_NAMES.values())


def test_station_met_screens_implausible_values(tmp_path):
    """Out-of-range values must be set to missing, not clipped.

    The Phase 5 cross-validation found the CPCB barometric-pressure column
    carrying daily means up to 1062 mmHg, which is above sea-level
    pressure at sea level, so the screen must be explicit and configurable.
    """
    directory = tmp_path / "cpcb"
    directory.mkdir()
    dates = pd.date_range("2024-01-01", "2024-01-03", freq="h")
    rows = pd.DataFrame(
        {
            "Timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "PM2.5 (µg/m³)": 80.0,
            "AT (°C)": [-500.0] * len(dates),  # implausible
            "RH (%)": 50.0,
        }
    )
    rows.to_csv(directory / "a.csv", index=False)
    result = load_station_met_daily(
        directory,
        valid_ranges={"AT (°C)": (-20.0, 55.0), "RH (%)": (0.0, 100.0)},
    )
    # The implausible temperature must not appear as a value.
    assert "AT" not in result.frame.columns or result.frame["AT"].isna().all()
    assert result.coverage["per_column_screened_values"].get("AT", 0) == len(dates)


def test_station_met_can_exclude_a_column(tmp_path):
    """A column excluded by policy must not reach the output."""
    directory = tmp_path / "cpcb"
    directory.mkdir()
    dates = pd.date_range("2024-01-01", "2024-01-02", freq="h")
    rows = pd.DataFrame(
        {
            "Timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "PM2.5 (µg/m³)": 80.0,
            "AT (°C)": 25.0,
            "BP (mmHg)": 725.0,
        }
    )
    rows.to_csv(directory / "a.csv", index=False)
    result = load_station_met_daily(directory, excluded_columns=("BP (mmHg)",))
    assert "BP" not in result.frame.columns
    assert "AT" in result.frame.columns
    assert result.coverage["excluded_columns"] == ["BP (mmHg)"]


def test_missing_directory_raises(tmp_path):
    """A missing input directory is an error, not an empty dataset."""
    with pytest.raises(FileNotFoundError):
        load_station_met_daily(tmp_path / "does_not_exist")


def test_merra_column_names_carry_the_midday_suffix():
    """The naming must state what the value is (D-010)."""
    for name in MERRA_COLUMNS.values():
        assert name.endswith("_midday")


def test_merra_missing_directory_raises(tmp_path):
    """A missing input directory is an error."""
    with pytest.raises(FileNotFoundError):
        load_merra_daily(tmp_path / "does_not_exist")
