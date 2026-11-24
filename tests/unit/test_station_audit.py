"""Tests for the CPCB station-local meteorological audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerosphinx.quality.station_audit import (
    STATION_LOCAL_COLUMNS,
    audit_station_columns,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def synthetic_cpcb(tmp_path):
    """A synthetic CPCB export with a genuine hourly meteorological cycle."""
    directory = tmp_path / "cpcb"
    directory.mkdir()
    dates = pd.date_range("2024-01-01", "2024-03-31", freq="h")
    hours = dates.hour.to_numpy()
    rows = pd.DataFrame(
        {
            "Timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "PM2.5 (µg/m³)": 80 + 20 * np.sin(hours * 2 * np.pi / 24),
            "AT (°C)": 22 + 8 * np.sin((hours - 9) * 2 * np.pi / 24),
            "RH (%)": 60 - 15 * np.sin((hours - 9) * 2 * np.pi / 24),
            "WS (m/s)": 1.0 + 0.5 * np.abs(np.sin(hours * 2 * np.pi / 24)),
            "WD (deg)": np.where(np.sin(hours * 2 * np.pi / 24) > 0, 90.0, 270.0),
            "RF (mm)": 0.0,
            "TOT-RF (mm)": 0.0,
            "SR (W/mt2)": np.clip(400 * np.sin((hours - 6) * 2 * np.pi / 24), 0, None),
            "BP (mmHg)": 750.0 + 2 * np.sin(hours * 2 * np.pi / 24),
            "VWS (m/s)": -1.0 + 0.3 * np.sin(hours * 2 * np.pi / 24),
        }
    )
    rows.to_csv(
        directory / "City_wise_raw_data_1Hr_2024_Faridabad_1Hr.csv", index=False
    )
    return directory


def test_audit_covers_every_candidate_column(synthetic_cpcb):
    """Every candidate column present in the export must be audited."""
    records = audit_station_columns(synthetic_cpcb)
    audited = {record.source_column for record in records}
    expected = set(STATION_LOCAL_COLUMNS) & set(
        pd.read_csv(next(synthetic_cpcb.glob("*.csv")), nrows=0).columns
    )
    assert audited == expected


def test_audit_record_has_every_required_field(synthetic_cpcb):
    """The audit must supply every field the authorization requires."""
    record = audit_station_columns(synthetic_cpcb)[0].as_dict()
    required = {
        "variable",
        "source_column",
        "physical_meaning",
        "unit",
        "station_or_source",
        "measurement_timestamp",
        "availability_timestamp",
        "missing_percentage",
        "valid_range",
        "quality_flag",
        "forecast_usable",
    }
    assert required.issubset(record.keys())


def test_hourly_measurement_is_detected(synthetic_cpcb):
    """A column with a genuine hourly cycle must be identified as such."""
    records = {r.source_column: r for r in audit_station_columns(synthetic_cpcb)}
    temperature = records["AT (°C)"]
    assert temperature.evidence["timestamps_on_exact_hour"] is True
    assert temperature.evidence["mean_distinct_values_per_day"] > 12
    assert temperature.evidence["diurnal_cycle_present"] is True
    assert temperature.evidence["looks_like_hourly_measurement"] is True


def test_daily_aggregate_repeated_across_hours_is_detected(synthetic_cpcb):
    """A column that repeats one value per day must not be called hourly."""
    records = {r.source_column: r for r in audit_station_columns(synthetic_cpcb)}
    rainfall = records["RF (mm)"]
    assert rainfall.evidence["mean_distinct_values_per_day"] <= 1.5
    assert rainfall.evidence["looks_like_hourly_measurement"] is False


def test_availability_unknown_blocks_forecast_use(synthetic_cpcb):
    """Without a known availability time, no variable may be forecast-usable.

    This is the authorisation's central requirement: a measurement whose
    publication time is unknown cannot be used in a forecast issued before
    that time.
    """
    records = audit_station_columns(synthetic_cpcb)
    assert all(record.availability_timestamp == "UNKNOWN" for record in records)
    assert all(record.forecast_usable is False for record in records)
    # Without a reanalysis comparison the station-locality is also
    # unverified, so the flag reflects the weaker of the two conditions.
    assert all(
        record.quality_flag in {"timing unverified", "provenance unverified"}
        for record in records
    )


def test_missing_percentage_is_computed(synthetic_cpcb):
    """The missing percentage must reflect the actual data."""
    records = {r.source_column: r for r in audit_station_columns(synthetic_cpcb)}
    assert records["AT (°C)"].missing_percentage == pytest.approx(0.0, abs=0.01)
    assert records["BP (mmHg)"].missing_percentage == pytest.approx(0.0, abs=0.01)


def test_valid_range_is_reported(synthetic_cpcb):
    """The observed range must be recorded for each variable."""
    records = {r.source_column: r for r in audit_station_columns(synthetic_cpcb)}
    assert records["AT (°C)"].valid_range.startswith("[14.")
    assert records["BP (mmHg)"].valid_range.startswith("[748.")


def test_comparison_without_merra_is_reported_as_unknown(synthetic_cpcb):
    """Without a MERRA-2 directory the station-locality test cannot run."""
    records = audit_station_columns(synthetic_cpcb, merra_dir=None)
    assert all(record.station_or_source == "UNKNOWN" for record in records)
    assert all(record.forecast_usable is False for record in records)


def test_missing_directory_raises(tmp_path):
    """A missing input directory is an error, not an empty audit."""
    with pytest.raises(FileNotFoundError):
        audit_station_columns(tmp_path / "does_not_exist")
