"""Tests for the MERRA-2 validation.

Covers the Phase 5 gate criteria: dimension, unit, grid and temporal
coverage verification, plus the physical cross-validation against the CPCB
station.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerosphinx.quality.merra_validation import (
    MERRA_EXPECTED_UNITS,
    MERRA_PLAUSIBLE_RANGES,
    station_cross_check,
    validate_merra,
)

pytestmark = pytest.mark.unit


def _write_granule(
    directory,
    name: str = "MERRA2_400.tavg1_2d_slv_Nx.20240101.SUB.nc",
    times=("2024-01-01 11:30", "2024-01-01 12:30", "2024-01-01 13:30"),
    t2m=(290.15, 290.65, 291.15),
    t2m_units: str = "K",
    spatial_box: str = "77.2 77.4 28.3 28.5",
) -> "object":
    """Write one MERRA-2-like granule and return its path."""
    import xarray as xr

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    stamps = pd.to_datetime(list(times))

    def column(values):
        return np.array([[[float(v)]] for v in values], dtype=float)

    dataset = xr.Dataset(
        {
            "T2M": (("time", "lat", "lon"), column(t2m)),
            "TS": (("time", "lat", "lon"), column([289.15, 289.65, 290.15])),
            "QV2M": (("time", "lat", "lon"), column([0.008, 0.0075, 0.007])),
            "SLP": (("time", "lat", "lon"), column([101500.0, 101510.0, 101520.0])),
            "PS": (("time", "lat", "lon"), column([99200.0, 99210.0, 99220.0])),
        },
        coords={
            "lat": ("lat", [28.5]),
            "lon": ("lon", [77.5]),
            "time": ("time", stamps.to_numpy()),
        },
        attrs={
            "history_L34RS": (
                "Created by L34RS v1.4.4 @ NASA GES DISC on July 17 2025 "
                f"10:12:13. Spatial: {spatial_box} Variables: PS QV10M QV2M SLP"
            )
        },
    )
    dataset["T2M"].attrs["units"] = t2m_units
    dataset["TS"].attrs["units"] = "K"
    dataset["QV2M"].attrs["units"] = "kg kg-1"
    dataset["SLP"].attrs["units"] = "Pa"
    dataset["PS"].attrs["units"] = "Pa"
    dataset.to_netcdf(path)
    dataset.close()
    return path


@pytest.fixture
def synthetic_merra(tmp_path):
    """A synthetic MERRA-2 granule with a known physical state."""
    directory = tmp_path / "merra"
    _write_granule(directory)
    return directory


def test_structure_is_audited(synthetic_merra):
    """The audit must record the dimensions and the variable inventory."""
    audit = validate_merra(synthetic_merra).as_dict()
    assert audit["n_granules"] == 1
    assert len(audit["dimension_signatures"]) == 1
    assert audit["days_per_granule"] == {1: 1}
    assert "T2M" in audit["variable_presence"]


def test_unit_mismatch_is_detected(tmp_path):
    """A variable with the wrong unit must be reported."""
    directory = tmp_path / "merra"
    _write_granule(directory, t2m_units="degC")
    audit = validate_merra(directory).as_dict()
    assert "T2M" in audit["unit_mismatches"]


def test_out_of_range_value_is_detected(tmp_path):
    """A physically impossible value must be reported."""
    directory = tmp_path / "merra"
    _write_granule(directory, t2m=(999.0, 290.65, 291.15))
    audit = validate_merra(directory).as_dict()
    assert audit["out_of_range_values"].get("T2M", 0) >= 1


def test_missing_value_is_counted(tmp_path):
    """A NaN value must be counted."""
    import xarray as xr

    directory = tmp_path / "merra"
    path = _write_granule(directory)
    dataset = xr.open_dataset(path)
    values = np.asarray(dataset["T2M"].values).copy()
    values[0, 0, 0] = np.nan
    dataset["T2M"].values = values
    # netCDF4 cannot overwrite an open file, so write a new one.
    dataset.to_netcdf(directory / "with_nan.nc")
    dataset.close()
    (directory / path.name).unlink()

    audit = validate_merra(directory).as_dict()
    assert audit["n_missing_values"] >= 1


def test_grid_cell_is_recorded(synthetic_merra):
    """The grid cell and the subsetting box must be recorded."""
    audit = validate_merra(synthetic_merra).as_dict()
    assert "(28.5, 77.5)" in audit["grid_cells"]
    assert any("77.2 77.4 28.3 28.5" in key for key in audit["subsetting_boxes"])


def test_local_hours_are_the_ist_equivalent(synthetic_merra):
    """The UTC timesteps must be reported at their local station time."""
    audit = validate_merra(synthetic_merra).as_dict()
    # 11:30, 12:30, 13:30 UTC is 17:00, 18:00, 19:00 at UTC+5:30.
    assert audit["local_hours_ist"] == ["17:00", "18:00", "19:00"]


def test_temporal_coverage_is_reported(synthetic_merra):
    """The date range and any missing dates must be reported."""
    audit = validate_merra(synthetic_merra).as_dict()
    assert audit["date_start"] == "2024-01-01"
    assert audit["date_end"] == "2024-01-01"
    assert audit["n_missing_dates"] == 0


def test_expected_units_are_declared():
    """Every expected variable must have a declared unit and range."""
    for name in ("T2M", "TS", "QV2M", "SLP", "U10M", "V10M"):
        assert name in MERRA_EXPECTED_UNITS
        assert MERRA_PLAUSIBLE_RANGES[name][0] < MERRA_PLAUSIBLE_RANGES[name][1]


def test_cross_check_aligns_utc_to_local(tmp_path):
    """The station match must use the local time, not the UTC time.

    A station observation recorded at 17:00 IST is the same instant as a
    MERRA-2 value at 11:30 UTC. Matching on the raw UTC hour would pair
    the wrong observations and could manufacture a spurious correlation.
    """
    merra_dir = tmp_path / "merra"
    _write_granule(merra_dir)

    cpcb_dir = tmp_path / "cpcb"
    cpcb_dir.mkdir()
    rows = []
    for hour in (17, 18, 19):
        rows.append(
            {
                "Timestamp": f"2024-01-01 {hour:02d}:00:00",
                "PM2.5 (µg/m³)": 80.0,
                "AT (°C)": 17.0,
                "RH (%)": 50.0,
                "BP (mmHg)": 725.0,
            }
        )
    pd.DataFrame(rows).to_csv(cpcb_dir / "a.csv", index=False)

    cross = station_cross_check(merra_dir, cpcb_dir)
    assert len(cross) == 3
    # The MERRA value at 11:30 UTC must be matched with the station's 17:00 IST.
    matched = cross[cross.utc_time.dt.hour == 11]
    assert len(matched) == 1
    assert matched["local_time"].iloc[0].hour == 17


def test_cross_check_computes_the_elevation_term(tmp_path):
    """The sea-level minus surface pressure difference must be reported."""
    merra_dir = tmp_path / "merra"
    _write_granule(merra_dir)
    cpcb_dir = tmp_path / "cpcb"
    cpcb_dir.mkdir()
    pd.DataFrame(
        [
            {
                "Timestamp": f"2024-01-01 {hour:02d}:00:00",
                "PM2.5 (µg/m³)": 80.0,
                "AT (°C)": 17.0,
                "RH (%)": 50.0,
                "BP (mmHg)": 725.0,
            }
            for hour in (17, 18, 19)
        ]
    ).to_csv(cpcb_dir / "a.csv", index=False)

    cross = station_cross_check(merra_dir, cpcb_dir)
    # SLP 101500 Pa - PS 99200 Pa = 2300 Pa = 23 hPa.
    assert cross["SLP_minus_PS_hPa"].mean() == pytest.approx(23.0, abs=0.01)
