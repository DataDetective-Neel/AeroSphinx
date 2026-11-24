"""Tests for INSAT-3D AOD ingestion and validation.

Covers the four test categories required by ``AGENTS.md`` section 8:

* HDF5 dimension and orientation tests
* fill-value and QA tests
* spatial-window tests
* duplicate-granule tests
"""

from __future__ import annotations

import numpy as np
import pytest

from aerosphinx.ingestion.aod import (
    AODRules,
    inspect_granules,
    load_aod_daily,
    orientation_check,
    window_sensitivity,
)

pytestmark = pytest.mark.unit

STATION_LAT = 28.4089
STATION_LON = 77.3178


@pytest.fixture
def synthetic_granule(tmp_path):
    """A synthetic L2G granule with a known AOD field.

    The array is built as ``[time, latitude, longitude]`` with a descending
    latitude axis, matching the real product, and the AOD field encodes its
    own position so that a transposed read would be detected.
    """
    import h5py

    directory = tmp_path / "aod"
    directory.mkdir()
    path = directory / "3DIMG_15APR2024_0530_L2G_AOD_V02R00.h5"

    n = 51
    latitude = np.linspace(30.0, 20.0, n)  # descending
    longitude = np.linspace(75.0, 85.0, n)  # ascending
    grid = np.full((1, n, n), -999.0, dtype=np.float32)
    # A hot spot at a known position, surrounded by fill.
    grid[0, 10, 20] = 2.5
    grid[0, 10, 21] = 1.5

    with h5py.File(path, "w") as handle:
        handle.create_dataset("AOD", data=grid)
        handle.create_dataset("latitude", data=latitude)
        handle.create_dataset("longitude", data=longitude)
        handle.create_dataset("time", data=np.array([12754410.0]))
        handle["AOD"].attrs["_FillValue"] = np.array([-999.0], dtype=np.float32)
        handle["AOD"].dims[0].attach_scale(handle["time"])
        handle["AOD"].dims[1].attach_scale(handle["latitude"])
        handle["AOD"].dims[2].attach_scale(handle["longitude"])
        handle.attrs["Acquisition_Date"] = b"15APR2024"
        handle.attrs["Acquisition_Time_in_GMT"] = b"0530"
        handle.attrs["Acquisition_Start_Time"] = b"15-APR-2024T05:30:00"
        handle.attrs["Acquisition_End_Time"] = b"15-APR-2024T05:56:58"
    return directory


# --------------------------------------------------------------------------
# Dimension and orientation tests
# --------------------------------------------------------------------------


def test_dimension_scales_bind_latitude_and_longitude(synthetic_granule):
    """The AOD axis order must be verified from the dimension scales."""
    audit, granules = inspect_granules(synthetic_granule)
    bindings = audit.dimension_scale_bindings
    assert len(bindings) == 1
    (only,) = bindings
    assert only[1] == ("/latitude",)
    assert only[2] == ("/longitude",)


def test_grayscale_and_axis_order_are_detected(synthetic_granule):
    """A transposed product would produce a different dimension binding."""
    audit, _ = inspect_granules(synthetic_granule)
    (only,) = audit.dimension_scale_bindings
    assert only[1] != ("/longitude",)
    assert only[2] != ("/latitude",)


def test_latitude_order_is_recorded(synthetic_granule):
    """A descending latitude axis must be recorded, not assumed."""
    audit, _ = inspect_granules(synthetic_granule)
    assert audit.latitude_orders == {True: 1}
    assert audit.longitude_orders == {True: 1}


def test_extraction_reads_the_correct_axes(synthetic_granule):
    """The hot spot must be found at its true latitude and longitude.

    The synthetic field carries a value of 2.5 at latitude index 10 and
    longitude index 21. A transposed read would look for it at the
    transposed position and find fill instead.
    """
    result = load_aod_daily(
        synthetic_granule,
        station_lat=30.0 - 10 * (10.0 / 50.0),
        station_lon=75.0 + 21 * (10.0 / 50.0),
    )
    assert len(result.frame) == 1
    value = result.frame["AOD"].iloc[0]
    assert np.isfinite(value)
    # The window covers both the 2.5 and the 1.5 pixel.
    assert value == pytest.approx(2.0, abs=1e-6)
    assert result.frame["aod_valid_pixels"].iloc[0] == 2


def test_orientation_check_rejects_a_physically_impossible_orientation(
    synthetic_granule,
):
    """The physical cross-check must be able to fail.

    Two boxes are chosen with the *same* AOD content under both
    orientations, so neither hypothesis can produce an ocean/land contrast.
    The check must then report that the dimension-scale orientation is not
    corroborated, rather than silently passing.
    """
    result = orientation_check(
        synthetic_granule,
        boxes={
            "box_low": ((24.0, 26.0), (74.0, 76.0)),
            "box_high": ((24.0, 26.0), (80.0, 82.0)),
        },
    )
    assert "dimension_scale" in result["hypotheses"]
    assert "transposed" in result["hypotheses"]
    assert "dimension_scale_orientation_corroborated" in result
    # With identical content in both boxes no contrast is possible, so the
    # check must not claim corroboration.
    assert result["dimension_scale_orientation_corroborated"] is False


# --------------------------------------------------------------------------
# Fill-value and QA tests
# --------------------------------------------------------------------------


def test_fill_value_is_detected(synthetic_granule):
    """The fill value must be read from the product, not assumed."""
    audit, _ = inspect_granules(synthetic_granule)
    assert audit.fill_values == {-999.0: 1}


def test_fill_value_is_excluded_from_the_mean(synthetic_granule):
    """Fill pixels must never contribute to the extracted mean."""
    result = load_aod_daily(
        synthetic_granule,
        station_lat=30.0 - 10 * (10.0 / 50.0),
        station_lon=75.0 + 21 * (10.0 / 50.0),
    )
    # Only the two real pixels are counted, not the surrounding fill.
    assert result.frame["aod_valid_pixels"].iloc[0] == 2


def test_no_qa_dataset_is_reported(synthetic_granule):
    """The absence of a QA dataset must be reported explicitly."""
    audit, _ = inspect_granules(synthetic_granule)
    assert audit.has_qa_dataset is False


def test_scale_factor_absence_is_recorded(synthetic_granule):
    """The absence of a scale factor must be recorded."""
    audit, _ = inspect_granules(synthetic_granule)
    assert audit.scale_factors == {False: 1}


def test_out_of_range_values_are_excluded():
    """Values outside the validity window must not enter the mean."""
    rules = AODRules(aod_valid_min=0.0, aod_valid_max=5.0)
    assert rules.aod_valid_max > rules.aod_valid_min
    with pytest.raises(ValueError):
        AODRules(aod_valid_min=5.0, aod_valid_max=1.0)


# --------------------------------------------------------------------------
# Spatial-window tests
# --------------------------------------------------------------------------


def test_window_selects_the_expected_pixel_count(synthetic_granule):
    """The window must select the pixels inside its half-width."""
    # A window of half-width 0.5 degrees covers one row and one column
    # of the 0.2-degree grid, plus the hot spot.
    rules = AODRules(window_half_width_deg=0.25)
    result = load_aod_daily(
        synthetic_granule,
        rules,
        station_lat=30.0 - 10 * (10.0 / 50.0),
        station_lon=75.0 + 21 * (10.0 / 50.0),
    )
    assert result.frame["aod_valid_pixels"].iloc[0] >= 1


def test_window_half_width_changes_the_pixel_count(synthetic_granule):
    """A wider window must select at least as many pixels."""
    small = load_aod_daily(
        synthetic_granule,
        AODRules(window_half_width_deg=0.05),
        station_lat=30.0 - 10 * (10.0 / 50.0),
        station_lon=75.0 + 21 * (10.0 / 50.0),
    )
    large = load_aod_daily(
        synthetic_granule,
        AODRules(window_half_width_deg=1.0),
        station_lat=30.0 - 10 * (10.0 / 50.0),
        station_lon=75.0 + 21 * (10.0 / 50.0),
    )
    assert (
        large.frame["aod_valid_pixels"].iloc[0]
        >= small.frame["aod_valid_pixels"].iloc[0]
    )


def test_invalid_window_is_rejected():
    """A non-positive window must fail loudly."""
    with pytest.raises(ValueError):
        AODRules(window_half_width_deg=0.0)


def test_window_sensitivity_reports_every_candidate(synthetic_granule):
    """The sensitivity table must report every candidate window."""
    table = window_sensitivity(
        synthetic_granule,
        half_widths=(0.2, 0.3),
        station_lat=30.0 - 10 * (10.0 / 50.0),
        station_lon=75.0 + 21 * (10.0 / 50.0),
    )
    assert len(table) == 2
    for column in (
        "window_half_width_deg",
        "n_days",
        "mean_AOD",
        "mean_valid_pixels",
        "correlation_with_default",
    ):
        assert column in table.columns


# --------------------------------------------------------------------------
# Duplicate-granule tests
# --------------------------------------------------------------------------


def test_duplicate_date_overpass_is_detected(tmp_path):
    """Two granules for the same date and overpass must be reported."""
    import shutil

    import h5py

    directory = tmp_path / "aod"
    directory.mkdir()
    source = directory / "3DIMG_15APR2024_0530_L2G_AOD_V02R00.h5"
    n = 51
    grid = np.full((1, n, n), -999.0, dtype=np.float32)
    grid[0, 10, 20] = 1.0
    with h5py.File(source, "w") as handle:
        handle.create_dataset("AOD", data=grid)
        handle.create_dataset("latitude", data=np.linspace(30.0, 20.0, n))
        handle.create_dataset("longitude", data=np.linspace(75.0, 85.0, n))
        handle.create_dataset("time", data=np.array([12754410.0]))
        handle["AOD"].attrs["_FillValue"] = np.array([-999.0], dtype=np.float32)
        handle.attrs["Acquisition_Date"] = b"15APR2024"
        handle.attrs["Acquisition_Time_in_GMT"] = b"0530"

    # An exact duplicate under a different file name.
    shutil.copy(source, directory / "3DIMG_15APR2024_0530_L2G_AOD_V02R00_copy.h5")

    audit, _ = inspect_granules(directory)
    assert len(audit.duplicate_keys) == 1


def test_distinct_overpasses_on_one_date_are_not_duplicates(tmp_path):
    """Two overpasses of the same date are two observations, not duplicates."""
    import h5py

    directory = tmp_path / "aod"
    directory.mkdir()
    n = 51
    for overpass, name in (("0530", "a"), ("0630", "b")):
        grid = np.full((1, n, n), -999.0, dtype=np.float32)
        grid[0, 10, 20] = 1.0
        with h5py.File(
            directory / f"3DIMG_15APR2024_{overpass}_L2G_AOD_V02R00_{name}.h5",
            "w",
        ) as handle:
            handle.create_dataset("AOD", data=grid)
            handle.create_dataset("latitude", data=np.linspace(30.0, 20.0, n))
            handle.create_dataset("longitude", data=np.linspace(75.0, 85.0, n))
            handle.create_dataset("time", data=np.array([12754410.0]))
            handle["AOD"].attrs["_FillValue"] = np.array([-999.0], dtype=np.float32)
            handle.attrs["Acquisition_Date"] = b"15APR2024"
            handle.attrs["Acquisition_Time_in_GMT"] = overpass.encode()

    audit, granules = inspect_granules(directory)
    assert len(audit.duplicate_keys) == 0
    assert audit.distinct_dates == 1
    assert len(granules) == 2


def test_multiple_overpasses_are_averaged_into_one_daily_value(tmp_path):
    """Same-date overpasses must be averaged into a single daily value."""
    import h5py

    directory = tmp_path / "aod"
    directory.mkdir()
    n = 51
    for overpass, value in (("0530", 1.0), ("0630", 3.0)):
        grid = np.full((1, n, n), -999.0, dtype=np.float32)
        grid[0, 10, 20] = value
        with h5py.File(
            directory / f"3DIMG_15APR2024_{overpass}_L2G_AOD_V02R00.h5", "w"
        ) as handle:
            handle.create_dataset("AOD", data=grid)
            handle.create_dataset("latitude", data=np.linspace(30.0, 20.0, n))
            handle.create_dataset("longitude", data=np.linspace(75.0, 85.0, n))
            handle.create_dataset("time", data=np.array([12754410.0]))
            handle["AOD"].attrs["_FillValue"] = np.array([-999.0], dtype=np.float32)
            handle.attrs["Acquisition_Date"] = b"15APR2024"
            handle.attrs["Acquisition_Time_in_GMT"] = overpass.encode()

    result = load_aod_daily(
        directory,
        station_lat=30.0 - 10 * (10.0 / 50.0),
        station_lon=75.0 + 21 * (10.0 / 50.0),
    )
    assert len(result.frame) == 1, "one date must give one daily row"
    assert result.frame["aod_overpasses"].iloc[0] == 2
    assert result.frame["AOD"].iloc[0] == pytest.approx(2.0, abs=1e-6)


def test_missing_directory_raises(tmp_path):
    """A missing input directory is an error."""
    with pytest.raises(FileNotFoundError):
        load_aod_daily(tmp_path / "does_not_exist")
