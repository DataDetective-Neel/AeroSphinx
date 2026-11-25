"""Tests for the Phase 6 availability mask and fusion alignment gate.

The forward-fill defect (D-003) is pinned here: a missing environmental
observation must stay missing, and the fused frame must carry an explicit
per-source availability mask rather than an imputed value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerosphinx.features.availability import (
    MISSING_VALUE_POLICY,
    SOURCE_GROUPS,
    FusedDataset,
    FusionRules,
    assert_fusion_alignment,
    build_fused_dataset,
    support_columns,
    validate_fused_dataset,
    value_columns,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def ground() -> pd.DataFrame:
    """Four ground days with a calendar gap on 2024-01-03."""
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"]
            ),
            "PM25": [100.0, 110.0, 120.0, 130.0],
            "PM25_Lag_1": [np.nan, 100.0, np.nan, np.nan],
        }
    )


def _merra(dates: list[str]) -> pd.DataFrame:
    """A MERRA-2 daily frame with every canonical value column."""
    frame = pd.DataFrame({"Date": pd.to_datetime(dates)})
    for offset, column in enumerate(value_columns("merra")):
        frame[column] = [280.0 + offset + 10.0 * index for index in range(len(dates))]
    return frame


@pytest.fixture
def merra_frame() -> pd.DataFrame:
    """MERRA-2 on two ground dates plus one date outside the ground calendar."""
    return _merra(["2024-01-01", "2024-01-04", "2024-01-06"])


def _row(frame: pd.DataFrame, date: str) -> pd.Series:
    return frame.loc[frame["Date"] == pd.Timestamp(date)].iloc[0]


# --------------------------------------------------------------------------
# The D-003 correction: no manufactured observations
# --------------------------------------------------------------------------


def test_missing_environmental_values_are_not_forward_filled(ground, merra_frame):
    """A gap in a source must stay NaN; the previous day's value may not leak."""
    result = build_fused_dataset(ground, merra=merra_frame)
    jan2 = _row(result.frame, "2024-01-02")

    assert jan2["merra_observed"] is np.False_ or jan2["merra_observed"] == False
    for column in value_columns("merra"):
        assert pd.isna(jan2[column]), f"{column} was carried forward"

    # And the value on the next observed date is not the stale one either.
    jan4 = _row(result.frame, "2024-01-04")
    assert jan4["merra_observed"] == True
    assert jan4["T2M_midday"] == pytest.approx(290.0)


def test_observation_age_records_staleness_without_supplying_a_value(ground):
    """The age is a diagnostic; it never fills the missing value."""
    result = build_fused_dataset(ground, merra=_merra(["2024-01-01"]))
    ages = result.frame["merra_observation_age_days"].to_list()
    assert ages == [0.0, 1.0, 3.0, 4.0]
    # Every unobserved row still carries no value.
    unobserved = result.frame[~result.frame["merra_observed"]]
    assert unobserved[list(value_columns("merra"))].isna().all().all()


def test_observation_age_is_missing_before_the_first_observation(ground):
    """A row preceding the first source observation has no age to report."""
    result = build_fused_dataset(ground, merra=_merra(["2024-01-02"]))
    assert np.isnan(result.frame.loc[0, "merra_observation_age_days"])
    assert result.frame.loc[1, "merra_observation_age_days"] == 0.0


def test_fused_frame_preserves_the_ground_calendar(ground, merra_frame):
    """Environmental sources must never extend the ground calendar."""
    result = build_fused_dataset(ground, merra=merra_frame)
    assert list(result.frame["Date"]) == list(pd.to_datetime(ground["Date"]))
    assert pd.Timestamp("2024-01-03") not in set(result.frame["Date"])
    assert pd.Timestamp("2024-01-06") not in set(result.frame["Date"])
    # The out-of-calendar source date is reported, not silently dropped.
    assert (
        result.summary["per_source"]["merra"]["source_dates_outside_ground_calendar"]
        == 1
    )


def test_omitted_source_keeps_the_schema_stable(ground):
    """An omitted source must still yield its columns and a False mask."""
    result = build_fused_dataset(ground)
    for source in SOURCE_GROUPS:
        assert f"{source}_observed" in result.frame.columns
        assert f"{source}_observation_age_days" in result.frame.columns
        assert not result.frame[f"{source}_observed"].any()
        for column in value_columns(source):
            assert column in result.frame.columns
            assert result.frame[column].isna().all()
    assert (result.frame["n_sources_observed"] == 0).all()
    assert not result.frame["all_sources_observed"].any()
    assert result.summary["requested_sources"] == []
    assert result.summary["missing_value_policy"] == MISSING_VALUE_POLICY


def test_aod_zero_pixel_day_is_not_an_observation(ground):
    """A granule with no valid pixel is not a measured AOD value."""
    aod = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01"]),
            "AOD": [np.nan],
            "aod_valid_pixels": [0],
            "aod_overpasses": [1],
        }
    )
    result = build_fused_dataset(ground, aod=aod)
    jan1 = _row(result.frame, "2024-01-01")
    assert jan1["aod_observed"] == False
    assert pd.isna(jan1["AOD"])
    # Support metadata is 0, not a manufactured measurement.
    assert jan1["aod_valid_pixels"] == 0
    assert jan1["aod_overpasses"] == 1


def test_aod_with_valid_pixels_is_observed(ground):
    """A value supported by at least one pixel is observed."""
    aod = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01"]),
            "AOD": [1.4],
            "aod_valid_pixels": [6],
            "aod_overpasses": [1],
        }
    )
    result = build_fused_dataset(ground, aod=aod)
    jan1 = _row(result.frame, "2024-01-01")
    assert jan1["aod_observed"] == True
    assert jan1["AOD"] == pytest.approx(1.4)
    assert jan1["aod_observation_age_days"] == 0.0


def test_combined_mask_counts_only_requested_sources(ground, merra_frame):
    """The combined availability must not count a source that was not fused."""
    result = build_fused_dataset(ground, merra=merra_frame)
    assert result.summary["requested_sources"] == ["merra"]
    assert result.frame["n_sources_observed"].max() == 1
    # Jan 1 is observed by MERRA and so is a complete case for this fusion.
    assert _row(result.frame, "2024-01-01")["all_sources_observed"] == True
    assert _row(result.frame, "2024-01-02")["all_sources_observed"] == False


def test_summary_reports_observed_and_unobserved_target_means(ground, merra_frame):
    """Availability must be described against the target, not only counted."""
    result = build_fused_dataset(ground, merra=merra_frame)
    merra = result.summary["per_source"]["merra"]
    assert merra["fused_rows_observed"] == 2
    assert merra["target_mean_when_observed"] == pytest.approx(110.0)
    assert merra["target_mean_when_unobserved"] == pytest.approx(120.0)
    assert merra["target_mean_difference"] == pytest.approx(-10.0)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_duplicate_ground_dates_raise(ground):
    duplicated = pd.concat([ground, ground.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate dates"):
        build_fused_dataset(duplicated)


def test_duplicate_source_dates_raise(ground, merra_frame):
    duplicated = pd.concat([merra_frame, merra_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate dates"):
        build_fused_dataset(ground, merra=duplicated)


def test_source_missing_a_required_column_raises(ground):
    incomplete = _merra(["2024-01-01"]).drop(columns=["T2M_midday"])
    with pytest.raises(KeyError, match="T2M_midday"):
        build_fused_dataset(ground, merra=incomplete)


def test_invalid_fusion_rules_raise():
    with pytest.raises(ValueError):
        FusionRules(min_observed_values=0)


# --------------------------------------------------------------------------
# The alignment gate
# --------------------------------------------------------------------------


def test_a_clean_fusion_passes_every_alignment_check(ground, merra_frame):
    result = build_fused_dataset(ground, merra=merra_frame)
    validation = validate_fused_dataset(result, daily=ground, merra=merra_frame)
    assert validation["passed"], validation["failed_checks"]
    assert validation["n_failed"] == 0
    assert_fusion_alignment(validation)


def test_alignment_gate_rejects_a_forward_filled_value(ground, merra_frame):
    """The exact D-003 failure mode must fail the gate."""
    result = build_fused_dataset(ground, merra=merra_frame)
    tampered = result.frame.copy()
    # Manufacture an observation on Jan 2 by copying Jan 1 forward.
    for column in value_columns("merra"):
        tampered.loc[1, column] = tampered.loc[0, column]

    dataset = FusedDataset(
        frame=tampered,
        summary=result.summary,
        rules=result.rules,
        requested_sources=result.requested_sources,
    )
    validation = validate_fused_dataset(dataset, daily=ground, merra=merra_frame)
    assert not validation["passed"]
    assert "merra:value_without_observation_flag" in validation["failed_checks"]
    with pytest.raises(ValueError, match="Phase 6 alignment gate"):
        assert_fusion_alignment(validation)


def test_alignment_gate_rejects_a_false_observation_flag(ground, merra_frame):
    """Clearing a flag while keeping the value is also a failure."""
    result = build_fused_dataset(ground, merra=merra_frame)
    tampered = result.frame.copy()
    tampered.loc[0, "merra_observed"] = False

    dataset = FusedDataset(
        frame=tampered,
        summary=result.summary,
        rules=result.rules,
        requested_sources=result.requested_sources,
    )
    validation = validate_fused_dataset(dataset, daily=ground, merra=merra_frame)
    assert not validation["passed"]
    assert "merra:value_without_observation_flag" in validation["failed_checks"]
    assert "merra:observation_age_is_consistent" in validation["failed_checks"]


def test_alignment_gate_rejects_a_calendar_extension(ground, merra_frame):
    """A row outside the ground calendar must fail the gate."""
    result = build_fused_dataset(ground, merra=merra_frame)
    extra = result.frame.iloc[[0]].copy()
    extra.loc[:, "Date"] = pd.Timestamp("2024-01-06")
    tampered = pd.concat([result.frame, extra], ignore_index=True)

    dataset = FusedDataset(
        frame=tampered,
        summary=result.summary,
        rules=result.rules,
        requested_sources=result.requested_sources,
    )
    validation = validate_fused_dataset(dataset, daily=ground, merra=merra_frame)
    assert not validation["passed"]
    assert "row_count_matches_ground_series" in validation["failed_checks"]
    assert "ground_calendar_preserved" in validation["failed_checks"]


def test_support_columns_are_declared_for_aod_only():
    """Only AOD has support-metadata columns."""
    assert support_columns("aod") == ("aod_valid_pixels", "aod_overpasses")
    assert support_columns("merra") == ()
    assert "AOD" in value_columns("aod")
    assert "aod_valid_pixels" not in value_columns("aod")
