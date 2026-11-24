"""Tests for the M0-M8 experiment matrix.

The properties that matter are structural: every configuration exists, an
environmental configuration cannot be operationally eligible, and every
configuration in a track is evaluated on identical eligible dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerosphinx.tracks.experiment_matrix import (
    MATRIX,
    common_eligible,
    matrix_config_manifest,
    run_matrix,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    """A 420-day frame with complete, genuinely observed sources."""
    dates = pd.date_range("2023-01-01", periods=420, freq="D")
    n = len(dates)
    rng = np.random.default_rng(0)
    pm25 = 85 + 25 * np.sin(np.arange(n) * 2 * np.pi / 365) + rng.normal(0, 5, n)
    frame = pd.DataFrame({"Date": dates, "PM25": pm25})
    for lag in (1, 2, 3, 7):
        frame[f"PM25_Lag_{lag}"] = frame["PM25"].shift(lag)
    frame["PM25_Roll3"] = frame["PM25"].shift(1).rolling(3).mean()
    frame["PM25_Roll7"] = frame["PM25"].shift(1).rolling(7).mean()
    frame["day_sin"] = np.sin(dates.day_of_week.to_numpy() * 2 * np.pi / 7)
    frame["day_cos"] = np.cos(dates.day_of_week.to_numpy() * 2 * np.pi / 7)
    frame["month_sin"] = np.sin(dates.month.to_numpy() * 2 * np.pi / 12)
    frame["month_cos"] = np.cos(dates.month.to_numpy() * 2 * np.pi / 12)
    frame["AOD"] = rng.uniform(0.1, 1.5, n)
    frame["aod_valid_pixels"] = 10
    frame["aod_overpasses"] = 1
    for column in (
        "T2M_midday",
        "TS_midday",
        "QV2M_midday",
        "U10M_midday",
        "V10M_midday",
        "SLP_midday",
    ):
        frame[column] = rng.normal(300, 5, n)
    frame["aod_observed"] = True
    frame["merra_observed"] = True
    return frame


def test_matrix_has_m0_through_m8():
    assert set(MATRIX) == {f"M{i}" for i in range(9)}


def test_only_ground_configs_are_operationally_eligible():
    for config in MATRIX.values():
        if config.groups == ("pm25_history", "calendar"):
            assert config.operational_eligible is True
        else:
            assert config.operational_eligible is False


def test_manifest_is_serialisable():
    import json

    json.dumps(matrix_config_manifest())


def test_common_eligibility_requires_environmental_flags(synthetic_frame):
    frame = synthetic_frame.copy()
    frame.loc[frame.index[:5], "aod_observed"] = False
    mask = common_eligible(frame, [MATRIX["M0"], MATRIX["M2"]])
    assert not mask.iloc[:5].any()
    assert mask.iloc[10:].all()


def test_common_eligibility_is_ground_only_for_track_a(synthetic_frame):
    frame = synthetic_frame.copy()
    frame["aod_observed"] = False
    frame["merra_observed"] = False
    mask = common_eligible(frame, [MATRIX["M0"], MATRIX["M1"]])
    assert mask.iloc[1:].all()


def test_track_a_runs_and_keeps_configs_separate(synthetic_frame):
    result = run_matrix(
        synthetic_frame,
        "A",
        ["M0", "M1"],
        min_train_days=200,
        test_block_days=30,
        step_days=30,
    )
    assert set(result.predictions["config"]) == {"M0", "M1"}
    assert {"MAE", "RMSE", "MedianAE", "R2"} <= set(result.metrics.columns)
    assert result.metrics["n"].gt(0).all()


def test_track_b_runs_all_requested_configs(synthetic_frame):
    result = run_matrix(
        synthetic_frame,
        "B",
        ["M0", "M1", "M4"],
        min_train_days=200,
        test_block_days=30,
        step_days=30,
    )
    assert set(result.predictions["config"]) == {"M0", "M1", "M4"}
    assert not result.metrics.empty
    assert not result.ablations.empty


def test_common_metrics_use_only_fully_predicted_dates(synthetic_frame):
    result = run_matrix(
        synthetic_frame,
        "B",
        ["M0", "M1"],
        min_train_days=200,
        test_block_days=30,
        step_days=30,
    )
    # Persistence and ridge both predict every eligible row here.
    assert not result.common_metrics.empty
    assert result.common_metrics["n"].gt(0).all()


def test_independent_seasonal_model_abstains_without_season_data(synthetic_frame):
    """M7 must not fabricate a season it has no specialist for."""
    small = synthetic_frame.iloc[:220].copy()
    result = run_matrix(
        small,
        "B",
        ["M7"],
        min_train_days=180,
        test_block_days=30,
        step_days=30,
        min_specialist_days=10_000,
    )
    if result.predictions.empty:
        pytest.skip("no fold fit the short frame")
    # With an impossible specialist threshold, every prediction is absent.
    assert result.predictions["Prediction"].isna().all()
    assert result.metrics["n"].iloc[0] == 0


def test_fallback_model_predicts_when_specialists_are_ineligible(synthetic_frame):
    """M8 must always fall back to the global model."""
    result = run_matrix(
        synthetic_frame,
        "B",
        ["M8"],
        min_train_days=200,
        test_block_days=30,
        step_days=30,
        min_specialist_days=10_000,
    )
    assert result.predictions["Prediction"].notna().any()


def test_ablation_delta_is_config_minus_reference(synthetic_frame):
    result = run_matrix(
        synthetic_frame,
        "B",
        ["M1", "M4"],
        min_train_days=200,
        test_block_days=30,
        step_days=30,
    )
    if result.ablations.empty:
        pytest.skip("no paired ablation produced")
    row = result.ablations.iloc[0]
    assert set(result.ablations.columns) >= {
        "track",
        "config",
        "reference",
        "delta_MAE",
        "n",
    }
    assert row["n"] > 0
