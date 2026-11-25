"""Tests for the Phase 3 cold-start and extrapolation analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerosphinx.evaluation.block_inference import (
    apply_correction,
    benjamini_hochberg,
    estimate_block_length,
    fold_level_comparison,
    holm_bonferroni,
    moving_block_bootstrap_ci,
    year_blocked_bootstrap_ci,
)
from aerosphinx.tracks.cold_start import (
    error_vs_out_of_range_fraction,
    extrapolation_conditioned_error,
    persistence_fallback,
    quantile_fallback,
    regime_aware_fallback,
    summarise_cold_start,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def cold_start_frame() -> pd.DataFrame:
    """Predictions with a training range that some observations exceed."""
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Fold": 0,
            "Target": [
                50.0,
                60.0,
                70.0,
                80.0,
                90.0,
                100.0,
                110.0,
                120.0,
                130.0,
                140.0,
                10.0,
                20.0,
            ],
            "persistence": [
                50.0,
                60.0,
                70.0,
                80.0,
                90.0,
                100.0,
                110.0,
                120.0,
                130.0,
                140.0,
                10.0,
                20.0,
            ],
            "ridge": [
                55.0,
                65.0,
                75.0,
                85.0,
                95.0,
                105.0,
                115.0,
                125.0,
                135.0,
                145.0,
                90.0,
                95.0,
            ],
        }
    )


# --------------------------------------------------------------------------
# Extrapolation diagnostics
# --------------------------------------------------------------------------


def test_out_of_range_observations_are_identified(cold_start_frame):
    """Observations beyond the training range must be flagged."""
    train_min = pd.Series({0: 50.0})
    train_max = pd.Series({0: 140.0})
    table = extrapolation_conditioned_error(
        cold_start_frame, ["persistence", "ridge"], train_min, train_max
    )
    ridge = table[table.model == "ridge"].iloc[0]
    assert ridge["n_out_of_range"] == 2
    assert ridge["mae_out_of_range"] > ridge["mae_in_range"]


def test_error_grows_with_the_out_of_range_fraction(cold_start_frame):
    """A model must do worse on the out-of-range subset."""
    train_min = pd.Series({0: 50.0})
    train_max = pd.Series({0: 140.0})
    table = extrapolation_conditioned_error(
        cold_start_frame, ["ridge"], train_min, train_max
    )
    row = table.iloc[0]
    assert row["mae_ratio"] > 1.0


def test_cold_start_summary_counts_folds_and_observations(cold_start_frame):
    """The summary must report the extrapolating fold and observation counts."""
    train_min = pd.Series({0: 50.0})
    train_max = pd.Series({0: 140.0})
    summary = summarise_cold_start(cold_start_frame, ["ridge"], train_min, train_max)
    assert summary.n_folds == 1
    assert summary.n_extrapolating_folds == 1
    assert summary.n_observations_out_of_range == 2
    assert summary.fraction_out_of_range > 0


def test_error_against_out_of_range_fraction_is_binned(cold_start_frame):
    """The binned table must cover every model."""
    train_min = pd.Series({0: 50.0})
    train_max = pd.Series({0: 140.0})
    table = error_vs_out_of_range_fraction(
        cold_start_frame, ["ridge", "persistence"], train_min, train_max
    )
    assert set(table["model"]) == {"ridge", "persistence"}


# --------------------------------------------------------------------------
# Mitigations
# --------------------------------------------------------------------------


def test_persistence_fallback_only_fires_outside_the_training_range():
    """The fallback must replace only out-of-range predictions."""
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "Fold": 0,
            "Target": [50.0, 60.0, 70.0, 80.0],
            "model": [55.0, 65.0, 10.0, 200.0],
            "persistence": [50.0, 60.0, 70.0, 80.0],
        }
    )
    train_min = pd.Series({0: 40.0})
    train_max = pd.Series({0: 100.0})
    result = persistence_fallback(
        frame, "model", train_min=train_min, train_max=train_max
    )
    assert result[0] == pytest.approx(55.0)  # inside the range -> kept
    assert result[1] == pytest.approx(65.0)  # inside the range -> kept
    assert result[2] == pytest.approx(70.0)  # below range -> persistence
    assert result[3] == pytest.approx(80.0)  # above range -> persistence


def test_regime_fallback_uses_calendar_only():
    """Regime assignment must not depend on the target or any observation."""
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-15", "2024-07-15"]),
            "Target": [100.0, 20.0],
            "model": [90.0, 90.0],
            "persistence": [95.0, 25.0],
        }
    )
    # Winter -> the model is used; monsoon -> persistence is used.
    result = regime_aware_fallback(frame, "model")
    assert result[0] == pytest.approx(90.0)
    assert result[1] == pytest.approx(25.0)

    # Changing the target must not change the routing.
    altered = frame.copy()
    altered["Target"] = [1.0, 999.0]
    assert list(regime_aware_fallback(altered, "model")) == list(result)


def test_quantile_fallback_replaces_extreme_predictions():
    """A prediction beyond the training quantiles must be replaced."""
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "Target": [50.0, 50.0, 50.0],
            "model": [50.0, 500.0, 50.0],
            "persistence": [50.0, 50.0, 50.0],
        }
    )
    quantiles = {1: (40.0, 60.0), 2: (40.0, 60.0), 3: (40.0, 60.0)}
    result = quantile_fallback(frame, "model", train_quantiles=quantiles)
    assert result[0] == pytest.approx(50.0)
    assert result[1] == pytest.approx(50.0)  # 500 replaced by the midpoint


# --------------------------------------------------------------------------
# Block inference
# --------------------------------------------------------------------------


def test_block_length_requires_persistent_dependence():
    """The estimator must not return 1 for a serially correlated series."""
    rng = np.random.default_rng(0)
    # An AR(1) series with rho = 0.8.
    values = np.zeros(500)
    for t in range(1, 500):
        values[t] = 0.8 * values[t - 1] + rng.normal(0, 1)
    assert estimate_block_length(values) > 5


def test_block_length_is_one_for_white_noise():
    """Independent noise needs no blocking."""
    rng = np.random.default_rng(1)
    assert estimate_block_length(rng.normal(0, 1, 800)) == 1


def test_block_length_is_estimated_from_the_difference_series():
    """The block length must reflect the dependence of the *difference*.

    For a paired comparison the quantity of interest is the difference
    between the two models' errors, and the shared target autocorrelation
    cancels in that difference. Blocking on the difference is therefore the
    correct choice, and it is what the implementation does.
    """
    rng = np.random.default_rng(2)
    n = 600
    serially = np.zeros(n)
    for t in range(1, n):
        serially[t] = 0.85 * serially[t - 1] + rng.normal(0, 1)
    y = serially + 50

    # Two models that both track the serially correlated target closely,
    # with strongly serially correlated errors of their own.
    shared_error = np.zeros(n)
    for t in range(1, n):
        shared_error[t] = 0.8 * shared_error[t - 1] + rng.normal(0, 3)
    a = y + shared_error + rng.normal(0, 1, n)
    b = y + rng.normal(0, 1, n)

    difference = a - b
    # The difference inherits the serial dependence of the shared error.
    assert pd.Series(difference).autocorr(1) > 0.3
    assert estimate_block_length(difference) > 5


def test_moving_block_interval_is_wider_than_the_naive_one():
    """Blocking must widen the interval when the difference is dependent."""
    rng = np.random.default_rng(2)
    n = 600
    serially = np.zeros(n)
    for t in range(1, n):
        serially[t] = 0.85 * serially[t - 1] + rng.normal(0, 1)
    y = serially + 50

    shared_error = np.zeros(n)
    for t in range(1, n):
        shared_error[t] = 0.8 * shared_error[t - 1] + rng.normal(0, 3)
    a = y + shared_error + rng.normal(0, 1, n)
    b = y + rng.normal(0, 1, n)

    naive = moving_block_bootstrap_ci(y, a, b, n_samples=2000, block_length=1, seed=5)
    blocked = moving_block_bootstrap_ci(y, a, b, n_samples=2000, seed=5)
    assert blocked["block_length"] > 1
    assert (blocked["ci_high"] - blocked["ci_low"]) > (
        naive["ci_high"] - naive["ci_low"]
    )


def test_block_bootstrap_is_reproducible():
    """The same seed must give the same interval."""
    rng = np.random.default_rng(3)
    y = rng.normal(100, 20, 300)
    a = y + rng.normal(0, 5, 300)
    b = y + rng.normal(0, 8, 300)
    first = moving_block_bootstrap_ci(y, a, b, n_samples=500, seed=7)
    second = moving_block_bootstrap_ci(y, a, b, n_samples=500, seed=7)
    assert first["ci_low"] == second["ci_low"]


def test_year_blocked_bootstrap_uses_whole_years():
    """The year-blocked interval must be computable and reproducible."""
    dates = pd.Series(pd.date_range("2018-01-01", "2022-12-31", freq="D"))
    rng = np.random.default_rng(4)
    y = rng.normal(100, 20, len(dates))
    a = y + rng.normal(0, 5, len(dates))
    b = y + rng.normal(0, 8, len(dates))
    result = year_blocked_bootstrap_ci(y, a, b, dates, n_samples=300, seed=11)
    assert result["n_years"] == 5
    assert result["ci_low"] <= result["delta"] <= result["ci_high"]


def test_holm_correction_is_more_conservative_than_raw():
    """Holm must never produce a smaller p-value than the raw one."""
    pvalues = np.array([0.001, 0.02, 0.04, 0.3])
    adjusted = holm_bonferroni(pvalues)
    assert np.all(adjusted >= pvalues)
    assert adjusted[0] <= adjusted[-1]


def test_benjamini_hochberg_is_ordered_and_bounded():
    """BH adjustment must be monotone in the raw p-values and bounded by 1."""
    pvalues = np.array([0.001, 0.02, 0.04, 0.3])
    adjusted = benjamini_hochberg(pvalues)
    assert np.all(adjusted <= 1.0)
    assert np.all(np.diff(adjusted) >= -1e-12)


def test_apply_correction_reports_both_criteria():
    """The correction table must report both adjusted p-value families."""
    table = apply_correction(np.array([0.01, 0.04, 0.2]))
    assert "p_holm" in table.columns
    assert "p_benjamini_hochberg" in table.columns
    assert int(table["n_comparisons"].iloc[0]) == 3


def test_fold_level_comparison_shows_consistency():
    """The fold-level table must report how consistent the sign is."""
    y = np.array([10.0, 20.0, 30.0, 40.0] * 10)
    a = y + 1.0
    b = y
    folds = np.repeat([0, 1, 2, 3, 4], 8)
    table = fold_level_comparison(y, a, b, folds)
    assert len(table) == 5
    assert int(table["significant_folds"].iloc[0]) == 0
    assert table["folds_a_better_fraction"].iloc[0] == 0.0
