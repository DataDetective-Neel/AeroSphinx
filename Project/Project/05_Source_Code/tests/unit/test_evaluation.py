"""Unit tests for evaluation splits, metrics and uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerosphinx.evaluation.block_inference import (
    estimate_block_length,
    moving_block_bootstrap_ci,
)
from aerosphinx.evaluation.metrics import (
    AQI_LABELS,
    aqi_category,
    evaluate,
    metric_table,
    regression_metrics,
)
from aerosphinx.evaluation.splits import (
    assert_no_leakage,
    expanding_window_folds,
    rolling_origin_folds,
)
from aerosphinx.evaluation.uncertainty import (
    compare_models,
    paired_bootstrap_ci,
    paired_wilcoxon,
)

pytestmark = pytest.mark.unit


def daily_dates(n: int = 1000, start: str = "2016-01-01") -> pd.Series:
    return pd.Series(pd.date_range(start, periods=n, freq="D"))


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def test_folds_are_strictly_chronological():
    """No fold may train on data at or after its own test block."""
    dates = daily_dates(1000)
    for fold in rolling_origin_folds(dates, min_train_days=365, step_days=30):
        assert (
            fold.train_end
            < fold.validation_start
            < fold.validation_end
            < fold.test_start
        )
        assert fold.test_end > fold.test_start


def test_training_window_expands():
    """The rolling-origin protocol must expand, not slide, the training set."""
    dates = daily_dates(1000)
    folds = rolling_origin_folds(dates, min_train_days=365, step_days=30)
    for previous, current in zip(folds, folds[1:]):
        assert current.train_start == previous.train_start
        assert current.train_end > previous.train_end


def test_fold_split_returns_disjoint_blocks():
    """The three blocks of a fold must not overlap."""
    dates = daily_dates(1000)
    frame = pd.DataFrame({"Date": dates, "PM25": np.arange(1000, dtype=float)})
    for fold in rolling_origin_folds(dates, min_train_days=365, step_days=30):
        train, validation, test = fold.split(frame)
        assert not train.empty and not test.empty
        assert (
            train["Date"].max() < validation["Date"].min()
            if not validation.empty
            else True
        )
        assert (
            validation["Date"].max() < test["Date"].min()
            if not validation.empty
            else True
        )


def test_folds_stop_at_the_end_of_the_record():
    """The last fold must not extend beyond the available data."""
    dates = daily_dates(1000)
    folds = rolling_origin_folds(dates, min_train_days=365, step_days=30)
    assert folds[-1].test_end <= dates.max()


def test_no_leakage_assertion_catches_a_bad_fold():
    """A fold whose training block overlaps its test block must be rejected."""
    dates = daily_dates(1000)
    frame = pd.DataFrame({"Date": dates, "PM25": np.arange(1000, dtype=float)})
    folds = rolling_origin_folds(dates, min_train_days=365, step_days=30)
    good = folds[0]
    assert_no_leakage(good, frame)

    from aerosphinx.evaluation.splits import Fold

    bad = Fold(
        index=0,
        train_start=good.train_start,
        train_end=good.test_end,  # deliberately overlaps the test block
        validation_start=None,
        validation_end=None,
        test_start=good.test_start,
        test_end=good.test_end,
    )
    with pytest.raises(AssertionError, match="leaks"):
        assert_no_leakage(bad, frame)


def test_expanding_window_helper_disables_validation():
    """The no-validation helper must produce folds without a validation block."""
    dates = daily_dates(1000)
    folds = expanding_window_folds(dates, min_train_days=365, step_days=30)
    assert folds
    assert all(fold.validation_start is None for fold in folds)


def test_invalid_split_parameters_are_rejected():
    """Nonsensical split parameters must fail loudly."""
    dates = daily_dates(1000)
    with pytest.raises(ValueError):
        rolling_origin_folds(dates, min_train_days=0)
    with pytest.raises(ValueError):
        rolling_origin_folds(dates, test_block_days=0)
    with pytest.raises(ValueError):
        rolling_origin_folds(dates, step_days=0)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_aqi_bands_match_the_cpcb_pm25_subindex():
    """The category boundaries must match the CPCB PM2.5 bands exactly."""
    assert aqi_category([0])[0] == "Good"
    assert aqi_category([29.9])[0] == "Good"
    assert aqi_category([30])[0] == "Satisfactory"
    assert aqi_category([60])[0] == "Moderate"
    assert aqi_category([90])[0] == "Poor"
    assert aqi_category([120])[0] == "Very Poor"
    assert aqi_category([250])[0] == "Severe"
    assert aqi_category([500])[0] == "Severe"
    assert list(AQI_LABELS) == [
        "Good",
        "Satisfactory",
        "Moderate",
        "Poor",
        "Very Poor",
        "Severe",
    ]


def test_regression_metrics_on_a_perfect_prediction():
    """A perfect prediction must score zero error and R2 of one."""
    y = np.array([10.0, 20.0, 30.0, 40.0])
    scores = regression_metrics(y, y)
    assert scores["MAE"] == pytest.approx(0.0)
    assert scores["RMSE"] == pytest.approx(0.0)
    assert scores["R2"] == pytest.approx(1.0)
    assert scores["Mean_Bias"] == pytest.approx(0.0)


def test_mean_bias_sign_convention():
    """A positive bias must mean over-prediction."""
    y = np.array([10.0, 20.0, 30.0])
    scores = regression_metrics(y, y + 5.0)
    assert scores["Mean_Bias"] == pytest.approx(5.0)


def test_aqi_macro_f1_is_reported_and_can_differ_from_accuracy():
    """Macro-F1 must be present and must expose majority-class collapse.

    A model that always predicts the majority category has high accuracy
    and near-zero macro-F1.
    """
    y = np.concatenate([np.full(90, 100.0), np.full(10, 10.0)])
    always_majority = np.full(100, 100.0)
    scores = evaluate(y, always_majority)
    assert scores["AQI_Accuracy"] == pytest.approx(0.90)
    assert scores["AQI_Macro_F1"] < 0.5


def test_metric_table_groups_by_a_supplied_label():
    """The grouped table must produce one row per model and group."""
    y = np.array([10.0, 20.0, 30.0, 40.0])
    predictions = {"a": y, "b": y + 1}
    groups = pd.Series(["x", "x", "y", "y"])
    table = metric_table(y, predictions, groups)
    assert len(table) == 4
    assert set(table["group"]) == {"x", "y"}


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------


def test_paired_test_detects_a_real_improvement():
    """A clearly better model must be detected as significantly better."""
    rng = np.random.default_rng(0)
    y = rng.normal(100, 20, 400)
    better = y + rng.normal(0, 5, 400)
    worse = y + rng.normal(0, 25, 400)

    test = paired_wilcoxon(y, better, worse)
    assert test["delta"] < 0
    assert test["pvalue"] < 0.05

    boot = paired_bootstrap_ci(y, better, worse, n_samples=500, seed=42)
    assert boot["ci_high"] < 0, "the interval must exclude zero"
    assert boot["significant"] is True


def test_paired_test_reports_no_difference_for_identical_models():
    """Identical models must not be reported as significantly different."""
    rng = np.random.default_rng(1)
    y = rng.normal(100, 20, 200)
    p = y + rng.normal(0, 5, 200)
    test = paired_wilcoxon(y, p, p)
    assert test["pvalue"] == 1.0
    assert test["significant"] is False


def test_bootstrap_interval_is_reproducible():
    """The same seed must give the same interval."""
    rng = np.random.default_rng(2)
    y = rng.normal(100, 20, 300)
    a = y + rng.normal(0, 5, 300)
    b = y + rng.normal(0, 10, 300)
    first = paired_bootstrap_ci(y, a, b, n_samples=1000, seed=42)
    second = paired_bootstrap_ci(y, a, b, n_samples=1000, seed=42)
    assert first["ci_low"] == second["ci_low"]
    assert first["ci_high"] == second["ci_high"]


def test_compare_models_marks_the_baseline_as_reference():
    """The baseline row must be labelled as the reference."""
    y = np.array([10.0, 20.0, 30.0, 40.0] * 20)
    predictions = {"persistence": y, "other": y + 1}
    table = compare_models(y, predictions, baseline="persistence", n_samples=200)
    assert set(table["model"]) == {"persistence", "other"}
    assert table[table.model == "persistence"]["verdict"].iloc[0] == "reference"


def test_compare_models_requires_the_baseline():
    """A missing baseline must be an error, not a silent default."""
    with pytest.raises(KeyError):
        compare_models(
            np.array([1.0, 2.0]), {"a": np.array([1.0, 2.0])}, baseline="missing"
        )


def test_constant_error_series_is_not_blocked():
    """A series with no variance has no serial dependence to break.

    Pandas divides by its zero standard deviation while computing the
    autocorrelation, so this used to emit a RuntimeWarning on every test run.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert estimate_block_length(np.ones(120)) == 1
        assert estimate_block_length(np.zeros(60)) == 1


def test_block_bootstrap_interval_for_a_constant_series_is_degenerate():
    values = np.ones(120)
    result = moving_block_bootstrap_ci(values, values, values)
    assert result["delta"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0)
    assert result["ci_high"] == pytest.approx(0.0)
    assert result["significant"] is False
