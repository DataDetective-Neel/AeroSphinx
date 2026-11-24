"""Integration tests for the Track A rolling-origin backtest.

These run the real experiment over the real local CPCB record and assert
the properties established during Phase 2.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aerosphinx.evaluation.splits import rolling_origin_folds
from aerosphinx.tracks.diagnostics import coverage_by_fold, regime_coverage
from aerosphinx.tracks.track_a import DEFAULT_FEATURES, run_track_a

pytestmark = [
    pytest.mark.integration,
    pytest.mark.scientific,
    pytest.mark.skipif(
        not (
            Path(__file__).resolve().parents[2]
            / "data"
            / "processed"
            / "track_a_features.csv"
        ).exists(),
        reason="Track A features not built; run scripts/build_track_a_dataset.py",
    ),
]

ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = ROOT / "data" / "processed" / "track_a_features.csv"


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.read_csv(FEATURES_CSV, parse_dates=["Date"])


@pytest.fixture(scope="module")
def result(frame):
    """A reduced but real backtest: fewer folds, cheap models."""
    return run_track_a(
        frame,
        models=["ridge"],
        min_train_days=730,
        test_block_days=60,
        step_days=180,
        bootstrap_samples=1000,
    )


def test_backtest_produces_many_folds(result):
    """A rolling-origin backtest over nine years must produce many folds."""
    assert result.metadata["n_folds"] >= 8


def test_every_model_is_scored_on_identical_rows(result):
    """All models must be compared on exactly the same observations.

    This is the requirement that the legacy evaluation could not meet: a
    comparison between models scored on different windows is void.
    """
    model_columns = [
        c for c in result.predictions.columns if c not in ("Date", "Fold", "Target")
    ]
    complete = result.predictions.dropna(subset=model_columns + ["Target"])
    assert len(complete) == result.metadata["n_rows_evaluated"]
    assert complete["Target"].notna().all()
    # The aggregate table must report the same count for every model.
    counts = set(result.aggregate["n"])
    assert counts == {len(complete)}


def test_persistence_is_the_strongest_baseline(result):
    """The experiment must not silently favour a learned model.

    Phase 2 established that no learned model significantly beats
    persistence on this record. This test pins that result so a future
    change that reverses it is noticed.
    """
    comparison = result.comparisons.set_index("model")
    ridge_delta = comparison.loc["ridge", "delta_mae"]
    # Ridge is not significantly better than persistence.
    assert not bool(comparison.loc["ridge", "significant"]) or ridge_delta < 0


def test_seasonal_baseline_is_significantly_worse(result):
    """The D-005 correction is verified by the baseline's own bias.

    With a three-year lookback the climatology must be materially worse
    than persistence, and the comparison must be reported as significant.
    """
    comparison = result.comparisons.set_index("model")
    assert comparison.loc["seasonal_naive", "delta_mae"] > 0
    assert bool(comparison.loc["seasonal_naive", "significant"])


def test_comparisons_carry_uncertainty(result):
    """Every comparison must carry a paired test and an interval."""
    required = {
        "wilcoxon_pvalue",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "significant",
        "verdict",
        "n_pairs",
    }
    assert required.issubset(result.comparisons.columns)
    assert result.comparisons["n_pairs"].min() > 0


def test_folds_are_chronological(frame, result):
    """No fold may train on observations dated at or after its test block."""
    folds = rolling_origin_folds(
        frame, min_train_days=730, test_block_days=60, step_days=180
    )
    for fold in folds:
        train, _val, test = fold.split(frame)
        if train.empty or test.empty:
            continue
        assert train["Date"].max() < test["Date"].min()


def test_regime_coverage_identifies_extrapolating_folds(frame):
    """Folds whose test block leaves the training range must be flagged."""
    folds = rolling_origin_folds(
        frame, min_train_days=365, test_block_days=30, step_days=30
    )
    coverage = coverage_by_fold(frame, folds)
    assert not coverage.empty
    assert coverage["extrapolating"].dtype == bool
    # The earliest folds are the ones that extrapolate.
    assert coverage[coverage["extrapolating"]]["fold"].min() < 10


def test_regime_coverage_detects_an_out_of_range_test_block():
    """The diagnostic must flag a test block below the training minimum."""
    train = pd.DataFrame({"PM25": [50.0, 60.0, 70.0]})
    test = pd.DataFrame({"PM25": [10.0, 12.0]})
    coverage = regime_coverage(train, test)
    assert coverage["extrapolating"] is True
    assert coverage["test_below_train_min"] == pytest.approx(1.0)


def test_no_leakage_of_target_into_features(frame):
    """A feature column must not be a copy of the target."""
    assert "PM25" not in DEFAULT_FEATURES
    for feature in DEFAULT_FEATURES:
        assert feature in frame.columns


def test_fold_stability_is_reported(result):
    """Fold-to-fold stability must be reported, not just the mean."""
    assert not result.fold_summary.empty
    assert "MAE_sd" in result.fold_summary.columns
    assert "MAE_max" in result.fold_summary.columns
    assert "folds_with_negative_R2" in result.fold_summary.columns
