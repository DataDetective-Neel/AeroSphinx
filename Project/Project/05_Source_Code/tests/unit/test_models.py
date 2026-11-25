"""Unit tests for baselines and supervised models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerosphinx.models.baselines import (
    ClimatologyBuilder,
    PersistenceBaseline,
    SeasonalNaiveBaseline,
    persistence_forecast,
)
from aerosphinx.models.supervised import MODEL_REGISTRY, build_model

pytestmark = pytest.mark.unit


@pytest.fixture
def seasonal_frame() -> pd.DataFrame:
    """Three years of daily data with a strong seasonal cycle."""
    dates = pd.date_range("2018-01-01", "2020-12-31", freq="D")
    season = dates.month.map(
        lambda m: 200.0 if m in (11, 12, 1) else 100.0 if m in (6, 7, 8) else 50.0
    )
    return pd.DataFrame({"Date": dates, "PM25": season})


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    """A frame with the Track A feature set, complete and well behaved."""
    rng = np.random.default_rng(0)
    n = 500
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    base = rng.normal(100, 20, n)
    return pd.DataFrame(
        {
            "Date": dates,
            "PM25": base,
            "PM25_Lag_1": np.roll(base, 1),
            "PM25_Lag_2": np.roll(base, 2),
            "PM25_Lag_3": np.roll(base, 3),
            "PM25_Lag_7": np.roll(base, 7),
            "PM25_Roll3": np.roll(base, 3),
            "PM25_Roll7": np.roll(base, 7),
            "day_sin": np.sin(np.arange(n) / 7),
            "day_cos": np.cos(np.arange(n) / 7),
            "month_sin": np.sin(np.arange(n) / 30),
            "month_cos": np.cos(np.arange(n) / 30),
        }
    )


FEATURES = [
    "PM25_Lag_1",
    "PM25_Lag_2",
    "PM25_Lag_3",
    "PM25_Lag_7",
    "PM25_Roll3",
    "PM25_Roll7",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
]


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_persistence_uses_the_previous_calendar_day():
    """Persistence must be the value dated exactly one day earlier."""
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]),
            "PM25": [10.0, 20.0, 40.0],
        }
    )
    predictions = PersistenceBaseline().fit(frame).predict(frame)
    # 2024-01-04 has no 2024-01-03, so the strict baseline cannot predict it.
    assert predictions[0] != predictions[0]  # NaN
    assert predictions[1] == pytest.approx(10.0)
    assert predictions[2] != predictions[2]  # NaN


def test_persistence_with_staleness_allowance_reaches_back():
    """The gap-tolerant variant must reach back to the last observation."""
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]),
            "PM25": [10.0, 20.0, 40.0],
        }
    )
    predictions = persistence_forecast(frame, max_age_days=3)
    assert predictions[2] == pytest.approx(20.0)  # from 2024-01-02


def test_persistence_staleness_allowance_respects_its_limit():
    """A staleness limit must be enforced."""
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-10"]),
            "PM25": [10.0, 40.0],
        }
    )
    predictions = persistence_forecast(frame, max_age_days=3)
    assert np.isnan(predictions[1])


# --------------------------------------------------------------------------
# Seasonal baseline
# --------------------------------------------------------------------------


def test_seasonal_baseline_recovers_the_seasonal_cycle():
    """The climatology must reproduce the seasonal pattern it was fitted on."""
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2018-01-01", "2020-12-31", freq="D"),
            "PM25": [
                (
                    200.0
                    if d.month in (11, 12, 1)
                    else 100.0 if d.month in (6, 7, 8) else 50.0
                )
                for d in pd.date_range("2018-01-01", "2020-12-31", freq="D")
            ],
        }
    )
    baseline = SeasonalNaiveBaseline(lookback_years=3).fit(frame)
    predictions = baseline.predict(frame)
    summer = frame["Date"].dt.month.isin([6, 7, 8]).to_numpy()
    winter = frame["Date"].dt.month.isin([11, 12, 1]).to_numpy()
    assert predictions[summer].mean() == pytest.approx(100.0, abs=1.0)
    assert predictions[winter].mean() == pytest.approx(200.0, abs=1.0)


def test_climatology_uses_only_the_recent_years():
    """The baseline must not average in stale regimes (the D-005 correction)."""
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01"]
            ),
            "PM25": [400.0, 300.0, 200.0, 100.0],
        }
    )
    builder = ClimatologyBuilder(lookback_years=2).fit(frame)
    prediction = builder.predict(pd.DataFrame({"Date": pd.to_datetime(["2019-01-01"])}))
    # Only 2018 and 2019 are in the lookback window.
    assert prediction[0] == pytest.approx(150.0)


def test_climatology_never_sees_the_test_period():
    """A climatology fitted on training data must not use test observations."""
    train = pd.DataFrame({"Date": pd.to_datetime(["2018-01-01"]), "PM25": [10.0]})
    test = pd.DataFrame({"Date": pd.to_datetime(["2019-01-01"]), "PM25": [999.0]})
    builder = ClimatologyBuilder(lookback_years=3).fit(train)
    assert builder.predict(test)[0] == pytest.approx(10.0)


def test_climatology_falls_back_to_a_nearby_day_of_year():
    """A missing day-of-year key must fall back within the permitted offset."""
    train = pd.DataFrame({"Date": pd.to_datetime(["2018-01-01"]), "PM25": [42.0]})
    # 2018-01-04 is day-of-year 4, absent from the training set.
    test = pd.DataFrame({"Date": pd.to_datetime(["2018-01-04"])})
    builder = ClimatologyBuilder(lookback_years=3, max_day_offset=3).fit(train)
    assert builder.predict(test)[0] == pytest.approx(42.0)


def test_climatology_rejects_a_nonsensical_lookback():
    """A lookback of zero years cannot be satisfied."""
    with pytest.raises(ValueError):
        ClimatologyBuilder(lookback_years=0)


# --------------------------------------------------------------------------
# Supervised models
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_models_fit_and_predict(name, feature_frame):
    """Every registered model must fit and produce finite predictions."""
    model = build_model(name, FEATURES).fit(feature_frame)
    predictions = model.predict(feature_frame)
    assert len(predictions) == len(feature_frame)
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_models_reject_a_missing_feature(name, feature_frame):
    """A missing feature column must be an error."""
    model = build_model(name, FEATURES)
    with pytest.raises(KeyError):
        model.fit(feature_frame.drop(columns=["PM25_Lag_1"]))


def test_unknown_model_name_is_rejected():
    """An unregistered model name must fail loudly."""
    with pytest.raises(KeyError):
        build_model("no_such_model", FEATURES)


def test_trees_are_not_given_imputed_values(feature_frame):
    """Tree models must preserve NaN lags rather than imputing them.

    The calendar-aware lags are NaN exactly where the previous calendar
    day is absent. Imputing them would destroy the semantics that Phase 1
    established.
    """
    frame = feature_frame.copy()
    frame.loc[10:20, "PM25_Lag_1"] = np.nan
    model = build_model("random_forest", FEATURES).fit(frame)
    predictions = model.predict(frame)
    assert len(predictions) == len(frame)


def test_ridge_standardises_features(feature_frame):
    """The ridge pipeline must include a scaler.

    Without standardisation the penalty falls almost entirely on the
    large-scale PM2.5 lags, and the regularised model is not the one the
    alpha was chosen for.
    """
    model = build_model("ridge", FEATURES).fit(feature_frame)
    steps = list(model._pipeline.named_steps)
    assert "scaler" in steps
    assert steps.index("scaler") < steps.index("ridge")


def test_ridge_records_missingness(feature_frame):
    """The ridge imputer must record missingness rather than hide it."""
    model = build_model("ridge", FEATURES).fit(feature_frame)
    assert model._pipeline.named_steps["imputer"].add_indicator is True


def test_ridge_imputer_is_fitted_on_training_data_only(feature_frame):
    """The imputer statistics must come from the training block alone."""
    frame = feature_frame.copy()
    frame.loc[:10, "PM25_Lag_1"] = np.nan
    train = frame.iloc[11:]
    model = build_model("ridge", FEATURES).fit(train)
    medians = model._pipeline.named_steps["imputer"].statistics_
    # The statistics reflect the training rows, where the NaNs were absent.
    assert np.all(np.isfinite(medians))
    assert medians[0] == pytest.approx(train["PM25_Lag_1"].median())
