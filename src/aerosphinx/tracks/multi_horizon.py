"""Multi-horizon forecasting for AeroSphinx Track A.

Phase 2 established that no learned model significantly beats persistence at
**one-day** horizon. That leaves one obvious question, and it is the
central experiment of Phase 3:

> Does skill relative to persistence **grow** with horizon?

Persistence copies the most recent observation. Its error grows with
horizon because the true state drifts. A model with seasonal and temporal
context might degrade more slowly. Or it might not.

Two methods are implemented, and they are genuinely different:

* **Direct** - a separate model per horizon, trained with the target
  shifted *h* days forward. Each horizon is fitted independently, so the
  model can specialise.
* **Recursive** - a single one-day model whose predictions are fed back as
  later lag inputs. One model, applied repeatedly, with error compounding.

The **horizon-appropriate persistence baseline** is the value observed *h*
calendar days before the target. Using the one-day persistence baseline to
judge a day-5 forecast would be wrong: it would flatter the learned models
by comparing them against a baseline that is itself decaying.

Availability contract
---------------------
A forecast at horizon *h* may use only information available at the
**forecast issue date**:

- PM2.5 observed on or before the issue date;
- calendar features of the target date;
- environmental covariates **only if genuinely available at issue time**.

No future observed AOD or MERRA-2 is ever used. The issue date is recorded
on every forecast so the contract can be verified after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aerosphinx.models.baselines import persistence_forecast
from aerosphinx.models.supervised import build_model

__all__ = [
    "HorizonForecast",
    "MultiHorizonResult",
    "run_multi_horizon",
    "direct_forecast",
    "recursive_forecast",
]


@dataclass
class HorizonForecast:
    """One forecast at one horizon."""

    date: pd.Timestamp
    issue_date: pd.Timestamp
    horizon: int
    fold: int
    actual: float
    predicted: float
    baseline: float

    def as_dict(self) -> dict:
        return {
            "Date": self.date,
            "issue_date": self.issue_date,
            "horizon": self.horizon,
            "fold": self.fold,
            "actual": self.actual,
            "predicted": self.predicted,
            "baseline": self.baseline,
            "absolute_error": (
                abs(self.predicted - self.actual)
                if np.isfinite(self.actual)
                else np.nan
            ),
            "baseline_absolute_error": (
                abs(self.baseline - self.actual) if np.isfinite(self.actual) else np.nan
            ),
        }


@dataclass
class MultiHorizonResult:
    """Outcome of a multi-horizon backtest."""

    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    comparisons: pd.DataFrame
    within_issue: pd.DataFrame = field(default_factory=pd.DataFrame)
    lag_reference: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "metrics": self.metrics.to_dict(orient="records"),
            "comparisons": self.comparisons.to_dict(orient="records"),
        }


def _horizon_target(
    frame: pd.DataFrame, horizon: int, target: str, date_column: str
) -> pd.DataFrame:
    """Frame whose target is the value ``horizon`` days ahead.

    The row dated *t* carries the target observed at *t + horizon*. The
    features remain those of date *t*, which is what a model issuing a
    forecast for *t + horizon* would have.
    """
    data = frame.sort_values(date_column).copy()
    lookup = data.set_index(date_column)[target]
    target_dates = data[date_column] + pd.Timedelta(days=horizon)
    data["_target_future"] = target_dates.map(lookup)
    data["_target_date"] = target_dates
    return data


def _make_features(
    frame: pd.DataFrame,
    date_column: str,
    target: str,
    lags: tuple[int, ...] = (1, 2, 3, 7),
    rolling_windows: tuple[int, ...] = (3, 7),
) -> pd.DataFrame:
    """Calendar-aware features for the *issue* date of each row.

    Every lag is calendar-aware: it is the observation dated exactly *k*
    calendar days before the issue date, or missing.
    """
    from aerosphinx.features.lags import calendar_lag, calendar_rolling_stats

    data = frame.sort_values(date_column).reset_index(drop=True).copy()
    for lag in lags:
        data[f"PM25_Lag_{lag}"] = calendar_lag(data, target, lag).to_numpy()
    for window in rolling_windows:
        stats = calendar_rolling_stats(data, target, window)
        data[f"PM25_Roll{window}"] = stats[f"PM25_roll{window}_mean"].to_numpy()
        data[f"PM25_Roll{window}_n"] = stats[f"PM25_roll{window}_n"].to_numpy()

    dates = pd.to_datetime(data[date_column])
    day_of_week = dates.dt.day_of_week
    data["day_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    data["day_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    month = dates.dt.month
    data["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return data


def direct_forecast(
    train: pd.DataFrame,
    issue_frame: pd.DataFrame,
    horizon: int,
    features: list[str],
    model_name: str,
    target: str = "PM25",
    date_column: str = "Date",
) -> np.ndarray:
    """Train a horizon-specific model and predict the horizon-ahead target.

    ``train`` must carry a ``_target_future`` column holding the value
    observed ``horizon`` days after each row's date.
    """
    usable = train.dropna(subset=features + ["_target_future"])
    if usable.empty:
        return np.full(len(issue_frame), np.nan)

    model = build_model(model_name, features, target="_target_future")
    model.fit(usable)
    return model.predict(issue_frame)


def recursive_forecast(
    train: pd.DataFrame,
    issue_frame: pd.DataFrame,
    horizon: int,
    features: list[str],
    model_name: str,
    target: str = "PM25",
    date_column: str = "Date",
) -> np.ndarray:
    """Apply a one-day model repeatedly, feeding predictions back.

    The model is trained on the one-day-ahead target and then applied
    iteratively. Its predictions replace the observed values in the lag
    features for subsequent steps, which is what makes the forecast
    recursive rather than a re-estimation.

    Only the *forecast issue date* history is used. No observation dated
    after the issue date is ever consulted.
    """
    usable = train.dropna(subset=features + ["_target_future"])
    if usable.empty:
        return np.full(len(issue_frame), np.nan)

    # The recursive model predicts the *next* day from the current day's
    # features, so it is trained on the one-day-ahead target.
    model = build_model(model_name, features, target="_target_future")
    model.fit(usable)

    predictions = np.full(len(issue_frame), np.nan)
    for position, (_, row) in enumerate(issue_frame.iterrows()):
        working = row.copy()
        # The lag values as they stand at the issue date.
        lag1 = working.get("PM25_Lag_1", np.nan)
        lag2 = working.get("PM25_Lag_2", np.nan)

        for step in range(1, horizon + 1):
            x = working[features].to_frame().T.astype(float)
            predicted = float(model.predict(x)[0])
            if step == horizon:
                predictions[position] = predicted

            # Feed the prediction back for the next step: the value just
            # predicted becomes "yesterday" for the following day.
            working["PM25_Lag_1"] = predicted
            working["PM25_Lag_2"] = lag1
            working["PM25_Lag_3"] = lag2
            if "PM25_Roll3" in working.index:
                working["PM25_Roll3"] = np.nanmean([predicted, lag1, lag2])
            if "PM25_Roll7" in working.index:
                working["PM25_Roll7"] = np.nan
            lag1, lag2 = predicted, lag1

    return predictions


def persistence_error_by_lag(
    frame: pd.DataFrame,
    target_dates,
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5),
    target: str = "PM25",
    date_column: str = "Date",
) -> pd.DataFrame:
    """Persistence error at each lag, on **identical target dates**.

    The multi-horizon backtest issues one forecast per issue date, so each
    horizon forecasts a *different* target date. Comparing horizons
    directly therefore confounds horizon difficulty with target-date
    difficulty.

    This function removes the confound: for a fixed set of target dates, it
    evaluates persistence at every lag, so the only thing that varies is
    the lag. It is the reference against which the horizon effect should
    be read.
    """
    lookup = frame.set_index(date_column)[target]
    target_dates = pd.to_datetime(pd.Series(target_dates)).dropna().unique()

    rows = []
    for target_date in target_dates:
        for horizon in horizons:
            source = target_date - pd.Timedelta(days=horizon)
            if source not in lookup.index or target_date not in lookup.index:
                continue
            actual = lookup.loc[target_date]
            predicted = lookup.loc[source]
            if not (np.isfinite(actual) and np.isfinite(predicted)):
                continue
            rows.append(
                {
                    "target_date": target_date,
                    "horizon": int(horizon),
                    "abs_error": abs(predicted - actual),
                }
            )

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    summary = (
        table.groupby("horizon")["abs_error"]
        .agg(["size", "mean", "std"])
        .reset_index()
        .rename(columns={"size": "n", "mean": "mae", "std": "sd"})
    )
    reference = summary.loc[summary["horizon"] == summary["horizon"].min(), "mae"]
    if len(reference):
        summary["increase_vs_h1"] = summary["mae"] - float(reference.iloc[0])
    return summary


def run_multi_horizon(
    frame: pd.DataFrame,
    features: list[str] | None = None,
    models: list[str] | None = None,
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5),
    methods: tuple[str, ...] = ("direct", "recursive"),
    min_train_days: int = 365,
    test_block_days: int = 30,
    step_days: int = 30,
    baseline: str = "persistence",
    bootstrap_samples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
    date_column: str = "Date",
    target: str = "PM25",
) -> MultiHorizonResult:
    """Run the multi-horizon rolling-origin backtest.

    For each fold, the issue date is the **last day of the fold's test
    block**. From that single issue date, forecasts are made for the
    following ``horizon`` days. This mirrors an operational setting: one
    forecast issued at one time, covering the next several days.

    Only observations dated on or before the issue date are used, so the
    availability contract holds by construction.

    Parameters
    ----------
    frame:
        The daily series with the target column.
    features:
        Feature columns for the learned models.
    models:
        Names of supervised models to fit.
    horizons:
        Forecast horizons in days.
    methods:
        ``"direct"`` and/or ``"recursive"``.
    baseline:
        The reference model for paired comparisons.
    """
    from aerosphinx.evaluation.block_inference import (
        apply_correction,
        moving_block_bootstrap_ci,
        year_blocked_bootstrap_ci,
    )
    from aerosphinx.evaluation.metrics import evaluate
    from aerosphinx.evaluation.splits import rolling_origin_folds

    data = frame.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(date_column).reset_index(drop=True)

    if target not in data.columns:
        raise KeyError(f"frame is missing the target column {target!r}")

    # The features are built here rather than taken from the input frame,
    # so the module is self-contained and the feature contract is
    # guaranteed to match the horizon semantics.
    if features is None:
        features = [
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
    models = list(models or ["ridge", "random_forest", "xgboost"])

    folds = rolling_origin_folds(
        data,
        min_train_days=min_train_days,
        validation_days=0,
        test_block_days=test_block_days,
        step_days=step_days,
        date_column=date_column,
    )
    if not folds:
        raise ValueError("no folds could be constructed")

    rows: list[dict] = []

    for fold in folds:
        train, _validation, test = fold.split(data, date_column)
        if train.empty or test.empty:
            continue

        # The issue date is the last day of the fold's test block. Only
        # observations up to and including that date may be used.
        issue_date = test[date_column].max()
        train = train[train[date_column] <= issue_date].copy()
        if train.empty:
            continue

        # The issue row: the observation on the issue date, carrying the
        # features available at that moment. Its features are built from
        # the frame truncated at the issue date, so no observation after
        # the issue date can enter them.
        issue_frame = data[data[date_column] <= issue_date].copy()
        issue_rows = _make_features(issue_frame, date_column, target)
        issue_rows = issue_rows[issue_rows[date_column] == issue_date]
        if issue_rows.empty:
            continue

        # Training rows for the one-day-ahead target, used by the recursive
        # method. The direct method uses a horizon-specific model instead.
        one_day_train = _horizon_target(data, 1, target, date_column)
        one_day_train = one_day_train[one_day_train[date_column] <= issue_date].copy()
        one_day_train = one_day_train[
            one_day_train["_target_date"] <= issue_date
        ].copy()
        one_day_train = _make_features(one_day_train, date_column, target)

        # Training targets for each horizon.
        for horizon in horizons:
            # The shifted training frame carries, for every date up to the
            # issue date, the value observed `horizon` days later. Only
            # rows whose future target is known (that is, whose target
            # date is at or before the issue date) may train the model:
            # training on a future target would leak the answer.
            shifted = _horizon_target(data, horizon, target, date_column)
            shifted = shifted[shifted[date_column] <= issue_date].copy()
            shifted = shifted[shifted["_target_date"] <= issue_date].copy()
            shifted = _make_features(shifted, date_column, target)

            target_date = issue_date + pd.Timedelta(days=horizon)
            actual = (
                data.loc[data[date_column] == target_date, target].iloc[0]
                if (data[date_column] == target_date).any()
                else np.nan
            )
            if not np.isfinite(actual):
                continue

            # The baseline at this horizon: the observation h calendar days
            # before the target date. The horizon-appropriate baseline is
            # essential - comparing a day-5 forecast against the day-1
            # persistence value would flatter the learned models.
            baseline_frame = data[data[date_column] <= target_date].copy()
            baseline_preds = persistence_forecast(
                baseline_frame,
                value_column=target,
                date_column=date_column,
                periods=horizon,
            )
            target_positions = np.flatnonzero(
                (baseline_frame[date_column] == target_date).to_numpy()
            )
            baseline_at_target = (
                float(baseline_preds[target_positions[0]])
                if target_positions.size
                else np.nan
            )

            for method in methods:
                for model_name in models:
                    if method == "direct":
                        predicted = direct_forecast(
                            shifted,
                            issue_rows,
                            horizon,
                            features,
                            model_name,
                            target=target,
                            date_column=date_column,
                        )[0]
                    else:
                        predicted = recursive_forecast(
                            one_day_train,
                            issue_rows,
                            horizon,
                            features,
                            model_name,
                            target=target,
                            date_column=date_column,
                        )[0]

                    rows.append(
                        {
                            "Date": target_date,
                            "issue_date": issue_date,
                            "horizon": horizon,
                            "fold": fold.index,
                            "method": method,
                            "model": model_name,
                            "actual": float(actual),
                            "predicted": float(predicted),
                            "baseline": float(baseline_at_target),
                        }
                    )

    if not rows:
        raise ValueError("no forecasts were produced")

    forecasts = pd.DataFrame(rows)

    # Per-horizon, per-model metrics against the horizon-appropriate
    # baseline, on rows where both are defined.
    metric_rows = []
    for (horizon, method, model_name), group in forecasts.groupby(
        ["horizon", "method", "model"]
    ):
        usable = group.dropna(subset=["predicted", "baseline", "actual"])
        if usable.empty:
            continue
        scores = evaluate(usable["actual"].to_numpy(), usable["predicted"].to_numpy())
        baseline_scores = evaluate(
            usable["actual"].to_numpy(), usable["baseline"].to_numpy()
        )
        metric_rows.append(
            {
                "horizon": int(horizon),
                "method": method,
                "model": model_name,
                **scores,
                "baseline_MAE": baseline_scores["MAE"],
                "baseline_RMSE": baseline_scores["RMSE"],
                "baseline_R2": baseline_scores["R2"],
                "baseline_Observed_SD": baseline_scores["Observed_SD"],
            }
        )
    metrics = pd.DataFrame(metric_rows)

    # Paired comparisons against the horizon-appropriate baseline, with
    # block-aware inference and a multiple-comparison correction across
    # the whole family of horizon-method-model comparisons.
    comparison_rows = []
    for (horizon, method, model_name), group in forecasts.groupby(
        ["horizon", "method", "model"]
    ):
        usable = group.dropna(subset=["predicted", "baseline", "actual"])
        if usable.empty:
            continue
        y = usable["actual"].to_numpy()
        predicted = usable["predicted"].to_numpy()
        baseline_values = usable["baseline"].to_numpy()

        block = moving_block_bootstrap_ci(
            y,
            predicted,
            baseline_values,
            dates=usable["Date"].to_numpy(),
            n_samples=bootstrap_samples,
            alpha=alpha,
            seed=seed,
        )
        year_block = year_blocked_bootstrap_ci(
            y,
            predicted,
            baseline_values,
            dates=usable["Date"].to_numpy(),
            n_samples=bootstrap_samples,
            alpha=alpha,
            seed=seed,
        )
        comparison_rows.append(
            {
                "horizon": int(horizon),
                "method": method,
                "model": model_name,
                "n": int(len(usable)),
                "mae_model": float(np.abs(y - predicted).mean()),
                "mae_baseline": float(np.abs(y - baseline_values).mean()),
                "delta_mae": block["delta"],
                "block_ci_low": block["ci_low"],
                "block_ci_high": block["ci_high"],
                "block_significant": block["significant"],
                "block_length_days": block["block_length"],
                "year_ci_low": year_block["ci_low"],
                "year_ci_high": year_block["ci_high"],
                "year_significant": year_block["significant"],
            }
        )

    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        corrections = apply_correction(
            comparisons["delta_mae"].abs().to_numpy(), alpha=alpha
        )
        comparisons["p_holm"] = corrections["p_holm"].to_numpy()
        comparisons["p_benjamini_hochberg"] = corrections[
            "p_benjamini_hochberg"
        ].to_numpy()
        comparisons["n_comparisons"] = corrections["n_comparisons"].to_numpy()

    metadata = {
        "horizons": list(horizons),
        "methods": list(methods),
        "models": models,
        "features": features,
        "baseline": f"{baseline} at the horizon-appropriate lag",
        "n_folds": int(forecasts["fold"].nunique()),
        "n_forecasts": int(len(forecasts)),
        "min_train_days": min_train_days,
        "test_block_days": test_block_days,
        "step_days": step_days,
        "bootstrap_samples": bootstrap_samples,
        "alpha": alpha,
        "seed": seed,
        "issue_date_rule": "last day of the fold's test block",
        "availability_contract": (
            "only observations dated on or before the issue date are used; "
            "no future observed AOD or MERRA-2 is ever consulted"
        ),
        "comparability_note": (
            "Each horizon forecasts a DIFFERENT target date from the same "
            "issue date, so cross-horizon MAE differences partly reflect "
            "target-date difficulty rather than horizon difficulty. The "
            "within-issue-date comparison in within_issue.csv isolates the "
            "horizon effect."
        ),
    }

    # Within-issue-date comparison: for each issue date, compare each
    # horizon's error to the same issue date's horizon-1 error. This
    # removes target-date difficulty, which differs between horizons.
    within = _within_issue_comparison(forecasts, baseline)

    # Confound-free baseline: persistence evaluated at every lag on the
    # identical set of target dates the backtest used.
    lag_reference = persistence_error_by_lag(
        data,
        forecasts["Date"].unique(),
        horizons=horizons,
        target=target,
        date_column=date_column,
    )

    return MultiHorizonResult(
        forecasts=forecasts,
        metrics=metrics,
        comparisons=comparisons,
        within_issue=within,
        lag_reference=lag_reference,
        metadata=metadata,
    )


def _within_issue_comparison(forecasts: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Compare each horizon to horizon 1 within the same issue date.

    Because each horizon predicts a different target date, a raw
    cross-horizon MAE comparison confounds horizon difficulty with
    target-date difficulty. Holding the issue date fixed removes the
    latter: the same forecast origin is compared across horizons.
    """
    rows = []
    for (method, model_name, issue_date), group in forecasts.groupby(
        ["method", "model", "issue_date"]
    ):
        by_horizon = group.set_index("horizon")
        if 1 not in by_horizon.index:
            continue
        reference = by_horizon.loc[1]
        if not np.isfinite(reference["predicted"]):
            continue
        for horizon, row in by_horizon.iterrows():
            if horizon == 1:
                continue
            if not np.isfinite(row["predicted"]):
                continue
            rows.append(
                {
                    "method": method,
                    "model": model_name,
                    "issue_date": issue_date,
                    "horizon": int(horizon),
                    "abs_error": abs(row["predicted"] - row["actual"]),
                    "abs_error_h1": abs(reference["predicted"] - reference["actual"]),
                    "baseline_abs_error": abs(row["baseline"] - row["actual"]),
                    "baseline_abs_error_h1": abs(
                        reference["baseline"] - reference["actual"]
                    ),
                }
            )

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    table["model_vs_h1"] = table["abs_error"] - table["abs_error_h1"]
    table["baseline_vs_h1"] = (
        table["baseline_abs_error"] - table["baseline_abs_error_h1"]
    )
    table["model_minus_baseline"] = table["abs_error"] - table["baseline_abs_error"]
    return table
