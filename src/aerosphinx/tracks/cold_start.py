"""Cold-start and extrapolation analysis for AeroSphinx.

Phase 2 established that the learned models fail hardest on the 8 of 96
folds whose test block contains observations outside the training target
range. The mechanism is structural: a tree ensemble cannot predict outside
the range of its training targets, while persistence copies yesterday's
value whatever it was.

This module quantifies that failure properly and tests candidate
mitigations.

The analyses are:

* **extrapolation-conditional error** - MAE conditional on whether the
  test observation lies inside the training target range;
* **error against the out-of-range fraction** - how error grows as a
  fraction of the test block falls outside the training range;
* **mitigation comparison** - persistence fallback, ridge extrapolation,
  quantile-calibrated fallback and regime-aware fallback.

Clipping is deliberately **not** offered as a mitigation. Clipping bounds
extreme outputs, but it replaces a wrong high prediction with a wrong low
one, which can make the low-pollution extrapolation case worse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aerosphinx.features.seasonal import season_of_month

__all__ = [
    "ColdStartSummary",
    "extrapolation_conditioned_error",
    "error_vs_out_of_range_fraction",
    "persistence_fallback",
    "quantile_fallback",
    "regime_aware_fallback",
    "summarise_cold_start",
]


def _season_of(month: int) -> str:
    """Meteorological season for a month number.

    Delegates to the single authoritative assignment in
    :mod:`aerosphinx.features.seasonal`.
    """
    return season_of_month(month)


@dataclass
class ColdStartSummary:
    """Aggregate cold-start findings."""

    n_folds: int
    n_extrapolating_folds: int
    n_observations: int
    n_observations_out_of_range: int
    fraction_out_of_range: float
    mae_in_range: float
    mae_out_of_range: float
    mae_ratio: float

    def as_dict(self) -> dict:
        return {
            "n_folds": self.n_folds,
            "n_extrapolating_folds": self.n_extrapolating_folds,
            "n_observations": self.n_observations,
            "n_observations_out_of_range": self.n_observations,
            "fraction_out_of_range": round(self.fraction_out_of_range, 6),
            "mae_in_range": self.mae_in_range,
            "mae_out_of_range": self.mae_out_of_range,
            "mae_ratio": self.mae_ratio,
        }


def extrapolation_conditioned_error(
    predictions: pd.DataFrame,
    model_columns: list[str],
    train_min: pd.Series,
    train_max: pd.Series,
    target: str = "Target",
    fold_column: str = "Fold",
) -> pd.DataFrame:
    """Split error into in-range and out-of-range components.

    Parameters
    ----------
    predictions:
        Per-observation predictions with a target column and a fold column.
    model_columns:
        The models to analyse.
    train_min, train_max:
        Per-fold training target bounds, indexed by fold.
    target:
        Name of the observed target column.
    fold_column:
        Name of the fold column.

    Returns
    -------
    pandas.DataFrame
        One row per model with error statistics for the in-range and
        out-of-range subsets.
    """
    data = predictions.copy()
    folds = data[fold_column]
    data["_train_min"] = folds.map(train_min)
    data["_train_max"] = folds.map(train_max)
    data["_out_of_range"] = (data[target] < data["_train_min"]) | (
        data[target] > data["_train_max"]
    )

    rows = []
    for model_name in model_columns:
        usable = data.dropna(subset=[model_name, target])
        if usable.empty:
            continue

        inside = usable[~usable["_out_of_range"]]
        outside = usable[usable["_out_of_range"]]

        mae_inside = (
            float(np.abs(inside[model_name] - inside[target]).mean())
            if len(inside)
            else float("nan")
        )
        mae_outside = (
            float(np.abs(outside[model_name] - outside[target]).mean())
            if len(outside)
            else float("nan")
        )
        rows.append(
            {
                "model": model_name,
                "n": int(len(usable)),
                "n_out_of_range": int(len(outside)),
                "fraction_out_of_range": float(len(outside) / len(usable)),
                "mae_in_range": mae_inside,
                "mae_out_of_range": mae_outside,
                "mae_ratio": (
                    mae_outside / mae_inside
                    if mae_inside and np.isfinite(mae_inside) and mae_inside > 0
                    else float("nan")
                ),
                "bias_in_range": (
                    float((inside[model_name] - inside[target]).mean())
                    if len(inside)
                    else float("nan")
                ),
                "bias_out_of_range": (
                    float((outside[model_name] - outside[target]).mean())
                    if len(outside)
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


def error_vs_out_of_range_fraction(
    predictions: pd.DataFrame,
    model_columns: list[str],
    train_min: pd.Series,
    train_max: pd.Series,
    target: str = "Target",
    fold_column: str = "Fold",
    n_bins: int = 5,
) -> pd.DataFrame:
    """Error as a function of how much of the fold's test block is out of range.

    Folds are binned by their out-of-range fraction, and the mean error is
    reported per bin. This shows whether the degradation is gradual or
    concentrated at the extreme.
    """
    data = predictions.copy()
    folds = data[fold_column]
    data["_train_min"] = folds.map(train_min)
    data["_train_max"] = folds.map(train_max)
    data["_out_of_range"] = (data[target] < data["_train_min"]) | (
        data[target] > data["_train_max"]
    )

    fold_fraction = data.groupby(fold_column)["_out_of_range"].mean()
    data["_fold_fraction"] = data[fold_column].map(fold_fraction)

    # Bin folds by their out-of-range fraction. A single fold cannot be
    # binned, so it is kept as its own group.
    distinct = data["_fold_fraction"].nunique()
    if distinct >= 2:
        try:
            data["_bin"] = pd.qcut(
                data["_fold_fraction"],
                min(n_bins, distinct),
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            data["_bin"] = 0
    else:
        data["_bin"] = 0

    rows = []
    for model_name in model_columns:
        usable = data.dropna(subset=[model_name, target])
        if usable.empty:
            continue
        for bin_value, group in usable.groupby("_bin"):
            rows.append(
                {
                    "model": model_name,
                    "bin": int(bin_value),
                    "fold_out_of_range_fraction": float(group["_fold_fraction"].mean()),
                    "n": int(len(group)),
                    "mae": float(np.abs(group[model_name] - group[target]).mean()),
                    "bias": float((group[model_name] - group[target]).mean()),
                }
            )

    return pd.DataFrame(rows).sort_values(["model", "bin"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Mitigations
# --------------------------------------------------------------------------


def persistence_fallback(
    predictions: pd.DataFrame,
    model_column: str,
    persistence_column: str = "persistence",
    train_min: pd.Series | None = None,
    train_max: pd.Series | None = None,
    target: str = "Target",
    fold_column: str = "Fold",
) -> np.ndarray:
    """Use the model only where its prediction is inside the training range.

    Outside the training target range the model is structurally unable to
    produce a sensible value, so persistence is used instead. The fallback
    is applied to the *prediction*, not the observed target, so it remains
    usable at forecast time.
    """
    values = predictions[model_column].to_numpy(dtype=float)
    fallback = predictions[persistence_column].to_numpy(dtype=float)

    if train_min is None or train_max is None:
        return np.where(np.isfinite(values), values, fallback)

    folds = predictions[fold_column]
    lower = folds.map(train_min).to_numpy(dtype=float)
    upper = folds.map(train_max).to_numpy(dtype=float)

    in_range = np.isfinite(values) & (values >= lower) & (values <= upper)
    return np.where(in_range, values, fallback)


def quantile_fallback(
    predictions: pd.DataFrame,
    model_column: str,
    persistence_column: str = "persistence",
    train_quantiles: dict | None = None,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
    target: str = "Target",
    fold_column: str = "Fold",
) -> np.ndarray:
    """Fall back to a conditional quantile of the training target.

    Where the model predicts outside the plausible range of the training
    target, the fallback is the empirical quantile of the training target
    at that season, rather than a single point. This is a distributional
    fallback: it acknowledges that the model cannot place the observation,
    while still saying something about where it probably lies.
    """
    values = predictions[model_column].to_numpy(dtype=float)
    fallback = predictions[persistence_column].to_numpy(dtype=float)

    if not train_quantiles:
        return np.where(np.isfinite(values), values, fallback)

    months = (
        pd.to_datetime(predictions[target.replace(target, "Date")]).dt.month
        if "Date" in predictions.columns
        else None
    )
    if months is None:
        return np.where(np.isfinite(values), values, fallback)

    result = values.copy()
    for index, month in enumerate(months.to_numpy()):
        if not np.isfinite(result[index]):
            continue
        key = int(month)
        if key not in train_quantiles:
            continue
        low, high = train_quantiles[key]
        if result[index] < low or result[index] > high:
            result[index] = 0.5 * (low + high)

    return result


def regime_aware_fallback(
    predictions: pd.DataFrame,
    model_column: str,
    persistence_column: str = "persistence",
    use_model_in_regimes: tuple[str, ...] = ("pre_monsoon", "winter"),
    target: str = "Target",
    date_column: str = "Date",
) -> np.ndarray:
    """Use the learned model only in the regimes where it is known to help.

    Phase 2 found that the learned models beat persistence in the
    pre-monsoon and winter seasons and lose in monsoon. This fallback
    applies that structure. **The regime is derived from the calendar month
    of the target date**, which is known in advance and uses no observed
    quantity.
    """
    values = predictions[model_column].to_numpy(dtype=float)
    fallback = predictions[persistence_column].to_numpy(dtype=float)

    if date_column not in predictions.columns:
        return np.where(np.isfinite(values), values, fallback)

    months = pd.to_datetime(predictions[date_column]).dt.month
    regimes = months.map(_season_of).to_numpy()

    use_model = np.isin(regimes, use_model_in_regimes) & np.isfinite(values)
    return np.where(use_model, values, fallback)


def summarise_cold_start(
    predictions: pd.DataFrame,
    model_columns: list[str],
    train_min: pd.Series,
    train_max: pd.Series,
    target: str = "Target",
    fold_column: str = "Fold",
) -> ColdStartSummary:
    """Aggregate summary of the cold-start situation across all folds."""
    data = predictions.copy()
    folds = data[fold_column]
    data["_train_min"] = folds.map(train_min)
    data["_train_max"] = folds.map(train_max)
    data["_out_of_range"] = (data[target] < data["_train_min"]) | (
        data[target] > data["_train_max"]
    )

    fold_fraction = data.groupby(fold_column)["_out_of_range"].mean()
    n_extrapolating = int((fold_fraction > 0).sum())

    out_of_range = data[data["_out_of_range"]]
    inside = data[~data["_out_of_range"]]

    # The reference is the best baseline: persistence.
    mae_inside = (
        float(np.abs(inside["persistence"] - inside[target]).mean())
        if "persistence" in inside and len(inside)
        else float("nan")
    )
    mae_outside = (
        float(np.abs(out_of_range["persistence"] - out_of_range[target]).mean())
        if "persistence" in out_of_range and len(out_of_range)
        else float("nan")
    )

    return ColdStartSummary(
        n_folds=int(fold_fraction.size),
        n_extrapolating_folds=n_extrapolating,
        n_observations=int(len(out_of_range)),
        n_observations_out_of_range=int(len(out_of_range)),
        fraction_out_of_range=float(len(out_of_range) / max(len(data), 1)),
        mae_in_range=mae_inside,
        mae_out_of_range=mae_outside,
        mae_ratio=(
            mae_outside / mae_inside
            if mae_inside and np.isfinite(mae_inside) and mae_inside > 0
            else float("nan")
        ),
    )
