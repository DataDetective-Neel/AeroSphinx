"""Regime and coverage diagnostics for Track A.

The Phase 2 backtest exposed a failure mode that neither a single-window
evaluation nor a point-estimate comparison would reveal: in the earliest
folds the **learned** models were far worse than the persistence baseline,
while in later folds they were comparable.

The mechanism is regime extrapolation. A tree ensemble cannot predict
outside the range of its training targets. When the training window is
short and dominated by high-PM2.5 months, and the test block is a low-PM2.5
monsoon period, the model is structurally unable to produce the required
values. The persistence baseline has no such limitation - it simply copies
yesterday's value, whatever that value was.

These diagnostics quantify that situation so the result can be reported
honestly rather than as an unexplained model failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "regime_coverage",
    "coverage_by_fold",
    "cold_start_report",
]


def regime_coverage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    value_column: str = "PM25",
) -> dict:
    """Compare the target distribution of a test block with its training block.

    Returns
    -------
    dict
        ``train_min``, ``train_max``, ``test_min``, ``test_max``,
        ``train_p05``, ``train_p95``, ``test_p05``, ``test_p95``,
        ``test_below_train_min`` (fraction of test observations below the
        training minimum), ``test_above_train_max`` and
        ``extrapolating`` (whether the test block lies materially outside
        the training range in either direction).
    """
    train_values = pd.to_numeric(train[value_column], errors="coerce").dropna()
    test_values = pd.to_numeric(test[value_column], errors="coerce").dropna()

    if train_values.empty or test_values.empty:
        return {
            "train_min": float("nan"),
            "train_max": float("nan"),
            "test_min": float("nan"),
            "test_max": float("nan"),
            "train_p05": float("nan"),
            "train_p95": float("nan"),
            "test_p05": float("nan"),
            "test_p95": float("nan"),
            "test_below_train_min": float("nan"),
            "test_above_train_max": float("nan"),
            "extrapolating": False,
        }

    train_min = float(train_values.min())
    train_max = float(train_values.max())
    test_min = float(test_values.min())
    test_max = float(test_values.max())

    below = float(np.mean(test_values < train_min))
    above = float(np.mean(test_values > train_max))

    return {
        "train_min": train_min,
        "train_max": train_max,
        "test_min": test_min,
        "test_max": test_max,
        "train_p05": float(train_values.quantile(0.05)),
        "train_p95": float(train_values.quantile(0.95)),
        "test_p05": float(test_values.quantile(0.05)),
        "test_p95": float(test_values.quantile(0.95)),
        "test_below_train_min": below,
        "test_above_train_max": above,
        "extrapolating": bool(below > 0.0 or above > 0.0),
    }


def coverage_by_fold(
    frame: pd.DataFrame,
    folds,
    value_column: str = "PM25",
    date_column: str = "Date",
) -> pd.DataFrame:
    """Compute the regime-coverage diagnostics for every fold."""
    rows = []
    for fold in folds:
        train, _validation, test = fold.split(frame, date_column)
        if train.empty or test.empty:
            continue
        rows.append({"fold": fold.index, **regime_coverage(train, test, value_column)})
    return pd.DataFrame(rows)


def cold_start_report(
    fold_metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    baseline: str = "persistence",
) -> pd.DataFrame:
    """Attribute each fold's baseline-relative performance to regime coverage.

    A fold is marked ``extrapolating`` when any test observation lies
    outside the training target range - the condition under which a tree
    ensemble cannot produce the required values.

    Returns
    -------
    pandas.DataFrame
        One row per fold with the MAE of each model relative to the
        baseline, plus the coverage flags.
    """
    if fold_metrics.empty or coverage.empty:
        return pd.DataFrame()

    pivot = fold_metrics.pivot_table(index="fold", columns="model", values="MAE")
    if baseline not in pivot.columns:
        return pd.DataFrame()

    result = pivot.copy()
    for column in pivot.columns:
        if column != baseline:
            result[f"{column}_minus_{baseline}"] = pivot[column] - pivot[baseline]

    merged = result.reset_index().merge(coverage, on="fold", how="left")
    merged["extrapolating"] = merged.get("extrapolating", False).fillna(False)
    return merged
