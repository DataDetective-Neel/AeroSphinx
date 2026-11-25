"""Evaluation metrics for AeroSphinx.

Two families of metric live here, and the distinction matters:

* **Regression metrics** (MAE, RMSE, R2, bias) measure concentration error.
* **AQI category metrics** measure agreement of the *PM2.5 concentration
  category* against the CPCB National AQI PM2.5 sub-index bands.

The AQI category computed here is **not** the official overall AQI, which is
a multi-pollutant index. The columns are named accordingly and the
distinction is enforced by the docstrings, per
``docs/reference/EXPERIMENT_PROTOCOL.md`` section 5.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "AQI_BOUNDS",
    "AQI_LABELS",
    "aqi_category",
    "regression_metrics",
    "aqi_metrics",
    "evaluate",
    "metric_table",
]


# CPCB National AQI breakpoints for the PM2.5 sub-index, in ug/m3.
# The official bands are: 0-30 Good, 30-60 Satisfactory, 60-90 Moderate,
# 90-120 Poor, 120-250 Very Poor, above 250 Severe.
AQI_BOUNDS = [30.0, 60.0, 90.0, 120.0, 250.0]
AQI_LABELS = np.array(
    ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
)


def aqi_category(values) -> np.ndarray:
    """Map PM2.5 concentrations to CPCB PM2.5 concentration categories.

    This is the **PM2.5 sub-index band**, not the official multi-pollutant
    AQI. ``np.digitize`` with these bounds reproduces the CPCB bands
    exactly: 30.0 -> Satisfactory, 60.0 -> Moderate, and so on.

    Parameters
    ----------
    values:
        PM2.5 concentrations in ug/m3.

    Returns
    -------
    numpy.ndarray
        Category labels, one per input value.
    """
    values = np.asarray(values, dtype=float)
    return AQI_LABELS[np.digitize(values, AQI_BOUNDS)]


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE, R2, mean bias and the observed standard deviation.

    ``Mean_Bias`` is ``mean(predicted - observed)``; a positive value means
    the model over-predicts. ``Observed_SD`` is reported alongside R2 so a
    reader can see what R2 = 0 corresponds to in absolute terms.

    A caveat on R2 that this function makes visible: R2 is
    ``1 - SSE/SST``, so on a test block with a small observed standard
    deviation a small systematic offset produces a hugely negative R2. The
    Phase 2 backtest produced a fold with observed SD of 6.1 ug/m3 and R2 of
    -186. **R2 is reported for comparability with prior work, but MAE is the
    metric to judge a fold on when the test block has low variance.**
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0:
        return {
            "n": 0,
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "R2": float("nan"),
            "Mean_Bias": float("nan"),
            "Observed_SD": float("nan"),
            "Observed_Mean": float("nan"),
        }

    error = y_pred - y_true
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    return {
        "n": int(y_true.size),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "Mean_Bias": float(np.mean(error)),
        "Observed_SD": float(np.std(y_true, ddof=0)),
        "Observed_Mean": float(np.mean(y_true)),
    }


def aqi_metrics(y_true, y_pred) -> dict[str, float]:
    """Accuracy, macro-F1 and confusion matrix for the PM2.5 category.

    Macro-F1 is reported because plain accuracy is misleading under class
    imbalance, which is severe here: most days are Moderate or Poor, so a
    model that always predicts the majority class scores well on accuracy
    while being useless.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0:
        return {
            "AQI_Accuracy": float("nan"),
            "AQI_Macro_F1": float("nan"),
            "AQI_ConfusionMatrix": [],
            "AQI_Labels": list(AQI_LABELS),
        }

    true_labels = aqi_category(y_true)
    pred_labels = aqi_category(y_pred)

    accuracy = float(np.mean(true_labels == pred_labels))

    f1_scores = []
    for label in AQI_LABELS:
        true_positive = int(np.sum((true_labels == label) & (pred_labels == label)))
        false_positive = int(np.sum((true_labels != label) & (pred_labels == label)))
        false_negative = int(np.sum((true_labels == label) & (pred_labels != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)

    matrix = []
    for true_label in AQI_LABELS:
        row = [
            int(np.sum((true_labels == true_label) & (pred_labels == pred_label)))
            for pred_label in AQI_LABELS
        ]
        matrix.append(row)

    return {
        "AQI_Accuracy": accuracy,
        "AQI_Macro_F1": float(np.mean(f1_scores)),
        "AQI_ConfusionMatrix": matrix,
        "AQI_Labels": list(AQI_LABELS),
    }


def evaluate(y_true, y_pred) -> dict[str, float]:
    """Combined regression and AQI category metrics for one prediction set."""
    metrics = regression_metrics(y_true, y_pred)
    metrics.update(aqi_metrics(y_true, y_pred))
    return metrics


def metric_table(
    y_true,
    predictions: dict[str, np.ndarray],
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    """Build a comparison table of several models' metrics.

    Parameters
    ----------
    y_true:
        Observed target values.
    predictions:
        Mapping of model name to predicted values.
    groups:
        Optional grouping labels (season, year, month). When supplied the
        table gains a ``group`` column and one row per (model, group).

    Returns
    -------
    pandas.DataFrame
        One row per model (or per model-group pair).
    """
    y_true = np.asarray(y_true, dtype=float)

    rows = []
    for name, values in predictions.items():
        values = np.asarray(values, dtype=float)
        if groups is None:
            rows.append({"model": name, **evaluate(y_true, values)})
            continue

        groups = pd.Series(groups).reset_index(drop=True)
        for group in sorted(groups.dropna().unique()):
            mask = (groups == group).to_numpy()
            if not mask.any():
                continue
            rows.append(
                {
                    "model": name,
                    "group": str(group),
                    **evaluate(y_true[mask], values[mask]),
                }
            )

    return pd.DataFrame(rows)
