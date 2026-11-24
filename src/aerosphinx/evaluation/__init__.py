"""Evaluation: splits, metrics and uncertainty quantification."""

from aerosphinx.evaluation.metrics import (
    AQI_BOUNDS,
    AQI_LABELS,
    aqi_category,
    evaluate,
    metric_table,
    regression_metrics,
)
from aerosphinx.evaluation.splits import (
    Fold,
    assert_no_leakage,
    expanding_window_folds,
    rolling_origin_folds,
)
from aerosphinx.evaluation.uncertainty import (
    compare_models,
    paired_bootstrap_ci,
    paired_wilcoxon,
)

__all__ = [
    "AQI_BOUNDS",
    "AQI_LABELS",
    "aqi_category",
    "evaluate",
    "metric_table",
    "regression_metrics",
    "Fold",
    "assert_no_leakage",
    "expanding_window_folds",
    "rolling_origin_folds",
    "compare_models",
    "paired_bootstrap_ci",
    "paired_wilcoxon",
]
