"""Uncertainty quantification for model comparisons.

The Phase 0 audit established that the shipped baseline's headline claim -
that the ensemble beats persistence - was **not statistically significant**
(paired Wilcoxon p = 0.1495; paired bootstrap 95% CI on the MAE difference
[-6.47, +1.39], spanning zero). Point estimates alone had concealed that.

This module makes the required statistics the default rather than an
afterthought. Per ``docs/reference/EXPERIMENT_PROTOCOL.md`` section 5.3, a claim of
improvement requires the interval to **exclude zero**.

All comparisons are **paired**: the same test observations are scored by
both models, which removes between-day variance from the difference and is
far more sensitive than comparing two independent error distributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "paired_wilcoxon",
    "paired_bootstrap_ci",
    "compare_models",
    "bootstrap_seed",
]


DEFAULT_SEED = 42


def bootstrap_seed(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """A seeded generator, so every reported interval is reproducible."""
    return np.random.default_rng(seed)


def _absolute_errors(y_true, y_pred) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return np.abs(y_true[mask] - y_pred[mask])


def paired_wilcoxon(
    y_true,
    y_pred_a,
    y_pred_b,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test on per-observation absolute errors.

    Parameters
    ----------
    y_true:
        Observed target values.
    y_pred_a, y_pred_b:
        Predictions from the two models being compared.
    alternative:
        ``"two-sided"`` (default), ``"less"`` or ``"greater"``.

    Returns
    -------
    dict
        ``statistic``, ``pvalue``, ``n_pairs``, ``mean_error_a``,
        ``mean_error_b``, ``delta`` (A minus B; negative favours A) and
        ``significant`` at the 5% level.
    """
    from scipy import stats

    y_true = np.asarray(y_true, dtype=float)
    a = np.asarray(y_pred_a, dtype=float)
    b = np.asarray(y_pred_b, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(a) & np.isfinite(b)
    errors_a = np.abs(y_true[mask] - a[mask])
    errors_b = np.abs(y_true[mask] - b[mask])

    if errors_a.size == 0:
        return {
            "statistic": float("nan"),
            "pvalue": float("nan"),
            "n_pairs": 0,
            "mean_error_a": float("nan"),
            "mean_error_b": float("nan"),
            "delta": float("nan"),
            "significant": False,
        }

    delta = float(errors_a.mean() - errors_b.mean())

    if np.allclose(errors_a, errors_b):
        # Every difference is zero; the test is undefined but the models are
        # indistinguishable on these observations.
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "n_pairs": int(errors_a.size),
            "mean_error_a": float(errors_a.mean()),
            "mean_error_b": float(errors_b.mean()),
            "delta": 0.0,
            "significant": False,
        }

    try:
        result = stats.wilcoxon(errors_a, errors_b, alternative=alternative)
        statistic = float(result.statistic)
        pvalue = float(result.pvalue)
    except ValueError:
        statistic = float("nan")
        pvalue = float("nan")

    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "n_pairs": int(errors_a.size),
        "mean_error_a": float(errors_a.mean()),
        "mean_error_b": float(errors_b.mean()),
        "delta": delta,
        "significant": bool(np.isfinite(pvalue) and pvalue < 0.05),
    }


def paired_bootstrap_ci(
    y_true,
    y_pred_a,
    y_pred_b,
    n_samples: int = 10000,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Bootstrap confidence interval for the paired MAE difference.

    The bootstrap resamples *days* with replacement, keeping each day's two
    errors together, which respects the paired structure.

    Returns
    -------
    dict
        ``delta`` (A minus B), ``ci_low``, ``ci_high``, ``n_pairs``,
        ``p_a_better`` (the bootstrap fraction of samples in which A has the
        lower MAE) and ``significant`` (whether the interval excludes zero).
    """
    y_true = np.asarray(y_true, dtype=float)
    a = np.asarray(y_pred_a, dtype=float)
    b = np.asarray(y_pred_b, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(a) & np.isfinite(b)
    errors_a = np.abs(y_true[mask] - a[mask])
    errors_b = np.abs(y_true[mask] - b[mask])

    if errors_a.size == 0:
        return {
            "delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_pairs": 0,
            "p_a_better": float("nan"),
            "significant": False,
            "n_samples": n_samples,
        }

    rng = bootstrap_seed(seed)
    n = errors_a.size
    indices = rng.integers(0, n, size=(n_samples, n))

    means_a = errors_a[indices].mean(axis=1)
    means_b = errors_b[indices].mean(axis=1)
    differences = means_a - means_b

    low = float(np.percentile(differences, 100 * alpha / 2))
    high = float(np.percentile(differences, 100 * (1 - alpha / 2)))
    delta = float(errors_a.mean() - errors_b.mean())

    return {
        "delta": delta,
        "ci_low": low,
        "ci_high": high,
        "n_pairs": int(n),
        "p_a_better": float(np.mean(differences < 0)),
        "significant": bool(low > 0 or high < 0),
        "n_samples": n_samples,
    }


def compare_models(
    y_true,
    predictions: dict[str, np.ndarray],
    baseline: str,
    n_samples: int = 10000,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Compare every model against a baseline, with paired statistics.

    The baseline is the reference, so a negative ``delta`` means the model
    has a lower MAE than the baseline.

    Parameters
    ----------
    y_true:
        Observed target values.
    predictions:
        Mapping of model name to predictions. Must contain ``baseline``.
    baseline:
        Name of the reference model.
    n_samples:
        Bootstrap resamples.
    alpha:
        Significance level.
    seed:
        Bootstrap seed, for reproducibility.

    Returns
    -------
    pandas.DataFrame
        One row per model, with the paired test, the bootstrap interval and
        an explicit ``verdict`` column.
    """
    if baseline not in predictions:
        raise KeyError(f"baseline {baseline!r} is not among the predictions")

    y_true = np.asarray(y_true, dtype=float)
    reference = np.asarray(predictions[baseline], dtype=float)

    rows = []
    for name, values in predictions.items():
        values = np.asarray(values, dtype=float)
        test = paired_wilcoxon(y_true, values, reference)
        boot = paired_bootstrap_ci(
            y_true, values, reference, n_samples=n_samples, alpha=alpha, seed=seed
        )

        if name == baseline:
            verdict = "reference"
        elif boot["significant"] and boot["delta"] < 0:
            verdict = "better than baseline (significant)"
        elif boot["significant"]:
            verdict = "worse than baseline (significant)"
        else:
            verdict = "no significant difference"

        rows.append(
            {
                "model": name,
                "baseline": baseline,
                "n_pairs": test["n_pairs"],
                "mae_model": test["mean_error_a"],
                "mae_baseline": test["mean_error_b"],
                "delta_mae": test["delta"],
                "wilcoxon_statistic": test["statistic"],
                "wilcoxon_pvalue": test["pvalue"],
                "bootstrap_ci_low": boot["ci_low"],
                "bootstrap_ci_high": boot["ci_high"],
                "p_better_than_baseline": boot["p_a_better"],
                "significant": boot["significant"],
                "verdict": verdict,
            }
        )

    return pd.DataFrame(rows)
