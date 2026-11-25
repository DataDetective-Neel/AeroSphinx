"""Block-aware uncertainty quantification and multiple-comparison control.

The Phase 3 audit found that the Phase 2 bootstrap resampled individual days
as if they were independent. They are not: daily PM2.5 absolute errors have
a lag-1 autocorrelation of +0.31 and a lag-30 autocorrelation of +0.16, so
the effective sample size is materially smaller than the nominal one. Folds
are dependent too: test blocks are contiguous 30-day windows advancing 30
days, so adjacent folds test adjacent calendar periods and share nearly all
their training history.

This module provides the corrections:

* :func:`moving_block_bootstrap_ci` - resamples **contiguous blocks** of
  days, so the serial correlation within a block is preserved.
* :func:`year_blocked_bootstrap_ci` - resamples **whole calendar years**,
  the most conservative block structure available here.
* :func:`holm_bonferroni` and :func:`benjamini_hochberg` - control the
  family-wise error rate across the many model comparisons Phase 3 performs.
* :func:`fold_level_comparison` - the paired test computed **per fold**,
  so the dependence structure is visible rather than assumed away.

The autocorrelation length is *estimated from the data*, not assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "estimate_block_length",
    "moving_block_bootstrap_ci",
    "year_blocked_bootstrap_ci",
    "holm_bonferroni",
    "benjamini_hochberg",
    "fold_level_comparison",
    "apply_correction",
]


def estimate_block_length(
    values: np.ndarray,
    max_lag: int = 60,
    threshold: float = 0.05,
    significance: float = 0.05,
) -> int:
    """Estimate the block length needed to break serial dependence.

    A lag is treated as *significantly* correlated when its sample
    autocorrelation exceeds the threshold by more than the sampling noise
    would explain. The sample autocorrelation of white noise is not zero:
    with *n* observations its standard error is approximately
    ``1/sqrt(n)``, so a fixed threshold of 0.05 flags roughly half the
    lags of a genuinely independent series as "correlated" and produces an
    absurdly long block.

    The block length is therefore the smallest lag from which the
    autocorrelation is **not significantly** different from zero for all
    subsequent lags. Using "the last significant lag" rather than "the
    first insignificant lag" keeps the block long enough to break the
    dependence without being driven by a single noisy lag.

    Parameters
    ----------
    values:
        A time-ordered series.
    max_lag:
        Largest lag to examine.
    threshold:
        Minimum |autocorrelation| for a lag to count as correlated.
    significance:
        Multiple-testing level for deciding whether an observed
        autocorrelation exceeds sampling noise.

    Returns
    -------
    int
        The estimated block length, at least 1.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 20:
        return 1

    # A constant series has no serial dependence to break, and pandas divides
    # by its zero standard deviation while computing the autocorrelation. The
    # answer is 1 either way; returning early avoids a RuntimeWarning that would
    # otherwise appear in every test run and train readers to ignore warnings.
    spread = float(np.nanstd(values))
    if not np.isfinite(spread) or spread == 0.0:
        return 1

    series = pd.Series(values)
    lags = list(range(1, min(max_lag, n // 2) + 1))
    autocorrelation = np.array([series.autocorr(lag) for lag in lags], dtype=float)
    autocorrelation = np.nan_to_num(autocorrelation, nan=0.0)

    # Standard error of the sample autocorrelation under independence.
    standard_error = 1.0 / np.sqrt(n)

    # Bonferroni over the lags examined: a lag counts as correlated only if
    # it exceeds the threshold AND exceeds the noise band.
    noise_band = (
        standard_error * np.sqrt(2 * np.log(lags[-1] / significance))
        if lags
        else standard_error
    )
    significant = (np.abs(autocorrelation) > threshold) & (
        np.abs(autocorrelation) > noise_band
    )

    above = np.flatnonzero(significant)
    if above.size == 0:
        return 1
    return int(above[-1] + 2)


def _block_indices(
    n: int, block_length: int, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw resampled indices made of contiguous blocks.

    Each block starts at a uniformly chosen position and runs forward,
    wrapping around the end of the series. The wrap keeps every block the
    same length without ever indexing outside the array.
    """
    if n == 0:
        return np.empty((n_samples, 0), dtype=int)

    block_length = max(1, min(block_length, n))
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(n_samples, n_blocks))
    offsets = np.arange(block_length)
    # Modulo wrapping keeps the indices inside the array.
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    return indices.reshape(n_samples, -1)[:, :n]


def moving_block_bootstrap_ci(
    y_true,
    y_pred_a,
    y_pred_b,
    dates: pd.Series | None = None,
    n_samples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
    block_length: int | None = None,
) -> dict:
    """Block bootstrap CI for the paired MAE difference.

    Contiguous blocks of days are resampled with replacement, preserving the
    serial correlation within each block.

    Parameters
    ----------
    y_true:
        Observed target values.
    y_pred_a, y_pred_b:
        The two models being compared.
    dates:
        Observation dates, used to order the series before blocking. When
        omitted the input order is assumed to be chronological.
    n_samples:
        Bootstrap resamples.
    alpha:
        Significance level.
    seed:
        Bootstrap seed, for reproducibility.
    block_length:
        Block length in days. Estimated from the data when omitted.

    Returns
    -------
    dict
        ``delta``, ``ci_low``, ``ci_high``, ``block_length``,
        ``n_pairs``, ``p_a_better``, ``significant`` and ``n_samples``.
    """
    y_true = np.asarray(y_true, dtype=float)
    a = np.asarray(y_pred_a, dtype=float)
    b = np.asarray(y_pred_b, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(a) & np.isfinite(b)
    y_true = y_true[mask]
    errors_a = np.abs(y_true - a[mask])
    errors_b = np.abs(y_true - b[mask])

    if errors_a.size == 0:
        return {
            "delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "block_length": 0,
            "n_pairs": 0,
            "p_a_better": float("nan"),
            "significant": False,
            "n_samples": n_samples,
        }

    # Order chronologically before blocking.
    if dates is not None:
        order = np.argsort(np.asarray(pd.to_datetime(dates))[mask])
        errors_a = errors_a[order]
        errors_b = errors_b[order]

    if block_length is None:
        block_length = estimate_block_length(errors_a - errors_b)

    rng = np.random.default_rng(seed)
    indices = _block_indices(errors_a.size, block_length, n_samples, rng)

    means_a = errors_a[indices].mean(axis=1)
    means_b = errors_b[indices].mean(axis=1)
    differences = means_a - means_b

    low = float(np.percentile(differences, 100 * alpha / 2))
    high = float(np.percentile(differences, 100 * (1 - alpha / 2)))

    return {
        "delta": float(errors_a.mean() - errors_b.mean()),
        "ci_low": low,
        "ci_high": high,
        "block_length": int(block_length),
        "n_pairs": int(errors_a.size),
        "p_a_better": float(np.mean(differences < 0)),
        "significant": bool(low > 0 or high < 0),
        "n_samples": n_samples,
    }


def year_blocked_bootstrap_ci(
    y_true,
    y_pred_a,
    y_pred_b,
    dates,
    n_samples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap CI resampling whole calendar years.

    This is the most conservative block structure available: a year is
    resampled as an indivisible unit, so no year contributes twice and the
    between-year variability is fully preserved. It is correspondingly the
    widest interval and the hardest test of significance.
    """
    y_true = np.asarray(y_true, dtype=float)
    a = np.asarray(y_pred_a, dtype=float)
    b = np.asarray(y_pred_b, dtype=float)
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)

    mask = np.isfinite(y_true) & np.isfinite(a) & np.isfinite(b)
    years = dates[mask].dt.year.to_numpy()
    errors_a = np.abs(y_true[mask] - a[mask])
    errors_b = np.abs(y_true[mask] - b[mask])

    if errors_a.size == 0:
        return {
            "delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_pairs": 0,
            "n_years": 0,
            "p_a_better": float("nan"),
            "significant": False,
            "n_samples": n_samples,
        }

    unique_years = np.unique(years)
    if unique_years.size < 2:
        return moving_block_bootstrap_ci(
            y_true, a, b, n_samples=n_samples, alpha=alpha, seed=seed
        )

    per_year = {int(year): np.flatnonzero(years == year) for year in unique_years}

    rng = np.random.default_rng(seed)
    n_years = unique_years.size
    sampled = rng.integers(0, n_years, size=(n_samples, n_years))

    means_a = np.empty(n_samples)
    means_b = np.empty(n_samples)
    for index in range(n_samples):
        picked = np.concatenate(
            [per_year[int(unique_years[j])] for j in sampled[index]]
        )
        means_a[index] = errors_a[picked].mean()
        means_b[index] = errors_b[picked].mean()

    differences = means_a - means_b
    low = float(np.percentile(differences, 100 * alpha / 2))
    high = float(np.percentile(differences, 100 * (1 - alpha / 2)))

    return {
        "delta": float(errors_a.mean() - errors_b.mean()),
        "ci_low": low,
        "ci_high": high,
        "n_pairs": int(errors_a.size),
        "n_years": int(n_years),
        "p_a_better": float(np.mean(differences < 0)),
        "significant": bool(low > 0 or high < 0),
        "n_samples": n_samples,
    }


# --------------------------------------------------------------------------
# Multiple-comparison control
# --------------------------------------------------------------------------


def holm_bonferroni(pvalues) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment.

    Controls the family-wise error rate under the null hypothesis that all
    comparisons in the family are uninformative. More powerful than the
    plain Bonferroni correction, and the appropriate default for a small
    family of confirmatory comparisons.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = pvalues.size
    if n == 0:
        return pvalues

    order = np.argsort(pvalues)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (n - rank) * pvalues[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjustment.

    Controls the expected proportion of false discoveries among rejections,
    which is the right criterion when the comparisons are exploratory rather
    than confirmatory. Phase 3 compares many models across many horizons, so
    both criteria are reported.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = pvalues.size
    if n == 0:
        return pvalues

    order = np.argsort(pvalues)[::-1]
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for rank, index in enumerate(order):
        candidate = min(1.0, pvalues[index] * n / (n - rank))
        running = min(running, candidate)
        adjusted[index] = running
    return adjusted


def apply_correction(
    pvalues, alpha: float = 0.05, method: str = "holm"
) -> pd.DataFrame:
    """Apply a multiple-comparison correction to a family of p-values.

    Returns
    -------
    pandas.DataFrame
        The raw p-values, both adjusted p-values, and the significance
        verdict under each correction.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    holm = holm_bonferroni(pvalues)
    bh = benjamini_hochberg(pvalues)

    return pd.DataFrame(
        {
            "p_value": pvalues,
            "p_holm": holm,
            "p_benjamini_hochberg": bh,
            "significant_holm": holm < alpha,
            "significant_bh": bh < alpha,
            "alpha": alpha,
            "n_comparisons": int(pvalues.size),
            "correction_method": method,
        }
    )


# --------------------------------------------------------------------------
# Fold-level comparison
# --------------------------------------------------------------------------


def fold_level_comparison(
    y_true,
    y_pred_a,
    y_pred_b,
    folds,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired MAE difference computed within each fold.

    Reporting the per-fold difference alongside the pooled one makes the
    dependence structure visible: if the sign of the difference flips
    between folds, the pooled result is describing an average over
    heterogeneous regimes rather than a consistent effect.
    """
    y_true = np.asarray(y_true, dtype=float)
    a = np.asarray(y_pred_a, dtype=float)
    b = np.asarray(y_pred_b, dtype=float)
    folds = np.asarray(folds)

    mask = np.isfinite(y_true) & np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return pd.DataFrame()

    y = y_true[mask]
    ea = np.abs(y - a[mask])
    eb = np.abs(y - b[mask])
    fold_values = folds[mask]

    rows = []
    for fold in np.unique(fold_values):
        selected = fold_values == fold
        mae_a = float(ea[selected].mean())
        mae_b = float(eb[selected].mean())
        rows.append(
            {
                "fold": int(fold),
                "n": int(selected.sum()),
                "mae_a": mae_a,
                "mae_b": mae_b,
                "delta": mae_a - mae_b,
                "a_better": bool(mae_a < mae_b),
            }
        )

    table = pd.DataFrame(rows)
    # Summary columns are broadcast to every row so the table can be
    # filtered without losing the consistency information.
    table["folds_a_better_fraction"] = float(table["a_better"].mean())
    table["significant_folds"] = int((table["delta"] < 0).sum())
    table["alpha"] = alpha
    return table
