"""The M0-M8 experiment matrix for multisource seasonal PM2.5 forecasting.

Every configuration is evaluated on **identical eligible dates** and with the
same rolling-origin folds, so a difference between configurations is a
difference in model or inputs, never a difference in the evaluation
population.

Tracks
------
``A``
    Retrospective ground-only. CPCB history and calendar features only; the
    only operational-eligible track.
``B``
    Retrospective multisource. AOD and MERRA-2 are same-day observations and
    are **retrospective only** (D-002), so Track B may never be described as
    an operational forecast.

A source is eligible on a date only when its ``<source>_observed`` flag is
true. The Track B evaluation frame is therefore the intersection of dates on
which AOD and MERRA-2 are both genuinely observed. Missing values are never
filled; a configuration that cannot predict a row abstains on that row, and
its coverage is reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aerosphinx.evaluation.block_inference import (
    holm_bonferroni,
    moving_block_bootstrap_ci,
    year_blocked_bootstrap_ci,
)
from aerosphinx.evaluation.metrics import evaluate
from aerosphinx.evaluation.splits import rolling_origin_folds
from aerosphinx.features.groups import assemble_features
from aerosphinx.features.seasonal import SEASONS, assign_season, season_features
from aerosphinx.models.supervised import build_model

__all__ = [
    "MATRIX",
    "MIN_REPORTABLE_N",
    "MatrixConfig",
    "MatrixResult",
    "matrix_config_manifest",
    "common_eligible",
    "run_matrix",
]

GROUND_GROUPS: tuple[str, ...] = ("pm25_history", "calendar")

MIN_REPORTABLE_N: int = 30
"""Below this many paired predictions a metric is reported as suppressed.

A season or year with fewer than 30 pairs cannot support an MAE, and an R2 on
a handful of autocorrelated days is actively misleading (the pre-fix matrix
published R2 = -125.5 on two monsoon days). The row is still emitted so the
count stays visible; the metric columns are left empty and ``reported`` is
``False``. The API and the UI surface the empty value rather than a number.
"""


@dataclass(frozen=True)
class MatrixConfig:
    """One row of the experiment matrix.

    ``season_mode`` is one of ``none``, ``categorical``, ``interaction``,
    ``independent`` or ``fallback``.
    """

    id: str
    name: str
    groups: tuple[str, ...]
    family: str
    season_mode: str
    operational_eligible: bool
    question: str


MATRIX: dict[str, MatrixConfig] = {
    "M0": MatrixConfig(
        "M0",
        "Persistence baseline",
        GROUND_GROUPS,
        "persistence",
        "none",
        True,
        "How well does yesterday's PM2.5 predict tomorrow's?",
    ),
    "M1": MatrixConfig(
        "M1",
        "CPCB-only global",
        GROUND_GROUPS,
        "learned",
        "none",
        True,
        "Does a learned model using ground history alone beat persistence?",
    ),
    "M2": MatrixConfig(
        "M2",
        "CPCB + INSAT AOD",
        GROUND_GROUPS + ("aod",),
        "learned",
        "none",
        False,
        "Does AOD add predictive information on matched dates?",
    ),
    "M3": MatrixConfig(
        "M3",
        "CPCB + MERRA-2",
        GROUND_GROUPS + ("merra",),
        "learned",
        "none",
        False,
        "Does MERRA-2 add predictive information on matched dates?",
    ),
    "M4": MatrixConfig(
        "M4",
        "CPCB + INSAT + MERRA-2",
        GROUND_GROUPS + ("aod", "merra"),
        "learned",
        "none",
        False,
        "Does full multisource fusion beat the ground-only model?",
    ),
    "M5": MatrixConfig(
        "M5",
        "Multisource global + season",
        GROUND_GROUPS + ("aod", "merra"),
        "learned",
        "categorical",
        False,
        "Does a season categorical feature improve one global model?",
    ),
    "M6": MatrixConfig(
        "M6",
        "Multisource + season interactions",
        GROUND_GROUPS + ("aod", "merra"),
        "learned",
        "interaction",
        False,
        "Do season-feature interaction terms improve one global model?",
    ),
    "M7": MatrixConfig(
        "M7",
        "Four independent seasonal models",
        GROUND_GROUPS + ("aod", "merra"),
        "learned",
        "independent",
        False,
        "Does a specialist per season beat the global model?",
    ),
    "M8": MatrixConfig(
        "M8",
        "Seasonal with global fallback",
        GROUND_GROUPS + ("aod", "merra"),
        "learned",
        "fallback",
        False,
        "Does seasonal selection with an unconditional global fallback help?",
    ),
}

#: Pairs compared for the source ablation. Negative delta = first is better.
ABLATION_PAIRS: tuple[tuple[str, str], ...] = (
    # Track A: the learned ground-only model against persistence. This is the
    # pair behind the paper's Track A headline, so it carries the same
    # block-aware interval as every Track B ablation.
    ("M1", "M0"),
    ("M2", "M1"),
    ("M3", "M1"),
    ("M4", "M1"),
    ("M4", "M0"),
    ("M5", "M4"),
    ("M6", "M4"),
    ("M7", "M4"),
    ("M8", "M4"),
    ("M7", "M1"),
    ("M8", "M1"),
)


def matrix_config_manifest(config_ids=None) -> list[dict]:
    """Machine-readable description of the matrix."""
    ids = list(config_ids) if config_ids is not None else list(MATRIX)
    records = []
    for config_id in ids:
        config = MATRIX[config_id]
        records.append(
            {
                "id": config.id,
                "name": config.name,
                "groups": list(config.groups),
                "family": config.family,
                "season_mode": config.season_mode,
                "operational_eligible": config.operational_eligible,
                "question": config.question,
            }
        )
    return records


def _required_flags(config: MatrixConfig) -> list[str]:
    flags = []
    if "aod" in config.groups:
        flags.append("aod_observed")
    if "merra" in config.groups:
        flags.append("merra_observed")
    return flags


def common_eligible(frame: pd.DataFrame, configs) -> pd.Series:
    """Dates eligible for **every** configuration in a comparison.

    A date must carry the target, a calendar-aware one-day lag, and a genuine
    observation of every environmental source any configuration requires.
    This is what makes the ablation a like-for-like comparison.
    """
    if "PM25" not in frame.columns or "PM25_Lag_1" not in frame.columns:
        raise KeyError("frame must carry PM25 and PM25_Lag_1")
    mask = frame["PM25"].notna() & frame["PM25_Lag_1"].notna()
    flags: list[str] = []
    for config in configs:
        for flag in _required_flags(config):
            if flag not in flags:
                flags.append(flag)
    for flag in flags:
        if flag not in frame.columns:
            raise KeyError(f"frame is missing the availability flag {flag!r}")
        mask = mask & frame[flag].astype(bool)
    return mask


def _base_features(config: MatrixConfig, frame: pd.DataFrame) -> list[str]:
    features = [
        feature
        for feature in assemble_features(config.groups)
        if feature in frame.columns
    ]
    if not features:
        raise ValueError(f"{config.id}: no feature columns present in the frame")
    return features


def _design(frame: pd.DataFrame, base: list[str], mode: str) -> pd.DataFrame:
    """Build the design matrix for a non-seasonal or one-global-model mode."""
    X = frame[base].reset_index(drop=True).copy()
    if mode in {"categorical", "interaction"}:
        seasons = season_features(frame["Date"]).reset_index(drop=True)["season"]
        for season in SEASONS:
            indicator = seasons.eq(season).to_numpy(dtype=float)
            if mode == "categorical":
                X[f"season_{season}"] = indicator
            else:
                for feature in base:
                    X[f"{feature}_x_{season}"] = (
                        frame[feature].reset_index(drop=True).to_numpy(dtype=float)
                        * indicator
                    )
    return X


def _stable_columns(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Drop near-constant design columns using **training-only** variance.

    An interaction term can be zero on every training row of a season and
    carry a variance near 1e-17. A standardiser then divides by that tiny
    scale, and a linear model extrapolates catastrophically on the test
    block. Such a column carries no estimable information, so it is removed
    before fitting. The selection uses the training block only.
    """
    values = X_train.to_numpy(dtype=float)
    std = np.nanstd(values, axis=0)
    keep = std > 1e-8
    if not keep.any():
        keep = np.ones(X_train.shape[1], dtype=bool)
    return X_train.loc[:, keep], X_test.loc[:, keep]


def _fit_global(X_train, y_train, X_test, model_name: str) -> np.ndarray:
    X_train, X_test = _stable_columns(
        X_train.reset_index(drop=True), X_test.reset_index(drop=True)
    )
    model = build_model(model_name, list(X_train.columns), target="PM25")
    train_frame = X_train.copy()
    train_frame["PM25"] = np.asarray(y_train, dtype=float)
    model.fit(train_frame)
    predictions = np.asarray(model.predict(X_test), dtype=float)
    # A non-finite prediction is an abstention, never a reported value.
    predictions[~np.isfinite(predictions)] = np.nan
    return predictions


def _seasonal_predict(
    config: MatrixConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    base: list[str],
    model_name: str,
    min_specialist_days: int,
    use_fallback: bool,
) -> np.ndarray:
    train_seasons = assign_season(train["Date"]).to_numpy()
    test_seasons = assign_season(test["Date"]).to_numpy()
    y_train = pd.to_numeric(train["PM25"], errors="coerce").to_numpy(dtype=float)

    predictions = np.full(len(test), np.nan, dtype=float)
    if use_fallback:
        predictions = _fit_global(
            _design(train, base, "none"),
            y_train,
            _design(test, base, "none"),
            model_name,
        )

    for season in SEASONS:
        in_train = train_seasons == season
        in_test = test_seasons == season
        if not in_test.any():
            continue
        if int(in_train.sum()) < min_specialist_days:
            # No specialist: independent mode abstains, fallback keeps the
            # global prediction. The threshold is declared, not tuned.
            continue
        specialist = _fit_global(
            _design(train[in_train], base, "none"),
            y_train[in_train],
            _design(test[in_test], base, "none"),
            model_name,
        )
        predictions[in_test] = specialist

    return predictions


def _fit_predict(
    config: MatrixConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
    min_specialist_days: int,
) -> np.ndarray:
    if config.family == "persistence":
        return pd.to_numeric(test["PM25_Lag_1"], errors="coerce").to_numpy(dtype=float)

    base = _base_features(config, train)
    if config.season_mode == "independent":
        return _seasonal_predict(
            config,
            train,
            test,
            base,
            model_name,
            min_specialist_days,
            use_fallback=False,
        )
    if config.season_mode == "fallback":
        return _seasonal_predict(
            config,
            train,
            test,
            base,
            model_name,
            min_specialist_days,
            use_fallback=True,
        )

    y_train = pd.to_numeric(train["PM25"], errors="coerce").to_numpy(dtype=float)
    return _fit_global(
        _design(train, base, config.season_mode),
        y_train,
        _design(test, base, config.season_mode),
        model_name,
    )


def _empty_frames() -> dict[str, pd.DataFrame]:
    return {
        "metrics": pd.DataFrame(
            columns=[
                "track",
                "config",
                "model",
                "n",
                "n_missing",
                "folds",
                "MAE",
                "RMSE",
                "MedianAE",
                "R2",
                "Mean_Bias",
            ]
        ),
        "by_season": pd.DataFrame(
            columns=[
                "track",
                "config",
                "model",
                "season",
                "n",
                "MAE",
                "RMSE",
                "R2",
                "reported",
            ]
        ),
        "by_year": pd.DataFrame(
            columns=[
                "track",
                "config",
                "model",
                "year",
                "n",
                "MAE",
                "RMSE",
                "R2",
                "reported",
            ]
        ),
        "fold_metrics": pd.DataFrame(
            columns=["track", "config", "model", "Fold", "n", "MAE", "RMSE"]
        ),
        "common_metrics": pd.DataFrame(
            columns=["track", "config", "model", "n", "MAE", "RMSE", "MedianAE", "R2"]
        ),
        "ablations": pd.DataFrame(
            columns=[
                "track",
                "config",
                "reference",
                "delta_MAE",
                "n",
                "ci_low",
                "ci_high",
                "block_length_days",
                "year_ci_low",
                "year_ci_high",
                "p_value",
                "p_holm",
                "significant",
                "reported",
            ]
        ),
    }


@dataclass
class MatrixResult:
    """All artefacts produced by one track of the matrix."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    by_season: pd.DataFrame
    by_year: pd.DataFrame
    fold_metrics: pd.DataFrame
    common_metrics: pd.DataFrame
    ablations: pd.DataFrame
    configs: list[dict]
    folds: list[dict]


def _overall(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (track, config, model), group in predictions.groupby(
        ["track", "config", "model"], sort=False
    ):
        usable = group.dropna(subset=["Prediction", "Target"])
        metrics = evaluate(usable["Target"], usable["Prediction"])
        errors = np.abs(
            usable["Prediction"].to_numpy(dtype=float)
            - usable["Target"].to_numpy(dtype=float)
        )
        rows.append(
            {
                "track": track,
                "config": config,
                "model": model,
                "n": int(metrics["n"]),
                "n_missing": int(group["Prediction"].isna().sum()),
                "folds": int(group["Fold"].nunique()),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MedianAE": float(np.median(errors)) if errors.size else float("nan"),
                "R2": metrics["R2"],
                "Mean_Bias": metrics["Mean_Bias"],
            }
        )
    return pd.DataFrame(rows)


def _grouped(predictions: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    """MAE/RMSE/R2 per (track, config, model, group).

    Groups smaller than :data:`MIN_REPORTABLE_N` keep their row — the count is
    itself information — but carry no metric, so nothing downstream can quote
    an R2 computed on two days.
    """
    rows = []
    usable = predictions.dropna(subset=["Prediction", "Target"])
    for (track, config, model), group in usable.groupby(
        ["track", "config", "model"], sort=False
    ):
        for key, subset in group.groupby(column, sort=False):
            metrics = evaluate(subset["Target"], subset["Prediction"])
            n = int(metrics["n"])
            reportable = n >= MIN_REPORTABLE_N
            rows.append(
                {
                    "track": track,
                    "config": config,
                    "model": model,
                    label: key,
                    "n": n,
                    "MAE": metrics["MAE"] if reportable else float("nan"),
                    "RMSE": metrics["RMSE"] if reportable else float("nan"),
                    "R2": metrics["R2"] if reportable else float("nan"),
                    "reported": reportable,
                }
            )
    return pd.DataFrame(rows)


def _common(predictions: pd.DataFrame):
    """Metrics and aligned matrices on dates predicted by every config."""
    rows = []
    aligned: dict[str, tuple[pd.Series, pd.DataFrame]] = {}
    for track in predictions["track"].unique():
        subset = predictions[predictions["track"] == track]
        pivot = subset.pivot_table(
            index="Date", columns="config", values="Prediction", aggfunc="first"
        )
        if pivot.empty:
            continue
        target = subset.groupby("Date")["Target"].first()
        common = pivot.dropna()
        if common.empty:
            continue
        aligned[track] = (target.reindex(common.index), common)
        for config in common.columns:
            y = target.reindex(common.index).to_numpy(dtype=float)
            p = common[config].to_numpy(dtype=float)
            metrics = evaluate(y, p)
            errors = np.abs(p - y)
            model = subset.loc[subset["config"] == config, "model"].iloc[0]
            rows.append(
                {
                    "track": track,
                    "config": config,
                    "model": model,
                    "n": int(metrics["n"]),
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "MedianAE": (
                        float(np.median(errors)) if errors.size else float("nan")
                    ),
                    "R2": metrics["R2"],
                }
            )
    return pd.DataFrame(rows), aligned


def _ablate(aligned) -> pd.DataFrame:
    """Paired MAE differences with block-aware uncertainty.

    Track A has always reported a moving-block bootstrap interval and a
    Holm-adjusted p-value against persistence; Track B — the track that
    carries the central negative claim — reported a bare point estimate.
    Both tracks are now treated identically: every ablation pair carries its
    paired delta, a bootstrap interval that respects serial correlation, a
    two-sided p-value derived from the bootstrap sign, and a Holm adjustment
    across the ablation family *within the track* (the comparisons are
    analysed per track, so a Track B delta is never penalised by the size of
    the Track A family).

    A pair whose interval spans zero is reported as ``not_significant`` even
    when the point estimate looks large, and a pair below
    :data:`MIN_REPORTABLE_N` carries no delta at all.
    """
    rows = []
    for track, (target, common) in aligned.items():
        y = target.to_numpy(dtype=float)
        dates = pd.Series(pd.to_datetime(common.index))
        track_rows = []
        for config, reference in ABLATION_PAIRS:
            if config not in common.columns or reference not in common.columns:
                continue
            a = common[config].to_numpy(dtype=float)
            b = common[reference].to_numpy(dtype=float)
            n = int(np.isfinite(y).sum())
            interval = moving_block_bootstrap_ci(y, a, b, dates=dates)
            # The conservative check: resample whole calendar years. Eight
            # years is a small number of resampling units, so this interval is
            # much wider than the day-level one and is the honest guard against
            # a single season carrying the whole difference.
            yearly = year_blocked_bootstrap_ci(y, a, b, dates=dates)
            p_a_better = interval.get("p_a_better", float("nan"))
            p_value = (
                float(min(1.0, 2 * min(p_a_better, 1 - p_a_better)))
                if np.isfinite(p_a_better)
                else float("nan")
            )
            reportable = n >= MIN_REPORTABLE_N
            track_rows.append(
                {
                    "track": track,
                    "config": config,
                    "reference": reference,
                    "delta_MAE": (
                        float(interval["delta"]) if reportable else float("nan")
                    ),
                    "n": n,
                    "ci_low": float(interval["ci_low"]) if reportable else float("nan"),
                    "ci_high": (
                        float(interval["ci_high"]) if reportable else float("nan")
                    ),
                    "block_length_days": int(interval.get("block_length", 0)),
                    "year_ci_low": (
                        float(yearly["ci_low"]) if reportable else float("nan")
                    ),
                    "year_ci_high": (
                        float(yearly["ci_high"]) if reportable else float("nan")
                    ),
                    "p_value": p_value if reportable else float("nan"),
                    "reported": reportable,
                }
            )
        if track_rows:
            pvalues = [row["p_value"] for row in track_rows]
            adjusted = holm_bonferroni(
                [1.0 if not np.isfinite(p) else p for p in pvalues]
            )
            for row, p_holm in zip(track_rows, adjusted):
                row["p_holm"] = float(p_holm)
                row["significant"] = bool(
                    row["reported"]
                    and np.isfinite(p_holm)
                    and p_holm < 0.05
                    and not (row["ci_low"] <= 0.0 <= row["ci_high"])
                )
            rows.extend(track_rows)
    return pd.DataFrame(rows)


def run_matrix(
    frame: pd.DataFrame,
    track: str,
    config_ids,
    model_name: str = "ridge",
    min_train_days: int = 60,
    test_block_days: int = 30,
    step_days: int = 30,
    validation_days: int = 0,
    min_specialist_days: int = 30,
) -> MatrixResult:
    """Run one track of the M0-M8 matrix on a shared eligible frame."""
    if track not in {"A", "B"}:
        raise ValueError("track must be 'A' or 'B'")
    config_ids = list(config_ids)
    configs = [MATRIX[config_id] for config_id in config_ids]
    for config in configs:
        if config.family == "persistence" and config.groups != GROUND_GROUPS:
            raise ValueError("the persistence baseline must be ground-only")

    eligible = frame.loc[common_eligible(frame, configs)].copy()
    eligible["Date"] = pd.to_datetime(eligible["Date"])
    eligible = eligible.sort_values("Date").reset_index(drop=True)
    folds = rolling_origin_folds(
        eligible["Date"],
        min_train_days=min_train_days,
        validation_days=validation_days,
        test_block_days=test_block_days,
        step_days=step_days,
    )

    records: list[dict] = []
    for fold in folds:
        train, _, test = fold.split(eligible)
        if train.empty or test.empty:
            continue
        for config in configs:
            predictions = _fit_predict(
                config, train, test, model_name, min_specialist_days
            )
            for date, target, prediction in zip(
                test["Date"].to_numpy(),
                test["PM25"].to_numpy(dtype=float),
                predictions,
            ):
                records.append(
                    {
                        "track": track,
                        "config": config.id,
                        "model": (
                            "persistence"
                            if config.family == "persistence"
                            else model_name
                        ),
                        "Fold": fold.index,
                        "Date": date,
                        "Target": target,
                        "Prediction": prediction,
                    }
                )

    columns = ["track", "config", "model", "Fold", "Date", "Target", "Prediction"]
    prediction_frame = pd.DataFrame(records, columns=columns)
    if prediction_frame.empty:
        empty = _empty_frames()
        return MatrixResult(
            predictions=prediction_frame,
            metrics=empty["metrics"],
            by_season=empty["by_season"],
            by_year=empty["by_year"],
            fold_metrics=empty["fold_metrics"],
            common_metrics=empty["common_metrics"],
            ablations=empty["ablations"],
            configs=matrix_config_manifest(config_ids),
            folds=[fold.as_dict() for fold in folds],
        )

    prediction_frame["season"] = assign_season(prediction_frame["Date"]).to_numpy()
    prediction_frame["year"] = pd.to_datetime(
        prediction_frame["Date"]
    ).dt.year.to_numpy()

    common_metrics, aligned = _common(prediction_frame)
    return MatrixResult(
        predictions=prediction_frame,
        metrics=_overall(prediction_frame),
        by_season=_grouped(prediction_frame, "season", "season"),
        by_year=_grouped(prediction_frame, "year", "year"),
        fold_metrics=_grouped(prediction_frame, "Fold", "Fold"),
        common_metrics=common_metrics,
        ablations=_ablate(aligned),
        configs=matrix_config_manifest(config_ids),
        folds=[fold.as_dict() for fold in folds],
    )
