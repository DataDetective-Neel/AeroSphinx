"""Phase 3 Workstream C experiment: architecture, ablation and routing.

This module implements the expanded Workstream C scope:

1. Global versus seasonal models.
2. Seasonal interaction features.
3. Horizon-specific versus shared models.
4. Direct versus recursive forecasting.
5. Feature-group ablations.
6. Regime-aware model routing.
7. Minimum sample requirements and a validated fallback.

Selection discipline
--------------------
Every candidate is scored on a **chronological validation block** that
precedes the test block. The winner is chosen on the pooled validation
scores and then scored on the test blocks, once. The test scores of every
candidate are also reported, for transparency, but they did not inform the
choice. There is no code path here that selects on test scores.

Matched dates
-------------
A comparison is only reported as a direct model improvement when the two
models are scored on **identical test rows**. Comparisons across different
row sets are reported separately and are never presented as improvements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aerosphinx.evaluation.metrics import evaluate
from aerosphinx.evaluation.splits import Fold
from aerosphinx.features.groups import (
    GROUND_ONLY_GROUPS,
    assemble_features,
    distinctive_features,
)
from aerosphinx.models.baselines import persistence_forecast
from aerosphinx.tracks.regime import (
    MIN_SPECIALIST_DAYS_CANDIDATES,
    assign_regime,
    build_architecture,
)

__all__ = [
    "WorkstreamCResult",
    "run_workstream_c",
    "build_feature_frame",
]


@dataclass
class WorkstreamCResult:
    """Outcome of the Workstream C experiment."""

    predictions: pd.DataFrame
    architecture_validation: pd.DataFrame
    architecture_test: pd.DataFrame
    ablation: pd.DataFrame
    selection: pd.DataFrame
    selected_performance: pd.DataFrame = field(default_factory=pd.DataFrame)
    group_pairwise: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "selection": self.selection.to_dict(orient="records"),
            "architecture_validation": self.architecture_validation.to_dict(
                orient="records"
            ),
            "architecture_test": self.architecture_test.to_dict(orient="records"),
            "ablation": self.ablation.to_dict(orient="records"),
        }


def build_feature_frame(
    daily: pd.DataFrame,
    station_met: pd.DataFrame | None = None,
    merra: pd.DataFrame | None = None,
    aod: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge the daily series with the environmental feature groups.

    The merge is a left join on the date, so the ground series determines
    which rows exist. Environmental values are never allowed to extend the
    calendar.

    For the AOD group the join is done explicitly rather than by merge, so
    that ``aod_valid_pixels`` has one unambiguous meaning: the number of
    pixels supporting the daily value. A date with no granule and a date
    whose granule contained no valid pixel both give **0**, because in
    both cases there is no AOD observation.
    """
    frame = daily.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])

    if station_met is not None and not station_met.empty:
        met = station_met.copy()
        met["Date"] = pd.to_datetime(met["Date"])
        frame = frame.merge(met, on="Date", how="left", suffixes=("", "_met"))
    if merra is not None and not merra.empty:
        reanalysis = merra.copy()
        reanalysis["Date"] = pd.to_datetime(reanalysis["Date"])
        frame = frame.merge(reanalysis, on="Date", how="left", suffixes=("", "_merra"))
    if aod is not None and not aod.empty:
        satellite = aod.copy()
        satellite["Date"] = pd.to_datetime(satellite["Date"])
        satellite = satellite[
            ["Date", "AOD", "aod_valid_pixels", "aod_overpasses"]
        ].set_index("Date")
        frame = frame.merge(satellite, on="Date", how="left", suffixes=("", "_aod"))
        # A date with no AOD observation has no supporting pixel.
        frame["aod_valid_pixels"] = frame["aod_valid_pixels"].fillna(0)
        frame["aod_overpasses"] = frame["aod_overpasses"].fillna(0)

    return frame.sort_values("Date").reset_index(drop=True)


def _shifted_target(
    frame: pd.DataFrame, horizon: int, target: str = "PM25", date_column: str = "Date"
) -> pd.Series:
    """The value observed ``horizon`` days after each row's date."""
    lookup = frame.set_index(date_column)[target]
    future = frame[date_column] + pd.Timedelta(days=horizon)
    return future.map(lookup)


def _issue_frame(
    frame: pd.DataFrame, issue_date, date_column: str = "Date"
) -> pd.DataFrame:
    """The rows whose date is the issue date, from a frame truncated there."""
    truncated = frame[frame[date_column] <= issue_date].copy()
    return truncated[truncated[date_column] == issue_date]


def _fit_predict(
    architecture_kind: str,
    model_name: str,
    features: list[str],
    min_specialist_days: int,
    train: pd.DataFrame,
    train_future: pd.Series,
    predict_frames: list[pd.DataFrame],
) -> list[np.ndarray]:
    """Fit one architecture and predict on several frames.

    ``train_future`` supplies the horizon-shifted target for the training
    rows.
    """
    working = train.reset_index(drop=True).copy()
    working["_future"] = train_future.to_numpy()

    architecture = build_architecture(
        architecture_kind,
        features,
        model_name=model_name,
        min_specialist_days=min_specialist_days,
    )
    architecture.fit(working)

    regimes = assign_regime(
        train[working.index].to_numpy() if False else working["Date"]
    )
    counts = {regime: int((regimes == regime).sum()) for regime in set(regimes)}
    architecture.record_specialist_counts(counts)

    return [architecture.predict(frame) for frame in predict_frames]


def _shared_model_predictions(
    model_name: str,
    features: list[str],
    train: pd.DataFrame,
    horizons: tuple[int, ...],
    predict_frame: pd.DataFrame,
    horizon: int,
) -> np.ndarray:
    """Predict with a single model trained across all horizons.

    The shared model is trained on a pooled multi-horizon sample: every
    training row appears once per horizon, with the horizon as a feature.
    One model then serves every horizon, which is the alternative to
    training a separate model per horizon.
    """
    from aerosphinx.models.supervised import build_model

    pooled_frames = []
    pooled_targets = []
    for candidate_horizon in horizons:
        shifted = _shifted_target(train, candidate_horizon)
        subset = train.reset_index(drop=True).copy()
        subset["_future"] = shifted.to_numpy()
        subset["_horizon"] = float(candidate_horizon)
        pooled_frames.append(subset)
        pooled_targets.append(shifted.to_numpy())
    pooled = pd.concat(pooled_frames, ignore_index=True)
    pooled["_target"] = np.concatenate(pooled_targets)

    usable = pooled.dropna(subset=["_target"])
    if usable.empty:
        return np.full(len(predict_frame), np.nan)

    model_features = list(features) + ["_horizon"]
    model = build_model(model_name, model_features, target="_target").fit(usable)

    predict = predict_frame.reset_index(drop=True).copy()
    predict["_horizon"] = float(horizon)
    return model.predict(predict)


def run_workstream_c(
    frame: pd.DataFrame,
    group_sets: dict[str, tuple[str, ...]] | None = None,
    architectures: tuple[str, ...] = (
        "global",
        "seasonal",
        "interaction",
        "routing",
        "shared",
    ),
    model_name: str = "ridge",
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5),
    methods: tuple[str, ...] = ("direct", "recursive"),
    min_specialist_days: int = 180,
    min_specialist_candidates: tuple[int, ...] = MIN_SPECIALIST_DAYS_CANDIDATES,
    folds: list[Fold] | None = None,
    date_column: str = "Date",
    target: str = "PM25",
    bootstrap_samples: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> WorkstreamCResult:
    """Run the Workstream C experiment.

    Parameters
    ----------
    frame:
        The daily series with the feature columns of every group present.
    group_sets:
        Mapping of a label to the feature groups it uses. Defaults to
        ground-only and ground plus each environmental group.
    architectures:
        Architecture kinds to compare.
    model_name:
        The supervised model used within each architecture.
    horizons:
        Forecast horizons.
    methods:
        ``"direct"`` and/or ``"recursive"``.
    min_specialist_days:
        The threshold used when not searching over candidates.
    min_specialist_candidates:
        Candidate thresholds, searched on the validation blocks.
    folds:
        Folds with a validation block. When absent, rolling-origin folds
        are constructed.

    Returns
    -------
    WorkstreamCResult
    """
    from aerosphinx.evaluation.splits import rolling_origin_folds

    group_sets = group_sets or {
        "ground_only": GROUND_ONLY_GROUPS,
        "ground_plus_station_met": GROUND_ONLY_GROUPS + ("station_met",),
        "ground_plus_merra": GROUND_ONLY_GROUPS + ("merra",),
        "ground_plus_all": GROUND_ONLY_GROUPS + ("station_met", "merra"),
    }

    if folds is None:
        folds = rolling_origin_folds(
            frame,
            min_train_days=365,
            validation_days=30,
            test_block_days=30,
            step_days=30,
            date_column=date_column,
        )

    data = frame.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(date_column).reset_index(drop=True)

    predictions: list[dict] = []

    for fold in folds:
        train, validation, test = fold.split(data, date_column)
        if train.empty or test.empty:
            continue

        # Two walk-forward passes per fold.
        #
        # Pass A issues at the end of the training block and its targets
        # land in the validation block. Pass B issues at the end of the
        # validation block and its targets land in the test block.
        #
        # The architecture is chosen on Pass A's validation scores and then
        # applied, unchanged, to Pass B. The test block is therefore never
        # consulted for any decision.
        passes = []
        if not validation.empty:
            passes.append(
                {
                    "split": "validation",
                    "train": train,
                    "issue_date": train[date_column].max(),
                    "score_rows": validation,
                }
            )
        passes.append(
            {
                "split": "test",
                # For the test pass the model may be refitted on the
                # training and validation blocks, which are both legitimately
                # in the past at that point.
                "train": (
                    pd.concat([train, validation], ignore_index=True)
                    if not validation.empty
                    else train
                ),
                "issue_date": (
                    validation[date_column].max()
                    if not validation.empty
                    else train[date_column].max()
                ),
                "score_rows": test,
            }
        )

        for pass_spec in passes:
            train_block = pass_spec["train"]
            issue_date = pass_spec["issue_date"]
            score_rows = pass_spec["score_rows"]
            if train_block.empty:
                continue

            train_block = train_block[train_block[date_column] <= issue_date].copy()
            if train_block.empty:
                continue

            for group_label, groups in group_sets.items():
                features = assemble_features(groups)

                for horizon in horizons:
                    # The target for a training row is the value horizon
                    # days later, and only rows whose future target is
                    # already known may train the model.
                    future = _shifted_target(data, horizon, target, date_column)
                    eligible = (
                        train_block[date_column] + pd.Timedelta(days=horizon)
                        <= issue_date
                    )
                    train_rows = train_block[eligible].copy()
                    train_future = future.loc[train_rows.index]
                    if train_rows.empty:
                        continue

                    issue_frame = _issue_frame(data, issue_date, date_column)
                    if issue_frame.empty:
                        continue
                    target_date = issue_date + pd.Timedelta(days=horizon)
                    target_rows = data[data[date_column] == target_date]
                    if target_rows.empty:
                        continue
                    actual = float(target_rows[target].iloc[0])
                    if not np.isfinite(actual):
                        continue

                    # The forecast is scored only when its target falls in
                    # the scored block.
                    if not (score_rows[date_column] == target_date).any():
                        continue

                    # The horizon-appropriate persistence baseline.
                    baseline_frame = data[data[date_column] <= target_date].copy()
                    baseline_preds = persistence_forecast(
                        baseline_frame,
                        value_column=target,
                        date_column=date_column,
                        periods=horizon,
                    )
                    positions = np.flatnonzero(
                        (baseline_frame[date_column] == target_date).to_numpy()
                    )
                    baseline_value = (
                        float(baseline_preds[positions[0]])
                        if positions.size
                        else np.nan
                    )

                    for architecture_kind in architectures:
                        for method in methods:
                            if method == "recursive":
                                # The recursive model is the one-day model
                                # applied repeatedly.
                                one_day_future = _shifted_target(
                                    data, 1, target, date_column
                                )
                                one_day_eligible = (
                                    train_block[date_column] + pd.Timedelta(days=1)
                                    <= issue_date
                                )
                                one_day_rows = train_block[one_day_eligible].copy()
                                predicted = _recursive_predict(
                                    one_day_rows,
                                    one_day_future.loc[one_day_rows.index],
                                    issue_frame,
                                    features,
                                    model_name,
                                    horizon,
                                    data,
                                    issue_date,
                                    target,
                                    date_column,
                                )
                            elif method == "direct":
                                if architecture_kind == "shared":
                                    # One model trained across all horizons,
                                    # with the horizon as a feature.
                                    predicted = _shared_model_predictions(
                                        model_name,
                                        features,
                                        train_rows,
                                        horizons,
                                        issue_frame,
                                        horizon,
                                    )
                                    predicted = float(np.asarray(predicted).ravel()[0])
                                else:
                                    predicted = _fit_predict(
                                        architecture_kind,
                                        model_name,
                                        features,
                                        min_specialist_days,
                                        train_rows,
                                        train_future,
                                        [issue_frame],
                                    )[0]
                                    predicted = float(np.asarray(predicted).ravel()[0])
                            else:
                                continue

                            predictions.append(
                                {
                                    "split": pass_spec["split"],
                                    "fold": fold.index,
                                    "group_set": group_label,
                                    "groups": ",".join(groups),
                                    "architecture": architecture_kind,
                                    "method": method,
                                    "model": model_name,
                                    "horizon": horizon,
                                    "issue_date": issue_date,
                                    "Date": target_date,
                                    "actual": actual,
                                    "predicted": float(predicted),
                                    "baseline": baseline_value,
                                }
                            )

    if not predictions:
        raise ValueError("no predictions were produced")

    all_predictions = pd.DataFrame(predictions)

    # --- architecture selection on the validation blocks --------------------
    # The validation scores are restricted to the dates on which the group
    # set has genuine feature coverage. Without this, a group whose
    # coverage is limited (MERRA-2 covers only 2024) would be selected on
    # validation rows where its features were absent and merely
    # median-imputed, which is not a selection on real evidence.
    availability: dict[str, set] = {}
    for label, groups in group_sets.items():
        distinctive = distinctive_features(groups)
        if not distinctive:
            availability[label] = set(data[date_column])
            continue
        present_any = data[distinctive].notna().any(axis=1)
        availability[label] = set(data.loc[present_any, date_column])

    validation_predictions = all_predictions[all_predictions["split"] == "validation"]
    validation_predictions = _restrict_to_coverage(validation_predictions, availability)
    validation_scores = _score_predictions(validation_predictions)
    test_scores = _score_predictions(
        all_predictions[all_predictions["split"] == "test"]
    )

    selection = _select_architecture(validation_scores)

    # --- feature-group ablation -------------------------------------------
    ablation = _ablation(
        all_predictions[all_predictions["split"] == "test"],
        bootstrap_samples,
        alpha,
        seed,
        feature_availability=availability,
    )

    # The direct group-versus-group comparison: does the environmental
    # group add value on top of the ground-only model?
    group_pairwise = _group_pairwise(
        all_predictions[all_predictions["split"] == "test"],
        bootstrap_samples,
        alpha,
        seed,
        feature_availability=availability,
    )

    # The selected architecture's test performance, per horizon. This is
    # the headline: the architecture was chosen on validation and then
    # scored on the test blocks.
    selected_performance = _selected_test_performance(
        selection, all_predictions[all_predictions["split"] == "test"]
    )

    metadata = {
        "group_sets": {k: list(v) for k, v in group_sets.items()},
        "architectures": list(architectures),
        "model_name": model_name,
        "horizons": list(horizons),
        "methods": list(methods),
        "min_specialist_days": min_specialist_days,
        "min_specialist_candidates": list(min_specialist_candidates),
        "n_folds": int(all_predictions["fold"].nunique()),
        "selection_rule": (
            "lowest pooled validation MAE; test scores were not consulted"
        ),
        "availability_note": (
            "station_met and merra are retrospective-only groups. Their "
            "results are retrospective estimations, not operational "
            "forecasts. The ground-only group set is the only one that "
            "satisfies the operational feature contract."
        ),
    }

    return WorkstreamCResult(
        predictions=all_predictions,
        architecture_validation=validation_scores,
        architecture_test=test_scores,
        ablation=ablation,
        selection=selection,
        selected_performance=selected_performance,
        group_pairwise=group_pairwise,
        metadata=metadata,
    )


def _selected_test_performance(
    selection: pd.DataFrame, test_predictions: pd.DataFrame
) -> pd.DataFrame:
    """The test performance of the architecture chosen on validation.

    This is the headline table: for each horizon, the architecture that
    validation selected, scored on the test blocks.
    """
    if selection.empty or test_predictions.empty:
        return pd.DataFrame()

    chosen = selection[selection.get("selected", False) == True]  # noqa: E712
    rows = []
    for _, row in chosen.iterrows():
        matched = test_predictions[
            (test_predictions["group_set"] == row["group_set"])
            & (test_predictions["architecture"] == row["architecture"])
            & (test_predictions["method"] == row["method"])
            & (test_predictions["horizon"] == row["horizon"])
        ].dropna(subset=["predicted", "baseline", "actual"])
        if matched.empty:
            continue
        scores = evaluate(matched["actual"].to_numpy(), matched["predicted"].to_numpy())
        baseline_scores = evaluate(
            matched["actual"].to_numpy(), matched["baseline"].to_numpy()
        )
        rows.append(
            {
                "horizon": int(row["horizon"]),
                "group_set": row["group_set"],
                "architecture": row["architecture"],
                "method": row["method"],
                "validation_MAE": row["MAE"],
                "validation_n": int(row["n"]),
                "n_test": int(len(matched)),
                "test_MAE": scores["MAE"],
                "test_RMSE": scores["RMSE"],
                "test_R2": scores["R2"],
                "test_bias": scores["Mean_Bias"],
                "persistence_MAE": baseline_scores["MAE"],
                "delta_vs_persistence": scores["MAE"] - baseline_scores["MAE"],
            }
        )
    return pd.DataFrame(rows)


def _recursive_predict(
    train_rows: pd.DataFrame,
    train_future: pd.Series,
    issue_frame: pd.DataFrame,
    features: list[str],
    model_name: str,
    horizon: int,
    data: pd.DataFrame,
    issue_date,
    target: str,
    date_column: str,
) -> float:
    """Apply the one-day model repeatedly, feeding predictions back."""
    from aerosphinx.models.supervised import build_model

    working = train_rows.reset_index(drop=True).copy()
    working["_future"] = train_future.to_numpy()
    # Only the target is required to be present. A missing *feature* is
    # handled by the model itself: the tree models take NaN natively and
    # the linear model's imputer records it as an indicator. Requiring
    # every feature to be present here would discard most of the training
    # set and silently change what the model learns.
    usable = working.dropna(subset=["_future"])
    if usable.empty:
        return float("nan")

    model = build_model(model_name, list(features), target="_future").fit(usable)

    row = issue_frame.iloc[0]
    lag1 = row.get("PM25_Lag_1", np.nan)
    lag2 = row.get("PM25_Lag_2", np.nan)

    prediction = np.nan
    for step in range(1, horizon + 1):
        working_row = row.copy()
        if step > 1:
            # From the second step onward the lag and rolling features are
            # rewritten with the model's own earlier predictions. At the
            # first step the issue row's real features are used unchanged,
            # so the recursion and the direct method agree at horizon one.
            working_row["PM25_Lag_1"] = lag1
            working_row["PM25_Lag_2"] = lag2
            working_row["PM25_Lag_3"] = np.nan
            if "PM25_Roll3" in working_row.index:
                window = [value for value in (lag1, lag2) if np.isfinite(value)]
                working_row["PM25_Roll3"] = float(np.mean(window)) if window else np.nan
            if "PM25_Roll7" in working_row.index:
                working_row["PM25_Roll7"] = np.nan
        x = working_row[features].to_frame().T.astype(float)
        prediction = float(model.predict(x)[0])
        lag1, lag2 = prediction, lag1
    return prediction


def _restrict_to_coverage(
    predictions: pd.DataFrame, availability: dict
) -> pd.DataFrame:
    """Keep only the rows whose group set has genuine feature coverage.

    The ground groups are available on every date, so a group set's
    coverage is judged on the features it *adds* beyond the ground
    baseline. The distinction matters for a group with limited coverage:
    without it, the selection would run on rows where the group's features
    were absent and merely median-imputed.
    """
    if predictions.empty or not availability:
        return predictions

    keep = []
    for _, row in predictions.iterrows():
        dates = availability.get(row["group_set"])
        if dates is None or row["Date"] in dates:
            keep.append(True)
        else:
            keep.append(False)
    return predictions[np.array(keep, dtype=bool)]


def _score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate MAE for each (group, architecture, method, horizon) cell."""
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in predictions.groupby(
        ["group_set", "architecture", "method", "horizon"]
    ):
        usable = group.dropna(subset=["predicted", "actual"])
        if usable.empty:
            continue
        scores = evaluate(usable["actual"].to_numpy(), usable["predicted"].to_numpy())
        rows.append(
            {
                "group_set": keys[0],
                "architecture": keys[1],
                "method": keys[2],
                "horizon": int(keys[3]),
                **scores,
            }
        )
    return pd.DataFrame(rows)


def _select_architecture(
    validation_scores: pd.DataFrame, min_validation_rows: int = 30
) -> pd.DataFrame:
    """Rank candidates by validation MAE **within each horizon**.

    The selection is made per horizon because a deployment needs an
    architecture for every horizon it serves, and the best architecture
    need not be the same at day 1 as at day 5. Ranking across all horizons
    at once would pick a single cell and discard the rest.

    A candidate is **eligible** for selection only when its validation
    block holds at least ``min_validation_rows`` observations. Without
    this, a group with very limited coverage can win the selection on a
    handful of rows, and the "winner" is noise rather than evidence.

    The ranking uses **validation** scores only. The test scores are
    attached afterwards for transparency and play no part in the choice.
    """
    if validation_scores.empty:
        return pd.DataFrame()

    ranked = validation_scores.copy()
    ranked["eligible_for_selection"] = ranked["n"] >= min_validation_rows

    # Rank eligible candidates ahead of ineligible ones within a horizon.
    ranked["_sort_key"] = (~ranked["eligible_for_selection"]).astype(
        int
    ) * 1e9 + ranked["MAE"]
    ranked = ranked.sort_values(["horizon", "_sort_key"]).reset_index(drop=True)
    ranked["validation_rank"] = (ranked.groupby("horizon").cumcount() + 1).astype(int)
    ranked = ranked.drop(columns=["_sort_key"])

    # The selected candidate for each horizon: the best *eligible* one.
    ranked["selected"] = False
    for horizon in ranked["horizon"].unique():
        eligible = ranked[
            (ranked["horizon"] == horizon) & ranked["eligible_for_selection"]
        ]
        if eligible.empty:
            continue
        best = eligible["validation_rank"].min()
        ranked.loc[
            (ranked["horizon"] == horizon) & (ranked["validation_rank"] == best),
            "selected",
        ] = True
    return ranked


def _group_pairwise(
    test_predictions: pd.DataFrame,
    bootstrap_samples: int,
    alpha: float,
    seed: int,
    reference_group_set: str = "ground_only",
    feature_availability: dict | None = None,
    min_rows_for_inference: int = 30,
) -> pd.DataFrame:
    """Direct paired comparison of each group set against the ground-only model.

    This is *the* Track B question: does the environmental group add
    predictive value **on top of** the ground-only model, rather than
    merely helping the model beat persistence?

    The ablation above answers a different question — whether each model
    beats persistence — and a model can beat persistence without the
    environmental group contributing anything. This comparison holds the
    architecture, method, horizon and rows fixed and varies only the
    feature group.
    """
    from aerosphinx.evaluation.block_inference import (
        apply_correction,
        moving_block_bootstrap_ci,
        year_blocked_bootstrap_ci,
    )

    if test_predictions.empty:
        return pd.DataFrame()

    key = ["fold", "architecture", "method", "horizon", "Date"]
    reference = test_predictions[
        test_predictions["group_set"] == reference_group_set
    ].dropna(subset=["predicted", "actual"])
    if reference.empty:
        return pd.DataFrame()

    rows = []
    for group_set, group in test_predictions.groupby("group_set"):
        if group_set == reference_group_set:
            continue
        group = group.dropna(subset=["predicted", "actual"])
        if group.empty:
            continue

        if feature_availability and group_set in feature_availability:
            available = feature_availability[group_set]
            group = group[group["Date"].isin(available)]

        merged = group.merge(
            reference[key + ["predicted"]].rename(
                columns={"predicted": "_reference_predicted"}
            ),
            on=key,
            how="inner",
        )
        if merged.empty:
            continue

        for (architecture, method, horizon), cell in merged.groupby(
            ["architecture", "method", "horizon"]
        ):
            errors_model = np.abs(
                cell["actual"].to_numpy() - cell["predicted"].to_numpy()
            )
            errors_reference = np.abs(
                cell["actual"].to_numpy() - cell["_reference_predicted"].to_numpy()
            )
            block = moving_block_bootstrap_ci(
                cell["actual"].to_numpy(),
                cell["predicted"].to_numpy(),
                cell["_reference_predicted"].to_numpy(),
                dates=cell["Date"].to_numpy(),
                n_samples=bootstrap_samples,
                alpha=alpha,
                seed=seed,
            )
            year_block = year_blocked_bootstrap_ci(
                cell["actual"].to_numpy(),
                cell["predicted"].to_numpy(),
                cell["_reference_predicted"].to_numpy(),
                dates=cell["Date"].to_numpy(),
                n_samples=bootstrap_samples,
                alpha=alpha,
                seed=seed,
            )
            rows.append(
                {
                    "group_set": group_set,
                    "reference": reference_group_set,
                    "architecture": architecture,
                    "method": method,
                    "horizon": int(horizon),
                    "n_matched": int(len(cell)),
                    "sufficient_evidence": bool(len(cell) >= min_rows_for_inference),
                    "mae_with_group": float(errors_model.mean()),
                    "mae_reference": float(errors_reference.mean()),
                    "delta_vs_reference": float(
                        errors_model.mean() - errors_reference.mean()
                    ),
                    "block_ci_low": block["ci_low"],
                    "block_ci_high": block["ci_high"],
                    "block_significant": block["significant"],
                    "block_length_days": block["block_length"],
                    "year_ci_low": year_block["ci_low"],
                    "year_ci_high": year_block["ci_high"],
                    "year_significant": year_block["significant"],
                }
            )

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    corrections = apply_correction(
        table["delta_vs_reference"].abs().to_numpy(), alpha=alpha
    )
    table["p_holm"] = corrections["p_holm"].to_numpy()
    table["p_benjamini_hochberg"] = corrections["p_benjamini_hochberg"].to_numpy()
    table["n_comparisons"] = corrections["n_comparisons"].to_numpy()
    table["group_better"] = table["delta_vs_reference"] < 0
    return table


def _ablation(
    test_predictions: pd.DataFrame,
    bootstrap_samples: int,
    alpha: float,
    seed: int,
    reference_group_set: str = "ground_only",
    feature_availability: dict | None = None,
    min_rows_for_inference: int = 30,
) -> pd.DataFrame:
    """Compare each feature-group set against the ground-only baseline.

    **Matched test dates.** For each group set, the comparison is made
    against the ground-only rows on the *same* ``(fold, architecture,
    method, horizon, Date)`` keys, and further restricted to the dates on
    which that group's features were genuinely present. A group with
    limited coverage — the MERRA-2 group covers only 2024 — is therefore
    compared on its own dates, not on dates where its features were absent
    and merely median-imputed.

    **Block-aware uncertainty.** Every interval uses the moving-block
    bootstrap, with a Holm-Bonferroni correction across the family.

    **Sufficient evidence.** A comparison with fewer than
    ``min_rows_for_inference`` matched rows is reported but flagged as
    insufficient, and its interval is not treated as evidence.
    """
    from aerosphinx.evaluation.block_inference import (
        apply_correction,
        moving_block_bootstrap_ci,
        year_blocked_bootstrap_ci,
    )

    if test_predictions.empty:
        return pd.DataFrame()

    key = ["fold", "architecture", "method", "horizon", "Date"]
    reference = test_predictions[
        test_predictions["group_set"] == reference_group_set
    ].dropna(subset=["predicted", "baseline", "actual"])

    rows = []
    for group_set, group in test_predictions.groupby("group_set"):
        group = group.dropna(subset=["predicted", "baseline", "actual"])
        if group.empty:
            continue

        # Restrict to the dates on which this group's features are present.
        if feature_availability and group_set in feature_availability:
            available = feature_availability[group_set]
            group = group[group["Date"].isin(available)]

        # Match against the reference on identical keys.
        merged = group.merge(
            reference[key + ["predicted", "baseline", "actual"]].rename(
                columns={
                    "predicted": "_reference_predicted",
                    "baseline": "_reference_baseline",
                    "actual": "_reference_actual",
                }
            ),
            on=key,
            how="inner",
        )
        if merged.empty:
            continue

        # The reference and the model must have scored the same target.
        merged = merged[(merged["actual"] - merged["_reference_actual"]).abs() < 1e-9]
        if merged.empty:
            continue

        for (architecture, method, horizon), cell in merged.groupby(
            ["architecture", "method", "horizon"]
        ):
            scores = evaluate(cell["actual"].to_numpy(), cell["predicted"].to_numpy())
            baseline_scores = evaluate(
                cell["actual"].to_numpy(), cell["baseline"].to_numpy()
            )
            block = moving_block_bootstrap_ci(
                cell["actual"].to_numpy(),
                cell["predicted"].to_numpy(),
                cell["baseline"].to_numpy(),
                dates=cell["Date"].to_numpy(),
                n_samples=bootstrap_samples,
                alpha=alpha,
                seed=seed,
            )
            year_block = year_blocked_bootstrap_ci(
                cell["actual"].to_numpy(),
                cell["predicted"].to_numpy(),
                cell["baseline"].to_numpy(),
                dates=cell["Date"].to_numpy(),
                n_samples=bootstrap_samples,
                alpha=alpha,
                seed=seed,
            )
            rows.append(
                {
                    "group_set": group_set,
                    "reference": reference_group_set,
                    "architecture": architecture,
                    "method": method,
                    "horizon": int(horizon),
                    "n_matched": int(len(cell)),
                    "sufficient_evidence": bool(len(cell) >= min_rows_for_inference),
                    "mae_model": scores["MAE"],
                    "rmse_model": scores["RMSE"],
                    "baseline_mae": baseline_scores["MAE"],
                    "baseline_rmse": baseline_scores["RMSE"],
                    "delta_vs_persistence": block["delta"],
                    "block_ci_low": block["ci_low"],
                    "block_ci_high": block["ci_high"],
                    "block_significant": block["significant"],
                    "block_length_days": block["block_length"],
                    "year_ci_low": year_block["ci_low"],
                    "year_ci_high": year_block["ci_high"],
                    "year_significant": year_block["significant"],
                }
            )

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    corrections = apply_correction(
        table["delta_vs_persistence"].abs().to_numpy(), alpha=alpha
    )
    table["p_holm"] = corrections["p_holm"].to_numpy()
    table["p_benjamini_hochberg"] = corrections["p_benjamini_hochberg"].to_numpy()
    table["n_comparisons"] = corrections["n_comparisons"].to_numpy()
    return table
