"""Compare a 10-hidden-layer MLP against the tuned AeroSphinx ensemble."""

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import aerosphinx_pipeline as aero

ARCHITECTURES = {
    "10": [(16, 12) + (8,) * 8],
    "3-5": [(16, 12, 8), (16, 12, 8, 6), (16, 12, 8, 6, 4)],
}
MLP_PARAMS = dict(solver="lbfgs", alpha=2.0, max_iter=2000,
                  random_state=aero.RANDOM_STATE)
HORIZON = 5


def fit_mlp(train, features, config, layers):
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=layers, **MLP_PARAMS),
    )
    target = aero._forward_transform(train[aero.TARGET], config["target_transform"])
    model.fit(train[features], target)

    def predict_raw(frame):
        return aero._inverse_transform(
            model.predict(frame[features]), config["target_transform"]
        )

    return predict_raw


def predict(frame, predict_raw, config):
    return aero.apply_anchor(predict_raw(frame), frame["PM25_Lag_1"], config)


def score_recursive(daily, test, features, predictors, config):
    """Score complete five-day windows with identical dates for every model."""
    daily = daily.set_index("Date")
    test_dates = set(test["Date"])
    records = []
    for start in test["Date"]:
        dates = pd.date_range(start, periods=HORIZON)
        if not all(date in test_dates for date in dates):
            continue
        prior_dates = pd.date_range(end=start - pd.Timedelta(days=1), periods=7)
        seed = daily[aero.TARGET].reindex(prior_dates)
        if seed.isna().any():
            continue
        histories = {name: seed.tolist() for name in predictors}
        for step, date in enumerate(dates, 1):
            observed = daily.loc[date]
            record = {"horizon": step, "actual": observed[aero.TARGET]}
            for name, predict_raw in predictors.items():
                history = histories[name]
                row = aero._update_recursive_features(observed, history)
                frame = row[features].to_frame().T.astype(float)
                prediction = predict(frame, predict_raw, config)[0]
                history.append(prediction)
                record[name] = prediction
            records.append(record)
    if not records:
        raise ValueError("No complete five-day test windows with seven days of history.")

    scored = pd.DataFrame(records)
    return {
        "origins": len(scored) // HORIZON,
        "predictions": len(scored),
        "metrics": {
            name: {
                **aero.evaluate(scored["actual"], scored[name]),
                "MAE_by_horizon": {
                    str(step): float((rows["actual"] - rows[name]).abs().mean())
                    for step, rows in scored.groupby("horizon")
                },
            }
            for name in predictors
        },
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=root / "results" / "daily_multisource_dataset.csv")
    parser.add_argument("--depth", choices=ARCHITECTURES, default="10",
                        help="Use 10 hidden layers or repeat the 3–5 layer search")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        suffix = "_10_layers" if args.depth == "10" else ""
        args.output = root / "results" / f"neural_network_comparison{suffix}.json"

    config = aero.CONFIGURATIONS["tuned"]
    daily = pd.read_csv(args.dataset, parse_dates=["Date"]).sort_values("Date")
    features, _ = aero.select_available_features(daily, config)
    complete = daily.dropna(subset=features + [aero.TARGET])
    train, test, split = aero.chronological_split(complete, "2024-05-17", "2024-06-18")
    development = train[train["Date"] < aero.VALIDATION_START]
    validation = train[train["Date"].between(aero.VALIDATION_START, aero.VALIDATION_END)]
    if development.empty or validation.empty:
        raise ValueError("Dataset must cover both development and validation dates.")

    candidates = []
    for layers in ARCHITECTURES[args.depth]:
        predictor = fit_mlp(development, features, config, layers)
        metrics = aero.evaluate(validation[aero.TARGET], predict(validation, predictor, config))
        candidates.append({"hidden_layers": layers, **metrics})
    winner = min(candidates, key=lambda candidate: candidate["MAE"])

    rf, xgb = aero.fit_ensemble(train, features, config)

    def ensemble_raw(frame):
        return aero.raw_ensemble(rf, xgb, frame, features, config)[2]

    def persistence_raw(frame):
        return frame["PM25_Lag_1"].to_numpy()

    predictors = {
        "RF_XGB_Ensemble": ensemble_raw,
        "Selected_MLP": fit_mlp(train, features, config, winner["hidden_layers"]),
        "Persistence": persistence_raw,
    }
    report = {
        "library_versions": {
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost")
        },
        "split": asdict(split),
        "features": features,
        "configuration": config,
        "mlp_parameters": MLP_PARAMS,
        "validation": {
            "start": aero.VALIDATION_START,
            "end": aero.VALIDATION_END,
            "n_development": len(development),
            "n_validation": len(validation),
            "selection_metric": "MAE",
            "candidates": candidates,
            "selected_hidden_layers": winner["hidden_layers"],
        },
        "daily_test": {
            name: aero.evaluate(test[aero.TARGET], predict(test, predictor, config))
            for name, predictor in predictors.items()
        },
        "recursive_test": score_recursive(daily, test, features, predictors, config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Selected hidden layers: {winner['hidden_layers']}")
    for label, scores in [("Daily", report["daily_test"]),
                          ("Recursive", report["recursive_test"]["metrics"])]:
        print(f"\n{label} test:")
        print(pd.DataFrame(scores).T[["MAE", "RMSE", "R2", "AQI_Accuracy"]].to_string())
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
