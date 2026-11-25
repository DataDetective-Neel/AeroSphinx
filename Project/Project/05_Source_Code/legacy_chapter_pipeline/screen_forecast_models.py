"""Screen model families on expanding windows before the May–June holdout."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import aerosphinx_pipeline as aero
from compare_neural_network import ARCHITECTURES, fit_mlp, predict, score_recursive

VALIDATION_WINDOWS = [
    ("2024-02-17", "2024-03-16"),
    ("2024-03-17", "2024-04-16"),
    ("2024-04-17", "2024-05-16"),
]
DEPTHS = (3, 10, 12, 16, 20)


def fit_ridge(train, features, config):
    transform = config["target_transform"]
    target = aero._forward_transform(train[aero.TARGET], transform)
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(train[features], target)

    def predict_raw(frame):
        return aero._inverse_transform(model.predict(frame[features]), transform)

    return predict_raw


def main():
    results = Path(__file__).resolve().parent / "results"
    daily = pd.read_csv(results / "daily_multisource_dataset.csv", parse_dates=["Date"])
    config = aero.CONFIGURATIONS["tuned"]
    features, _ = aero.select_available_features(daily, config)
    complete = daily.dropna(subset=features + [aero.TARGET]).sort_values("Date")
    history_features = [feature for feature in features if feature in aero.RECURSIVE_FEATURES]
    rows = []

    for start, end in VALIDATION_WINDOWS:
        train, validation, _ = aero.chronological_split(complete, start, end)
        rf, xgb = aero.fit_ensemble(train, features, config)

        def ensemble(frame):
            return aero.raw_ensemble(rf, xgb, frame, features, config)[2]

        predictors = {
            "RF + XGBoost": ensemble,
            "Persistence": lambda frame: frame["PM25_Lag_1"],
            "Ridge (all features)": fit_ridge(train, features, config),
            "Ridge (PM2.5 history)": fit_ridge(train, history_features, config),
        }
        for depth in DEPTHS:
            layers = ARCHITECTURES["3-5"][0] if depth == 3 else (16, 12) + (8,) * (depth - 2)
            predictors[f"MLP ({depth} layers)"] = fit_mlp(train, features, config, layers)

        forecasts = score_recursive(daily, validation, features, predictors, config)
        for name, predictor in predictors.items():
            scores = aero.evaluate(validation[aero.TARGET], predict(validation, predictor, config))
            rows.append({
                "model": name,
                "validation_start": start,
                "validation_end": end,
                "training_days": len(train),
                "daily_predictions": len(validation),
                "forecast_origins": forecasts["origins"],
                "daily_mae": scores["MAE"],
                "five_day_mae": forecasts["metrics"][name]["MAE"],
            })
        print(f"Scored {start} through {end} on {len(validation)} days "
              f"and {forecasts['origins']} forecast origins.", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(results / "model_screening.csv", index=False)
    print(table.groupby("model")[["daily_mae", "five_day_mae"]]
          .mean().sort_values("five_day_mae").round(2).to_string())


if __name__ == "__main__":
    main()
