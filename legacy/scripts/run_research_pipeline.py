"""Publication-grade AeroSphinx PM2.5 research pipeline.

This script builds a leakage-aware daily PM2.5 modeling dataset from local CPCB,
INSAT AOD, and MERRA-2 files, trains Random Forest and XGBoost regressors with a
chronological split, evaluates a weighted ensemble, and writes figures/reports.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".codex_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


RANDOM_STATE = 42
TARGET = "PM25"
EPS = 1e-6
STATION_LAT = 28.4089
STATION_LON = 77.3178

BASE_FEATURES = [
    "AOD",
    "T2M",
    "TS",
    "QV2M",
    "U10M",
    "V10M",
    "SLP",
    "PBLH",
    "Temp_Diff",
    "Humidity_Ratio",
    "PM25_Lag_1",
    "PM25_Lag_2",
    "PM25_Lag_3",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
]

MERRA_REQUIRED = ["T2M", "TS", "QV2M", "U10M", "V10M", "SLP", "PBLH"]
MERRA_OPTIONAL_FALLBACKS = {"PBLH": ["PBLTOP"]}


@dataclass
class SplitSummary:
    strategy: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    n_train: int
    n_validation: int
    n_test: int


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def find_pm25_column(columns: Iterable[str]) -> str:
    for col in columns:
        if "PM2.5" in col or col.strip().upper() == "PM25":
            return col
    raise ValueError("Could not identify PM2.5 column")


def load_cpcb_daily(data_dir: Path, min_hours: int = 18) -> pd.DataFrame:
    records = []
    for csv_path in sorted(data_dir.glob("City_wise_raw_data_1Hr_*_Faridabad_1Hr.csv")):
        df = pd.read_csv(csv_path)
        pm_col = find_pm25_column(df.columns)
        hourly = pd.DataFrame(
            {
                "Timestamp": pd.to_datetime(df["Timestamp"], errors="coerce"),
                TARGET: pd.to_numeric(df[pm_col], errors="coerce"),
            }
        ).dropna(subset=["Timestamp"])
        hourly = hourly.set_index("Timestamp").sort_index()
        daily = hourly[TARGET].resample("D").agg(["mean", "count"]).reset_index()
        daily = daily[daily["count"] >= min_hours]
        daily["Date"] = daily["Timestamp"].dt.normalize()
        daily[TARGET] = daily["mean"]
        daily["source_file"] = csv_path.name
        records.append(daily[["Date", TARGET, "count", "source_file"]])

    if not records:
        raise FileNotFoundError(f"No CPCB hourly CSVs found in {data_dir}")

    out = pd.concat(records, ignore_index=True)
    out = out.groupby("Date", as_index=False).agg({TARGET: "mean", "count": "sum"})
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def parse_aod_date(path: Path, attrs: dict | None = None) -> pd.Timestamp | None:
    if attrs:
        value = attrs.get("Acquisition_Date")
        if value is not None:
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")
            elif hasattr(value, "tobytes"):
                value = value.tobytes().decode("utf-8", errors="ignore").strip("\x00")
            parsed = pd.to_datetime(str(value), format="%d%b%Y", errors="coerce")
            if pd.notna(parsed):
                return parsed.normalize()

    match = re.search(r"3DIMG_(\d{2}[A-Z]{3}\d{4})_", path.name.upper())
    if match:
        parsed = pd.to_datetime(match.group(1), format="%d%b%Y", errors="coerce")
        if pd.notna(parsed):
            return parsed.normalize()
    return None


def load_aod_daily(aod_dirs: list[Path], lat: float, lon: float, window_deg: float = 0.25) -> pd.DataFrame:
    import h5py

    rows = []
    for directory in aod_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.h5")):
            with h5py.File(path, "r") as f:
                date = parse_aod_date(path, dict(f.attrs))
                if date is None:
                    continue
                aod = np.asarray(f["AOD"][0], dtype=float)
                lat_arr = np.asarray(f["latitude"][:], dtype=float)
                lon_arr = np.asarray(f["longitude"][:], dtype=float)

                lat_mask = (lat_arr >= lat - window_deg) & (lat_arr <= lat + window_deg)
                lon_mask = (lon_arr >= lon - window_deg) & (lon_arr <= lon + window_deg)
                subset = aod[np.ix_(lat_mask, lon_mask)]
                valid = subset[(subset >= 0) & (subset <= 5)]
                rows.append(
                    {
                        "Date": date,
                        "AOD": float(np.nanmean(valid)) if valid.size else np.nan,
                        "aod_valid_pixels": int(valid.size),
                        "aod_file": path.name,
                    }
                )

    if not rows:
        raise FileNotFoundError("No readable AOD HDF5 files found")

    df = pd.DataFrame(rows)
    return (
        df.groupby("Date", as_index=False)
        .agg({"AOD": "mean", "aod_valid_pixels": "sum", "aod_file": "count"})
        .rename(columns={"aod_file": "aod_overpasses"})
        .sort_values("Date")
        .reset_index(drop=True)
    )


def load_merra_daily(merra_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    import xarray as xr

    rows = []
    missing_counts = {var: 0 for var in MERRA_REQUIRED}
    for path in sorted(merra_dir.glob("*.nc")):
        ds = xr.open_dataset(path)
        try:
            row = {"Date": pd.to_datetime(ds["time"].values[0]).normalize()}
            for var in MERRA_REQUIRED:
                source_var = var if var in ds else None
                if source_var is None:
                    for fallback in MERRA_OPTIONAL_FALLBACKS.get(var, []):
                        if fallback in ds:
                            source_var = fallback
                            break
                if source_var is None:
                    row[var] = np.nan
                    missing_counts[var] += 1
                    continue
                row[var] = float(ds[source_var].mean(skipna=True).values)
            rows.append(row)
        finally:
            ds.close()

    if not rows:
        raise FileNotFoundError(f"No MERRA-2 NetCDF files found in {merra_dir}")

    df = pd.DataFrame(rows)
    df = df.groupby("Date", as_index=False).mean(numeric_only=True)
    return df.sort_values("Date").reset_index(drop=True), {k: v for k, v in missing_counts.items() if v}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("Date").copy()
    out["Temp_Diff"] = out["TS"] - out["T2M"]
    out["Humidity_Ratio"] = out["QV2M"] / (out["T2M"] + EPS)
    for lag in [1, 2, 3]:
        out[f"PM25_Lag_{lag}"] = out[TARGET].shift(lag)
    day_of_year = out["Date"].dt.dayofyear
    days_in_year = np.where(out["Date"].dt.is_leap_year, 366, 365)
    out["day_sin"] = np.sin(2 * np.pi * day_of_year / days_in_year)
    out["day_cos"] = np.cos(2 * np.pi * day_of_year / days_in_year)
    month = out["Date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out


def split_chronologically(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitSummary]:
    preferred_train = model_df[(model_df["Date"].dt.year >= 2016) & (model_df["Date"].dt.year <= 2022)]
    preferred_val = model_df[model_df["Date"].dt.year == 2023]
    preferred_test = model_df[model_df["Date"].dt.year == 2024]

    if len(preferred_train) > 0 and len(preferred_val) > 0 and len(preferred_test) > 0:
        train, val, test = preferred_train, preferred_val, preferred_test
        strategy = "preferred_year_split_2016_2022_2023_2024"
    else:
        n = len(model_df)
        if n < 30:
            raise ValueError(f"Need at least 30 complete daily rows for fallback split; found {n}")
        train_end = max(1, int(n * 0.60))
        val_end = max(train_end + 1, int(n * 0.80))
        train = model_df.iloc[:train_end]
        val = model_df.iloc[train_end:val_end]
        test = model_df.iloc[val_end:]
        strategy = "fallback_single_period_chronological_60_20_20"

    def dates(frame: pd.DataFrame) -> tuple[str, str]:
        return frame["Date"].min().date().isoformat(), frame["Date"].max().date().isoformat()

    tr0, tr1 = dates(train)
    va0, va1 = dates(val)
    te0, te1 = dates(test)
    summary = SplitSummary(strategy, tr0, tr1, va0, va1, te0, te1, len(train), len(val), len(test))
    return train, val, test, summary


def aqi_category(values: np.ndarray | pd.Series) -> np.ndarray:
    bins = [-np.inf, 30, 60, 90, 120, 250, np.inf]
    labels = np.array(["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"])
    idx = np.digitize(np.asarray(values, dtype=float), bins[1:-1], right=False)
    return labels[idx]


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "AQI_Accuracy": float(accuracy_score(aqi_category(y_true), aqi_category(y_pred))),
    }


def tune_rf(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> tuple[RandomForestRegressor, dict]:
    param_grid = [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf, "max_features": mf}
        for n in [200, 400]
        for d in [6, 10, None]
        for leaf in [1, 3]
        for mf in ["sqrt", None]
    ]
    best = None
    best_score = np.inf
    for params in param_grid:
        model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
        model.fit(train[features], train[TARGET])
        pred = model.predict(val[features])
        score = mean_absolute_error(val[TARGET], pred)
        if score < best_score:
            best_score = score
            best = params
    final_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **best)
    final_model.fit(pd.concat([train, val])[features], pd.concat([train, val])[TARGET])
    return final_model, {"params": best, "validation_MAE": float(best_score)}


def tune_xgb(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> tuple[XGBRegressor, dict]:
    param_grid = [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": 0.85, "colsample_bytree": 0.85}
        for n in [150, 300]
        for d in [2, 3, 4]
        for lr in [0.03, 0.06, 0.10]
    ]
    best = None
    best_score = np.inf
    for params in param_grid:
        model = XGBRegressor(
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            n_jobs=-1,
            reg_lambda=1.0,
            **params,
        )
        model.fit(train[features], train[TARGET], verbose=False)
        pred = model.predict(val[features])
        score = mean_absolute_error(val[TARGET], pred)
        if score < best_score:
            best_score = score
            best = params
    final_params = dict(best)
    final_model = XGBRegressor(
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        n_jobs=-1,
        reg_lambda=1.0,
        **final_params,
    )
    final_model.fit(pd.concat([train, val])[features], pd.concat([train, val])[TARGET], verbose=False)
    return final_model, {"params": final_params, "validation_MAE": float(best_score)}


def make_predictions(rf: RandomForestRegressor, xgb: XGBRegressor, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = frame[["Date", TARGET]].copy()
    out["RF"] = rf.predict(frame[features])
    out["XGB"] = xgb.predict(frame[features])
    out["Ensemble"] = 0.6 * out["RF"] + 0.4 * out["XGB"]
    out["Residual"] = out[TARGET] - out["Ensemble"]
    out["Actual_AQI_Class"] = aqi_category(out[TARGET])
    out["Predicted_AQI_Class"] = aqi_category(out["Ensemble"])
    return out


def weighted_feature_importance(rf: RandomForestRegressor, xgb: XGBRegressor, features: list[str]) -> pd.DataFrame:
    rf_imp = np.asarray(rf.feature_importances_, dtype=float)
    xgb_imp = np.asarray(xgb.feature_importances_, dtype=float)
    rf_imp = rf_imp / rf_imp.sum() if rf_imp.sum() else rf_imp
    xgb_imp = xgb_imp / xgb_imp.sum() if xgb_imp.sum() else xgb_imp
    ensemble_imp = 0.6 * rf_imp + 0.4 * xgb_imp
    out = pd.DataFrame({"feature": features, "importance": ensemble_imp, "rf_importance": rf_imp, "xgb_importance": xgb_imp})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def recursive_forecast(
    rf: RandomForestRegressor,
    xgb: XGBRegressor,
    feature_source: pd.DataFrame,
    origin_date: pd.Timestamp,
    features: list[str],
    horizon: int = 5,
) -> pd.DataFrame:
    source = feature_source.set_index("Date").sort_index()
    history = source.loc[:origin_date, TARGET].dropna().tolist()
    if len(history) < 3:
        raise ValueError("Need at least three observed PM2.5 values before forecast origin")

    preds = []
    rolling = history[-3:]
    for step in range(1, horizon + 1):
        forecast_date = origin_date + pd.Timedelta(days=step)
        if forecast_date not in source.index:
            break
        row = source.loc[forecast_date].copy()
        row["PM25_Lag_1"] = rolling[-1]
        row["PM25_Lag_2"] = rolling[-2]
        row["PM25_Lag_3"] = rolling[-3]
        x = row[features].to_frame().T
        rf_pred = float(rf.predict(x)[0])
        xgb_pred = float(xgb.predict(x)[0])
        ensemble = 0.6 * rf_pred + 0.4 * xgb_pred
        preds.append(
            {
                "Date": forecast_date,
                "horizon_day": step,
                "actual_PM25": float(row[TARGET]) if pd.notna(row[TARGET]) else np.nan,
                "predicted_PM25": ensemble,
                "RF": rf_pred,
                "XGB": xgb_pred,
            }
        )
        rolling.append(ensemble)
        rolling = rolling[-3:]
    return pd.DataFrame(preds)


def plot_outputs(
    cpcb_daily: pd.DataFrame,
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    forecast: pd.DataFrame,
    figures_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(predictions["Date"], predictions[TARGET], marker="o", label="Actual")
    ax.plot(predictions["Date"], predictions["Ensemble"], marker="o", label="Predicted")
    ax.set_title("Actual vs Predicted Daily PM2.5")
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures_dir / "actual_vs_predicted_pm25.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color="black", linewidth=1)
    ax.scatter(predictions["Date"], predictions["Residual"], c=predictions["Residual"], cmap="coolwarm", edgecolor="black")
    ax.set_title("Residuals over Test Period")
    ax.set_ylabel("Actual - Predicted PM2.5")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures_dir / "residual_plot.png", dpi=200)
    plt.close(fig)

    top = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], color="#2a9d8f")
    ax.set_title("Weighted Ensemble Feature Importance")
    ax.set_xlabel("Normalized importance")
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_importance.png", dpi=200)
    plt.close(fig)

    monthly = cpcb_daily.assign(month=cpcb_daily["Date"].dt.month).groupby("month")[TARGET].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(monthly.index, monthly.values, marker="o", color="#264653")
    ax.set_xticks(range(1, 13))
    ax.set_title("Monthly PM2.5 Seasonality from CPCB Daily Means")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean PM2.5 (ug/m3)")
    fig.tight_layout()
    fig.savefig(figures_dir / "monthly_pm25_seasonality.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(forecast["horizon_day"], forecast["predicted_PM25"], marker="o", label="Recursive forecast")
    if forecast["actual_PM25"].notna().any():
        ax.plot(forecast["horizon_day"], forecast["actual_PM25"], marker="o", label="Actual")
    ax.set_title("Five-Day Recursive Forecast Example")
    ax.set_xlabel("Forecast horizon day")
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.set_xticks(forecast["horizon_day"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "five_day_forecast_example.png", dpi=200)
    plt.close(fig)


def write_report(
    report_path: Path,
    audit: dict,
    split: SplitSummary,
    metrics: dict,
    importance: pd.DataFrame,
    files_written: list[str],
) -> None:
    lines = [
        "# AeroSphinx Scientific Audit Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Repository Audit",
        "- Multiple duplicate model scripts were found at the repository root and under `ML model/`.",
        "- Legacy scripts use random train/test splits and do not implement recursive five-day forecasting.",
        "- `pawanai-project` contains synthetic/random data generation and is not suitable for publication metrics.",
        "- Several CSVs and notebooks are exploratory or stale; the reproducible entry point is now `scripts/run_research_pipeline.py`.",
        "",
        "## Scientific Issues Found",
    ]
    for issue in audit["scientific_issues"]:
        lines.append(f"- {issue}")

    lines.extend(["", "## Code Issues Found"])
    for issue in audit["code_issues"]:
        lines.append(f"- {issue}")

    lines.extend(
        [
            "",
            "## Data Coverage",
            f"- CPCB daily rows: {audit['coverage']['cpcb_daily_rows']} ({audit['coverage']['cpcb_start']} to {audit['coverage']['cpcb_end']}).",
            f"- AOD daily rows: {audit['coverage']['aod_daily_rows']} ({audit['coverage']['aod_start']} to {audit['coverage']['aod_end']}).",
            f"- MERRA daily rows: {audit['coverage']['merra_daily_rows']} ({audit['coverage']['merra_start']} to {audit['coverage']['merra_end']}).",
            f"- Complete model rows after leakage-safe lags and missing-data filtering: {audit['coverage']['model_rows']}.",
            "",
            "## Chronological Split",
            f"- Strategy: {split.strategy}",
            f"- Train: {split.train_start} to {split.train_end} (n={split.n_train})",
            f"- Validation: {split.validation_start} to {split.validation_end} (n={split.n_validation})",
            f"- Test: {split.test_start} to {split.test_end} (n={split.n_test})",
            "",
            "## Final Test Metrics",
        ]
    )
    for model_name, values in metrics.items():
        lines.append(
            f"- {model_name}: MAE={values['MAE']:.3f}, RMSE={values['RMSE']:.3f}, "
            f"R2={values['R2']:.3f}, AQI accuracy={values['AQI_Accuracy']:.3f}"
        )

    lines.extend(["", "## Feature Importance Ranking"])
    for _, row in importance.head(17).iterrows():
        lines.append(f"- {row['feature']}: {row['importance']:.5f}")

    lines.extend(
        [
            "",
            "## Recommended Springer Text",
            "The final empirical pipeline aggregated CPCB hourly PM2.5 observations to daily means, extracted station-window INSAT AOD from local HDF5 granules, and averaged available MERRA-2 meteorological variables from local NetCDF files. Lagged PM2.5 predictors were computed strictly from prior days and the model was evaluated using a chronological split. Because the local satellite and MERRA-2 files only covered a 2024 subset and the MERRA-2 files did not include PBLH, the full 2016-2022/2023/2024 multi-source protocol could not be executed without additional historical AOD/MERRA-2 data and a valid boundary-layer-height product. Reported metrics should therefore be described as a limited strict multi-source benchmark over the available 2024 window, not as a full urban-India generalization result.",
            "",
            "## Files Written",
        ]
    )
    for file in files_written:
        lines.append(f"- `{file}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AeroSphinx reproducible PM2.5 research pipeline.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--aod-dir", type=Path, action="append", default=[ROOT / "ML model" / "aod_folder", ROOT / "AOD daata"])
    parser.add_argument("--merra-dir", type=Path, default=ROOT / "ML model" / "merra_downloads")
    parser.add_argument("--min-hours", type=int, default=18)
    args = parser.parse_args()

    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    models_dir = ROOT / "models"
    ensure_dirs([results_dir, figures_dir, models_dir])

    cpcb_daily = load_cpcb_daily(args.data_dir, min_hours=args.min_hours)
    aod_daily = load_aod_daily(args.aod_dir, STATION_LAT, STATION_LON)
    merra_daily, missing_merra_counts = load_merra_daily(args.merra_dir)

    combined = cpcb_daily.merge(aod_daily, on="Date", how="left").merge(merra_daily, on="Date", how="left")
    combined = engineer_features(combined)

    unavailable_features = [feature for feature in BASE_FEATURES if feature not in combined.columns or combined[feature].isna().all()]
    model_features = [feature for feature in BASE_FEATURES if feature not in unavailable_features]
    model_df = combined.dropna(subset=model_features + [TARGET]).sort_values("Date").reset_index(drop=True)

    train, val, test, split_summary = split_chronologically(model_df)
    rf_model, rf_tuning = tune_rf(train, val, model_features)
    xgb_model, xgb_tuning = tune_xgb(train, val, model_features)

    pred = make_predictions(rf_model, xgb_model, test, model_features)
    metrics = {
        "RandomForest": metrics_dict(pred[TARGET], pred["RF"]),
        "XGBoost": metrics_dict(pred[TARGET], pred["XGB"]),
        "WeightedEnsemble_0.6RF_0.4XGB": metrics_dict(pred[TARGET], pred["Ensemble"]),
    }

    importance = weighted_feature_importance(rf_model, xgb_model, model_features)

    forecast_origin = model_df["Date"].iloc[-6]
    forecast = recursive_forecast(rf_model, xgb_model, model_df, forecast_origin, model_features, horizon=5)

    plot_outputs(cpcb_daily, pred, importance, forecast, figures_dir)

    combined.to_csv(results_dir / "daily_multisource_dataset.csv", index=False)
    model_df.to_csv(results_dir / "modeling_dataset_complete_cases.csv", index=False)
    pred.to_csv(results_dir / "test_predictions.csv", index=False)
    forecast.to_csv(results_dir / "five_day_recursive_forecast.csv", index=False)
    importance.to_csv(results_dir / "feature_importance.csv", index=False)

    joblib.dump(rf_model, models_dir / "random_forest_pm25.joblib")
    joblib.dump(xgb_model, models_dir / "xgboost_pm25.joblib")

    coverage = {
        "cpcb_daily_rows": int(len(cpcb_daily)),
        "cpcb_start": cpcb_daily["Date"].min().date().isoformat(),
        "cpcb_end": cpcb_daily["Date"].max().date().isoformat(),
        "aod_daily_rows": int(len(aod_daily.dropna(subset=["AOD"]))),
        "aod_start": aod_daily["Date"].min().date().isoformat(),
        "aod_end": aod_daily["Date"].max().date().isoformat(),
        "merra_daily_rows": int(len(merra_daily)),
        "merra_start": merra_daily["Date"].min().date().isoformat(),
        "merra_end": merra_daily["Date"].max().date().isoformat(),
        "model_rows": int(len(model_df)),
        "model_start": model_df["Date"].min().date().isoformat(),
        "model_end": model_df["Date"].max().date().isoformat(),
    }

    scientific_issues = [
        "Preferred 2016-2022/2023/2024 split is impossible for the strict multi-source dataset because local AOD and MERRA-2 files cover only a 2024 subset.",
        "Local MERRA-2 NetCDF files do not contain PBLH; the model excludes PBLH and reports it as unavailable rather than fabricating it.",
        "AOD granules include fill values (-999) and invalid negative values; these are filtered before daily aggregation.",
        "The resulting strict multi-source benchmark is small and should not be reported as a full urban-India generalization result.",
    ]
    fully_missing = [var for var, count in missing_merra_counts.items() if count >= len(merra_daily)]
    partially_missing = {
        var: count for var, count in missing_merra_counts.items() if 0 < count < len(merra_daily)
    }
    if fully_missing:
        scientific_issues.append(f"MERRA variables absent from all local files: {', '.join(fully_missing)}.")
    if partially_missing:
        details = ", ".join(f"{var} missing on {count}/{len(merra_daily)} days" for var, count in partially_missing.items())
        scientific_issues.append(f"MERRA variables partially missing before complete-case filtering: {details}.")
    if unavailable_features:
        scientific_issues.append(f"Unavailable requested model features removed from empirical training: {', '.join(unavailable_features)}.")

    code_issues = [
        "Legacy scripts use random train_test_split(), causing time-series leakage risk.",
        "Legacy scripts hard-code prediction dates and manually inject AOD values.",
        "Several scripts reference missing files such as ../data/merged_pm_aod.csv and sample_insat_aod.nc.",
        "The old target column encoding appears corrupted in source CSV headers; the new pipeline detects the PM2.5 column by name pattern.",
        "Duplicate root and ML model scripts are byte-identical and should not both be maintained.",
        "Synthetic demo code in pawanai-project should be excluded from manuscript performance claims.",
    ]

    audit = {
        "coverage": coverage,
        "features_requested": BASE_FEATURES,
        "features_used": model_features,
        "features_unavailable": unavailable_features,
        "split": asdict(split_summary),
        "rf_tuning": rf_tuning,
        "xgb_tuning": xgb_tuning,
        "metrics": metrics,
        "scientific_issues": scientific_issues,
        "code_issues": code_issues,
    }

    (results_dir / "metrics.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    files_written = [
        "results/daily_multisource_dataset.csv",
        "results/modeling_dataset_complete_cases.csv",
        "results/test_predictions.csv",
        "results/five_day_recursive_forecast.csv",
        "results/feature_importance.csv",
        "results/metrics.json",
        "results/audit_report.md",
        "figures/actual_vs_predicted_pm25.png",
        "figures/residual_plot.png",
        "figures/feature_importance.png",
        "figures/monthly_pm25_seasonality.png",
        "figures/five_day_forecast_example.png",
        "models/random_forest_pm25.joblib",
        "models/xgboost_pm25.joblib",
    ]
    write_report(results_dir / "audit_report.md", audit, split_summary, metrics, importance, files_written)

    print(json.dumps({"metrics": metrics, "split": asdict(split_summary), "features_used": model_features}, indent=2))


if __name__ == "__main__":
    main()
