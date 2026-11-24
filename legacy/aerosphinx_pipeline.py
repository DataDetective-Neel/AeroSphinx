from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor
RANDOM_STATE = 42
TARGET = "PM25"
EPS = 1e-6

# Faridabad CPCB monitoring station (Sector 16A), Haryana, Delhi-NCR.
STATION_LAT = 28.4089
STATION_LON = 77.3178
STATION_NAME = "Faridabad"

# Half-width of the square window used for nearest-neighbour collocation of
# satellite grid cells to the station coordinate (+/- 0.25 deg ~ 27 km).
COLLOCATION_WINDOW_DEG = 0.25

# Section 4, Pre-processing: PM2.5 above 500 ug/m3 or below zero is missing.
PM25_VALID_MIN = 0.0
PM25_VALID_MAX = 500.0

# Minimum number of valid hourly CPCB samples required to accept a daily mean.
MIN_HOURS_PER_DAY = 18

# Section 4: residual gaps forward-filled within a three-day window.
FORWARD_FILL_LIMIT_DAYS = 3

# Section 3: y_ensemble = 0.6 * y_RF + 0.4 * y_XGB, clipped to [0, 500].
ENSEMBLE_WEIGHT_RF = 0.6
ENSEMBLE_WEIGHT_XGB = 0.4

# Section 5: hyperparameters selected manually from values established for
# tabular environmental datasets of comparable size, not by automated search.
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# The two reproducible model configurations.
#
#   target_transform    "log1p" fits the ensemble on log1p(PM2.5); "none" fits
#                       the raw concentration as in the literal Section 3 text.
#   recency_halflife    days; training weight = 0.5 ** (age / halflife).
#                       None disables recency weighting.
#   persistence_anchor  alpha in y = (1-alpha) * y_model + alpha * y_(t-1).
#                       0.0 is the literal Section 3 ensemble.
#   extra_features      leakage-safe rolling means added to the feature vector.
CONFIGURATIONS = {
    "tuned": {
        "target_transform": "log1p",
        "recency_halflife": None,
        "persistence_anchor": 0.4,
        "extra_features": ["PM25_Roll3", "PM25_Roll7"],
        "description": (
            "Validation-selected configuration: log1p target, 0.4 persistence "
            "anchor, three- and seven-day rolling-mean features."
        ),
    },
    "chapter": {
        "target_transform": "none",
        "recency_halflife": None,
        "persistence_anchor": 0.0,
        "extra_features": [],
        "description": (
            "Literal Section 3 formulation: unweighted ensemble on the raw "
            "concentration, no anchor, no rolling features."
        ),
    },
}
DEFAULT_CONFIG = "tuned"

# Validation window used for configuration selection. It ends before the test
# window begins, so no test observation influences any modelling choice.
VALIDATION_START = "2024-04-10"
VALIDATION_END = "2024-05-16"

# Section 5: CPCB National Air Quality Index bands for PM2.5 (ug/m3).
AQI_BOUNDS = [30, 60, 90, 120, 250]
AQI_LABELS = np.array(
    ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
)

# MERRA-2 fields requested by the chapter. PBLH is requested but is not present
# in the subsetted tavg1_2d_slv_Nx granules distributed with this study.
MERRA_REQUESTED = ["T2M", "TS", "QV2M", "U10M", "V10M", "SLP", "PBLH"]
MERRA_FALLBACKS = {"PBLH": ["PBLTOP"]}

# Section 3, feature vector X_t. Rolling means are appended by configuration.
BASE_FEATURE_ORDER = [
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

# Features whose values must be recomputed from predictions during recursion.
RECURSIVE_FEATURES = ["PM25_Lag_1", "PM25_Lag_2", "PM25_Lag_3",
                      "PM25_Roll3", "PM25_Roll7"]

FEATURE_DESCRIPTIONS = {
    "AOD": ("INSAT-3D aerosol optical depth at 550 nm, station window mean",
            "Column aerosol loading, the primary satellite predictor"),
    "T2M": ("MERRA-2 2 m air temperature (K)",
            "Thermal state controlling vertical mixing"),
    "TS": ("MERRA-2 surface skin temperature (K)",
           "Surface heating driving convective dispersion"),
    "QV2M": ("MERRA-2 2 m specific humidity (kg/kg)",
             "Hygroscopic growth of aerosol particles"),
    "U10M": ("MERRA-2 10 m zonal wind (m/s)",
             "East-west advection of pollutants"),
    "V10M": ("MERRA-2 10 m meridional wind (m/s)",
             "North-south advection of pollutants"),
    "SLP": ("MERRA-2 sea-level pressure (Pa)",
            "Synoptic regime and stagnation indicator"),
    "PBLH": ("MERRA-2 planetary boundary layer height (m)",
             "Dilution volume; requested but unavailable in local granules"),
    "Temp_Diff": ("TS - T2M (K)",
                  "Near-surface stability proxy for inversion strength"),
    "Humidity_Ratio": ("QV2M / T2M",
                       "Combined moisture-temperature aerosol growth term"),
    "PM25_Lag_1": ("PM2.5 on day t-1 (ug/m3)",
                   "Day-to-day persistence of pollution episodes"),
    "PM25_Lag_2": ("PM2.5 on day t-2 (ug/m3)", "Two-day pollution memory"),
    "PM25_Lag_3": ("PM2.5 on day t-3 (ug/m3)", "Three-day pollution memory"),
    "PM25_Roll3": ("Mean PM2.5 over days t-1 to t-3 (ug/m3)",
                   "Short-term pollution level, less noisy than a single lag"),
    "PM25_Roll7": ("Mean PM2.5 over days t-1 to t-7 (ug/m3)",
                   "Weekly background level of the current episode"),
    "day_sin": ("sin(2*pi*d/7), d = day of week",
                "Cyclic weekly encoding of sub-weekly emission cycles"),
    "day_cos": ("cos(2*pi*d/7), d = day of week",
                "Cyclic weekly encoding, quadrature component"),
    "month_sin": ("sin(2*pi*m/12), m = month",
                  "Cyclic seasonal encoding of the annual pollution cycle"),
    "month_cos": ("cos(2*pi*m/12), m = month",
                  "Cyclic seasonal encoding, quadrature component"),
}


@dataclass
class SplitSummary:
    """Chronological hold-out description (Section 5)."""

    strategy: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int


# --------------------------------------------------------------------------
# Layer 1: Data Acquisition
# --------------------------------------------------------------------------


def _find_pm25_column(columns) -> str:
    """CPCB exports vary in header encoding; locate the PM2.5 column by name."""
    for col in columns:
        upper = col.strip().upper().replace(" ", "")
        if "PM2.5" in col or upper.startswith("PM25"):
            return col
    raise ValueError(f"No PM2.5 column found among: {list(columns)}")


def load_cpcb_daily(cpcb_dir: Path, min_hours: int = MIN_HOURS_PER_DAY) -> pd.DataFrame:
    """Aggregate CPCB hourly PM2.5 CSV exports to quality-screened daily means.

    Section 4, Pre-processing: mixed date formats are standardised with UTC
    anchoring, and PM2.5 values outside [0, 500] ug/m3 are treated as missing.
    A daily mean is retained only when at least ``min_hours`` valid hourly
    samples are present.
    """
    frames = []
    files = sorted(cpcb_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CPCB CSV files found in {cpcb_dir}")

    for path in files:
        raw = pd.read_csv(path)
        pm_col = _find_pm25_column(raw.columns)
        hourly = pd.DataFrame(
            {
                "Timestamp": pd.to_datetime(raw["Timestamp"], errors="coerce"),
                TARGET: pd.to_numeric(raw[pm_col], errors="coerce"),
            }
        ).dropna(subset=["Timestamp"])

        invalid = (hourly[TARGET] < PM25_VALID_MIN) | (hourly[TARGET] > PM25_VALID_MAX)
        hourly.loc[invalid, TARGET] = np.nan

        daily = (
            hourly.set_index("Timestamp")
            .sort_index()[TARGET]
            .resample("D")
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"Timestamp": "Date", "mean": TARGET, "count": "valid_hours"})
        )
        frames.append(daily[["Date", TARGET, "valid_hours"]])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.groupby("Date", as_index=False).agg(
        {TARGET: "mean", "valid_hours": "sum"}
    )
    combined = combined[combined["valid_hours"] >= min_hours]
    return combined.dropna(subset=[TARGET]).sort_values("Date").reset_index(drop=True)


def _parse_insat_date(path: Path, attrs: dict) -> pd.Timestamp | None:
    """Read the acquisition date from an INSAT-3D granule, or from its name."""
    value = attrs.get("Acquisition_Date")
    if value is not None:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        elif hasattr(value, "tobytes"):
            value = value.tobytes().decode("utf-8", errors="ignore").strip("\x00")
        parsed = pd.to_datetime(str(value).strip(), format="%d%b%Y", errors="coerce")
        if pd.notna(parsed):
            return parsed.normalize()

    match = re.search(r"3DIMG_(\d{2}[A-Z]{3}\d{4})_", path.name.upper())
    if match:
        parsed = pd.to_datetime(match.group(1), format="%d%b%Y", errors="coerce")
        if pd.notna(parsed):
            return parsed.normalize()
    return None


def load_aod_daily(
    aod_dirs: list[Path],
    lat: float = STATION_LAT,
    lon: float = STATION_LON,
    window_deg: float = COLLOCATION_WINDOW_DEG,
) -> pd.DataFrame:
    """Extract station-collocated INSAT-3D AOD from L2G HDF5 granules.

    Section 4: nearest-neighbour spatial collocation assigns each CPCB station
    the AOD of the surrounding satellite grid cells. Fill values (-999) and
    physically impossible retrievals are removed before averaging, and multiple
    same-day overpasses are averaged into one daily value.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("h5py is required to read INSAT-3D HDF5 granules") from exc

    records = []
    for directory in aod_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.h5")):
            with h5py.File(path, "r") as handle:
                date = _parse_insat_date(path, dict(handle.attrs))
                if date is None:
                    continue
                grid = np.asarray(handle["AOD"][0], dtype=float)
                lats = np.asarray(handle["latitude"][:], dtype=float)
                lons = np.asarray(handle["longitude"][:], dtype=float)

            lat_mask = (lats >= lat - window_deg) & (lats <= lat + window_deg)
            lon_mask = (lons >= lon - window_deg) & (lons <= lon + window_deg)
            window = grid[np.ix_(lat_mask, lon_mask)]
            valid = window[(window >= 0.0) & (window <= 5.0)]

            records.append(
                {
                    "Date": date,
                    "AOD": float(np.nanmean(valid)) if valid.size else np.nan,
                    "aod_valid_pixels": int(valid.size),
                }
            )

    if not records:
        raise FileNotFoundError(f"No readable INSAT-3D granules in {aod_dirs}")

    frame = pd.DataFrame(records)
    return (
        frame.groupby("Date", as_index=False)
        .agg(AOD=("AOD", "mean"), aod_valid_pixels=("aod_valid_pixels", "sum"),
             aod_overpasses=("AOD", "size"))
        .sort_values("Date")
        .reset_index(drop=True)
    )


def load_merra_daily(merra_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read MERRA-2 reanalysis NetCDF granules as daily means over the station box.

    Section 4: MERRA-2 fields are spatially averaged over the Faridabad bounding
    box contained in each subsetted granule. Requested variables that no granule
    provides are returned as NaN and reported, never substituted.
    """
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover
        raise ImportError("xarray and netCDF4 are required to read MERRA-2 files") from exc

    files = sorted(merra_dir.glob("*.nc")) + sorted(merra_dir.glob("*.nc4"))
    if not files:
        raise FileNotFoundError(f"No MERRA-2 NetCDF files found in {merra_dir}")

    rows = []
    missing_counts = {var: 0 for var in MERRA_REQUESTED}
    for path in files:

        dataset = xr.open_dataset(path)
        try:
            row = {"Date": pd.to_datetime(dataset["time"].values[0]).normalize()}
            for var in MERRA_REQUESTED:
                source = var if var in dataset else None
                if source is None:
                    for fallback in MERRA_FALLBACKS.get(var, []):
                        if fallback in dataset:
                            source = fallback
                            break
                if source is None:
                    row[var] = np.nan
                    missing_counts[var] += 1
                    continue
                row[var] = float(dataset[source].mean(skipna=True).values)
            rows.append(row)
        finally:
            dataset.close()

    daily = pd.DataFrame(rows).groupby("Date", as_index=False).mean(numeric_only=True)
    daily = daily.sort_values("Date").reset_index(drop=True)
    return daily, {k: v for k, v in missing_counts.items() if v}


# --------------------------------------------------------------------------
# Layer 2: Harmonisation
# --------------------------------------------------------------------------


def harmonise(
    cpcb: pd.DataFrame,
    aod: pd.DataFrame,
    merra: pd.DataFrame,
    ffill_limit: int = FORWARD_FILL_LIMIT_DAYS,
) -> pd.DataFrame:
    """Merge the three daily streams onto a common calendar and fill short gaps.

    Section 4: the three sources are merged into a unified tabular dataset;
    missing values are forward-filled within a three-day window, with residual
    gaps excluded from training downstream.
    """
    merged = (
        cpcb.merge(aod, on="Date", how="left")
        .merge(merra, on="Date", how="left")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    gap_columns = [c for c in ["AOD"] + MERRA_REQUESTED if c in merged.columns]
    merged[gap_columns] = merged[gap_columns].ffill(limit=ffill_limit)
    return merged


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the physically motivated feature set of Section 3.

    Lags and rolling means use only previous days, so no future information
    enters the feature vector. Calendar variables are encoded as sine-cosine
    pairs to preserve the circular topology of weekly and annual cycles:

        day_sin   = sin(2*pi*d/7),   day_cos   = cos(2*pi*d/7),   d = day of week
        month_sin = sin(2*pi*m/12),  month_cos = cos(2*pi*m/12),  m = month
    """
    out = frame.sort_values("Date").copy()

    out["Temp_Diff"] = out["TS"] - out["T2M"]
    out["Humidity_Ratio"] = out["QV2M"] / (out["T2M"] + EPS)

    for lag in (1, 2, 3):
        out[f"PM25_Lag_{lag}"] = out[TARGET].shift(lag)
    out["PM25_Roll3"] = out[TARGET].shift(1).rolling(3).mean()
    out["PM25_Roll7"] = out[TARGET].shift(1).rolling(7).mean()

    day_of_week = out["Date"].dt.day_of_week
    out["day_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    out["day_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

    month = out["Date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    return out


def select_available_features(
    frame: pd.DataFrame, config: dict
) -> tuple[list[str], list[str]]:
    """Split the requested feature set into available and unavailable columns."""
    requested = BASE_FEATURE_ORDER + list(config["extra_features"])
    unavailable = [
        feature
        for feature in requested
        if feature not in frame.columns or frame[feature].isna().all()
    ]
    available = [f for f in requested if f not in unavailable]
    return available, unavailable


# --------------------------------------------------------------------------
# Layer 3: Modelling
# --------------------------------------------------------------------------


def chronological_split(
    model_df: pd.DataFrame, test_start: str, test_end: str
) -> tuple[pd.DataFrame, pd.DataFrame, SplitSummary]:
    """Reserve the most recent window for testing (Section 5).

    Everything strictly before ``test_start`` is used for training, so no
    observation from the future informs the fitted model.
    """
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end)

    train = model_df[model_df["Date"] < start]
    test = model_df[(model_df["Date"] >= start) & (model_df["Date"] <= end)]

    if len(train) < 30:
        raise ValueError(f"Training set too small: {len(train)} rows")
    if len(test) < 5:
        raise ValueError(f"Test set too small: {len(test)} rows")

    summary = SplitSummary(
        strategy="chronological_holdout_most_recent_window",
        train_start=train["Date"].min().date().isoformat(),
        train_end=train["Date"].max().date().isoformat(),
        test_start=test["Date"].min().date().isoformat(),
        test_end=test["Date"].max().date().isoformat(),
        n_train=len(train),
        n_test=len(test),
    )
    return train, test, summary


def _recency_weights(train: pd.DataFrame, halflife: int | None) -> np.ndarray | None:
    """Exponential recency weights: weight = 0.5 ** (age_in_days / halflife)."""
    if not halflife:
        return None
    age = (train["Date"].max() - train["Date"]).dt.days.to_numpy(dtype=float)
    return 0.5 ** (age / float(halflife))


def _forward_transform(values: np.ndarray, transform: str) -> np.ndarray:
    return np.log1p(values) if transform == "log1p" else values


def _inverse_transform(values: np.ndarray, transform: str) -> np.ndarray:
    return np.expm1(values) if transform == "log1p" else values


def fit_ensemble(
    train: pd.DataFrame, features: list[str], config: dict
) -> tuple[RandomForestRegressor, XGBRegressor]:
    """Fit the Random Forest and XGBoost members of the ensemble (Section 3)."""
    y = _forward_transform(train[TARGET].to_numpy(dtype=float),
                           config["target_transform"])
    weights = _recency_weights(train, config["recency_halflife"])

    rf = RandomForestRegressor(**RF_PARAMS).fit(train[features], y, sample_weight=weights)
    xgb = XGBRegressor(**XGB_PARAMS).fit(train[features], y, sample_weight=weights)
    return rf, xgb


def raw_ensemble(
    rf: RandomForestRegressor,
    xgb: XGBRegressor,
    frame: pd.DataFrame,
    features: list[str],
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Model-only prediction, on the natural PM2.5 scale, before anchoring."""
    rf_raw = rf.predict(frame[features])
    xgb_raw = xgb.predict(frame[features])
    blended = ENSEMBLE_WEIGHT_RF * rf_raw + ENSEMBLE_WEIGHT_XGB * xgb_raw

    transform = config["target_transform"]
    return (
        _inverse_transform(rf_raw, transform),
        _inverse_transform(xgb_raw, transform),
        _inverse_transform(blended, transform),
    )


def apply_anchor(model_pred: np.ndarray, previous_day: np.ndarray, config: dict) -> np.ndarray:
    """Damp the model towards the previous observation and clip to physical range.

        y = (1 - alpha) * y_model + alpha * y_(t-1)

    With alpha = 0 this is the literal Section 3 ensemble.
    """
    alpha = float(config["persistence_anchor"])
    combined = (1.0 - alpha) * np.asarray(model_pred, dtype=float)
    if alpha > 0:
        combined = combined + alpha * np.asarray(previous_day, dtype=float)
    return np.clip(combined, PM25_VALID_MIN, PM25_VALID_MAX)


def ensemble_predict(
    rf: RandomForestRegressor,
    xgb: XGBRegressor,
    frame: pd.DataFrame,
    features: list[str],
    config: dict,
) -> pd.DataFrame:
    """Full prediction pipeline for a frame of observed feature rows."""
    rf_pred, xgb_pred, model_pred = raw_ensemble(rf, xgb, frame, features, config)
    previous = frame["PM25_Lag_1"].to_numpy(dtype=float)

    out = frame[["Date", TARGET]].copy()
    out["RF"] = apply_anchor(rf_pred, previous, config)
    out["XGB"] = apply_anchor(xgb_pred, previous, config)
    out["Ensemble"] = apply_anchor(model_pred, previous, config)
    out["Model_Only"] = np.clip(model_pred, PM25_VALID_MIN, PM25_VALID_MAX)
    out["Residual"] = out[TARGET] - out["Ensemble"]
    out["Actual_AQI_Class"] = aqi_category(out[TARGET])
    out["Predicted_AQI_Class"] = aqi_category(out["Ensemble"])
    return out


def aqi_category(values) -> np.ndarray:
    """Map PM2.5 concentrations to CPCB National AQI bands (Section 5)."""
    return AQI_LABELS[np.digitize(np.asarray(values, dtype=float), AQI_BOUNDS)]


def evaluate(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE, R2 and AQI-band accuracy (Section 5, Evaluation Metrics)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "AQI_Accuracy": float(accuracy_score(aqi_category(y_true), aqi_category(y_pred))),
        "Mean_Bias": float(np.mean(y_pred - y_true)),
        "Observed_SD": float(np.std(y_true, ddof=0)),
    }


def run_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    cpcb_daily: pd.DataFrame,
    config: dict,
) -> dict[str, dict[str, float]]:
    """Single-source and naive baselines used to answer RQ1.

    RQ1 asks whether multi-source fusion improves on single-source baselines,
    which requires those baselines on exactly the same test days.
    """
    results: dict[str, dict[str, float]] = {}
    y_true = test[TARGET].values
    calendar = {"day_sin", "day_cos", "month_sin", "month_cos"}
    history = {f for f in features if f.startswith("PM25_")}

    ablations = {
        "Satellite_AOD_only": [f for f in features if f == "AOD" or f in calendar],
        "Meteorology_only": [f for f in features if f not in history and f != "AOD"],
        "Ground_history_only": [f for f in features if f in history or f in calendar],
        "No_satellite_AOD": [f for f in features if f != "AOD"],
    }

    for name, subset in ablations.items():
        if not subset:
            continue
        rf, xgb = fit_ensemble(train, subset, config)
        results[name] = evaluate(
            y_true, ensemble_predict(rf, xgb, test, subset, config)["Ensemble"].values
        )

    # Persistence: tomorrow equals today, the standard naive benchmark.
    results["Persistence_baseline"] = evaluate(y_true, test["PM25_Lag_1"].values)

    # Day-of-year climatology built only from years preceding the test period.
    past = cpcb_daily[cpcb_daily["Date"] < test["Date"].min()].copy()
    past["doy"] = past["Date"].dt.day_of_year
    climatology = past.groupby("doy")[TARGET].mean()
    clim_pred = test["Date"].dt.day_of_year.map(climatology).astype(float)
    if clim_pred.notna().all():
        results["Climatology_baseline"] = evaluate(y_true, clim_pred.values)

    return results


def weighted_feature_importance(
    rf: RandomForestRegressor, xgb: XGBRegressor, features: list[str]
) -> pd.DataFrame:
    """Normalised ensemble feature importance (Section 6.4, Figure 6)."""
    rf_imp = np.asarray(rf.feature_importances_, dtype=float)
    xgb_imp = np.asarray(xgb.feature_importances_, dtype=float)
    rf_imp = rf_imp / rf_imp.sum() if rf_imp.sum() else rf_imp
    xgb_imp = xgb_imp / xgb_imp.sum() if xgb_imp.sum() else xgb_imp
    ensemble = ENSEMBLE_WEIGHT_RF * rf_imp + ENSEMBLE_WEIGHT_XGB * xgb_imp

    return (
        pd.DataFrame(
            {
                "feature": features,
                "importance": ensemble,
                "importance_percent": 100.0 * ensemble / ensemble.sum(),
                "rf_importance": rf_imp,
                "xgb_importance": xgb_imp,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def select_configuration(
    model_df: pd.DataFrame, test_start: str
) -> pd.DataFrame:
    """Search model configurations on a validation window before the test period.

    This is the search that produced the shipped ``tuned`` configuration. It
    never touches the test window: candidates are fitted on data before
    VALIDATION_START and scored on the validation window only.
    """
    v_start, v_end = pd.Timestamp(VALIDATION_START), pd.Timestamp(VALIDATION_END)
    if pd.Timestamp(test_start) <= v_end:
        raise ValueError("Validation window must end before the test window begins")

    rows = []
    grid = itertools.product(
        [[], ["PM25_Roll3", "PM25_Roll7"]],       # extra features
        ["none", "log1p"],                        # target transform
        [None, 60, 90],                           # recency half-life
        [0.0, 0.2, 0.4, 0.6],                     # persistence anchor
    )
    for extra, transform, halflife, alpha in grid:
        config = {
            "target_transform": transform,
            "recency_halflife": halflife,
            "persistence_anchor": alpha,
            "extra_features": extra,
        }
        features, _ = select_available_features(model_df, config)
        frame = model_df.dropna(subset=features + [TARGET])

        core = frame[frame["Date"] < v_start]
        validation = frame[(frame["Date"] >= v_start) & (frame["Date"] <= v_end)]
        if len(core) < 30 or len(validation) < 10:
            continue

        rf, xgb = fit_ensemble(core, features, config)
        pred = ensemble_predict(rf, xgb, validation, features, config)
        scores = evaluate(pred[TARGET], pred["Ensemble"])
        rows.append(
            {
                "rolling_features": bool(extra),
                "target_transform": transform,
                "recency_halflife": halflife,
                "persistence_anchor": alpha,
                "n_validation": len(validation),
                "validation_MAE": scores["MAE"],
                "validation_RMSE": scores["RMSE"],
                "validation_R2": scores["R2"],
            }
        )

    return pd.DataFrame(rows).sort_values("validation_MAE").reset_index(drop=True)


# --------------------------------------------------------------------------
# Layer 4: Forecasting
# --------------------------------------------------------------------------


def _update_recursive_features(row: pd.Series, history: list[float]) -> pd.Series:
    """Rewrite lag and rolling features from a running prediction history.

    ``history`` holds the most recent values first-in-last-out, so history[-1]
    is the previous day. During a forecast those values are predictions, which
    is what makes the forecast recursive rather than a re-estimation.
    """
    row = row.copy()
    row["PM25_Lag_1"] = history[-1]
    row["PM25_Lag_2"] = history[-2]
    row["PM25_Lag_3"] = history[-3]
    if "PM25_Roll3" in row.index:
        row["PM25_Roll3"] = float(np.mean(history[-3:]))
    if "PM25_Roll7" in row.index:
        row["PM25_Roll7"] = float(np.mean(history[-7:]))
    return row


def recursive_forecast(
    rf: RandomForestRegressor,
    xgb: XGBRegressor,
    frame: pd.DataFrame,
    origin_date: pd.Timestamp,
    features: list[str],
    config: dict,
    horizon: int = 5,
) -> pd.DataFrame:
    """Five-day recursive forecast over days that already have observations.

    Used for evaluation (Section 6.5): the forecast is issued inside the
    held-out window, so all forecast days are days the model has never seen,
    and observed values are available for scoring. Meteorological and calendar
    inputs are taken as given, matching the chapter's assumption that forward
    meteorology is supplied by an external forecast product.
    """
    source = frame.set_index("Date").sort_index()
    observed = source.loc[:origin_date, TARGET].dropna().tolist()
    if len(observed) < 7:
        raise ValueError("At least seven observed days are required before the origin")

    history = list(observed[-7:])
    rows = []
    for step in range(1, horizon + 1):
        forecast_date = origin_date + pd.Timedelta(days=step)
        if forecast_date not in source.index:
            break

        row = _update_recursive_features(source.loc[forecast_date], history)
        x = row[features].to_frame().T.astype(float)
        _, _, model_pred = raw_ensemble(rf, xgb, x, features, config)
        prediction = float(apply_anchor(model_pred, [history[-1]], config)[0])

        actual = float(row[TARGET]) if pd.notna(row[TARGET]) else np.nan
        rows.append(
            {
                "Date": forecast_date,
                "horizon_day": step,
                "actual_PM25": actual,
                "predicted_PM25": prediction,
                "absolute_error": abs(prediction - actual) if np.isfinite(actual) else np.nan,
                "predicted_AQI_class": aqi_category([prediction])[0],
            }
        )
        history = (history + [prediction])[-7:]

    return pd.DataFrame(rows)


def forward_forecast(
    rf: RandomForestRegressor,
    xgb: XGBRegressor,
    frame: pd.DataFrame,
    features: list[str],
    config: dict,
    horizon: int = 5,
) -> pd.DataFrame:
    """Operational forecast for days beyond the last observation.

    This is the mode an operational deployment would run. No observed PM2.5
    exists for these days, so every lag and rolling input is a previous
    prediction. Meteorological and satellite inputs are persisted from the most
    recent observed day, which is a placeholder for the NWP forecast fields
    (NCMRWF, ECMWF-IFS or GFS) that Sections 8 and 9 identify as the
    requirement for genuine prospective forecasting. Calendar features are
    computed exactly, since they are known in advance.

    The persisted-meteorology assumption is recorded in the output so that the
    forecast is never mistaken for one driven by a real weather forecast.
    """
    ordered = frame.sort_values("Date")
    last = ordered.iloc[-1]
    last_date = last["Date"]
    history = ordered[TARGET].dropna().tolist()[-7:]
    if len(history) < 7:
        raise ValueError("At least seven observed days are required to forecast forward")

    persisted = {
        f: float(last[f])
        for f in features
        if f not in RECURSIVE_FEATURES and not f.startswith(("day_", "month_"))
    }

    rows = []
    for step in range(1, horizon + 1):
        forecast_date = last_date + pd.Timedelta(days=step)

        row = pd.Series(dtype=float)
        for name, value in persisted.items():
            row[name] = value
        day_of_week = forecast_date.day_of_week
        row["day_sin"] = math.sin(2 * math.pi * day_of_week / 7.0)
        row["day_cos"] = math.cos(2 * math.pi * day_of_week / 7.0)
        row["month_sin"] = math.sin(2 * math.pi * forecast_date.month / 12.0)
        row["month_cos"] = math.cos(2 * math.pi * forecast_date.month / 12.0)
        for name in RECURSIVE_FEATURES:
            if name in features:
                row[name] = np.nan
        row = _update_recursive_features(row, history)

        x = row[features].to_frame().T.astype(float)
        _, _, model_pred = raw_ensemble(rf, xgb, x, features, config)
        prediction = float(apply_anchor(model_pred, [history[-1]], config)[0])

        rows.append(
            {
                "Date": forecast_date,
                "horizon_day": step,
                "predicted_PM25": prediction,
                "predicted_AQI_class": aqi_category([prediction])[0],
                "meteorology_source": f"persisted from {last_date.date()}",
            }
        )
        history = (history + [prediction])[-7:]

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures (Figures 3-7 of the chapter)
# --------------------------------------------------------------------------


def make_figures(
    cpcb_daily: pd.DataFrame,
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    forecast: pd.DataFrame,
    figures_dir: Path,
) -> list[str]:
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 10})
    written = []

    # Figure 3: monthly PM2.5 climatology.
    monthly = (
        cpcb_daily.assign(month=cpcb_daily["Date"].dt.month)
        .groupby("month")[TARGET]
        .mean()
        .reindex(range(1, 13))
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(monthly.index, monthly.values, marker="o", color="#264653", linewidth=2)
    ax.fill_between(monthly.index, 0, monthly.values, color="#264653", alpha=0.12)
    ax.axhline(40, color="#c1121f", linestyle="--", linewidth=1,
               label="CPCB annual standard (40)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_xlabel("Month")
    ax.set_ylabel(r"Mean PM$_{2.5}$ ($\mu$g/m$^3$)")
    ax.set_title(f"Monthly mean PM$_{{2.5}}$ climatology, {STATION_NAME} (CPCB)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "fig3_monthly_pm25_climatology.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path.name)

    # Figure 4: observed versus predicted over the test period.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(predictions["Date"], predictions[TARGET], marker="o", markersize=4,
            color="#264653", label="Observed")
    ax.plot(predictions["Date"], predictions["Ensemble"], marker="s", markersize=4,
            color="#e76f51", label="Predicted (ensemble)")
    ax.set_ylabel(r"PM$_{2.5}$ ($\mu$g/m$^3$)")
    ax.set_title("Observed versus predicted daily PM$_{2.5}$, held-out test period")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = figures_dir / "fig4_observed_vs_predicted.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path.name)

    # Figure 5: residuals.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = np.where(predictions["Residual"] >= 0, "#2a9d8f", "#e76f51")
    ax.bar(predictions["Date"], predictions["Residual"], color=colors, width=0.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(predictions["Residual"].mean(), color="#023047", linestyle="--",
               linewidth=1,
               label=f"Mean residual = {predictions['Residual'].mean():.1f}")
    ax.set_ylabel(r"Observed - Predicted ($\mu$g/m$^3$)")
    ax.set_title("Residuals over the test period")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, axis="y")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = figures_dir / "fig5_residuals.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path.name)

    # Figure 6: feature importance.
    top = importance.head(18).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(top["feature"], top["importance"], color="#2a9d8f")
    ax.set_xlabel("Normalised ensemble importance")
    ax.set_title("Weighted ensemble feature importance")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    path = figures_dir / "fig6_feature_importance.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path.name)

    # Figure 7: five-day recursive forecast.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(forecast["horizon_day"], forecast["predicted_PM25"], marker="s",
            color="#e76f51", linewidth=2, label="Recursive forecast")
    if forecast["actual_PM25"].notna().any():
        ax.plot(forecast["horizon_day"], forecast["actual_PM25"], marker="o",
                color="#264653", linewidth=2, label="Observed")
    ax.set_xticks(forecast["horizon_day"])
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_ylabel(r"PM$_{2.5}$ ($\mu$g/m$^3$)")
    ax.set_title("Five-day recursive forecast versus observations")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "fig7_five_day_forecast.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path.name)

    return written


# --------------------------------------------------------------------------
# Chapter tables
# --------------------------------------------------------------------------


def write_chapter_tables(
    path: Path,
    config_name: str,
    config: dict,
    coverage: dict,
    split: SplitSummary,
    metrics: dict,
    baselines: dict,
    importance: pd.DataFrame,
    forecast: pd.DataFrame,
    forward: pd.DataFrame,
    features_used: list[str],
    features_unavailable: list[str],
) -> None:
    """Emit Tables 1-4 of the chapter, populated from this run."""
    lines: list[str] = []
    add = lines.append

    add("# AeroSphinx: chapter tables generated from the local data")
    add("")
    add(f"Generated {datetime.now().isoformat(timespec='seconds')} "
        f"(configuration `{config_name}`, random_state={RANDOM_STATE}).")
    add("")
    add(f"{config['description']}")
    add("")

    add("## Table 1 Principal data sources")
    add("")
    add("| Source | Product | Variable | Native resolution | Coverage used | Rows |")
    add("|---|---|---|---|---|---|")
    add(f"| ISRO / SAC INSAT-3D | Imager L2G AOD V02R00 | AOD at 550 nm | "
        f"~4 km, per overpass | {coverage['aod_start']} to {coverage['aod_end']} | "
        f"{coverage['aod_daily_rows']} |")
    add(f"| CPCB, India | National Air Quality Monitoring Programme | PM2.5 | "
        f"hourly, station | {coverage['cpcb_start']} to {coverage['cpcb_end']} | "
        f"{coverage['cpcb_daily_rows']} |")
    add(f"| NASA GMAO | MERRA-2 tavg1_2d_slv_Nx | "
        f"{', '.join(v for v in MERRA_REQUESTED if v not in features_unavailable)} | "
        f"0.5 x 0.625 deg, hourly | {coverage['merra_start']} to {coverage['merra_end']} | "
        f"{coverage['merra_daily_rows']} |")
    add("")
    if features_unavailable:
        add(f"Requested but unavailable in the local granules: "
            f"{', '.join(features_unavailable)}.")
        add("")

    add("## Table 2 Engineered features used in the ensemble model")
    add("")
    add("| Feature | Definition | Physical motivation |")
    add("|---|---|---|")
    for feature in features_used:
        definition, motivation = FEATURE_DESCRIPTIONS[feature]
        add(f"| `{feature}` | {definition} | {motivation} |")
    add("")

    add("## Table 3 Performance on the held-out test period")
    add("")
    add(f"Test period {split.test_start} to {split.test_end}, n = {split.n_test} days; "
        f"training {split.train_start} to {split.train_end}, n = {split.n_train} days.")
    add("")
    add("| Model | MAE | RMSE | R2 | AQI-band accuracy | Mean bias |")
    add("|---|---|---|---|---|---|")
    for name, values in metrics.items():
        add(f"| {name} | {values['MAE']:.2f} | {values['RMSE']:.2f} | "
            f"{values['R2']:.3f} | {values['AQI_Accuracy']:.3f} | "
            f"{values['Mean_Bias']:+.2f} |")
    add("")
    add("### Table 3b Single-source and naive baselines (RQ1)")
    add("")
    add("| Baseline | MAE | RMSE | R2 | AQI-band accuracy | Mean bias |")
    add("|---|---|---|---|---|---|")
    for name, values in baselines.items():
        add(f"| {name.replace('_', ' ')} | {values['MAE']:.2f} | {values['RMSE']:.2f} | "
            f"{values['R2']:.3f} | {values['AQI_Accuracy']:.3f} | "
            f"{values['Mean_Bias']:+.2f} |")
    add("")
    observed_sd = list(metrics.values())[0]["Observed_SD"]
    add(f"Observed standard deviation of PM2.5 over the test days: "
        f"{observed_sd:.2f} ug/m3. R2 is measured against this spread, so an R2 "
        f"of 0 corresponds to an RMSE of {observed_sd:.2f} ug/m3.")
    add("")

    add("## Table 4 Five-day recursive forecast, evaluated in the held-out window")
    add("")
    add("| Horizon (day) | Date | Observed | Predicted | Absolute error |")
    add("|---|---|---|---|---|")
    for _, row in forecast.iterrows():
        observed = f"{row['actual_PM25']:.2f}" if np.isfinite(row["actual_PM25"]) else "n/a"
        error = f"{row['absolute_error']:.2f}" if np.isfinite(row["absolute_error"]) else "n/a"
        add(f"| {int(row['horizon_day'])} | {row['Date'].date()} | {observed} | "
            f"{row['predicted_PM25']:.2f} | {error} |")
    add("")
    errors = forecast["absolute_error"].dropna()
    if len(errors):
        add(f"Mean absolute forecast error across the horizon: {errors.mean():.2f} "
            f"ug/m3 (range {errors.min():.2f} to {errors.max():.2f}).")
        add("")

    if not forward.empty:
        add("### Table 4b Operational forward forecast beyond the last observation")
        add("")
        add("| Horizon (day) | Date | Predicted PM2.5 | CPCB band |")
        add("|---|---|---|---|")
        for _, row in forward.iterrows():
            add(f"| {int(row['horizon_day'])} | {row['Date'].date()} | "
                f"{row['predicted_PM25']:.2f} | {row['predicted_AQI_class']} |")
        add("")
        add(f"Meteorological inputs {forward.iloc[0]['meteorology_source']}; "
            "an operational deployment would substitute NWP forecast fields.")
        add("")

    add("## Feature importance ranking (Figure 6)")
    add("")
    add("| Rank | Feature | Normalised importance | Share of total |")
    add("|---|---|---|---|")
    for rank, row in enumerate(importance.itertuples(index=False), start=1):
        add(f"| {rank} | `{row.feature}` | {row.importance:.5f} | "
            f"{row.importance_percent:.1f}% |")
    add("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Terminal results
# --------------------------------------------------------------------------


def print_results(split, metrics, baselines, importance, forecast, forward):
    """Print all key results to the terminal."""
    cols = ["MAE", "RMSE", "R2", "AQI_Accuracy", "Mean_Bias"]

    def show(title, df):
        print(f"\n{title}")
        print(df.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 74)
    print("RESULTS")
    print("=" * 74)
    print(f"Train: {split.train_start} to {split.train_end} (n={split.n_train})")
    print(f"Test : {split.test_start} to {split.test_end} (n={split.n_test})")

    show("Model performance (held-out test period)",
         pd.DataFrame(metrics).T[cols])
    show("Baselines (RQ1: single-source and naive)",
         pd.DataFrame(baselines).T[cols])

    sd = next(iter(metrics.values()))["Observed_SD"]
    print(f"\nObserved SD of PM2.5 in test window: {sd:.2f} ug/m3 "
          f"(R2 = 0 corresponds to RMSE = {sd:.2f})")

    show("Top 10 feature importances (%)",
         importance.head(10).set_index("feature")[["importance_percent"]])

    f = forecast.assign(Date=forecast["Date"].dt.date).set_index("horizon_day")
    show("5-day recursive forecast (evaluated in test window)",
         f[["Date", "actual_PM25", "predicted_PM25",
            "absolute_error", "predicted_AQI_class"]])

    if not forward.empty:
        w = forward.assign(Date=forward["Date"].dt.date).set_index("horizon_day")
        show("Operational forward forecast (weather persisted, placeholder)",
             w[["Date", "predicted_PM25", "predicted_AQI_class"]])
    print("=" * 74)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args(root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the AeroSphinx PM2.5 estimation and forecasting results."
    )
    parser.add_argument("--cpcb-dir", type=Path, default=root / "data" / "cpcb")
    parser.add_argument("--aod-dir", type=Path, action="append", default=None)
    parser.add_argument("--merra-dir", type=Path, default=root / "data" / "merra2")
    parser.add_argument(
        "--results-dir", type=Path, default=root / "results" / "legacy_chapter"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=root / "docs" / "figures"
    )
    parser.add_argument("--models-dir", type=Path, default=root / "models")
    parser.add_argument("--config", choices=sorted(CONFIGURATIONS), default=DEFAULT_CONFIG,
                        help="Model configuration to run")
    parser.add_argument("--select", action="store_true",
                        help="Re-run the validation configuration search and exit")
    parser.add_argument("--test-start", type=str, default="2024-05-17")
    parser.add_argument("--test-end", type=str, default="2024-06-18")
    parser.add_argument("--min-hours", type=int, default=MIN_HOURS_PER_DAY)
    parser.add_argument("--forecast-horizon", type=int, default=5,
                        help="Horizon of the in-window evaluated forecast")
    parser.add_argument("--forecast-days", type=int, default=5,
                        help="Horizon of the operational forward forecast")
    parser.add_argument("--forecast-origin", type=str, default=None,
                        help="Forecast issue date; defaults to the last five days "
                             "of the held-out test window")
    args = parser.parse_args()
    if not args.aod_dir:
        args.aod_dir = [root / "data" / "aod"]
    return args


def main() -> int:
    # This file lives in legacy/ but reads data/ and writes results/ from the
    # repository root, so the baseline keeps landing in the same place it did
    # before the 2026-11-23 restructure.
    root = Path(__file__).resolve().parents[1]
    args = parse_args(root)
    config = dict(CONFIGURATIONS[args.config])

    for directory in (args.results_dir, args.figures_dir, args.models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"AeroSphinx multi-source PM2.5 pipeline  [config: {args.config}]")
    print("=" * 74)

    # Layer 1: acquisition -------------------------------------------------
    cpcb_daily = load_cpcb_daily(args.cpcb_dir, min_hours=args.min_hours)
    aod_daily = load_aod_daily(args.aod_dir)
    merra_daily, merra_missing = load_merra_daily(args.merra_dir)
    print(f"[1/4] acquisition  CPCB {len(cpcb_daily)} days, "
          f"AOD {int(aod_daily['AOD'].notna().sum())} days, "
          f"MERRA-2 {len(merra_daily)} days")

    # Layer 2: harmonisation ----------------------------------------------
    combined = engineer_features(harmonise(cpcb_daily, aod_daily, merra_daily))
    features_used, features_unavailable = select_available_features(combined, config)
    model_df = (
        combined.dropna(subset=features_used + [TARGET])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    print(f"[2/4] harmonised   {len(model_df)} complete daily records "
          f"({model_df['Date'].min().date()} to {model_df['Date'].max().date()})")
    if features_unavailable:
        print(f"      unavailable features excluded: {', '.join(features_unavailable)}")

    # Optional: configuration search --------------------------------------
    if args.select:
        table = select_configuration(model_df, args.test_start)
        out = args.results_dir / "configuration_selection.csv"
        table.to_csv(out, index=False)
        print(f"\nValidation window {VALIDATION_START} to {VALIDATION_END}, "
              f"scored on {int(table['n_validation'].iloc[0])} days.")
        print(table.head(10).to_string(index=False))
        print(f"\nFull table written to {out}")
        return 0

    # Layer 3: modelling ---------------------------------------------------
    train, test, split = chronological_split(model_df, args.test_start, args.test_end)
    rf_model, xgb_model = fit_ensemble(train, features_used, config)
    predictions = ensemble_predict(rf_model, xgb_model, test, features_used, config)

    metrics = {
        "RandomForest": evaluate(predictions[TARGET], predictions["RF"]),
        "XGBoost": evaluate(predictions[TARGET], predictions["XGB"]),
        "WeightedEnsemble_0.6RF_0.4XGB": evaluate(
            predictions[TARGET], predictions["Ensemble"]
        ),
    }
    if config["persistence_anchor"] > 0:
        metrics["Ensemble_without_anchor"] = evaluate(
            predictions[TARGET], predictions["Model_Only"]
        )
    baselines = run_baselines(train, test, features_used, cpcb_daily, config)
    importance = weighted_feature_importance(rf_model, xgb_model, features_used)

    headline = metrics["WeightedEnsemble_0.6RF_0.4XGB"]
    print(f"[3/4] modelling    train n={split.n_train}, test n={split.n_test} "
          f"({split.test_start} to {split.test_end})")
    print(f"      ensemble     MAE={headline['MAE']:.2f}  RMSE={headline['RMSE']:.2f}  "
          f"R2={headline['R2']:.3f}  bias={headline['Mean_Bias']:+.2f}  "
          f"AQI={headline['AQI_Accuracy']:.3f}")

    # Layer 4: forecasting -------------------------------------------------
    if args.forecast_origin:
        origin = pd.Timestamp(args.forecast_origin)
    else:
        test_positions = model_df.index[model_df["Date"].isin(test["Date"])]
        origin = model_df["Date"].iloc[max(0, test_positions[-1] - args.forecast_horizon)]

    forecast = recursive_forecast(
        rf_model, xgb_model, model_df, origin, features_used, config, args.forecast_horizon
    )
    forward = forward_forecast(
        rf_model, xgb_model, model_df, features_used, config, args.forecast_days
    )

    errors = forecast["absolute_error"].dropna()
    if len(errors):
        print(f"[4/4] forecasting  evaluated 5-day error {errors.min():.1f} to "
              f"{errors.max():.1f} ug/m3 (mean {errors.mean():.1f})")
    if not forward.empty:
        first, last = forward.iloc[0], forward.iloc[-1]
        print(f"      forward      {first['Date'].date()} {first['predicted_PM25']:.1f} "
              f"-> {last['Date'].date()} {last['predicted_PM25']:.1f} ug/m3 "
              f"({last['predicted_AQI_class']})")

    # Full results in the terminal -----------------------------------------
    print_results(split, metrics, baselines, importance, forecast, forward)

    # Outputs --------------------------------------------------------------
    figures_written = make_figures(
        cpcb_daily, predictions, importance, forecast, args.figures_dir
    )

    combined.to_csv(args.results_dir / "daily_multisource_dataset.csv", index=False)
    model_df.to_csv(args.results_dir / "modelling_dataset_complete_cases.csv", index=False)
    predictions.to_csv(args.results_dir / "test_predictions.csv", index=False)
    forecast.to_csv(args.results_dir / "five_day_recursive_forecast.csv", index=False)
    forward.to_csv(args.results_dir / "operational_forward_forecast.csv", index=False)
    importance.to_csv(args.results_dir / "feature_importance.csv", index=False)

    joblib.dump(rf_model, args.models_dir / "random_forest_pm25.joblib")
    joblib.dump(xgb_model, args.models_dir / "xgboost_pm25.joblib")

    coverage = {
        "cpcb_daily_rows": int(len(cpcb_daily)),
        "cpcb_start": cpcb_daily["Date"].min().date().isoformat(),
        "cpcb_end": cpcb_daily["Date"].max().date().isoformat(),
        "aod_daily_rows": int(aod_daily["AOD"].notna().sum()),
        "aod_start": aod_daily["Date"].min().date().isoformat(),
        "aod_end": aod_daily["Date"].max().date().isoformat(),
        "merra_daily_rows": int(len(merra_daily)),
        "merra_start": merra_daily["Date"].min().date().isoformat(),
        "merra_end": merra_daily["Date"].max().date().isoformat(),
        "model_rows": int(len(model_df)),
        "model_start": model_df["Date"].min().date().isoformat(),
        "model_end": model_df["Date"].max().date().isoformat(),
    }

    data_notes = [
        "Model rows require CPCB, INSAT-3D AOD and MERRA-2 to be jointly "
        "available on the same day, after a three-day forward fill.",
        "INSAT-3D and MERRA-2 granules held locally cover 2024 only, so the "
        "multi-source model is trained and tested within 2024 even though the "
        "CPCB record extends back to 2016.",
        f"Model configuration '{args.config}' selected on the validation window "
        f"{VALIDATION_START} to {VALIDATION_END}, which ends before the test "
        "window begins; the test set influenced no modelling choice.",
        "The operational forward forecast persists meteorological and satellite "
        "inputs from the last observed day as a placeholder for NWP forecast "
        "fields; it is not driven by a real weather forecast.",
    ]
    if features_unavailable:
        data_notes.append(
            "Requested features absent from every local granule and therefore "
            f"excluded rather than imputed: {', '.join(features_unavailable)}."
        )
    for var, count in merra_missing.items():
        if 0 < count < len(merra_daily):
            data_notes.append(
                f"MERRA-2 variable {var} missing on {count} of {len(merra_daily)} "
                "days before complete-case filtering."
            )

    audit = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "configuration_name": args.config,
        "configuration": {k: v for k, v in config.items() if k != "description"},
        "random_state": RANDOM_STATE,
        "station": {"name": STATION_NAME, "lat": STATION_LAT, "lon": STATION_LON},
        "coverage": coverage,
        "features_used": features_used,
        "features_unavailable": features_unavailable,
        "hyperparameters": {
            "RandomForest": {k: v for k, v in RF_PARAMS.items() if k != "n_jobs"},
            "XGBoost": {k: v for k, v in XGB_PARAMS.items() if k != "n_jobs"},
        },
        "split": asdict(split),
        "validation_window": {"start": VALIDATION_START, "end": VALIDATION_END},
        "metrics": metrics,
        "baselines": baselines,
        "five_day_forecast": forecast.assign(Date=forecast["Date"].astype(str)).to_dict("records"),
        "forward_forecast": forward.assign(Date=forward["Date"].astype(str)).to_dict("records"),
        "data_notes": data_notes,
        "figures": figures_written,
    }
    (args.results_dir / "metrics.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    write_chapter_tables(
        args.results_dir / "chapter_tables.md",
        args.config, config, coverage, split, metrics, baselines,
        importance, forecast, forward, features_used, features_unavailable,
    )

    print("-" * 74)
    print(f"results  -> {args.results_dir}")
    print(f"figures  -> {args.figures_dir}")
    print(f"models   -> {args.models_dir}")
    print("Chapter tables written to results/chapter_tables.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())