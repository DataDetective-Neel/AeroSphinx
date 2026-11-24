# AeroSphinx: chapter tables generated from the local data

Generated 2026-09-01T04:06:20 (configuration `tuned`, random_state=42).

Validation-selected configuration: log1p target, 0.4 persistence anchor, three- and seven-day rolling-mean features.

## Table 1 Principal data sources

| Source | Product | Variable | Native resolution | Coverage used | Rows |
|---|---|---|---|---|---|
| ISRO / SAC INSAT-3D | Imager L2G AOD V02R00 | AOD at 550 nm | ~4 km, per overpass | 2024-01-01 to 2024-06-18 | 139 |
| CPCB, India | National Air Quality Monitoring Programme | PM2.5 | hourly, station | 2016-01-01 to 2024-12-31 | 2979 |
| NASA GMAO | MERRA-2 tavg1_2d_slv_Nx | T2M, TS, QV2M, U10M, V10M, SLP | 0.5 x 0.625 deg, hourly | 2024-01-01 to 2024-06-20 | 172 |

Requested but unavailable in the local granules: PBLH.

## Table 2 Engineered features used in the ensemble model

| Feature | Definition | Physical motivation |
|---|---|---|
| `AOD` | INSAT-3D aerosol optical depth at 550 nm, station window mean | Column aerosol loading, the primary satellite predictor |
| `T2M` | MERRA-2 2 m air temperature (K) | Thermal state controlling vertical mixing |
| `TS` | MERRA-2 surface skin temperature (K) | Surface heating driving convective dispersion |
| `QV2M` | MERRA-2 2 m specific humidity (kg/kg) | Hygroscopic growth of aerosol particles |
| `U10M` | MERRA-2 10 m zonal wind (m/s) | East-west advection of pollutants |
| `V10M` | MERRA-2 10 m meridional wind (m/s) | North-south advection of pollutants |
| `SLP` | MERRA-2 sea-level pressure (Pa) | Synoptic regime and stagnation indicator |
| `Temp_Diff` | TS - T2M (K) | Near-surface stability proxy for inversion strength |
| `Humidity_Ratio` | QV2M / T2M | Combined moisture-temperature aerosol growth term |
| `PM25_Lag_1` | PM2.5 on day t-1 (ug/m3) | Day-to-day persistence of pollution episodes |
| `PM25_Lag_2` | PM2.5 on day t-2 (ug/m3) | Two-day pollution memory |
| `PM25_Lag_3` | PM2.5 on day t-3 (ug/m3) | Three-day pollution memory |
| `day_sin` | sin(2*pi*d/7), d = day of week | Cyclic weekly encoding of sub-weekly emission cycles |
| `day_cos` | cos(2*pi*d/7), d = day of week | Cyclic weekly encoding, quadrature component |
| `month_sin` | sin(2*pi*m/12), m = month | Cyclic seasonal encoding of the annual pollution cycle |
| `month_cos` | cos(2*pi*m/12), m = month | Cyclic seasonal encoding, quadrature component |
| `PM25_Roll3` | Mean PM2.5 over days t-1 to t-3 (ug/m3) | Short-term pollution level, less noisy than a single lag |
| `PM25_Roll7` | Mean PM2.5 over days t-1 to t-7 (ug/m3) | Weekly background level of the current episode |

## Table 3 Performance on the held-out test period

Test period 2024-05-17 to 2024-06-18, n = 33 days; training 2024-01-07 to 2024-05-16, n = 130 days.

| Model | MAE | RMSE | R2 | AQI-band accuracy | Mean bias |
|---|---|---|---|---|---|
| RandomForest | 17.80 | 23.35 | -0.022 | 0.545 | +6.45 |
| XGBoost | 16.85 | 22.17 | 0.079 | 0.485 | +3.91 |
| WeightedEnsemble_0.6RF_0.4XGB | 17.24 | 22.81 | 0.025 | 0.515 | +5.41 |
| Ensemble_without_anchor | 19.89 | 26.05 | -0.271 | 0.545 | +9.02 |

### Table 3b Single-source and naive baselines (RQ1)

| Baseline | MAE | RMSE | R2 | AQI-band accuracy | Mean bias |
|---|---|---|---|---|---|
| Satellite AOD only | 19.10 | 24.23 | -0.100 | 0.636 | +12.60 |
| Meteorology only | 20.40 | 27.05 | -0.371 | 0.545 | +8.42 |
| Ground history only | 17.10 | 23.78 | -0.060 | 0.636 | +7.27 |
| No satellite AOD | 17.13 | 22.78 | 0.028 | 0.576 | +5.88 |
| Persistence baseline | 19.92 | 24.19 | -0.096 | 0.424 | +0.00 |
| Climatology baseline | 35.87 | 44.32 | -2.681 | 0.242 | -35.85 |

Observed standard deviation of PM2.5 over the test days: 23.10 ug/m3. R2 is measured against this spread, so an R2 of 0 corresponds to an RMSE of 23.10 ug/m3.

## Table 4 Five-day recursive forecast, evaluated in the held-out window

| Horizon (day) | Date | Observed | Predicted | Absolute error |
|---|---|---|---|---|
| 1 | 2024-06-14 | 92.25 | 104.12 | 11.88 |
| 2 | 2024-06-15 | 79.07 | 104.51 | 25.45 |
| 3 | 2024-06-16 | 93.43 | 104.12 | 10.69 |
| 4 | 2024-06-17 | 78.75 | 103.78 | 25.04 |
| 5 | 2024-06-18 | 93.93 | 106.93 | 13.00 |

Mean absolute forecast error across the horizon: 17.21 ug/m3 (range 10.69 to 25.45).

### Table 4b Operational forward forecast beyond the last observation

| Horizon (day) | Date | Predicted PM2.5 | CPCB band |
|---|---|---|---|
| 1 | 2024-06-22 | 97.05 | Poor |
| 2 | 2024-06-23 | 111.00 | Poor |
| 3 | 2024-06-24 | 119.41 | Poor |
| 4 | 2024-06-25 | 126.84 | Very Poor |
| 5 | 2024-06-26 | 134.61 | Very Poor |

Meteorological inputs persisted from 2024-06-21; an operational deployment would substitute NWP forecast fields.

## Feature importance ranking (Figure 6)

| Rank | Feature | Normalised importance | Share of total |
|---|---|---|---|
| 1 | `month_sin` | 0.39529 | 39.5% |
| 2 | `PM25_Roll3` | 0.11331 | 11.3% |
| 3 | `PM25_Lag_1` | 0.06028 | 6.0% |
| 4 | `PM25_Roll7` | 0.05680 | 5.7% |
| 5 | `V10M` | 0.04715 | 4.7% |
| 6 | `U10M` | 0.04234 | 4.2% |
| 7 | `Temp_Diff` | 0.04091 | 4.1% |
| 8 | `month_cos` | 0.03964 | 4.0% |
| 9 | `T2M` | 0.03290 | 3.3% |
| 10 | `TS` | 0.02782 | 2.8% |
| 11 | `SLP` | 0.02281 | 2.3% |
| 12 | `PM25_Lag_3` | 0.02072 | 2.1% |
| 13 | `PM25_Lag_2` | 0.02006 | 2.0% |
| 14 | `AOD` | 0.01934 | 1.9% |
| 15 | `Humidity_Ratio` | 0.01764 | 1.8% |
| 16 | `QV2M` | 0.01633 | 1.6% |
| 17 | `day_sin` | 0.01389 | 1.4% |
| 18 | `day_cos` | 0.01278 | 1.3% |

