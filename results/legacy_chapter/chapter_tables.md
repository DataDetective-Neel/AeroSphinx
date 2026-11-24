# AeroSphinx: chapter tables generated from the local data

Generated 2026-10-26T12:31:31 (configuration `tuned`, random_state=42).

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
| RandomForest | 17.82 | 23.37 | -0.024 | 0.545 | +6.46 |
| XGBoost | 16.94 | 22.42 | 0.058 | 0.485 | +3.74 |
| WeightedEnsemble_0.6RF_0.4XGB | 17.30 | 22.90 | 0.017 | 0.545 | +5.34 |
| Ensemble_without_anchor | 20.12 | 26.18 | -0.284 | 0.515 | +8.91 |

### Table 3b Single-source and naive baselines (RQ1)

| Baseline | MAE | RMSE | R2 | AQI-band accuracy | Mean bias |
|---|---|---|---|---|---|
| Satellite AOD only | 19.26 | 24.36 | -0.112 | 0.636 | +12.39 |
| Meteorology only | 20.12 | 26.81 | -0.347 | 0.545 | +8.22 |
| Ground history only | 16.62 | 23.67 | -0.050 | 0.636 | +7.15 |
| No satellite AOD | 17.20 | 22.98 | 0.011 | 0.576 | +5.66 |
| Persistence baseline | 19.92 | 24.19 | -0.096 | 0.424 | +0.00 |
| Climatology baseline | 35.87 | 44.32 | -2.681 | 0.242 | -35.85 |

Observed standard deviation of PM2.5 over the test days: 23.10 ug/m3. R2 is measured against this spread, so an R2 of 0 corresponds to an RMSE of 23.10 ug/m3.

## Table 4 Five-day recursive forecast, evaluated in the held-out window

| Horizon (day) | Date | Observed | Predicted | Absolute error |
|---|---|---|---|---|
| 1 | 2024-06-14 | 92.25 | 104.46 | 12.22 |
| 2 | 2024-06-15 | 79.07 | 103.84 | 24.78 |
| 3 | 2024-06-16 | 93.43 | 104.60 | 11.17 |
| 4 | 2024-06-17 | 78.75 | 104.30 | 25.55 |
| 5 | 2024-06-18 | 93.93 | 107.08 | 13.15 |

Mean absolute forecast error across the horizon: 17.37 ug/m3 (range 11.17 to 25.55).

### Table 4b Operational forward forecast beyond the last observation

| Horizon (day) | Date | Predicted PM2.5 | CPCB band |
|---|---|---|---|
| 1 | 2024-06-22 | 96.60 | Poor |
| 2 | 2024-06-23 | 112.26 | Poor |
| 3 | 2024-06-24 | 122.18 | Very Poor |
| 4 | 2024-06-25 | 129.64 | Very Poor |
| 5 | 2024-06-26 | 138.51 | Very Poor |

Meteorological inputs persisted from 2024-06-21; an operational deployment would substitute NWP forecast fields.

## Feature importance ranking (Figure 6)

| Rank | Feature | Normalised importance | Share of total |
|---|---|---|---|
| 1 | `month_sin` | 0.37934 | 37.9% |
| 2 | `PM25_Roll3` | 0.11222 | 11.2% |
| 3 | `month_cos` | 0.06287 | 6.3% |
| 4 | `PM25_Lag_1` | 0.06187 | 6.2% |
| 5 | `PM25_Roll7` | 0.05850 | 5.9% |
| 6 | `V10M` | 0.04517 | 4.5% |
| 7 | `U10M` | 0.04291 | 4.3% |
| 8 | `Temp_Diff` | 0.04111 | 4.1% |
| 9 | `T2M` | 0.03280 | 3.3% |
| 10 | `TS` | 0.02635 | 2.6% |
| 11 | `SLP` | 0.02212 | 2.2% |
| 12 | `AOD` | 0.01925 | 1.9% |
| 13 | `PM25_Lag_2` | 0.01887 | 1.9% |
| 14 | `PM25_Lag_3` | 0.01847 | 1.8% |
| 15 | `Humidity_Ratio` | 0.01627 | 1.6% |
| 16 | `QV2M` | 0.01587 | 1.6% |
| 17 | `day_sin` | 0.01329 | 1.3% |
| 18 | `day_cos` | 0.01271 | 1.3% |

