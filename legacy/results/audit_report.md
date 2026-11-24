# AeroSphinx Scientific Audit Report

Generated: 2026-06-07T14:22:34

## Repository Audit
- Multiple duplicate model scripts were found at the repository root and under `ML model/`.
- Legacy scripts use random train/test splits and do not implement recursive five-day forecasting.
- `pawanai-project` contains synthetic/random data generation and is not suitable for publication metrics.
- Several CSVs and notebooks are exploratory or stale; the reproducible entry point is now `scripts/run_research_pipeline.py`.

## Scientific Issues Found
- Preferred 2016-2022/2023/2024 split is impossible for the strict multi-source dataset because local AOD and MERRA-2 files cover only a 2024 subset.
- Local MERRA-2 NetCDF files do not contain PBLH; the model excludes PBLH and reports it as unavailable rather than fabricating it.
- AOD granules include fill values (-999) and invalid negative values; these are filtered before daily aggregation.
- The resulting strict multi-source benchmark is small and should not be reported as a full urban-India generalization result.
- MERRA variables absent from all local files: PBLH.
- MERRA variables partially missing before complete-case filtering: V10M missing on 2/172 days.
- Unavailable requested model features removed from empirical training: PBLH.

## Code Issues Found
- Legacy scripts use random train_test_split(), causing time-series leakage risk.
- Legacy scripts hard-code prediction dates and manually inject AOD values.
- Several scripts reference missing files such as ../data/merged_pm_aod.csv and sample_insat_aod.nc.
- The old target column encoding appears corrupted in source CSV headers; the new pipeline detects the PM2.5 column by name pattern.
- Duplicate root and ML model scripts are byte-identical and should not both be maintained.
- Synthetic demo code in pawanai-project should be excluded from manuscript performance claims.

## Data Coverage
- CPCB daily rows: 3022 (2016-01-01 to 2024-12-31).
- AOD daily rows: 139 (2024-01-01 to 2024-06-18).
- MERRA daily rows: 172 (2024-01-01 to 2024-06-20).
- Complete model rows after leakage-safe lags and missing-data filtering: 139.

## Chronological Split
- Strategy: fallback_single_period_chronological_60_20_20
- Train: 2024-01-07 to 2024-04-13 (n=83)
- Validation: 2024-04-15 to 2024-05-17 (n=28)
- Test: 2024-05-18 to 2024-06-18 (n=28)

## Final Test Metrics
- RandomForest: MAE=38.721, RMSE=44.421, R2=-2.484, AQI accuracy=0.143
- XGBoost: MAE=33.017, RMSE=39.388, R2=-1.739, AQI accuracy=0.357
- WeightedEnsemble_0.6RF_0.4XGB: MAE=36.367, RMSE=42.213, R2=-2.146, AQI accuracy=0.214

## Feature Importance Ranking
- month_sin: 0.14591
- PM25_Lag_1: 0.13400
- PM25_Lag_2: 0.10076
- day_cos: 0.08825
- day_sin: 0.08088
- TS: 0.05653
- U10M: 0.05527
- Temp_Diff: 0.05273
- V10M: 0.05028
- PM25_Lag_3: 0.04143
- T2M: 0.04067
- QV2M: 0.03991
- SLP: 0.03240
- Humidity_Ratio: 0.03206
- month_cos: 0.02725
- AOD: 0.02168

## Recommended Springer Text
The final empirical pipeline aggregated CPCB hourly PM2.5 observations to daily means, extracted station-window INSAT AOD from local HDF5 granules, and averaged available MERRA-2 meteorological variables from local NetCDF files. Lagged PM2.5 predictors were computed strictly from prior days and the model was evaluated using a chronological split. Because the local satellite and MERRA-2 files only covered a 2024 subset and the MERRA-2 files did not include PBLH, the full 2016-2022/2023/2024 multi-source protocol could not be executed without additional historical AOD/MERRA-2 data and a valid boundary-layer-height product. Reported metrics should therefore be described as a limited strict multi-source benchmark over the available 2024 window, not as a full urban-India generalization result.

## Files Written
- `results/daily_multisource_dataset.csv`
- `results/modeling_dataset_complete_cases.csv`
- `results/test_predictions.csv`
- `results/five_day_recursive_forecast.csv`
- `results/feature_importance.csv`
- `results/metrics.json`
- `results/audit_report.md`
- `figures/actual_vs_predicted_pm25.png`
- `figures/residual_plot.png`
- `figures/feature_importance.png`
- `figures/monthly_pm25_seasonality.png`
- `figures/five_day_forecast_example.png`
- `models/random_forest_pm25.joblib`
- `models/xgboost_pm25.joblib`
