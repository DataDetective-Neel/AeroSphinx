# AeroSphinx — reproducibility package

Code and data accompanying the book chapter

> **A Multi-Source Machine Learning Framework for Satellite-Fused PM2.5 Estimation
> and Five-Day Atmospheric Forecasting over Urban India**
> Raju Bhakta and Indraneel Chatterjee
> Mathematics and Basic Sciences, NIIT University, NH-8, Delhi–Jaipur Highway,
> Alwar, Neemrana, Rajasthan-301705, India

Everything the chapter reports — Tables 1–4, Figures 3–7 and every metric in
Section 6 — is produced by a single script from the raw files in `data/`.
No number is hard-coded, and no value is copied from an earlier run.

---

## Quick start

```bash
python -m pip install -r requirements.txt
python aerosphinx_pipeline.py
```

Runtime is roughly two to four minutes on a standard laptop CPU. No GPU and no
network access are required; all input files are local.

Optional arguments:

```bash
python aerosphinx_pipeline.py --config chapter   # literal Section 3 formulation
python aerosphinx_pipeline.py --select           # validation configuration search
python aerosphinx_pipeline.py --forecast-days 10 # longer operational forecast
python aerosphinx_pipeline.py \
    --cpcb-dir data/cpcb --aod-dir data/aod --merra-dir data/merra2 \
    --test-start 2024-05-17 --test-end 2024-06-18 --min-hours 18
```

## Neural network comparison

```bash
python compare_neural_network.py
```

This CPU-only experiment uses the existing `results/daily_multisource_dataset.csv`
(regenerate it with the main pipeline if needed) and writes
`results/neural_network_comparison_10_layers.json`. It refits the tuned RF/XGBoost
ensemble and compares it with a multilayer perceptron with 10 hidden layers.
Both use the same feature columns, log target, persistence adjustment, and
chronological test dates. Feature scaling is fitted on training rows only.

Run `python compare_neural_network.py --depth 3-5` to repeat the earlier
experiment. That run selects a depth by daily validation MAE, then refits
and tests the selected model. Its output is `results/neural_network_comparison.json`.
Both reports record the installed library versions.

The report separates daily estimates (observed PM2.5 lags) from recursive
five-day forecasts (predicted PM2.5 lags after day one). Both models see the
same historical satellite and meteorological inputs for forecast dates, so
this evaluates recursion under known inputs. Forecast windows overlap; their
errors are descriptive. The main chapter pipeline continues to use the ensemble.

## Two reproducible configurations

| | `tuned` (default) | `chapter` |
|---|---|---|
| Target | `log1p(PM2.5)`, inverted for reporting | raw concentration |
| Features | base set + `PM25_Roll3`, `PM25_Roll7` | base set |
| Output | `0.6 · y_model + 0.4 · y_(t-1)` | `y_model` |
| Test MAE / RMSE / R² | **17.24 / 22.81 / +0.025** | 23.29 / 30.28 / −0.718 |

The `chapter` configuration is the literal Section 3 and Section 5 formulation.
The `tuned` configuration adds three documented choices, each selected on a
validation window (10 April to 16 May 2024) that **closes before the test window
opens**, so the test set influenced no modelling decision. `--select` regenerates
that search and writes the full table to `results/configuration_selection.csv`.

Model hyperparameters are fixed as published in Section 5 in both cases and are
never searched.

## Repository layout

```text
aerosphinx_pipeline.py     Single reproducible entry point (all four layers)
requirements.txt           Pinned minimum library versions
data/cpcb/                 CPCB hourly PM2.5 exports, Faridabad, 2016-2024
data/aod/                  INSAT-3D Imager L2G AOD V02R00 granules (HDF5)
data/merra2/               MERRA-2 tavg1_2d_slv_Nx granules (NetCDF)
results/                   Metrics, predictions, feature ranking, chapter tables
figures/                   Figures 3-7 at 300 dpi
models/                    Fitted Random Forest and XGBoost artifacts
```

## What the script does

The four layers of Section 4 map one-to-one onto functions in the script.

| Layer | Section 4 name | Functions |
|---|---|---|
| 1 | Data Acquisition | `load_cpcb_daily`, `load_aod_daily`, `load_merra_daily` |
| 2 | Harmonisation | `harmonise`, `engineer_features` |
| 3 | Modelling | `chronological_split`, `fit_ensemble`, `ensemble_predict`, `evaluate` |
| 4 | Forecasting | `recursive_forecast` |

Methodological choices, all matching the chapter text:

* PM2.5 values outside `[0, 500]` µg/m³ are treated as missing; a daily mean is
  kept only when at least 18 valid hourly samples are present.
* INSAT-3D AOD is collocated to the station by a ±0.25° window; fill values
  (−999) and impossible retrievals are removed before daily averaging.
* MERRA-2 fields are averaged over the Faridabad bounding box of each granule.
* Remaining gaps are forward-filled within a three-day window; anything still
  missing is excluded rather than imputed.
* Calendar variables use cyclic encodings, `day_sin = sin(2πd/7)` over the week
  and `month_sin = sin(2πm/12)` over the year, with matching cosine terms.
* Lags use only prior days, and the split is chronological, so no future
  information reaches the model.
* The ensemble is `0.6 · RF + 0.4 · XGB`, clipped to `[0, 500]` µg/m³.
* Hyperparameters are fixed as stated in Section 5 and are not searched.
* The five-day forecast is genuinely recursive: from day two onward the lag and
  rolling inputs are previous predictions, never observations.

## Forecasting, in two modes

`recursive_forecast()` — **evaluated mode**. The forecast is issued inside the
held-out window, so all five forecast days are days the model has never seen and
observations exist for scoring. Meteorological inputs are taken as given, which
matches the chapter's assumption that forward meteorology arrives from an
external forecast product. This is what Table 4 and Figure 7 report.

`forward_forecast()` — **operational mode**. The forecast runs past the last
observation, so no observed PM2.5 exists for any forecast day and every lag and
rolling input is a previous prediction. Meteorological and satellite inputs are
persisted from the most recent observed day, as a placeholder for the NWP fields
(NCMRWF, ECMWF-IFS, GFS) that Sections 8 and 9 identify as the requirement for
genuine prospective forecasting. Each row records `meteorology_source` so the
output is never mistaken for a real weather-forecast-driven product. Written to
`results/operational_forward_forecast.csv`; horizon set by `--forecast-days`.

Running both is deliberate: the difference between them isolates the cost of
recursive lag substitution from the cost of imperfect meteorology.

## Generated outputs

| File | Contents |
|---|---|
| `results/chapter_tables.md` | Tables 1, 2, 3, 3b and 4, ready to paste |
| `results/metrics.json` | Full machine-readable audit of the run |
| `results/test_predictions.csv` | Per-day observed, RF, XGB, ensemble, residual |
| `results/five_day_recursive_forecast.csv` | Evaluated forecast by horizon day |
| `results/operational_forward_forecast.csv` | Forward forecast beyond last observation |
| `results/configuration_selection.csv` | Validation search (`--select` only) |
| `results/feature_importance.csv` | Ensemble, RF and XGB importances |
| `results/daily_multisource_dataset.csv` | Full harmonised daily dataset |
| `results/modelling_dataset_complete_cases.csv` | Rows actually used for modelling |
| `figures/fig3_monthly_pm25_climatology.png` | Figure 3 |
| `figures/fig4_observed_vs_predicted.png` | Figure 4 |
| `figures/fig5_residuals.png` | Figure 5 |
| `figures/fig6_feature_importance.png` | Figure 6 |
| `figures/fig7_five_day_forecast.png` | Figure 7 |

## Data provenance and known constraints

| Stream | Provider | Local coverage |
|---|---|---|
| PM2.5 | CPCB National Air Quality Monitoring Programme | 2016-01-01 to 2024-12-31, hourly |
| AOD 550 nm | ISRO / SAC, INSAT-3D Imager L2G V02R00 | 2024-01-01 to 2024-06-18 |
| Meteorology | NASA GMAO, MERRA-2 `tavg1_2d_slv_Nx` | 2024-01-01 to 2024-06-20 |

Two constraints follow directly from these files and are reported by the script
rather than worked around:

1. **The multi-source model is a 2024 model.** A modelling row needs CPCB, AOD
   and MERRA-2 on the same day. The satellite and reanalysis granules held
   locally cover January to June 2024 only, so although the CPCB record reaches
   back to 2016, the fused dataset is confined to 2024. Extending the study to
   earlier years requires downloading the corresponding AOD and MERRA-2
   granules; the script picks them up automatically once they are in `data/`.

2. **PBLH is unavailable.** Planetary boundary layer height appears in the
   theoretical formulation of Section 3, but the subsetted MERRA-2 granules
   distributed with this study contain neither `PBLH` nor `PBLTOP`. The script
   detects this, drops the feature, and records it in `metrics.json` under
   `features_unavailable` instead of substituting a proxy. Re-subsetting
   MERRA-2 with `PBLH` included is the single highest-value data improvement
   available to this framework.

## Reproducibility

`random_state=42` throughout, fixed hyperparameters, no automated search, and a
chronological split, so repeated runs on the same files give identical numbers.
The exact configuration of each run is written to `results/metrics.json`.

## Acknowledgements

INSAT-3D Imager AOD products: Indian Space Research Organisation (ISRO), Space
Applications Centre (SAC), Ahmedabad. MERRA-2 reanalysis: NASA Global Modeling
and Assimilation Office (GMAO). Ground-level air-quality observations: Central
Pollution Control Board (CPCB), Government of India.
