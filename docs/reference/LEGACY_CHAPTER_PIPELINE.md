# The legacy chapter pipeline

The single-file pipeline that produced the book chapter's tables, figures and
metrics. It is retained as the reproducibility baseline and is **not** the
current research code: every number it produces belongs to one 33-day test
window (2024-05-17 to 2024-06-18), which cannot characterise seasonal
behaviour. The Tracks in `src/aerosphinx/` replace it; see
`../reports/PHASE7_REPORT.md` for the comparison.

The script itself lives at `legacy/aerosphinx_pipeline.py`. Only its output
directory changed in the 2026-11-23 restructure — it now writes to
`results/legacy_chapter/` and `docs/figures/` — so the numbers are byte-for-byte
what the chapter reports.

## Running it

The single-script pipeline that produced the book chapter's tables, figures
and metrics is retained unchanged as the reproducibility baseline:

```bash
python -m pip install -r requirements.txt
python legacy/aerosphinx_pipeline.py
```

Optional arguments:

```bash
python legacy/aerosphinx_pipeline.py --config chapter   # literal Section 3 formulation
python legacy/aerosphinx_pipeline.py --select           # validation configuration search
python legacy/aerosphinx_pipeline.py --forecast-days 10 # longer operational forecast
python legacy/aerosphinx_pipeline.py \
    --cpcb-dir data/cpcb --aod-dir data/aod --merra-dir data/merra2 \
    --test-start 2024-05-17 --test-end 2024-06-18 --min-hours 18
```

## Two reproducible configurations

These belong to the legacy single-script pipeline, not to the Tracks. Its test
window is **33 days** (2024-05-17 to 2024-06-18) and its training window 130
days; a 33-day window cannot characterise seasonal behaviour, which is
assumption F3 in `docs/reference/SCIENTIFIC_ASSUMPTIONS.md`. The numbers are here because
the chapter reports them, and they are labelled so they are not mistaken for
the Track A result above.

| | `tuned` (default) | `chapter` |
|---|---|---|
| Target | `log1p(PM2.5)`, inverted for reporting | raw concentration |
| Features | base set + `PM25_Roll3`, `PM25_Roll7` | base set |
| Output | `0.6 · y_model + 0.4 · y_(t-1)` | `y_model` |
| Test days (n) | 33 | 33 |
| Test MAE / RMSE / R² | **17.24 / 22.81 / +0.025** | 23.29 / 30.28 / −0.718 |

The `chapter` configuration is the literal Section 3 and Section 5 formulation.
The `tuned` configuration adds three documented choices, each selected on a
validation window (10 April to 16 May 2024, **37 days**) that closes before the
test window opens, so the test set influenced no modelling decision. Note that
37 autocorrelated days choosing among 36 candidate configurations is a weak
selection signal: the spread across the whole validation grid is ~0.35 MAE
(21.92 to 22.27) while the test-window difference between configurations is
6 MAE, so the choice is close to arbitrary and should not be read as tuning.
`--select` regenerates the search and writes the full table to
`results/legacy_chapter/configuration_selection.csv`.

Model hyperparameters are fixed as published in Section 5 in both cases and are
never searched.

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
`results/legacy_chapter/operational_forward_forecast.csv`; horizon set by `--forecast-days`.

Running both is deliberate: the difference between them isolates the cost of
recursive lag substitution from the cost of imperfect meteorology.

## Generated outputs

| File | Contents |
|---|---|
| `results/legacy_chapter/chapter_tables.md` | Tables 1, 2, 3, 3b and 4, ready to paste |
| `results/legacy_chapter/metrics.json` | Full machine-readable audit of the run |
| `results/legacy_chapter/test_predictions.csv` | Per-day observed, RF, XGB, ensemble, residual |
| `results/legacy_chapter/five_day_recursive_forecast.csv` | Evaluated forecast by horizon day |
| `results/legacy_chapter/operational_forward_forecast.csv` | Forward forecast beyond last observation |
| `results/legacy_chapter/configuration_selection.csv` | Validation search (`--select` only) |
| `results/legacy_chapter/feature_importance.csv` | Ensemble, RF and XGB importances |
| `results/legacy_chapter/daily_multisource_dataset.csv` | Full harmonised daily dataset |
| `results/legacy_chapter/modelling_dataset_complete_cases.csv` | Rows actually used for modelling |
| `docs/figures/fig3_monthly_pm25_climatology.png` | Figure 3 |
| `docs/figures/fig4_observed_vs_predicted.png` | Figure 4 |
| `docs/figures/fig5_residuals.png` | Figure 5 |
| `docs/figures/fig6_feature_importance.png` | Figure 6 |
| `docs/figures/fig7_five_day_forecast.png` | Figure 7 |
