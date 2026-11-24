# AeroSphinx — Limitations

**Phase 0.** The limitations of the current repository and of the underlying
data. These constrain what may be claimed, now and after later phases.

Evidence: `docs/reports/REPOSITORY_AUDIT.md`, `docs/reference/DEFECT_REGISTER.md`,
`results/baseline/data_audit.json`.

---

## 1. Limitations of the current results

### 1.1 The headline model has essentially no explanatory power

The weighted ensemble achieves **R² = 0.017** on the held-out window, with
RMSE 22.90 against an observed standard deviation of 23.10 µg/m³. It
explains roughly **2% of the variance** in daily PM2.5 over the test period.

This is the most important limitation. No downstream analysis of feature
importance, error patterns or "what drives pollution" is interpretable when
the model itself has no skill.

### 1.2 There is no evidence that multisource fusion helps

On identical test rows, the **ground-history-only ablation (MAE 16.617)
outperforms the full multisource ensemble (MAE 17.298)**. Satellite-only
(19.260) and meteorology-only (20.119) are both worse than the persistence
baseline (19.918).

The project's central research question — does satellite and meteorological
information add measurable value? — is **currently answered "no"**, or more
precisely "not demonstrated on this data". The plan treats the multisource
track as a constrained experiment precisely because this outcome was
foreseeable.

**Revision 2026-11-23.** The comparison above is the legacy 33-day evaluation.
On the canonical matrix it is made on 49 Track B dates and now carries a
paired block-bootstrap interval and a Holm adjustment: AOD +0.21 [−2.19,
+2.61], MERRA-2 +4.24 [+1.26, +7.18] at Holm p = 0.059, full fusion +4.70
[+0.30, +10.41] at Holm p = 0.34. So the limitation is stronger than "does not
help": at this sample size the data cannot separate any environmental gain or
loss from zero, and the honest description of the result is *underpowered*,
not *negative*. See `docs/reports/PHASE7_REPORT.md` §10.

### 1.3 The improvement over persistence is not statistically significant

Paired Wilcoxon signed-rank test on the 33 test days: **p = 0.1495**.
Paired bootstrap 95% CI on the MAE difference: **[-6.47, +1.39]**, spanning
zero. The improvement is directionally consistent but not established.

Furthermore, the ensemble's advantage comes largely from a **0.4 persistence
anchor** applied after prediction. Without the anchor its MAE is 20.118 —
worse than persistence. The model is largely a persistence model with seasonal
and meteorological decoration.

### 1.4 The five-day "forecast" is a retrospective estimation

Forecast days receive the **observed** AOD and MERRA-2 values for those days.
The five-day forecast therefore measures estimation accuracy under perfect
environmental knowledge, not forecasting skill. As shipped, the repository
contains **no valid operational forecast**, because no future covariate
supply exists locally.

### 1.5 The model is largely a seasonal climatology

`month_sin` alone carries **37.93%** of the ensemble's normalised importance,
against 23.26% for the three PM2.5 history features combined. The model has
learned "what does May look like" more than "what drives aerosol loading".

### 1.6 The climatology baseline is mis-calibrated

The seasonal-naive baseline under-predicts by **-35.85 µg/m³** (MAE 35.87,
R² -2.68) because the test window sits in a higher-PM2.5 period than the
2016-2023 same-day-of-year average. This inflates every other model's
apparent competence.

---

## 2. Limitations of the data

### 2.1 The fused dataset is 166 rows over six months of one year

| Stream | Coverage | Effect |
|---|---|---|
| CPCB | 2016-01-01 to 2024-12-31 (2,979 daily rows) | full 9-year record available |
| INSAT-3D AOD | 2024-01-01 to 2024-06-18 (166 dates) | **limits everything** |
| MERRA-2 | 2024-01-01 to 2024-06-20 (172 days) | **limits everything** |
| **Fused frame** | **2024-01-07 to 2024-06-21, 166 rows** | single season, single station |

Track A has ~18× more data available than the multisource frame. Any
multisource conclusion is drawn from a 166-row, one-season, one-station
sample.

### 2.2 Single station, single city

One CPCB monitoring station in Faridabad, Haryana. Nothing in the data
supports generalisation to other cities, to rural sites, or to the Delhi-NCR
region as a whole. The plan's guidance against expanding to multiple cities
until the gates pass is well founded.

### 2.3 AOD support is thin and unflagged

Median **12 valid pixels** per granule in the station window; **27 granules
yield zero valid pixels**. The granules contain **no QA or cloud-flag
dataset**, so a valid pixel cannot be distinguished from a clear-sky
retrieval. A daily AOD derived from a single pixel is not a
station-representative measurement.

### 2.4 MERRA-2 features are midday single-cell values

The granules hold a **1×1 grid cell** at (28.5, 77.5) with **three hourly
timesteps** (11:30, 12:30, 13:30 UTC). The pipeline's docstrings describe
them as spatially averaged daily means; they are neither. The sampling window
covers peak boundary-layer mixing and is unrepresentative of the night-time
accumulation that dominates winter PM2.5. The grid point sits ~10 km north
and ~17 km east of the station.

### 2.5 The CPCB record changes character across the nine years

Integer-valued hourly readings fall from **96.1% (2016) to 0.3% (2023)**;
mean absolute successive-hour difference falls from **19.3 to 9.9 µg/m³**.
Early years are raw hourly instrument values; later years are progressively
smoothed or pre-aggregated. A Track A model spanning the full record is
trained across a change in measurement character, and no current test detects
it. The cause is **UNKNOWN** from local files alone.

### 2.6 The record has 123 calendar gaps

Longest gap **25 days** (2017-10-09 to 2017-11-03). Combined with the
row-based lag defect, some features labelled "yesterday" are weeks old.

### 2.7 Boundary layer height is unavailable

`PBLH` is absent from all 172 MERRA-2 granules, and `PBLTOP` is also absent.
PBLH is the single most physically relevant variable for aerosol dilution,
and its absence is a material constraint on Track B's mechanistic claims.

### 2.8 No future covariate supply exists

No NWP product (NCMRWF, ECMWF-IFS, GFS) and no satellite forecast is present.
This is what makes the retrospective/operational distinction binding rather
than academic, and it is why Phase 8 is conditional.

---

## 3. Limitations of the evaluation

### 3.1 A single 33-day test window

The test period covers 17 May – 18 June 2024 — one season, one episode type.
33 observations cannot characterise monsoon onset, winter peaks, or the
lockdown-affected 2020–2021 period. No rolling-origin backtesting has been
performed.

### 3.2 No uncertainty quantification on any published metric

The shipped `results/legacy_chapter/metrics.json` reports point estimates only. Paired
tests and bootstrap intervals are absent.

### 3.3 One split, one validation window

Configuration selection uses a single fixed validation window
(2024-04-10 to 2024-05-16) on a single split. Selection on one window
risks overfitting the validation period itself.

### 3.4 No significance testing on any comparison

Every claim of improvement is currently a comparison of point estimates.

### 3.5 No ablation on matched rows over multiple windows

The existing ablation table is correctly built on identical rows, which is
good, but there is only one window and no fold structure.

---

## 4. Limitations of the repository and engineering

| Limitation | Consequence |
|---|---|
| Single-file pipeline (1,373 lines) | No unit-testable modules; every change risks the whole pipeline |
| No test suite at all | No regression protection; no way to prove a fix |
| No `pyproject.toml`, `Makefile`, `AGENTS.md`, `requirements.lock` | No pinned, reproducible environment definition |
| Tracked binary model artefacts with no recorded provenance | Artefact-to-result linkage is inferable, not provable |
| Legacy pipelines and notebooks absent from the worktree | **Resolved in Phase 0** — restored under `legacy/`, 0 content loss |
| Seven duplicated CPCB files in git history | Latent double-counting hazard in aggregation |
| `requirements.txt` states Python 3.9; code needs 3.10+ | Environment floor is wrong |
| No CI | Reproducibility is asserted by hand |
| No experiment manifest for any existing run | Provenance of published results is partly reconstructed |

---

## 5. What this repository does **not** claim

Stated explicitly so that no reader over-reads the results.

- It does **not** demonstrate a working air-quality forecasting system.
- It does **not** demonstrate that satellite or meteorological data improves
  PM2.5 prediction.
- It does **not** demonstrate generalisation beyond one station in one city.
- It does **not** demonstrate seasonal or multi-year robustness.
- It does **not** provide official AQI; it provides a PM2.5 concentration
  category.
- It does **not** provide a physically validated aerosol model; its feature
  importance is dominated by a calendar term.

---

## 6. What would change these limitations

| Limitation | Resolution | Phase |
|---|---|---|
| No skill (R² 0.017) | More data, better features, honest baselines | 1–3 |
| No multisource benefit | More AOD coverage; matched-date ablations | 4–7 |
| Not significant | Rolling-origin folds; paired tests; more test days | 3 |
| Retrospective, not forecasting | Real NWP covariates at issue time | 8 |
| 166-row frame | Download INSAT and MERRA-2 for 2016–2023 | 4–5 (external dependency) |
| Thin AOD support | Higher-cadence granules; QA flags | 4 (external dependency) |
| Midday single-cell MERRA | Re-subset with full 24-hour coverage and a collocation box | 5 (external dependency) |
| Record heterogeneity | Document upstream provenance; test for shift | 1 (external dependency) |
| Stale lags, unsafe aggregation | Calendar-aware lags; hour-weighted means | 1 |
| No tests, no CI | Module split; test suite; CI | 1–9 |

---

## 7. Standing caveat on data availability

Four of the improvements above depend on data that **cannot be obtained from
the local checkout** and require either network access or explicit
permission: pre-2024 INSAT granules, a MERRA-2 subset with `PBLH` and full
temporal coverage, upstream CPCB provenance documentation, and an
issue-time-available NWP product. These are recorded as **BLOCKED** in
`docs/reference/DEFECT_REGISTER.md` (B-001 to B-003) and in
`docs/reference/SCIENTIFIC_ASSUMPTIONS.md` §H.

Phases 1–3 (Track A) are **not** blocked by any of these and can proceed
immediately.
