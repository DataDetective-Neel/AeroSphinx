# AeroSphinx — Experiment Protocol

**Phase 0.** The protocol that every experiment in this repository must
follow. Written before Phase 1 so that later results are comparable by
construction rather than by retrospective reconciliation.

The protocol is designed around the constraint the Phase 0 audit established:
**the fused multisource frame is 166 rows over six months of one year, while
the CPCB record alone spans nine years and 2,979 accepted daily days.** Track
A therefore has roughly 18× the data available, and the protocol is built so
that Track A is not handicapped by Track B's coverage limits.

---

## 1. Experimental identity

Every run is identified by a **run ID** with the form:

```
<track>-<config>-<split>-<YYYYMMDDTHHMMSSZ>-<short-sha>
```

Example: `A-tuned-roll1-20261112T141203Z-859fd08`.

The run ID is written into every artefact that run produces, and into the
experiment manifest. Two runs may never share an ID.

---

## 2. The two-track contract

### 2.1 Track A — historical CPCB forecasting

**Purpose.** Establish how far historical ground measurements alone can
predict future PM2.5.

**Data.** CPCB daily series only: 2,979 accepted days, 2016-01-01 to
2024-12-31.

**Allowed inputs.** Historical PM2.5 values dated strictly before the target
date; calendar features; **and**, where explicitly marked, station-local
meteorology from the CPCB export itself (`AT`, `RH`, `WS`, `WD`, `RF`, `SR`,
`BP`, `VWS`) — subject to the provenance check in H4 of
`docs/reference/SCIENTIFIC_ASSUMPTIONS.md`.

**Forbidden.** Any AOD, MERRA-2 or other externally sourced value that is not
available at forecast issue time.

**Experiments.**

| ID | Inputs | Question |
|---|---|---|
| A0 | Previous-day PM2.5 | Persistence baseline |
| A1 | Historical PM2.5 + calendar | Seasonal baseline |
| A2 | Historical PM2.5 + calendar + engineered temporal features | Does temporal ML beat persistence? |
| A3 | A2 + station-local meteorology | Does local meteorology add value? |
| A4 | A2 + CPCB-only MERRA/AOD **if** future availability can be demonstrated | Can environmental covariates ever be used operationally? |
| A5 | Direct horizon-specific models | Independent day 1–5 forecasts |
| A6 | Recursive forecasting | How does error compound? |

A0–A3 are the minimum viable Track A. A4 is conditional on H1 resolving.
A5–A6 are required before Track A can be called a forecasting system.

### 2.2 Track B — multisource estimation

**Purpose.** Measure whether satellite and meteorological information adds
value over ground-only models.

**Data.** CPCB + INSAT-3D AOD + MERRA-2, on matched dates.

**Mode.** **Retrospective estimation only.** Track B may not be described as
forecasting unless and until a future covariate supply is demonstrated
(Phase 8).

**Experiments.**

| ID | Inputs | Question |
|---|---|---|
| B0 | CPCB historical features | Ground-only benchmark |
| B1 | AOD only | How informative is satellite aerosol alone? |
| B2 | MERRA-2 only | How informative is meteorology alone? |
| B3 | CPCB + AOD | Does satellite add value? |
| B4 | CPCB + MERRA-2 | Does meteorology add value? |
| B5 | CPCB + AOD + MERRA-2 | Does full fusion beat its components? |
| B6 | Availability-aware combination | How robust is the model when a source is missing? |

**B0 vs B5 is the headline comparison** and must be run on identical dates and
identical target rows. If B0 is evaluated on 100 dates and B5 on 25, the
comparison is void — this is the single most common reporting error the plan
warns about, and the current repository's own ablation table is currently
built correctly on matched rows and must stay that way.

### 2.3 Shared infrastructure, separate assumptions

Track A and Track B may share ingestion, provenance tracking, feature
contracts, experiment management and evaluation utilities. They **must not**
share a feature matrix, a split definition, or a metric table without an
explicit statement that the comparison is invalid.

---

## 3. Availability contract

This is the core methodological mechanism. Every feature must be tagged with
its availability class, and the tag determines where it may be used.

| Class | Definition | Permitted in retrospective estimation | Permitted in operational forecast |
|---|---|---|---|
| `observed_ground` | CPCB PM2.5 at the target date | Target only | Target only |
| `observed_ground_history` | CPCB PM2.5 dated before the target date | Yes | Yes |
| `observed_satellite` | INSAT AOD at the target date | Yes | **No** |
| `observed_satellite_history` | INSAT AOD dated before the target date | Yes | Yes, with stated latency |
| `reanalysis` | MERRA-2 reanalysis at the target date | Yes | **No** |
| `reanalysis_history` | MERRA-2 dated before the target date | Yes | Yes, with stated latency |
| `forecast_covariate` | NWP or satellite forecast valid at the target date | Yes | Yes |
| `calendar` | Deterministic function of the target date | Yes | Yes |

**Rules.**

1. No feature may enter a model without an availability class.
2. A forward-filled value is never assigned the class of the column it
   fills. It is `imputed` and must be excluded or flagged as such.
3. Every forecast record carries the **forecast issue timestamp** and the
   **timestamp of each input observation**. A forecast whose inputs cannot be
   shown to have been available at issue time is invalid and must not be
   reported.
4. When a required source is unavailable, the system selects a model trained
   and evaluated on that exact feature combination, or falls back to a
   documented alternative. It never silently substitutes a value.

### 3.1 Issue-time record

Every forecast output must record:

```
forecast_issue_timestamp
target_date
horizon_days
model_id, model_version, git_sha
input_manifest            (feature -> value -> source -> observation timestamp)
source_coverage           (per source: present, age_days, valid_pixels)
predicted_pm25
prediction_interval       (only if calibrated and validated)
```

---

## 4. Feature construction rules

1. **Calendar-aware lags.** Lags are computed against an explicit calendar.
   If the previous calendar day is absent, the lag is `NaN` — never a stale
   observation relabelled as yesterday's. (Fixes D-001.)
2. **Hour-weighted daily means.** Aggregate hourly records across all files
   first, then compute one daily mean per date. File-level daily means are
   never averaged. (Fixes D-013.)
3. **No duplicate hours.** Deduplicate on timestamp before aggregation, and
   assert zero duplicates as a test.
4. **Rolling windows** use only strictly prior calendar days and report the
   number of contributing observations.
5. **Imputation is explicit.** Any imputed value carries an
   `_imputed` suffix or a paired boolean indicator, plus an age in days.
6. **No full-dataset statistics before splitting.** Normalisation, scaling or
   imputation values are fitted on the training partition only.
7. **Coverage is a feature.** `aod_valid_pixels`, `aod_coverage_fraction`,
   `merra_timesteps_present` and per-source presence indicators are model
   inputs, not just diagnostics. (Fixes D-003, D-004.)
8. **Physically accurate naming.** A feature name states what the quantity is.
   `Humidity_Ratio` is replaced. (Fixes D-009.)
9. **Documented extraction windows.** The collocation window and temporal
   window are configuration, and sensitivity to both is reported.

---

## 5. Evaluation protocol

### 5.1 Splits

**Track A: rolling-origin backtesting.**

Training windows expand; each fold forecasts a strictly later block.

```
Fold 1:  train [-----------] validate [--] test [--]
Fold 2:  train [---------------] validate [--] test [--]
Fold 3:  train [-------------------] validate [--] test [--]
```

Fold boundaries are **generated from the validated dataset**, never
hardcoded. A minimum training length is enforced. Every fold records its
exact train/validate/test boundaries in the manifest.

**Track B: matched-date ablation.**

All ablation variants are fitted and scored on **one identical row set**.
Where a source is missing for some dates, the comparison set is restricted to
dates where all compared sources are present, and the restriction is stated
in the results.

### 5.2 Metrics

Report for every model and every fold:

- MAE
- RMSE
- R²
- Mean bias error
- Performance by forecast horizon
- Performance by month and by season
- Performance against the persistence and seasonal-naive baselines
- Error distributions; prediction intervals where supportable
- AQI **concentration category** accuracy, macro-F1, and confusion matrix

### 5.3 Uncertainty — mandatory

A single point estimate is not a reportable result.

- **Paired tests** for every model-vs-baseline comparison (Wilcoxon
  signed-rank on per-observation errors).
- **Bootstrap confidence intervals** on metric differences, using a fixed
  seed.
- **Fold-level results** reported individually as well as aggregated.
- A claim of improvement requires the interval to **exclude zero**. The
  current headline comparison does not meet this bar (p = 0.1495) and must be
  reported as such until it does.

### 5.4 Statistical honesty

- Do not present an improvement because one MAE is numerically smaller.
- Do not compare metrics computed on different row sets.
- Do not report a metric for a model that could not be fitted.
- Do not tune on held-out data.
- Report the baseline's own bias, so a shared systematic offset cannot hide
  inside a relative comparison. (Fixes D-005.)

---

## 6. Model selection

Candidate models, in the order they are tried:

1. Persistence baseline
2. Seasonal naive baseline
3. Regularized linear regression (ridge)
4. Random Forest
5. XGBoost

Additional models are added only when a baseline failure or a specific
research question justifies them.

**Tuning discipline.** Hyperparameter search is performed on a validation
window strictly before the test window, on a single pre-declared split, and
the search space is recorded. Selecting a configuration by inspecting test
performance is prohibited. The current `select_configuration` implements this
correctly for its single split; it must be generalised to the rolling-origin
folds.

**Ensembling.** If an ensemble is used, its weights are fixed a priori or
selected on validation data only, and the unweighted members are reported
alongside it. The current ensemble's apparent skill comes substantially from
a persistence anchor applied post-hoc; both the anchored and unanchored
variants are always reported.

---

## 7. Forecasting protocol

### 7.1 Two modes, one interface

| | Mode A (ground-based) | Mode B (environmental) |
|---|---|---|
| Status | **Implement first** | Conditional on Phase 8 |
| Inputs | Observed PM2.5 history + calendar | + future-available AOD and NWP meteorology |
| Direct | Separate models for days 1–5 | Separate models for days 1–5 |
| Recursive | Predictions re-enter as later lags | Same, with covariate substitution rules |

### 7.2 Recursive forecasting rules

- The anchor parameter is re-selected or disabled per horizon.
- Predictions re-entering the feature vector are marked as predicted, not
  observed.
- The recursion depth is a reported parameter; results are shown per depth.
- A recursive forecast is never described as operational unless every
  environmental input satisfies the availability contract.

### 7.3 Leakage tests — mandatory before any forecast result is reported

1. **Issue-time availability test** — no input value derives from an
   observation dated after the forecast issue time.
2. **Future-observation leakage test** — asserted at unit level for both
   direct and recursive modes.
3. **Recursive propagation test** — verifies that predictions, not
   observations, feed the next step's lag features.
4. **Missing-covariate fallback test** — verifies that a documented fallback
   model is selected when a source is absent, and that no value is
   fabricated.

---

## 8. Experiment manifest

Every run writes a manifest containing:

- Git SHA and branch
- Environment and package versions
- Input hashes (SHA-256 of every raw file consumed)
- Dataset counts and date range
- Feature list with availability class per feature
- Source availability rules in force
- Split boundaries (exact dates, row counts, and the row-set hash)
- Model configuration and random seed
- Metrics, with uncertainty intervals
- Prediction artefact locations
- Run ID

The manifest is machine-readable JSON. `results/baseline/manifest.json` and
`results/baseline/data_audit.json` are the Phase 0 exemplars.

---

## 9. Reporting standards

- Report fold-level and aggregate results separately.
- Always publish the baseline table alongside the headline table.
- Always publish uncertainty.
- State the test window's season and coverage explicitly.
- State the number of test observations alongside every metric.
- Distinguish "no evidence of improvement" from "evidence of no improvement".
- Do not describe Track B results as forecasting unless Phase 8 has been
  completed and passed.
- Do not describe a PM2.5 concentration category as the official AQI.

---

## 10. Phase gates and acceptance criteria

| Phase | Gate — all must pass before proceeding |
|---|---|
| 0 | Audit complete, baseline reproducible, findings approved. **DONE** |
| 1 | CPCB ingestion produces a validated daily series with a provenance record; duplicate-hour, unequal-file-size, timestamp and missing-hour tests pass; daily means are hour-weighted; lags are calendar-aware. |
| 2 | Track A features and baselines implemented; persistence and seasonal-naive baselines defined and computed on identical rows. |
| 3 | Rolling-origin backtesting implemented; at least 3 folds; paired tests and bootstrap intervals reported; honest reporting of whether Track A beats persistence. |
| 4 | INSAT extraction validated: dimension/orientation, fill-value, window and duplicate-granule tests pass; valid-pixel coverage reported per date. |
| 5 | MERRA extraction validated: dimension, timestamp, unit, grid-extraction and aggregation tests pass; feature naming corrected to match the actual data. |
| 6 | Fusion and availability layer implemented with an explicit availability mask; no silent imputation; missing-source behaviour documented. |
| 7 | Track B ablations run on matched dates; improvement claim requires a CI excluding zero, otherwise the negative result is reported as the finding. |
| 8 | Operational forecast demonstrated with genuinely issue-time-available covariates; fallback behaviour verified by test. |
| 9 | Full test suite green; every headline result reproducible from a clean environment; manifests complete for every run. |

**A gate is not passed by assertion.** Each requires executed tests with
reported output. `NOT RUN` is an acceptable gate outcome; a falsely reported
`PASS` is not.
