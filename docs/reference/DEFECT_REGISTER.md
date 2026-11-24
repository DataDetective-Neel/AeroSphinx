# AeroSphinx — Defect Register

Phase 0 repository autopsy. Every entry below is classified as **VERIFIED**
(directly supported by evidence reproduced in this repository), **INFERRED**
(logically derived but not directly confirmed) or **BLOCKED** (cannot be
settled without data or permission that is not available).

Machine-readable evidence: `results/baseline/data_audit.json`,
`results/baseline/manifest.json`.

Severity reflects scientific impact on the research claims, not code aesthetics.

## Resolution status

| Defect | Severity | Status | Resolved in |
|---|---|---|---|
| D-001 Row-based PM2.5 lags are not calendar-aware | Critical | **FIXED** | Phase 1 — `src/aerosphinx/features/lags.py` |
| D-002 Forecast consumes future observed satellite/reanalysis data | Critical | OPEN | Phase 8 |
| D-003 Forward-fill manufactures environmental observations | High | **FIXED** (Phase 6) | `src/aerosphinx/features/availability.py` |
| D-004 27 INSAT granules contribute zero valid pixels | High | **QUANTIFIED** (Phase 4) | `src/aerosphinx/ingestion/aod.py` |
| D-005 Climatology baseline mis-specified | High | **FIXED** (Phase 2) | `src/aerosphinx/models/baselines.py` |
| D-006 Improvement over persistence not significant | High | **ADDRESSED** (Phase 2) | `src/aerosphinx/evaluation/uncertainty.py` |
| D-007 Fusion does not beat the ground-only ablation | High | OPEN (finding, not a code defect) | Phase 7 |
| D-008 PBLH unavailable; `PBLTOP` fallback is a latent trap | Medium | OPEN | Phase 5 |
| D-009 `Humidity_Ratio` mis-named and mis-defined | Medium | OPEN | Phase 5 |
| D-010 MERRA-2 features are midday single-cell values | Medium | **FIXED** (Phase 5) | `src/aerosphinx/ingestion/merra.py` |
| D-011 Feature importance dominated by `month_sin` | Medium | **PARTIALLY ADDRESSED** (Phase 2) | `docs/reports/PHASE2_REPORT.md` |
| D-012 Duplicate CPCB source files | Medium | **FIXED** | Phase 0 — `legacy/`, `data/cpcb/` |
| D-013 Mean-of-file-means aggregation is unsafe | Medium | **FIXED** | Phase 1 — `src/aerosphinx/ingestion/cpcb.py` |
| D-014 Legacy pipelines absent from the worktree | Medium | **FIXED** | Phase 0 — `legacy/` |
| D-015 CRLF line-ending conversion | Low | Documented, no action required | — |
| D-016 Operational forecast extrapolates an upward drift | Low | OPEN | Phase 8 |
| D-017 Baseline defect shrank the headline evaluation set | High | **FIXED** (Phase 3 audit) | `src/aerosphinx/models/baselines.py` |
| D-018 Bootstrap overstated the effective sample size | High | **FIXED** (Phase 3 audit) | `src/aerosphinx/evaluation/block_inference.py` |
| D-019 Horizon-appropriate baseline was not used | Critical | **FIXED** (Phase 3 Workstream A) | `src/aerosphinx/tracks/multi_horizon.py` |
| D-020 The CPCB barometric pressure column is corrupt | High | **FIXED** (Phase 5) | `config/data_quality.yaml` |

---

## D-001 — Row-based PM2.5 lags are not calendar-aware

| | |
|---|---|
| **Severity** | Critical |
| **Status** | VERIFIED |
| **Location** | `aerosphinx_pipeline.py:427-430` (`engineer_features`) |
| **Evidence** | `results/baseline/data_audit.json` → `cpcb.rows_where_lag1_is_not_previous_calendar_day` |

**Observed behaviour.** Lags are computed with `out[TARGET].shift(lag)`, which
advances by *row position*, not by calendar day. The accepted CPCB daily series
has **123 calendar gaps**, so for 123 of 2979 rows the value labelled
`PM25_Lag_1` is not the previous calendar day. The worst case is **25 days**
old (2017-10-09 → 2017-11-03). `PM25_Roll3` and `PM25_Roll7` inherit the same
defect through the same shift-then-roll construction.

**Impact.** A feature advertised as "PM2.5 on day t-1" can in fact be a
3-week-old observation. This inflates apparent temporal skill wherever the
series is gappy, and directly contradicts the plan's requirement that "a
missing date must not make a two-day-old observation masquerade as yesterday's
measurement". It also makes `PM25_Roll3`/`PM25_Roll7` semantically undefined
whenever a gap falls inside the window.

**Proposed correction.** Re-index the daily frame onto a complete
`date_range` before computing lags, or use `Series.reindex(dates).shift(lag)`
against an explicit calendar. Carry an `observation_age_days` column so
staleness is visible to the model and to the reader.

**Test proving the correction.** A unit test asserting that for every row the
value in `PM25_Lag_1` equals the observation dated exactly one calendar day
earlier, and that rows whose previous calendar day is missing carry `NaN`
rather than a stale value.

**Resolution applied (Phase 1).** `src/aerosphinx/features/lags.py` implements
`calendar_lag`, `calendar_lags`, `calendar_rolling`, `calendar_rolling_stats`
and `observation_age_days`. Every temporal feature reindexes onto the full
calendar before shifting, so a missing date becomes an explicit `NaN` row
rather than being bridged by a row-position shift. On the real record this
changes **123 rows** from a stale value (mean 3.5 days old, worst case
**25 days**) to `NaN`.

Proof tests: `tests/unit/test_lags.py` (including
`test_row_based_shift_would_have_been_wrong`, which pins the legacy defect so
it cannot silently return) and
`tests/integration/test_track_a_ingestion.py::test_calendar_aware_lag_rejects_stale_values`.

---

## D-002 — The 5-day "forecast" consumes future observed satellite and reanalysis data

| | |
|---|---|
| **Severity** | Critical |
| **Status** | VERIFIED |
| **Location** | `aerosphinx_pipeline.py:774` (`recursive_forecast`), `aerosphinx_pipeline.py:833-843` (`forward_forecast`) |
| **Evidence** | Reproduced directly; see `docs/reports/REPOSITORY_AUDIT.md` §5 |

**Observed behaviour.** `recursive_forecast` reads the target day's row
directly from the merged observation frame:

```python
row = _update_recursive_features(source.loc[forecast_date], history)
```

That row carries the **observed** AOD and **observed** MERRA-2 values for the
forecast day itself. With a forecast origin of 2024-06-16, the model received
genuine satellite AOD for 2024-06-17 and 2024-06-18 and a *forward-filled*
AOD value for 2024-06-19 through 2024-06-21. The docstring states that
"meteorological and calendar inputs are taken as given, matching the chapter's
assumption that forward meteorology is supplied by an external forecast
product" — but no external forecast product exists in this repository. The
values are reanalysis and satellite observations from the future.

`forward_forecast` has the mirror-image problem: it persists the last
observed meteorological values forward and labels them
`"persisted from <date>"`. That label is honest, but the resulting numbers are
not forecasts.

**Impact.** This is the single largest scientific risk in the repository. The
five-day forecast table and Figure 7 present retrospective estimation as
operational forecasting. Under the plan's availability contract, future
observed AOD and future observed reanalysis are **forbidden** inputs to an
operational forecast. As shipped, the pipeline cannot produce a valid
operational forecast at all.

**Proposed correction.** Split the feature contract explicitly into a
retrospective mode and an operational mode. Track B retrospective estimation
may use same-day observed AOD/MERRA. Any operational forecast must carry an
explicit issue timestamp, may only use covariates genuinely available at that
timestamp, and must fall back to a CPCB-only model when future covariates are
unavailable.

**Test proving the correction.** A leakage test asserting that no feature
value used at forecast step *h* derives from an observation dated after the
forecast issue time.

---

## D-003 — Free forward-fill silently manufactures environmental observations

| | |
|---|---|
| **Severity** | High |
| **Status** | VERIFIED |
| **Location** | `aerosphinx_pipeline.py:407-408` (`harmonise`) |
| **Evidence** | `results/baseline/data_audit.json` → `alignment.rows_with_forward_filled_aod` |

**Observed behaviour.** After merging the three sources, `harmonise` applies
`ffill(limit=3)` to AOD and every MERRA-2 variable. The filled rows are
indistinguishable from genuinely observed rows in the modelling frame: there
is no source-presence indicator, no observation-age column, and no record of
which values were imputed. Of the 166 modelling rows, **7 carry a
forward-filled AOD value** and 1 carries a forward-filled MERRA-2 value.

**Impact.** The plan requires that "never silently fill missing environmental
measurements with fabricated values" and that each prediction record its
"feature availability timestamps" and "source coverage and missingness". A
forward-filled AOD is a stale value presented as a contemporaneous one, which
is exactly the failure mode that an availability-aware model must not
inherit. It also distorts the multisource ablation, because a "fusion" model
may be partly running on imputed satellite inputs while the ground-only
comparison sees real observations.

**Proposed correction.** Replace filling with an explicit availability mask.
Emit `aod_observed`, `merra_observed` and per-source `observation_age_days`
columns; never substitute a value for a missing observation in a model that
claims to use it.

**Test proving the correction.** A test asserting that no modelling row
contains an AOD or MERRA-2 value that is not backed by a same-day source
record, unless the row is explicitly flagged as imputed.

**Resolution applied (Phase 6).**
`src/aerosphinx/features/availability.py` builds the canonical fused frame
with an explicit availability mask. The ground series defines the row
calendar exactly: **2,979 rows, 2016-01-01 to 2024-12-31, 0 extended and 0
dropped**. Every row carries `<source>_observed` and
`<source>_observation_age_days` for `station_met`, `merra` and `aod`.
Missing values are preserved as `NaN`; nothing is forward-filled,
interpolated or imputed. AOD support counts are `0` where no AOD
observation exists, because they are support metadata rather than
measurements.

The 33-check alignment gate (`validate_fused_dataset`) verifies calendar
integrity, mask consistency, AOD pixel support and — for all 17 columns —
same-day provenance against the source frame. On the real record **0 rows
carry a value without a same-day observation** and **0 checks fail**. The
published dataset is `data/processed/fused_dataset.csv`; the audit is
`results/runs/F-fusion-20261113T071533Z/alignment_audit.json`.

Proof tests: `tests/unit/test_fusion_availability.py` (including
`test_missing_environmental_values_are_not_forward_filled` and
`test_alignment_gate_rejects_a_forward_filled_value`, which pins the exact
defect) and `tests/integration/test_phase6_fusion.py`
(`test_no_environmental_value_without_a_same_day_observation`,
`test_no_source_value_is_forward_filled`).

The legacy `aerosphinx_pipeline.py` retains its `ffill(limit=3)` because it
is the frozen Phase 0 baseline, preserved unchanged under the
repository-preservation rule. It is not the research path and must not be
used to build a Track B experiment.

**Coverage consequence.** The mask makes the real limitation visible: all
three sources are observed together on only **139 of 2,979 rows (4.7%)**,
because AOD and MERRA-2 cover 2024 only. Phase 7 must compare groups on
matched dates; the masked frame is deliberately not reduced to those 139
rows, which would remove all winter and pre-monsoon ground history.

---

## D-004 — 27 of 169 INSAT granules contribute zero valid pixels

| | |
|---|---|
| **Severity** | High |
| **Status** | **QUANTIFIED (Phase 4)** |
| **Location** | `src/aerosphinx/ingestion/aod.py` |
| **Evidence** | `results/runs/D-aod-20261113T042245Z/` |

**Observed behaviour.** The station collocation window is +/-0.25 degrees
around 28.4089 N, 77.3178 E on a 0.1-degree INSAT-3D grid, i.e. at most
5 x 5 = 25 pixels. Across 169 granules, **27 yield no valid pixel at all**
inside that window (fill value -999 throughout), and the median granule
contributes only **12** valid pixels. The pipeline records `aod_valid_pixels`
but never uses it as a feature, a filter, or a reported confidence measure.

**Impact.** A daily AOD value derived from a single pixel is not a station
representative measurement, yet it enters the model as an ordinary covariate
with no indication of its support. This is the plan's requirement to "report
valid pixel coverage for every date" and to treat coverage as a first-class
signal. The absence of any QA or cloud-flag dataset in the granules
(`has_qa_flag_dataset: false`) means there is no way to distinguish "clear
sky, low aerosol" from "partially cloudy, retrieval uncertain" using these
files.

**Proposed correction.** Add `aod_valid_pixels` and `aod_coverage_fraction`
as features and as a documented quality gate, and report the distribution of
pixel support alongside every satellite-derived result.

**Resolution (Phase 4).** The `aod` feature group now carries `AOD`,
`aod_valid_pixels` and `aod_overpasses`, so the support of every satellite
value is visible to the model. The coverage report is published per date
(`aod_coverage.csv`) and the missingness was found to be **informative
rather than random**: days with no valid AOD pixel have a mean ground PM2.5
of **119.38 ug/m3** against **105.34 ug/m3** on days with one
(Mann-Whitney p = 0.032). The satellite record is least available exactly
when pollution is worst. The thin support is the binding limitation: the
median granule contributes **6** valid pixels out of a possible 25, and the
granules carry **no QA or cloud-flag dataset**, so a present pixel means a
retrieval was attempted rather than that it is trustworthy.

---

## D-005 — Climatology baseline is mis-specified and materially under-predicts

| | |
|---|---|
| **Severity** | High |
| **Status** | VERIFIED |
| **Location** | `aerosphinx_pipeline.py:632-638` (`run_baselines`) |
| **Evidence** | Reproduced: MAE 35.87, bias -35.85 against observed mean 100.63 |

**Observed behaviour.** The seasonal-naive baseline groups **all** CPCB
history by day-of-year and averages across years, then maps test dates
through that average. The test window (17 May - 18 June 2024) happens to sit
in a high-PM2.5 period relative to the 2016-2023 mean for the same
day-of-year, so the baseline predicts a mean of 64.78 against an observed
mean of 100.63 — a systematic under-prediction of -35.85 ug/m3, MAE 35.87 and
R2 of -2.68.

**Impact.** A baseline that is this badly mis-calibrated makes every other
model look competent by comparison. It flatters the ensemble (MAE 17.30)
without any real evidence that the ensemble is skilful. The plan explicitly
warns that "a small number of test days means large uncertainty" and requires
baselines that are meaningful comparators.

**Proposed correction.** Build the seasonal baseline from the most recent
available years only (or use a quantile-matched climatology), and report the
baseline's own bias so that a shared systematic offset cannot hide inside a
relative comparison.

**Resolution applied (Phase 2).** `ClimatologyBuilder` restricts the
climatology to a configurable lookback window and reports its own fitted
range. `SeasonalNaiveBaseline.select_lookback` chooses the lookback **per
fold on an inner validation block** carved from the training data, so no test
observation influences the choice; the selected lookback moves from 1 year in
the early folds to 9 years later as the expanding window makes longer
histories available.

The baseline's own bias is now reported alongside its metrics. With fair
tuning the seasonal baseline improves (MAE 38.08 to 37.24 over the 96-fold
backtest) but remains **significantly worse than persistence** (delta
+15.83 ug/m3, p = 9.1e-87), with a residual bias of +3.75 ug/m3. So the
defect was real and is corrected, but the correction does not rescue the
baseline: a day-of-year climatology cannot predict day-to-day variability in
Delhi-NCR PM2.5.

---

## D-006 — The reported improvement over persistence is not statistically significant

| | |
|---|---|
| **Severity** | High |
| **Status** | VERIFIED |
| **Location** | `results/legacy_chapter/metrics.json` → `metrics.WeightedEnsemble_0.6RF_0.4XGB`, `baselines.Persistence_baseline` |
| **Evidence** | Paired Wilcoxon signed-rank test, n = 33 |

**Observed behaviour.** The ensemble reports MAE 17.298 against a persistence
baseline of 19.918 on the same 33 test days. A paired Wilcoxon signed-rank
test on the per-day absolute errors gives **p = 0.1495**, and a paired
bootstrap of the MAE difference gives a 95% confidence interval of
**[-6.47, +1.39]**, which spans zero.

**Impact.** The headline result — that the multisource model beats simple
persistence — is **not statistically supported** at the conventional 0.05
level. Presenting the MAE difference alone materially overstates the result.
The plan requires "fold-level results, paired comparisons and appropriate
uncertainty estimates" and states that one must not "claim statistically
significant improvement merely because one MAE is numerically smaller".

**Proposed correction.** Publish the paired test and bootstrap interval
alongside every headline metric. Do not describe the ensemble as beating
persistence until the interval excludes zero.

**Resolution applied (Phase 2).** `src/aerosphinx/evaluation/uncertainty.py`
provides `paired_wilcoxon` and `paired_bootstrap_ci` (paired on the day, with
a fixed seed for reproducibility), and `compare_models` produces a table
with an explicit `verdict` column. Every comparison in the Track A backtest
now carries a p-value, a bootstrap interval and a verdict, so a numerically
smaller MAE can no longer be mistaken for a demonstrated improvement.

The extended finding, from the 96-fold backtest over the full nine-year
record: **no learned model significantly beats persistence at one-day
horizon.** Random Forest matches it (delta -0.013, p = 0.86), ridge matches
it (delta +0.101, p = 0.52), and XGBoost is significantly worse (delta
+0.965, p = 0.018). The learned models *do* beat persistence in the
pre-monsoon and winter seasons, which is the physically expected pattern,
but the effect is not large enough to move the nine-year aggregate past
significance. See `docs/reports/PHASE2_REPORT.md`.

---

## D-007 — The ensemble does not beat its own ground-only ablation

| | |
|---|---|
| **Severity** | High |
| **Status** | VERIFIED |
| **Location** | `results/legacy_chapter/metrics.json` → `baselines` |
| **Evidence** | Reproduced; all 81 numeric metrics reproduce exactly |

**Observed behaviour.** On the identical 33 test rows, the ablations give:

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| `Ground_history_only` | **16.617** | 23.671 | -0.050 |
| `No_satellite_AOD` | 17.205 | 22.976 | 0.011 |
| `WeightedEnsemble` (full) | 17.298 | 22.902 | **0.017** |
| `Persistence_baseline` | 19.918 | 24.186 | -0.096 |
| `Satellite_AOD_only` | 19.260 | 24.355 | -0.112 |
| `Meteorology_only` | 20.119 | 26.811 | -0.347 |

The full multisource ensemble is **worse on MAE** than the ground-history-only
ablation, and the fused model has an R2 of **0.017** — essentially no
explanatory power. The satellite-only and meteorology-only models are both
worse than persistence.

**Impact.** There is currently **no evidence that satellite or meteorological
information adds predictive value**. The repository's central research
question — the entire justification for Track B — is not supported by its own
results. Any claim of multisource benefit would be a scientific overclaim.
Note also that the full ensemble's apparent advantage comes substantially from
the 0.4 persistence anchor: the unanchored ensemble has MAE 20.118, i.e.
worse than persistence.

**Proposed correction.** Report the full ablation table as the primary
Track B result, state plainly that fusion did not beat ground-only, and
treat the demonstration of incremental value as the open research question
rather than a settled result.

---


**Revision 2026-11-23 — inference added, and it changes the claim.** The
figures above come from the legacy 33-day evaluation. On the canonical matrix
(Phase 7) the comparison is made on 49 Track B dates and now carries a paired
moving-block bootstrap interval and a Holm adjustment, alongside every other
Track B ablation:

| Comparison | Δ MAE | 95% CI | Holm p | Distinguishable from zero |
|---|---|---|---|---|
| M2 − M1 (adds AOD) | +0.21 | [−2.19, +2.61] | 1.000 | no |
| M3 − M1 (adds MERRA-2) | +4.24 | [+1.26, +7.18] | 0.059 | no |
| M4 − M1 (full fusion) | +4.70 | [+0.30, +10.41] | 0.340 | no |

The defect as stated — the ensemble does not beat its own ground-only
ablation — remains true as an ordering of point estimates, and the separate
finding that Track B's ground-only model does not beat persistence on the full
80-date set (28.59 vs 22.98) is unaffected. What is no longer supported by the
data is a *measured* cost of adding a source: at n = 49 with daily
autocorrelation, no environmental delta is separable from zero. The phase
report's §10 states the same thing. Registry: `results/CANONICAL.json`.

## D-008 — PBLH is unavailable and the configured fallback is wrong

| | |
|---|---|
| **Severity** | Medium |
| **Status** | VERIFIED |
| **Location** | `aerosphinx_pipeline.py:124-125` (`MERRA_REQUESTED`, `MERRA_FALLBACKS`) |
| **Evidence** | `results/baseline/data_audit.json` → `merra2.requested_variables_missing_in_all_granules` |

**Observed behaviour.** The pipeline requests `PBLH` and declares
`MERRA_FALLBACKS = {"PBLH": ["PBLTOP"]}`. **Neither `PBLH` nor `PBLTOP` is
present in any of the 172 granules** — both are missing in all 172 files. The
pipeline therefore drops the feature and reports `PBLH` under
`features_unavailable`, which is the correct *outcome*.

**Impact.** The outcome is right but the mechanism is not. The plan explicitly
forbids "PBLTOP-to-PBLH substitution without established scientific
equivalence", and `PBLTOP` (pressure at top of boundary layer) is not a
substitute for `PBLH` (boundary layer height) in any case. The fallback is
dead code that would silently activate if a future granule happened to
contain `PBLTOP`, producing a physically invalid feature. The README
correctly documents the current absence, but the code carries a latent trap.

**Proposed correction.** Remove the `PBLTOP` fallback entirely. If boundary
layer height is required, it must be obtained by re-subsetting MERRA-2 with
`PBLH` included.

---

## D-009 — `Humidity_Ratio` is mis-named and mis-defined

| | |
|---|---|
| **Severity** | Medium |
| **Status** | VERIFIED |
| **Location** | `aerosphinx_pipeline.py:425`, `aerosphinx_pipeline.py:171-172` |
| **Evidence** | `results/baseline/data_audit.json` → `merra2.variable_metadata` |

**Observed behaviour.** The feature is defined as `QV2M / (T2M + EPS)` and
described as "Combined moisture-temperature aerosol growth term". MERRA-2
`QV2M` is 2-metre **specific humidity** (kg/kg, verified from the granule
attribute `units='kg kg-1'`, `long_name='2-meter_specific_humidity'`) and
`T2M` is 2-metre air temperature in K. Their ratio is not a humidity ratio
in any conventional sense.

**Impact.** The plan states plainly that "QV2M/T2M is not a conventional
humidity-ratio calculation" and requires the feature to be replaced. As
labelled, the feature invites a reader to mis-interpret it. Specific humidity
itself (`QV2M`) is already in the feature list, so the derived ratio adds a
weakly-motivated column with a misleading name.

**Proposed correction.** Rename to a physically accurate name (e.g.
`SpecificHumidityPerKelvin`), or drop it and retain `QV2M` directly. If a
hygroscopic-growth proxy is wanted, define it explicitly and justify the
form.

---

## D-010 — MERRA-2 granules contain 3 hourly steps per day, not a daily series

| | |
|---|---|
| **Severity** | Medium |
| **Status** | VERIFIED |
| **Location** | `data/merra2/*.SUB.nc` (granule metadata), `aerosphinx_pipeline.py:340-381` |
| **Evidence** | `results/baseline/data_audit.json` → `merra2.time_signatures_utc` |

**Observed behaviour.** The product is `tavg1_2d` (2-daily, 1-hourly
averaged), but the distributed granules were subsetted to **timestamps 12, 13
and 14 only** (11:30, 12:30 and 13:30 UTC), on a **1 x 1 grid** at
28.5 N, 77.5 E. The granule's own `history_L34RS` attribute records
`Spatial: 77.2 77.4 28.3 28.5. Timesteps: 12 13 14`. The pipeline averages
those three values and calls the result a daily mean.

**Impact.** Every MERRA-2 feature in the model is a **midday window mean from
a single grid cell**, not a daily mean over a collocation box. The code
docstring claims the values are "spatially averaged over the Faridabad
bounding box", which is not what the data supports. Consequences: the
features are not diurnally representative, the station-grid offset
(28.5, 77.5 versus 28.4089, 77.3178) is unquantified, and the temporal
sampling is biased toward the hours of peak boundary-layer mixing.

**Proposed correction.** Describe the features accurately as
midday-window single-cell values, or re-subset MERRA-2 with full 24-hour
coverage and a deliberate collocation box. Either way, correct the docstring
and the feature descriptions, and quantify the grid-point offset.

**Resolution (Phase 3 Workstream C, naming half).** The extraction now lives
in `src/aerosphinx/ingestion/merra.py` and every variable carries a
`_midday` suffix, because that is what the value is: the mean over the three
hourly timesteps the distributed granules contain (11:30, 12:30, 13:30 UTC)
at a single 1x1 grid cell. The misleading docstrings are gone and the
provenance is recorded in the run manifest.

**Resolution (Phase 5, complete).** The Phase 5 validation verified all 172
granules: identical dimensions ``(lat=1, lon=1, time=3)``, identical grid
cell, zero unit mismatches, zero out-of-range values, zero missing values,
and a complete 172-day coverage with no gaps. The naming now states the
truth, and the physical cross-validation against the CPCB station confirmed
the values: T2M correlates **+0.9728** with the station's own air
temperature (mean difference +0.96 C), QV2M correlates +0.6038 with the
station's implied specific humidity, and SLP minus PS gives +23.32 hPa, an
implied model elevation of **185 m**, consistent with Faridabad's ~200 m.

The remaining finding is a *coverage* limitation rather than a defect: the
three UTC timesteps are **17:00, 18:00 and 19:00 IST**, an evening window
rather than the Indian midday, so the sample is offset from the diurnal
pollution peak. The MERRA-2 group therefore remains **disabled**: 172 days of
a 3,288-day record gives only 5 matched test rows, and no conclusion about
its predictive value is possible.

---

## D-011 — Feature importance is dominated by the month term

| | |
|---|---|
| **Severity** | Medium |
| **Status** | VERIFIED |
| **Location** | `results/legacy_chapter/feature_importance.csv` |
| **Evidence** | `month_sin` 37.93%, `month_cos` 6.29% |

**Observed behaviour.** `month_sin` carries **37.93%** of the ensemble's
normalised importance — roughly six times the next feature and more than the
three PM2.5 history features combined (11.22 + 6.19 + 5.85 = 23.26%).

**Impact.** The model is substantially a seasonal climatology with a
persistence anchor, not a physical model of aerosol processes. This is
consistent with the near-zero R2 and with the climatology baseline being
poorly specified (D-005): the model has learned "what does May look like"
rather than "what drives aerosol loading". The physical features (AOD, wind,
temperature) are being used as noise-fillers. Reporting this model as a
multisource environmental model would misrepresent it.

**Proposed correction.** Report importance with the seasonal terms separated
out, and test the hypothesis explicitly: fit a month-only model and compare
it against the full feature set on identical rows.

---

## D-012 — Duplicate CPCB source files across two repository locations

| | |
|---|---|
| **Severity** | Medium |
| **Status** | VERIFIED |
| **Location** | `data/City_wise_raw_data_1Hr_*.csv` (git) vs `ML model/CPCB Data/City_wise_raw_data_1Hr_*.csv` (git) |
| **Evidence** | Byte-identical SHA-256 for 2017-2023 (7 pairs) |

**Observed behaviour.** Seven CPCB files exist twice in the tracked history
at `data/` and at `ML model/CPCB Data/` with **identical SHA-256 digests**.
The current pipeline reads only `data/cpcb/`, so it is not double-counting
today, but the aggregation step in `load_cpcb_daily` sums `valid_hours`
across files and would silently double-count coverage if both directories
were ever passed in.

**Impact.** A latent double-counting hazard in the aggregation contract, and
wasted repository space. The plan explicitly requires a "duplicate-hour test"
and warns against "a mean of file-level daily means when files contain unequal
numbers of hourly readings".

**Proposed correction.** Deduplicate to a single canonical location, and add
an aggregation test that asserts no hour is counted twice across input files.

---

## D-013 — Mean-of-file-means aggregation is structurally unsafe

| | |
|---|---|
| **Severity** | Medium |
| **Status** | **FIXED (Phase 1)** |
| **Location** | `src/aerosphinx/ingestion/cpcb.py` → `aggregate_daily` |
| **Evidence** | `results/baseline/data_audit.json` → `cpcb.mean_of_means_vs_hour_weighted_mismatch_days` |

**Observed behaviour.** The loader aggregates each file to daily means
first, then averages those per-file daily means across files by date. This
is the exact pattern the plan warns against. On the **current** data it
produces **0 mismatching days out of 2979** against a true hour-weighted
daily mean (max absolute difference 1.1e-13), because each date appears in
exactly one file.

**Impact.** No numerical impact today, but the correctness of the daily series
depends on a property of the input layout (one file per year, no overlap)
rather than on the algorithm. Adding a single overlapping file — including
the duplicate files noted in D-012 — would silently change every affected
daily mean.

**Proposed correction.** Aggregate at the hourly level across all files, then
compute one daily mean per date. Keep the file-level provenance separately so
the aggregation remains auditable.

**Resolution applied (Phase 1).** `aggregate_daily` now pools every unique
hourly observation across all files and computes one hourly-weighted mean per
date. The result on the real record is **identical to the legacy output**
(max absolute difference 0.0 over all 2,979 days), confirming the correction
is behaviour-preserving on data whose files do not overlap — but it is now
correct by construction rather than by accident of file layout.

Proof tests: `tests/unit/test_cpcb_ingestion.py::test_daily_mean_is_hour_weighted_across_files`
(a 20/4 hour split that the mean-of-file-means would score 55.0 and the
hour-weighted mean scores 25.0) and
`::test_aggregation_is_independent_of_file_split`.

---

## D-014 — Legacy pipelines, notebooks and documentation are absent from the worktree

| | |
|---|---|
| **Severity** | Medium |
| **Status** | VERIFIED (recoverable from git history) |
| **Location** | 66 tracked paths deleted in the working tree |

**Observed behaviour.** The working tree deletes **407** tracked paths. 341
of them are byte-identical to files relocated under `data/`
(`data/aod`, `data/merra2`, `data/cpcb`). **66** exist nowhere on disk; of
those, 11 are exact duplicates of another tracked path and 55 are unique,
recoverable only from commit `859fd08`. The unique set includes every legacy
Python pipeline and notebook: `legacy/scripts/run_research_pipeline.py` (662 lines),
`ML model/ML_Model.py`, `ML_Model.py`, `pawanai-project/pawanai.py` (1106
lines), `Data_Reading.ipynb`, plus chapter drafts, `.pbix`/`.docx`/`.pptx`
artefacts and the original `legacy/results/audit_report.md`.

**Impact.** The plan's repository-preservation section requires that legacy
results and model artefacts be preserved, and states "do not delete, overwrite,
push or rewrite history without explicit permission". Committing the current
working tree as-is would remove these from the active tree. They remain
recoverable from history, but a `git checkout` of a future commit would make
them harder to find, and the audit trail between the legacy pipeline and the
current one would be lost.

**Proposed correction.** Before the restructuring is committed, restore the
legacy source artefacts into a `legacy/` directory as the plan's architecture
specifies, so the experimental history remains inspectable. The 341 relocated
data files should be committed as renames, which git detects automatically.

**Resolution applied (Phase 0).** The 50 genuinely unique artefacts were
recovered from `859fd08` and restored under `legacy/`, preserving their
original relative paths, and verified byte-identical to the baseline blobs.
The 341 relocated data files are committed as renames (399 renames detected).
A final integrity check confirms **0 content loss** across all 407 removed
HEAD paths. `legacy/README.md` documents the provenance and the rules
governing the directory.

---

## D-015 — CRLF line-ending conversion in the relocated CPCB files

| | |
|---|---|
| **Severity** | Low |
| **Status** | VERIFIED |
| **Location** | `data/cpcb/*.csv` vs git `data/City_wise_raw_data_1Hr_*.csv` |

**Observed behaviour.** The nine relocated CPCB files use CRLF line endings
while the tracked originals use LF. After normalising CRLF to LF the two sets
are **byte-identical for all nine files**, confirming no data alteration.

**Impact.** None on parsing — pandas handles both. It does mean the raw
digests differ from the historical ones, which matters for manifest
provenance. The Phase 0 manifest records the current digests precisely so
this is traceable.

**Proposed correction.** None required. Note the convention in the data
dictionary so future hash comparisons are not misread as data changes.

---

## D-016 — Operational forward forecast extrapolates an upward drift

| | |
|---|---|
| **Severity** | Low |
| **Status** | VERIFIED |
| **Location** | `results/legacy_chapter/operational_forward_forecast.csv` |
| **Evidence** | 96.6 -> 112.3 -> 122.2 -> 129.6 -> 138.5 ug/m3 |

**Observed behaviour.** The forward forecast rises monotonically across all
five days, ending at 138.5 ug/m3 ("Very Poor"). The driver is the recursion:
each prediction re-enters the model as `PM25_Lag_1` and, with the 0.4
persistence anchor, compounds the error in one direction.

**Impact.** The persistence anchor, selected on the validation window, is
applied unchanged at every horizon of a five-day recursion, where its
optimal value would very likely differ. The result is presented as an
operational outlook, and the anchor was tuned for a **one-step** horizon.

**Proposed correction.** Treat the anchor as a per-horizon parameter, or
disable it for horizons beyond one. Label the output clearly as a
persistence-dominated projection, not a forecast, until real future
covariates are available.

---

## D-017 — A baseline defect silently shrank the headline evaluation set

| | |
|---|---|
| **Severity** | High |
| **Status** | **FIXED** (Phase 3 audit) |
| **Location** | `src/aerosphinx/models/baselines.py`, `src/aerosphinx/tracks/track_a.py` |

**Observed behaviour.** The Phase 2 headline set required *every* model,
including the seasonal-naive baseline, to be defined. The baseline's
climatology was built from the most recent year of the training block, and
in the early folds the training block is one partial year, so it covered
only 260 of 366 day-of-years. **409 rows were undefined for the baseline**
and, because the set required all models to be defined, were removed from
the evaluation of **every** model. 15 of 96 folds lost the baseline
entirely.

**Impact.** The evaluation set was 2,150 rows instead of 2,510, and the
removed rows concentrated in exactly the early cold-start folds where the
learned models fail hardest — **understating the cold-start problem**.

**Resolution.** The headline set is now defined by the models under test; a
baseline that cannot predict a row is scored as NaN there and reported as
undefined rather than removing the row. `ClimatologyBuilder` additionally
guarantees complete day-of-year coverage via a per-day-of-year fallback
from older years and a nearest-day fill. Undefined seasonal predictions
fell from 409 rows to 0, and 15 folds were restored.

---

## D-018 — The bootstrap overstated the effective sample size

| | |
|---|---|
| **Severity** | High |
| **Status** | **FIXED** (Phase 3 audit) |
| **Location** | `src/aerosphinx/evaluation/block_inference.py` |

**Observed behaviour.** The Phase 2 bootstrap resampled individual days,
treating them as independent. Daily PM2.5 absolute errors have lag-1
autocorrelation +0.31 and lag-30 +0.16, so the effective sample size is
materially smaller than nominal. Folds are dependent too: test blocks are
contiguous 30-day windows advancing 30 days, so adjacent folds test
adjacent calendar periods and share nearly all their training history.

**Impact.** The intervals were narrower than they should have been. **This
changed a conclusion**: Random Forest versus persistence went from
"significantly worse" (naive p = 0.032) to "not significantly different"
under both moving-block and year-blocked inference, and the Holm-adjusted
p is 0.095.

**Resolution.** `block_inference.py` adds a moving-block bootstrap with a
data-estimated block length, a year-blocked bootstrap, Holm-Bonferroni and
Benjamini-Hochberg corrections, and a fold-level paired comparison. The
block length is estimated from the *difference* series, which is correct for
a paired comparison because the shared target autocorrelation cancels in
the difference.

---

## D-019 — The forecast baseline was not horizon-appropriate

| | |
|---|---|
| **Severity** | Critical |
| **Status** | **FIXED** (Phase 3 Workstream A) |
| **Location** | `src/aerosphinx/models/baselines.py`, `src/aerosphinx/tracks/multi_horizon.py` |

**Observed behaviour.** The multi-horizon backtest compared a day-*h*
forecast against the **day-1** persistence value. The correct baseline for
a day-*h* forecast is the observation *h* calendar days earlier. The
mismatch produced a misleading result: it made the learned models look
consistently *worse* with horizon (delta +2.3 to +12.7), which reads as
"learned models degrade and persistence holds up".

**Impact.** The Phase 3 conclusion was inverted by this defect. Two
diagnostics exposed it: the baseline's apparent error did not grow with
horizon (it fell after h=2, which is impossible for a degrading
baseline), and each horizon forecasts a different target date, so the
comparison confounded horizon difficulty with target-date difficulty.

**Resolution.** `persistence_forecast` gained a `periods` argument and the
multi-horizon runner uses it. With the correct baseline the result
reverses: the learned models beat persistence from **day 2 onward**, and
the within-issue-date comparison confirms the effect on identical forecast
origins.

---

## D-020 — The CPCB barometric pressure column is corrupt

| | |
|---|---|
| **Severity** | High |
| **Status** | **FIXED (Phase 5)** |
| **Location** | `config/data_quality.yaml`, `src/aerosphinx/ingestion/station_met.py` |
| **Evidence** | `results/runs/D-merra-20261113T052451Z/` |

**Observed behaviour.** The Phase 5 physical cross-validation of MERRA-2
against the CPCB station produced an impossible result: a daily correlation
of **-0.214** between the station's `BP` column and the MERRA-2 sea-level
pressure at the same site. Three independent checks confirm the column is
unreliable:

1. **Impossible magnitudes.** 107 of 3,288 days have a daily mean above
   800 mmHg, reaching **1062 mmHg = 1416 hPa** — above sea-level pressure
   *at sea level*, so impossible for any station. Concentrated in 2023-01/02
   and 2024-07 to 2024-11.
2. **Anti-correlation with the reanalysis.** Daily correlation of -0.214 with
   MERRA-2 SLP and -0.212 with MERRA-2 surface pressure.
3. **An implausible diurnal phase.** The column's daily maximum falls at
   **10h** and its minimum at 17h. Surface pressure at a fixed elevation
   must be highest near dawn and lowest in the afternoon. The station's own
   `AT` column is correctly phased (max 14h, min 05h), and MERRA-2's fields
   show the expected evening behaviour.

The other eight station columns pass every screen, with zero implausible
days.

**Impact.** The corrupt values are extreme outliers. They were supplied to
the model as a feature in the Workstream C station-meteorology experiment.
I tested whether they caused the reported harm and **the hypothesis was
wrong**: the 22 corrupt-BP dates in the evaluation set carry only 1.4% of the
total absolute error, because they are low-pollution late-2024 days that
are easy to predict. Removing the column nevertheless changes the result
materially: the station-meteorology group's count of *significantly worse*
comparisons falls from **2 to 0**, and its best delta improves from +3.08 to
+2.13. The group goes from significantly harmful to merely neutral. The null
conclusion — no demonstrated benefit — is unchanged.

**Resolution.** The `BP` column is screened (a plausibility range is now
declared for all nine station columns) and excluded from the
`station_met` feature group. The Workstream C run was re-executed with the
corrected feature set; the prior run is preserved at
`results/runs/C-regime-20261112T201612Z/`.

**Open.** The diurnal phase inversion suggests a *transformation* applied to
the column rather than sporadic instrument faults, so the same treatment may
affect other CPCB exports. See B-039 and B-040.

---

## Open questions requiring a decision (BLOCKED)

| ID | Question | Why it is blocked |
|---|---|---|
| B-001 | Is a boundary-layer-height product obtainable for 2024? | Requires a MERRA-2 re-subset with `PBLH`; needs network access and a data request. |
| B-002 | Can INSAT-3D granules for 2016-2023 be obtained? | Without them the fused dataset stays confined to 2024 and Track B cannot be evaluated on the full CPCB record. |
| B-003 | Are the 27 zero-pixel AOD granules retrievable at higher temporal resolution, or are they genuinely cloud-flagged? | The distributed granules contain no QA or cloud-flag dataset, so this cannot be determined from local files. |
| B-004 | Should the chapter draft, `.pbix` and `.docx` artefacts be retained in the repository? | These are deliverables, not research inputs; retention is a project decision. |
| B-005 | Is the `.pbix`/PowerBI work part of the research record or an application artefact to be isolated? | Affects whether it belongs in `legacy/` or in a separate application area, per the plan's `pawanai-project` guidance. |
| B-006 | What is the intended definition of "availability-aware" fallback behaviour? | The plan describes it conceptually but does not specify the fallback policy, so implementation would be speculative. |
| B-007 | What is the provenance of the root-level legacy CSV files (`aod_data_1..3.csv`, `combined_pm2.5_para*.csv`, `pm25_cpcb_05AM_*.csv`, `predicted_pm25_jan_to_dec_2019.csv`)? | The plan requires determining their provenance before moving or deleting them. They are preserved under `legacy/` unmodified so the question can be answered later from the data itself; answering it needs upstream context that is not in the repository. |
| B-008 | Should the 124 Track A rows without a complete 1-day lag be dropped, or handled with gap-skipping lags? | Changes the usable Track A dataset size by roughly 4%. Phase 2 must choose and report the choice. |
| B-009 | Should a day carrying a duplicated hour be excluded entirely, or resolved by a documented precedence rule? | No effect on the current record (no duplicate hours exist). Matters only if overlapping CPCB files are ever supplied. The current policy excludes the day. |
| B-010 | Can the provenance and QA conventions of the CPCB export itself be established, particularly for the station-local meteorological columns? | **PARTIALLY RESOLVED by the Phase 3 Workstream D audit.** The nine columns are confirmed genuine hourly station-local measurements, not reanalysis. What remains unknown is the **availability timestamp**, so none is forecast-usable yet. See B-021 and B-022. |
| B-011 | Should the cold-start folds (2017) be reported as a headline result, or excluded as an unrepresentative warm-up? | **DECIDED: keep them.** All 96 folds are retained in every headline metric; in-range and extrapolating folds are reported as separate diagnostics. |
| B-012 | Is a regime-conditional model - persistence in monsoon, a learned model otherwise - in scope? | **DECIDED: approved as an experiment, not as the primary model.** A regime-aware fallback was tested in Workstream B and is the only mitigation that worked; the validated routing belongs to Workstream C. |
| B-013 | Should the regime split (winter/pre-monsoon vs the rest) be re-derived on validation folds? | **ANSWERED in Workstream C.** The routing architecture's regime assignment is calendar-only and its specialists are selected on validation folds with a minimum-sample threshold. The seasonal *split* itself is a fixed calendar definition, not a fitted parameter, so it does not require re-derivation. |
| B-014 | Is a continuous regime weighting better than a hard split? | Open. A blend of model and persistence by a season-dependent weight is the natural alternative to a hard split. |
| B-015 | Can the cold-start *coverage* failure be addressed directly? | Open. The failure is that the tree ensembles cannot reach the low-pollution regime, so weighting the training set toward rare regimes is a candidate. |
| B-016 | Does the regime-aware fallback still help at multi-day horizons? | **ANSWERED in Workstream A: no.** The learned models beat persistence from h=2 onward using ground history alone, so a persistence fallback would remove skill rather than add it. |
| B-017 | Is the multi-horizon advantage driven by the calendar features or the PM2.5 history? | Open. An ablation would separate them and determines how hard the task facing environmental covariates is. |
| B-018 | Should day-1 and multi-day use separate models? | **PARTIALLY ANSWERED in Workstream C.** The shared multi-horizon model wins at h=2, 4 and 5; the regime-conditioned models win at h=1 and h=3. A single shared model is not optimal, and the selection is made per horizon. |
| B-019 | Does the multi-horizon advantage survive the nine-year record's heterogeneity? | Open. The year-on-year MAE trend remains confounded with the measurement change identified in Phase 0. |
| B-020 | Can recursive forecasting be salvaged with per-horizon damping? | **ANSWERED in Workstream C: no.** Direct beats recursive at every horizon beyond 1; the recursive model degrades from MAE 24.57 to 66.93 against the direct model's 36.24 at h=5. |
| B-021 | Does CPCB publish hourly data with a timestamp, and what is the publication latency relative to the observation hour? | **Blocks making the station-local columns forecast-usable.** This is a CPCB data-provenance question. The columns are preserved, retrospective-only, until answered. |
| B-022 | Is a declared one-hour publication lag acceptable for retrospective estimation, recorded as an assumption rather than a verified fact? | **DECIDED: retrospective-only.** The Phase 3 authorisation ruled that no publication lag may be assumed and no availability timestamp may be fabricated. |
| B-023 | Which of the nine audited station-local variables carry genuine PM2.5 signal? | **ANSWERED in Workstream C: none, as a group.** The nine-column group is better in 7 of 25 direct comparisons and worse in 18, with 0 significantly better after Holm correction. Individual columns remain untested (B-029). |
| B-024 | Should `BP` (22.4% missing) and `VWS` (43.3% missing) be excluded on completeness grounds before any modelling? | **ANSWERED in Phase 5: `BP` excluded, for a stronger reason.** It fails three independent physical checks (see D-020). `VWS` remains included; it passes its screens. |
| B-038 | Can MERRA-2 granules be obtained for 2016–2023 (B-002)? | Without them the group stays untestable: 172 days of a 3,288-day record gives 5 matched test rows. |
| B-039 | Is the station `BP` corruption a localised instrument fault or a systematic transformation? | The diurnal phase inversion points to a transformation, so the same treatment may affect other CPCB exports. |
| B-040 | Should every source be cross-validated against every other source? | The `BP` defect was found by cross-validating MERRA-2 against the station — a different group from the one being validated. Worth doing systematically. |
| B-041 | Should the MERRA-2 subset be re-requested with full 24-hour coverage and an 08:00–14:00 UTC window (13:30–19:30 IST) that brackets the Indian midday? | The current window samples 17:00–19:00 IST, an evening window offset from the pollution peak. |
| B-042 | With `BP` excluded, does the station-meteorology group help under a tree model rather than the linear one? | Open. The Workstream C experiment used ridge throughout. |
