# AeroSphinx — Phase 3 Experimental Protocol

**Status: protocol written before the Phase 3 experiments are run.**
Approved by the Phase 3 authorization. This document fixes the statistical
methodology in advance so that no methodological choice can be made after
seeing the results.

---

## 1. Revised research question

> **Under what atmospheric regimes and data-availability conditions can
> environmental information improve PM2.5 predictions beyond historical
> persistence, particularly across multi-day horizons and distribution
> shifts?**

This supersedes the Phase 0 framing. Ground-history saturation does **not**
prove that satellite fusion will help; it establishes that environmental
predictors deserve investigation.

### Research problems

| # | Problem | Status |
|---|---|---|
| RP-1 | **Persistence saturation.** No learned model demonstrates significant aggregate one-day improvement over persistence. | Established (Phase 2, corrected) |
| RP-2 | **Distribution-shift vulnerability.** Tree ensembles fail when the test target range exceeds the training target range. | Established (Phase 2) |
| RP-3 | **Environmental and regime-dependent value.** Meteorology, satellite and regime-aware models may add value where ground-history models struggle. | **Open — the hypothesis under test** |

---

## 2. Phase 3 audit corrections (already applied)

See `docs/reports/PHASE3_AUDIT.md`. Three corrections were applied **before** any
Phase 3 experiment ran.

| ID | Correction | Effect |
|---|---|---|
| D-016 | The headline evaluation set is defined by the **models under test**, not by whether an auxiliary baseline can predict. | Evaluation set grew from 2,150 to **2,510** rows |
| D-016/017 | The climatology covers every day-of-year, with a per-day-of-year fallback from older years. | Undefined seasonal predictions fell from **409 rows to 0**; 15 folds that lost the baseline entirely are restored |
| D-018 | Moving-block and year-blocked bootstrap replace the naive day-level bootstrap; Holm and Benjamini-Hochberg corrections added. | **Changed a conclusion** — see §4 |

### The superseded Phase 2 baseline

| Model | Phase 2 MAE (2,150 rows) | Corrected MAE (2,510 rows) |
|---|---|---|
| Persistence | 21.411 | 21.831 |
| Seasonal naive | 37.241 | 38.012 |
| Ridge | 21.511 | 22.175 |
| Random Forest | 21.398 | 22.985 |
| XGBoost | 22.376 | 24.519 |

The Phase 2 evaluation set had removed exactly the rows where the learned
models fail hardest. The corrected set is larger and the learned models
score worse on it — **the audit strengthened the Phase 2 conclusion rather
than weakening it.** The audited run is preserved at
`results/runs/A-roll20261112T142317Z/` and is not overwritten.

---

## 3. Workstream A — Multi-horizon forecasting (days 1–5)

### Design

For each horizon *h* in 1…5, and for each of the 96 folds, predict PM2.5
for the day *h* days after the fold's test-block origin.

**Two methods are compared:**

| Method | Description |
|---|---|
| **Direct** | A separate model per horizon, trained on the fold's training block with the target shifted *h* days. |
| **Recursive** | A single one-day model whose predictions are fed back as later lag inputs. |

**Horizon-appropriate baseline.** Persistence at horizon *h* is the observed
value *h* calendar days before the target. This is the correct reference:
persistence at horizon 1 is not the right baseline for a day-5 forecast.

### Feature contract

At horizon *h*, a model may use:
- PM2.5 observed on or before the **forecast issue date** (the last day of
  the fold's training block);
- calendar features of the target date;
- environmental covariates **only if genuinely available at issue time**.

Track A has no environmental covariates, so the feature set is the
calendar-aware PM2.5 history plus calendar features. Track B covariates
enter only under the Phase 3 Workstream D rules.

### Reported per horizon

MAE, RMSE, mean bias, R² (supplementary, with observed SD), 95% confidence
intervals, and skill relative to the horizon-appropriate persistence
baseline, with the paired test, block-aware interval and fold-level
consistency.

### Hypothesis under test

> Persistence degrades with horizon. Learned models gain relative value at
> longer horizons.

This is a **hypothesis to test, not an assumption**. A null result — that
learned models never gain — is a valid outcome.

---

## 4. Workstream B — Cold-start and extrapolation diagnostics

### Diagnostic report

Per fold: training target minimum and maximum; number and fraction of test
observations outside the training range; MAE conditional on extrapolation;
MAE conditional on in-range prediction; error as a function of the
proportion of out-of-range test observations.

Results are reported in three groupings, per the B-011 decision:

1. **All 96 folds** — the primary result.
2. **In-range folds** — diagnostic.
3. **Extrapolating folds** — diagnostic.

No fold is removed from the primary result.

### Candidate mitigations to be tested

| # | Strategy | Rationale |
|---|---|---|
| 1 | **Persistence fallback** | Use the learned model only when its prediction lies inside the training target range; otherwise fall back to persistence. |
| 2 | **Ridge with standardised features** | A linear model extrapolates beyond the training range rather than being bounded by it. |
| 3 | **Quantile-calibrated fallback** | Predict a conditional quantile and compare against the empirical distribution of the training target. |
| 4 | **Regime-aware fallback** | Use persistence in regimes where it is known to be strong (monsoon), the learned model elsewhere — supported by the Phase 2 per-season result. |

**Clipping is explicitly excluded as a primary fix.** Clipping a prediction
to the training range bounds extreme outputs but can make the
low-pollution extrapolation problem worse, by replacing a wrong high
prediction with a wrong low one.

### Statistical methodology

- **Moving-block bootstrap** with the block length estimated from the data
  (the smallest lag from which the autocorrelation of the *difference* series
  stays below 0.05 for all subsequent lags).
- **Year-blocked bootstrap** resampling whole calendar years — the most
  conservative structure available.
- **Holm-Bonferroni** correction across the family of comparisons, with
  Benjamini-Hochberg reported alongside.
- **Fold-level** paired differences reported alongside the pooled result, so
  a sign flip across folds is visible.

---

## 5. Workstream C — Regime-conditional modelling

### Regimes

Four meteorological seasons: winter, pre-monsoon, monsoon, post-monsoon.

**Regime assignment uses only information available at prediction time.** The
regime of a target date is a deterministic function of its calendar month,
which is known in advance. **No assignment may use the target value, any
future observation, or any measured pollution quantity.** This is asserted
by a test.

### Models compared

| # | Model |
|---|---|
| C1 | Global model (one model for all regimes) |
| C2 | Seasonal specialists (one model per regime) |
| C3 | Global model with seasonal interaction features |
| C4 | Regime-aware routing with a validated global fallback |

### Specialist guard

A specialist is used **only when its training set contains at least
``min_specialist_days`` observations**. Otherwise the global model is used.
The threshold is selected on the validation folds, not chosen by inspecting
test results.

### Selection discipline

Model selection, the routing threshold and any hyperparameter choice are
made on **chronological validation folds** that precede the test folds.
The final test folds are touched once, after every choice is fixed.

The question is whether regime specialization produces **stable
out-of-sample gains**, not whether it scores better in the seasons where it
is already known to work. Consistency across folds is therefore reported.

---

## 6. Workstream D — CPCB station-local meteorology audit

No model may be trained on a station-local column until it has been audited.
The audit produces one record per variable:

```
variable
source_column
physical_meaning
unit
station_or_source
measurement_timestamp
availability_timestamp
missing_percentage
valid_range
quality_flag
forecast_usable
```

**The distinction between measurement time and availability time is
essential.** A meteorological measurement recorded for 10:00 but uploaded at
16:00 cannot be used in a forecast issued at noon. Where the availability
time cannot be established from the local files, `availability_timestamp`
is recorded as **UNKNOWN** and `forecast_usable` is **false**.

**Comparison order**, as authorized:

1. CPCB station-local meteorology
2. Historical MERRA-2 meteorology
3. INSAT AOD
4. Combined environmental feature groups
5. Availability-aware model routing

Station-local and MERRA-2 are compared as **separate feature groups**
first, and only combined afterwards.

---

## 7. Preserved findings

The following remain true and must not be re-litigated or quietly dropped:

- **96 folds, all retained**, including the cold-start folds (B-011).
- The legacy 33-day benchmark is **separately identified** and never
  replaces the full-history result.
- The legacy pipeline `aerosphinx_pipeline.py` is **untouched**.
- Existing metrics and outputs are **preserved**; every new experiment is
  written to a **new run directory with a new run ID**.

---

## 8. Forecast information contract

- Future observed AOD and future observed MERRA-2 are **never** used in
  operational forecast mode.
- Every forecast records explicit feature availability timestamps.
- Direct and recursive multi-horizon methods are both reported.
- Missing or unsupported future covariates are **reported**, never
  silently filled.

---

## 9. Acceptance criteria

| # | Criterion |
|---|---|
| 1 | Day 1–5 evaluation with horizon-specific baselines |
| 2 | All 96 folds retained |
| 3 | Cold-start performance independently quantified |
| 4 | Block-aware uncertainty and corrected statistical comparisons |
| 5 | Seasonal specialists compared against a global model |
| 6 | Regime routing has a tested fallback |
| 7 | CPCB station-local meteorological columns documented |
| 8 | Station-local versus MERRA-2 experiments separated |
| 9 | No future environmental observations used operationally |
| 10 | Every experiment reproducible from its run manifest |

---

## 10. Execution order

The protocol is fixed before any experiment runs. Work proceeds one
workstream at a time, and evidence is reported after each:

1. Workstream B (cold-start diagnostics) — depends only on the corrected
   Phase 2 infrastructure.
2. Workstream A (multi-horizon) — the primary experiment.
3. Workstream C (regime-conditional) — depends on A's feature contract.
4. Workstream D (station-local meteorology) — depends on A's
   forecast-information contract.

**No methodological choice is made after seeing the results.** If a
methodological parameter must be adjusted during the work, the adjustment
is made on validation folds and recorded in the run manifest with the
reason.
