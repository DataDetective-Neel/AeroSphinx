# AeroSphinx — Phase 2 Report: Track A Feature Engineering and Baselines

**Phase 2 complete.** Rolling-origin backtest over the full nine-year CPCB
record, with paired uncertainty on every comparison. Awaiting approval to
begin Phase 3 (Track A forecasting and backtesting refinement).

Reproduce:

```bash
make install-dev && make test
python scripts/build_track_a_dataset.py
python scripts/run_track_a.py
```

---

## 1. Headline result

> **On the full nine-year CPCB record, no learned model significantly beats
> the persistence baseline.** Random Forest matches it (Δ MAE −0.013,
> p = 0.86), ridge matches it (Δ MAE +0.101, p = 0.52), and XGBoost is
> significantly worse (Δ MAE +0.965, p = 0.018).

This is a **negative result**, and it is the correct scientific finding for
this data. It is reported here rather than buried, because it directly
answers Track A's question — *"how much predictive value do temporal patterns
and environmental covariates add over simple historical persistence?"* — and
the answer on the current data is: **very little, at one-day horizon.**

It also revises the Phase 0 picture. The legacy pipeline's headline
(MAE 17.30 vs persistence 19.92 on a single 33-day window) looked like a
modest improvement; under a 96-fold backtest with paired statistics, that
apparent edge does not survive.

---

## 2. What was built

| Component | Path | Purpose |
|---|---|---|
| Evaluation metrics | `src/aerosphinx/evaluation/metrics.py` | MAE/RMSE/R²/bias + AQI category accuracy, macro-F1, confusion matrix |
| Rolling-origin splits | `src/aerosphinx/evaluation/splits.py` | Expanding-window backtesting with leakage assertions |
| Uncertainty | `src/aerosphinx/evaluation/uncertainty.py` | Paired Wilcoxon + paired bootstrap CI |
| Baselines | `src/aerosphinx/models/baselines.py` | Persistence, climatology with per-fold lookback selection |
| Supervised models | `src/aerosphinx/models/supervised.py` | Ridge, Random Forest, XGBoost |
| Regime diagnostics | `src/aerosphinx/tracks/diagnostics.py` | Cold-start / extrapolation detection |
| Track A runner | `src/aerosphinx/tracks/track_a.py` | The experiment itself |
| CLI | `scripts/run_track_a.py` | Runs the backtest, writes results + manifest |

---

## 3. Test results

```
$ .venv/bin/python -m pytest tests/ -q
86 passed in 3.26s
```

| Suite | Count | Status |
|---|---|---|
| Unit — evaluation (splits, metrics, uncertainty) | 21 | **PASS** |
| Unit — models (baselines, supervised) | 19 | **PASS** |
| Unit — CPCB ingestion (Phase 1) | 13 | **PASS** |
| Unit — calendar-aware lags (Phase 1) | 14 | **PASS** |
| Integration — Track A ingestion (Phase 1) | 13 | **PASS** |
| Integration — Track A experiment (Phase 2) | 6 | **PASS** |

---

## 4. The experiment

| | |
|---|---|
| Folds | **96** |
| Test blocks | 30 days each, advancing 30 days |
| Training window | expanding, from 365 days |
| Test rows evaluated | **2,150** |
| Period covered | 2017-01-30 to 2024-12-31 |
| Models | persistence, seasonal-naive, ridge, Random Forest, XGBoost |
| Uncertainty | paired Wilcoxon + 10,000-sample paired bootstrap, seed 42 |

**Every model is scored on identical rows within every fold** — the
requirement the legacy single-window evaluation could not meet. Rows where
the baseline is undefined (no previous calendar day) are excluded from all
models equally; 2,682 rows were predicted, 2,150 evaluated.

---

## 5. Results

### 5.1 Aggregate

| Model | n | MAE | RMSE | R² | Bias | AQI acc. | AQI macro-F1 |
|---|---|---|---|---|---|---|---|
| **Persistence** | 2150 | **21.411** | 32.027 | 0.713 | +0.04 | **0.562** | **0.544** |
| Seasonal naive | 2150 | 37.241 | 51.856 | 0.248 | +3.75 | 0.385 | 0.306 |
| Ridge | 2150 | 21.511 | **31.227** | 0.727 | +0.06 | 0.538 | 0.435 |
| **Random Forest** | 2150 | **21.398** | **31.076** | **0.730** | +1.82 | 0.558 | 0.503 |
| XGBoost | 2150 | 22.376 | 32.481 | 0.705 | +1.80 | 0.543 | 0.494 |

### 5.2 Paired comparisons against persistence

| Model | Δ MAE | Wilcoxon p | Bootstrap 95% CI | Verdict |
|---|---|---|---|---|
| Seasonal naive | +15.830 | 9.1e-87 | [+14.32, +17.35] | **worse** (significant) |
| Ridge | +0.101 | 0.517 | [−0.567, +0.765] | **no significant difference** |
| Random Forest | −0.013 | 0.856 | [−0.640, +0.601] | **no significant difference** |
| XGBoost | +0.965 | 0.018 | [+0.241, +1.685] | **worse** (significant) |

### 5.3 Stability across folds

| Model | MAE mean | MAE sd | MAE max | median fold R² | folds with R² < 0 |
|---|---|---|---|---|---|
| Persistence | 22.20 | 11.72 | 54.05 | +0.054 | 36/81 |
| Ridge | 22.41 | 11.91 | 53.99 | +0.074 | 32/81 |
| Random Forest | **22.06** | **11.08** | 55.51 | +0.073 | **31/81** |
| XGBoost | 23.20 | 11.97 | 51.84 | +0.033 | 36/81 |

### 5.4 By season (MAE)

| Season | Persistence | Ridge | RF | XGBoost | Seasonal naive |
|---|---|---|---|---|---|
| Monsoon | 11.18 | 12.03 | 11.85 | 11.63 | 18.42 |
| Pre-monsoon | 19.75 | 18.61 | 18.72 | 20.22 | 33.42 |
| Post-monsoon | 26.83 | 28.18 | 28.19 | 29.89 | 51.44 |
| Winter | 33.68 | 33.23 | 32.86 | 34.41 | 57.60 |

The learned models beat persistence in **pre-monsoon and winter** — the
high-variance, high-pollution seasons — and lose in **monsoon**. That is the
physically expected pattern and a point in favour of the models; it is simply
not large enough to move the nine-year aggregate past significance.

### 5.5 By year (MAE)

| Year | Persistence | Ridge | RF | XGBoost | Seasonal naive |
|---|---|---|---|---|---|
| 2017 | 27.38 | 34.88 | 32.48 | 34.72 | 70.27 |
| 2018 | 26.26 | 25.88 | 25.55 | 27.47 | 48.67 |
| 2019 | 24.13 | 26.70 | 26.34 | 26.98 | 39.49 |
| 2020 | 20.20 | 19.64 | 20.15 | 22.15 | 39.04 |
| 2021 | 23.72 | 23.04 | 22.20 | 22.91 | 36.77 |
| 2022 | 20.23 | 19.96 | 20.08 | 20.47 | 29.02 |
| 2023 | 17.88 | 17.54 | 17.67 | 18.25 | 32.00 |
| 2024 | 18.59 | 18.26 | 18.58 | 19.17 | 33.74 |

**2017 is the outlier**: every model is far worse than persistence. That is
the cold-start effect, quantified below.

---

## 6. Why the learned models fail — and where

The failure is **not** uniform, and diagnosing it was the most valuable part
of this phase.

### 6.1 Regime extrapolation (cold start)

**8 of 96 folds** require the model to predict values outside the range of
its training targets. In those folds the learned models fail badly:

| Model | Δ MAE on extrapolating folds | Δ MAE on in-range folds |
|---|---|---|
| Random Forest | **+17.11** | +0.27 |
| Ridge | +1.52 | +0.48 |
| XGBoost | **+25.25** | +1.49 |

The mechanism is structural. A tree ensemble **cannot predict outside the
range of its training targets**. In fold 5 the training data (383 days from
2016-2017) contained only **3 observations below 30 µg/m³**, so the model's
lowest attainable prediction was 33.4 µg/m³, while the test block
(monsoon 2017) had values down to **13.4 µg/m³**. The model was asked for a
number it had no way to produce.

**Persistence has no such limitation** — it copies yesterday's value,
whatever that value was. That asymmetry is the single reason persistence is
hard to beat, and it is a real property of the problem rather than an
artefact of the experiment.

**Consequence:** the legacy pipeline's single 33-day test window contained
none of these folds. Its result was an accident of *when* it was evaluated.

### 6.2 R² is unstable on low-variance blocks

One fold produced **R² = −186** for Random Forest. The cause is arithmetic,
not a model failure: R² = 1 − SSE/SST, and that fold's test block had an
observed SD of **6.1 µg/m³**. A systematic offset of a few µg/m³ against a
near-zero SST produces an arbitrarily large negative R².

**MAE is the metric to judge a fold on when the test block has low
variance.** This is now documented in the `regression_metrics` docstring and
the per-fold R² column is reported with the observed SD beside it.

### 6.3 The seasonal baseline is now fairly tuned

The D-005 correction must not become a strawman. The climatology lookback is
therefore **selected per fold on an inner validation block** carved from the
training data — no test observation is involved.

Selected lookbacks: 1 year (26 folds), 2 (7), 3 (19), 5 (20), 9 (24) — the
selection shifts from 1 year in the early folds to 9 years later, as the
expanding window makes longer histories available.

With fair tuning, the seasonal baseline improves (MAE 38.08 → 37.24) but
remains **significantly worse than persistence** (Δ +15.83, p = 9.1e-87). Its
bias is still positive (+3.75 µg/m³). So the D-005 correction is real but
does not rescue the baseline — a day-of-year climatology simply cannot
predict day-to-day variability in Delhi-NCR PM2.5.

---

## 7. Two bugs found and fixed during Phase 2

Both were caught by tests or diagnostics written to the protocol.

**Bug 1 — ridge was not standardised.** The feature set mixes calendar
encodings in [−1, 1] with PM2.5 lags in the tens or hundreds. Ridge penalises
coefficient magnitude, so without scaling the penalty fell almost entirely
on the large-scale features. The effect was severe: ridge was **significantly
worse** than persistence (Δ +0.97, p = 0.031) before the fix and **not
significantly different** after (Δ +0.22, p = 0.36), with R² improving from
0.706 to 0.725.

**Bug 2 — missingness was silently imputed.** The calendar-aware lags are
`NaN` exactly where the previous calendar day is absent. That is
information, not an inconvenience. The imputer now records missingness as an
indicator column (`add_indicator=True`), so the model can learn "yesterday
was missing" rather than silently receiving a median value.

Both are now pinned by tests:
`test_ridge_standardises_features`, `test_ridge_records_missingness`,
`test_ridge_imputer_is_fitted_on_training_data_only`.

---

## 8. Defect status

| Defect | Status |
|---|---|
| **D-005** Climatology baseline mis-specified | **FIXED** — per-fold lookback selected on inner validation; the baseline's own bias is now reported |
| **D-006** Improvement over persistence not significant | **ADDRESSED** — every comparison now carries paired statistics and an interval. The finding is confirmed and extended: on the full nine-year record, no model beats persistence significantly |
| **D-011** Feature importance dominated by `month_sin` | **PARTIALLY ADDRESSED** — the seasonal baseline now quantifies how much the seasonal term is worth, and the per-season breakdown shows where the learned models actually help |
| **D-001** Calendar-aware lags | **FIXED** (Phase 1) |
| **D-013** Hour-weighted aggregation | **FIXED** (Phase 1) |
| D-002, D-003, D-004, D-007–D-010, D-016 | OPEN (Phases 4–8) |

---

## 9. Phase 2 gate

| Criterion | Status |
|---|---|
| Track A features implemented | **PASS** — calendar-aware lags, rolling stats, cyclic calendar encodings |
| Persistence baseline defined and computed | **PASS** — strict calendar-aware variant |
| Seasonal-naive baseline defined and computed | **PASS** — with per-fold validation-selected lookback (the D-005 correction) |
| Both baselines computed on identical rows | **PASS** — 2,150 rows, every model |
| Rolling-origin backtest implemented | **PASS** — 96 folds, expanding window |
| Paired uncertainty reported | **PASS** — Wilcoxon + bootstrap CI on every comparison |
| Fold-level stability reported | **PASS** — mean, sd, max, R² distribution |
| Per-season and per-year breakdowns | **PASS** |
| Honest reporting | **PASS** — the negative result is the headline |
| No tuning on held-out data | **PASS** — climatology lookback selected on an inner validation block |
| No new source edits to legacy pipeline | **PASS** — `aerosphinx_pipeline.py` untouched |

**Phase 2 gate: PASSED.**

---

## 10. Interpretation and what it means for the project

**Track A's research question, answered.** How much predictive value do
temporal patterns add over simple historical persistence? On this record, at
one-day horizon: **almost none that can be demonstrated with statistical
confidence.**

That is not a nullity. It is a **constraint that redirects the project**:

1. **Multi-day horizons are where the value must be.** A one-day-ahead
   forecast for a pollutant with a strong persistence signal is close to
   saturated. The interesting question is whether skill survives at day 3–5,
   where persistence degrades but a model with seasonal and meteorological
   context might not. **Phase 3 should test that directly.**

2. **The cold-start asymmetry is itself a research contribution.** Tree
   ensembles cannot extrapolate outside their training range; persistence
   can. For an operational deployment that must handle regime shifts
   (monsoon onset, a policy change, an exceptional dust event), this is a
   real design constraint, not a footnote. The availability-aware framework
   the plan proposes is the natural home for it.

3. **The learned models do help where it matters.** In pre-monsoon and winter
   — the seasons with the highest pollution and the largest error — ridge and
   Random Forest beat persistence. A system that switched between a
   persistence-like model and a learned model by regime would be better
   than either.

4. **Track B is now better motivated, not less.** Track A shows that
   ground-history alone saturates at one-day horizon. If satellite and
   meteorological information is going to add value, it will be through
   *environmental* context rather than through better use of PM2.5 history —
   which is exactly what Track B is designed to test.

---

## 11. Remaining risks

1. **The nine-year record is not homogeneous.** Integer-valued hourly
   readings fall from 96.1% (2016) to 0.3% (2023). The per-year table shows
   MAE falling steadily from 27.4 (2017) to 17.9 (2023) for *every* model,
   including persistence. Some of that is a genuine change in pollution
   regimes; some may be a change in measurement character. **Phase 3 must
   test for distribution shift** before interpreting the year-on-year trend.
   This is the single largest interpretive risk in the current results.

2. **One station.** Nothing supports generalisation beyond Faridabad.

3. **Station-local meteorology is still unused** — blocked on B-010. This is
   now more important: Track A shows ground history alone saturates, so the
   available environmental covariates are the obvious next source of signal.

4. **Hyperparameters are fixed, not searched.** Ridge `alpha`, RF depth and
   XGBoost settings were carried over from the legacy chapter. A per-fold
   validation-based search is Phase 3 work and could change the ordering.

5. **No multi-horizon evaluation yet.** Everything here is one-day-ahead.

---

## 12. Recommended Phase 3 scope

1. **Direct multi-horizon models** (day 1–5), the primary remaining Track A
   experiment. Test whether skill *grows* relative to persistence at longer
   horizons, where persistence degrades.
2. **Recursive forecasting** with per-horizon anchor handling.
3. **Per-fold hyperparameter selection** on the inner validation block.
4. **Distribution-shift diagnostics** across the record, to separate a real
   regime change from a measurement artefact.
5. **Prediction intervals**, so the output carries uncertainty rather than a
   bare point estimate.

---

## 13. Open questions for the reviewer

| ID | Question | Impact |
|---|---|---|
| **B-008** | The 124 rows without a complete 1-day lag: drop them, or use gap-skipping lags? | ~4% of the dataset; the strict choice is currently in force |
| **B-010** | Provenance and QA conventions of the CPCB export's station-local columns | Blocks the most promising source of new Track A signal |
| **B-011** | Should the cold-start folds (2017) be reported as a headline result, or excluded as an unrepresentative warm-up? | Currently reported in full, with the mechanism quantified. Excluding them would favour the learned models, so the honest default is to keep them |
| **B-012** | Is a "regime-conditional" model — persistence in monsoon, learned model otherwise — in scope for the project? | The per-season table suggests it would beat either model alone |
