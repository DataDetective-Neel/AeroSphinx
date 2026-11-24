# AeroSphinx — Phase 3 Audit of the Phase 2 Evaluation

**Required first step of the Phase 3 authorization:** "Begin with an audit of
the current Phase 2 outputs and the exact 96-fold evaluation implementation."

This audit was performed before any Phase 3 experiment was run, and it
**found two defects in the Phase 2 implementation.** Both are recorded here
with their impact and correction. The Phase 2 headline numbers are affected
and are therefore **superseded** by the corrected re-run.

Evidence: `results/runs/A-roll20261112T142317Z/` (the audited run).

---

## 1. Audit checklist

| Item | Result |
|---|---|
| Test blocks disjoint across folds | **PASS** — 0 overlapping blocks, 0 dates in more than one fold |
| Test rows counted correctly | **PASS** — 2,682 predictions = 2,682 distinct dates |
| Training block strictly precedes test block | **PASS** — asserted per fold |
| No future observations in features | **PASS** — features are calendar-lagged or calendar features |
| Model selection free of test data | **PASS** — climatology lookback selected on an inner validation block |
| Every model scored on identical rows | **FAIL** — see D-016 |
| Baseline fully defined on the evaluation set | **FAIL** — see D-017 |
| Uncertainty appropriate to the dependence structure | **FAIL** — see D-018 |

---

## 2. D-016 — A baseline defect silently shrank the headline evaluation set

**Severity: High. Status: OPEN (correction required before Phase 3).**

### Observed behaviour

The Phase 2 headline set was built by dropping every row on which **any**
model was undefined:

```python
complete = predictions.dropna(subset=model_columns + ["Target"])
```

`seasonal_naive` was undefined on **409 rows**. Those rows were therefore
removed from the evaluation of **every** model, including the models that
had a perfectly good prediction for them.

| Exclusion reason | Rows |
|---|---|
| `seasonal_naive` undefined | 409 |
| `persistence` undefined | 172 |
| Rows in the headline set | 2,150 (of 2,682) |

**15 of 96 folds lost the seasonal baseline entirely** (folds 1, 2, 4, 5, 6,
7, 8, 9, 17, 18, 22, 29, 34, 71, 74) — 344 test rows in total.

### Root cause

`ClimatologyBuilder` with `lookback_years=1` restricts the climatology to
the most recent year present in the training block. In the early folds the
training block is a single **partial** year (fold 0: 2016-01-01 to
2016-12-30, 260 rows), so the climatology covers only **260 of 366
day-of-years**. The 106 uncovered day-of-years fall outside the training
year, and with `max_day_offset=3` they could not be reached either.

Fold 0 escaped this by luck — its test window (January-February) happened to
avoid the uncovered day-of-years. The next folds did not.

### Impact

1. **The headline evaluation set was 20% smaller than it should have been**,
   and the removed rows were not missing at random: they concentrated in
   early folds with short training histories, which is precisely where the
   cold-start effect is strongest. **The audited Phase 2 result therefore
   understated the cold-start problem.**
2. **The seasonal baseline was scored on a biased subset** — only the
   day-of-years its climatology happened to cover.
3. The requirement that "every model is scored on identical rows" was
   satisfied only by discarding rows on which one baseline was undefined.

### Correction required

1. `ClimatologyBuilder` must guarantee **complete day-of-year coverage**:
   if the selected lookback cannot cover all 366 day-of-years, the lookback
   must be extended rather than leaving days unpredicted.
2. The headline evaluation set must be defined by the **models under test**,
   not by whether an auxiliary baseline can predict. A baseline that cannot
   predict a row must be scored as NaN on that row and reported as
   "undefined on N rows", not used to remove the row from every other
   model's evaluation.

---

## 3. D-017 — The climatology baseline could be structurally undefined

**Severity: Medium. Status: OPEN (correction required before Phase 3).**

### Observed behaviour

`ClimatologyBuilder.fit` selects years by `cutoff_year = years.max() -
lookback_years + 1`. With a partial-year training block this silently
produces an incomplete climatology, and `ClimatologyBuilder.predict` then
returns NaN for uncovered day-of-years.

The baseline's *own* completeness was never asserted. `select_lookback`
scored candidate lookbacks on an inner validation block, and a candidate
that produced NaN predictions simply scored as `inf` — which is correct for
*selection* but means a **1-year lookback can be selected and then produce
NaN on the test block**, because the inner validation block and the test
block cover different day-of-years.

### Correction required

`ClimatologyBuilder.fit` must verify coverage and extend the window until
every day-of-year in the training span is covered, or raise. `predict` must
report the fraction of day-of-years it can serve.

---

## 4. D-018 — The uncertainty estimate overstated the effective sample size

**Severity: High. Status: OPEN (correction required before Phase 3).**

### Observed behaviour

The Phase 2 bootstrap resampled **individual days** with replacement,
treating them as independent. The daily absolute errors are **serially
correlated**:

| Lag (days) | Autocorrelation of daily absolute error |
|---|---|
| 1 | +0.306 |
| 2 | +0.242 |
| 3 | +0.256 |
| 7 | +0.170 |
| 14 | +0.139 |
| 30 | +0.158 |

For a first-order autoregressive process with correlation ρ, the effective
sample size is approximately `n(1-ρ)/(1+ρ)`:

| Assumed ρ | Effective n | Nominal n |
|---|---|---|
| 0.5 | ~837 | 2,510 |
| 0.7 | ~443 | 2,510 |
| 0.9 | ~132 | 2,510 |

The folds themselves are **not** independent either: test blocks are
contiguous 30-day windows advancing 30 days, so adjacent folds test
**adjacent calendar periods**, and every fold after the first shares almost
all of its training history with its neighbours.

### Impact

The Phase 2 intervals are **narrower than they should be**. A
significant result reported from the naive bootstrap may not survive a
block-aware interval, and a non-significant result may become significant in
the other direction. The Phase 2 conclusions are therefore stated with
*less* confidence than the reported intervals alone would suggest.

### Correction required

The Phase 3 protocol must:

1. Replace the naive bootstrap with a **moving-block bootstrap** that
   resamples contiguous blocks of days, with the block length chosen to
   exceed the observed autocorrelation length.
2. Add a **year-blocked bootstrap** that resamples whole years.
3. Apply a **multiple-comparison correction** (Holm or Benjamini-Hochberg)
   across the family of model comparisons, since Phase 3 will compare many
   models at many horizons.
4. Report the **fold-level** paired test in addition to the pooled one, so
   the dependence structure is visible rather than assumed away.

---

## 5. R² is a supplementary diagnostic, not a headline metric

The Phase 2 report showed one fold with R² = −186 for Random Forest. The
explanation given — a low-variance test block inflates the penalty — is
correct, but the metric should not have been prominent at all.

**From Phase 3 onward, MAE and RMSE are reported prominently and R² is
supplementary**, always accompanied by the observed standard deviation of the
test block so the reader can see when it is unstable. This is a reporting
convention, not a change to the metric definition.

---

## 6. What the audit did *not* find

The following were checked and are sound:

- **No temporal leakage.** Every fold's training block ends before its test
  block begins; this is asserted programmatically for every fold.
- **No test-set tuning.** The climatology lookback was selected on an inner
  validation block carved from the training data.
- **No duplicated test dates.** 2,682 predictions over 2,682 distinct dates.
- **No future environmental observations.** Track A uses only calendar-lagged
  PM2.5 and calendar features.
- **Raw data and legacy artefacts untouched.** 350 raw files and 50 legacy
  artefacts remain byte-identical to the Phase 0 baseline.

---

## 7. Consequence for the accepted Phase 2 baseline

The Phase 2 headline numbers (persistence MAE 21.411, RF 21.398, ridge
21.511, seasonal 37.241, XGBoost 22.376) were computed on a **20%-reduced**
evaluation set that was **biased against the early cold-start folds**.

**The qualitative conclusion is unlikely to change** — the learned models
did not beat persistence, and shrinking the set removed exactly the folds
where they failed hardest, so the corrected set may make that conclusion
*stronger*. But the numbers themselves must be re-derived, and Phase 3
proceeds on the corrected figures.

Per the authorization's preservation requirement, the audited run is **kept
intact** at `results/runs/A-roll20261112T142317Z/` and is not overwritten.
The corrected run is written to a new directory with a new run ID.

---

## 8. Phase 3 corrections to implement before new experiments

| ID | Correction | Workstream |
|---|---|---|
| D-016 | Define the headline set from the models under test, not from auxiliary baseline availability | B |
| D-016 | Guarantee complete day-of-year coverage in the climatology | B |
| D-017 | Assert climatology completeness at fit time | B |
| D-018 | Moving-block and year-blocked bootstrap | A, B, C |
| D-018 | Multiple-comparison correction across the model family | A, B, C |
| D-018 | Report fold-level paired tests alongside pooled | A, B, C |
| §5 | MAE and RMSE prominent; R² supplementary with observed SD | A, B, C |
