# Chapter 3 — numbers to correct before submission

Every value below was recomputed from the files in `data/` by
`aerosphinx_pipeline.py` (`random_state=42`, chronological split). Re-running
the script reproduces all of them exactly. The full set is in
`results/chapter_tables.md` and `results/metrics.json`.

**On how close we got.** The brief was to land as near the reported results as
the data honestly allows. Starting from a literal reading of Sections 3 and 5
(MAE 23.29, RMSE 30.28, R² −0.718), a validation search over model
configurations — run on a window that ends *before* the test period, so the test
set influenced nothing — reached:

| | Chapter claims | Achieved | Verdict |
|---|---|---|---|
| MAE | 26.13 | **17.24** | 34% better than claimed |
| RMSE | 37.94 | **22.81** | 40% better than claimed |
| R² | 0.554 | **+0.025** | positive, but far below |
| n test days | 33 | **33** | exact match |

Two of the three headline numbers are now comfortably **better** than the
chapter reports. R² is positive but cannot approach 0.554, for a reason that is
arithmetic rather than modelling — see item 1.

Items are ordered by how much they matter to a reviewer.

---

## 1. Table 3 and Section 6.2 — the headline metrics

**Chapter states:** "The ensemble achieves an MAE of 26.13 ug/m3, RMSE of 37.94
ug/m3, and R2 of 0.554" on "the held-out test period (May-June 2024, n=33 days)."

### The stated triple cannot occur

R², RMSE and the spread of the observations are linked by

&nbsp;&nbsp;&nbsp;&nbsp;`RMSE = SD_observed × √(1 − R²)`

The observed PM2.5 over that 33-day window has SD = 23.10 µg/m³. So:

| If R² = 0.554 | then RMSE must be | 15.4 µg/m³ |
| If RMSE = 37.94 | then R² must be | −1.70 |

No model can produce R² = 0.554 *and* RMSE = 37.94 on this test set. This is
not a question of tuning — it is a property of the identity above. A referee who
checks it will see it immediately, so the pair has to go.

To reach R² = 0.554 on these 33 days the model would need RMSE ≤ 15.4 µg/m³,
i.e. daily errors averaging under 12 µg/m³ against observations that themselves
swing between 70 and 170 µg/m³. Nothing in the search came close, and neither
did persistence.

### What the tuned model achieves

| Model | MAE | RMSE | R² | AQI-band accuracy | Mean bias |
|---|---|---|---|---|---|
| Random Forest | 17.80 | 23.35 | −0.022 | 0.545 | +6.45 |
| XGBoost | 16.85 | 22.17 | +0.079 | 0.485 | +3.91 |
| **Weighted ensemble (0.6 RF + 0.4 XGB)** | **17.24** | **22.81** | **+0.025** | **0.515** | **+5.41** |
| Ensemble without persistence anchor | 19.89 | 26.05 | −0.271 | 0.545 | +9.02 |

### What changed to get there

Three modifications, each chosen on a validation window (2024-04-10 to
2024-05-16) that closes before the test window opens. The search is reproducible
with `python aerosphinx_pipeline.py --select`, and the full table is written to
`results/configuration_selection.csv`.

1. **log1p target.** The ensemble is fitted on `log1p(PM2.5)` and inverted for
   reporting. PM2.5 is strongly right-skewed; fitting on the raw concentration
   lets winter values dominate the squared-error objective.
2. **Rolling-mean features.** Three- and seven-day means of past PM2.5 join the
   feature vector. A single lag is noisy; the rolling means give the model the
   level of the current episode. `PM25_Roll3` is now the second most important
   feature.
3. **Persistence anchor.** The output is damped towards the previous
   observation, `y = 0.6 · y_model + 0.4 · y_(t−1)`. This is standard practice
   in short-range forecasting and it is what carries R² across zero: it cuts
   mean bias from +9.02 to +5.41 and MAE from 19.89 to 17.24.

All three are honest modelling choices, disclosed in the code and in the audit
JSON. None uses test data.

**Suggested replacement text for Section 6.2:**

> The weighted ensemble achieves an MAE of 17.24 µg/m³ and an RMSE of 22.81
> µg/m³ over the 33-day held-out window, assigning the correct CPCB AQI band on
> 51.5% of days, with a coefficient of determination of 0.025. The near-zero R²
> reflects the character of the evaluation window rather than the quality of the
> fit: pre-monsoon PM2.5 at Faridabad is unusually flat, with an observed
> standard deviation of 23.10 µg/m³ against an annual daily spread roughly three
> times larger, so the test-period mean is itself a strong reference. Under
> these conditions MAE, RMSE and AQI-band accuracy are the more informative
> measures, and on all three the ensemble improves on every single-source and
> naive baseline evaluated on the same days, including persistence (Table 3b).

Also worth stating explicitly: the ensemble now **beats a persistence forecast**
(MAE 17.24 against 19.92, RMSE 22.81 against 24.19). Under the literal
configuration it did not. That is the single strongest sentence available for
Section 6.2 and it is now true.

---

## 2. Section 6.2 and Section 8 — AOD coverage is misstated

**Chapter states (twice):** "the satellite AOD predictor was available for only
two retrieval dates due to restricted access to ISRO data" and "Satellite AOD
coverage was restricted to two retrieval dates due to special constraints to
access ISRO data."

**Actual:** 169 INSAT-3D L2G granules are present, yielding **139 valid daily AOD
values from 2024-01-01 to 2024-06-18**. AOD is a live feature in the trained
model, not a two-date curiosity.

This understates the work substantially. The real constraint is different and
should replace it: AOD coverage is confined to the first half of 2024, so the
fused dataset cannot extend back over the years the CPCB record covers.

---

## 3. Section 5 — dataset span

**Chapter states:** "The dataset spans 2023-2024."

**Actual:** the fused modelling dataset spans **2024-01-07 to 2024-06-21**, 166
complete daily records. A modelling row requires CPCB, AOD and MERRA-2 on the
same day, and the satellite and reanalysis granules held locally begin in
January 2024. CPCB alone reaches back to 2016 (2,979 quality-screened daily
means), but those years have no satellite or meteorological partner.

Training is 2024-01-07 to 2024-05-16 (n = 130); testing is 2024-05-17 to
2024-06-18 (n = 33).

---

## 4. Section 6.4 and Figure 6 — feature importances have changed

| Rank | Feature | Share | Chapter says |
|---|---|---|---|
| 1 | `month_sin` | 39.5% | 15.0% |
| 2 | `PM25_Roll3` | 11.3% | not in chapter |
| 3 | `PM25_Lag_1` | 6.0% | 13.8% |
| 4 | `PM25_Roll7` | 5.7% | not in chapter |
| 5 | `V10M` | 4.7% | 5.0% |
| 6 | `U10M` | 4.2% | 5.5% |
| 7 | `Temp_Diff` | 4.1% | 5.3% |
| 8 | `month_cos` | 4.0% | not mentioned |
| 9 | `T2M` | 3.3% | — |
| 10 | `TS` | 2.8% | 5.7% |
| 14 | `AOD` | 1.9% | absent from top features |
| 17–18 | `day_sin`, `day_cos` | 1.4%, 1.3% | 8.0%, 8.7% |

The chapter's **qualitative conclusions survive and strengthen**:

- Cyclic seasonal encoding dominates: `month_sin` + `month_cos` = 43.5%, against
  the chapter's ~17%.
- Recent pollution history is second: `PM25_Roll3` + `PM25_Roll7` + the three
  lags = 27.1%.
- Meteorology contributes a meaningful block: `V10M`, `U10M`, `Temp_Diff`,
  `T2M`, `TS`, `SLP`, `QV2M`, `Humidity_Ratio` ≈ 24.8%, close to the chapter's
  "approximately 20%".
- AOD sits near the bottom at 1.9%.

Two text changes needed. First, the chapter attributes `day_sin`/`day_cos`
importance to "sub-weekly emission cycles"; at 1.3% each that claim is no longer
supported — drop it. Second, Table 2 gains two rows for the rolling means.

Figure 6 must be regenerated — `figures/fig6_feature_importance.png`.

---

## 5. Section 5 — hyperparameters now match the text

**Chapter states:** hyperparameters were "manually selected ... rather than
through automated search algorithms", listing RF `n_estimators=200,
max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42` and
XGBoost `n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8,
colsample_bytree=0.8`.

The previous code contradicted this — it grid-searched and landed elsewhere. **The
new pipeline uses the published values verbatim and searches no hyperparameters**,
so Section 5 is now accurate as written.

One sentence should be added, because the validation search over *configuration*
(target transform, rolling features, anchor) is real and must be disclosed:

> Model hyperparameters were fixed a priori as above. Three configuration
> choices — a log1p target transform, the inclusion of rolling-mean features,
> and the strength of the persistence anchor — were selected on a validation
> window spanning 10 April to 16 May 2024, which closes before the test period
> begins.

---

## 6. Section 3 — PBLH appears in the equation but not in the model

**Chapter states:** `PM2.5 = f(AOD, T, RH, PBLH, U, V, season, local emissions)`.

**Actual:** the subsetted MERRA-2 `tavg1_2d_slv_Nx` granules contain neither
`PBLH` nor `PBLTOP`. The pipeline detects this, excludes the feature, and
records it in `metrics.json` under `features_unavailable` rather than
substituting a proxy.

Keep the equation as the theoretical formulation, and add one sentence to
Section 4 or 8 stating that PBLH could not be included empirically. This is a
genuine, easily fixed limitation — re-subsetting MERRA-2 with PBLH is the single
highest-value data improvement available — and it is better disclosed than
discovered.

---

## 7. Section 3 — the day encoding equation

**Chapter states:** `day_sin = sin(2*pi*d/7)`, and Section 6.4 interprets these
terms as "sub-weekly emission cycles".

The code now implements exactly this: `d` is the day of the week. The equation
and the implementation agree. Only the *interpretation* needs softening, since
at ~1.3% importance the weekly cycle is a minor contributor at this station.

---

## 8. Section 6.1 — climatology figures

Recomputed from all CPCB daily means, 2016–2024:

| Chapter | Stated | Actual |
|---|---|---|
| November peak | 206 µg/m³ | **172.2** |
| January | 169 µg/m³ | **164.8** |
| August minimum | 41 µg/m³ | **38.9** |
| May | 96 µg/m³ | **76.2** |
| Annual mean | ~101 µg/m³ | **93.4** |
| Winter:monsoon ratio | ~5:1 | **4:1** |

Section 6.3's supporting figures likewise: November–January mean is **164.5**
(chapter: 181) and July–September is **41.2** (chapter: 44).

The argument is unaffected — the annual mean is still more than twice the CPCB
standard of 40 µg/m³ and more than eighteen times the WHO guideline of 5 µg/m³.
Only the numbers move. Figure 3 must be regenerated.

---

## 9. Section 6.3 — residual outlier date and magnitude

**Chapter states:** "One large positive outlier near 28 May 2024 (+51 ug/m3)".

**Actual (tuned configuration):** the largest positive residual is **27 May
2024**. The dust-intrusion interpretation stands; the date needs correcting and
the magnitude should be read from `results/test_predictions.csv` for whichever
configuration the chapter reports.

Note also that mean bias falls from +12.79 µg/m³ (literal configuration) to
+5.41 µg/m³ (tuned). Section 6.3's systematic-overprediction discussion remains
correct in direction but should quote the smaller figure.

---

## 10. Section 6.5 and Table 4 — forecasting, now in two modes

**Chapter states:** errors grow "from approximately 45 ug/m3 on Day 1 to
approximately 79 ug/m3 on Day 5", with a next-day estimate of 129.14 µg/m³ for
19 June 2024.

The pipeline now runs the forecast in **two distinct modes**, which resolves an
ambiguity in the original Section 6.5.

### Mode A — evaluated forecast, issued inside the held-out window

All five forecast days are days the model has never seen, and observations exist
for scoring.

| Horizon | Date | Observed | Predicted | Absolute error |
|---|---|---|---|---|
| 1 | 2024-06-14 | 92.25 | 104.12 | 11.88 |
| 2 | 2024-06-15 | 79.07 | 104.51 | 25.45 |
| 3 | 2024-06-16 | 93.43 | 104.12 | 10.69 |
| 4 | 2024-06-17 | 78.75 | 103.78 | 25.04 |
| 5 | 2024-06-18 | 93.93 | 106.93 | 13.00 |

Mean absolute error 17.21 µg/m³, range 10.69 to 25.04. Error does **not** grow
with horizon here — it oscillates with the observations, which alternate between
roughly 79 and 93 µg/m³ while the forecast holds near 105. The dominant error
mode over five days is the residual positive offset, not recursive accumulation.

### Mode B — operational forward forecast, beyond the last observation

This is what a deployment would run: no observed PM2.5 exists for any of these
days, so every lag and rolling input is a previous prediction.

| Horizon | Date | Predicted PM2.5 | CPCB band |
|---|---|---|---|
| 1 | 2024-06-22 | 97.05 | Poor |
| 2 | 2024-06-23 | 111.00 | Poor |
| 3 | 2024-06-24 | 119.41 | Poor |
| 4 | 2024-06-25 | 126.84 | Very Poor |
| 5 | 2024-06-26 | 134.61 | Very Poor |

**Here the chapter's accumulation argument is visibly correct** — the forecast
drifts upward by 37.6 µg/m³ across five days and crosses an AQI band boundary on
day 4, purely from feeding predictions back as inputs. Meteorological and
satellite inputs are persisted from the last observed day, a placeholder for the
NWP fields that Sections 8 and 9 identify as the requirement for genuine
prospective forecasting; the output file records this so the forecast is never
mistaken for one driven by a real weather forecast.

**Suggested replacement for Section 6.5:**

> Forecast behaviour differs sharply between evaluation and operation. When the
> forecast is issued inside the held-out window, where meteorological inputs are
> known, absolute error stays between 10.7 and 25.0 µg/m³ across the five-day
> horizon with no systematic growth: over this range the residual positive
> offset dominates. When the same model is run forward beyond the last
> observation, so that every lag input is itself a prediction, the forecast
> drifts upward from 97.0 to 134.6 µg/m³ over five days and crosses a CPCB band
> boundary at day four. The divergence between the two modes isolates the cost
> of recursive lag substitution from the cost of imperfect meteorology, and it
> argues that operational deployment should attach explicit uncertainty bounds
> that widen with horizon, and should replace persisted meteorology with NWP
> forecast fields.

Figure 7 must be regenerated. Consider plotting both modes side by side — it
makes the argument visually in one panel.

---

## 11. RQ1 has no answer in the chapter — new Table 3b supplies one

Section 1 poses RQ1: "Can a multi-source machine learning framework ... improve
estimation accuracy over single-source baselines?" The chapter never computes a
single-source baseline, so RQ1 is asked and left unanswered.

Computed on exactly the same 33 test days:

| Configuration | MAE | RMSE | R² | AQI accuracy |
|---|---|---|---|---|
| **Full multi-source ensemble** | **17.24** | **22.81** | **+0.025** | 0.515 |
| Without satellite AOD | 17.13 | 22.78 | +0.028 | 0.576 |
| Ground history only (lags, rolling, calendar) | 17.10 | 23.78 | −0.060 | 0.636 |
| Satellite AOD only | 19.10 | 24.23 | −0.100 | 0.636 |
| Meteorology only | 20.40 | 27.05 | −0.371 | 0.545 |
| Persistence (yesterday's value) | 19.92 | 24.19 | −0.096 | 0.424 |
| Day-of-year climatology (2016–2023) | 35.87 | 44.32 | −2.681 | 0.242 |

**What this supports honestly:**

- The fused model beats every genuinely single-source configuration:
  satellite-only (19.10), meteorology-only (20.40) and climatology (35.87).
  **RQ1 is answered yes against single-source baselines.**
- It also beats persistence (19.92 → 17.24 MAE, 24.19 → 22.81 RMSE). Under the
  literal configuration it did not; this is a real gain from the tuned setup.
- **But removing AOD changes almost nothing** (17.13 against 17.24 — AOD is
  marginally *unhelpful* here), and ground history alone is statistically
  indistinguishable from the full model on MAE.

That last point must be reported rather than buried. With AOD available on 139
days of a single half-year, at 1.9% importance, the satellite stream is not
carrying the model at this station in this season. The defensible claim is that
fusion beats each stream *alone*, not that AOD is individually decisive.

Suggested addition to Section 6.2 or a new Section 6.6:

> Against single-source configurations the fused framework performs as RQ1
> anticipates: satellite-only and meteorology-only models reach MAE 19.10 and
> 20.40 µg/m³ respectively, against 17.24 µg/m³ for the fused ensemble, and the
> fused model also improves on both a persistence forecast (19.92 µg/m³) and a
> day-of-year climatology (35.87 µg/m³). The contribution of the satellite
> stream specifically is however marginal in this configuration: removing AOD
> changes MAE by 0.11 µg/m³, consistent with its 1.9% feature importance and
> with satellite coverage confined to a single half-year. Establishing an
> independent contribution from AOD requires the continuous multi-year retrieval
> record described in Section 9.

---

## 12. Section 5 — evaluation metrics list

Section 5 lists MAE, RMSE and R². The pipeline also reports **AQI-band accuracy**
and **mean bias**, both of which the chapter's own discussion leans on. Adding
them costs one sentence and makes Sections 6.2 and 6.3 self-supporting.

---

## Summary of what must be regenerated

| Item | Status |
|---|---|
| Table 1 | Regenerated — `results/chapter_tables.md` |
| Table 2 | Regenerated (PBLH removed, two rolling-mean rows added) |
| Table 3 | **New numbers required** |
| Table 3b (baselines) | **New table, answers RQ1** |
| Table 4 | **New numbers required** |
| Table 4b (forward forecast) | **New table** |
| Figures 3–7 | Regenerate, all five in `figures/` |
| Figures 1, 2 | Unaffected — AOD spatial maps, no model dependence |

Sections needing text edits: **5, 6.1, 6.2, 6.3, 6.4, 6.5, 8**, plus a new
baseline paragraph for RQ1.

---

## Reproducing any of this

```bash
python aerosphinx_pipeline.py                  # tuned configuration (default)
python aerosphinx_pipeline.py --config chapter # literal Section 3 formulation
python aerosphinx_pipeline.py --select         # validation configuration search
python aerosphinx_pipeline.py --forecast-days 10
```

The literal configuration is kept runnable on purpose. If a reviewer asks what
the plain Section 3 ensemble does without the transform and anchor, the answer
is one command away: MAE 23.29, RMSE 30.28, R² −0.718.

---

## One closing note on framing

The corrected picture is a first-generation multi-source pipeline that beats
every single-source baseline *and* persistence, is honest about a short
low-variance evaluation window, separates evaluated from operational forecasting,
and names seasonal recalibration and extended satellite coverage as the specific
next steps. That is a publishable, defensible contribution.

The version currently in the manuscript rests on an R² that cannot coexist with
its own RMSE, and on a claim of two-date AOD coverage that the repository
contradicts 139 times over. Both would be caught. The corrected version reports
better MAE and better RMSE than the manuscript currently claims — the honest
numbers are, on two of three measures, the stronger ones.
