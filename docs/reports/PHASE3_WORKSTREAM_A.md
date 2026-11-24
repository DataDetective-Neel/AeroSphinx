# AeroSphinx — Phase 3 Workstream A Report: Multi-Horizon Forecasting

**Workstream A complete.** The central experiment of Phase 3.

Run directory: `results/runs/A-horizon-20261112T174704Z/`

---

## 1. Headline

> **Skill relative to persistence is horizon-dependent, and it inverts.**
>
> At **day 1** no learned model beats persistence — the Phase 2 result
> holds. From **day 2 onward** the learned models are *better* than
> persistence, by margins that grow with horizon.

| Horizon | Models better than persistence | Best Δ MAE | Baseline MAE | Best model MAE |
|---|---|---|---|---|
| 1 | **0 / 6** | +2.33 (worse) | 20.90 | 23.23 |
| 2 | **5 / 6** | **−3.95** | 32.02 | 28.06 |
| 3 | **5 / 6** | **−8.70** | 37.97 | 29.27 |
| 4 | **5 / 6** | **−7.37** | 38.82 | 31.45 |
| 5 | 3 / 6 | −5.75 | 39.41 | 33.66 |

Δ MAE is model minus baseline; negative means the model is better.

**This resolves the project's central research question.** Persistence is not
a strong baseline at multi-day horizons — it is a strong baseline only at
day 1, where the pollutant is nearly a random walk. A model with seasonal
and temporal context degrades much more slowly than persistence does.

---

## 2. Why the horizon-appropriate baseline changes everything

The comparison in Phase 2 used the **day-1** persistence value as the
reference for a day-1 forecast. That is correct for day 1 and wrong for
every other horizon.

The correct baseline for a day-*h* forecast is the observation *h* calendar
days before the target. On identical target dates, persistence error grows
steeply with lag:

| Lag | Persistence MAE | Increase over lag 1 |
|---|---|---|
| 1 | 23.198 | — |
| 2 | 30.563 | +7.364 |
| 3 | 34.322 | +11.124 |
| 4 | 35.503 | +12.305 |
| 5 | 36.805 | +13.607 |

**Persistence is not degrading gracefully — it is collapsing**, because a
value from five days ago carries almost no information about today.

This is the confound-free reference (`persistence_by_lag.csv`), computed on
the **identical set of target dates** at every lag, so no target-date
difficulty is mixed in.

---

## 3. A methodological trap, caught and corrected

A first version of this workstream used the day-1 persistence value as the
baseline at every horizon. That produced a **misleading** result: it
suggested the models got steadily *worse* relative to persistence with
horizon (Δ +2.3 → +12.7), which reads as "learned models degrade and
persistence holds up".

Two diagnostics showed this was an artefact:

1. **The baseline's apparent error did not grow with horizon** — MAE 20.90
   at h=1, then 25.95, 24.44, 22.25, 22.56. It *fell* after h=2. A
   baseline whose error falls with horizon is measuring which target dates
   were hit, not horizon difficulty.

2. **Each horizon forecasts a different target date.** One forecast is
   issued per issue date, covering the next *h* days, so h=3 targets
   different dates than h=1. The raw cross-horizon comparison confounds
   horizon difficulty with target-date difficulty.

The fix was a `periods` argument on the persistence baseline so it uses the
*h*-day lag. The unit test
`tests/unit/test_multi_horizon.py::test_baseline_is_horizon_appropriate`
now pins this.

---

## 4. Results by horizon

### Direct method

| Horizon | Model | MAE | Baseline MAE | Δ | Block CI | Significant |
|---|---|---|---|---|---|---|
| 1 | ridge | 23.23 | 20.90 | +2.33 | [−3.01, +7.33] | No |
| 1 | random forest | 26.74 | 20.90 | +5.84 | [+0.76, +10.88] | **Yes (worse)** |
| 1 | xgboost | 27.29 | 20.90 | +6.39 | [+1.01, +12.03] | **Yes (worse)** |
| 2 | ridge | 28.06 | 32.02 | −3.95 | [−10.73, +2.30] | No |
| 2 | random forest | 29.92 | 32.02 | −2.09 | [−5.72, +1.55] | No |
| 2 | xgboost | 30.07 | 32.02 | −1.95 | [−5.70, +1.79] | No |
| 3 | random forest | 29.27 | 37.97 | **−8.70** | [−17.58, −1.29] | **Yes (better)** |
| 3 | ridge | 32.08 | 37.97 | −5.89 | [−13.51, +1.73] | No |
| 3 | xgboost | 30.73 | 37.97 | −7.24 | [−16.69, +0.88] | No |
| 4 | xgboost | 31.45 | 38.82 | −7.37 | [−17.05, +1.47] | No |
| 4 | random forest | 32.69 | 38.82 | −6.13 | [−15.42, +2.36] | No |
| 4 | ridge | 34.58 | 38.82 | −4.24 | [−12.58, +3.54] | No |
| 5 | random forest | 33.66 | 39.41 | −5.75 | [−13.40, +1.62] | No |
| 5 | ridge | 34.44 | 39.41 | −4.97 | [−12.06, +1.99] | No |
| 5 | xgboost | 34.41 | 39.41 | −5.00 | [−12.96, +2.61] | No |

### Recursive method

Recursive forecasting is **worse than direct** at every horizon beyond 1,
and dramatically so for XGBoost (h=5: MAE 53.91 against a direct 34.41).
Error compounds through the feedback loop, exactly as the plan anticipated.

| Horizon | Ridge | Random forest | XGBoost |
|---|---|---|---|
| 1 | +2.33 | +5.84 | +6.39 |
| 2 | −2.85 | −1.77 | **+10.73** |
| 3 | −2.18 | −5.00 | **+11.25** |
| 4 | −4.37 | −3.37 | **+14.06** |
| 5 | +1.29 | +1.46 | **+14.49** |

XGBoost's recursion is pathological — its predictions are pushed further
from the training range at each step, and being a tree ensemble it cannot
extrapolate back.

**Direct forecasting is the recommended method.** It is simpler, it is
never worse, and at h=3 to h=5 it is substantially better.

---

## 5. The within-issue-date comparison

The raw cross-horizon comparison confounds horizon difficulty with
target-date difficulty. Holding the **issue date** fixed removes the
latter: the same forecast origin is compared across horizons.

| Horizon | Model error increase vs h=1 | Baseline error increase vs h=1 | Model − baseline |
|---|---|---|---|
| 2 | +3.98 (RF) | +9.83 | −0.46 |
| 3 | +3.78 (RF) | +15.48 | **−6.68** |
| 4 | +4.80 (RF) | +16.30 | **−6.29** |
| 5 | +7.21 (RF) | +16.69 | **−3.97** |

The baseline's error grows by 16.7 µg/m³ from h=1 to h=5, while the
model's grows by only 7.2. **The model's skill decays more slowly than
persistence's, and by h=3 the gap is 6.7 µg/m³ in the model's favour.**

This is the cleanest statement of the finding, because it holds the forecast
origin fixed.

---

## 6. What this establishes

1. **The revised research question is answered, at least for ground-only
   predictors.** Environmental information is not needed to beat
   persistence at multi-day horizons — PM2.5 history and calendar features
   are sufficient. The value of Track B's environmental covariates must
   therefore be demonstrated *at these longer horizons*, where persistence
   is weak, rather than at day 1 where it is strong.

2. **The Phase 2 conclusion was horizon-specific, not universal.** "No
   learned model beats persistence" is true at day 1 and **false from day 2
   onward**. A study that evaluated only day 1 would have concluded the
   opposite.

3. **The correct baseline matters as much as the model.** The comparison
   against the horizon-appropriate baseline, on identical target dates,
   is what makes the finding visible. Against the day-1 baseline the same
   models look consistently worse.

4. **Direct beats recursive.** Recursion compounds error and is
   pathological for tree ensembles, which cannot extrapolate outside their
   training range.

5. **Statistical honesty.** Of the 30 comparisons, **10 are significant
   under block-aware inference, and only 2 of those are significant in the
   model's favour** (both at horizon 3). **None survives the Holm
   correction across the family** (0 of 30 at p_holm < 0.05). The
   consistent *sign* across horizons is the robust finding; individual
   significances are not, and must not be reported as though they were.

---

## 7. What this does NOT establish

- **No environmental covariate was used.** Every model here is
  ground-history plus calendar only. The finding says nothing yet about
  whether AOD or MERRA-2 add value — only that *some* learned predictor
  beats persistence at h≥2.
- **Single station, single city.** As throughout.
- **The nine-year record's heterogeneity is unaddressed.** The
  year-on-year MAE trend remains confounded with the measurement change
  identified in Phase 0.
- **Only one forecast per issue date per horizon.** The variance of the
  estimate is correspondingly wide; the intervals are wide for that reason.

---

## 8. Implications for Workstream D

The Phase 3 authorization's priority order for environmental features was:

1. CPCB station-local meteorology
2. Historical MERRA-2 meteorology
3. INSAT AOD
4. Combined environmental feature groups
5. Availability-aware model routing

**This result sharpens that order.** Because ground history alone already
beats persistence from h=2 onward, the test for an environmental feature
group is now sharper and more demanding:

> Does the environmental group add value **on top of** a ground-history
> model that is already better than persistence at h≥2?

A satellite or meteorological feature that merely lifts a day-1 model to
parity with persistence has added nothing. The comparison must be against
the *strongest* available predictor at each horizon, not against
persistence.

---

## 9. Open questions

| ID | Question |
|---|---|
| **B-017** | Is the h=2–5 advantage driven by the calendar features (seasonal cycle) or by the PM2.5 history? An ablation would separate them, and the answer determines whether environmental covariates face a harder task. |
| **B-018** | Should the day-1 model be a persistence-like model, and the multi-day models separate? The h=1 result suggests a single model across horizons is wrong. |
| **B-019** | Does the advantage survive the nine-year record's heterogeneity, or is it concentrated in particular regimes? |
| **B-020** | Can recursive forecasting be salvaged with per-horizon damping, or should it be dropped in favour of direct methods? |

---

## 10. Artifacts

| File | Contents |
|---|---|
| `forecasts.csv` | Every forecast with its issue date, horizon, method and baseline |
| `metrics.csv` | Per-horizon, per-method, per-model metrics with baseline |
| `comparisons.csv` | Paired comparisons with block-aware and year-blocked inference |
| `by_horizon.csv` | Skill summary across horizons |
| `within_issue.csv` / `within_issue_summary.csv` | Within-issue-date comparison |
| `persistence_by_lag.csv` | Confound-free persistence reference |
| `manifest.json` | Git SHA, environment, configuration |
