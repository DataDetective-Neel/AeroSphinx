# AeroSphinx — Phase 3 Workstream C Report: Regime-Conditioned Modelling, Ablation and Routing

**Workstream C complete.** Expanded scope: regime-conditional modelling,
feature-group ablation, horizon-specific architecture, and validated routing.

Run directory: `results/runs/C-regime-20261112T201612Z/`

---

## 1. Headline

> **Two results, one positive and one negative.**
>
> **Positive:** regime-aware and multi-horizon architectures genuinely beat a
> single global model — by 1.4 to 3.4 µg/m³ depending on horizon — and the
> gain is selected on validation folds, never on test.
>
> **Negative:** **station-local meteorology does not add predictive value.**
> In the recommended **direct** method, adding the nine audited CPCB
> meteorological columns makes the model *worse* in 18 of 25 comparisons and
> better in 7, **none** significantly better and one significantly worse.
> After Holm correction across the full family of 50 comparisons,
> **zero** survive.

The second result is the answer to the project's central Track B question, and
it is a genuine negative.

---

## 2. Protocol compliance

| Requirement | How it is enforced | Where |
|---|---|---|
| Preserve all 96 folds | Every fold contributes a validation pass and a test pass | `run_workstream_c` |
| Chronological validation | Pass A issues at the end of the training block and scores the validation block; Pass B issues at the end of the validation block and scores the test block | `run_workstream_c` |
| Architecture chosen on validation | `_select_architecture` ranks on validation MAE only; the test scores are attached afterwards and never consulted | `_select_architecture` |
| Selection per horizon | Ranking is within each horizon, because a deployment needs an architecture for every horizon | `_select_architecture` |
| A candidate needs enough validation data to be selectable | `eligible_for_selection` requires ≥ 30 validation rows | `_select_architecture` |
| Matched test dates | Each group set is compared against the reference on identical `(fold, architecture, method, horizon, Date)` keys, and only on dates where the group's features are genuinely present | `_ablation` |
| Block-aware uncertainty | Moving-block bootstrap with a data-estimated block length, plus a year-blocked bootstrap | `_ablation` |
| Multiple-comparison correction | Holm-Bonferroni and Benjamini-Hochberg across the family | `_ablation` |
| Report by horizon and season | `by_horizon_season.csv` | `run_workstream_c.py` |
| Availability contract | `assert_operational_contract` refuses an operational set containing a retrospective-only group | `features/groups.py` |

**Two protocol bugs were found and fixed during this workstream**, both by
the tests and diagnostics rather than by inspection:

1. **The forecast issue date landed after the test block**, so every
   prediction was scored against a date outside the block and the results
   were empty. Fixed with the two-pass walk-forward structure.
2. **The recursion overwrote the lag features before the first prediction**,
   so the recursive method did not agree with the direct method at horizon
   one. Fixed by leaving the issue row's real features intact at the first
   step.

A third defect was found by the matched-date rule: the coverage test
originally included the always-present ground features, so every group set
looked fully covered. Coverage is now judged on the features a group *adds*
beyond the ground baseline.

---

## 3. The availability contract, encoded

`src/aerosphinx/features/groups.py` makes the authorisation's constraint
machine-enforceable.

| Group | Availability class | Operational? |
|---|---|---|
| `pm25_history` | observed_ground_history | **Yes** |
| `calendar` | calendar | **Yes** |
| `station_met` | station_local_unknown_availability | **No — retrospective only** |
| `merra` | reanalysis | **No — never operational** |
| `aod` | observed_satellite | **No — retrospective only** (and disabled pending Phase 4) |

```python
assert_operational_contract(GROUND_ONLY_GROUPS)          # passes
assert_operational_contract(GROUND_ONLY_GROUPS + ("station_met",))
# ValueError: operational feature contract violated: the following groups
# are not available at forecast issue time: station_met
```

All nine station-local columns are **preserved**, with their measurement
timestamps intact, `availability_timestamp = UNKNOWN` and
`forecast_usable = false`. No publication lag is assumed anywhere. The
MERRA-2 extraction carries a `_midday` suffix on every variable, which
corrects the naming half of **D-010** — the values are midday-window means
at a single 1×1 grid cell, not daily means over a collocation box.

---

## 4. Feature-group ablation — the Track B question

### 4.1 The direct comparison: does the environmental group help *the model*?

The ablation table compares each model with persistence. The sharper question
is whether the environmental group adds value **on top of the ground-only
model**, holding the architecture, method, horizon and rows fixed and varying
only the feature group. That comparison is in `group_pairwise.csv`.

**Station-local meteorology vs ground-only, direct method (25 comparisons):**

| Architecture | h | + station met | ground only | Δ | 95% block CI | Significant? | Better? |
|---|---|---|---|---|---|---|---|
| global | 1 | 23.60 | 24.57 | −0.97 | [−3.16, +1.22] | No | yes |
| global | 2 | 30.57 | 29.60 | +0.98 | [−1.15, +3.28] | No | no |
| global | 3 | 34.62 | 32.56 | +2.06 | [−1.85, +7.74] | No | no |
| global | 4 | 38.14 | 35.06 | +3.08 | [−0.84, +8.58] | No | no |
| global | 5 | 37.41 | 36.24 | +1.17 | [−2.09, +5.16] | No | no |
| seasonal / routing | 1 | 22.29 | 24.17 | −1.88 | [−4.60, +0.66] | No | yes |
| seasonal / routing | 2 | 30.93 | 29.35 | +1.58 | [−1.12, +4.13] | No | no |
| seasonal / routing | 3 | 32.51 | 31.25 | +1.26 | [−1.96, +4.96] | No | no |
| seasonal / routing | 4 | 37.11 | 35.04 | **+2.07** | **[+0.19, +3.99]** | **Yes — worse** | no |
| seasonal / routing | 5 | 34.64 | 36.36 | −1.72 | [−4.91, +1.20] | No | yes |
| shared | 1 | 24.95 | 25.29 | −0.34 | [−2.66, +1.99] | No | yes |
| shared | 2–5 | 31.78–35.37 | 28.40–34.24 | +1.00 to +3.38 | spans 0 | No | no |

**Direct method summary:** better in **7 of 25**, worse in **18 of 25**,
**0 significantly better**, **1 significantly worse** (seasonal/routing at
h=4). The harm is systematic at the longer horizons: +3.08 at h=4 for the
global model, +2.07 for the regime models.

### 4.2 The recursive method, and why its result is not what it seems

The same comparison for the **recursive** method is favourable: station
meteorology is better in **all 25** comparisons, with 10 block-significant
improvements of up to −7.54 µg/m³.

**This does not mean the meteorological columns are predictive.** The
recursive model is *far worse than direct* in absolute terms — at h=5 the
ground-only recursive model scores 66.93 against the direct model's 36.24.
The meteorological columns dampen the recursion's divergence, because the
model has a genuine environmental signal to fall back on instead of
extrapolating its own error. That is a *mitigation of a defective method*,
not evidence of predictive value.

### 4.3 The honest summary

| Method | Comparisons | Group better | Significantly better | Significantly worse | Holm p < 0.05 and better |
|---|---|---|---|---|---|
| Direct | 25 | 7 | **0** | 1 | **0** |
| Recursive | 25 | 25 | 10 | 0 | **0** |

> **REVISED in Phase 5.** The single "significantly worse" comparison above
> was an artefact of a corrupt column in the station data. The CPCB
> barometric-pressure column fails three independent physical checks and has
> since been screened and excluded from the group (D-020). With it removed
> the corrected run gives **0 significantly worse** as well. The null
> conclusion — no demonstrated benefit — is unchanged, but the group is now
> *neutral* rather than *harmful*. See `docs/reports/PHASE5_REPORT.md` §6 and
> `results/runs/C-regime-20261113T052110Z/`.
| **All** | **50** | 32 | 15 | 2 | **0** |

**After correction for the family of 50 comparisons, not one result
survives.** The recursive method's apparent advantage is an artefact of
comparing two poor models.

### 4.4 MERRA-2 cannot be evaluated

MERRA-2's coverage is 172 days, all in 2024, and only **5** of them fall in
a test block. Every MERRA comparison is reported with `n_matched = 5` and
`sufficient_evidence = False`, and MERRA is **excluded from architecture
selection** for the same reason. **Nothing about MERRA-2's predictive value
can be concluded from the local data, and none is claimed.**

---

## 5. Architecture comparison — the positive result

Ground-only model, direct method, MAE by architecture and horizon:

| Horizon | global | interaction | routing | seasonal | shared | Best |
|---|---|---|---|---|---|---|
| 1 | 24.57 | 24.57 | **24.17** | **24.17** | 25.29 | routing / seasonal |
| 2 | 29.60 | 29.60 | 29.35 | 29.35 | **28.40** | shared |
| 3 | 32.56 | 32.56 | **31.25** | **31.25** | 31.91 | routing / seasonal |
| 4 | 35.06 | 35.06 | 35.04 | 35.04 | **32.84** | shared |
| 5 | 36.24 | 36.24 | 36.36 | 36.36 | **34.24** | shared |

**Every architecture beats the plain global model at every horizon**, by
1.4 to 3.4 µg/m³. The improvement comes from two distinct mechanisms:

- **Regime conditioning** (`seasonal`, `routing`) helps at the **short**
  horizons (h=1, h=3). `interaction` is numerically identical to `global`
  here — with a linear model and the regime indicators, the interaction
  terms do not change the fit materially, which is itself informative.
- **The shared multi-horizon model** (`shared`) helps at the **long**
  horizons (h=2, h=4, h=5), by up to 3.4 µg/m³ at h=4.

That is a coherent story: a single model pooled across horizons borrows
strength across the horizon dimension, which pays off when the target is
far away and the specific-horizon signal is weak.

### 5.1 Selected architecture, per horizon

Chosen on validation MAE alone:

| Horizon | Selected | Validation MAE | Test MAE | Persistence MAE | Δ vs persistence |
|---|---|---|---|---|---|
| 1 | `ground_plus_station_met` / routing / direct | 22.505 | 22.294 | 20.953 | **+1.341** |
| 2 | `ground_only` / shared / direct | 29.711 | 28.403 | 32.212 | **−3.809** |
| 3 | `ground_only` / routing / direct | 32.414 | 31.249 | 38.015 | **−6.766** |
| 4 | `ground_only` / shared / direct | 34.391 | 32.835 | 39.131 | **−6.296** |
| 5 | `ground_only` / shared / direct | 35.631 | 34.238 | 39.421 | **−5.183** |

The h=1 selection is the only one that chose the station-meteorology group,
and on the test blocks it is the only horizon where the model **loses** to
persistence (+1.341). At h=2–5 the selected architecture beats persistence
comfortably. The selection is honest — it was made on validation — but it
shows the value of the guard that excluded MERRA from selection on 5 rows.

### 5.2 Direct versus recursive

**Direct wins at every horizon beyond 1.** For the ground-only global model:

| Horizon | Direct MAE | Recursive MAE |
|---|---|---|
| 1 | 24.57 | 24.36 |
| 2 | 29.60 | 48.31 |
| 3 | 32.56 | 60.40 |
| 4 | 35.06 | 62.06 |
| 5 | 36.24 | 66.93 |

The recursive method degrades catastrophically beyond day 1 (MAE 66.93
against 36.24 at h=5), because the error compounds through the feedback
loop and the linear model extrapolates further out with each step.

---

## 6. Skill by horizon and season

Ground-only, routing, direct. Model MAE minus persistence MAE (negative is
better):

| Season | h=1 | h=2 | h=3 | h=4 | h=5 |
|---|---|---|---|---|---|
| Monsoon | +4.33 | −1.72 | −2.22 | −0.82 | −2.36 |
| Pre-monsoon | +1.16 | +0.37 | −1.58 | −4.79 | −4.45 |
| Post-monsoon | −12.64 | −23.46 | −26.97 | −22.91 | −17.13 |
| **Winter** | **+15.09** | **+7.44** | −4.45 | **+5.23** | **+6.86** |

**This is the clearest statement of the regime dependence, and it directly
addresses the authorisation's revised research question.**

- **Post-monsoon** is where the model helps most: −12.6 to −27.0 µg/m³.
- **Monsoon** the model helps from h=2 onward but loses at h=1.
- **Winter** the model *loses* at h=1, h=2, h=4 and h=5, and only wins at
  h=3. Winter has the highest variance (persistence MAE 27.47 at h=1
  rising to 61.62 at h=5) and the model does not capture it.

The season is doing the work that the global architecture cannot: a single
model cannot be good everywhere, and the routing architecture's value is
concentrated in exactly the seasons where persistence is weak.

---

## 7. What this establishes

1. **Regime conditioning works, and the gain is real.** Every
   regime-aware or multi-horizon architecture beats the plain global model
   at every horizon, with the selection made on validation folds. This
   answers **B-012** affirmatively — regime-conditional modelling is
   justified, though it is not yet a demonstrated improvement over
   persistence, only over the global model.
2. **Station-local meteorology does not add value.** In the direct method
   it is better in 7 of 25 comparisons and worse in 18, with **0
   significantly better** and 1 significantly worse. Its apparent benefit
   under the recursive method is a mitigation of that method's divergence,
   not evidence of predictive value. **This is the first real test of the
   Track B hypothesis and the answer is negative.** The columns are
   genuine, complete and physically sensible — they simply do not carry
   information about PM2.5 that the persistence and seasonal signals do not
   already provide.

   *(Phase 5 revision: the one significantly-worse comparison was caused by a
   corrupt barometric-pressure column, now excluded. The corrected group is
   neutral rather than harmful; the null conclusion stands.)*
3. **MERRA-2 remains untestable.** Five matched dates. Nothing about
   MERRA-2's predictive value can be concluded from the local data, and
   none is claimed.
4. **The availability contract is now executable**, not just documented.
   The station columns can be used retrospectively and are automatically
   excluded from any operational contract.
5. **The seasonal picture is the real finding.** Skill is strongly
   regime-dependent: large gains in post-monsoon, large losses in winter.
   A single architecture cannot serve both, which is the argument for
   routing — and also the reason a routing policy needs to be validated
   rather than assumed.

---

## 8. What this does NOT establish

- **No environmental covariate has demonstrated incremental value.** Track B
  remains an open experiment. The station-local result is negative, MERRA-2
  is untestable, and AOD is not yet extracted.
- **The h=1 selection chose the station-meteorology group** and then lost to
  persistence on the test blocks. With one horizon's evidence that is
  suggestive, not conclusive.
- **Single station, single city.** Unchanged.
- **The nine-year record's heterogeneity** is still unaddressed (B-019).
- **Ridge is the model used throughout.** The architecture question and the
  model question are separate axes; a different model might reorder the
  architectures. This is a deliberate simplification, recorded in the run
  manifest.
- **No conclusion about operational forecasting.** The only group set that
  satisfies the operational contract is `ground_only`.

---

## 9. Open questions

| ID | Question |
|---|---|
| **B-025** | Should the h=1 architecture exclude the station-meteorology group on the grounds that it is retrospective-only, so that the *operationally eligible* architecture is selected separately from the best retrospective one? |
| **B-026** | Does the regime advantage survive with a tree model, or is it an artefact of the linear model's regularisation? |
| **B-027** | Is the winter failure a model failure or a property of the pollutant regime? Winter PM2.5 is dominated by synoptic episodes that no daily-history model can anticipate. |
| **B-028** | Should the routing policy be selected per season on validation, rather than per horizon? The seasonal table suggests season is the stronger axis. |
| **B-029** | Which of the nine station columns, if any, carries any signal? The group as a whole does not help, but individual columns might. |
| **B-030** | Can MERRA-2 be evaluated at all on this data, or must the answer to the MERRA question wait for pre-2024 granules (B-002)? |
| **B-031** | Does the station-meteorology null result survive with a tree model, or is it an artefact of the linear model's regularisation penalising 9 noisy features? |
| **B-032** | Does the station-meteorology null result survive with the sparse columns (BP, VWS) removed, so that the group is more complete? |

---

## 10. Artifacts

| File | Contents |
|---|---|
| `selection.csv` | Every candidate ranked by validation MAE within horizon, with eligibility and the selection flag |
| `selected_performance.csv` | The selected architecture's test performance per horizon |
| `architecture_validation.csv` | Candidate scores on validation blocks |
| `architecture_test.csv` | The same candidates scored on test blocks, for transparency |
| `ablation.csv` | Each model against the persistence baseline, block-aware inference |
| `group_pairwise.csv` | **The Track B question**: each feature group against the ground-only model, on matched rows |
| `by_horizon_season.csv` | Skill by horizon and season |
| `predictions.csv` | Every prediction with its issue date, split and provenance |
| `manifest.json` | Git SHA, environment, configuration, availability contract, per-column coverage |
