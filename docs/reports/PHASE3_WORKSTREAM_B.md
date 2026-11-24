# AeroSphinx — Phase 3 Workstream B Report: Cold-Start and Extrapolation

**Workstream B complete.** All 96 folds retained, per the B-011 decision.

Reproduce:

```bash
python scripts/run_track_a.py        # corrected baseline
python scripts/run_cold_start.py      # this workstream
```

Run directory: `results/runs/B-coldstart-20261112T164037Z/`

---

## 1. Headline

> **The cold-start failure is real, precisely characterised, and only one
> mitigation works — but the mitigation that works is not the one that
> targets the failure.**

| Model | MAE in range | MAE out of range | Ratio | Bias in range | Bias out of range |
|---|---|---|---|---|---|
| Persistence | 21.86 | 15.86 | 0.73 | −0.00 | −4.85 |
| Seasonal naive | 37.94 | 56.76 | 1.50 | +7.11 | +30.71 |
| Ridge | 22.84 | 42.55 | 1.86 | +0.90 | +9.07 |
| Random Forest | 23.03 | **77.48** | **3.36** | +2.85 | **+50.49** |
| XGBoost | 24.34 | **95.77** | **3.93** | +3.35 | **+71.39** |

**22 observations (0.82%)** across 8 of 96 folds fall outside the training
target range. On those observations the tree ensembles' error is **3.4x and
3.9x** their in-range error, with a large positive bias — they predict far
too high.

**Persistence is *better* out of range than in range** (ratio 0.73). It has
no structural limitation: it copies yesterday's value, whatever it was.

---

## 2. The mechanism, precisely

The Phase 2 report attributed the cold-start failure to a tree ensemble
being unable to predict outside its training target range. **That
explanation is incomplete, and the correction matters.**

Direct measurement:

| | |
|---|---|
| Observations outside the training range | 22 |
| RF **predictions** outside the training range | **0** |
| XGBoost predictions outside the training range | **0** |
| Ridge predictions outside the training range | 7 |

**The tree ensembles never predict outside the training range.** They are
bounded by it — that part of the explanation holds. But the failure is that
they predict **inside** the range and are still 50–71 µg/m³ too high:

> On the out-of-range observations, Random Forest's mean prediction is
> **100.6 µg/m³** against a training minimum of **19.2 µg/m³** and a bias of
> **+50.5 µg/m³**.

The model cannot reach the low-pollution regime at all. The training data
in those folds contains only a handful of sub-30 µg/m³ observations, so the
ensemble has no leaf in that region and defaults to the nearest thing it
knows.

---

## 3. Why the obvious mitigation fails

### Persistence fallback on the prediction range

Falls back to persistence when the model's prediction lies outside the
training range. **It fires 0 times for Random Forest and XGBoost and 7
times for ridge** — because the models never leave the range.

| Model | Mitigation fired | MAE |
|---|---|---|
| Random Forest | 0 / 2510 | 22.985 (unchanged) |
| XGBoost | 0 / 2510 | 24.519 (unchanged) |

The failure is operationally **undetectable**: at forecast time nothing
signals that the model is about to be 50 µg/m³ too high.

### Quantile-triggered fallback

A more promising trigger: fall back when the prediction falls outside the
training target's 5th–95th percentile. But the diagnostic is poor:

| Model | Triggers on | True positives | False triggers |
|---|---|---|---|
| Ridge | 81 rows | 9 / 22 | 72 |
| Random Forest | 73 rows | **2 / 22** | 71 |
| XGBoost | 86 rows | **2 / 22** | 84 |

It detects 2 of 22 genuine failures while misfiring on 71 in-range rows. A
mitigation that fires on 71 wrong rows to catch 2 right ones is not usable.

### Clipping — excluded by the protocol

Clipping to the training range bounds the output but replaces a wrong high
prediction with a wrong low one. In the low-pollution extrapolation case
that is *worse*, not better. It is excluded as a primary fix, per the
protocol.

---

## 4. The mitigation that works: regime-aware fallback

**Approach.** Use the learned model in the regimes where Phase 2 showed it
beats persistence (pre-monsoon, winter), and persistence elsewhere
(monsoon, post-monsoon).

**No leakage.** The regime is a deterministic function of the target
date's **calendar month**, which is known in advance. No target value, no
future observation and no measured pollution quantity enters the routing.
This is asserted by
`tests/unit/test_cold_start.py::test_regime_fallback_uses_calendar_only`,
which changes the target column and verifies the routing does not change.

**Fixed row set.** All mitigations are compared on the **same 2,510 rows**,
so a mitigation cannot look better merely by dropping the rows it handles
badly.

| Model | Mitigation | Fired | MAE | Bias | Δ vs persistence | Block CI | Significant |
|---|---|---|---|---|---|---|---|
| Ridge | none | 0 | 22.175 | +0.42 | +0.345 | [−0.32, +1.02] | No |
| Ridge | persistence fallback | 7 | 22.168 | +0.43 | +0.337 | [−0.33, +1.01] | No |
| Ridge | **regime-aware** | 1244 | **21.586** | **−0.22** | **−0.245** | [−0.75, +0.27] | No |
| Random Forest | none | 0 | 22.985 | +3.03 | +1.155 | [−0.14, +2.92] | No |
| Random Forest | persistence fallback | 0 | 22.985 | +3.03 | +1.155 | [−0.14, +2.92] | No |
| Random Forest | **regime-aware** | 1244 | **21.509** | **+0.64** | **−0.322** | [−0.83, +0.16] | No |
| XGBoost | none | 0 | 24.519 | +3.60 | +2.688 | [+0.86, +5.28] | **Yes (worse)** |
| XGBoost | persistence fallback | 0 | 24.519 | +3.60 | +2.688 | [+0.86, +5.28] | **Yes (worse)** |
| XGBoost | **regime-aware** | 1244 | **22.339** | **+1.08** | **+0.508** | [−0.12, +1.17] | **No** |

**What it achieves:**

1. **Every model's bias collapses** — Random Forest's +3.03 becomes +0.64.
   The learned models' positive bias was concentrated in the regimes where
   persistence is stronger.
2. **Random Forest's point estimate now edges ahead of persistence**
   (MAE 21.509 vs 21.831), with a block-aware interval that includes zero
   ([−0.83, +0.16]). It is **not** a demonstrated improvement.
3. **XGBoost's significant deficit disappears** — from significantly worse
   to not significantly different.
4. **Ridge's deficit disappears** and its point estimate also edges ahead
   (21.586 vs 21.831).

**Honest reading.** The regime-aware fallback produces a *better point
estimate* for all three models and removes the two significant deficits.
But **no model is significantly better than persistence under block-aware
inference.** The fallback improves the models to the point of parity with
persistence; it does not produce a demonstrated win.

The regime split (winter, pre-monsoon) was **derived from the Phase 2
per-season table**, i.e. from knowledge of where the models work. That is a
form of selection informed by the evaluation, and the protocol's Workstream
C requires that regime configuration be chosen on **validation folds**. This
workstream is therefore an *experiment* that establishes the direction; the
properly selected version belongs to Workstream C.

---

## 5. Statistical methodology notes

### The block length is estimated from the *difference* series

For a paired comparison the quantity of interest is the difference between
the two models' errors. The shared target autocorrelation **cancels** in that
difference, so the difference is less autocorrelated than either error
series. Blocking on the difference is therefore correct — blocking on the
individual errors would over-block.

This is a subtlety worth stating, because the Phase 2 audit's autocorrelation
figures (lag-1 +0.31) were for the *error* series, not the difference.

### The estimator ignores sampling noise

A fixed autocorrelation threshold of 0.05 flags roughly half the lags of a
genuinely independent series as "correlated" (the standard error of a sample
autocorrelation is ≈ 1/√n). The estimator therefore requires a lag to
exceed both the threshold and a Bonferroni-adjusted noise band. White noise
now yields a block length of 1; an AR(0.85) series yields 61.

### Three results under three inference methods

| Model | Naive day-level | Moving-block | Year-blocked | Holm |
|---|---|---|---|---|
| Ridge | p=0.128 | [−0.32, +1.02] | [−0.43, +1.48] | p=0.256 |
| Random Forest | **p=0.032 (sig.)** | [−0.14, +2.92] | [−0.48, +4.07] | p=0.095 |
| XGBoost | p<1e-5 (sig.) | [+0.86, +5.28] | [+0.36, +7.07] | p<1e-5 |

**The naive test called Random Forest significantly worse; the block-aware
and corrected tests do not.** This is the audit's D-018 finding made
concrete.

---

## 6. What this workstream establishes

1. **The cold-start vulnerability is confirmed and quantified** — 22
   observations, 8 folds, error ratios of 1.9x–3.9x, biases up to +71.
2. **The mechanism is refined.** It is not that trees predict outside their
   range; it is that they **cannot reach** the low-pollution regime and
   default to a high value. The failure is a coverage failure, not a
   bounding failure.
3. **Detection at forecast time is the hard part.** Neither the prediction
   range nor the training quantile reliably identifies the at-risk rows.
   The failure is operationally silent.
4. **Regime-aware routing is the only tested mitigation that works**, and it
   works for a different reason than it was designed for: it removes the
   *bias* rather than catching the *extreme cases*.
5. **A useful negative result:** mitigation that targets the diagnosed
   mechanism (range-based fallback) does not work, because the mechanism was
   misdiagnosed. Mitigation guided by *where the model is good* works.

---

## 7. Open questions for Workstream C

| ID | Question |
|---|---|
| **B-013** | Should the regime split (winter/pre-monsoon vs the rest) be re-derived on validation folds, so the routing is not informed by the test-period per-season table? |
| **B-014** | Is a **continuous** regime weighting better than a hard split — for example, blending model and persistence by a season-dependent weight? |
| **B-015** | Can the *coverage* failure be addressed directly — by weighting the training set toward rare regimes, or by fitting a model on the training distribution's lower tail? |
| **B-016** | Does the regime-aware fallback still help at multi-day horizons, where persistence degrades? This is the key question for Workstream A. |

---

## 8. Artifacts

| File | Contents |
|---|---|
| `cold_start_summary.json` | Aggregate cold-start picture |
| `extrapolation_conditioned.csv` | In-range vs out-of-range error per model |
| `error_vs_out_of_range.csv` | Error against the out-of-range fraction |
| `mitigations.csv` | Mitigation comparison with block-aware inference |
| `manifest.json` | Git SHA, environment, configuration, source run |
