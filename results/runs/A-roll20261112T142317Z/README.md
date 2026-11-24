# Run: A-roll20261112T142317Z

**This run is the audited Phase 2 baseline and is SUPERSEDED.**

It is preserved intact, per the Phase 3 authorization's requirement to
"preserve existing metrics and outputs" and not to "overwrite the accepted
baseline".

## Why it is superseded

The Phase 3 audit (`docs/reports/PHASE3_AUDIT.md`) found two defects in the
evaluation that this run exhibits:

1. **D-016** — the headline evaluation set required *every* model,
   including the seasonal-naive baseline, to be defined. The baseline's
   climatology could not cover every day-of-year from a partial training
   year, so 409 rows were removed from the evaluation of **all** models.
   15 of 96 folds lost the baseline entirely. The evaluation set was
   therefore 2,150 rows instead of 2,510, and the removed rows
   concentrated in exactly the early cold-start folds where the learned
   models fail hardest — **understating the cold-start problem**.

2. **D-018** — the bootstrap resampled individual days, treating
   serially correlated errors (lag-1 autocorrelation +0.31) as independent,
   which understates the interval width.

## The corrected run

`../A-roll20261112T163107Z/` contains the corrected evaluation:
2,510 rows, complete climatology coverage, and block-aware inference.

## What changed in the conclusion

| | This run (audited) | Corrected run |
|---|---|---|
| Evaluation rows | 2,150 | 2,510 |
| Persistence MAE | 21.411 | 21.831 |
| Random Forest MAE | 21.398 | 22.985 |
| Ridge MAE | 21.511 | 22.175 |
| XGBoost MAE | 22.376 | 24.519 |
| RF vs persistence | "no significant difference" | "not significantly worse" (block-aware) |

The corrected evaluation makes the learned models look **worse**, because
the set that was removed was the set where they failed hardest. **The
audit strengthened the Phase 2 conclusion**: no learned model significantly
beats persistence, and under block-aware inference Random Forest is not
even significantly *worse*.

## Metrics in this directory

The files here are the audited artefacts and are **not** regenerated. They
describe the evaluation as it was run, including the defect.
