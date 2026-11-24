# Results

Everything in this directory is a **persisted artefact**: a file some script
writes, which a claim can then be checked against. Nothing here is hand-edited,
and `aerosphinx check-claims` fails the build if a published number stops
matching what is on disk.

## The canonical set

| Directory | Produced by | What it is |
|---|---|---|
| `CANONICAL.json` | `results/CANONICAL.json` (hand-maintained registry) | One entry per published claim: the artefact, the row, the value, the README token |
| `experiment_matrix/` | `make run-matrix` | The M0–M8 matrix. **This is the canonical evaluation** — 97 rolling-origin folds for Track A, 3 for Track B, every configuration on identical eligible dates |
| `alignment/` | `make run-alignment` | Daily coverage, per-source missingness, alignment report |
| `sufficiency/` | `make run-sufficiency` | Per-season × per-source SUFFICIENT / INSUFFICIENT status |
| `baseline/` | `make audit` | Phase 0 data audit and its manifest |
| `runs/` | individual experiments | One directory per run, named `<track>-<purpose>-<UTC timestamp>`. Exploratory: a run supports a report, it is never itself the published number |
| `legacy_chapter/` | `legacy/aerosphinx_pipeline.py` | Chapter-pipeline outputs on its 33-day test window. Superseded by the matrix; kept because the chapter cites them |

## Why the matrix and not a run

`runs/` contains several Track A backtests that disagree with each other in
sign. That is not a contradiction — they evaluate different eligible sets,
because each was written at a different point in the protocol's evolution. The
matrix is the one that applies the final protocol to every configuration at
once, so it is the only series of numbers the README quotes. If you find
yourself comparing a `runs/` figure with a README figure, you are comparing a
diagnostic with a result.

## Layout conventions

* A run directory is immutable once written. A correction is a new run.
* `manifest.json` in each directory records the Git SHA, the environment, input
  hashes and whether the tree was dirty. A manifest with `dirty: true` means the
  numbers cannot be reproduced from any commit — treat it as a draft.
* CSV floats are written with ten significant digits, so a regeneration in a
  different NumPy or pandas build does not produce a spurious diff.

## Regenerating

```bash
make run-matrix           # the canonical matrix
make check-claims         # README numbers still match the artefacts
make check-reproducibility # re-run the matrix and diff it against what is committed
```
