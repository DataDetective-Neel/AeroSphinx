#!/usr/bin/env python
"""Phase 3 Workstream B - cold-start and extrapolation diagnostics.

Quantifies the distribution-shift vulnerability established in Phase 2 and
tests candidate mitigations.

Outputs, under ``results/runs/<run_id>/``:

* ``cold_start_summary.json``   - the aggregate cold-start picture
* ``extrapolation_conditioned.csv`` - error in-range vs out-of-range
* ``error_vs_out_of_range.csv``  - error against the out-of-range fraction
* ``mitigations.csv``            - candidate mitigations compared
* ``mitigation_by_fold.csv``     - per-fold mitigation behaviour

Usage:
    python scripts/run_cold_start.py [--run-id <id>]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerosphinx.evaluation.block_inference import (  # noqa: E402
    apply_correction,
    moving_block_bootstrap_ci,
    year_blocked_bootstrap_ci,
)
from aerosphinx.evaluation.metrics import evaluate  # noqa: E402
from aerosphinx.evaluation.splits import rolling_origin_folds  # noqa: E402
from aerosphinx.quality.provenance import (  # noqa: E402
    build_provenance,
    write_provenance,
)
from aerosphinx.tracks.cold_start import (  # noqa: E402
    error_vs_out_of_range_fraction,
    extrapolation_conditioned_error,
    persistence_fallback,
    regime_aware_fallback,
    summarise_cold_start,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--features-csv", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    import yaml

    common = yaml.safe_load((root / "config" / "common.yaml").read_text())
    track_a = yaml.safe_load((root / "config" / "track_a.yaml").read_text())
    evaluation = track_a["evaluation"]

    features_csv = args.features_csv or (
        root / common["paths"]["processed_dir"] / "track_a_features.csv"
    )
    frame = pd.read_csv(features_csv, parse_dates=["Date"])

    # Use the corrected Phase 2 run if present, otherwise rebuild the
    # predictions from the folds.
    runs = sorted((root / common["paths"]["results_dir"] / "runs").glob("A-roll*"))
    predictions = None
    for run in reversed(runs):
        candidate = run / "predictions.csv"
        if candidate.exists():
            predictions = pd.read_csv(candidate, parse_dates=["Date"])
            break
    if predictions is None:
        raise SystemExit("No Phase 2 run found; run scripts/run_track_a.py first.")

    print(f"[1/4] analysing {len(predictions)} predictions from {runs[-1].name}")

    # Per-fold training target bounds.
    folds = rolling_origin_folds(
        frame,
        min_train_days=evaluation["min_train_days"],
        validation_days=0,
        test_block_days=evaluation["test_block_days"],
        step_days=evaluation["step_days"],
    )
    train_min, train_max = {}, {}
    for fold in folds:
        train, _val, test = fold.split(frame)
        if train.empty or test.empty:
            continue
        values = pd.to_numeric(train["PM25"], errors="coerce").dropna()
        if values.empty:
            continue
        train_min[fold.index] = float(values.min())
        train_max[fold.index] = float(values.max())
    train_min = pd.Series(train_min)
    train_max = pd.Series(train_max)

    model_columns = [
        c
        for c in predictions.columns
        if c not in ("Date", "Fold", "Target", "seasonal_lookback")
    ]

    summary = summarise_cold_start(predictions, model_columns, train_min, train_max)
    print(
        f"[2/4] cold start: {summary.n_extrapolating_folds} of {summary.n_folds} "
        f"folds extrapolate; {summary.n_observations_out_of_range} observations "
        f"({summary.fraction_out_of_range:.2%}) fall outside the training range"
    )

    conditioned = extrapolation_conditioned_error(
        predictions, model_columns, train_min, train_max
    )
    by_fraction = error_vs_out_of_range_fraction(
        predictions, model_columns, train_min, train_max
    )

    print("\n      error conditional on extrapolation:")
    display = conditioned[
        [
            "model",
            "n",
            "n_out_of_range",
            "mae_in_range",
            "mae_out_of_range",
            "mae_ratio",
            "bias_in_range",
            "bias_out_of_range",
        ]
    ]
    print(display.to_string(index=False))

    # ---- Candidate mitigations -----------------------------------------
    # All mitigations are compared on a **fixed row set** - the rows where
    # every model and the persistence baseline are defined - so the
    # comparison is paired and no mitigation can look better merely by
    # dropping the rows it handles badly.
    fixed = predictions.dropna(
        subset=[m for m in model_columns if m != "persistence"]
        + ["persistence", "Target"]
    ).copy()
    print(f"\n      mitigation comparison on a fixed set of {len(fixed)} rows")

    mitigation_rows = []
    for model_name in model_columns:
        if model_name == "persistence":
            continue
        variants = {
            "none": fixed[model_name].to_numpy(dtype=float),
            "persistence_fallback": persistence_fallback(
                fixed, model_name, train_min=train_min, train_max=train_max
            ),
            "regime_aware_fallback": regime_aware_fallback(fixed, model_name),
        }
        for variant, values in variants.items():
            usable = fixed.copy()
            usable["_variant"] = values
            usable = usable.dropna(subset=["_variant", "Target"])
            if usable.empty:
                continue
            scores = evaluate(
                usable["Target"].to_numpy(), usable["_variant"].to_numpy()
            )
            block = moving_block_bootstrap_ci(
                usable["Target"].to_numpy(),
                usable["_variant"].to_numpy(),
                usable["persistence"].to_numpy(),
                dates=usable["Date"].to_numpy(),
                n_samples=evaluation["bootstrap_samples"],
                alpha=evaluation["alpha"],
                seed=common["random_seed"],
            )
            year_block = year_blocked_bootstrap_ci(
                usable["Target"].to_numpy(),
                usable["_variant"].to_numpy(),
                usable["persistence"].to_numpy(),
                dates=usable["Date"].to_numpy(),
                n_samples=evaluation["bootstrap_samples"],
                alpha=evaluation["alpha"],
                seed=common["random_seed"],
            )
            # How often did this mitigation actually fire?
            if variant == "none":
                n_fired = 0
            elif variant == "persistence_fallback":
                lower = usable["Fold"].map(train_min).to_numpy(dtype=float)
                upper = usable["Fold"].map(train_max).to_numpy(dtype=float)
                original = usable[model_name].to_numpy(dtype=float)
                n_fired = int(np.sum(~((original >= lower) & (original <= upper))))
            else:
                months = usable["Date"].dt.month.to_numpy()
                from aerosphinx.tracks.cold_start import _season_of

                regimes = np.array([_season_of(int(m)) for m in months])
                n_fired = int(np.sum(~np.isin(regimes, ("pre_monsoon", "winter"))))

            mitigation_rows.append(
                {
                    "model": model_name,
                    "mitigation": variant,
                    "n": scores["n"],
                    "n_mitigation_fired": n_fired,
                    "mae": scores["MAE"],
                    "rmse": scores["RMSE"],
                    "bias": scores["Mean_Bias"],
                    "delta_vs_persistence": block["delta"],
                    "block_ci_low": block["ci_low"],
                    "block_ci_high": block["ci_high"],
                    "block_significant": block["significant"],
                    "year_ci_low": year_block["ci_low"],
                    "year_ci_high": year_block["ci_high"],
                    "year_significant": year_block["significant"],
                    "block_length_days": block["block_length"],
                }
            )

    mitigations = pd.DataFrame(mitigation_rows)
    if not mitigations.empty:
        corrections = apply_correction(
            mitigations["delta_vs_persistence"].abs().to_numpy(),
            alpha=evaluation["alpha"],
        )
        mitigations["p_holm"] = corrections["p_holm"].to_numpy()

    print("\n[3/4] candidate mitigations:")
    if not mitigations.empty:
        print(
            mitigations[
                [
                    "model",
                    "mitigation",
                    "n",
                    "n_mitigation_fired",
                    "mae",
                    "bias",
                    "delta_vs_persistence",
                    "block_ci_low",
                    "block_ci_high",
                    "block_significant",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )

    run_id = (
        args.run_id
        or f"B-coldstart-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir = args.out_dir or (root / common["paths"]["results_dir"] / "runs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditioned.to_csv(out_dir / "extrapolation_conditioned.csv", index=False)
    by_fraction.to_csv(out_dir / "error_vs_out_of_range.csv", index=False)
    if not mitigations.empty:
        mitigations.to_csv(out_dir / "mitigations.csv", index=False)

    (out_dir / "cold_start_summary.json").write_text(
        json.dumps(summary.as_dict(), indent=2) + "\n"
    )

    provenance = build_provenance(
        run_id=run_id,
        root=root,
        inputs={
            "features_csv": str(features_csv.relative_to(root)),
            "source_run": runs[-1].name,
        },
        dataset={
            "n_predictions": int(len(predictions)),
            "n_folds": int(predictions["Fold"].nunique()),
        },
        rules={
            "evaluation": evaluation,
            "workstream": "B - cold-start and extrapolation diagnostics",
            "mitigations_tested": [
                "none",
                "persistence_fallback",
                "regime_aware_fallback",
            ],
            "clipping_excluded": True,
        },
        outputs={
            "run_dir": _relative(out_dir, root),
            "files": sorted(p.name for p in out_dir.glob("*")),
        },
    )
    write_provenance(provenance, out_dir / "manifest.json")

    print(f"[4/4] wrote {_relative(out_dir, root)}/")
    return 0


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
