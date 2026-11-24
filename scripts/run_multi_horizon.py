#!/usr/bin/env python
"""Phase 3 Workstream A - multi-horizon forecasting (days 1-5).

The central experiment of Phase 3: does skill relative to persistence grow
with forecast horizon?

Outputs, under ``results/runs/<run_id>/``:

* ``forecasts.csv``       - every forecast with its issue date
* ``metrics.csv``        - per-horizon, per-method, per-model metrics
* ``comparisons.csv``    - paired comparisons vs the horizon-appropriate baseline
* ``by_horizon.csv``     - skill summary across horizons
* ``manifest.json``      - Git SHA, environment, configuration

Usage:
    python scripts/run_multi_horizon.py [--models ridge random_forest xgboost]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerosphinx.quality.provenance import (  # noqa: E402
    build_provenance,
    write_provenance,
)
from aerosphinx.tracks.multi_horizon import run_multi_horizon  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--features-csv", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--methods", nargs="*", default=["direct", "recursive"])
    parser.add_argument("--min-train-days", type=int, default=None)
    parser.add_argument("--test-block-days", type=int, default=None)
    parser.add_argument("--step-days", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    common = yaml.safe_load((root / "config" / "common.yaml").read_text())
    track_a = yaml.safe_load((root / "config" / "track_a.yaml").read_text())
    evaluation = track_a["evaluation"]

    features_csv = args.features_csv or (
        root / common["paths"]["processed_dir"] / "track_a_features.csv"
    )
    if not features_csv.exists():
        raise SystemExit(
            f"{features_csv} not found. Run scripts/build_track_a_dataset.py first."
        )
    frame = pd.read_csv(features_csv, parse_dates=["Date"])

    print(f"[1/3] multi-horizon backtest over horizons {args.horizons}")
    result = run_multi_horizon(
        frame,
        models=args.models,
        horizons=tuple(args.horizons),
        methods=tuple(args.methods),
        min_train_days=args.min_train_days or evaluation["min_train_days"],
        test_block_days=args.test_block_days or evaluation["test_block_days"],
        step_days=args.step_days or evaluation["step_days"],
        bootstrap_samples=args.bootstrap_samples or evaluation["bootstrap_samples"],
        alpha=evaluation["alpha"],
        seed=common["random_seed"],
    )
    print(
        f"      {result.metadata['n_forecasts']:,} forecasts over "
        f"{result.metadata['n_folds']} folds"
    )

    print("[2/3] per-horizon skill against the horizon-appropriate baseline")
    comparison_display = (
        result.comparisons[
            [
                "horizon",
                "method",
                "model",
                "n",
                "mae_model",
                "mae_baseline",
                "delta_mae",
                "block_ci_low",
                "block_ci_high",
                "block_significant",
            ]
        ]
        if not result.comparisons.empty
        else pd.DataFrame()
    )
    if not comparison_display.empty:
        print(comparison_display.round(3).to_string(index=False))

    # Skill summary: does the gap close with horizon?
    if not result.comparisons.empty:
        summary = (
            result.comparisons.groupby("horizon")
            .agg(
                n_comparisons=("model", "size"),
                best_delta_mae=("delta_mae", "min"),
                median_delta_mae=("delta_mae", "median"),
            )
            .reset_index()
        )
        significant = result.comparisons[result.comparisons["block_significant"]]
        better = significant[significant["delta_mae"] < 0].groupby("horizon").size()
        worse = significant[significant["delta_mae"] > 0].groupby("horizon").size()
        summary["n_significantly_better"] = (
            summary["horizon"].map(better).fillna(0).astype(int)
        )
        summary["n_significantly_worse"] = (
            summary["horizon"].map(worse).fillna(0).astype(int)
        )
        summary = (
            summary.drop(columns=["n_significantly_better_x"])
            if "n_significantly_better_x" in summary.columns
            else summary
        )
        print("\n      skill against persistence by horizon:")
        print(summary.round(3).to_string(index=False))

    run_id = (
        args.run_id
        or f"A-horizon-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir = args.out_dir or (root / common["paths"]["results_dir"] / "runs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    result.forecasts.to_csv(out_dir / "forecasts.csv", index=False)
    result.metrics.to_csv(out_dir / "metrics.csv", index=False)
    result.comparisons.to_csv(out_dir / "comparisons.csv", index=False)
    if not result.lag_reference.empty:
        result.lag_reference.to_csv(out_dir / "persistence_by_lag.csv", index=False)
        print("\n      confound-free persistence reference (identical target dates):")
        print(result.lag_reference.round(3).to_string(index=False))
    if not result.within_issue.empty:
        result.within_issue.to_csv(out_dir / "within_issue.csv", index=False)
        # The within-issue comparison removes target-date difficulty.
        within_summary = (
            result.within_issue.groupby(["method", "model", "horizon"])
            .agg(
                n=("model_vs_h1", "size"),
                model_error_increase=("model_vs_h1", "mean"),
                baseline_error_increase=("baseline_vs_h1", "mean"),
                model_minus_baseline=("model_minus_baseline", "mean"),
            )
            .reset_index()
        )
        within_summary.to_csv(out_dir / "within_issue_summary.csv", index=False)
        print("\n      within-issue-date comparison (removes target-date difficulty):")
        print(within_summary.round(3).to_string(index=False))
    if not result.comparisons.empty:
        summary.to_csv(out_dir / "by_horizon.csv", index=False)

    (out_dir / "metrics.json").write_text(
        json.dumps(result.as_dict(), indent=2, default=str) + "\n"
    )

    provenance = build_provenance(
        run_id=run_id,
        root=root,
        inputs={
            "features_csv": str(features_csv.relative_to(root)),
            "features_sha256": _sha256(features_csv),
        },
        dataset={
            "rows": int(len(frame)),
            "date_range": [
                frame["Date"].min().date().isoformat(),
                frame["Date"].max().date().isoformat(),
            ],
        },
        rules={
            "evaluation": evaluation,
            "workstream": "A - multi-horizon forecasting",
            "configuration": result.metadata,
        },
        outputs={
            "run_dir": _relative(out_dir, root),
            "files": sorted(p.name for p in out_dir.glob("*")),
        },
    )
    write_provenance(provenance, out_dir / "manifest.json")

    print(f"[3/3] wrote {_relative(out_dir, root)}/")
    return 0


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
