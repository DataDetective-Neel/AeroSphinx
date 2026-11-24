#!/usr/bin/env python
"""Track A experiment runner - Phase 2.

Runs the rolling-origin backtest over the full CPCB record and writes the
results, comparisons and a run manifest.

Outputs, under ``results/runs/<run_id>/``:

* ``metrics.json``          - aggregate metrics, fold summary, metadata
* ``fold_metrics.csv``      - per-fold metrics for every model
* ``predictions.csv``       - per-observation predictions
* ``comparisons.csv``       - paired tests and bootstrap intervals vs baseline
* ``by_season.csv``         - per-season breakdown
* ``by_year.csv``           - per-year breakdown
* ``manifest.json``         - Git SHA, environment, input hashes, configuration

Usage:
    python scripts/run_track_a.py [--root .]
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

from aerosphinx.evaluation.metrics import evaluate  # noqa: E402
from aerosphinx.quality.provenance import (  # noqa: E402
    build_provenance,
    write_provenance,
)
from aerosphinx.tracks.track_a import DEFAULT_FEATURES, run_track_a  # noqa: E402


def _season(dates: pd.Series) -> pd.Series:
    month = dates.dt.month
    return pd.Series(
        [
            (
                "winter"
                if m in (12, 1, 2)
                else (
                    "pre_monsoon"
                    if m in (3, 4, 5)
                    else "monsoon" if m in (6, 7, 8, 9) else "post_monsoon"
                )
            )
            for m in month
        ],
        index=dates.index,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--features-csv", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--baseline", default="persistence")
    parser.add_argument("--min-train-days", type=int, default=None)
    parser.add_argument("--test-block-days", type=int, default=None)
    parser.add_argument("--step-days", type=int, default=None)
    parser.add_argument("--lookback-years", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
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

    min_train_days = args.min_train_days or evaluation["min_train_days"]
    test_block_days = args.test_block_days or evaluation["test_block_days"]
    step_days = args.step_days or evaluation["step_days"]
    lookback_years = args.lookback_years or 3
    bootstrap_samples = args.bootstrap_samples or evaluation["bootstrap_samples"]

    run_id = f"A-roll{run_id_suffix()}"
    print(f"[1/4] running Track A rolling-origin backtest ({run_id})")
    result = run_track_a(
        frame,
        features=DEFAULT_FEATURES,
        models=args.models,
        min_train_days=min_train_days,
        validation_days=(
            evaluation["min_validation_days"]
            if evaluation["min_validation_days"]
            else 0
        ),
        test_block_days=test_block_days,
        step_days=step_days,
        baseline=args.baseline,
        climatology_lookback_years=lookback_years,
        bootstrap_samples=bootstrap_samples,
        alpha=evaluation["alpha"],
        seed=common["random_seed"],
    )

    metadata = result.metadata
    print(
        f"      folds: {metadata['n_folds']}   "
        f"rows evaluated: {metadata['n_rows_evaluated']:,} of "
        f"{metadata['n_rows_total']:,}"
    )

    print("[2/4] aggregate metrics")
    display = result.aggregate[
        ["model", "n", "MAE", "RMSE", "R2", "Mean_Bias", "AQI_Accuracy", "AQI_Macro_F1"]
    ]
    print(display.to_string(index=False))

    print(f"[3/4] paired comparisons against {args.baseline}")
    comparison_display = result.comparisons[
        [
            "model",
            "delta_mae",
            "wilcoxon_pvalue",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "significant",
            "verdict",
        ]
    ]
    print(comparison_display.to_string(index=False))

    # Cold-start attribution: the learned models fail hardest where the test
    # block lies outside the training target range.
    if not result.cold_start.empty:
        extrapolating = result.cold_start[result.cold_start["extrapolating"]]
        print(
            f"\n      regime coverage: {len(extrapolating)} of {len(result.cold_start)} "
            f"folds require extrapolating outside the training target range"
        )
        if not extrapolating.empty:
            delta_columns = [
                c
                for c in extrapolating.columns
                if c.startswith("random_forest")
                or c.startswith("xgboost")
                or c.startswith("ridge")
            ]
            for column in delta_columns:
                if column.endswith(f"_minus_{args.baseline}"):
                    inside = result.cold_start[~result.cold_start["extrapolating"]][
                        column
                    ].mean()
                    outside = extrapolating[column].mean()
                    print(
                        f"      {column:38s} extrapolating {outside:+7.2f} "
                        f"vs in-range {inside:+7.2f}"
                    )

    out_dir = args.out_dir or (root / common["paths"]["results_dir"] / "runs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    result.predictions.to_csv(out_dir / "predictions.csv", index=False)
    result.fold_metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    result.comparisons.to_csv(out_dir / "comparisons.csv", index=False)
    result.fold_summary.to_csv(out_dir / "fold_summary.csv", index=False)
    if not result.coverage.empty:
        result.coverage.to_csv(out_dir / "regime_coverage.csv", index=False)
    if not result.cold_start.empty:
        result.cold_start.to_csv(out_dir / "cold_start.csv", index=False)
    if not result.fold_complete.empty:
        result.fold_complete.to_csv(out_dir / "fold_complete_metrics.csv", index=False)
    if not result.block_inference.empty:
        result.block_inference.to_csv(out_dir / "block_inference.csv", index=False)
        print("\n      block-aware inference (moving-block and year-blocked):")
        block_display = result.block_inference[
            [
                "model",
                "delta_mae",
                "block_length_days",
                "block_ci_low",
                "block_ci_high",
                "year_ci_low",
                "year_ci_high",
                "p_holm",
                "folds_a_better",
            ]
        ]
        print(block_display.to_string(index=False))

    # Seasonal and per-year breakdowns on the complete evaluation rows.
    model_columns = [
        c
        for c in result.predictions.columns
        if c not in ("Date", "Fold", "Target", "seasonal_lookback")
    ]
    complete = result.predictions.dropna(subset=model_columns + ["Target"])
    complete = complete.reset_index(drop=True)

    for label, groups in (
        ("by_season", _season(complete["Date"])),
        ("by_year", complete["Date"].dt.year.astype(str)),
    ):
        rows = []
        for group in sorted(groups.unique()):
            mask = (groups == group).to_numpy()
            y = complete.loc[mask, "Target"].to_numpy()
            for model_name in model_columns:
                rows.append(
                    {
                        "group": group,
                        "model": model_name,
                        **evaluate(y, complete.loc[mask, model_name].to_numpy()),
                    }
                )
        pd.DataFrame(rows).to_csv(out_dir / f"{label}.csv", index=False)

    metrics_payload = {
        "run_id": run_id,
        "metadata": metadata,
        "aggregate": result.aggregate.to_dict(orient="records"),
        "fold_summary": result.fold_summary.to_dict(orient="records"),
        "comparisons": result.comparisons.to_dict(orient="records"),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, default=str) + "\n"
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
            "date_range": metadata["date_range"],
            "rows_evaluated": metadata["n_rows_evaluated"],
            "n_folds": metadata["n_folds"],
        },
        rules={
            "evaluation": evaluation,
            "features": track_a["features"],
            "models": metadata["models"],
            "baseline": args.baseline,
            "climatology_lookback_years": lookback_years,
        },
        outputs={
            "run_dir": _relative(out_dir, root),
            "files": sorted(p.name for p in out_dir.glob("*")),
        },
    )
    write_provenance(provenance, out_dir / "manifest.json")

    print(f"[4/4] wrote {_relative(out_dir, root)}/")
    return 0


def run_id_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    """Path relative to the repository root, or absolute if outside it."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
