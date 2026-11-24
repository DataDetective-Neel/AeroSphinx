#!/usr/bin/env python
"""Run the M0-M8 multisource seasonal experiment matrix.

Track A runs M0-M1 on the ground-only record. Track B runs M0-M8 on the
intersection of dates where AOD and MERRA-2 are genuinely observed. All
configurations inside a track share identical eligible dates and identical
rolling-origin folds.

Outputs (``results/experiment_matrix/``):

* ``matrix_metrics.csv``         overall metrics per config
* ``matrix_common_metrics.csv``  metrics on dates every config predicted
* ``matrix_by_season.csv``       MAE/RMSE/R2 per season
* ``matrix_by_year.csv``         stability by year
* ``matrix_fold_metrics.csv``    per-fold metrics
* ``matrix_ablations.csv``       paired MAE differences
* ``predictions.csv``            per-row predictions
* ``matrix_configs.json``        the matrix definition
* ``manifest.json``              provenance and fold boundaries

Usage:
    python scripts/run_experiment_matrix.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerosphinx.quality.provenance import (  # noqa: E402
    build_provenance,
    write_provenance,
)
from aerosphinx.tracks.experiment_matrix import (  # noqa: E402
    MATRIX,
    run_matrix,
)


def _combine(frames):
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return frames[0] if frames else pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--model", default="ridge", choices=["ridge", "random_forest", "xgboost"]
    )
    parser.add_argument("--track-a-min-train", type=int, default=365)
    parser.add_argument("--track-b-min-train", type=int, default=60)
    parser.add_argument("--test-block", type=int, default=30)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--min-specialist-days", type=int, default=30)
    args = parser.parse_args()

    root = args.root.resolve()
    input_csv = args.input or (root / "data" / "processed" / "fused_dataset.csv")
    out_dir = args.out_dir or (root / "results" / "experiment_matrix")
    if not input_csv.exists():
        print(f"ERROR: {input_csv} is missing; run `make run-phase6` first")
        return 1

    frame = pd.read_csv(input_csv, parse_dates=["Date"])
    run_id = args.run_id or (
        f"matrix-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        model_name=args.model,
        test_block_days=args.test_block,
        step_days=args.step,
        min_specialist_days=args.min_specialist_days,
    )
    print("[1/3] Track A: ground-only (M0, M1)")
    track_a = run_matrix(
        frame, "A", ["M0", "M1"], min_train_days=args.track_a_min_train, **common
    )
    print(f"      folds: {len(track_a.folds)}  predictions: {len(track_a.predictions)}")

    print("[2/3] Track B: multisource retrospective (M0-M8)")
    track_b = run_matrix(
        frame, "B", list(MATRIX), min_train_days=args.track_b_min_train, **common
    )
    print(f"      folds: {len(track_b.folds)}  predictions: {len(track_b.predictions)}")

    results = [track_a, track_b]
    artifacts = {
        "matrix_metrics.csv": _combine([r.metrics for r in results]),
        "matrix_common_metrics.csv": _combine([r.common_metrics for r in results]),
        "matrix_by_season.csv": _combine([r.by_season for r in results]),
        "matrix_by_year.csv": _combine([r.by_year for r in results]),
        "matrix_fold_metrics.csv": _combine([r.fold_metrics for r in results]),
        "matrix_ablations.csv": _combine([r.ablations for r in results]),
        "predictions.csv": _combine([r.predictions for r in results]),
    }
    for name, table in artifacts.items():
        # Ten significant digits: enough for every metric this study reports,
        # and stable across pandas/numpy builds. Without it a regeneration in a
        # different environment rewrites every float with a different last
        # digit, which makes the artefact diff unreadable and hides the real
        # change in noise.
        table.to_csv(out_dir / name, index=False, float_format="%.10g")

    (out_dir / "matrix_configs.json").write_text(
        json.dumps(
            {"configs": track_b.configs, "track_a": track_a.configs},
            indent=2,
            default=str,
        )
        + "\n"
    )

    print("[3/3] writing artefacts")
    for name in artifacts:
        print(f"      {name}")

    provenance = build_provenance(
        run_id=run_id,
        root=root,
        # The run writes its own artefacts; only source changes count as dirt.
        ignore_dirty=(
            (
                str(out_dir.relative_to(root))
                if out_dir.is_relative_to(root)
                else str(out_dir)
            ),
        ),
        inputs=(
            {"fused_dataset": str(input_csv.relative_to(root))}
            if input_csv.is_relative_to(root)
            else {"fused_dataset": str(input_csv)}
        ),
        dataset={
            "rows": int(len(frame)),
            "track_a_folds": len(track_a.folds),
            "track_b_folds": len(track_b.folds),
            "track_a_predictions": int(len(track_a.predictions)),
            "track_b_predictions": int(len(track_b.predictions)),
        },
        rules={
            "phase": "6-11 - M0-M8 experiment matrix",
            "configs": track_b.configs,
            "model": args.model,
            "min_train_days_track_a": args.track_a_min_train,
            "min_train_days_track_b": args.track_b_min_train,
            "test_block_days": args.test_block,
            "step_days": args.step,
            "min_specialist_days": args.min_specialist_days,
            "evaluation": "identical eligible dates within each track",
            "track_b_note": (
                "retrospective only: AOD and MERRA-2 are same-day observations "
                "(D-002); never an operational forecast"
            ),
        },
        outputs={
            "run_dir": (
                str(out_dir.relative_to(root))
                if out_dir.is_relative_to(root)
                else str(out_dir)
            ),
            "files": sorted(path.name for path in out_dir.glob("*")),
        },
    )
    write_provenance(provenance, out_dir / "manifest.json")

    print("\nTrack A metrics")
    print(
        track_a.metrics.to_string(index=False)
        if not track_a.metrics.empty
        else "  none"
    )
    print("\nTrack B metrics")
    print(
        track_b.metrics.to_string(index=False)
        if not track_b.metrics.empty
        else "  none"
    )
    print(f"\nwrote {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
