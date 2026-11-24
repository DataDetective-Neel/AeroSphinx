#!/usr/bin/env python
"""Phase 3 Workstream C - regime-conditional modelling, ablation and routing.

Executes the expanded Workstream C scope:

1. Global versus seasonal models.
2. Seasonal interaction features.
3. Horizon-specific versus shared models.
4. Direct versus recursive forecasting.
5. Feature-group ablations.
6. Regime-aware model routing.
7. Minimum sample requirements and a validated fallback.

Outputs, under ``results/runs/<run_id>/``:

* ``architecture_validation.csv`` - candidate scores on validation blocks
* ``architecture_test.csv``      - the same candidates scored on test blocks
* ``selection.csv``              - the architecture chosen on validation
* ``ablation.csv``               - feature-group comparison, block-aware
* ``by_horizon_season.csv``      - breakdown by horizon and season
* ``predictions.csv``            - every prediction with its provenance
* ``manifest.json``              - Git SHA, environment, configuration

Usage:
    python scripts/run_workstream_c.py [--model ridge] [--horizons 1 2 3 4 5]
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

from aerosphinx.features.groups import (  # noqa: E402
    GROUND_ONLY_GROUPS,
    assert_operational_contract,
    describe_groups,
    partition_by_availability,
)
from aerosphinx.ingestion.aod import load_aod_daily  # noqa: E402
from aerosphinx.ingestion.merra import load_merra_daily  # noqa: E402
from aerosphinx.ingestion.station_met import load_station_met_daily  # noqa: E402
from aerosphinx.quality.provenance import (  # noqa: E402
    build_provenance,
    write_provenance,
)
from aerosphinx.tracks.regime import assign_regime  # noqa: E402
from aerosphinx.tracks.workstream_c import (  # noqa: E402
    build_feature_frame,
    run_workstream_c,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--methods", nargs="*", default=["direct", "recursive"])
    parser.add_argument(
        "--architectures",
        nargs="*",
        default=["global", "seasonal", "interaction", "routing", "shared"],
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    common = yaml.safe_load((root / "config" / "common.yaml").read_text())
    quality = yaml.safe_load((root / "config" / "data_quality.yaml").read_text())
    evaluation = yaml.safe_load((root / "config" / "track_a.yaml").read_text())[
        "evaluation"
    ]

    print("[1/4] building the feature frame")
    # The daily file carries the target and coverage; the temporal and
    # calendar features are rebuilt here so that the feature contract is
    # defined in one place.
    features_csv = root / common["paths"]["processed_dir"] / "track_a_features.csv"
    daily = pd.read_csv(features_csv, parse_dates=["Date"])

    station_coords = common["project"]["station"]
    station = load_station_met_daily(
        root / common["paths"]["cpcb_dir"],
        valid_ranges={
            key: tuple(value)
            for key, value in quality["cpcb"]["station_met_valid_ranges"].items()
        },
        excluded_columns=tuple(quality["cpcb"]["station_met_excluded_columns"]),
    )
    merra = load_merra_daily(root / common["paths"]["merra_dir"])
    aod = load_aod_daily(
        root / common["paths"]["aod_dir"],
        station_lat=station_coords["latitude"],
        station_lon=station_coords["longitude"],
    )

    frame = build_feature_frame(daily, station.frame, merra.frame, aod.frame)
    print(f"      {len(frame)} daily rows, {len(frame.columns)} columns")
    print(f"      station_met: {station.coverage['n_days_accepted']} days accepted")
    print(f"      merra: {len(merra.frame)} days")
    print(
        f"      aod: {len(aod.frame)} days "
        f"({int((aod.frame['aod_valid_pixels'] == 0).sum())} with no valid pixel)"
    )

    group_sets = {
        "ground_only": GROUND_ONLY_GROUPS,
        "ground_plus_station_met": GROUND_ONLY_GROUPS + ("station_met",),
        "ground_plus_merra": GROUND_ONLY_GROUPS + ("merra",),
        "ground_plus_aod": GROUND_ONLY_GROUPS + ("aod",),
    }

    # The operational contract: only the ground-only set qualifies.
    operational, retrospective = partition_by_availability(
        [g for groups in group_sets.values() for g in groups]
    )
    print(
        f"\n      availability: {len(set(operational))} operationally eligible "
        f"group(s), {len(set(retrospective))} retrospective-only"
    )
    assert_operational_contract(GROUND_ONLY_GROUPS)
    print("      ground_only satisfies the operational feature contract")

    print(
        f"\n[2/4] running the experiment "
        f"({len(group_sets)} group sets x {len(args.architectures)} architectures "
        f"x {len(args.horizons)} horizons)"
    )
    result = run_workstream_c(
        frame,
        group_sets=group_sets,
        architectures=tuple(args.architectures),
        model_name=args.model,
        horizons=tuple(args.horizons),
        methods=tuple(args.methods),
        bootstrap_samples=args.bootstrap_samples,
        alpha=evaluation["alpha"],
        seed=common["random_seed"],
    )

    print("[3/4] architecture selection (on validation blocks)")
    selection = result.selection
    if not selection.empty:
        chosen = selection[selection.get("selected", False) == True]  # noqa: E712
        print("      selected per horizon (chosen on validation MAE alone):")
        print(
            chosen[["group_set", "architecture", "method", "horizon", "n", "MAE"]]
            .round(3)
            .to_string(index=False)
        )
        print("\n      their test performance:")
        if not result.selected_performance.empty:
            print(
                result.selected_performance[
                    [
                        "horizon",
                        "group_set",
                        "architecture",
                        "method",
                        "n_test",
                        "test_MAE",
                        "test_RMSE",
                        "test_R2",
                        "persistence_MAE",
                        "delta_vs_persistence",
                    ]
                ]
                .round(3)
                .to_string(index=False)
            )
        print("\n      top 6 candidates per horizon by validation MAE:")
        for horizon in sorted(selection["horizon"].unique()):
            top = selection[selection["horizon"] == horizon].head(3)
            print(
                f"        h={horizon}: "
                + ", ".join(
                    f"{r.group_set}/{r.architecture}/{r.method} {r.MAE:.2f}"
                    for _, r in top.iterrows()
                )
            )

    print("\n      feature-group ablation on test blocks:")
    if not result.ablation.empty:
        print(
            result.ablation[
                [
                    "group_set",
                    "architecture",
                    "method",
                    "horizon",
                    "n_matched",
                    "sufficient_evidence",
                    "mae_model",
                    "baseline_mae",
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
        or f"C-regime-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir = args.out_dir or (root / common["paths"]["results_dir"] / "runs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    result.architecture_validation.to_csv(
        out_dir / "architecture_validation.csv", index=False
    )
    result.architecture_test.to_csv(out_dir / "architecture_test.csv", index=False)
    selection.to_csv(out_dir / "selection.csv", index=False)
    result.ablation.to_csv(out_dir / "ablation.csv", index=False)
    if not result.group_pairwise.empty:
        result.group_pairwise.to_csv(out_dir / "group_pairwise.csv", index=False)
        print("\n      DIRECT group-vs-group comparison (does the environmental group")
        print("      add value on top of the ground-only model?):")
        gp = result.group_pairwise
        print(
            gp[
                [
                    "group_set",
                    "architecture",
                    "method",
                    "horizon",
                    "n_matched",
                    "sufficient_evidence",
                    "mae_with_group",
                    "mae_reference",
                    "delta_vs_reference",
                    "block_ci_low",
                    "block_ci_high",
                    "block_significant",
                    "group_better",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )
    result.predictions.to_csv(out_dir / "predictions.csv", index=False)
    if not result.selected_performance.empty:
        result.selected_performance.to_csv(
            out_dir / "selected_performance.csv", index=False
        )

    # Breakdown by horizon and season, on the test blocks.
    test_predictions = result.predictions[result.predictions["split"] == "test"].copy()
    if not test_predictions.empty:
        test_predictions["season"] = assign_regime(test_predictions["Date"])
        breakdown = (
            test_predictions.groupby(
                ["group_set", "architecture", "method", "horizon", "season"]
            )
            .apply(
                lambda g: pd.Series(
                    {
                        "n": int(len(g)),
                        "mae": float((g["predicted"] - g["actual"]).abs().mean()),
                        "baseline_mae": float(
                            (g["baseline"] - g["actual"]).abs().mean()
                        ),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        breakdown.to_csv(out_dir / "by_horizon_season.csv", index=False)

    (out_dir / "metrics.json").write_text(
        json.dumps(result.as_dict(), indent=2, default=str) + "\n"
    )

    provenance = build_provenance(
        run_id=run_id,
        root=root,
        inputs={
            "daily_csv": str(
                (
                    root / common["paths"]["processed_dir"] / "track_a_daily.csv"
                ).relative_to(root)
            ),
            "cpcb_dir": str((root / common["paths"]["cpcb_dir"]).relative_to(root)),
            "merra_dir": str((root / common["paths"]["merra_dir"]).relative_to(root)),
        },
        dataset={
            "rows": int(len(frame)),
            "date_range": [
                frame["Date"].min().date().isoformat(),
                frame["Date"].max().date().isoformat(),
            ],
            "station_met_coverage": station.coverage,
            "merra_timesteps_utc": merra.timesteps_utc,
            "merra_grid_cell": list(merra.grid_cell),
        },
        rules={
            "evaluation": evaluation,
            "workstream": "C - regime-conditional modelling, ablation, routing",
            "configuration": result.metadata,
            "feature_groups": describe_groups(
                sorted({g for groups in group_sets.values() for g in groups})
            ),
            "availability_contract": {
                "ground_only": "operationally eligible",
                "station_met": "retrospective only - availability timestamp UNKNOWN",
                "merra": "retrospective only - reanalysis is never operational",
            },
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
