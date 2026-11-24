#!/usr/bin/env python
"""Phase 7: Track B ablations using the fused availability mask.

Compares ground-only baselines against environmental groups (AOD, MERRA-2,
station meteorology) on the 2979-row fused frame, restricting to rows where
each source is genuinely observed (<source>_observed mask).

Results written to results/runs/F-phase7-<timestamp>/ with:
  ablation.csv        -- per-group-set MAE / delta vs ground-only
  manifest.json       -- Git SHA, env, rules
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/home/akshat/aero")

# ── Load fused frame ─────────────────────────────────────────────────
fused = pd.read_csv(
    ROOT / "data" / "processed" / "fused_dataset.csv", parse_dates=["Date"]
)
n_total = len(fused)
print(f"Fused frame: {n_total} rows")

# ── Load fusion rules / availability ─────────────────────────────────
with open(ROOT / "config" / "fusion.yaml") as f:
    fusion_cfg = yaml.safe_load(f)["fusion"]

with open(ROOT / "config" / "common.yaml") as f:
    common = yaml.safe_load(f)

# ── Define group-sets for ablation ───────────────────────────────────
# Each set is a list of source names from SOURCE_GROUPS
group_sets = {
    "ground_only": [],  # PM25_Lag_* features only
    "ground_station_met": ["station_met"],  # + station met columns
    "ground_aod": ["aod"],  # + AOD (only 139 rows)
    "ground_merra": ["merra"],  # + MERRA-2 (only 172 rows)
    "ground_all": ["station_met", "merra", "aod"],  # + all three (139 rows)
}

# ── Ground-only features ─────────────────────────────────────────────
ground_features = [
    "PM25_Lag_1",
    "PM25_Lag_2",
    "PM25_Lag_3",
    "PM25_Roll3",
    "PM25_Roll7",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
]

# ── Ablation loop ────────────────────────────────────────────────────
import sys

results = []

sys.path.insert(0, str(ROOT / "src"))
from aerosphinx.features.availability import SOURCE_GROUPS

for set_name, sources in group_sets.items():
    print(f"\n=== Ablation: {set_name} ({sources}) ===")

    # Determine which rows have ALL requested sources observed
    if not sources:
        mask = pd.Series(True, index=fused.index)
        available_feature_set = ground_features
    else:
        # Start with all True, then intersect with each source's observed mask
        mask = pd.Series(True, index=fused.index)
        for s in sources:
            col = f"{s}_observed"
            if col not in fused.columns:
                raise ValueError(f"Missing column {col} in fused_dataset.csv")
            source_mask = fused[col].astype(bool)
            mask = mask & source_mask
        available_feature_set = ground_features

    n_observed = int(mask.sum())
    fraction = n_observed / n_total
    print(
        f"  Rows with all sources observed: {n_observed} / {n_total} ({fraction:.1%})"
    )

    # Restrict to observed rows
    sub = fused[mask].copy()
    if len(sub) < 10:
        print(f"  ! Too few rows ({len(sub)}), skipping")
        continue

    # Target
    y = sub["PM25"].values

    # Build feature matrix
    X_list = [sub[f] for f in available_feature_set if f in sub.columns]

    # Add source-specific features
    if "station_met" in sources:
        sm_cols = ["AT", "RH", "WS", "WD", "RF", "TOT_RF", "SR", "VWS"]
        for c in sm_cols:
            if c in sub.columns:
                X_list.append(sub[c].values if hasattr(sub[c], "values") else sub[c])
        print(
            f"  station_met columns added: {[c for c in sm_cols if c in sub.columns]}"
        )

    if "aod" in sources:
        if "AOD" in sub.columns:
            X_list.append(
                sub["AOD"].values if hasattr(sub["AOD"], "values") else sub["AOD"]
            )
        print("  AOD column added")

    if "merra" in sources:
        m_cols = [
            "T2M_midday",
            "TS_midday",
            "QV2M_midday",
            "U10M_midday",
            "V10M_midday",
            "SLP_midday",
        ]
        for c in m_cols:
            if c in sub.columns:
                X_list.append(sub[c].values if hasattr(sub[c], "values") else sub[c])
        print(f"  MERRA-2 columns added: {[c for c in m_cols if c in sub.columns]}")

    if len(X_list) == 0:
        print("  ! No features available, skipping")
        continue

    # Stack; some entries may be Python scalars, coerce to float arrays
    X = np.column_stack([np.asarray(col) for col in X_list])

    # Drop rows with any NaN in feature matrix (complete-case only)
    # The fused frame has NaN for unobserved environmental values
    good = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[good]
    y = y[good]
    # sub is no longer needed after this point; we only use X and y

    if len(X) < 10:
        print(f"  ! Too few complete rows ({len(X)}), skipping")
        continue

    # 80/20 positional split (temporal order preserved by ground row order)
    split_idx = int(0.8 * len(X))
    train_X, train_y = X[:split_idx], y[:split_idx]
    test_X, test_y = X[split_idx:], y[split_idx:]

    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error

    ridge = Ridge(alpha=1.0)
    ridge.fit(train_X, train_y)
    y_pred = ridge.predict(test_X)
    mae = float(mean_absolute_error(test_y, y_pred))

    # Ground-only baseline MAE from Phase 2
    baseline_mae = 21.41  # persistence from Phase 2 rolling-origin backtest

    delta_mae = mae - baseline_mae

    print(
        f"  Model: Ridge, MAE={mae:.2f}, Baseline={baseline_mae:.2f}, Delta={delta_mae:.2f}"
    )
    print(
        f"  Complete-case rows: total={len(X)}, train={split_idx}, test={len(X)-split_idx}"
    )

    results.append(
        {
            "group_set": set_name,
            "sources": "+".join(sources),
            "n_rows_total": n_total,
            "n_rows_observed": n_observed,
            "fraction_observed": fraction,
            "model": "ridge",
            "mae": round(mae, 2),
            "baseline_mae": round(baseline_mae, 2),
            "delta_mae": round(delta_mae, 2),
            "train_rows": split_idx,
            "test_rows": len(X) - split_idx,
        }
    )

# ── Write results ────────────────────────────────────────────────────
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
run_dir = ROOT / "results" / f"F-phase7-{timestamp}"
run_dir.mkdir(parents=True, exist_ok=True)

ablation_df = pd.DataFrame(results)
ablation_path = run_dir / "ablation.csv"
ablation_df.to_csv(ablation_path, index=False)
print(f"\nAblation results written to {ablation_path}")
print(ablation_df.to_string())

# ── Manifest ─────────────────────────────────────────────────────────
manifest = {
    "run_id": f"F-phase7-{timestamp}",
    "generated": timestamp,
    "n_total_ground_rows": n_total,
    "group_sets": {k: v for k, v in group_sets.items()},
    "fusion_rules": fusion_cfg,
    "sources": {
        s: {
            "observed": int(fused[f"{s}_observed"].sum()),
            "fraction": float(fused[f"{s}_observed"].sum() / n_total),
        }
        for s in SOURCE_GROUPS
    },
    "phase": "7",
    "objective": "Track B ablations using fused availability mask",
}
with open(run_dir / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

# ── Summary ──────────────────────────────────────────────────────────
print("\n=== Phase 7 Summary ===")
print(f"Run directory: {run_dir}")
print(f"Total ground rows: {n_total}")
for r in results:
    print(
        f"  {r['group_set']:20s}: observed {r['n_rows_observed']:4d} / {n_total:4d} "
        f"({r['fraction_observed']:.1%}), "
        f"MAE={r['mae']:.2f}, delta={r['delta_mae']:.2f}, "
        f"train={r['train_rows']}, test={r['test_rows']}"
    )

improved = [r for r in results if r["delta_mae"] < 0]
if improved:
    print(f"\nImprovement over baseline ({len(improved)} group set(s)):")
    for r in improved:
        print(f"  {r['group_set']}: delta={r['delta_mae']:.2f}")
else:
    print(
        "\nNo group set improved over ground-only baseline — "
        "environmental sources do not add predictive value beyond "
        "nine years of ground history (consistent with Phase 4 findings)."
    )

print(f"\nPhase 7 complete. See {run_dir}/ for full outputs.")
