"""Re-run the matrix and prove the committed numbers still reproduce.

``docs/reference/SCIENTIFIC_ASSUMPTIONS.md`` G1 claims that repeated runs give identical
numbers. Nothing checked that. This script regenerates the M0-M8 matrix into
a temporary directory and compares it field by field with the committed
artefacts, so a code change that silently moves a published metric fails CI
instead of being discovered by a reader.

Comparison is on values, not bytes: two runs of the same deterministic code
agree to ~1e-13 in the CSV text but not necessarily character for character.

Usage::

    aerosphinx check-reproducibility
    python scripts/check_reproducibility.py --tolerance 1e-6

Exits 0 when every compared field matches within tolerance, 1 on mismatch,
2 when the run itself fails.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

#: Artefacts compared, with the columns that identify a row.
_ARTEFACTS = {
    "matrix_metrics.csv": ["track", "config", "model"],
    "matrix_ablations.csv": ["track", "config", "reference"],
    "matrix_common_metrics.csv": ["track", "config", "model"],
}

#: Columns compared numerically.
_NUMERIC = ("MAE", "RMSE", "R2", "MedianAE", "Mean_Bias", "delta_MAE", "n")


def _compare(committed: pd.DataFrame, fresh: pd.DataFrame, keys: list[str], tol: float):
    problems: list[str] = []
    merged = committed.merge(
        fresh, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True
    )
    unmatched = merged[merged["_merge"] != "both"]
    if len(unmatched):
        problems.append(
            f"row set differs: {len(unmatched)} key(s) present on one side only"
        )
    for column in _NUMERIC:
        old, new = f"{column}_old", f"{column}_new"
        if old not in merged.columns or new not in merged.columns:
            continue
        difference = (merged[old] - merged[new]).abs()
        offending = merged[difference > tol]
        if len(offending):
            keys_text = ", ".join(
                str(tuple(offending.iloc[i][k] for k in keys))
                for i in range(min(3, len(offending)))
            )
            problems.append(
                f"{column} moved by up to {difference.max():.3g} in {len(offending)} "
                f"row(s): {keys_text}"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--keep", action="store_true", help="keep the temp directory")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    committed_dir = root / "results" / "experiment_matrix"
    if not committed_dir.is_dir():
        print(f"no committed matrix at {committed_dir}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="aerosphinx-repro-") as tmp:
        out_dir = Path(tmp) / "matrix"
        command = [
            sys.executable,
            str(root / "scripts" / "run_experiment_matrix.py"),
            "--root",
            str(root),
            "--out-dir",
            str(out_dir),
        ]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        if result.returncode != 0:
            print("matrix regeneration failed:", file=sys.stderr)
            print((result.stderr or result.stdout)[-2000:], file=sys.stderr)
            return 2

        failures = 0
        for filename, keys in _ARTEFACTS.items():
            old_path, new_path = committed_dir / filename, out_dir / filename
            if not old_path.is_file() or not new_path.is_file():
                print(f"SKIP  {filename} (not present on both sides)")
                continue
            problems = _compare(
                pd.read_csv(old_path), pd.read_csv(new_path), keys, args.tolerance
            )
            if problems:
                failures += 1
                print(f"FAIL  {filename}")
                for problem in problems:
                    print(f"      {problem}")
            else:
                print(f"PASS  {filename} reproduces within {args.tolerance:g}")

        if args.keep:
            print(f"fresh artefacts kept at {out_dir}")

    if failures:
        print(
            "\nA mismatch means the committed artefact no longer follows from "
            "the code. Regenerate results/experiment_matrix/ and re-run "
            "check-claims.",
            file=sys.stderr,
        )
        return 1
    print("\nReproducibility: every compared field matched.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
