"""Verify every published number against the artefact it came from.

The repository publishes the same scientific numbers in three places: the
README, the phase reports, and the API/UI. Nothing linked them, so they
drifted — the README carried a Track A ridge delta of −0.65 while the
runner it tells you to execute produced +0.34, and ``api/server.py`` still
quoted the legacy 33-day fusion numbers.

``results/CANONICAL.json`` is the single registry that fixes this. Each entry
names one published claim, the artefact that produces it, how to select the
row, and the value the README must show. This script checks both halves:

1. **artefact** — the artefact still holds the recorded value (within
   tolerance). This catches a pipeline change that silently moves a number.
2. **readme** — the formatted value appears in the README. This catches a
   README edited without regenerating, and a regenerated artefact whose new
   value was never carried into the prose.

Run it after any regeneration::

    aerosphinx check-claims
    python scripts/check_claims.py --root . --verbose

Exits 0 when every claim holds, 1 when any claim fails, 2 when the registry
itself cannot be read. Wire it into CI so a stale number cannot be merged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_registry(root: Path) -> dict:
    path = root / "results" / "CANONICAL.json"
    if not path.is_file():
        raise FileNotFoundError(f"canonical registry not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_artifact(root: Path, rel_path: str):
    path = root / rel_path
    if not path.is_file():
        return None
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path)


def _select(frame, where: dict):
    """Filter a tabular artefact to a single row."""
    if frame is None:
        return None
    subset = frame
    for column, expected in where.items():
        if column not in subset.columns:
            return None
        subset = subset[subset[column].astype(str) == str(expected)]
    if len(subset) != 1:
        return None
    return subset.iloc[0]


def _value_from_json(payload, path: str):
    node = payload
    for part in path.split("."):
        if isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(node, dict):
            if part not in node:
                return None
            node = node[part]
        else:
            return None
    return node


def check_claim(root: Path, claim: dict) -> dict:
    """Resolve one claim to (passed, observed, expected, detail)."""
    artifact = _read_artifact(root, claim["artifact"])
    if artifact is None:
        return {
            "id": claim["id"],
            "passed": False,
            "observed": None,
            "expected": claim.get("value"),
            "detail": f"artefact missing: {claim['artifact']}",
        }

    if isinstance(artifact, dict):
        observed = _value_from_json(artifact, claim["json_path"])
        detail_prefix = claim["json_path"]
    else:
        row = _select(artifact, claim.get("where", {}))
        if row is None:
            return {
                "id": claim["id"],
                "passed": False,
                "observed": None,
                "expected": claim.get("value"),
                "detail": "no single row matched the selector",
            }
        column = claim["column"]
        if column not in row.index:
            return {
                "id": claim["id"],
                "passed": False,
                "observed": None,
                "expected": claim.get("value"),
                "detail": f"column {column!r} absent",
            }
        observed = row[column]
        detail_prefix = f"{column} where {claim.get('where', {})}"

    if observed is None or (isinstance(observed, float) and math.isnan(observed)):
        return {
            "id": claim["id"],
            "passed": False,
            "observed": None,
            "expected": claim.get("value"),
            "detail": f"artefact holds no value at {detail_prefix}",
        }

    expected = claim["value"]
    tolerance = claim.get("tolerance", 1e-6)
    try:
        passed_value = abs(float(observed) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        passed_value = str(observed) == str(expected)

    detail = f"{detail_prefix}: artefact {observed}"
    if passed_value:
        detail += " matches the registry"

    # The README half of the check.
    readme_token = claim.get("readme")
    if readme_token:
        readme = (root / "README.md").read_text(encoding="utf-8")
        if readme_token not in readme:
            return {
                "id": claim["id"],
                "passed": False,
                "observed": observed,
                "expected": expected,
                "detail": f"artefact ok, but README does not contain {readme_token!r}",
            }
        detail += f"; README carries {readme_token!r}"

    return {
        "id": claim["id"],
        "passed": bool(passed_value),
        "observed": observed,
        "expected": expected,
        "detail": detail,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify published numbers against persisted artefacts."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    try:
        registry = _load_registry(root)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        print(f"cannot read canonical registry: {error}", file=sys.stderr)
        return 2

    results = [check_claim(root, claim) for claim in registry["claims"]]
    failures = [result for result in results if not result["passed"]]

    width = max(len(result["id"]) for result in results) if results else 10
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        line = f"{status}  {result['id'].ljust(width)}"
        if args.verbose or not result["passed"]:
            line += f"  {result['detail']}"
        print(line)

    print(
        f"\n{len(results) - len(failures)} of {len(results)} claims verified"
        f" against {registry.get('generated_from', 'the persisted artefacts')}."
    )
    if failures:
        print(
            "\nA failing claim means the prose and the artefact disagree. "
            "Regenerate the artefact, then update results/CANONICAL.json and "
            "the README together.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
