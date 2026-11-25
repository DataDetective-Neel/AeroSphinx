"""AeroSphinx read-only API server.

Serves the real scientific artifacts already present in this repository over
HTTP so the frontend can consume them with no fabrication. Standard library
only — no new dependencies.

Endpoints
---------
GET /api/health              -> {"status": "ok", "generated": <iso>}
GET /api/status              -> repository + dataset + test summary
GET /api/dataset             -> fused-frame summary + per-column missingness
GET /api/experiments         -> Track A metrics from results/legacy_chapter/metrics.json
GET /api/series?start&end    -> daily series rows (date, pm25, sources)
GET /api/coverage            -> per-source observation counts + overlap
GET /api/manifests           -> list of run manifests with key fields
GET /api/experiment-matrix   -> M0-M8 metrics, common dates, ablations
GET /api/ablations           -> paired source/season ablation deltas
GET /api/seasonal-sufficiency-> per-season per-source sufficiency
GET /api/alignment           -> daily source coverage and missingness

Run:
    python api/server.py --port 8787

The server never mutates data. Missing artifacts are reported as
{"available": false} rather than invented.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PROCESSED = DATA / "processed"

FUSED = PROCESSED / "fused_dataset.csv"
LEGACY = RESULTS / "legacy_chapter"
METRICS = LEGACY / "metrics.json"
TRACK_A = PROCESSED / "track_a_daily.csv"
BASELINE_MANIFEST = RESULTS / "baseline" / "manifest.json"

# Columns we expose from the fused frame. Values are read verbatim; NaN is
# emitted as null, never coerced to zero.
SERIES_COLUMNS = [
    "Date",
    "PM25",
    "AOD",
    "T2M_midday",
    "RH",
    "WS",
    "station_met_observed",
    "merra_observed",
    "aod_observed",
    "n_sources_observed",
    "all_sources_observed",
]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _read_csv(path: Path):
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _as_number(value: str):
    """Return float or None (never 0 for a blank/NaN cell)."""
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except ValueError:
        return None
    if f != f:  # NaN
        return None
    return f


def _as_bool(value: str):
    if value in ("True", "true", "1"):
        return True
    if value in ("False", "false", "0"):
        return False
    return None


def git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


_GIT_STATUS: str | None = None
_GIT_STATUS_AT: float = 0.0
_METRICS_GEN: str | None = None
_METRICS_GEN_AT: float = 0.0


def _git_status_cached() -> bool:
    """Return True if the working tree has uncommitted changes.
    Results are cached for the process lifetime to avoid repeated git calls."""
    global _GIT_STATUS, _GIT_STATUS_AT
    import time as _time

    if _time.time() - _GIT_STATUS_AT < 30:
        return _GIT_STATUS  # type: ignore[return-value]
    _GIT_STATUS = bool(_git_status_raw())  # type: ignore[assignment]
    _GIT_STATUS_AT = _time.time()
    return _GIT_STATUS


#: Path prefixes whose modification does not mean "this deployment is running
#: uncommitted science". A Render build runs `npm ci`, which can rewrite the
#: lockfile under a different npm version, and the deploy tree carries build
#: output that is gitignored in a developer checkout. Neither changes a number.
_BUILD_NOISE = (
    "ui/package-lock.json",
    "ui/node_modules",
    "ui/dist",
    ".venv",
    "node_modules",
)


def _git_status_raw() -> str | None:
    """Porcelain status, minus build-tree noise. No caching."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        lines = [
            line
            for line in out.stdout.splitlines()
            if line.strip()
            and not any(line[3:].strip().strip('"').startswith(p) for p in _BUILD_NOISE)
        ]
        return "\n".join(lines)
    except Exception:
        return None


def _experiments_generated() -> str | None:
    """Return metrics generated timestamp, cached to avoid repeated file reads."""
    global _METRICS_GEN, _METRICS_GEN_AT
    import time as _time

    if METRICS.exists() and _time.time() - _METRICS_GEN_AT < 60:
        return _METRICS_GEN  # type: ignore[return-value]
    if METRICS.exists():
        try:
            _METRICS_GEN = METRICS.read_text(encoding="utf-8")
            import json as _json

            _METRICS_GEN = _json.loads(_METRICS_GEN).get("generated")
        except Exception:
            _METRICS_GEN = None
    _METRICS_GEN_AT = _time.time()
    return _METRICS_GEN


def dataset_summary():
    rows = _read_csv(FUSED)
    if rows is None:
        return {"available": False, "reason": "fused_dataset.csv not found"}
    n = len(rows)
    dates = [r.get("Date", "") for r in rows if r.get("Date")]
    missing = {}
    if rows:
        for col in rows[0].keys():
            blank = sum(1 for r in rows if r.get(col, "") == "")
            missing[col] = {"missing": blank, "fraction": round(blank / n, 4)}
    return {
        "available": True,
        "rows": n,
        "columns": len(rows[0]) if rows else 0,
        "start": dates[0] if dates else None,
        "end": dates[-1] if dates else None,
        "missingness": missing,
    }


def coverage_summary():
    rows = _read_csv(FUSED)
    if rows is None:
        return {"available": False}
    n = len(rows)

    def count(col):
        return sum(1 for r in rows if _as_bool(r.get(col, "")) is True)

    station = count("station_met_observed")
    merra = count("merra_observed")
    aod = count("aod_observed")
    allthree = count("all_sources_observed")
    return {
        "available": True,
        "rows": n,
        "sources": {
            "station_met": {"observed": station, "fraction": round(station / n, 4)},
            "merra": {"observed": merra, "fraction": round(merra / n, 4)},
            "aod": {"observed": aod, "fraction": round(aod / n, 4)},
        },
        "all_sources_observed": {
            "observed": allthree,
            "fraction": round(allthree / n, 4),
        },
    }


def experiments_summary():
    if not METRICS.exists():
        return {
            "available": False,
            "reason": "results/legacy_chapter/metrics.json not found",
        }
    try:
        m = json.loads(METRICS.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"available": False, "reason": "metrics.json is not valid JSON"}
    return {
        "available": True,
        "generated": m.get("generated"),
        "configuration": m.get("configuration_name"),
        "station": m.get("station"),
        "coverage": m.get("coverage"),
        "split": m.get("split"),
        "metrics": m.get("metrics"),
        "baselines": m.get("baselines"),
        "features_used": m.get("features_used"),
        "features_unavailable": m.get("features_unavailable"),
        "hyperparameters": m.get("hyperparameters"),
    }


def series(start: str | None, end: str | None, limit: int = 5000):
    rows = _read_csv(FUSED)
    if rows is None:
        return {"available": False, "reason": "fused_dataset.csv not found"}
    out = []
    for r in rows:
        d = r.get("Date", "")
        if start and d < start:
            continue
        if end and d > end:
            continue
        item = {"date": d}
        item["pm25"] = _as_number(r.get("PM25", ""))
        item["aod"] = _as_number(r.get("AOD", ""))
        item["t2m"] = _as_number(r.get("T2M_midday", ""))
        item["rh"] = _as_number(r.get("RH", ""))
        item["ws"] = _as_number(r.get("WS", ""))
        item["sources"] = {
            "station_met": _as_bool(r.get("station_met_observed", "")),
            "merra": _as_bool(r.get("merra_observed", "")),
            "aod": _as_bool(r.get("aod_observed", "")),
        }
        out.append(item)
        if len(out) >= limit:
            break
    return {"available": True, "count": len(out), "rows": out}


def manifests_summary():
    found = []
    for path in sorted(RESULTS.rglob("manifest.json")):
        try:
            m = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        found.append(
            {
                "path": str(path.relative_to(ROOT)),
                "run_id": m.get("run_id"),
                "git_sha": m.get("git_sha") or m.get("git", {}).get("sha"),
                "generated": m.get("generated") or m.get("timestamp"),
                "track": m.get("track"),
            }
        )
    return {"available": bool(found), "count": len(found), "manifests": found}


MATRIX_DIR = RESULTS / "experiment_matrix"
SUFFICIENCY_DIR = RESULTS / "sufficiency"
ALIGNMENT_DIR = RESULTS / "alignment"


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _clean(value):
    """Convert a CSV cell to JSON-safe number/bool/str/null.

    NaN and blank cells become null; they are never coerced to zero. This
    also keeps the response strictly valid JSON, which ``NaN`` is not.
    """
    if value is None:
        return None
    text = str(value)
    if text == "":
        return None
    if text in ("True", "true"):
        return True
    if text in ("False", "false"):
        return False
    number = _as_number(text)
    if number is not None:
        return number
    if text.lower() == "nan":
        return None
    return text


def _artifact_rows(path: Path):
    rows = _read_csv(path)
    if rows is None:
        return []
    return [{key: _clean(value) for key, value in row.items()} for row in rows]


def experiment_matrix_summary():
    metrics = MATRIX_DIR / "matrix_metrics.csv"
    if not metrics.exists():
        return {
            "available": False,
            "reason": "run `python scripts/run_experiment_matrix.py`",
        }
    manifest = _read_json(MATRIX_DIR / "manifest.json") or {}
    configs = _read_json(MATRIX_DIR / "matrix_configs.json") or {}
    return {
        "available": True,
        "generated": (
            manifest.get("generated_utc")
            or manifest.get("generated")
            or manifest.get("timestamp")
        ),
        "run_id": manifest.get("run_id"),
        "configs": configs,
        "metrics": _artifact_rows(metrics),
        "common": _artifact_rows(MATRIX_DIR / "matrix_common_metrics.csv"),
        "by_season": _artifact_rows(MATRIX_DIR / "matrix_by_season.csv"),
        "by_year": _artifact_rows(MATRIX_DIR / "matrix_by_year.csv"),
        "ablations": _artifact_rows(MATRIX_DIR / "matrix_ablations.csv"),
    }


def ablations_summary():
    path = MATRIX_DIR / "matrix_ablations.csv"
    if not path.exists():
        return {
            "available": False,
            "reason": "run `python scripts/run_experiment_matrix.py`",
        }
    return {
        "available": True,
        "rows": _artifact_rows(path),
        "common": _artifact_rows(MATRIX_DIR / "matrix_common_metrics.csv"),
    }


def seasonal_sufficiency_summary():
    path = SUFFICIENCY_DIR / "season_source_sufficiency.csv"
    if not path.exists():
        return {
            "available": False,
            "reason": "run `python scripts/run_season_sufficiency.py`",
        }
    report = _read_json(SUFFICIENCY_DIR / "season_sufficiency_report.json") or {}
    summary = report.get("summary") or {}
    return {
        "available": True,
        "rows": _artifact_rows(path),
        "criteria": summary.get("criteria"),
        "counts": summary.get("counts"),
        "by_season": summary.get("by_season"),
        "not_evaluable": summary.get("not_evaluable"),
    }


def alignment_summary():
    coverage = ALIGNMENT_DIR / "daily_source_coverage.csv"
    if not coverage.exists():
        return {
            "available": False,
            "reason": "run `python scripts/run_daily_alignment.py`",
        }
    report = _read_json(ALIGNMENT_DIR / "daily_alignment_report.json") or {}
    validation = report.get("validation") or {}
    return {
        "available": True,
        "coverage": _artifact_rows(coverage),
        "missingness": _artifact_rows(ALIGNMENT_DIR / "daily_source_missingness.csv"),
        "report": {
            "rows": report.get("rows"),
            "date_range": report.get("date_range"),
            "rows_with_all_sources_observed": report.get(
                "rows_with_all_sources_observed"
            ),
            "fraction_with_all_sources_observed": report.get(
                "fraction_with_all_sources_observed"
            ),
            "rows_with_no_source_observed": report.get("rows_with_no_source_observed"),
            "n_sources_distribution": report.get("n_sources_distribution"),
            "per_year": report.get("per_year"),
            "validation_passed": validation.get("passed"),
        },
    }


def forecasts_summary():
    """Persisted forecast projections read verbatim from the legacy artefacts."""
    operational_path = LEGACY / "operational_forward_forecast.csv"
    recursive_path = LEGACY / "five_day_recursive_forecast.csv"
    operational_rows = _artifact_rows(operational_path)
    recursive_rows = _artifact_rows(recursive_path)
    if not operational_rows and not recursive_rows:
        return {
            "available": False,
            "reason": "no persisted forecast projection found under results/legacy_chapter/",
            "operational": {
                "path": str(operational_path.relative_to(ROOT)),
                "rows": [],
            },
            "recursive": {"path": str(recursive_path.relative_to(ROOT)), "rows": []},
        }
    return {
        "available": True,
        "operational": {
            "path": str(operational_path.relative_to(ROOT)),
            "rows": operational_rows,
        },
        "recursive": {
            "path": str(recursive_path.relative_to(ROOT)),
            "rows": recursive_rows,
        },
    }


def status_summary():
    ds = dataset_summary()
    return {
        "generated": _now(),
        "repo": {
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "sha": git("rev-parse", "HEAD"),
            "short_sha": git("rev-parse", "--short", "HEAD"),
            "dirty": _git_status_cached(),
        },
        "dataset": ds,
        "coverage": coverage_summary(),
        "experiments": {
            "available": METRICS.exists(),
            "generated": _experiments_generated(),
        },
    }


ROUTES = {
    "/api/health": lambda q: {"status": "ok", "generated": _now()},
    "/api/status": lambda q: status_summary(),
    "/api/dataset": lambda q: dataset_summary(),
    "/api/experiments": lambda q: experiments_summary(),
    "/api/coverage": lambda q: coverage_summary(),
    "/api/series": lambda q: series(
        (q.get("start") or [None])[0], (q.get("end") or [None])[0]
    ),
    "/api/manifests": lambda q: manifests_summary(),
    "/api/experiment-matrix": lambda q: experiment_matrix_summary(),
    "/api/ablations": lambda q: ablations_summary(),
    "/api/seasonal-sufficiency": lambda q: seasonal_sufficiency_summary(),
    "/api/alignment": lambda q: alignment_summary(),
    "/api/forecast": lambda q: forecast_endpoint(q),
    "/api/forecasts": lambda q: forecasts_summary(),
}


def _model_needs(m: str) -> set[str]:
    """Return the feature set a model requires.
    persistence needs pm25_history.
    temporal_ml needs pm25_history.
    fusion needs pm25_history, aod, merra.
    """
    needs: dict[str, set[str]] = {
        "persistence": {"pm25_history"},
        "temporal_ml": {"pm25_history"},
        "fusion": {"pm25_history", "aod", "merra"},
    }
    return needs.get(m, set())


def _parse_forecast_payload(body: dict) -> dict | None:
    """Parse and validate the forecast payload.
    Returns dict with model, issue_date, horizon, or None on failure.
    """
    try:
        model = body.get("model")
        issue_date = body.get("issue_date")
        horizon = body.get("horizon")
        # Validate model is present
        if not model:
            return None
        # Validate horizon is an integer in range 1-5
        try:
            horizon = int(horizon)
        except (TypeError, ValueError):
            horizon = 1
        if horizon < 1 or horizon > 5:
            horizon = 1
        return {
            "model": str(model),
            "issue_date": str(issue_date) if issue_date else None,
            "horizon": horizon,
        }
    except Exception:
        return None


def _matrix_lookup(filename: str, **where) -> dict | None:
    """Return the single row of a matrix artefact matching ``where``.

    The forecast gate must quote numbers that exist in the artefacts, not
    numbers typed into this file. The gate text used to hard-code the legacy
    33-day figures (16.617 / 17.298), which no current artefact supports.
    """
    rows = _read_csv(MATRIX_DIR / filename)
    if not rows:
        return None
    for row in rows:
        if all(str(row.get(key, "")) == str(value) for key, value in where.items()):
            return row
    return None


def _metric_mae(track: str, config: str, model: str) -> float | None:
    row = _matrix_lookup("matrix_metrics.csv", track=track, config=config, model=model)
    return None if row is None else _as_number(row.get("MAE"))


def _ablation(config: str, reference: str) -> dict:
    return (
        _matrix_lookup(
            "matrix_ablations.csv", track="B", config=config, reference=reference
        )
        or {}
    )


def _fmt(value: float | None, digits: int = 2, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _gate_evidence() -> dict:
    """Every number the forecast gate quotes, resolved from the artefacts."""
    fusion = _ablation("M4", "M1")
    aod = _ablation("M2", "M1")
    return {
        "track_a_persistence": _fmt(_metric_mae("A", "M0", "persistence")),
        "track_a_ridge": _fmt(_metric_mae("A", "M1", "ridge")),
        "track_b_persistence": _fmt(_metric_mae("B", "M0", "persistence")),
        "track_b_ridge": _fmt(_metric_mae("B", "M1", "ridge")),
        "fusion_delta": _fmt(_as_number(fusion.get("delta_MAE")), signed=True),
        "fusion_low": _fmt(_as_number(fusion.get("ci_low")), signed=True),
        "fusion_high": _fmt(_as_number(fusion.get("ci_high")), signed=True),
        "fusion_p_holm": _fmt(_as_number(fusion.get("p_holm"))),
        "fusion_n": int(_as_number(fusion.get("n"))) if fusion.get("n") else None,
        "aod_delta": _fmt(_as_number(aod.get("delta_MAE")), signed=True),
        "aod_low": _fmt(_as_number(aod.get("ci_low")), signed=True),
        "aod_high": _fmt(_as_number(aod.get("ci_high")), signed=True),
    }


# Model feature requirements (same as _model_needs but used inline)
_MODEL_NEEDS: dict[str, set[str]] = {
    "persistence": {"pm25_history"},
    "temporal_ml": {"pm25_history"},
    "fusion": {"pm25_history", "aod", "merra"},
}


def forecast_endpoint(body: dict) -> dict:
    """POST /api/forecast – eligibility‑checked, no fabricated prediction."""
    payload = _parse_forecast_payload(body)
    if payload is None:
        return {
            "status": "UNSUPPORTED",
            "forecast": None,
            "interval": None,
            "reason": "Malformed or missing forecast payload.",
            "used_features": {},
        }

    model = payload["model"]
    issue_date = payload["issue_date"]
    horizon = payload["horizon"]

    # 1. Model existence check
    if model not in _MODEL_NEEDS:
        return {
            "status": "UNSUPPORTED",
            "forecast": None,
            "interval": None,
            "reason": f'Model "{model}" is not recognised. Valid models: {list(_MODEL_NEEDS.keys())}',
            "used_features": {},
        }

    needs = _MODEL_NEEDS[model]

    # 2. Horizon support check: published comparison only supports day 1–2
    if horizon >= 3:
        return {
            "status": "DEGRADED",
            "forecast": None,
            "interval": None,
            "reason": f"Horizon {horizon} compounds error; the published comparison only supports one–two days.",
            "used_features": {k: False for k in needs},
        }

    # 3. Retrieve series data for the issue date (std-lib only, no new deps)
    # We read fused_dataset.csv directly, mirroring the /api/series implementation
    rows = _read_csv(FUSED)
    if rows is None:
        return {
            "status": "ABSTAINED",
            "forecast": None,
            "interval": None,
            "reason": "fused_dataset.csv not available; cannot establish feature availability.",
            "used_features": {k: False for k in needs},
        }

    # Find the row matching the issue date
    point = None
    for r in rows:
        if r.get("Date") == issue_date:
            point = r
            break

    if point is None:
        return {
            "status": "ABSTAINED",
            "forecast": None,
            "interval": None,
            "reason": f"No ground observation found for issue date {issue_date}.",
            "used_features": {k: False for k in needs},
        }

    # 4. Feature‑availability check at issue time
    # A feature is "observed" if its value exists and is non‑empty in the fused frame.
    # This mirrors the /api/series availability logic.
    def _has_value(value) -> bool:
        return value is not None and str(value).strip() != ""

    sources = {
        "pm25_history": _has_value(point.get("PM25")),
        "aod": _has_value(point.get("AOD")),
        "merra": _has_value(point.get("T2M_midday")),
    }

    # Check that all model‑required features are observed
    missing: list[str] = []
    for feature in needs:
        if feature not in sources or not sources[feature]:
            missing.append(feature)

    if missing:
        feature_names = {
            "pm25_history": "PM2.5 history",
            "aod": "INSAT-3D AOD",
            "merra": "MERRA-2 reanalysis",
        }
        missing_desc = ", ".join(feature_names.get(f, f) for f in missing)
        return {
            "status": "UNSUPPORTED",
            "forecast": None,
            "interval": None,
            "reason": f"Required feature(s) not observed: {missing_desc}.",
            "used_features": {k: sources.get(k, False) for k in needs},
        }

    # 5. Evaluation gate: is a learned model established to beat persistence?
    # The numbers come from results/experiment_matrix/, never from this file.
    evidence = _gate_evidence()
    if model == "persistence":
        return {
            "status": "DEGRADED",
            "forecast": None,
            "interval": None,
            "reason": (
                "No learned model is established to beat persistence. On Track A "
                f"ridge scores {evidence['track_a_ridge']} MAE against "
                f"persistence's {evidence['track_a_persistence']}, and the paired "
                "difference is not significant under block-aware inference (D-006)."
            ),
            "used_features": {k: True for k in needs},
        }

    # 6. Evaluation gate: does fusion add anything? (D-007)
    if model == "fusion":
        return {
            "status": "DEGRADED",
            "forecast": None,
            "interval": None,
            "reason": (
                f"On the {evidence['fusion_n']} dates where every Track B "
                "configuration predicts, adding AOD changes MAE by "
                f"{evidence['aod_delta']} ({evidence['aod_low']} to "
                f"{evidence['aod_high']}) and full fusion by "
                f"{evidence['fusion_delta']} ({evidence['fusion_low']} to "
                f"{evidence['fusion_high']}, Holm-adjusted p = "
                f"{evidence['fusion_p_holm']}). None of these is distinguishable "
                "from zero, and Track B's ground-only model "
                f"({evidence['track_b_ridge']}) is worse than persistence "
                f"({evidence['track_b_persistence']}) on that set (D-007)."
            ),
            "used_features": {k: True for k in needs},
        }

    # 7. If model exists, horizon is supported, features are available,
    #    and no special gate applies — the repository has no inference endpoint.
    #    Return ABSTAINED with an explicit explanation rather than fabricating.
    return {
        "status": "ABSTAINED",
        "forecast": None,
        "interval": None,
        "reason": "The repository produces forecasts through offline Python scripts. There is no inference endpoint; no prediction is returned rather than inventing a value.",
        "used_features": {k: True for k in needs},
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "AeroSphinxAPI/1.0"

    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802, redirect to static
        """Serve the frontend SPA from ./ui/dist."""
        import os

        dist_path = os.path.join(ROOT, "ui", "dist")
        requested_path = self.path

        # If the route is one of our API routes, handle as before
        api_routes = [
            "/api/health",
            "/api/status",
            "/api/dataset",
            "/api/experiments",
            "/api/coverage",
            "/api/series",
            "/api/manifests",
            "/api/experiment-matrix",
            "/api/ablations",
            "/api/seasonal-sufficiency",
            "/api/alignment",
            "/api/forecast",
        ]
        if requested_path in api_routes or requested_path.startswith("/api/"):
            parsed = urlparse(requested_path)
            route = ROUTES.get(parsed.path)
            if route is None:
                self._json({"error": "not_found", "path": parsed.path}, 404)
                return
            try:
                payload = route(parse_qs(parsed.query))
                self._json(payload)
            except Exception as exc:
                self._json({"error": "server_error", "detail": str(exc)}, 500)
            return

        # Otherwise, try to serve static files from dist
        # First, check if the path includes a file extension (css, js, html, etc.)
        if (
            "." in requested_path[len("/api") :]
            if requested_path.startswith("/api")
            else "." in requested_path
        ):
            # It looks like a static asset request
            # Try to serve the file
            file_path = os.path.join(dist_path, requested_path.lstrip("/"))
            if os.path.exists(file_path) and os.path.isfile(file_path):
                try:
                    with open(file_path, "rb") as f:
                        content = f.read()

                    # Determine content type
                    ext = requested_path.split(".")[-1] if "." in requested_path else ""
                    content_types = {
                        "html": "text/html",
                        "js": "application/javascript",
                        "css": "text/css",
                        "png": "image/png",
                        "jpg": "image/jpeg",
                        "ico": "image/x-icon",
                        "json": "application/json",
                    }
                    ct = content_types.get(ext, "application/octet-stream")

                    self.send_response(200)
                    self.send_header("Content-Type", ct)
                    self.send_header("Cache-Control", "public, max-age=86400")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(content)
                    return
                except Exception:
                    pass

        # If no static file found, serve index.html for SPA routing
        index_path = os.path.join(dist_path, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
            return

        # Fallback 404
        self._json({"error": "not_found", "path": requested_path}, 404)

    def log_message(self, fmt, *args):  # keep stdout quiet
        pass

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        route = ROUTES.get(parsed.path)
        if route is None:
            self._json({"error": "not_found", "path": parsed.path}, 404)
            return
        try:
            # Read body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(body) if body else {}
            result = route(payload)
            self._json(result)
        except Exception:
            self._json(
                {"error": "server_error", "detail": "an unexpected error occurred"}, 500
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8787)))
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"AeroSphinx API on http://{args.host}:{args.port} (read-only)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
