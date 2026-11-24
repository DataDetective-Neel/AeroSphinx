#!/usr/bin/env python3
"""Check that the deployed service is running the commit checked out locally.

The Render blueprint deploys on push, so whether a push is live is a question
about one GET. This compares the commit reported by the deployed ``/api/status``
with the local ``HEAD`` and fails when they differ, so a stale deployment is
not mistaken for a successful release.

Usage:
    python scripts/check_deploy.py
    python scripts/check_deploy.py --url https://aerosphinx.onrender.com
    python scripts/check_deploy.py --sha 9b0738b

Exit status is 0 only when the deployed commit matches. Standard library only,
like the API it probes. The free Render tier sleeps when idle, so the first
request may take a while; retries absorb the cold start.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("AEROSPHINX_URL", "https://aerosphinx.onrender.com")


def local_sha(ref: str = "HEAD") -> str:
    """Return the full commit SHA of ``ref`` in the current repository."""
    result = subprocess.run(
        ["git", "rev-parse", ref],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not resolve {ref}: {result.stderr.strip() or 'git rev-parse failed'}"
        )
    return result.stdout.strip()


def fetch_status(base_url: str, timeout: float, attempts: int) -> dict:
    """Read ``/api/status`` from the deployment, retrying through cold starts."""
    url = base_url.rstrip("/") + "/api/status"
    request = urllib.request.Request(
        url, headers={"User-Agent": "aerosphinx-deploy-check"}
    )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(5 * attempt, 15))
    raise RuntimeError(f"could not read {url}: {last_error}")


def verdict(local: str, payload: dict) -> tuple[bool, str]:
    """Compare a local commit SHA with a deployed ``/api/status`` payload."""
    deployed = ((payload or {}).get("repo") or {}).get("sha")
    if not deployed:
        return False, "deployed /api/status did not report repo.sha"
    if deployed == local:
        return True, f"deployed commit matches local HEAD ({local[:7]})"
    if len(deployed) >= 7 and local.startswith(deployed):
        return True, f"deployed short SHA {deployed} matches local HEAD ({local[:7]})"
    return False, f"deployed {deployed[:7]} != local HEAD {local[:7]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"deployment base URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--sha",
        default="HEAD",
        help="local ref to compare against (default: HEAD)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="per-request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="request attempts before giving up (default: 3)",
    )
    args = parser.parse_args(argv)

    try:
        local = local_sha(args.sha)
    except RuntimeError as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1

    try:
        payload = fetch_status(args.url, args.timeout, args.attempts)
    except RuntimeError as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1

    matches, message = verdict(local, payload)
    repo = payload.get("repo") or {}
    print(f"{'PASS' if matches else 'FAIL'}  {message}")
    print(f"      url: {args.url.rstrip('/')}")
    print(f"      branch: {repo.get('branch')}  dirty: {repo.get('dirty')}")
    if matches and repo.get("dirty"):
        print("NOTE  the deployment reports uncommitted changes in its build tree")
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
