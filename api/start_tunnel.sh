#!/usr/bin/env bash
# AeroSphinx Cloudflare Tunnel launcher
#
# Serves the built UI + API from one Python process and exposes it publicly
# through a Cloudflare quick tunnel, which needs no account or configuration.
#
#   ./api/start_tunnel.sh              # http://127.0.0.1:8787
#   PORT=9000 ./api/start_tunnel.sh
#
# The public hostname is random and lives only as long as this script runs.
# For a stable URL, see the named-tunnel steps in the deployment notes.

set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-8787}"
HOST="${HOST:-127.0.0.1}"
LOG_DIR=".deploy"
mkdir -p "$LOG_DIR"

# Locate the cloudflared binary: explicit override, project-local, then PATH.
CLOUDFLARED="${CLOUDFLARED:-}"
if [ -z "$CLOUDFLARED" ]; then
    for candidate in ./.deploy/bin/cloudflared "$(command -v cloudflared || true)"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            CLOUDFLARED="$candidate"
            break
        fi
    done
fi
if [ -z "$CLOUDFLARED" ]; then
    echo "cloudflared not found: put it in .deploy/bin/ or set CLOUDFLARED." >&2
    exit 1
fi

# The API is standard-library only, so the venv is preferred but not required.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if [ -x .venv/bin/python ]; then
        PYTHON=.venv/bin/python
    else
        PYTHON=python3
    fi
fi

if [ ! -d ui/dist ]; then
    echo "ui/dist is missing; building the UI..."
    ( cd ui && npm ci --no-audit --no-fund >/dev/null 2>&1 && npm run build >/dev/null )
fi

cleanup() {
    [ -n "${TUNNEL_PID:-}" ] && kill "$TUNNEL_PID" 2>/dev/null || true
    [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting AeroSphinx on http://$HOST:$PORT ..."
"$PYTHON" api/server.py --host "$HOST" --port "$PORT" > "$LOG_DIR/api.log" 2>&1 &
SERVER_PID=$!

ready=""
for _ in $(seq 1 40); do
    if "$PYTHON" - "http://$HOST:$PORT/api/health" <<'PY' 2>/dev/null
import sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=1) as response:
    sys.exit(0 if response.status == 200 else 1)
PY
    then
        ready="yes"
        break
    fi
    sleep 0.5
done
if [ -z "$ready" ]; then
    echo "The server did not become healthy; see $LOG_DIR/api.log" >&2
    exit 1
fi

echo "Opening the Cloudflare tunnel ..."
"$CLOUDFLARED" tunnel --url "http://$HOST:$PORT" --no-autoupdate > "$LOG_DIR/tunnel.log" 2>&1 &
TUNNEL_PID=$!

URL=""
for _ in $(seq 1 60); do
    URL=$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/tunnel.log" 2>/dev/null | head -1 || true)
    [ -n "$URL" ] && break
    sleep 1
done

if [ -z "$URL" ]; then
    echo "The tunnel did not report a URL; see $LOG_DIR/tunnel.log" >&2
    exit 1
fi

echo ""
echo "=== AeroSphinx is publicly reachable ==="
echo "  $URL"
echo "  $URL/api/health"
echo ""
echo "Press Ctrl+C to stop the server and the tunnel."

wait
