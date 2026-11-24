#!/usr/bin/env bash
# AeroSphinx Deployment Script
# Starts the combined API + Frontend server on a single port
# Requires: python 3.12+, npm packages installed in ui/

set -e

echo "=== AeroSphinx Deployment ==="

# Change to project root
cd "$(dirname "$0")/.."

# Function to check if a port is in use
port_in_use() {
    local port=$1
    ss -tlnp "sport = :$port" 2>/dev/null || lsof -i ":$port" 2>/dev/null | tail -1 || echo "no"
}

# Ask for port
DEFAULT_PORT=8787
read -p "API port [${DEFAULT_PORT}]: " PORT
PORT=${PORT:-$DEFAULT_PORT}

# Check if port is available
if [ "$(port_in_use $PORT)" != "no" ]; then
    echo "WARNING: Port $PORT appears to be in use!"
    read -p "Continue anyway? (y/n) " yn
    [ "$yn" != "y" ] && exit 1
fi

# Check if UI dist exists, build if needed
if [ ! -d "ui/dist" ]; then
    echo "Building UI..."
    cd ui && npm run build 2>&1 | tail -3
    cd ..
fi

# Start the server in the background
echo "Starting AeroSphinx API + Frontend server on port $PORT..."
echo "  API:     http://127.0.0.1:$PORT/api/*"
echo "  Frontend: http://127.0.0.1:$PORT/"
echo ""

# Start server - it will run until killed
python api/server.py --port "$PORT" &

echo ""
echo "=== Deployment running ==="
echo "Press Ctrl+C to stop, or kill the background process"
echo ""

# Wait for interrupt
wait