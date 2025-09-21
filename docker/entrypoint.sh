#!/usr/bin/env bash
set -euo pipefail

echo "Starting Chrome Dino RL environment..."

# Start Xvfb in background
echo "Starting virtual display..."
Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Wait for X server to start
sleep 3

# Start Chromium with Dino game
echo "Starting Chromium with Chrome Dino..."
chromium \
  --no-sandbox \
  --disable-gpu \
  --disable-dev-shm-usage \
  --disable-extensions \
  --disable-plugins \
  --hide-crash-restore-bubble \
  --no-first-run \
  --no-default-browser-check \
  --user-data-dir=/tmp/chrome \
  --window-size=1280,720 \
  --start-maximized \
  https://chromedino.com &

CHROME_PID=$!

# Wait for browser to load
echo "Waiting for browser to load..."
sleep 5

# Function to cleanup processes
cleanup() {
    echo "Cleaning up processes..."
    kill $CHROME_PID 2>/dev/null || true
    kill $XVFB_PID 2>/dev/null || true
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# If no command specified, run interactive shell
if [ $# -eq 0 ]; then
    echo "No command specified. Starting interactive shell..."
    exec bash
fi

# Run the provided command
echo "Executing command: $*"
exec "$@"