#!/usr/bin/env bash
set -euo pipefail

# Defaults (overridable)
: "${DISPLAY:=:99}"
: "${XVFB_W:=1280}"
: "${XVFB_H:=720}"
: "${XVFB_D:=24}"
: "${CHROME_URL:=http://127.0.0.1:8080}"

# Pick a Chromium binary that exists
if command -v chromium >/dev/null 2>&1; then
  CHROME_BIN=chromium
elif command -v chromium-browser >/dev/null 2>&1; then
  CHROME_BIN=chromium-browser
else
  echo "No chromium binary found"; exit 1
fi

# Start X virtual framebuffer
Xvfb "$DISPLAY" -screen 0 ${XVFB_W}x${XVFB_H}x${XVFB_D} -nolisten tcp -nolisten unix &
XVFB_PID=$!

# Wait for X to be ready
for i in {1..20}; do
  if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then break; fi
  sleep 0.25
done

# Start nginx quietly (daemon on so this script can continue)
nginx -g 'daemon on;'

# Launch Chromium (headed) into the X server
"$CHROME_BIN" \
  --no-sandbox \
  --disable-dev-shm-usage \
  --disable-gpu \
  --disable-software-rasterizer \
  --no-first-run \
  --disable-extensions \
  --disable-infobars \
  --force-device-scale-factor=1 \
  --window-size="${XVFB_W},${XVFB_H}" \
  --window-position=50,50 \
  "$CHROME_URL" >/tmp/chromium.log 2>&1 &

# Small delay so page renders before your env’s first grab
sleep 2

# Focus the Chromium window (best-effort)
xdotool search --sync --onlyvisible --class "Chromium" windowactivate || true
xdotool search --sync --onlyvisible --name "T-Rex Runner" windowactivate || true

# Hand off to the container CMD (e.g., scripts.train_ppo)
exec "$@"