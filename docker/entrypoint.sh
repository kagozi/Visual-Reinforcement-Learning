#!/bin/sh
set -eu

# Defaults (overridable via env)
: "${DISPLAY:=:99}"
: "${XVFB_W:=1280}"
: "${XVFB_H:=800}"
: "${XVFB_D:=24}"
: "${CHROME_URL:=https://chromedino.com}"   # public clone works headless; chrome://dino won't in container

# Start X virtual framebuffer
Xvfb "$DISPLAY" -screen 0 ${XVFB_W}x${XVFB_H}x${XVFB_D} -nolisten tcp -nolisten unix &
XVFB_PID=$!

# Wait for X to be ready (avoid races)
for i in 1 2 3 4 5 6 7 8 9 10; do
  if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then break; fi
  sleep 0.3
done

# Launch Chromium into the X server (no-sandbox required in most containers)
chromium \
  --no-sandbox \
  --disable-dev-shm-usage \
  --disable-gpu \
  --disable-software-rasterizer \
  --window-size=1200,700 \
  --window-position=50,50 \
  "$CHROME_URL" >/tmp/chromium.log 2>&1 &
CHROME_PID=$!

# Small delay so page renders before your env first grab
sleep 2

# Hand off to the trainer command from docker-compose (PPO/DQN/etc)
exec "$@"
