# Dockerfile
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DISPLAY=:99 \
    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
    PYTORCH_ENABLE_MPS_FALLBACK=1

# --- System deps: Xvfb + xdotool + Chromium + nginx + basic GUI libs + git ---
RUN apt-get update && apt-get install -y --no-install-recommends \
      xvfb xdotool x11-utils \
      nginx git \
      chromium \
      libgl1 libglib2.0-0 libgtk-3-0 libnss3 libasound2 \
      fonts-dejavu tzdata ca-certificates curl \
      tesseract-ocr python3-dev gcc \
  && ln -sf /usr/bin/chromium /usr/local/bin/chromium-browser || true \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps (Torch CPU wheel index for portability) ---
COPY requirements-docker.txt /app/
RUN pip install --upgrade pip wheel \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-docker.txt

# --- Project code ---
COPY . /app

# --- Clone the open-source game and serve it with nginx on :8080 ---
RUN mkdir -p /var/www/trex \
 && git clone --depth=1 https://github.com/kagozi/t-rex-runner.git /var/www/trex \
 && rm -rf /var/www/trex/.git

# nginx server block for the game (use printf to avoid Docker heredoc parsing issues)
RUN printf '%s\n' \
  'server {' \
  '  listen 8080;' \
  '  server_name _;' \
  '  root /var/www/trex;' \
  '  index index.html;' \
  '  access_log /var/log/nginx/trex_access.log;' \
  '  error_log  /var/log/nginx/trex_error.log;' \
  '  location / { try_files $uri /index.html; }' \
  '}' \
  > /etc/nginx/conf.d/trex.conf \
 && rm -f /etc/nginx/sites-enabled/default

# --- Entrypoint ---
COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# Point Chromium at the local nginx by default
ENV CHROME_URL=http://127.0.0.1:8080

ENTRYPOINT ["/entrypoint.sh"]
