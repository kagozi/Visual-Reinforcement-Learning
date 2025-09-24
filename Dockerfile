FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DISPLAY=:99 \
    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
    PYTORCH_ENABLE_MPS_FALLBACK=1

# System deps for headless GUI + OpenCV + Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb xdotool chromium x11-utils \
    libgl1 libglib2.0-0 libgtk-3-0 libnss3 libasound2 \
    fonts-dejavu tzdata ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (pin numpy<2; add torch CPU via extra index)
COPY requirements-docker.txt /app/
RUN pip install --upgrade pip wheel \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-docker.txt

# Project
COPY . /app

# Entrypoint (convert CRLF->LF just in case, then make executable)
COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
