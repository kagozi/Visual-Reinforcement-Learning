# Dockerfile (with architecture specification)
FROM --platform=linux/amd64 python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DISPLAY=:99 \
    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
    PYTORCH_ENABLE_MPS_FALLBACK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb xdotool chromium \
    libgl1 libglib2.0-0 libgtk-3-0 libnss3 libasound2 \
    fonts-dejavu tzdata ca-certificates curl \
    linux-headers-generic build-essential \
    python3-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements-docker.txt /app/

# Install Python packages
RUN pip install --upgrade pip wheel && \
    pip install --no-cache-dir \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy application code
COPY . /app

# Copy and setup entrypoint (ensure Unix line endings)
COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
