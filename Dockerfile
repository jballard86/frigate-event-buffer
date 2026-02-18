# Frigate Event Buffer - State-Aware Orchestrator
# ================================================
#
# Environment Variables (override config.yaml):
#   MQTT_BROKER     - MQTT broker IP (required)
#   MQTT_PORT       - MQTT broker port (default: 1883)
#   BUFFER_IP       - Buffer container's reachable IP (required, used in notification URLs)
#   FRIGATE_URL     - Frigate API base URL (required)
#   STORAGE_PATH    - Storage directory (default: /app/storage)
#   RETENTION_DAYS  - Days to retain events (default: 3)
#   FLASK_PORT      - Flask server port (default: 5055)
#   LOG_LEVEL       - Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
#
# Volume Mounts:
#   /app/storage     - Event storage (clips, snapshots, summaries)
#   /app/config.yaml - Configuration file (optional, for camera/label filtering)
#
# Example Docker Run:
#   docker run -d \
#     -p 5055:5055 \
#     -v /mnt/user/appdata/frigate_buffer:/app/storage \
#     -v /mnt/user/appdata/frigate_buffer/config.yaml:/app/config.yaml:ro \
#     frigate-buffer

# -----------------------------------------------------------------------------
# FFmpeg with NVENC is supplied by the build context (see BUILD_NVENC.md).
# Run scripts/build-ffmpeg-nvenc.sh on a host with NVIDIA GPU first; it writes
# ffmpeg-nvenc-artifacts/ at repo root. Then build with context .: -f Dockerfile .
# -----------------------------------------------------------------------------
FROM python:3.12-slim

# FFmpeg (NVENC) from pre-built artifacts in context. If this COPY fails, run
# scripts/build-ffmpeg-nvenc.sh from repo root, then build with context .
COPY ffmpeg-nvenc-artifacts/ffmpeg /usr/local/bin/ffmpeg
COPY ffmpeg-nvenc-artifacts/ffprobe /usr/local/bin/ffprobe
COPY ffmpeg-nvenc-artifacts/lib /usr/local/lib
RUN ldconfig /usr/local/lib

# OpenCV headless runtime (libxcb, libGL, X11) so cv2 import works
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
        libxcb-shm0 \
        libxext6 \
        libxrender1 \
        libsm6 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies only (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project and source (code-only changes only invalidate from here)
COPY pyproject.toml README.md* ./
COPY src/ ./src/

# Install the app only, no deps (fast when only code changed)
RUN pip install --no-cache-dir --no-deps .

# Copy the rest of your files
COPY config.example.yaml .
RUN mkdir -p /app/storage

EXPOSE 5055

HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5055/status || exit 1

CMD ["python", "-m", "frigate_buffer.main"]
