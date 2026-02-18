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

FROM python:3.12-slim

# 1. System deps: FFmpeg (NVENC) + OpenCV headless runtime (libxcb, libGL, X11)
ARG FFMPEG_BTBN_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        xz-utils \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
        libxcb-shm0 \
        libxext6 \
        libxrender1 \
        libsm6 \
    && curl -sL "${FFMPEG_BTBN_URL}" -o /tmp/ffmpeg.tar.xz && \
    tar -xJf /tmp/ffmpeg.tar.xz -C /tmp && \
    cp /tmp/ffmpeg-*/bin/ffmpeg /usr/local/bin/ && \
    cp /tmp/ffmpeg-*/bin/ffprobe /usr/local/bin/ && \
    rm -rf /tmp/ffmpeg.tar.xz /tmp/ffmpeg-* && \
    apt-get remove -y curl xz-utils && apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

    WORKDIR /app

    # 1. Install dependencies only (cached unless requirements.txt changes)
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # 2. Copy project and source (code-only changes only invalidate from here)
    COPY pyproject.toml README.md* ./
    COPY src/ ./src/

    # 3. Install the app only, no deps (fast when only code changed)
    RUN pip install --no-cache-dir --no-deps .

    # 4. Copy the rest of your files
    COPY config.example.yaml .
    RUN mkdir -p /app/storage
    
    EXPOSE 5055

HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5055/status || exit 1 [cite: 7]

CMD ["python", "-m", "frigate_buffer.main"]
