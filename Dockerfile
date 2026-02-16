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

FROM python:3.10-slim

# Install ffmpeg for video transcoding
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project and install package (src layout)
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Example config
COPY config.example.yaml .

# Create storage directory
RUN mkdir -p /app/storage

# Expose Flask port
EXPOSE 5055

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5055/status || exit 1

CMD ["python", "-m", "frigate_buffer.main"]
