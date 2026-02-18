# Same content as repo root Dockerfile; keep in sync. Use with build context = repo/stack root (e.g. docker build -f src/Dockerfile . or compose build with context: .).
# FFmpeg artifacts: run scripts/build-ffmpeg-nvenc.sh first (writes to src/ffmpeg-nvenc-artifacts/).
# COPY paths use src/ so the build context must be the repo/stack root (.).
FROM python:3.12-slim

COPY src/ffmpeg-nvenc-artifacts/ffmpeg /usr/local/bin/ffmpeg
COPY src/ffmpeg-nvenc-artifacts/ffprobe /usr/local/bin/ffprobe
COPY src/ffmpeg-nvenc-artifacts/lib /usr/local/lib
RUN ldconfig /usr/local/lib

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

COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/pyproject.toml ./
COPY src/frigate_buffer/ ./src/frigate_buffer/
COPY src/config.example.yaml .
RUN mkdir -p /app/storage

RUN pip install --no-cache-dir --no-deps .

EXPOSE 5055

HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5055/status || exit 1

CMD ["python", "-m", "frigate_buffer.main"]
