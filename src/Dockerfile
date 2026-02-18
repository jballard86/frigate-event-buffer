# Same as repo root Dockerfile; use when build context is "src" (e.g. docker build -f src/Dockerfile src).
# FFmpeg artifacts must be in context: run scripts/build-ffmpeg-nvenc.sh first (writes to src/ffmpeg-nvenc-artifacts/).
FROM python:3.12-slim

COPY ffmpeg-nvenc-artifacts/ffmpeg /usr/local/bin/ffmpeg
COPY ffmpeg-nvenc-artifacts/ffprobe /usr/local/bin/ffprobe
COPY ffmpeg-nvenc-artifacts/lib /usr/local/lib
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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md* ./
COPY frigate_buffer/ ./src/frigate_buffer/
COPY config.example.yaml .
RUN mkdir -p /app/storage

RUN pip install --no-cache-dir --no-deps .

EXPOSE 5055

HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5055/status || exit 1

CMD ["python", "-m", "frigate_buffer.main"]
