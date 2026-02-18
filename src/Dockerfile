# Same content as repo root Dockerfile; keep in sync.
# Build context: either repo/stack root (.) or repo root (e.g. src/). Use SRC_PREFIX to match.
# - Context = stack root (e.g. docker build -f src/Dockerfile .): SRC_PREFIX=src/ (default).
# - Context = repo root (e.g. from inside src/: docker build -f Dockerfile --build-arg SRC_PREFIX= .): SRC_PREFIX=.
# FFmpeg artifacts: run scripts/build-ffmpeg-nvenc.sh first (writes to src/ffmpeg-nvenc-artifacts/ or ffmpeg-nvenc-artifacts/).
FROM python:3.12-slim
ARG SRC_PREFIX=src/

COPY ${SRC_PREFIX}ffmpeg-nvenc-artifacts/ffmpeg /usr/local/bin/ffmpeg
COPY ${SRC_PREFIX}ffmpeg-nvenc-artifacts/ffprobe /usr/local/bin/ffprobe
COPY ${SRC_PREFIX}ffmpeg-nvenc-artifacts/lib /usr/local/lib
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

COPY ${SRC_PREFIX}requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ${SRC_PREFIX}pyproject.toml ./
COPY ${SRC_PREFIX}frigate_buffer/ ./src/frigate_buffer/
COPY ${SRC_PREFIX}config.example.yaml .
RUN mkdir -p /app/storage

RUN pip install --no-cache-dir --no-deps .

EXPOSE 5055

HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5055/status || exit 1

CMD ["python", "-m", "frigate_buffer.main"]
