# Build from repo root: docker build -t frigate-buffer:latest .
# FFmpeg with NVENC comes from multi-stage build; no script or artifact folder required. See BUILD_NVENC.md.
ARG FFMPEG_NVENC_IMAGE=jrottenberg/ffmpeg:7.0-nvidia2204
FROM ${FFMPEG_NVENC_IMAGE} AS ffmpeg_nvenc

FROM python:3.12-slim
# Copy FFmpeg + ffprobe and shared libs from the NVENC-enabled image (jrottenberg uses PREFIX=/opt/ffmpeg; libs in lib64).
COPY --from=ffmpeg_nvenc /opt/ffmpeg/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg_nvenc /opt/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe
COPY --from=ffmpeg_nvenc /opt/ffmpeg/lib64/. /usr/local/lib/
RUN ldconfig /usr/local/lib 2>/dev/null || true

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

COPY pyproject.toml ./
COPY src/frigate_buffer/ ./src/frigate_buffer/
COPY config.example.yaml .
RUN mkdir -p /app/storage

RUN pip install --no-cache-dir --no-deps .

EXPOSE 5055

HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5055/status || exit 1

CMD ["python", "-m", "frigate_buffer.main"]
