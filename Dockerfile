# Build from repo root: docker build -t frigate-buffer:latest .
# FFmpeg with NVENC comes from multi-stage build; no script or artifact folder required. See BUILD_NVENC.md.
# Final stage is Ubuntu 22.04 to match the FFmpeg donor (jrottenberg) and avoid distro/ABI mismatch with NVIDIA libs.
ARG FFMPEG_NVENC_IMAGE=jrottenberg/ffmpeg:7.0-nvidia2204
FROM ${FFMPEG_NVENC_IMAGE} AS ffmpeg_nvenc

FROM ubuntu:22.04
# Copy FFmpeg + ffprobe and shared libs from the NVENC-enabled image (jrottenberg 7.0-nvidia2204 uses /usr/local).
COPY --from=ffmpeg_nvenc /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg_nvenc /usr/local/bin/ffprobe /usr/local/bin/ffprobe
COPY --from=ffmpeg_nvenc /usr/local/lib/. /usr/local/lib/
RUN ldconfig /usr/local/lib 2>/dev/null || true

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libxcb-shm0 \
    libxext6 \
    libxrender1 \
    libsm6 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src/frigate_buffer/ ./src/frigate_buffer/
COPY config.example.yaml .
RUN mkdir -p /app/storage

RUN python3.12 -m pip install --no-cache-dir --no-deps .

EXPOSE 5055

HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5055/status || exit 1

CMD ["python3.12", "-m", "frigate_buffer.main"]
