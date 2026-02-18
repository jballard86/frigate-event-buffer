# Build from repo root: docker build -t frigate-buffer:latest .
# FFmpeg with NVENC comes from multi-stage build; no script or artifact folder required. See BUILD_NVENC.md.
# Final stage is Ubuntu 24.04 (Python 3.12 in default repos; no PPA). Ubuntu base matches FFmpeg donor and avoids distro/ABI mismatch with NVIDIA libs.
ARG FFMPEG_NVENC_IMAGE=jrottenberg/ffmpeg:7.0-nvidia2204
FROM ${FFMPEG_NVENC_IMAGE} AS ffmpeg_nvenc

FROM ubuntu:24.04
# Copy FFmpeg + ffprobe and shared libs from the NVENC-enabled image (jrottenberg 7.0-nvidia2204 uses /usr/local).
COPY --from=ffmpeg_nvenc /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg_nvenc /usr/local/bin/ffprobe /usr/local/bin/ffprobe
COPY --from=ffmpeg_nvenc /usr/local/lib/. /usr/local/lib/
RUN ldconfig /usr/local/lib 2>/dev/null || true

ENV DEBIAN_FRONTEND=noninteractive
# Ubuntu 24.04 marks Python as externally managed (PEP 668); allow pip system install in this container.
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
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
RUN ln -sf /usr/bin/python3.12 /usr/bin/python

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
