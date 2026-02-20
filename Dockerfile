# Build from repo root: docker build -t frigate-buffer:latest .
# FFmpeg from multi-stage build provides NVDEC (decode) for ffmpegcv; encoding is not used (clips are stored as-is).
# Final stage is Ubuntu 24.04 (Python 3.12). Ubuntu base matches FFmpeg donor for NVIDIA lib compatibility.
ARG FFMPEG_NVENC_IMAGE=jrottenberg/ffmpeg:7.0-nvidia2204
FROM ${FFMPEG_NVENC_IMAGE} AS ffmpeg_nvenc

# Gather all FFmpeg/CUDA/NPP libs from donor into one tree (libnppig etc. may live in lib64 or cuda/lib64).
FROM ${FFMPEG_NVENC_IMAGE} AS ffmpeg_libs
RUN mkdir -p /out/lib && \
    cp -a /usr/local/lib/. /out/lib/ 2>/dev/null || true && \
    (cp -an /usr/local/lib64/. /out/lib/ 2>/dev/null || true) && \
    (cp -an /usr/local/cuda/lib64/. /out/lib/ 2>/dev/null || true)

FROM ubuntu:24.04
# Copy FFmpeg + ffprobe and shared libs (including NPP/CUDA libs like libnppig.so.12) from the NVENC image.
COPY --from=ffmpeg_nvenc /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg_nvenc /usr/local/bin/ffprobe /usr/local/bin/ffprobe
COPY --from=ffmpeg_libs /out/lib/. /usr/local/lib/
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
    libgomp1 \
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
