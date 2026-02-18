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
# Stage 1: Build FFmpeg with NVENC (and libx264 fallback)
# Uses nvidia/cuda runtime for libnvidia-encode at link time; at container run
# time the same lib is provided by NVIDIA Container Toolkit.
# -----------------------------------------------------------------------------
ARG FFMPEG_VERSION=7.0.2
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS ffmpeg-builder
ARG FFMPEG_VERSION
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    git \
    libass-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libmp3lame-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libx264-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    meson \
    nasm \
    ninja-build \
    pkg-config \
    texinfo \
    wget \
    yasm \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# nv-codec-headers: required for --enable-nvenc (headers only; libnvidia-encode comes from CUDA runtime)
RUN git clone --depth 1 https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers \
    && cd /tmp/nv-codec-headers && make install && cd / && rm -rf /tmp/nv-codec-headers

# Download and build FFmpeg with NVENC + libx264 (fallback used when GPU path unavailable)
RUN wget -q https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz -O /tmp/ffmpeg.tar.xz \
    && tar xf /tmp/ffmpeg.tar.xz -C /tmp && rm /tmp/ffmpeg.tar.xz

RUN cd /tmp/ffmpeg-${FFMPEG_VERSION} && \
    ./configure \
    --prefix=/opt/ffmpeg \
    --enable-gpl \
    --enable-nonfree \
    --enable-nvenc \
    --enable-libx264 \
    --enable-libass \
    --enable-libfreetype \
    --enable-libgnutls \
    --enable-libmp3lame \
    --enable-libvorbis \
    --disable-debug \
    --disable-doc \
    --disable-ffplay \
    --enable-shared \
    --extra-cflags="-I/usr/local/cuda/include" \
    --extra-ldflags="-L/usr/local/cuda/lib64" \
    && make -j$(nproc) && make install && rm -rf /tmp/ffmpeg-${FFMPEG_VERSION}

# -----------------------------------------------------------------------------
# Stage 2: Runtime image with Python app and FFmpeg (NVENC) from builder
# -----------------------------------------------------------------------------
FROM python:3.12-slim

# Copy FFmpeg binaries and shared libs from builder. libnvidia-encode is NOT
# copied; it is provided at runtime by NVIDIA Container Toolkit when using
# deploy.resources.reservations.devices (nvidia GPU).
COPY --from=ffmpeg-builder /opt/ffmpeg/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg-builder /opt/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe
COPY --from=ffmpeg-builder /opt/ffmpeg/lib /usr/local/lib
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
