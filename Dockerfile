# syntax=docker/dockerfile:1
# Build from repo root: docker build -t frigate-buffer:latest .
# Base: Ubuntu 24.04 + CUDA 12.6 runtime for NVDEC/NVENC. FFmpeg 6.1 from distro. libyuv and libspdlog are vendored in wheels/ and copied into /usr/lib/x86_64-linux-gnu/ for NeLux wheel ABI match (zero-copy GPU compilation).
# NeLux wheel is vendored in wheels/ (built against FFmpeg 6.1 / Ubuntu 24.04); do not use PyPI.
# Copy vendored C++ shared libraries from NeLux image to prevent ABI mismatch
# Use BuildKit (default in Docker 23+) for pip cache on app install: faster code-only rebuilds.
ARG USE_GUI_OPENCV=false
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y universe && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    curl \
    ca-certificates \
    libgomp1 \
    && apt-get purge -y software-properties-common && apt-get autoremove -y --purge \
    && rm -rf /var/lib/apt/lists/*

# Copy vendored C++ shared libraries for NeLux (match wheel ABI)
COPY wheels/libyuv.so* /usr/lib/x86_64-linux-gnu/
COPY wheels/libspdlog.so* /usr/lib/x86_64-linux-gnu/

# Refresh the Linux dynamic linker cache so it sees the new files
RUN ldconfig

# Prefer python3.12 for the rest of the build and runtime.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

WORKDIR /app

COPY requirements.txt .
COPY wheels/ ./wheels/

# Install deps; when USE_GUI_OPENCV=false, swap to opencv-python-headless (no X11).
RUN if [ "$USE_GUI_OPENCV" = "true" ]; then \
        pip3 install --no-cache-dir --break-system-packages -r requirements.txt; \
    else \
        pip3 install --no-cache-dir --break-system-packages -r requirements.txt && \
        pip3 uninstall -y opencv-python opencv-python-headless 2>/dev/null || true && \
        pip3 install --no-cache-dir --force-reinstall --break-system-packages opencv-python-headless; \
    fi

COPY pyproject.toml ./
COPY src/frigate_buffer/ ./src/frigate_buffer/
COPY config.example.yaml ./
RUN mkdir -p /app/storage

# Cache pip wheels so code-only rebuilds reuse deps (BuildKit cache mount).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages .

EXPOSE 5055

HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5055/status || exit 1

CMD ["python3", "-m", "frigate_buffer.main"]
