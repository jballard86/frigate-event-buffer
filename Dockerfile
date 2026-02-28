# syntax=docker/dockerfile:1
# Build from repo root: docker build -t frigate-buffer:latest .
# Base: Ubuntu 24.04 + CUDA 12.6 runtime for NVDEC/NVENC. FFmpeg from distro. PyNvVideoCodec from PyPI. No vendored wheels.
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


# Prefer python3.12 for the rest of the build and runtime.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

WORKDIR /app

COPY requirements.txt .

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
COPY examples/config.example.yaml ./
COPY run_server.py ./
RUN mkdir -p /app/storage

# Cache pip wheels so code-only rebuilds reuse deps (BuildKit cache mount).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages .

EXPOSE 5055

HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5055/status || exit 1

CMD ["python3", "run_server.py"]
