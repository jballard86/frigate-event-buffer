# Build from repo root: docker build -t frigate-buffer:latest .
# FFmpeg from multi-stage build provides NVDEC (decode) for ffmpegcv.
# Final stage is python:3.12-slim-bookworm. FFmpeg + CUDA/NPP libs copied from donor for hardware decode only.
ARG FFMPEG_DONOR_IMAGE=jrottenberg/ffmpeg:7.0-nvidia2204
# Set USE_GUI_OPENCV=true for faster builds (~1-2 min vs 15+ min) during development.
# Trade-off: GUI version adds ~100-150 MB to image size and includes X11 dependencies.
ARG USE_GUI_OPENCV=false
FROM ${FFMPEG_DONOR_IMAGE} AS ffmpeg_donor

# Gather all FFmpeg/CUDA/NPP libs from donor into one tree (libnppig etc. may live in lib64 or cuda/lib64).
# Copy donor system libs required by ffmpeg/ffprobe (from ldd); exclude glibc/toolchain so the final image keeps Debian's.
FROM ${FFMPEG_DONOR_IMAGE} AS ffmpeg_libs
RUN mkdir -p /out/lib && \
    cp -a /usr/local/lib/. /out/lib/ 2>/dev/null || true && \
    (cp -an /usr/local/lib64/. /out/lib/ 2>/dev/null || true) && \
    (cp -an /usr/local/cuda/lib64/. /out/lib/ 2>/dev/null || true) && \
    cp -L /lib/x86_64-linux-gnu/libcrypto.so.3 /lib/x86_64-linux-gnu/libexpat.so.1 \
        /lib/x86_64-linux-gnu/libgomp.so.1 /lib/x86_64-linux-gnu/libssl.so.3 \
        /lib/x86_64-linux-gnu/libz.so.1 /out/lib/

FROM python:3.12-slim-bookworm
# Copy FFmpeg + ffprobe and shared libs (NPP/CUDA for NVDEC) from donor. No GUI libs; opencv-python-headless only.
COPY --from=ffmpeg_donor /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg_donor /usr/local/bin/ffprobe /usr/local/bin/ffprobe
COPY --from=ffmpeg_libs /out/lib/. /usr/local/lib/
RUN echo '/usr/local/lib' > /etc/ld.so.conf.d/local.conf && ldconfig

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install requirements; conditionally skip opencv swap for faster GUI builds during development.
RUN if [ "$USE_GUI_OPENCV" = "true" ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt && \
        pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true && \
        pip install --no-cache-dir --force-reinstall opencv-python-headless; \
    fi

# (Opencv fix above: When USE_GUI_OPENCV=false, uninstalls opencv-python (GUI) from ultralytics
# and reinstalls headless only so the container has no X11/libxcb dependency. Set USE_GUI_OPENCV=true
# for faster builds (~1-2 min vs 15+ min) during development; trade-off is ~100-150 MB larger image.)

COPY pyproject.toml ./
COPY src/frigate_buffer/ ./src/frigate_buffer/
COPY config.example.yaml ./
RUN mkdir -p /app/storage

# Install package with deps from pyproject.toml so all runtime deps (e.g. pyyaml) are present.
RUN pip install --no-cache-dir .

EXPOSE 5055

HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5055/status || exit 1

CMD ["python", "-m", "frigate_buffer.main"]
