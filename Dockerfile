# Build from repo root: docker build -t frigate-buffer:latest .
# FFmpeg from multi-stage build provides NVDEC (decode) for ffmpegcv; encoding is not used (clips are stored as-is).
# Final stage is python:3.12-slim-bookworm. FFmpeg + CUDA/NPP libs copied from donor for hardware decode only.
ARG FFMPEG_DONOR_IMAGE=jrottenberg/ffmpeg:7.0-nvidia2204
FROM ${FFMPEG_DONOR_IMAGE} AS ffmpeg_donor

# Gather all FFmpeg/CUDA/NPP libs from donor into one tree (libnppig etc. may live in lib64 or cuda/lib64).
FROM ${FFMPEG_DONOR_IMAGE} AS ffmpeg_libs
RUN mkdir -p /out/lib && \
    cp -a /usr/local/lib/. /out/lib/ 2>/dev/null || true && \
    (cp -an /usr/local/lib64/. /out/lib/ 2>/dev/null || true) && \
    (cp -an /usr/local/cuda/lib64/. /out/lib/ 2>/dev/null || true)

FROM python:3.12-slim-bookworm
# Copy FFmpeg + ffprobe and shared libs (NPP/CUDA for NVDEC) from donor. No GUI libs; opencv-python-headless only.
COPY --from=ffmpeg_donor /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg_donor /usr/local/bin/ffprobe /usr/local/bin/ffprobe
COPY --from=ffmpeg_libs /out/lib/. /usr/local/lib/
RUN echo '/usr/local/lib' > /etc/ld.so.conf.d/local.conf && ldconfig

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src/frigate_buffer/ ./src/frigate_buffer/
COPY config.example.yaml ./
RUN mkdir -p /app/storage

RUN pip install --no-cache-dir --no-deps .

# Ultralytics pulls in opencv-python (GUI); ensure only headless is present so no libxcb/X11 in container.
# Uninstall both variants then force-reinstall headless so cv2 is always present.
RUN pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless

EXPOSE 5055

HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5055/status || exit 1

CMD ["python", "-m", "frigate_buffer.main"]
