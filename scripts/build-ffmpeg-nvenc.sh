#!/usr/bin/env bash
# Build FFmpeg with NVENC in a one-off container that has GPU/driver access.
# Outputs to src/ffmpeg-nvenc-artifacts/ so "docker build -f src/Dockerfile src" includes them.
# Run from repo root on a host with NVIDIA GPU and Docker (e.g. Unraid Tower).
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Artifacts must live inside the build context. When context is "src", that is either
# REPO_ROOT/src (repo has scripts/ and src/ as siblings) or REPO_ROOT (repo is inside src/).
if [ -f "$REPO_ROOT/src/Dockerfile" ]; then
  OUT_DIR="$REPO_ROOT/src/ffmpeg-nvenc-artifacts"
else
  OUT_DIR="$REPO_ROOT/ffmpeg-nvenc-artifacts"
fi
FFMPEG_VERSION="${FFMPEG_VERSION:-7.0.2}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.2.0-devel-ubuntu22.04}"

echo "Building FFmpeg ${FFMPEG_VERSION} with NVENC (output: ${OUT_DIR})"
mkdir -p "$OUT_DIR"

docker run --rm --gpus all \
  -e DEBIAN_FRONTEND=noninteractive \
  -v "$OUT_DIR:/out" \
  "$CUDA_IMAGE" \
  bash -ec '
    apt-get update -qq && apt-get install -y --no-install-recommends \
      autoconf automake build-essential ca-certificates git libass-dev \
      libfreetype6-dev libmp3lame-dev libssl-dev libtool libva-dev libvdpau-dev \
      libvorbis-dev libx264-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev \
      meson nasm ninja-build pkg-config texinfo wget yasm zlib1g-dev
    git clone --depth 1 https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers
    cd /tmp/nv-codec-headers && make install && cd / && rm -rf /tmp/nv-codec-headers
    export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH}"
    wget -q "https://ffmpeg.org/releases/ffmpeg-'"${FFMPEG_VERSION}"'.tar.xz" -O /tmp/ffmpeg.tar.xz
    tar xf /tmp/ffmpeg.tar.xz -C /tmp && rm /tmp/ffmpeg.tar.xz
    cd /tmp/ffmpeg-'"${FFMPEG_VERSION}"'
    ./configure \
      --prefix=/opt/ffmpeg \
      --enable-gpl --enable-nonfree --enable-nvenc --enable-libx264 \
      --enable-libass --enable-libfreetype --enable-openssl \
      --enable-libmp3lame --enable-libvorbis \
      --disable-debug --disable-doc --disable-ffplay --enable-shared \
      --extra-cflags="-I/usr/local/include -I/usr/local/cuda/include" \
      --extra-ldflags="-L/usr/local/lib -L/usr/local/cuda/lib64" \
      2>&1 | tee /tmp/ffmpeg-config.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
    make -j$(nproc) && make install
    if ! /opt/ffmpeg/bin/ffmpeg -encoders 2>&1 | grep -q h264_nvenc; then
      echo "NVENC not in build" >&2
      echo "Configure summary (nvenc-related):" >&2
      grep -i nvenc /tmp/ffmpeg-config.log || true
      echo "Last 60 lines of configure output:" >&2
      tail -60 /tmp/ffmpeg-config.log >&2
      exit 1
    fi
    mkdir -p /out/lib
    cp -a /opt/ffmpeg/bin/ffmpeg /opt/ffmpeg/bin/ffprobe /out/
    cp -a /opt/ffmpeg/lib/. /out/lib/
    rm -rf /tmp/ffmpeg-'"${FFMPEG_VERSION}"
  '

echo "Done. Artifacts in ${OUT_DIR}. Build with: docker build -t frigate-buffer:latest -f src/Dockerfile src"
