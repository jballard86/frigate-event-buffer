#!/usr/bin/env bash
# Build Dockerfile.rocm and run in-container ROCm smoke with /dev/kfd + DRI (gpu-03 Phase 6).
# Usage: from repo root, on a Linux host with Docker, AMD GPU, /dev/kfd, and /dev/dri:
#   ./scripts/run_amd_rocm_docker_smoke.sh --strict --strict-native
# Optional: KFD=/dev/kfd RENDER_NODE=/dev/dri/renderD129 IMAGE=frigate-buffer:rocm ./scripts/run_amd_rocm_docker_smoke.sh ...
# Trailing args are passed to smoke_amd_rocm_torch.py (e.g. /tmp/clip.mp4 inside the container).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

KFD="${KFD:-/dev/kfd}"
RENDER_NODE="${RENDER_NODE:-/dev/dri/renderD128}"
IMAGE="${IMAGE:-frigate-buffer:rocm}"
BUILD="${BUILD:-1}"

if [[ ! -e "$KFD" ]]; then
  echo "error: AMD KFD device not found: $KFD (set KFD=...)" >&2
  exit 1
fi

if [[ ! -e "$RENDER_NODE" ]]; then
  echo "error: render node not found: $RENDER_NODE (set RENDER_NODE=...)" >&2
  exit 1
fi

if [[ "$BUILD" == "1" ]]; then
  docker build -f Dockerfile.rocm -t "$IMAGE" .
fi

# shellcheck disable=SC2068
exec docker run --rm \
  --device "${KFD}:${KFD}" \
  --device "${RENDER_NODE}:${RENDER_NODE}" \
  --group-add video \
  --group-add render \
  "$IMAGE" \
  python3 scripts/smoke_amd_rocm_torch.py "$@"
