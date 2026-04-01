#!/usr/bin/env bash
# Build Dockerfile.intel and run in-container smoke with DRI passthrough (Intel Arc / gpu-02 Phase 8).
# Usage: from repo root, on a Linux host with Docker and /dev/dri:
#   ./scripts/run_intel_arc_docker_smoke.sh
# Optional: RENDER_NODE=/dev/dri/renderD129 IMAGE=frigate-buffer:intel ./scripts/run_intel_arc_docker_smoke.sh --strict --vainfo
# Trailing args are passed to smoke_intel_gpu_path.py (e.g. a clip path inside the container).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RENDER_NODE="${RENDER_NODE:-/dev/dri/renderD128}"
IMAGE="${IMAGE:-frigate-buffer:intel}"
BUILD="${BUILD:-1}"

if [[ ! -e "$RENDER_NODE" ]]; then
  echo "error: render node not found: $RENDER_NODE (set RENDER_NODE=...)" >&2
  exit 1
fi

if [[ "$BUILD" == "1" ]]; then
  docker build -f Dockerfile.intel -t "$IMAGE" .
fi

# shellcheck disable=SC2068
exec docker run --rm \
  --device "${RENDER_NODE}:${RENDER_NODE}" \
  --group-add video \
  --group-add render \
  "$IMAGE" \
  python3 scripts/smoke_intel_gpu_path.py "$@"
