#!/usr/bin/env bash
# Build native/amd_decode extension (Linux). See native/amd_decode/README.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/native/amd_decode"
BUILD="${SRC}/build"
CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
export CMAKE_PREFIX_PATH
cmake -S "${SRC}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD}" -j"$(nproc 2>/dev/null || echo 4)"
echo "Built: ${BUILD}/ (add to PYTHONPATH to import frigate_amd_decode)"
