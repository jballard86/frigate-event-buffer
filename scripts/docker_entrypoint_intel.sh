#!/bin/sh
# Ensure frigate_intel_decode .so resolves libtorch at runtime (multi-stage copy path).
set -e
TORCH_LIB="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
exec "$@"
