# `frigate_intel_decode` (gpu-02)

C++ extension for [gpu-02 Intel Arc plan](../../docs/Multi_GPU_Support_Integration_Plan/gpu-02-intel-arc.md): **CMake + pybind11 + libtorch + FFmpeg** linked in one module.

**Phase 2:** ``IntelDecoderSession`` prefers **h264_qsv / hevc_qsv** + **QSV** device (`auto`); falls back to software libavcodec if QSV init fails. Set ``FRIGATE_INTEL_DECODE_FORCE_SW=1`` to skip QSV. Decoded frames transfer to **CPU** BCHW ``uint8``; Python ``gpu_backends/intel/decoder.py`` may **``.to(xpu)``** when Intel Extension for PyTorch is available.

## Prerequisites (Linux)

- CMake ≥ 3.18, C++17 compiler, `pkg-config`
- FFmpeg **development** packages: `libavcodec`, `libavformat`, `libavutil`, `libswscale`
- Python 3.10+ with **PyTorch** installed (pip wheel is fine). For parity with **`Dockerfile.intel`**, match **`torch` / `torchvision`** versions in root **`requirements-intel.txt`** when building the `.so`.
- Network for first CMake run (FetchContent downloads pybind11)

Example (Debian/Ubuntu):

```bash
sudo apt install build-essential cmake pkg-config \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  python3-dev
```

## Build

Point CMake at the same libtorch as your `torch` wheel:

```bash
cd native/intel_decode
export CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The shared library is under `build/` (e.g. `frigate_intel_decode*.so`). Add that directory to `PYTHONPATH` to `import frigate_intel_decode`.

On **Windows**, the project defaults to `INTEL_DECODE_BUILD=OFF`; use **WSL** or **Linux Docker** for this spike, or pass `-DINTEL_DECODE_BUILD=ON` with MSVC + FFmpeg + libtorch paths configured yourself.

## Smoke test

```bash
PYTHONPATH=native/intel_decode/build python -c "
import frigate_intel_decode as m
print(m.version())
t = m.decode_first_frame_bchw_rgb_sw('path/to/clip.mp4')
print(t.shape, t.dtype)
"
```

## Docker (multi-stage)

Root **`Dockerfile.intel`** builds this extension in stage 1 and copies **`frigate_intel_decode*.so`** into the runtime image. See **`docs/INSTALL.md`** (Intel GPU Docker) and **`docker-compose.intel.example.yml`**. On Arc hardware, **`docs/Multi_GPU_Support_Integration_Plan/intel-arc-hardware-smoke.md`** describes DRI smoke and **`scripts/run_intel_arc_docker_smoke.sh`**.

## Packaging note

Host / CI builds can still use **`scripts/build_intel_decode.sh`**. setuptools/scikit-build wheel packaging remains optional.
