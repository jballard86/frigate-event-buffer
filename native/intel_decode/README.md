# `frigate_intel_decode` (gpu-02)

C++ extension for [gpu-02 Intel Arc plan](../../docs/Multi_GPU_Support_Integration_Plan/gpu-02-intel-arc.md): **CMake + pybind11 + libtorch + FFmpeg** linked in one module.

**Decode:** ``IntelDecoderSession`` uses **QSV only** (**h264_qsv** / **hevc_qsv**) with QSV device ``auto``. Unsupported codecs or QSV init failure **throws**; there is no software libavcodec fallback (same policy as the rest of the project: no CPU decode path).

**DRM PRIME probe:** The session initializes a DRM hwframes context and probes mapping a hw frame to **DRM PRIME NV12**; read ``can_map_to_drm_prime()`` for driver/FFmpeg compatibility.

**XPU zero-copy (required):** Build with ``-DINTEL_DECODE_XPU_ZEROCOPY=ON`` (requires Intel oneAPI ``icpx`` + an XPU torch stack). Returns **XPU** BCHW uint8 when the DRM PRIME → DMA-BUF path works. This project does **not** support a host readback + swscale RGB fallback in the Intel path; failure to map/import is treated as a decode failure.

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

To enable XPU zero-copy (requires Intel oneAPI `icpx` and an XPU torch stack):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DINTEL_DECODE_XPU_ZEROCOPY=ON
cmake --build build -j
```

The shared library is under `build/` (e.g. `frigate_intel_decode*.so`). Add that directory to `PYTHONPATH` to `import frigate_intel_decode`.

On **Windows**, the project defaults to `INTEL_DECODE_BUILD=OFF`; use **WSL** or **Linux Docker** for this spike, or pass `-DINTEL_DECODE_BUILD=ON` with MSVC + FFmpeg + libtorch paths configured yourself.

## Smoke test

```bash
PYTHONPATH=native/intel_decode/build python -c "
import frigate_intel_decode as m
print(m.version())
t = m.decode_first_frame_bchw_rgb('path/to/clip.mp4')
print(t.shape, t.dtype)
"
```

## Docker (multi-stage)

Root **`Dockerfile.intel`** builds this extension in stage 1 and copies **`frigate_intel_decode*.so`** into the runtime image. See **`docs/INSTALL.md`** (Intel GPU Docker) and **`docker-compose.intel.example.yml`**. On Arc hardware, **`docs/Multi_GPU_Support_Integration_Plan/intel-arc-hardware-smoke.md`** describes DRI smoke and **`scripts/run_intel_arc_docker_smoke.sh`**.

## Packaging note

Host / CI builds can still use **`scripts/build_intel_decode.sh`**. setuptools/scikit-build wheel packaging remains optional.
