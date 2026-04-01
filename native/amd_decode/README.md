# `frigate_amd_decode` (gpu-03 Phase 3)

C++ extension for [gpu-03 AMD ROCm plan](../../docs/Multi_GPU_Support_Integration_Plan/gpu-03-amd-rocm.md): **CMake + pybind11 + libtorch + FFmpeg** in one module.

**Phase 3:** ``AmdDecoderSession`` prefers **VAAPI** on a DRM render node (default ``/dev/dri/renderD128`` + ``device_index``, or ``FRIGATE_AMD_VAAPI_DEVICE``). If device init or ``avcodec_open2`` fails, falls back to **software** libavcodec. Decoded frames are **CPU** BCHW ``uint8`` RGB (same transfer path as Intel QSV hw frames). **ROCm zero-copy** tensors from VAAPI/DMA-BUF are **not** in this phase; Python may ``.to(cuda)`` when using a ROCm torch build.

## Environment

| Variable | Effect |
|----------|--------|
| ``FRIGATE_AMD_DECODE_FORCE_SW=1`` | Skip VAAPI; software decode only. |
| ``FRIGATE_AMD_VAAPI_DEVICE`` | Override DRM node (e.g. ``/dev/dri/renderD129``). |

## Prerequisites (Linux)

- CMake ≥ 3.18, C++17 compiler, ``pkg-config``
- FFmpeg **development** packages: ``libavcodec``, ``libavformat``, ``libavutil``, ``libswscale``
- Python 3.10+ with **PyTorch** installed. For ROCm images, match **`requirements-rocm.txt`** torch pins when linking libtorch.
- Network on first CMake run (FetchContent downloads pybind11)

Example (Debian/Ubuntu):

```bash
sudo apt install build-essential cmake pkg-config \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  python3-dev
```

## Build

```bash
cd native/amd_decode
export CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The shared library is under ``build/`` (e.g. ``frigate_amd_decode*.so``). Add that directory to ``PYTHONPATH`` to ``import frigate_amd_decode``.

On **Windows**, the project defaults to ``AMD_DECODE_BUILD=OFF``; use **Linux** (WSL2 with ``/dev/dri``, container, or bare metal).

## Smoke test

```bash
PYTHONPATH=native/amd_decode/build python -c "
import frigate_amd_decode as m
print(m.version())
t = m.decode_first_frame_bchw_rgb('path/to/clip.mp4')
print(t.shape, t.dtype)
"
```

## Host script

From repo root: **`scripts/build_amd_decode.sh`** (same pattern as Intel).

## Docker

Root **`Dockerfile.rocm`** (multi-stage) builds this extension in stage 1 and copies **`frigate_amd_decode*.so`** into the ROCm runtime image. See **`docs/INSTALL.md`** (AMD GPU Docker) and **`docker-compose.rocm.example.yml`**. Manual CI: **`.github/workflows/rocm_docker_build.yml`**.
