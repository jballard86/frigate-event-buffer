# `frigate_amd_decode` (gpu-03)

C++ extension for [gpu-03 AMD ROCm plan](../../docs/Multi_GPU_Support_Integration_Plan/gpu-03-amd-rocm.md): **CMake + pybind11 + libtorch + FFmpeg** in one module.

**Decode:** ``AmdDecoderSession`` prefers **VAAPI** on a DRM render node (default ``/dev/dri/renderD128`` + ``device_index``, or ``FRIGATE_AMD_VAAPI_DEVICE``). If device init or ``avcodec_open2`` fails, falls back to **software** libavcodec.

**Zero-copy (optional HIP):** When CMake finds **ROCm HIP** (``AMD_DECODE_HIP=ON``, default on Linux), the session builds a **DRM** hwframes context and maps VAAPI frames to **DRM PRIME**; DMA-BUF fds are imported with **``hipImportExternalMemory``**, NV12 is converted to RGB on-GPU via **``nv12_to_rgb_hip.hip``**, and the module returns **``torch::kCUDA``** (ROCm) **uint8 BCHW** tensors. Call **``uses_zero_copy_decode()``** after construction to see if the fast path is active. If mapping/HIP fails, the code falls back to **``av_hwframe_transfer_data``** + **sws_scale** → **CPU** tensors unless **``FRIGATE_AMD_DECODE_STRICT_ZEROCOPY=1``** (then errors propagate). Set **``FRIGATE_AMD_DECODE_DISABLE_ZEROCOPY=1``** to force the CPU transfer path without rebuilding.

## Environment

| Variable | Effect |
|----------|--------|
| ``FRIGATE_AMD_DECODE_FORCE_SW=1`` | Skip VAAPI; software decode only. |
| ``FRIGATE_AMD_VAAPI_DEVICE`` | Override DRM node (e.g. ``/dev/dri/renderD129``). |
| ``FRIGATE_AMD_DECODE_DISABLE_ZEROCOPY=1`` | Skip HIP/DRM map; use CPU transfer for hw frames. |
| ``FRIGATE_AMD_DECODE_STRICT_ZEROCOPY=1`` | Do not fall back to CPU transfer when zero-copy fails. |

## Prerequisites (Linux)

- CMake ≥ 3.18, C++17 compiler, ``pkg-config``
- FFmpeg **development** packages: ``libavcodec``, ``libavformat``, ``libavutil``, ``libswscale``
- Python 3.10+ with **PyTorch** installed. For ROCm images, match **`requirements-rocm.txt`** torch pins when linking libtorch.
- **ROCm + HIP** (e.g. ``/opt/rocm``) for the zero-copy path; CMake sets ``CMAKE_HIP_ARCHITECTURES`` if unset (override for your GPU).
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

Disable HIP (CPU transfer only): ``cmake -S . -B build -DAMD_DECODE_HIP=OFF``.

The shared library is under ``build/`` (e.g. ``frigate_amd_decode*.so``). Add that directory to ``PYTHONPATH`` to ``import frigate_amd_decode``.

On **Windows**, the project defaults to ``AMD_DECODE_BUILD=OFF``; use **Linux** (WSL2 with ``/dev/dri``, container, or bare metal).

## Smoke test

```bash
PYTHONPATH=native/amd_decode/build python -c "
import frigate_amd_decode as m
print(m.version())
s = m.AmdDecoderSession('path/to/clip.mp4', 0)
print('zero_copy', s.uses_zero_copy_decode())
t = s.get_frame_at_index(0)
print(t.shape, t.dtype, t.device)
"
```

## Host script

From repo root: **`scripts/build_amd_decode.sh`** (same pattern as Intel).

## Docker

Root **`Dockerfile.rocm`** (multi-stage) builds this extension in stage 1 and copies **`frigate_amd_decode*.so`** into the ROCm runtime image. See **`docs/INSTALL.md`** (AMD GPU Docker) and **`docker-compose.rocm.example.yml`**. Manual CI: **`.github/workflows/rocm_docker_build.yml`**.
