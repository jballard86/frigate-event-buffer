# Building NeLux Wheel from Source (Linux)

NeLux does not publish pre-built Linux wheels on PyPI. This guide documents how to compile a wheel in an isolated Docker container so it can be vendored into downstream projects without repeating the 20+ minute build.

---

## Requirements

- **Docker** with BuildKit (default on Docker 20.10+)
- **NVIDIA Container Toolkit** (for GPU test step only — not needed for compilation)
- **~15 GB disk space** (CUDA base image ~4 GB, PyTorch ~2.5 GB, build artifacts ~8 GB)
- **Internet access** during build (pulls git repos and pip packages)

## Build Dependencies (installed automatically by the Dockerfile)

| Dependency | Source | Purpose |
|---|---|---|
| CUDA 12.6 toolkit | `nvidia/cuda:12.6.3-devel-ubuntu24.04` base image | nvcc compiler, CUDA headers |
| Python 3.12 | Ubuntu 24.04 system package | Build environment |
| FFmpeg 6.1 dev libs | Ubuntu 24.04 apt packages | Video encode/decode API (requires 5.1+ for `ch_layout` API) |
| nv-codec-headers | GitHub `FFmpeg/nv-codec-headers` | NVIDIA hardware codec definitions |
| libyuv | Chromium source + custom CMake config | Color space conversion |
| spdlog 1.14.1 | GitHub, built with **bundled fmt** | Logging (must use bundled fmt to avoid conflict with PyTorch's fmt v12) |
| libfmt-dev | Ubuntu 24.04 apt | Format library |
| PyTorch | pip | Tensor operations, CUDA extensions, cmake config |
| pybind11 | pip | Python/C++ bindings |
| scikit-build-core | pip | PEP 517 build backend |
| CMake 3.23+ | pip (overrides system cmake) | Build system |
| Ninja | Ubuntu 24.04 apt | Parallel build tool |

## Runtime Dependencies (needed in your app container)

- Python 3.12
- PyTorch (`import torch` **must** come before `import nelux`)
- NumPy
- FFmpeg 6.1+ shared libraries: libavcodec60, libavformat60, libavutil58, libswscale7, libswresample4, libavfilter9, libavdevice60
- NVIDIA GPU driver + CUDA runtime

---

## Files

You need one file: `Dockerfile.nelux`

```dockerfile
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv python3-dev \
      git cmake build-essential pkg-config ninja-build \
      libfmt-dev \
      libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
      libavdevice-dev libavfilter-dev libswresample-dev \
      software-properties-common && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers && \
    cd /tmp/nv-codec-headers && make install

RUN git clone https://chromium.googlesource.com/libyuv/libyuv /tmp/libyuv && \
    cd /tmp/libyuv && mkdir build && cd build && \
    cmake -GNinja .. -DCMAKE_INSTALL_PREFIX=/usr/local && \
    ninja && ninja install

RUN mkdir -p /usr/local/lib/cmake/libyuv && \
    printf '%s\n' \
      'set(libyuv_FOUND TRUE)' \
      'find_path(libyuv_INCLUDE_DIR NAMES libyuv.h PATH_SUFFIXES libyuv PATHS /usr/local/include)' \
      'find_library(libyuv_LIBRARY NAMES yuv PATHS /usr/local/lib)' \
      'if(libyuv_INCLUDE_DIR AND libyuv_LIBRARY)' \
      '  if(NOT TARGET libyuv::libyuv)' \
      '    add_library(libyuv::libyuv UNKNOWN IMPORTED)' \
      '    set_target_properties(libyuv::libyuv PROPERTIES' \
      '      IMPORTED_LOCATION "${libyuv_LIBRARY}"' \
      '      INTERFACE_INCLUDE_DIRECTORIES "${libyuv_INCLUDE_DIR}")' \
      '  endif()' \
      '  set(libyuv_LIBRARIES ${libyuv_LIBRARY})' \
      '  set(libyuv_INCLUDE_DIRS ${libyuv_INCLUDE_DIR})' \
      'endif()' \
      > /usr/local/lib/cmake/libyuv/libyuvConfig.cmake

RUN git clone --depth 1 --branch v1.14.1 https://github.com/gabime/spdlog.git /tmp/spdlog && \
    cd /tmp/spdlog && mkdir build && cd build && \
    cmake -GNinja .. -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DSPDLOG_FMT_EXTERNAL=OFF \
      -DSPDLOG_BUILD_SHARED=ON && \
    ninja && ninja install && ldconfig

RUN python3 -m venv /opt/build-env && \
    /opt/build-env/bin/pip install --upgrade pip setuptools build wheel torch 'cmake>=3.23' pybind11 scikit-build-core
```

---

## Step 1: Build the Base Image

Navigate to the directory containing `Dockerfile.nelux` and run:

```bash
docker build -t nelux-builder -f Dockerfile.nelux .
```

**Expected time:** 20–40 minutes on first run (mostly downloading CUDA image + PyTorch). Subsequent runs use Docker layer caching and complete in seconds unless the Dockerfile changes.

---

## Step 2: Compile the Wheel

Run an ephemeral container that clones NeLux, patches the build files, compiles, and copies the wheel to your current directory:

```bash
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  nelux-builder bash -c " \
    export PATH=/opt/build-env/bin:\$PATH && \
    git clone https://github.com/NevermindNilas/NeLux.git /tmp/NeLux && \
    cd /tmp/NeLux && \
    mkdir -p external/ffmpeg/include external/ffmpeg/lib && \
    cp -rL /usr/include/x86_64-linux-gnu/libavcodec /usr/include/x86_64-linux-gnu/libavformat \
       /usr/include/x86_64-linux-gnu/libavutil /usr/include/x86_64-linux-gnu/libavfilter \
       /usr/include/x86_64-linux-gnu/libswscale /usr/include/x86_64-linux-gnu/libswresample \
       /usr/include/x86_64-linux-gnu/libavdevice external/ffmpeg/include/ && \
    for lib in avcodec avformat avutil avfilter swscale swresample avdevice; do \
      cp -L /usr/lib/x86_64-linux-gnu/lib\${lib}.so external/ffmpeg/lib/; \
    done && \
    sed -i 's/find_package(Python3 3.13/find_package(Python3 3.12/' CMakeLists.txt && \
    sed -i 's/SPDLOG_FMT_EXTERNAL/SPDLOG_FMT_BUNDLED/' CMakeLists.txt && \
    sed -i '/FMT_HEADER_ONLY/d' CMakeLists.txt && \
    sed -i 's/requires-python = \">=3.13\"/requires-python = \">=3.12\"/' pyproject.toml && \
    python -m build --wheel --no-isolation && \
    cp dist/*.whl /workspace/ \
"
```

**Expected time:** 8–18 minutes depending on CPU core count (Ninja parallelizes across all cores).

**Output:** A file like `nelux-0.8.9-cp312-cp312-linux_x86_64.whl` in your current directory.

### What the patches do

| Patch | Reason |
|---|---|
| Copy FFmpeg headers/libs to `external/ffmpeg/` | NeLux CMakeLists.txt expects vendored FFmpeg at this path (Windows-centric build) |
| `cp -rL` and `cp -L` flags | Dereferences symlinks so actual files are copied (scikit-build-core copies source to temp dir, breaking symlinks) |
| `find_package(Python3 3.13` → `3.12` | NeLux defaults to Python 3.13; we build with 3.12 |
| `SPDLOG_FMT_EXTERNAL` → `SPDLOG_FMT_BUNDLED` | Matches how spdlog was compiled (bundled fmt avoids PyTorch fmt v12 conflict) |
| Remove `FMT_HEADER_ONLY` | Prevents conflicting fmt definitions |
| `requires-python >= 3.13` → `>= 3.12` | Allows pip to install the wheel on Python 3.12 |

---

## Step 3: Test the Wheel

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  nelux-builder bash -c " \
    /opt/build-env/bin/pip install numpy && \
    /opt/build-env/bin/pip install --force-reinstall /workspace/nelux-0.8.9-cp312-cp312-linux_x86_64.whl && \
    /opt/build-env/bin/python -c 'import torch; import nelux; print(nelux)' \
"
```

**Expected output:** `<module 'nelux' from '...'>` with no errors.

---

## Troubleshooting

### `cmake: not found`
The `cmake` package was missing from the apt install list. Ensure `cmake` is in the `apt-get install` line in the Dockerfile. Rebuild the image.

### `cp: cannot stat '/usr/include/libavcodec'`
FFmpeg headers are under the architecture-specific path on Ubuntu. Use `/usr/include/x86_64-linux-gnu/libavcodec` instead of `/usr/include/libavcodec`.

### `'basic_runtime' is not a member of 'fmt'` (spdlog/fmt conflict)
Ubuntu's system spdlog was compiled against fmt 8.x, but PyTorch bundles fmt v12. The fix is to build spdlog from source with `-DSPDLOG_FMT_EXTERNAL=OFF` (bundled fmt) and patch NeLux's CMakeLists to use `SPDLOG_FMT_BUNDLED` instead of `SPDLOG_FMT_EXTERNAL`.

### `'AVCodecContext' has no member named 'ch_layout'`
The FFmpeg version is too old. NeLux requires FFmpeg 5.1+ for the new channel layout API (`ch_layout` / `AVChannelLayout`). Ubuntu 22.04 ships FFmpeg 4.4. Solution: use Ubuntu 24.04 base image (FFmpeg 6.1).

### `ERROR: Package 'nelux' requires a different Python: 3.12.3 not in '>=3.13'`
The `pyproject.toml` metadata still specifies `>=3.13`. Add this sed to the build command:
```bash
sed -i 's/requires-python = ">=3.13"/requires-python = ">=3.12"/' pyproject.toml
```

### `ImportError: PyTorch must be imported before Nelux`
This is expected behavior, not an error. Always `import torch` before `import nelux`.

### `ModuleNotFoundError: No module named 'numpy'`
Install numpy in your environment. NeLux requires it at runtime.

### `nvidia/cuda:12.4.1-devel-ubuntu24.04: not found`
Not all CUDA versions have Ubuntu 24.04 tags. Use `nvidia/cuda:12.6.3-devel-ubuntu24.04` or check [NVIDIA's Docker Hub](https://hub.docker.com/r/nvidia/cuda/tags) for available tags.

### `DEBIAN_FRONTEND` / timezone prompt hangs
Ensure `ENV DEBIAN_FRONTEND=noninteractive` is set in the Dockerfile before any `apt-get install` commands.

### `ModuleNotFoundError: No module named 'distutils'`
Python 3.12 removed distutils (PEP 632). Ensure `setuptools` is installed in the build venv — it provides the distutils compatibility shim.

---

## Rebuilding for a New NeLux Version

The Docker base image layers are fully cached. To rebuild after a new NeLux release, just re-run Step 2. If NeLux changes its dependencies or Python version requirements, check the `CMakeLists.txt` and `pyproject.toml` in the new version and adjust the `sed` patches accordingly.

## App Container Integration

The wheel links against FFmpeg 6.1 shared libraries from Ubuntu 24.04. Your app container must have matching FFmpeg libs at runtime. Options:

1. **Switch app base image** to Ubuntu 24.04-based (e.g. build your own from `ubuntu:24.04`)
2. **Multi-stage copy** — copy FFmpeg `.so` files from the builder image into your app image
3. **Install FFmpeg 6.x** from a backport or PPA in your app container

The wheel file itself can be vendored into your project (e.g. `wheels/` directory) and installed with:
```bash
pip install wheels/nelux-0.8.9-cp312-cp312-linux_x86_64.whl
```
