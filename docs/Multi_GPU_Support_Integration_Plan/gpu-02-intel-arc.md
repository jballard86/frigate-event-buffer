# Sub-plan 2: Intel Arc GPU path (pybind11 + oneVPL / QSV)

**Depends on:** [gpu-01-nvidia-refactor-and-prep.md](./gpu-01-nvidia-refactor-and-prep.md) merged (protocols, registry, injection, `GPU_DEVICE_INDEX`, **`DecoderContextProto`**).  
**Scope:** Full **decode → BCHW tensor → YOLO → compilation → GIF** on **Intel Arc** (and compatible Intel GPUs), preserving **zero-copy decode** parity with PyNvVideoCodec via a **native C++ extension**, not TorchCodec/torchlib-xpu as the decode bridge.

---

## 1. Objectives

1. `GPU_VENDOR=intel` selects the Intel backend in the registry.
2. **Decode:** A **pybind11** extension links **FFmpeg** (hw decode path) with **Intel oneVPL / QSV** (and related surfaces), maps frames into **libtorch** tensors on **XPU** (or the agreed Intel GPU tensor path), and returns **`torch.Tensor`** to Python with the same **`DecoderContextProto`** contract as NVIDIA (BCHW `uint8` RGB, index/time/seek APIs).
3. **Python:** `services/gpu_backends/intel/decoder.py` **imports the compiled `.so`** (extension module) like any other Python package artifact and wraps it to satisfy **`DecoderContextProto`** — the orchestrator and `video.py` / `multi_clip_extractor.py` / `video_compilation.py` see **no difference** from other backends except registry selection.
4. **Compilation:** FFmpeg **`h264_qsv`** / **`hevc_qsv`** argv builders in `intel/ffmpeg_encode.py` (unchanged intent).
5. **GIF:** Intel-specific FFmpeg graphs in `intel/gif_ffmpeg.py`.
6. **YOLO:** **IPEX + `xpu`** (or current Intel-recommended stack) for inference; decoded tensors must be **consumable** by that stack without an extra host round-trip.
7. **Deployment:** **Multi-stage Docker** (see §6); runtime container is **individual** per deployment; host passes **device nodes** (e.g. `/dev/dri/renderD128`) via **`--device`** / compose `devices:` — same operational pattern as NVIDIA GPU containers today.

---

## 2. Proposed filesystem layout

**Python (under `src/frigate_buffer/services/gpu_backends/`):**

```
├── intel/
│   ├── __init__.py          # build_intel_backend() for registry
│   ├── decoder.py           # Thin: import native extension, adapt to DecoderContextProto
│   ├── runtime.py           # xpu empty_cache, memory, logging
│   ├── ffmpeg_encode.py     # h264_qsv / hevc_qsv argv builders
│   └── gif_ffmpeg.py        # Intel-specific hwaccel + filters
```

**Native (repo root or agreed `native/` tree — update MAP when implementing):**

```
native/
└── intel_decode/            # name TBD
    ├── CMakeLists.txt
    ├── pybind11 module target → produces e.g. frigate_intel_decode*.so
    └── src/                 # FFmpeg hw context, oneVPL/QSV, libtorch tensor build, DLPack
```

The **wheel/sdist** or **Docker copy step** must place the `.so` where `intel.decoder` can `import` it (e.g. same package namespace via `scikit-build-core` / setuptools extension, or documented `PYTHONPATH` — decide in implementation).

**Registry:** `elif vendor == "intel": return build_intel_backend()`.

---

## 3. Primary decode strategy (C++ bridge)

### 3.1 Native pipeline (conceptual)

1. Open input with **libavformat**; configure **hwaccel** appropriate for **QSV / VAAPI / oneVPL** per target OS (Linux container is the first-class target).
2. Decode to **hardware frames** (`AVHWFramesContext`); avoid readback to CPU for the hot path.
3. Convert to **RGB planar / BCHW** layout on GPU if needed (FFmpeg filters in-graph, or custom CUDA-less Intel paths — spike in implementation).
4. Wrap or copy into **libtorch** `at::Tensor` on **XPU** with **ABI compatibility** to the **same** `libtorch` version as the installed **`torch`** wheel.
5. Expose to Python via **pybind11** (return `torch::Tensor` or capsule + **DLPack** as PyNv-style).

### 3.2 Python adapter (`intel/decoder.py`)

- Implement **`create_decoder(...)`** context manager and **`DecoderContextProto`** by calling into the extension.
- **No** business logic duplication in Python beyond argument marshalling and error translation (log **`GPU_DECODE_INIT_FAILURE_PREFIX`** + `[intel]`).

### 3.3 Locking and logging

- Same **`GPU_LOCK`** as other vendors for decoder lifecycle and `get_frames`.
- Document **driver** and **FFmpeg** minimum versions in INSTALL.

---

## 4. Dependencies

### 4.1 Build-time (Stage 1 Docker / dev machines)

- **CMake**, **build-essential**, **C++17** (or project standard).
- **FFmpeg development headers** (`libavcodec-dev`, `libavformat-dev`, `libavutil-dev`, `libswscale-dev` or distro equivalents).
- **Intel oneVPL / media SDK** headers and **libtorch** (prebuilt) matching **torch** version.
- **pybind11** (FetchContent or system package).

### 4.2 Runtime (Stage 2 Docker / host)

- **Intel GPU userspace drivers** / compute runtime as today for IPEX.
- **No** compiler toolchain in the **final** image if multi-stage is done correctly — only **`.so`**, Python app, **torch+IPEX**, and **FFmpeg shared libs** actually needed at runtime (exact set to be minimized in implementation).

### 4.3 Python packages

- **`torch`**, **Intel Extension for PyTorch**, **Ultralytics** — pin a **tested matrix** with the **native** extension ABI.

---

## 5. Compilation and deployment (Docker)

### 5.1 Multi-stage build (required)

| Stage | Purpose | Contents |
|-------|---------|----------|
| **1 — builder** | Compile pybind11 extension | `build-essential`, `cmake`, `ninja` (optional), FFmpeg **-dev** packages, **libtorch** tarball or dev image, Intel **oneVPL** (or equivalent) dev files, Python headers for the target version, **pybind11**. Runs **CMake** → produces **`frigate_intel_decode*.so`** (name TBD). |
| **2 — runtime** | Run the app | **Base:** slim image with **Intel runtime** drivers/libs, **FFmpeg** runtime `.so` **only** (no `-dev`), **`torch` + IPEX** from pip, **application code**, and **copied artifact** from stage 1: **only the compiled extension `.so`** (and any minimal non-dev FFmpeg/vendor `.so` the extension links to). **No** `g++`, **no** `cmake** in final image. |

### 5.2 Host deployment (containers)

- Service runs as **its own container** (same as current design).
- Pass through GPU access with **`--device /dev/dri/renderD128`** (or the correct render node for the Arc GPU) and any **group adds** (`video`, `render`) your distro requires — **analogous** to NVIDIA’s `--gpus all` / device capability model, but for **DRI**.
- Document **compose** snippet mirroring today’s pattern: **`devices`**, **`group_add`**, optional **`privileged: false`** (prefer minimal).

### 5.3 Non-Docker installs

- Document **from-source** build of the extension on a dev machine with matching **libtorch** and **FFmpeg** dev packages.

---

## 6. FFmpeg: compilation and GIF (Python-side)

- **`intel/ffmpeg_encode.py`:** rawvideo stdin → **`h264_qsv`** / **`hevc_qsv`**; no **libx264** if MAP GPU-only encode still applies.
- **`intel/gif_ffmpeg.py`:** QSV/VAAPI-friendly decode/scale where possible; palette steps may stay CPU as on NVIDIA.

---

## 7. Configuration schema

- `GPU_VENDOR: "intel"`.
- `GPU_DEVICE_INDEX` / device selection passed into native decoder where applicable.
- `DETECTION_DEVICE`: default **`xpu:0`** (or auto from config merge).
- Optional nested `intel:` for QSV preset, quality, etc.

---

## 8. Testing

| Test type | Plan |
|-----------|------|
| Unit | **Mock** the native module in `intel/decoder.py` for CI; contract tests for **`DecoderContextProto`**. |
| Native | Optional **hardware** job with built `.so`. |
| FFmpeg | Argv snapshot tests for QSV encode/GIF. |

---

## 9. Documentation updates

- [docs/INSTALL.md](../INSTALL.md): **Build** vs **runtime** deps; **Docker multi-stage**; **`--device /dev/dri/...`**.
- [MAP.md](../../MAP.md): Intel decode = **pybind11 + FFmpeg + oneVPL/QSV**.
- [docs/maps/PROCESSING.md](../maps/PROCESSING.md): Native extension + **DecoderContextProto**.

---

## 10. Risks

| Risk | Mitigation |
|------|------------|
| libtorch ABI skew vs pip `torch` | Pin versions; build extension in CI against **same** torch as runtime |
| RGB surface format mismatches | Spike early on Arc hardware |
| IPEX + native tensor device edge cases | Integration tests on real XPU |

---

## 11. Acceptance criteria

- [ ] `GPU_VENDOR=intel` E2E with **native decode** `.so` on reference Arc Linux setup.
- [ ] Final Docker image **does not** contain compiler toolchain.
- [ ] **`DecoderContextProto`** satisfied; no PyNvVideoCodec import on Intel path.
- [ ] INSTALL documents **multi-stage** build and **device** passthrough.

---

## 12. Implementation order (suggested)

1. **Spike:** minimal pybind11 module: open file, decode one frame to tensor (even if not full API yet).
2. Flesh out **seek / batch indices** to match PyNv semantics.
3. **`intel/decoder.py`** + registry + **GPU_LOCK**.
4. FFmpeg QSV encode + GIF helpers; IPEX YOLO validation.
5. **Dockerfile.intel** multi-stage + compose docs.
6. Tests + MAP/PROCESSING.
7. **Phase 7 (repo):** CI **`docker run`** smoke **`--strict`** after Intel image build;
   optional **`multi_cam.intel`** QSV compilation preset/quality + env overrides;
   **`FfmpegCompilationEncodeProto.config`** plumbed from **`generate_compilation_video`**.
8. **Phase 8 (hardware):** Runbook **[intel-arc-hardware-smoke.md](./intel-arc-hardware-smoke.md)**,
   **`scripts/run_intel_arc_docker_smoke.sh`**, smoke **`--vainfo` / `--strict-dri`**,
   runtime **`vainfo`** package in **`Dockerfile.intel`**, optional self-hosted workflow
   **`.github/workflows/intel_arc_smoke.yml`**.

---

*End of sub-plan 2*
