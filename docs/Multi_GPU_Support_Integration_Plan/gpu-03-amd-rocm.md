# Sub-plan 3: AMD ROCm GPU path (pybind11 + AMF / VAAPI)

**Depends on:** [gpu-01-nvidia-refactor-and-prep.md](./gpu-01-nvidia-refactor-and-prep.md) merged (**`DecoderContextProto`**, registry, injection).  
**Scope:** **Hardware decode → BCHW tensor → YOLO → compilation → GIF** on **AMD** GPUs under **ROCm**, with **zero-copy decode** parity with PyNvVideoCodec via a **C++ pybind11 extension** — **not** TorchCodec ROCm, **not** a Python “copy to host then `tensor.to(device)`” fallback for the GPU backend.

---

## 1. Objectives

1. `GPU_VENDOR=amd` selects the AMD backend in the registry.
2. **Decode:** **pybind11** extension linking **FFmpeg** (`libavcodec` hwaccel) with **VAAPI** (Linux) and/or **AMF** where applicable, mapping hardware frames into **libtorch (ROCm build)** tensors and exposing **`torch.Tensor`** to Python with **`DecoderContextProto`** (BCHW `uint8` RGB, index/time/seek).
3. **Python:** `services/gpu_backends/amd/decoder.py` **imports the compiled `.so`** and wraps the native API — identical abstraction role to Intel’s `decoder.py` and NVIDIA’s PyNv wrapper.
4. **Inference:** **ROCm PyTorch** + **Ultralytics**; tensors from the extension must be valid on the ROCm device expected by `DETECTION_DEVICE`.
5. **Compilation:** **`h264_amf`** / **`hevc_amf`** via `amd/ffmpeg_encode.py`.
6. **GIF:** **`amd/gif_ffmpeg.py`** — VAAPI/AMF-friendly graphs; Linux-first.
7. **Deployment:** **Multi-stage Docker** (§5); **runtime container** only ships the **`.so`** and slim deps; host uses **`--device`** for **`/dev/kfd`**, **`/dev/dri/*`** (and any **render** nodes) exactly as standard ROCm container practice — same **individual-container** model as today.

---

## 2. Proposed filesystem layout

**Python:**

```
src/frigate_buffer/services/gpu_backends/
├── amd/
│   ├── __init__.py              # build_amd_backend()
│   ├── decoder.py               # Thin: import native extension → DecoderContextProto
│   ├── runtime.py               # ROCm empty_cache, logging
│   ├── ffmpeg_encode.py         # h264_amf argv builders
│   └── gif_ffmpeg.py            # AMF/VAAPI GIF pipeline
```

**Native:**

```
native/
└── amd_decode/                  # name TBD
    ├── CMakeLists.txt
    └── src/                     # FFmpeg hw (VAAPI/AMF), ROCm-importable memory, libtorch ROCm, pybind11
```

**Removed from plan (superseded by pivot):** separate **`decoder_torchcodec.py`**, **`decoder_copy_path.py`**, and multi-strategy **`AMD_DECODE_STRATEGY`** — the **only** GPU decode implementation for AMD in this plan is the **native extension**.

---

## 3. Primary decode strategy (C++ bridge)

### 3.1 Native pipeline (conceptual)

1. **libavformat** / **libavcodec** with **hwaccel** (**VAAPI** on Linux for many AMD setups; **AMF** where build and OS support it).
2. Decode to **hardware frames**; avoid CPU readback on the hot path.
3. **Interop** from FFmpeg hardware frames to **ROCm-accessible** memory, then **`at::Tensor`** (ROCm) with layout matching **BCHW uint8 RGB** (convert on GPU if source is NV12, etc.).
4. **pybind11** exports to Python; support **DLPack** if the tensor type supports it for downstream **crop_utils** / YOLO.

### 3.2 Python adapter (`amd/decoder.py`)

- Implements **`create_decoder`** + **`DecoderContextProto`** against the extension.
- **Single** code path — no runtime switch between “copy” and “native” for production GPU mode.

### 3.3 Locking

- **`GPU_LOCK`** for decoder + frame pulls, consistent with gpu-01.

---

## 4. Dependencies

### 4.1 Build-time (Docker Stage 1 / dev)

- **CMake**, **build-essential**, FFmpeg **development** headers/libs.
- **ROCm**-compatible **libtorch** (prebuilt) matching the **ROCm `torch`** wheel used at runtime.
- **pybind11**.

### 4.2 Runtime (Docker Stage 2)

- **ROCm user-space** stack as required by **`torch`** wheel.
- **FFmpeg** runtime libraries **without** `-dev` packages.
- **Compiled `.so`** from Stage 1 only — **no** GCC/CMake in final image.

### 4.3 Python

- **ROCm `torch` + `torchvision`** (AMD index / documented URLs).
- **Ultralytics** — version matrix in INSTALL.

---

## 5. Compilation and deployment (Docker)

### 5.1 Multi-stage build (required)

| Stage | Purpose | Contents |
|-------|---------|----------|
| **1 — builder** | Compile AMD decode extension | Toolchain, **cmake**, FFmpeg **-dev**, **libtorch (ROCm)**, ROCm **device** headers if needed for interop, Python dev headers, **pybind11**. Output: **`frigate_amd_decode*.so`** (name TBD). |
| **2 — runtime** | Run frigate-buffer | ROCm-capable **base** (or minimal runtime), **`torch` ROCm** from pip, app code, **FFmpeg** runtime `.so`, **only** the built **`.so`** from stage 1. **Strip** compilers and `-dev` packages. |

### 5.2 Host deployment (containers)

- One **container** per service instance; pass **AMD GPU** devices explicitly, e.g.:
  - **`--device /dev/kfd`**
  - **`--device /dev/dri/renderD128`** (or correct render node)
  - **`--group-add video`** / **`render`** as required by the host
- Same **operational style** as current GPU deployments: **document** `docker run` and **compose** `devices:` / `group_add` for ROCm.

### 5.3 Non-Docker

- From-source build instructions with **ROCm** + **FFmpeg** dev environment.

---

## 6. FFmpeg: compilation and GIF

- **`amd/ffmpeg_encode.py`:** rawvideo → **AMF** encode; validate accepted **pixel formats**.
- **`amd/gif_ffmpeg.py`:** VAAPI-first on Linux.

---

## 7. Configuration schema

- `GPU_VENDOR: "amd"`.
- `GPU_DEVICE_INDEX` (decoder + runtime hints).
- `DETECTION_DEVICE`: ROCm default from `runtime.default_detection_device()`.
- Optional `amd:` nested block for AMF rate control, etc.
- **Remove** `AMD_DECODE_STRATEGY` multi-mode switch from planning — **one** native implementation.

---

## 8. Testing

| Test type | Plan |
|-----------|------|
| Unit | Mock native module; **`DecoderContextProto`** contract tests |
| Hardware | `@pytest.mark.amd_gpu` optional |
| CI | Default: **mocked**; no ROCm in standard CI unless self-hosted runner |

---

## 9. Documentation

- [docs/INSTALL.md](../INSTALL.md): ROCm versions, **multi-stage** build, **`/dev/kfd` + /dev/dri** compose example.
- [MAP.md](../../MAP.md): AMD decode = **pybind11 + FFmpeg + AMF/VAAPI + libtorch ROCm**; **zero-copy** goal (document any unavoidable deviation if discovered during implementation — should be exception, not planned fallback).
- [docs/maps/PROCESSING.md](../maps/PROCESSING.md): Native module + **DecoderContextProto**.

---

## 10. Risks

| Risk | Mitigation |
|------|------------|
| DMA-BUF / fd import into ROCm tensor | Early spike on target SKU + driver |
| AMF vs VAAPI matrix (Linux vs Windows) | Phase 1 **Linux** + document Windows separately |
| libtorch ROCm ABI vs pip torch | Pin and build in CI with same wheel |

---

## 11. Acceptance criteria

- [ ] `GPU_VENDOR=amd` E2E with **native** `.so` on documented Linux + AMD GPU.
- [ ] Runtime image **slim** (no build toolchain).
- [ ] **DecoderContextProto** complete; no NVIDIA imports.
- [ ] MAP/INSTALL describe **multi-stage** Docker and **device** nodes.

---

## 12. Implementation order (suggested)

1. ROCm torch + YOLO smoke (tensors only).
2. **Spike:** pybind11 + FFmpeg VAAPI → one ROCm tensor frame.
3. Full **DecoderContext** API in native code + **`amd/decoder.py`**.
4. AMF encode + GIF; registry wiring.
5. **Dockerfile.rocm** multi-stage + docs.
6. Tests + PROCESSING.

---

*End of sub-plan 3*
