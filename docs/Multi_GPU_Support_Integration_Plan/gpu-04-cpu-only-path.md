# Sub-plan 4: CPU-only pipeline (explicit rule exceptions)

**Depends on:** [gpu-01-nvidia-refactor-and-prep.md](./gpu-01-nvidia-refactor-and-prep.md) merged (protocols, registry, `GpuBackend` injection).  
**Status:** **Opt-in only.** This path **does not** follow the project’s normal GPU pipeline rules in [MAP.md](../../MAP.md) and [docs/maps/PROCESSING.md](../maps/PROCESSING.md).

---

## 0. Purpose and non-goals

**Purpose**

- Allow development, CI, and **small deployments without any GPU** (e.g. laptops, constrained VMs).
- Provide a **single switch** (`GPU_VENDOR=cpu` or equivalent) so the rest of the app (MQTT, web, lifecycle) runs unchanged.

**Non-goals**

- **Not** parity with zero-copy GPU latency or throughput.
- **Not** a supported “production default”; documentation must state **performance and policy exceptions** clearly.
- **Not** removing or weakening NVIDIA/Intel/AMD paths — CPU is an **additional** backend.

---

## 1. Explicit exceptions to normal project rules

When the CPU backend is active, the following **MAP / PROCESSING prohibitions are intentionally waived** for this backend only:

| Normal rule (MAP / PROCESSING) | CPU backend behavior |
|--------------------------------|----------------------|
| No **CPU decoding fallbacks**; PyNvVideoCodec-only decode | **Allowed:** FFmpeg subprocess decode and/or **OpenCV `VideoCapture`** (or equivalent) to produce frames for the pipeline. |
| No **ffmpegcv** for decode/capture | **Policy call:** Either allow **ffmpegcv** only when `GPU_VENDOR=cpu` **or** use raw **subprocess FFmpeg** + numpy — pick one in implementation and document. |
| Compilation encode **GPU-only** (`h264_nvenc` / AMF / QSV) | **Allowed:** **`libx264`** (or other CPU encoders) for compilation MP4 output. |
| GIF: mandatory CUDA **scale_cuda** | **Allowed:** Fully CPU FFmpeg filter graph (scale + palettegen/paletteuse). |
| Production crops via **BCHW GPU tensors** only | **Pragmatic:** Accept **numpy HWC** or CPU tensors in the CPU backend; **minimize** divergence by converting to BCHW `torch` on **CPU** before shared code where cheap, or isolate CPU-specific branches behind `cpu/` modules. |
| **GPU_LOCK** serializes NVDEC | Retain a **single lock** for CPU decode if shared resources are fragile; or document why lock is relaxed (often less critical on CPU). |

**Cursor / agent rules:** Update [.cursor/rules](../../.cursor/rules) or a dedicated note (e.g. `docs/Multi_GPU_Support_Integration_Plan/CPU_BACKEND_EXCEPTIONS.md` linked from MAP) so automated agents **do not** “fix” CPU path by re-applying GPU-only mandates.

---

## 2. Proposed filesystem layout

```
src/frigate_buffer/services/gpu_backends/
├── cpu/
│   ├── __init__.py              # build_cpu_backend()
│   ├── decoder.py               # FFmpeg and/or OpenCV → DecoderContextProto (CPU tensors / numpy adapter)
│   ├── runtime.py               # no-op or logging-only: log_cpu_mode_warning(), empty_cache no-op
│   ├── ffmpeg_encode.py         # libx264 argv for compilation
│   └── gif_ffmpeg.py            # CPU scale + palette filters
```

**Registry:** `GPU_VENDOR=cpu` → `build_cpu_backend(config)`.

---

## 3. Configuration

- **`GPU_VENDOR: "cpu"`** (or separate **`FORCE_CPU_VIDEO_PIPELINE: true`** if you want to avoid overloading `GPU_VENDOR` — document one canonical choice).
- **`GPU_DEVICE_INDEX`:** ignored or used only for logging.
- **`DETECTION_DEVICE`:** default **`cpu`** when CPU vendor selected.
- Optional caps: **`CPU_DECODE_MAX_WIDTH`**, **`CPU_SIDE_CAR_FRAME_STRIDE`** (run YOLO every N frames more aggressively to save time), **`CPU_MAX_PARALLEL_DECODES=1`**.

**Startup:** Log **WARNING** once: CPU video pipeline active; GPU MAP rules waived; not recommended for production load.

---

## 4. DecoderContext compatibility

The CPU backend **must** still implement **`DecoderContextProto`** (sub-plan 1) so `video.py` / `multi_clip_extractor.py` / `video_compilation.py` stay thin:

- `get_frames(indices)` may **decode via CPU** and return **`torch.uint8` BCHW on `device="cpu"`** (or a documented numpy bridge that callers convert once at the boundary).
- Seek / time APIs must match **semantic** contract; performance of random access may be poor — document **linear decode caching** if you add it.

---

## 5. Dependencies

- **Default image:** CPU-only Docker can use a **slim base** (no CUDA, no PyNvVideoCodec). Make **`PyNvVideoCodec`** an **optional** dependency when `GPU_VENDOR=cpu` (pyproject optional extra `gpu-nvidia` vs base `cpu` install).
- **OpenCV** already present; **ffmpeg** binary required for compilation/GIF if using subprocess (align with current project assumptions).

---

## 6. Testing

| Area | Plan |
|------|------|
| Unit | Mock decoder; libx264 argv snapshots |
| CI | **Primary** path for full pytest without GPU runners |
| Contract | `DecoderContextProto` tests with **CPU** tensors |

---

## 7. Documentation updates

- [MAP.md](../../MAP.md): New subsection **“CPU backend (exceptions)”** — bullet list of waived rules + link to this file.
- [docs/maps/PROCESSING.md](../maps/PROCESSING.md): CPU row in registry; **does not** claim zero-copy.
- [docs/INSTALL.md](../INSTALL.md): “No GPU” quick start; optional dependencies.
- [examples/config.example.yaml](../../examples/config.example.yaml): commented `GPU_VENDOR: cpu` block with warning.

---

## 8. Risks

| Risk | Mitigation |
|------|------------|
| Agents refactor CPU path back to GPU-only | Dedicated doc + MAP exception section |
| Dual maintenance (GPU + CPU tensor paths) | Keep CPU logic inside `gpu_backends/cpu/` |
| User confuses CPU with production | Loud logs + INSTALL warnings |

---

## 9. Acceptance criteria

- [ ] `GPU_VENDOR=cpu` runs end-to-end on a machine **without** NVIDIA drivers (sidecar, extraction, compilation, GIF) with documented limitations.
- [ ] MAP explicitly lists **which** prohibitions do not apply to CPU backend.
- [ ] Default / documented production path remains **GPU-first** vendors.

---

## 10. Run order relative to other sub-plans

- **After sub-plan 1** (registry + injection).  
- **Independent** of sub-plans 2–3 (can implement in parallel with Intel/AMD).

---

*End of sub-plan 4*
