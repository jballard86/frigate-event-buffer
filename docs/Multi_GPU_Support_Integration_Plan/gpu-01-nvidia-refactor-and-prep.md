# Sub-plan 1: NVIDIA path refactor and prep for multi-vendor GPU

**Parent:** Multi-vendor GPU support — master plan file: `.cursor/plans/amd_intel_gpu_support_d5add60b.plan.md` (Cursor workspace; primary copy: `docs/Multi_GPU_Support_Integration_Plan/gpu-00-primary-multi-vendor-gpu-plan.md`).  
**Scope:** Preserve **byte-for-byte behavior** for existing NVIDIA deployments while introducing **abstractions, filesystem layout, and config** that Intel and AMD sub-plans plug into.  
**Out of scope for this sub-plan:** Implementing Intel or AMD **decode/encode bodies** (sub-plans 2–3: **C++ pybind11** extensions + thin Python wrappers). Sub-plan 1 only defines **`DecoderContextProto`** and registry hooks those backends will satisfy.

---

## 1. Objectives

1. **No regression** on NVIDIA: PyNvVideoCodec, NVDEC, `h264_nvenc`, CUDA GIF path, YOLO on CUDA — same semantics after refactor.
2. **Single backend registry:** Config key selects vendor implementation; default remains NVIDIA.
3. **Vendor code isolation:** All NVIDIA-specific imports live under `services/gpu_backends/nvidia/` so PyNvVideoCodec is not imported from generic modules.
4. **Neutral APIs:** `video.py`, `video_compilation.py`, `multi_clip_extractor.py`, and tests depend on **protocols + registry**, not on `torch.cuda` / `h264_nvenc` literals scattered everywhere.
5. **Documentation:** [MAP.md](../../MAP.md), [docs/maps/PROCESSING.md](../maps/PROCESSING.md), [docs/INSTALL.md](../INSTALL.md), [config.example.yaml](../../examples/config.example.yaml) updated to describe the new layout and `GPU_VENDOR` / `GPU_DEVICE_INDEX`.

---

## 2. Proposed filesystem and package layout

Add a **gpu_backends** package under services (per root MAP file-placement rules).

```
src/frigate_buffer/services/
├── gpu_backends/
│   ├── __init__.py              # Re-export get_active_backend() for convenience (optional)
│   ├── protocols.py             # typing.Protocol: DecodeContextProto, GpuRuntimeProto, FfmpegEncodeProto
│   ├── registry.py              # load_backend(config) -> GpuBackend bundle
│   ├── types.py                 # GpuBackend dataclass or NamedTuple: decoder_factory, runtime, ffmpeg_encode, gif_builder
│   └── nvidia/
│       ├── __init__.py          # register / export build_nvidia_backend()
│       ├── decoder.py           # PyNvVideoCodec: DecoderContext + create_decoder contextmanager
│       ├── runtime.py           # log_gpu_status (nvidia-smi), empty_cache, memory_summary, default_detection_device
│       ├── ffmpeg_encode.py     # h264_nvenc argv builder + error messages (moved from video_compilation patterns)
│       └── gif_ffmpeg.py        # cuda hwaccel + scale_cuda filter graph for generate_gif_from_clip
├── gpu_decoder.py               # DEPRECATED thin facade: delegate to registry → nvidia.decoder (remove in later release or keep one release)
```

**Intel / AMD (later sub-plans):** `gpu_backends/intel/decoder.py` and `gpu_backends/amd/decoder.py` **import** compiled **pybind11** extension modules (`.so`) built from a separate **`native/`** tree (see gpu-02 / gpu-03). To Python, the extension is **just another module**; it must still expose behavior matching **`DecoderContextProto`**. Update **MAP.md** when `native/` is added.

**Lock placement:** Keep **one app-wide lock** for decode serialization. Options (pick one in implementation):

- **A (minimal move):** `GPU_LOCK` remains in `video.py`, documented as **decode serialization for all vendors**; `gpu_backends` modules do not own the lock.
- **B (clearer):** Move `GPU_LOCK` to `services/gpu_runtime.py` or `gpu_backends/lock.py` and import from there in `video.py`, `multi_clip_extractor.py`, `video_compilation.py`.

Recommendation: **B** if touching imports anyway — avoids implying the lock is “video-only.”

**Backward compatibility:**

- `from frigate_buffer.services.gpu_decoder import create_decoder, DecoderContext` — either keep **working** via re-export from active backend’s decoder module when `GPU_VENDOR=nvidia`, or deprecate with warning pointing to `gpu_backends.registry`. Prefer **re-export from nvidia.decoder** through a thin `gpu_decoder.py` shim for **one major version** to reduce churn for forks.

---

## 3. Protocols and registry contract

### 3.1 `protocols.py` (conceptual)

Define **narrow** protocols so Intel/AMD can implement without inheriting heavy bases:

| Protocol | Responsibility |
|----------|----------------|
| `DecoderContextProto` | `__len__`, `frame_count`, `get_frames`, `get_frame_at_index`, `get_batch_frames`, `seek_to_index`, `get_index_from_time_in_seconds` (match current [gpu_decoder.py](../../src/frigate_buffer/services/gpu_decoder.py) surface) |
| `DecoderFactoryProto` | Context manager: `(clip_path, device_index) -> Iterator[DecoderContextProto]` |
| `GpuRuntimeProto` | `log_gpu_status()`, `empty_cache()`, `memory_summary() -> str | None`, `tensor_device_for_decode() -> str` (e.g. `"cuda:0"`), `default_detection_device(config) -> str | None` |
| `FfmpegCompilationEncodeProto` | Build argv list + human-readable error context for compilation pipeline (today: h264_nvenc) |
| `GifFfmpegProto` | Args + filter_complex string for GIF generation |

### 3.2 `registry.py`

- `get_gpu_backend(config: dict) -> GpuBackend` — cached per process after first read.
- Parse `GPU_VENDOR` (default `nvidia`). Valid values at end of sub-plan 1: **`nvidia` only**; `intel` / `amd` register **stub** that raises `NotImplementedError` with pointer to sub-plans, **or** omit until sub-plan 2/3 lands (prefer **no stub** — only register implemented vendors to avoid misconfiguration).
- Normalize legacy keys: **`CUDA_DEVICE_INDEX` → `GPU_DEVICE_INDEX`** with **fallback** read in config merge (document deprecation).

### 3.3 Config schema ([config.py](../../src/frigate_buffer/config.py))

Add to Voluptuous schema (exact names to match YAML style of project):

- `GPU_VENDOR`: string, optional, default `nvidia`.
- `GPU_DEVICE_INDEX`: int, optional, default `0` (replaces conceptual use of `CUDA_DEVICE_INDEX`; keep reading `CUDA_DEVICE_INDEX` if `GPU_DEVICE_INDEX` absent).

Optional later (sub-plans): vendor-specific nested dict — not required in sub-plan 1.

---

## 4. Refactoring steps (ordered)

### Phase A — Add packages without wiring (NVIDIA copy-paste)

1. Create `gpu_backends/protocols.py`, `types.py`, `registry.py` (minimal).
2. Move **body** of [gpu_decoder.py](../../src/frigate_buffer/services/gpu_decoder.py) into `gpu_backends/nvidia/decoder.py`; adjust empty-tensor device to use `torch.device` from a small helper (`runtime.tensor_device_for_decode()`).
3. Extract from [video_compilation.py](../../src/frigate_buffer/services/video_compilation.py): NVENC-specific FFmpeg argument building into `nvidia/ffmpeg_encode.py` (functions only; no behavior change).
4. Extract GIF FFmpeg command pieces from [video.py](../../src/frigate_buffer/services/video.py) into `nvidia/gif_ffmpeg.py`.

### Phase B — Wire registry

5. `registry.load_backend(config)` returns `GpuBackend` with NVIDIA implementations.
6. Replace direct `from gpu_decoder import create_decoder` in `video.py`, `multi_clip_extractor.py`, `video_compilation.py` with `get_gpu_backend(config).decoder.create_decoder` (or inject backend at `VideoService` / orchestrator init to avoid passing `config` into every call — **prefer injection** from orchestrator: `video_service` holds `self._gpu = backend`).

### Phase C — Centralize CUDA calls

7. Replace `torch.cuda.empty_cache()` / `memory_summary()` / `is_available()` in `video.py`, `multi_clip_extractor.py`, `video_compilation.py`, [quick_title_service.py](../../src/frigate_buffer/services/quick_title_service.py) with `backend.runtime.*` (NVIDIA impl delegates to `torch.cuda`).

### Phase D — Shim and tests

8. `gpu_decoder.py` re-exports `create_decoder`, `DecoderContext` from NVIDIA module **or** from registry when only NVIDIA exists.
9. Update [tests/test_gpu_decoder.py](../../tests/test_gpu_decoder.py): mock **nvidia.decoder** or registry; add `tests/test_gpu_backends_registry.py` for config parsing and default vendor.
10. Update all tests that patch `gpu_decoder.create_decoder` to patch the **stable** entrypoint (registry or `nvidia.decoder`).

### Phase E — Docs and map

11. Update MAP.md §Video & AI pipeline, PROCESSING.md registry section, INSTALL.md (GPU_VENDOR, GPU_DEVICE_INDEX).
12. `examples/config.example.yaml` comments for new keys.

---

## 5. Orchestrator / service construction

Today `VideoService` is constructed with config elsewhere. **Inject** `GpuBackend` (or a lazy resolver) at construction time:

- In [orchestrator.py](../../src/frigate_buffer/orchestrator.py) (or wherever `VideoService` is built), call `get_gpu_backend(config)` once and pass to `VideoService(__init__(..., gpu_backend=...))`.
- Avoid **global** singleton for backend unless already consistent with project patterns; if singleton is simpler for phase 1, document thread-safety (Gunicorn single worker already assumed per [wsgi.py](../../src/frigate_buffer/wsgi.py)).

---

## 6. Testing strategy

| Area | Action |
|------|--------|
| Unit | Registry: default vendor, legacy `CUDA_DEVICE_INDEX` fallback |
| Unit | NVIDIA decoder: existing mock tests, paths updated |
| Integration | Optional: single GPU smoke script under `scripts/` (existing bench patterns) |
| Regression | Full `pytest tests/` — no skipped NVIDIA-specific tests without GPU |

---

## 7. Docker / dependencies

- **Sub-plan 1:** No Dockerfile base image change; `PyNvVideoCodec` remains in [pyproject.toml](../../pyproject.toml).
- Optional: add **`[project.optional-dependencies]`** `gpu-nvidia` mirroring core deps for clarity in future split images (not required for success).

---

## 8. Acceptance criteria

- [ ] With `GPU_VENDOR` unset or `nvidia`, all existing tests pass; manual spot-check: sidecar generation, compilation, GIF on NVIDIA box.
- [ ] No `import PyNvVideoCodec` outside `gpu_backends/nvidia/`.
- [ ] `MAP.md` and `docs/maps/PROCESSING.md` describe `gpu_backends/` and registry.
- [ ] Config validates `GPU_VENDOR` and `GPU_DEVICE_INDEX`; `CUDA_DEVICE_INDEX` still works with documented deprecation.

---

## 9. Follow-on (handoff to sub-plans 2–4)

- Registry gains `intel` and `amd` when **native decode `.so`** + thin `decoder.py` wrappers land ([gpu-02](./gpu-02-intel-arc.md), [gpu-03](./gpu-03-amd-rocm.md)); both use **FFmpeg + vendor HW + libtorch + pybind11**, not Python-only decode bridges.
- Optional **`cpu`** vendor per [gpu-04-cpu-only-path.md](./gpu-04-cpu-only-path.md) (waives MAP GPU-only rules; dev/CI).
- `protocols.py` may need **optional** methods if a vendor cannot support sequential `get_batch_frames` identically — document extension process in PROCESSING.md.
- Constants: consider vendor-neutral rename for `NVDEC_INIT_FAILURE_PREFIX` → `GPU_DECODE_INIT_FAILURE_PREFIX` with NVIDIA-specific log suffix, or keep string for log grep compatibility (decision in implementation).

---

*End of sub-plan 1*
