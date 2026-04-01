# GPU vendor zero-copy pipeline scorecard

This document scores how closely **NVIDIA**, **AMD**, and **Intel** decode-to-tensor
paths match the project’s stated rules: **MAP.md** §1 (zero-copy GPU decode, no
CPU decode in the core path, BCHW tensors, `crop_utils`), **docs/maps/PROCESSING.md**,
and native extension READMEs. Scores are **1–100** for alignment with those rules
as **verified in code**, not theoretical driver behavior.

**Scoring rubric (summary):**

| Band | Meaning |
|------|--------|
| 90–100 | HW decode → GPU-resident BCHW with no systematic host readback; policy matches MAP prohibitions. |
| 70–89 | Strong HW path; minor extra GPU copies or documentation/telemetry gaps. |
| 50–69 | HW decode exists but pixels routinely cross PCIe to CPU (or SW decode allowed) before tensors. |
| <50 | Frequent CPU staging, optional/disabled zero-copy, or policy conflict with MAP. |

**Out of scope for scores:** GIF/ffprobe subprocess boundaries, compilation
**rawvideo pipe** (explicitly allowed as a boundary in MAP), and YOLO/JPEG export.
This is strictly about **decode → tensor device residency**.

---

## NVIDIA (PyNvVideoCodec)

**Documentation:** `MAP.md`, `docs/maps/PROCESSING.md`,
`src/frigate_buffer/services/gpu_backends/nvidia/decoder.py`.

**What the code does**

- NVDEC decode uses `SimpleDecoder(..., use_device_memory=True)` and returns
  DLPack-capable frames.
- Python converts via `torch.from_dlpack` and then does a **GPU-side `.clone()`**
  to make the tensor lifetime independent of the decoder context.

```44:76:src/frigate_buffer/services/gpu_backends/nvidia/decoder.py
    def get_frames(self, indices: list[int]) -> Any:
        # ...
        raw_frames = self._decoder.get_batch_frames_by_index(indices)
        tensors = [torch.from_dlpack(f) for f in raw_frames]
        batch = torch.stack(tensors, dim=0).clone()
        return batch
```

**Score: 90 / 100**

- **Strengths:** No CPU decode fallback; fails closed on init; tensors are device
  resident.
- **Deductions:** `.clone()` is an intentional extra device copy (lifetime safety
  over strictest possible “no extra copies” semantics).

---

## AMD (`frigate_amd_decode`)

**Documentation:** `native/amd_decode/README.md`, `docs/maps/PROCESSING.md`.

**What the code does (current)**

- **VAAPI + HIP zero-copy only:** VAAPI hw frames are mapped to DRM PRIME; DMA-BUF
  fds are imported via HIP; NV12→RGB runs on-device; output is **ROCm `cuda:N`**
  BCHW `uint8`.
- **No CPU fallbacks:** VAAPI failures, DRM PRIME map failures, HIP import failures,
  and kernel failures **raise** (no SW decode retry, no host readback).

```409:478:native/amd_decode/src/session.cpp
torch::Tensor AmdDecoderSession::avframe_to_bchw_rgb_(AVFrame* src) {
#if defined(AMD_DECODE_WITH_HIP) && AMD_DECODE_WITH_HIP
  if (!zero_copy_ready_ || !using_vaapi_) {
    throw std::runtime_error("AMD decode requires HIP zero-copy; session not ready");
  }
  if (src->format != AV_PIX_FMT_VAAPI || src->hw_frames_ctx == nullptr) {
    throw std::runtime_error("AMD decode expected VAAPI hw frames for zero-copy path");
  }
  return avframe_vaapi_drm_to_cuda_(src);
#else
  (void)src;
  throw std::runtime_error("AMD decode built without HIP; zero-copy path unavailable");
#endif
}
```

**Score: 92 / 100**

- **Strengths:** Fail-closed like NVIDIA; no host readback path; decoded tensors
  are device-resident BCHW.
- **Deductions:** Still does a colorspace/format conversion kernel (NV12→RGB) and
  currently `uses_zero_copy_decode()` is “true after at least one frame used”.

---

## Intel (`frigate_intel_decode`)

**Documentation:** `native/intel_decode/README.md`, `MAP.md` Intel note,
`docs/maps/PROCESSING.md`.

**What the code does (current)**

- **QSV-only** decode (`h264_qsv` / `hevc_qsv`)—no software libavcodec fallback.
- **XPU zero-copy only:** QSV hw frames are mapped to DRM PRIME, imported via
  DMA-BUF, and converted NV12→RGB on-device to return **XPU** BCHW `uint8`.
- There is **no** host readback + swscale fallback path; failure to map/import is
  treated as a decode failure.

**Score: 92 / 100**

- **Strengths:** Fail-closed like NVIDIA; no host readback path; `uses_zero_copy_decode()`
  is “true only when actually used”.
- **Deductions:** Requires `icpx` + XPU torch stack; availability is hardware/driver
  dependent, so decode may fail more often than the removed legacy CPU-RGB path.

---

## Cross-cutting improvements (all three)

1. **Observable decode contract:** log once per decoder session a small enum like
   `device_bchw` vs `cpu_staging` vs `software_decode` so policy compliance is
   visible in logs/metrics.
2. **Tests:** hardware smoke should assert `tensor.device` type and
   `uses_zero_copy_decode()` on a golden clip for AMD ROCm and Intel Arc.
3. **NVIDIA clone:** consider an opt-in “no clone” decode API for advanced callers
   that keep decoder contexts alive (reduce VRAM traffic at the cost of safety).

---

## Source index

| Vendor | Native | Python adapter |
|--------|--------|----------------|
| NVIDIA | N/A (PyNvVideoCodec) | `src/frigate_buffer/services/gpu_backends/nvidia/decoder.py` |
| AMD | `native/amd_decode/src/session.cpp` | `src/frigate_buffer/services/gpu_backends/amd/decoder.py` |
| Intel | `native/intel_decode/src/session.cpp`, `xpu_zerocopy.cpp` | `src/frigate_buffer/services/gpu_backends/intel/decoder.py` |

