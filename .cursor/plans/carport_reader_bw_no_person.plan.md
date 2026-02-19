# Plan: Carport Reader, No-Person Frames, B/W Stacking (updated)

## 1. Prevent carport reader death — fallback and verbose logging (updated)

**Goal:** Reader death during extraction should **never** happen: always **fall back** when a decoder fails, and **log verbosely** on any failure.

### 1.1 Design: fallback so reader death is avoided

- **At open (reopen loop):** For **each** camera we try NVDEC first; **on failure** we **fall back to CPU**. If **CPU open also fails**, do **not** skip immediately: **retry GPU (NVDEC)** once, then try CPU again. Only if GPU retry fails **and** CPU fails again do we skip that camera (log verbosely, do not add to `caps`).
- **Second and subsequent cameras:** To avoid **mid-stream** reader death (e.g. "read of closed file" when two NVDEC decoders run), use **CPU decode for the second camera** when config `decode_second_camera_cpu_only` is True. For index ≥ 1, if that option is set, **skip NVDEC** and open with `ffmpegcv.VideoCapture(path)` only. That way we never run two NVDEC decoders and reader death from dual-NVDEC should not occur.
- **Fallback semantics:** (1) NVDEC fails → try CPU. (2) CPU fails → **retry NVDEC** once → if NVDEC fails again, try CPU again → if CPU fails again, skip camera with verbose log. (3) For second camera with `decode_second_camera_cpu_only`, use CPU only; on CPU failure, retry GPU then CPU as above before skipping.

### 1.2 Verbose logging on every failure

- **NVDEC open failed (per camera):** Log at **WARNING** with: camera name, path, exception type and message, and "Falling back to CPU decode." Include the exact error so logs are diagnostic.
- **CPU open failed (first time):** Log at **WARNING** with: camera name, path, exception, and "Retrying GPU (NVDEC) then CPU." Then retry NVDEC once; if NVDEC retry fails, log again and try CPU again.
- **NVDEC retry failed:** Log at **WARNING** with: camera name, path, exception, and "GPU retry failed; trying CPU again."
- **CPU open failed (after GPU retry and CPU retry):** Log at **ERROR** with: camera name, path, exception type and message, and "Skipping camera for extraction (GPU and CPU decode both failed)."
- **Reader died mid-stream** (e.g. "read of closed file"): Log at **WARNING** with: camera name, path, exception, **sample_time_sec**, **frame_index** for that camera, **remaining_cameras** list, and a short hint: e.g. "Consider enabling multi_cam.decode_second_camera_cpu_only to use CPU for additional cameras and avoid NVDEC contention."
- **After reopen (success):** Keep existing INFO per-camera log (backend, path, duration_sec, fps) so we can confirm which cameras are active and which backend each uses.

### 1.3 Implementation (unchanged structure, add logging and skip)

**File:** [multi_clip_extractor.py](src/frigate_buffer/services/multi_clip_extractor.py)

- Add parameter `decode_second_camera_cpu_only: bool = False`.
- In the **reopen** loop: iterate over `clip_paths` with an index. For each camera: try NVDEC; on failure log (camera, path, error, "Falling back to CPU decode") and try CPU. If CPU fails: log (camera, path, error, "Retrying GPU then CPU"); **retry NVDEC** once; if NVDEC fails again try CPU again. If CPU fails again, log **ERROR** (camera, path, "Skipping camera for extraction (GPU and CPU decode both failed)") and do not set `caps[cam]`, set `durations[cam] = 0`. For **index ≥ 1** when `decode_second_camera_cpu_only` is True: use CPU only (no NVDEC); on CPU failure still apply the retry (retry GPU once then CPU once) before skipping.
- In the **reader death** `except _READ_ERRORS` blocks (initial read and advance loop): ensure log message includes sample_time_sec, frame_index, remaining_cameras, and the hint about `decode_second_camera_cpu_only`.

**Config:** Add `multi_cam.decode_second_camera_cpu_only` (bool, default false). Flatten in [config.py](src/frigate_buffer/config.py), pass from [ai_analyzer.py](src/frigate_buffer/services/ai_analyzer.py) into `extract_target_centric_frames`. Document in [config.example.yaml](config.example.yaml) and [MAP.md](MAP.md).

**Result:** Reader death from dual-NVDEC is avoided when the option is enabled; every open or mid-stream failure is clearly and verbosely logged; fallback (NVDEC→CPU) always happens on open failure so we don’t leave a broken reader in use.

---

## 2. Skip output when selected camera has zero person area (unchanged)

When using sidecars, before appending to `collected`, if the chosen camera’s person area at T is 0, skip the append (and optionally still update hysteresis state). Reduces "first 9 frames have no person" when only one camera is left.

---

## 3. Remove color+B/W stacking (unchanged)

In [file.py](src/frigate_buffer/managers/file.py), change `write_stitched_frame` to write only the color frame (no grayscale, no vstack). Docstring: write single color image.

---

## 4. Tests and MAP.md

- Tests: second camera CPU-only when config set; verbose logging on open failure and on reader death; skip zero-person append when sidecar in use; color-only frame write.
- MAP.md: decode_second_camera_cpu_only; fallback and verbose logging; color-only AI frame analysis.
