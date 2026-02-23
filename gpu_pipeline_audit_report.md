# GPU Pipeline Performance Audit Report

**Scope:** Full-codebase sweep of the Frigate Event Buffer video pipeline (MQTT event ingestion → summary video compile).  
**Architecture:** 100% GPU-native (PyNvVideoCodec, PyTorch/DLPack, FFmpeg h264_nvenc).  
**Date:** 2025-02-23.

---

## Executive Summary

The codebase largely adheres to the zero-copy GPU pipeline described in `MAP.md`. The audit identified **4 high-severity** issues (lock contention and redundant I/O), **5 medium-severity** issues (CPU/GPU boundary and memory), and **4 low-severity** items (minor copies and acceptable boundary cases). No code changes were applied; this document provides findings, severity, and fix recommendations only.

---

## 1. The CPU/GPU Boundary (Zero-Copy Violations)

### 1.1 [LOW] [ACCEPTED - BY DESIGN] YOLO bbox tensor → CPU/numpy inside detection path (`video.py`)

**Location:** `src/frigate_buffer/services/video.py`  
- `_run_detection_on_image`: ~298 — `xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else xyxy`  
- `_run_detection_on_batch`: ~373 — same pattern  

**Finding:** Bounding boxes are moved to CPU and converted to numpy so they can be stored in Python dicts and written to `detection.json`. This is the **serialization boundary** for the sidecar (output-only); no further frame processing is done on these values.

**Recommendation:** Accept as **boundary**: the only consumer is JSON serialization. Optionally add a one-line comment: `# Boundary: bbox to CPU/numpy for sidecar JSON only.`

---

### 1.2 [LOW] [ACCEPTED - BY DESIGN] Compilation frame → numpy at FFmpeg stdin boundary (`video_compilation.py`)

**Location:** `src/frigate_buffer/services/video_compilation.py` line 583  

```python
arr = cropped[0].permute(1, 2, 0).cpu().numpy()
frames_list.append(arr)
```

**Finding:** Each cropped frame is converted to HWC numpy and appended to a list, then later written to FFmpeg rawvideo stdin in `_encode_frames_via_ffmpeg`. This is the **encode boundary** (feeding FFmpeg); MAP.md explicitly allows “tensor → numpy for FFmpeg rawvideo stdin” at that boundary.

**Recommendation:** No change required. Optional: add a short comment that this is the encode boundary for FFmpeg stdin.

---

### 1.3 [LOW] [ACCEPTED - BY DESIGN] JPEG / file write boundaries (`ai_analyzer.py`, `file.py`, `quick_title_service.py`)

**Locations:**  
- `ai_analyzer.py` ~251, ~257: `t.cpu()` then `encode_jpeg`; `jpeg_bytes.cpu().numpy().tobytes()` for base64.  
- `file.py` ~63: `jpeg_bytes.cpu().numpy().tobytes()` for file write.  
- `quick_title_service.py` ~38: `_tensor_bchw_rgb_to_numpy_bgr` — `t.permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]]` used to pass image to `generate_quick_title` (encoding path).

**Finding:** All are at the **output boundary** (JPEG encoding for API/file or BGR for Gemini). No frame processing is done on CPU beyond encoding.

**Recommendation:** No change. Document in a single sentence in MAP or a code comment that “.cpu()/.numpy() at JPEG/base64/file and FFmpeg stdin are the only allowed boundaries.”

---

### 1.4 [LOW] [ACCEPTED - BY DESIGN] `crop_utils`: motion mask and timestamp overlay (`crop_utils.py`)

**Locations:**  
- ~263: `mask_cpu = thresh.squeeze().cpu().numpy()` then `cv2.findContours(mask_cpu, ...)`  
- ~314–317: `draw_timestamp_overlay` — tensor → HWC BGR numpy for `cv2.putText`

**Finding:** MAP.md already states that only the 1-bit motion mask is transferred to CPU for `cv2.findContours` (no GPU contour API in OpenCV). Timestamp overlay is the OpenCV drawing boundary. No production frame processing is done on full frames in numpy here.

**Recommendation:** No change. Keep the existing docstring that explains the findContours boundary.

---

### 1.5 [MEDIUM] [DONE] `ai_analyzer.py` numpy path: `cv2.resize` / `cv2.imencode` on full frame

**Location:** `src/frigate_buffer/services/ai_analyzer.py` ~216–220  

```python
if w > FRAME_MAX_WIDTH:
    scale = FRAME_MAX_WIDTH / w
    frame = cv2.resize(frame, (FRAME_MAX_WIDTH, int(h * scale)))
_, buf = cv2.imencode('.jpg', frame)
```

**Finding:** When the frame is numpy (HWC BGR), the **entire frame** is resized and encoded on CPU. If callers ever pass a frame that was already on GPU and then converted to numpy just for this path, that would be a redundant CPU round-trip. Currently the Gemini path accepts “numpy HWC BGR or torch.Tensor BCHW RGB”; tensor path uses `_frame_tensor_to_base64_url` (GPU resize + encode_jpeg). So the numpy path is used only when the frame is already numpy (e.g. legacy or non-GPU source).

**Recommendation:**  
- Ensure all production paths that send frames to the proxy use the **tensor path** when the frame originates from the GPU pipeline (e.g. `ExtractedFrame.frame`).  
- In `_frame_to_base64_url`, if the input is a tensor, always use `_frame_tensor_to_base64_url` (already the case by type check). No change required if no GPU-sourced frame is ever passed as numpy; otherwise add a single check: “if tensor, delegate to _frame_tensor_to_base64_url” and avoid converting tensor → numpy just to hit the cv2 path.

---

### 1.6 [LOW] [DONE] Debug logging in hot path (`video.py`)

**Location:** `src/frigate_buffer/services/video.py` ~295–299 and ~370–374  

**Finding:** `logger.info("_run_detection: converting xyxy tensor to CPU/numpy")` and similar logs run in the detection hot path. They add noise and trivial CPU work.

**Recommendation:** Remove or downgrade to `logger.debug` so production logs are not flooded.

---

## 2. Memory Management & Spikes

### 2.1 [MEDIUM] [DONE] Redundant `batch.float()` in `_run_detection_on_batch` (`video.py`)

**Location:** `src/frigate_buffer/services/video.py` ~334–349  

```python
batch = batch.float()
batch.div_(255.0)
# ...
batch_resized = F.interpolate(
    batch.float(), size=(target_h, target_w), mode="bilinear", align_corners=False
)
```

**Finding:** `batch` is already float after `batch.float(); batch.div_(255.0)`. Calling `batch.float()` again in `F.interpolate` is redundant (no extra copy for the float cast, but the call is unnecessary). The real cost is that `F.interpolate` allocates a **new tensor** for `batch_resized`; that is required. No extra full-sized copy from the second `float()`.

**Recommendation:** Use `batch` directly in `F.interpolate`: `batch_resized = F.interpolate(batch, size=(target_h, target_w), ...)` to avoid redundant call and clarify intent.

---

### 2.2 [LOW] [DONE] `run_detection_on_image`: non–in-place normalize (`video.py`)

**Location:** `src/frigate_buffer/services/video.py` ~267  

```python
t = t.float() / 255.0
```

**Finding:** This creates an extra tensor for the division. For a single image the impact is small.

**Recommendation:** Optional micro-optimization: `t = t.float(); t.div_(255.0)` to avoid the division copy. Low priority.

---

### 2.3 [MEDIUM] [DONE] Compilation: full slice frames held in RAM before encode (`video_compilation.py`)

**Location:** `src/frigate_buffer/services/video_compilation.py` ~570–582, ~419–421  

```python
for i in range(n_frames):
    # ...
    arr = cropped[0].permute(1, 2, 0).cpu().numpy()
    frames_list.append(arr)
# ...
_encode_frames_via_ffmpeg(frames_list, target_w, target_h, tmp_output_path)
```

**Finding:** All frames for a slice are accumulated in `frames_list` (numpy arrays in RAM). For a long slice at 20 fps this can be hundreds of frames (e.g. 600 frames × 1440×1080×3 ≈ 2.7 GB per slice). Only after the slice is fully collected are they passed to FFmpeg. This creates a **memory spike** and delays the start of encoding.

**Recommendation:** Stream frames to FFmpeg stdin instead of buffering the whole slice: open the FFmpeg process once per compilation (or per slice if process reuse is not trivial), and in the per-slice loop write each `arr.tobytes()` to `proc.stdin` as soon as it is produced; then `del arr` and optionally run `torch.cuda.empty_cache()` periodically. This reduces peak RAM and can improve perceived latency. Requires refactoring `_encode_frames_via_ffmpeg` to accept an iterator or a callback that writes frames, or to be called incrementally.

---

### 2.4 [LOW] [DONE] `gpu_decoder`: `torch.stack(..., dim=0).clone()` (`gpu_decoder.py`)

**Location:** `src/frigate_buffer/services/gpu_decoder.py` ~64  

**Finding:** The decoder returns DLPack-backed frames; cloning after stack is intentional so the decoder can be closed without invalidating the batch. This is documented and correct.

**Recommendation:** No change. Documented in `get_frames` docstring and inline comment.

---

### 2.5 [DONE] VRAM release after batches

**Locations:**  
- `video.py`: `del batch` and `torch.cuda.empty_cache()` after each detection batch; same in `finally`.  
- `multi_clip_extractor.py`: `del batch`, `torch.cuda.empty_cache()` after each frame.  
- `video_compilation.py`: `del batch` and `torch.cuda.empty_cache()` after each slice’s frame loop.

**Finding:** VRAM is released consistently after each batch/frame/slice. No missing `del` or `empty_cache` found in the decode/inference/compilation paths.

**Recommendation:** No change. `del batch` and `torch.cuda.empty_cache()` are now present in `finally` blocks in video.py, multi_clip_extractor.py, and video_compilation.py to guarantee VRAM release even when an error occurs.

---

## 3. Redundant I/O and File Parsing

### 3.1 [HIGH] [DONE] Same `detection.json` read twice in one compilation flow (`video_compilation.py`)

**Locations:**  
- `compile_ce_video` (lines 753–769): reads `detection.json` for each camera into `sidecars` and `native_sizes`.  
- `generate_compilation_video` (lines 621–646): called from `compile_ce_video` and again reads `detection.json` for each camera into `sidecar_cache`.

**Finding:** For a single `compile_ce_video` run, each camera’s `detection.json` is **parsed twice**: once in `compile_ce_video` (for timeline and trimming) and once in `generate_compilation_video` (for crop computation). Same file, same process, redundant disk and JSON parsing.

**Recommendation:**  
- Have `compile_ce_video` build the full sidecar dict (including `entries`, `native_width`, `native_height`) once per camera and pass it into `generate_compilation_video`.  
- Change `generate_compilation_video(slices, ce_dir, output_path, ...)` to accept an optional `sidecars: dict[str, dict] | None = None`. If provided, use it and do not read from disk; if `None`, keep the current read-from-disk behavior for backward compatibility.  
- In `compile_ce_video`, after building `sidecars` and `native_sizes`, build a `sidecar_cache`-style dict (full structure per camera) and call `generate_compilation_video(..., sidecars=sidecar_cache)`.

---

### 3.2 [MEDIUM] [DONE] detection.json read by multiple consumers in the same CE flow

**Flow:**  
1. Lifecycle: `generate_detection_sidecars_for_cameras` **writes** `detection.json` per camera.  
2. `on_ce_ready_for_analysis` → `analyze_multi_clip_ce` → **multi_clip_extractor** reads `detection.json` per camera (parallel, one read per camera).  
3. Lifecycle then calls **compile_ce_video** → **compile_ce_video** and **generate_compilation_video** read `detection.json` again (and, per 3.1, twice within compilation).

**Finding:** There is no shared in-process cache of parsed sidecars. Multi-clip extraction and compilation are separate phases; each reads from disk. For a single CE, each camera’s file is read at least twice (extraction + compilation) and, within compilation, twice more if 3.1 is not fixed.

**Recommendation:**  
- Fix 3.1 so compilation reads each file at most once.  
- Optionally introduce a small **per-CE cache** (e.g. in lifecycle or a dedicated helper): after sidecar generation, pass the parsed sidecar dicts (or file paths + parsed data) to both `analyze_multi_clip_ce` and `compile_ce_video` so that neither re-reads the same file in the same CE run. This may require signature changes so that extraction and compilation can accept pre-parsed sidecars when available.

---

### 3.3 [MEDIUM] [DONE] Redundant ffprobe on the same clip

**Locations:**  
- `video.py`: `_get_video_metadata(clip_path)` in `generate_detection_sidecar` (before GPU_LOCK, ~470) and in `_decoder_frame_count`.  
- `multi_clip_extractor.py`: `_get_fps_duration_from_path(path)` → `_get_video_metadata(path)` **inside GPU_LOCK** (see §4) when opening decoders.  
- `video_compilation.py`: `_get_video_metadata(clip_path)` **inside GPU_LOCK** when `frame_count <= 0` for a slice.

**Finding:** The same clip can be ffprobe’d in: (1) sidecar generation, (2) multi-clip extraction (when opening decoders), (3) compilation (per slice when decoder reports zero frame count). So one clip may be probed multiple times across the pipeline, and in (2) and (3) the probe runs **under GPU_LOCK** (see §4).

**Recommendation:**  
- Move all ffprobe calls **outside** GPU_LOCK (see §4).  
- Add an optional **metadata cache** keyed by clip path (e.g. in-memory, TTL or single-run): when opening a clip for decode or compilation, check the cache first and call `_get_video_metadata` only on cache miss; store result in cache. This reduces redundant subprocess and disk I/O.

**Done:** In-memory metadata cache `_METADATA_CACHE` keyed by clip path (realpath when file exists) was added in video.py. All callers (sidecar, multi_clip_extractor via `_get_fps_duration_from_path`, video_compilation) benefit via `_get_video_metadata`; repeated calls for the same clip return the cached tuple without running ffprobe again.

---

## 4. Lock Contention & Concurrency
 ### 4.1 [HIGH] [DONE] ffprobe inside GPU_LOCK (`video_compilation.py`)  

**Location:** `src/frigate_buffer/services/video_compilation.py` ~512–518  

```python
with GPU_LOCK:
    with create_decoder(clip_path, gpu_id=cuda_device_index) as ctx:
        frame_count = len(ctx)
        if frame_count <= 0:
            meta = _get_video_metadata(clip_path)  # subprocess + disk I/O
            fps = (meta[2] if meta and meta[2] > 0 else 30.0)
            # ...
        batch = ctx.get_frames(src_indices)
```

**Finding:** `_get_video_metadata(clip_path)` runs **ffprobe** (subprocess and disk read) **while holding GPU_LOCK**. No GPU operation is performed during ffprobe; the GPU is idle and other threads (e.g. other decoders, compilation slices) are blocked. This increases latency and underutilizes the GPU.

**Recommendation:**  
- Call `_get_video_metadata(clip_path)` **before** entering `with GPU_LOCK`. Compute `fps` and fallback `frame_count` from `meta` outside the lock.  
- Enter the lock only for `create_decoder`, `len(ctx)`, and `get_frames`. If `frame_count <= 0`, use the pre-fetched `meta` (or a safe default) without calling ffprobe again inside the lock.

---

### 4.2 [HIGH] [DONE] ffprobe inside GPU_LOCK (`multi_clip_extractor.py`)

**Location:** `src/frigate_buffer/services/multi_clip_extractor.py` ~267–306  

```python
with GPU_LOCK:
    for (cam, path) in clip_paths:
        # ...
        ctx = stack.enter_context(create_decoder(path, ...))
        count = len(ctx)
        path_meta = _get_fps_duration_from_path(path)  # -> _get_video_metadata(path)
        if path_meta is not None:
            fps, duration_sec = path_meta[0], path_meta[1]
            # ...
```

**Finding:** `_get_fps_duration_from_path(path)` calls `_get_video_metadata(path)` (ffprobe) **while holding GPU_LOCK**. The lock is held for the entire loop over all cameras, so multiple ffprobe subprocess runs occur with the GPU locked.

**Recommendation:**  
- Precompute fps/duration for all `clip_paths` **before** acquiring GPU_LOCK (e.g. loop over `clip_paths`, call `_get_fps_duration_from_path(path)` for each, store in a dict keyed by path or camera).  
- Inside GPU_LOCK, only open decoders and use `len(ctx)`; use the precomputed metadata when `len(ctx)` is 0 or for duration/fps. This keeps GPU_LOCK limited to decoder open and frame access.

---

### 4.3 [HIGH] [DONE] detection sidecar JSON write inside GPU_LOCK (`video.py`)

**Location:** `src/frigate_buffer/services/video.py` ~479–576  

```python
with GPU_LOCK:
    with create_decoder(clip_path, ...) as ctx:
        # ... decode, YOLO, build sidecar_entries ...
        payload = {"native_width": ..., "native_height": ..., "entries": sidecar_entries}
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ...)
        return True
```

**Finding:** The **entire** sidecar generation, including building `sidecar_entries` and the **disk write** (`open` + `json.dump`), runs inside GPU_LOCK. Decoder and get_frames are GPU work; building Python dicts and writing JSON to disk are CPU and I/O. Holding the lock during the write blocks any other thread from using the GPU (e.g. another camera’s sidecar or compilation).

**Recommendation:**  
- Build `sidecar_entries` and `payload` inside the lock (they depend on decode/YOLO results).  
- **Release the lock** (exit the `with create_decoder` and the `with GPU_LOCK`) before writing to disk.  
- After releasing the lock, open `sidecar_path` and `json.dump(payload, f)`. Ensure no decoder or GPU state is used after the lock is released. This keeps GPU_LOCK limited to decoder create and get_frames; JSON serialization and file write run outside the lock.

---

### 4.4 [MEDIUM] [DONE] GPU_LOCK scope in multi_clip_extractor: one lock for all decoder opens

**Location:** `src/frigate_buffer/services/multi_clip_extractor.py` ~267–306  

**Finding:** A single `with GPU_LOCK` wraps opening **all** decoders (one per camera). Combined with 4.2, this keeps the lock held for the whole loop (decoder open + ffprobe per camera). Design is “one lock for all GPU decoder access,” so broadening the lock is intentional; the main issue is doing **non-GPU work** (ffprobe) inside it. Fixing 4.2 will already reduce contention.

**Recommendation:** After moving ffprobe out of the lock (4.2), no further change is required for this finding. If future refactors allow opening decoders one at a time and releasing the lock between cameras, that could improve concurrency with other GPU users (e.g. compilation), but it would require ensuring PyNvVideoCodec/context usage is safe with interleaved access.

**Done:** With ffprobe moved out of the lock (4.2), GPU_LOCK scope in multi_clip_extractor is now strictly limited to GPU-heavy decoder operations (create_decoder, len(ctx), get_frames).

---

## 5. Summary Table

| ID   | Severity | Category        | Short description |
|------|----------|-----------------|-------------------|
| 1.5  | Medium   | CPU/GPU boundary| ai_analyzer numpy path: ensure no GPU frame sent via cv2 path |
| 1.6  | Low      | CPU/GPU boundary| Reduce detection hot-path logging |
| 2.1  | Medium   | Memory          | Remove redundant batch.float() in F.interpolate call |
| 2.2  | Low      | Memory          | Optional: in-place div in run_detection_on_image |
| 2.3  | Medium   | Memory          | Stream compilation frames to FFmpeg instead of buffering |
| 3.1  | High     | Redundant I/O   | Pass sidecars from compile_ce_video to generate_compilation_video |
| 3.2  | Medium   | Redundant I/O   | Consider per-CE sidecar cache for extraction + compilation |
| 3.3  | Medium   | Redundant I/O   | [DONE] Metadata cache for ffprobe in video.py (_METADATA_CACHE) |
| 4.1  | High     | Lock contention | Move ffprobe out of GPU_LOCK in video_compilation |
| 4.2  | High     | Lock contention | Move ffprobe out of GPU_LOCK in multi_clip_extractor |
| 4.3  | High     | Lock contention | Move detection sidecar JSON write outside GPU_LOCK in video.py |
| 4.4  | Medium   | Lock contention | [DONE] Addressed by fixing 4.2; GPU_LOCK scope strictly decoder ops |

---

## 6. Recommended Implementation Order

1. **High impact, low risk:** Move ffprobe and sidecar JSON write outside GPU_LOCK (4.1, 4.2, 4.3).  
2. **High impact, small refactor:** Pass pre-loaded sidecars from `compile_ce_video` into `generate_compilation_video` (3.1).  
3. **Medium impact:** Stream compilation frames to FFmpeg (2.3).  
4. **Medium/low:** Redundant `batch.float()` fix (2.1), optional metadata cache (3.3), and debug log trim (1.6).

---

*End of report. No code was modified; this document is for review and implementation planning only.*
