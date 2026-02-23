# Diagnostic: Sidecar Writing, Timeline EMA, and Compilation Fallback

This document answers three questions about the pipeline: sidecar writing (video.py), timeline generation (timeline_ema + compile_ce_video), and what triggers the "Compilation fallback to center crop" ERROR logs when YOLO inference has run successfully.

---

## 1. Sidecar Writing (video.py)

**Question:** After YOLO runs inference on the batches, is the code properly formatting and writing bounding boxes to `detection.json`, and are coordinates correctly translated from the batch?

### What the code does

- **Batch loop:** `generate_detection_sidecar` decodes the clip with PyNvVideoCodec, builds `indices = range(0, frame_count, detection_frame_interval)`, then processes in chunks of **BATCH_SIZE** (4 in `video.py`, not 16). For each chunk it calls `ctx.get_frames(chunk_indices)` → `_run_detection_on_batch(model, batch, ...)`.
- **Index alignment:** `det_lists = _run_detection_on_batch(...)` returns one list of detections per image in the batch. The loop does `for i, idx in enumerate(chunk_indices)` and uses `det_lists[i]` for frame index `idx`. So **frame index ↔ batch index is correct**; each sidecar entry gets the right `frame_number` and `timestamp_sec = idx / fps`.
- **Format:** Each entry is written as:
  - `frame_number`, `timestamp_sec`, `detections`
  - Each detection has `label`, `bbox`, `centerpoint`, `area` (from `_run_detection_on_batch` and optionally `_scale_detections_to_native`).
- **Scaling:** When `native_w > 0 and native_h > 0 and read_w > 0 and read_h > 0`, detections are scaled with `_scale_detections_to_native(det, read_w, read_h, native_w, native_h)`. That function assumes **bbox/centerpoint/area are in (read_w, read_h) space** (decoder frame size) and scales them to native (ffprobe width/height).

### Potential coordinate-space bug

- **Assumption:** `_scale_detections_to_native` assumes YOLO returns boxes in **decoder frame space** (read_w × read_h). When the input to YOLO is a **tensor batch**, Ultralytics may resize internally to `imgsz` (e.g. 640) for inference. In that case, `r.boxes.xyxy` would be in **imgsz space**, not read_w × read_h.
- **Effect:** If boxes are actually in imgsz space but we scale as if they were in (read_w, read_h), bbox/centerpoint/area in the sidecar will be wrong (e.g. far too small or wrong position). That would cause:
  - **Timeline EMA:** Person area at each time would be wrong → camera assignment could be wrong or degenerate (e.g. one camera always wins).
  - **Compilation:** Crop would be computed from wrong centers → effectively center crop or nonsensical panning.
- **Recommendation:** Verify the coordinate space when calling `model(batch, ..., imgsz=imgsz)` with a tensor. If Ultralytics returns boxes in imgsz space for tensor input, add a scaling step from **imgsz → (read_w, read_h)** before (or instead of) scaling read → native. If it already returns “original” tensor dimensions, the current scaling is correct.

### Other sidecar-writing details

- If ffprobe fails, `native_w, native_h` are 0; scaling is skipped and **raw YOLO coordinates** are written. If those are in imgsz space, they will be wrong. When `native_w <= 0 and read_w > 0`, the payload still gets `native_width`/`native_height` from `read_w`/`read_h` after the loop, so the written “native” size is consistent with the (possibly wrong) unscaled coordinates.
- The JSON payload is `{"native_width", "native_height", "entries": [{frame_number, timestamp_sec, detections}, ...]}`. Downstream code (timeline_ema, video_compilation, multi_clip_extractor) expects `entries` and, per entry, `timestamp_sec` and `detections` with `label`, `bbox` or `box`, `centerpoint`, `area`.

**Summary:** Format and batch indexing are correct. The main risk is a **coordinate-space mismatch** (YOLO in imgsz vs. decoder read size), which would corrupt sidecar bbox/center/area and explain wrong timeline and static/center crop in compilation.

---

## 2. Timeline Generation (timeline_ema.py + compile_ce_video)

**Question:** Is timeline_ema successfully reading the sidecars and producing a valid camera timeline, or an empty/invalid one that causes multi_clip_extractor (or compilation) to fail?

### How it works

- **timeline_ema.py** does **not** read files. It only implements:
  - `build_dense_times(step_sec, max_frames_min, multiplier, global_end)` → list of sample times in `[0, global_end]`.
  - `build_phase1_assignments(times, cameras, area_at_t, native_size_per_cam, ...)` → list of `(t_sec, camera)` using EMA + hysteresis + segment merge.
- The **caller** is responsible for loading sidecars and defining `area_at_t(camera, t_sec)`.

### In compile_ce_video (video_compilation.py)

1. **Load sidecars:** For each camera with a clip under `ce_dir`, it loads `ce_dir/<cam>/detection.json`. It fills:
   - `sidecars[cam] = data.get("entries") or []`
   - `native_sizes[cam] = (native_width, native_height)` (or `(0,0)` for legacy list format).
2. **Timeline:** Builds `dense_times = timeline_ema.build_dense_times(step_sec, max_frames_min, multiplier, global_end)` and:
   - `_person_area_at_time(cam, t_sec)` finds the **nearest** sidecar entry (by `timestamp_sec`) and sums `area` over detections whose `label` is in `("person", "people", "pedestrian")`.
   - `assignments = timeline_ema.build_phase1_assignments(dense_times, cameras, _person_area_at_time, native_sizes, ...)`.
3. **Slices:** `slices = assignments_to_slices(assignments, global_end)` builds one slice per assignment; each slice has `camera`, `start_sec`, `end_sec`.

So timeline_ema **does not** read sidecars; **compile_ce_video** does. If sidecars are missing or empty, `compile_ce_video` returns early:

- No cameras with clips → `"No cameras found with clips"` → return `None`.
- No sidecars loaded (all `detection.json` missing or unparseable) → `"No sidecars available in {ce_dir}"` → return `None`.
- `build_phase1_assignments` can return an empty list if `not times or not cameras`; then `"No camera assignments could be generated"` → return `None`.

If sidecar **entries** exist but **all** have empty or zero person area, `_person_area_at_time` is always 0 for every camera and every t. Then EMA stays 0, and the “best” camera is still chosen (e.g. first in list or tie-break), so you still get a non-empty assignment list and slices. So timeline_ema itself does **not** return an empty timeline just because there are no detections; you still get a valid list of (t, camera). The problem in that case would be downstream: compilation will then hit the “no detections” fallback per slice (see below).

### In multi_clip_extractor

- It loads sidecars via `_load_sidecar_for_camera` (expects `entries` + optional `native_width`/`native_height`).
- If **any** camera lacks a sidecar, it does **not** use timeline_ema for extraction; it logs `"Skipping multi-clip extraction: not all cameras have detection sidecars"` and returns `[]`. So frame extraction fails when any camera is missing `detection.json` or returns `None` from `_load_sidecar_for_camera`.

**Summary:** Timeline_ema does not read sidecars; the callers do. A valid (non-empty) timeline is produced as long as sidecars exist and are loaded and `times`/`cameras` are non-empty. Empty or zero person area everywhere still yields a valid assignment list; the failure mode then shows up as compilation fallback (no detections) or wrong camera choice if coordinates/areas are wrong (e.g. due to the sidecar coordinate-space issue above).

---

## 3. Fallback logs in video_compilation.py

**Question:** What specific condition triggers the “Compilation fallback to center crop” ERROR when YOLO has run successfully?

The relevant block in `generate_compilation_video` (after loading sidecars per camera and before the encode) is:

```python
for i, sl in enumerate(slices):
    cam = sl["camera"]
    ...
    entries = sidecar_data.get("entries") or []
    if not entries:
        logger.error(
            "Compilation fallback to center crop: no sidecar entries for camera=%s slice [%.2f, %.2f]; output will be static.",
            cam, t0, t1,
        )
    else:
        entry0 = _nearest_entry_at_t(entries, t0, ts_sorted)
        entry1 = _nearest_entry_at_t(entries, t1, ts_sorted)
        dets0 = (entry0 or {}).get("detections") or []
        dets1 = (entry1 or {}).get("detections") or []
        if not dets0 or not dets1:
            logger.error(
                "Compilation fallback to center crop: no detections in sidecar for camera=%s slice [%.2f, %.2f]; output will be static.",
                cam, t0, t1,
            )
    sl["crop_start"] = calculate_crop_at_time(...)  # always run; center if no detections
    sl["crop_end"] = ...
```

So there are two distinct ERROR conditions:

1. **"no sidecar entries for camera=... slice [t0, t1]"**  
   Triggered when **`entries` is empty** for that camera in the sidecar cache. So either:
   - The sidecar file for that camera was missing or failed to load (we already filled the cache with `entries: []` and logged "sidecar missing" or "error loading sidecar"), or
   - The sidecar was loaded but the payload had no `entries` (or empty list).  
   If YOLO ran and wrote the file successfully, this usually means the file for this camera wasn’t written, wasn’t read (e.g. wrong path), or the written JSON doesn’t have `entries` or has an empty list.

2. **"no detections in sidecar for camera=... slice [t0, t1]"**  
   Triggered when **the sidecar has entries** but either:
   - the **nearest entry to the slice start time `t0`** has no `detections` or an empty list, **or**
   - the **nearest entry to the slice end time `t1`** has no `detections` or an empty list.  
   So even if YOLO ran and wrote entries for every sampled frame, **if at the specific times t0 and t1 the nearest frames have no person** (empty `detections`), this ERROR is logged. The crop is still computed (via `calculate_crop_at_time`), which falls back to center when the nearest entry has no detections, so the output is a **static center crop** for that slice.

So when YOLO has run successfully, the condition that most often triggers the fallback is:

- **The nearest sidecar entry to the slice start or the slice end has an empty `detections` list.**  
  That can happen if:
  - Many frames have no person (e.g. DETECTION_FRAME_INTERVAL skips to frames with no person at boundaries), or
  - Person area/coordinates were written incorrectly (e.g. wrong coordinate space), so downstream could be treating “has detections” in a way that fails, or
  - The slice boundaries (t0, t1) fall between sampled frames and the nearest sampled frame has no person.

In all these cases, `calculate_crop_at_time` still runs and uses the center of the frame when there are no detections, so the compilation **succeeds** but the video is **static center crop** with no camera switching or panning.

---

## Summary table

| Component            | Finding |
|----------------------|--------|
| **Sidecar writing**  | Format and batch→frame mapping are correct. Risk: bbox/center/area may be in YOLO imgsz space instead of decoder (read_w/read_h) space; then scaling to native is wrong and can cause wrong timeline and static/center crop. |
| **Timeline (EMA)**   | timeline_ema does not read sidecars; compile_ce_video and multi_clip_extractor do. Timeline is valid as long as sidecars load and times/cameras are non-empty. Empty person area everywhere still yields assignments; wrong areas (e.g. from bad coordinates) can yield wrong or single-camera timeline. |
| **Fallback when YOLO OK** | Trigger: for each slice, **either** “no sidecar entries” for that camera **or** the **nearest entry to t0 or t1 has empty detections**. The latter is the usual case when YOLO ran: boundary frames with no person (or wrong coordinates) → ERROR logged and static center crop. |

---

## Recommended next steps

1. **Verify YOLO output coordinate space** when input is a tensor batch (with `imgsz`). If boxes are in imgsz space, add scaling imgsz → (read_w, read_h) before or inside `_scale_detections_to_native`.
2. **Inspect a real `detection.json`**: Confirm `entries` length, `timestamp_sec` range, and that entries at slice boundaries have non-empty `detections` with sensible `bbox`/`centerpoint`/`area` in native (or decoder) space.
3. **Soften the “no detections” rule** (optional): e.g. require only that **at least one** of the nearest entries to t0/t1 has detections, or use the nearest entry that **has** detections in the slice, so a single empty boundary frame doesn’t force center crop for the whole slice.
