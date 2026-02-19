# Plan: Fix Carport Selection and Person-in-Frame Alignment (revised)

## Findings from your feedback and logs

### 1. Carport reader is dying during extraction (root cause of “no carport frames”)

**Log:** `Reader process died for camera carport (/app/storage/events/test1/carport/clip.mp4); dropping camera for rest of extraction: read of closed file`

So the second camera **does** start; the FFmpeg reader for carport **dies mid-run**. After that we set `caps[cam] = None` and that camera is dropped for the rest of extraction, so carport never contributes frames. This is not hysteresis or bias — it’s a **reader lifecycle / stability** issue.

**Possible causes:**

- **GPU (NVDEC) contention:** Two NVDEC decoders (doorbell + carport) running in parallel may exhaust GPU memory or hit driver limits; the second process (carport) may be the one that gets killed or closes the pipe.
- **Different decode path:** If doorbell uses GPU and carport fell back to CPU (e.g. “GPU transcode failed, fallback” for carport transcode), we still open **both** for decode in the extractor with the same logic (try NVDEC first, then CPU). So both could be NVDEC in the extractor, or one NVDEC and one CPU. A flaky NVDEC process could still explain “read of closed file” for one of them.
- **Clip/codec:** Different resolution, codec, or length could make one decode path more likely to fail (e.g. one clip triggers a decoder bug).

**Conclusion:** We need **more logging** around open and on reader death so we can see which backend each camera uses, and at what time/sample the carport reader died. Optionally, consider **using CPU decode for the second camera** when two NVDEC streams are used, to reduce GPU load (configurable).

---

### 2. First-camera bias: decay to 0, and make it exponential

**Current behavior ([multi_clip_extractor.py](src/frigate_buffer/services/multi_clip_extractor.py) ~316–322):**

- Formula: `1.5 - 0.5 * (t_sec / global_end)` → **linear** from 1.5 at t=0 to **1.0** at end of clip.
- So the bias **never goes to 0**; it only goes from 1.5 down to 1.0 over the **entire** clip length.

**Your requirement:** Bias should go **down to 0** within **a few seconds** (or faster), and you asked if the reduction should be **exponential**.

**Research (exponential decay):**

- Standard form: `N(t) = N0 * exp(-t / τ)` where τ is the time constant.
- After one time constant (t = τ), value is about **0.37** of initial; after 3τ, about **0.05**.
- So with **τ ≈ 1 s** and initial 1.5: at t = 3 s we get ~0.075 (effectively 0 for selection purposes). So “down to 0 within a few seconds” is achieved with **exponential decay** and a **short τ** (e.g. 1 s).

**Proposed change:**

- **Exponential decay to (effectively) 0:**  
  `bias(t) = initial_bias * exp(-t_sec / τ)` when `cam == first_camera_bias`, else `1.0`.
- **Config:** Add `first_camera_bias_decay_seconds` (τ), default **1.0** (so within ~3 s bias is negligible). Optionally `first_camera_bias_initial` (default 1.5). Optionally cap to 0 after a window (e.g. 5 s) for clarity: `min(bias(t), 0)` after 5 s or use 0 when `t_sec > cap_seconds`.
- **No dependency on clip length:** Decay is time-based (seconds), not `t / global_end`, so bias drops in the first few seconds regardless of clip duration.

---

### 3. Hysteresis relaxation (unchanged intent)

- Allow switch when **current camera’s (bias-adjusted) person area < threshold** (e.g. 200 px²) so the other camera can win without requiring 1.3×.
- Optionally lower the switch multiplier from **1.3× to 1.2×** so switching is easier when both have person.
- New config: e.g. `person_area_switch_threshold` under `multi_cam` (default 200; 0 = disable, use only 1.3× rule).

---

## Revised plan (ordered)

### A. Logging to diagnose carport reader death

**File:** [src/frigate_buffer/services/multi_clip_extractor.py](src/frigate_buffer/services/multi_clip_extractor.py)

1. **After reopening caps (per camera):** Log at INFO (or DEBUG): camera name, path, decode backend (GPU/CPU), duration_sec, fps. This confirms both cameras opened and which backend each uses (e.g. “doorbell GPU 12.5s, carport CPU 12.5s” or “both GPU”).
2. **When a reader dies:** In the existing `except _READ_ERRORS` block, also log:
   - Sample time **T** (sec) and **frame index** for that camera (from `frame_index.get(cam, 0)`).
   - **Remaining cameras** (list of cameras still in `caps` with non-None).  
   This shows “carport died at T=5.2s, frame 156; doorbell still active.”
3. **Optional (DEBUG):** Log per-camera person area at the first sample time (and every Nth sample) when using sidecars, so we can see if carport ever had non-zero area before it died.

**Config:** No new keys for logging; use existing `log_level` (DEBUG for verbose area logs).

---

### B. First-camera bias: exponential decay to 0

**File:** [src/frigate_buffer/services/multi_clip_extractor.py](src/frigate_buffer/services/multi_clip_extractor.py)

- Replace the current linear formula with:
  - `bias(t_sec) = initial_bias * exp(-t_sec / τ)` for the primary camera; other cameras 1.0.
  - Use **τ (tau)** from config `first_camera_bias_decay_seconds` (default **1.0**).
  - Optional: **cap to 0** after `first_camera_bias_cap_seconds` (e.g. 5): for `t_sec > cap`, use 0 so the primary gets no boost after a few seconds.
- **Math:** Need `import math` and use `math.exp(-t_sec / tau)`. If tau <= 0, treat as no decay (e.g. constant 1.5 or 1.0).

**Config / schema:**

- [config.py](src/frigate_buffer/config.py): Under `multi_cam`, add:
  - `Optional('first_camera_bias_decay_seconds'): float` (default 1.0),
  - Optional: `Optional('first_camera_bias_initial'): float` (default 1.5),
  - Optional: `Optional('first_camera_bias_cap_seconds'): float` (default 0 = no cap, or 5 to force 0 after 5 s).
- [config.example.yaml](config.example.yaml) and [config.yaml](config.yaml): Document and add defaults. Flatten to e.g. `FIRST_CAMERA_BIAS_DECAY_SECONDS` in the flat config and pass into `extract_target_centric_frames` (via ai_analyzer from orchestrator). ai_analyzer already gets `primary_camera`; it needs to also pass decay_seconds (and optional initial/cap) into the extractor.

**MAP.md:** Update multi-clip section to state that first-camera bias uses **exponential decay to 0** with configurable time constant (default 1 s), and optional cap seconds.

---

### C. Hysteresis relaxation (low-area switch + optional 1.2×)

**File:** [src/frigate_buffer/services/multi_clip_extractor.py](src/frigate_buffer/services/multi_clip_extractor.py)

- **Low-area escape:** If current camera’s (bias-adjusted) person area is **&lt; threshold** (config `person_area_switch_threshold`, default 200), allow switching to any camera with **strictly larger** area without requiring 1.3×.
- **Optional:** Use **1.2×** instead of 1.3× for the “switch when other camera has X× current” rule (config or constant).
- **Config:** `person_area_switch_threshold` (int, default 200; 0 = disable low-area escape). Optionally `camera_switch_ratio` (float, default 1.2) for the multiplier.

**Config schema / defaults:** Add to `multi_cam` in config.py and example YAML; document in MAP.md.

---

### D. Optional: Prefer CPU decode for second camera when both would use NVDEC

**Goal:** Reduce chance of “read of closed file” by avoiding two simultaneous NVDEC decoders.

**Approach:** When opening caps after the FPS/duration pass, open the **first** camera (by clip_paths order) with the current logic (try NVDEC then CPU). For the **second and subsequent** cameras, optionally **skip NVDEC** and use CPU decode only when a new config is set (e.g. `multi_cam.decode_second_camera_cpu_only: true`). This is a conservative, configurable fix; we can enable it only if logs show both cameras using GPU when carport dies.

**File:** [multi_clip_extractor.py](src/frigate_buffer/services/multi_clip_extractor.py): In the reopen loop, if `decode_second_camera_cpu_only` is True and camera index &gt; 0, open with `ffmpegcv.VideoCapture(path)` only (no VideoCaptureNV). Requires passing this option into `extract_target_centric_frames` (e.g. via ai_analyzer from config).

---

### E. Detection-aligned sampling (“no person in frame” fix)

When sidecars exist, build sample times from **detection timestamps where person area &gt; 0**, then subsample and extract so the **frame at that timestamp** is output (image matches detection). Fallback to fixed-step grid when no person detections exist.

#### E.1. Build the merged list of candidate output times (T_det)

- From each camera’s sidecar, collect `timestamp_sec` for entries where **person area &gt; 0** (keep one list per camera).
- **Empty sidecar / missing camera:** One camera may have detections and the other none. When merging, handle `None` or empty lists safely: use safe list comprehensions or filters (e.g. `(sidecars.get(cam) or [])`, filter entries with person area &gt; 0, then flatten). Do not assume every camera has a non-empty list; merging must not raise when one list is empty or missing.
- Merge all camera lists into one sorted list of unique timestamps (or sort after concatenating). This is the **master T_det timeline** (ascending).

#### E.2. Subsample with minimum time gap (step_sec) — critical

- **Trap:** Taking the first `max_frames_min` timestamps from the merged list would extract many frames from the **start** of the event (e.g. 15 frames in the first 3 seconds) and ignore the rest of the clip.
- **Fix:** Enforce a **minimum time gap** between selected detection timestamps using **max_multi_cam_frames_sec** from config (e.g. `step_sec = 1` from `config.yaml`).
  - Iterate through the merged detection timestamps **in ascending order**.
  - For each candidate `t`: if `t >= last_extracted_timestamp + step_sec` (and we’re under `max_frames_min`), **select** this timestamp for extraction, then set `last_extracted_timestamp = t`.
  - Initialize `last_extracted_timestamp` to a value that allows the first valid timestamp to be selected (e.g. `-step_sec` or `-inf`).
- Result: selected T_det values are spread across the clip with at least `step_sec` seconds between them, up to `max_frames_min` frames.

#### E.3. Sequential read constraints (ffmpegcv — no seek)

- **Trap:** ffmpegcv readers do **not** support random access (e.g. `cap.set(CAP_PROP_POS_FRAMES)` or seek). They only support **forward sequential read** (see MAP.md).
- **Fix:** Do **not** use `cap.set()`, `seek()`, or any position-setting API. Keep the existing **sequential** strategy:
  - Iterate over the **selected T_det** list in **strictly ascending chronological order** (which the subsampling already produces).
  - Use the existing `while cap.read()` (or equivalent advance) loop: maintain a running notion of **current timestamp** per camera (e.g. from frame index and fps), advance all readers forward until each has reached the next needed time.
  - When the current timestamp **matches** (or first exceeds) the target T_det for an extraction, **save that frame** for output. No seeking—only forward reads and “save when current time matches target.”

---

## Implementation order

1. **Logging (A)** — So the next run shows why carport died (backend per camera, T and remaining cameras on death).
2. **First-camera bias exponential decay (B)** — Decay to 0 in a few seconds with τ and optional cap.
3. **Hysteresis (C)** — Low-area threshold and optional 1.2×.
4. **Detection-aligned sampling (E)** — So frames show a person when we have detections.
5. **Optional second-camera CPU (D)** — Only if logs indicate NVDEC contention.

---

## Summary table

| Issue | Cause | Change |
|-------|--------|--------|
| No carport frames | Carport reader dies (“read of closed file”) | Add logging at open and on reader death; optionally CPU decode for second camera. |
| First-camera bias never 0 | Linear 1.5→1.0 over full clip | Exponential decay: 1.5*exp(-t/τ), τ≈1s, optional cap; configurable. |
| Second camera rarely wins | Hysteresis 1.3× + no low-area escape | Allow switch when current area &lt; threshold; optionally 1.2×. |
| Frames often no person | Sample at T but use detection at T_det | Detection-aligned sampling (sample at detection times, output frame at that time). |

---

## Tests and MAP.md

- **Tests:** Add/update tests for: exponential bias decay (value at t=0, t=τ, t=3τ); hysteresis with low-area threshold (switch when current &lt; threshold); detection-aligned sampling — including **subsampling** (selected timestamps respect step_sec gap; not all from start of clip), **empty sidecar** (one camera has entries, other empty/None; no exception when merging), and fallback when no person detections. Mock or fixture for sidecar data.
- **MAP.md:** Update multi-clip extraction paragraph: reader-death handling and new logging; first-camera bias exponential decay (τ, optional cap); hysteresis low-area threshold and switch ratio; detection-aligned sampling; optional second-camera CPU decode.
