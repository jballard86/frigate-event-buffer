# Plan to merge multi-cam frame extractor into main project

This folder holds a **standalone mock** of the multi-cam frame extractor. Nothing here is merged into the main frigate-event-buffer app yet. When ready, use this plan to integrate.

---

## Current state

- **Location:** `multi_cam_frame_extract_mock/`
- **Contents:** `multi_cam_frame_extracter.py` (mock), `daily_report_to_proxy.py` (mock), `multi_cam_system_prompt.txt`, `report_prompt.txt`, `multi_cam_frame_extractor_input_output.md` (payload spec), this plan, and README.
- **Main project:** Unchanged; no references to this mock.

---

## Merge steps (to do later)

### 1. Config

- In **frigate_buffer/config.py**:
  - Add optional schema: `multi_cam` (e.g. max_multi_cam_frames_min, max_multi_cam_frames_sec, motion_threshold_px, crop_width, crop_height, final_review_image_count) and `gemini_proxy` (url, api_key, model, temperature, top_p, frequency_penalty, presence_penalty). Add `final_review_image_count` under `settings` if desired.
  - In `load_config()`, flatten these into the main config dict (e.g. GEMINI_PROXY_URL, GEMINI_PROXY_API_KEY, MAX_MULTI_CAM_FRAMES_SEC, etc.) with sensible defaults.
  - Allow env overrides for proxy URL and api_key (for secrets).
- In **config.example.yaml**: Add commented `multi_cam:` and `gemini_proxy:` blocks.

### 2. Move and wire the service

- Move **multi_cam_frame_extracter.py** into **frigate_buffer/services/** (same name or e.g. `multi_cam_frame_extractor.py`).
- Remove the one-line “Lives in multi_cam_frame_extract_mock” comment.
- Have the service read config from the main app:
  - **If run by the orchestrator:** Pass the orchestrator’s config (same object as rest of app).
  - **If run standalone:** Use the same config path list as main: `frigate_buffer.config.load_config()` and map nested keys (network, settings, multi_cam, gemini_proxy) into the flat keys the script expects, or add a small adapter that reads from the main config dict.

### 3. MQTT and per-frame data

- Subscribe to **both** `frigate/events` and `frigate/+/tracked_object_update`.
- **EventMetadataStore:** Populate **only** from `tracked_object_update` (frame_time, box, area, score); use `frigate/events` only for new/end and overlap logic.
- Parse camera from topic `frigate/{camera}/tracked_object_update`; correlate by event id + camera. Normalize box format (e.g. to [ymin, xmin, ymax, xmax]) if Frigate version differs.

### 4. Clip paths

- When integrated, resolve clip paths like the main app:
  - Single event: `{STORAGE_PATH}/{camera}/{timestamp}_{event_id}/clip.mp4`
  - Consolidated: `{STORAGE_PATH}/events/{ce_folder_name}/{camera}/clip.mp4`
- Use FileManager / TimelineLogger / `folder_for_event` (or equivalent) so both single-event and consolidated layouts work. Replace or augment the current “walk STORAGE_PATH for event_id in root” logic where appropriate.

### 5. Clip timing (vid_start) — do not forget

- Clips from Frigate export often include 5–10 seconds of pre-capture buffer before the event start. The script currently uses the first metadata timestamp as vid_start; that can make seeking wrong (e.g. seek to 0:00 is actually 10s before event).
- **Fix when merging:** Use clip start from the export request (same export_buffer_before logic as the main app: see frigate_buffer/services/download.py and lifecycle.py) or derive from folder name / export response. Otherwise crops may miss the action.

### 6. Gemini proxy call and analysis_result.json (bridge to Reporter)

- Implement the actual HTTP POST in **send_to_proxy** (called from `process_multi_cam_event`), not in the MQTT loop.
- Build request from config (URL, api_key, model, temperature, etc.), system prompt, and base64-encoded frames (cap count e.g. via final_review_image_count). Auth per proxy (e.g. Bearer token from config/env).
- **Critical:** Save the proxy response as **analysis_result.json** in the same folder as the frames (e.g. `multi_cam_frames/analysis_result.json`). Include: title, scene, shortSummary, confidence, potential_threat_level (from response) plus event_id, camera, start_time (for the Reporter). The Reporter (daily_report_to_proxy.py) scans event folders for this file to populate get_previous_day_recaps().
- Payload shapes: see **multi_cam_frame_extractor_input_output.md** in this folder (can be moved to `docs/` or kept next to the service).

### 7. Orchestrator integration (optional)

- If the multi-cam flow should be triggered by the main app instead of a separate process: from the orchestrator (or lifecycle), on overlap or CE close, call into the multi-cam service (e.g. pass event ids, config, and storage path). The service would then use the same MQTT-derived EventMetadataStore or receive event/clip metadata from the orchestrator.

### 8. Daily report script (daily_report_to_proxy.py)

- **Purpose:** At 1am (configurable), compile all multi-cam recap responses from the previous calendar day and send them to the Gemini proxy to produce one AI daily report.
- **Prompt:** Load from **report_prompt.txt** (same dir or config path); edit file to change report style.
- **Config:** Add `daily_report_schedule_hour` (default 1), `report_prompt_file`, `report_output_dir` (or derive from storage). Reuse proxy URL/api_key from gemini_proxy config.
- **Inputs:** Reporter populates recaps by scanning storage for **analysis_result.json** inside event folders (e.g. `.../multi_cam_frames/analysis_result.json`). Event folder date is derived from folder name (timestamp_*). Extractor must write this file after each proxy response (see §6).
- **Outputs:** Proxy returns `choices[0].message.content` (report text). Save to e.g. `{report_output_dir}/daily_reports/YYYY-MM-DD_report.md`.
- **Scheduling:** Use system cron, or `schedule` library, or a thread that sleeps until next 1am. When merged, can align with main app's daily_review_schedule_hour if desired.

### 9. Docs and cleanup

- Move **multi_cam_frame_extractor_input_output.md** into the repo (e.g. project root or `docs/`) and point to it from the README if desired.
- Update main **README.md** with a short “Multi-cam frame extractor” section (features, config, prompt files, link to payload doc). Include "Daily report" in that section.
- Remove or archive **multi_cam_frame_extract_mock/** after merge, or keep the folder only as reference and leave the code in `frigate_buffer/services/`.

---

## Order of work

1. Config schema and load_config (no behavior change).
2. Move multi_cam_frame_extracter into frigate_buffer/services and make it use main config.
3. Add tracked_object_update subscription and feed EventMetadataStore from it only.
4. Align clip path resolution and vid_start with main app.
5. Implement proxy HTTP call and wire config/env (frame extractor).
6. Optionally hook into orchestrator; then update README and clean up mock folder.
7. Daily report: move daily_report_to_proxy.py (e.g. to services/), add config for schedule and report path, define recap storage layout and implement get_previous_day_recaps; implement send_report_to_proxy; schedule at 1am (cron or in-process).
