# Plan to merge multi-cam frame extractor into main project

This folder holds a **standalone mock** of the multi-cam frame extractor. It is kept as **reference/sandbox only**; the mock files here are **not** modified by the main app.

**In-process AI analyzer:** The main project now includes an in-process analyzer in **`frigate_buffer/services/ai_analyzer.py`** (`GeminiAnalysisService`). It implements frame extraction from clips, Gemini proxy communication (OpenAI-compatible), and returns result to orchestrator; orchestrator persists via Frigate API and notifies HA (buffer does not publish to `frigate/reviews`). The orchestrator triggers it when a clip is ready (no MQTT bridge). The following items are **[x] Completed** via that service: **Frame Extraction**, **Proxy Communication**, **Service Integration**.

---

## Current state

- **Location:** `multi_cam_frame_extract_mock/`
- **Contents:** `multi_cam_frame_extracter.py` (mock), `daily_report_to_proxy.py` (mock), `multi_cam_system_prompt.txt`, `report_prompt.txt`, `multi_cam_frame_extractor_input_output.md` (payload spec), this plan, and README.
- **Main project:** Uses `frigate_buffer/services/ai_analyzer.py` for clip analysis; this mock folder is reference only.
- **Step 1 (Config) — [x] Done:** Schema and load_config in `frigate_buffer/config.py` and `config.example.yaml` support `multi_cam` and `gemini_proxy`; flat keys (e.g. `MAX_MULTI_CAM_FRAMES_MIN`, `GEMINI_PROXY_URL`) with No Google Fallback and Single API Key. See "What was done (Step 1)" and "Notes for future AI" below.

---

## Merge steps (to do later)

### 1. Config — [x] Completed

- **No Google Fallback:** Default proxy URL must be `""`. Do not default to any Google or external proxy URL; users must explicitly set the proxy in config or env.
- **Single API Key:** Use `GEMINI_API_KEY` only for proxy authentication. Do not create a separate proxy-specific API key config; the same key is used for the Gemini proxy (and any future proxy usage).
- **Implemented in** **frigate_buffer/config.py**: Optional schema `multi_cam` (max_multi_cam_frames_min, max_multi_cam_frames_sec, motion_threshold_px, crop_width, crop_height, multi_cam_system_prompt_file) and `gemini_proxy` (url, model, temperature, top_p, frequency_penalty, presence_penalty). Flattened to flat keys; default proxy URL `""`; env override `GEMINI_PROXY_URL`; API key from `GEMINI_API_KEY` only. **config.example.yaml** has commented `multi_cam:` and `gemini_proxy:` blocks.

### 2. Clip paths and Clip timing (vid_start)

- **Reasoning:** We cannot verify analysis accuracy if we are looking at the wrong part of the video (pre-capture buffer). Path resolution and clip timing must be correct in **ai_analyzer.py** immediately after config is available, before adding complex multi-cam logic.

**Clip paths**

- When integrated, resolve clip paths like the main app:
  - Single event: `{STORAGE_PATH}/{camera}/{timestamp}_{event_id}/clip.mp4`
  - Consolidated: `{STORAGE_PATH}/events/{ce_folder_name}/{camera}/clip.mp4`
- Use FileManager / TimelineLogger / `folder_for_event` (or equivalent) so both single-event and consolidated layouts work. Replace or augment the current "walk STORAGE_PATH for event_id in root" logic where appropriate.

**Clip timing (vid_start)**

- Clips from Frigate export often include 5–10 seconds of pre-capture buffer before the event start. The script currently uses the first metadata timestamp as vid_start; that can make seeking wrong (e.g. seek to 0:00 is actually 10s before event).
- **Fix when merging:** Use clip start from the export request (same export_buffer_before logic as the main app: see frigate_buffer/services/download.py and lifecycle.py) or derive from folder name / export response. Otherwise crops may miss the action.

### 3. Move and wire the service

- Move **multi_cam_frame_extracter.py** into **frigate_buffer/services/** (same name or e.g. `multi_cam_frame_extractor.py`).
- Remove the one-line "Lives in multi_cam_frame_extract_mock" comment.
- Have the service read config from the main app:
  - **If run by the orchestrator:** Pass the orchestrator's config (same object as rest of app).
  - **If run standalone:** Use the same config path list as main: `frigate_buffer.config.load_config()` and map nested keys (network, settings, multi_cam, gemini_proxy) into the flat keys the script expects, or add a small adapter that reads from the main config dict.

### 4. MQTT and per-frame data

- Subscribe to **both** `frigate/events` and `frigate/+/tracked_object_update`.
- **EventMetadataStore:** Populate **only** from `tracked_object_update` (frame_time, box, area, score); use `frigate/events` only for new/end and overlap logic.
- Parse camera from topic `frigate/{camera}/tracked_object_update`; correlate by event id + camera. Normalize box format (e.g. to [ymin, xmin, ymax, xmax]) if Frigate version differs.

### 5. Gemini proxy call and analysis_result.json (bridge to Reporter) — [x] Completed

- **Done in `frigate_buffer/services/ai_analyzer.py`:** HTTP POST in **send_to_proxy** (OpenAI-compatible), system prompt + base64 frames, Bearer auth. Response parsed and published to `frigate/reviews` (Main App and HA consume it). Optional: save **analysis_result.json** for Reporter (daily_report_to_proxy.py) if that script is merged later.
- Payload shapes: see **multi_cam_frame_extractor_input_output.md** in this folder (can be moved to `docs/` or kept next to the service).

### 6. Orchestrator integration (optional) — [x] Completed

- **Done:** The main app triggers analysis from the orchestrator/lifecycle when a clip is ready. `GeminiAnalysisService.analyze_clip(event_id, clip_path)` is called in a background thread; no separate process or MQTT bridge.

### 7. Daily report script (daily_report_to_proxy.py)

- **Purpose:** At 1am (configurable), compile all multi-cam recap responses from the previous calendar day and send them to the Gemini proxy to produce one AI daily report.
- **Prompt:** Load from **report_prompt.txt** (same dir or config path); edit file to change report style.
- **Config:** Add `daily_report_schedule_hour` (default 1), `report_prompt_file`, `report_output_dir` (or derive from storage). Reuse proxy URL and `GEMINI_API_KEY` from config.
- **Inputs:** Reporter populates recaps by scanning storage for **analysis_result.json** inside event folders (e.g. `.../multi_cam_frames/analysis_result.json`). Event folder date is derived from folder name (timestamp_*). Extractor must write this file after each proxy response (see §5).
- **Outputs:** Proxy returns `choices[0].message.content` (report text). Save to e.g. `{report_output_dir}/daily_reports/YYYY-MM-DD_report.md`.
- **Scheduling:** Use system cron, or `schedule` library, or a thread that sleeps until next 1am. When merged, can align with main app's daily_review_schedule_hour if desired.

### 8. Docs and cleanup

- Move **multi_cam_frame_extractor_input_output.md** into the repo (e.g. project root or `docs/`) and point to it from the README if desired.
- Update main **README.md** with a short "Multi-cam frame extractor" section (features, config, prompt files, link to payload doc). Include "Daily report" in that section.
- Remove or archive **multi_cam_frame_extract_mock/** after merge, or keep the folder only as reference and leave the code in `frigate_buffer/services/`.

---

## What was done (Step 1)

- **frigate_buffer/config.py:** `CONFIG_SCHEMA` now includes `Optional('multi_cam')` and `Optional('gemini_proxy')`. `load_config()` initializes and merges flat keys: `MAX_MULTI_CAM_FRAMES_MIN`, `MAX_MULTI_CAM_FRAMES_SEC`, `MOTION_THRESHOLD_PX`, `CROP_WIDTH`, `CROP_HEIGHT`, `MULTI_CAM_SYSTEM_PROMPT_FILE`, `GEMINI_PROXY_URL`, `GEMINI_PROXY_MODEL`, `GEMINI_PROXY_TEMPERATURE`, `GEMINI_PROXY_TOP_P`, `GEMINI_PROXY_FREQUENCY_PENALTY`, `GEMINI_PROXY_PRESENCE_PENALTY`. Default for `GEMINI_PROXY_URL` is `""`. If YAML has `gemini` but not `gemini_proxy`, URL and model are derived from `gemini`. Env `GEMINI_PROXY_URL` overrides. API key is **not** in gemini_proxy schema; use `GEMINI_API_KEY` (and existing `config['GEMINI']['api_key']`) only.
- **config.example.yaml:** Commented `multi_cam:` and `gemini_proxy:` blocks with descriptions (No Google Fallback, Single API Key).
- **tests/test_config_schema.py:** New tests for multi_cam/gemini_proxy present, defaults when omitted, backward compat (gemini only), `GEMINI_PROXY_URL` env override, invalid multi_cam type.

---

## Notes for future AI (next steps)

- **Step 2 (Clip paths and Clip timing):** Do this **before** moving the mock service or adding MQTT. Fix **ai_analyzer.py** (and any clip consumer) so clip paths match the main app (single-event and consolidated layouts) and **vid_start** / segment is correct (use export_buffer_before/export_buffer_after or export response; see frigate_buffer/services/download.py and lifecycle.py). Without this, analysis can run on the wrong part of the video (pre-capture buffer).
- **Step 3:** When moving multi_cam_frame_extracter into frigate_buffer/services, have it read from the **flat** config keys (e.g. `config['MAX_MULTI_CAM_FRAMES_MIN']`, `config['GEMINI_PROXY_URL']`). The mock uses a flat dict; no adapter needed if keys match. For proxy API key, use `config['GEMINI']['api_key']` (which is already filled from `GEMINI_API_KEY`).
- **Config keys reference:** Multi-cam: `MAX_MULTI_CAM_FRAMES_MIN`, `MAX_MULTI_CAM_FRAMES_SEC`, `MOTION_THRESHOLD_PX`, `CROP_WIDTH`, `CROP_HEIGHT`, `MULTI_CAM_SYSTEM_PROMPT_FILE`. Gemini proxy: `GEMINI_PROXY_URL`, `GEMINI_PROXY_MODEL`, `GEMINI_PROXY_TEMPERATURE`, `GEMINI_PROXY_TOP_P`, `GEMINI_PROXY_FREQUENCY_PENALTY`, `GEMINI_PROXY_PRESENCE_PENALTY`. Single API key: `config['GEMINI']['api_key']` or env `GEMINI_API_KEY`.

---

## Order of work

1. ~~Config schema and load_config~~ **Done.** No Google Fallback, Single API Key (`GEMINI_API_KEY` only).
2. Clip paths and clip timing (vid_start) in **ai_analyzer.py** so analysis uses correct segment and paths; required before multi-cam logic.
3. Move multi_cam_frame_extracter into frigate_buffer/services and make it use main config (flat keys above).
4. Add tracked_object_update subscription and feed EventMetadataStore from it only.
5. Proxy HTTP call and config/env are already wired; optional: save analysis_result.json for Reporter.
6. Orchestrator hook is done; update README and clean up mock folder when ready.
7. Daily report: move daily_report_to_proxy.py (e.g. to services/), add config for schedule and report path, define recap storage layout and implement get_previous_day_recaps; implement send_report_to_proxy; schedule at 1am (cron or in-process).
