# Plan to merge multi-cam frame extractor into main project

This folder holds a **standalone mock** of the multi-cam frame extractor. It is kept as **reference/sandbox only**; the mock files here are **not** modified by the main app.

**In-process AI analyzer:** The main project now includes an in-process analyzer in **`frigate_buffer/services/ai_analyzer.py`** (`GeminiAnalysisService`). It implements frame extraction from clips, Gemini proxy communication (OpenAI-compatible), and returns result to orchestrator; orchestrator persists via Frigate API and notifies HA (buffer does not publish to `frigate/reviews`). The orchestrator triggers it when a clip is ready (no MQTT bridge). The following items are **[x] Completed** via that service: **Frame Extraction**, **Proxy Communication**, **Service Integration**.

---

## Current state

- **Location:** `multi_cam_frame_extract_mock/`
- **Contents:** `multi_cam_frame_extracter.py` (mock), `daily_report_to_proxy.py` (mock), `multi_cam_system_prompt.txt`, `report_prompt.txt`, `multi_cam_frame_extractor_input_output.md` (payload spec), this plan, and README.
- **Main project:** Uses `frigate_buffer/services/ai_analyzer.py` for clip analysis; this mock folder is reference only.
- **Step 1 (Config) — [x] Done:** Schema and load_config in `frigate_buffer/config.py` and `config.example.yaml` support `multi_cam` and `gemini_proxy`; flat keys (e.g. `MAX_MULTI_CAM_FRAMES_MIN`, `GEMINI_PROXY_URL`) with No Google Fallback and Single API Key. See "What was done (Step 1)" and "Notes for future AI" below.
- **Step 2 (Clip paths and Clip timing) — [x] Done:** Smart seeking in `ai_analyzer.py` (buffer_offset, seek, limit, stride); orchestrator looks up event and passes timestamps to `analyze_clip`; lifecycle callback stays `(event_id, clip_path)`. See "What was done (Step 2)" below.

---

## Merge steps (to do later)

### 1. Config — [x] Completed

- **No Google Fallback:** Default proxy URL must be `""`. Do not default to any Google or external proxy URL; users must explicitly set the proxy in config or env.
- **Single API Key:** Use `GEMINI_API_KEY` only for proxy authentication. Do not create a separate proxy-specific API key config; the same key is used for the Gemini proxy (and any future proxy usage).
- **Implemented in** **frigate_buffer/config.py**: Optional schema `multi_cam` (max_multi_cam_frames_min, max_multi_cam_frames_sec, motion_threshold_px, crop_width, crop_height, multi_cam_system_prompt_file) and `gemini_proxy` (url, model, temperature, top_p, frequency_penalty, presence_penalty). Flattened to flat keys; default proxy URL `""`; env override `GEMINI_PROXY_URL`; API key from `GEMINI_API_KEY` only. **config.example.yaml** has commented `multi_cam:` and `gemini_proxy:` blocks.

### 2. Clip paths and Clip timing (vid_start) — [x] Completed

- **Reasoning:** We cannot verify analysis accuracy if we are looking at the wrong part of the video (pre-capture buffer). Path resolution and clip timing must be correct in **ai_analyzer.py** immediately after config is available, before adding complex multi-cam logic.

**Clip paths**

- When integrated, resolve clip paths like the main app:
  - Single event: `{STORAGE_PATH}/{camera}/{timestamp}_{event_id}/clip.mp4`
  - Consolidated: `{STORAGE_PATH}/events/{ce_folder_name}/{camera}/clip.mp4`
- Use FileManager / TimelineLogger / `folder_for_event` (or equivalent) so both single-event and consolidated layouts work. Replace or augment the current "walk STORAGE_PATH for event_id in root" logic where appropriate.

**Clip timing (vid_start)**

- Clips from Frigate export often include 5–10 seconds of pre-capture buffer before the event start. The script currently uses the first metadata timestamp as vid_start; that can make seeking wrong (e.g. seek to 0:00 is actually 10s before event).
- **Fix when merging:** Use clip start from the export request (same export_buffer_before logic as the main app: see frigate_buffer/services/download.py and lifecycle.py) or derive from folder name / export response. Otherwise crops may miss the action.

### 3. Move and wire the service — [x] Completed

- Logic ported into **frigate_buffer/services/ai_analyzer.py** (no separate multi_cam_frame_extracter in services). Service reads from main app config (flat keys).
- **If run by the orchestrator:** Pass the orchestrator's config (same object as rest of app).
- **If run standalone:** Use the same config path list as main: `frigate_buffer.config.load_config()` and map nested keys (network, settings, multi_cam, gemini_proxy) into the flat keys the script expects, or add a small adapter that reads from the main config dict.

### 4. MQTT and per-frame data — [x] Completed

- Subscribe to **both** `frigate/events` and `frigate/+/tracked_object_update`.
- **EventMetadataStore:** Populate **only** from `tracked_object_update` (frame_time, box, area, score); use `frigate/events` only for new/end and overlap logic.
- Parse camera from topic `frigate/{camera}/tracked_object_update`; correlate by event id + camera. Normalize box format (e.g. to [ymin, xmin, ymax, xmax]) if Frigate version differs.

### 5. Gemini proxy call and analysis_result.json (bridge to Reporter) — [x] Completed

- **Done in `frigate_buffer/services/ai_analyzer.py`:** HTTP POST in **send_to_proxy** (OpenAI-compatible), system prompt + base64 frames, Bearer auth. Response parsed and returned to orchestrator (buffer does not publish to `frigate/reviews`). After a successful proxy response, **analysis_result.json** is saved in the same directory as the clip (event folder) via **\_save_analysis_result**; this file is the source of truth for the Daily Reporter (Step 7). Required fields for Reporter: `shortSummary`, `title`, `potential_threat_level` (missing fields are logged but the full dict is still saved).
- Integration tests in **tests/test_integration_step_5_6.py** verify: (1) persistence of `analysis_result.json` with expected content, (2) orchestrator hand-off to Frigate and HA, (3) error handling (invalid JSON or 500: no crash, no corrupt file).
- Payload shapes: see **multi_cam_frame_extractor_input_output.md** in this folder (can be moved to `docs/` or kept next to the service).

### 6. Orchestrator integration (optional) — [x] Completed

- **Done:** The main app triggers analysis from the orchestrator/lifecycle when a clip is ready. The orchestrator looks up the event and calls `GeminiAnalysisService.analyze_clip(event_id, clip_path, event_start_ts, event_end_ts)` in a background thread; no separate process or MQTT bridge.

### 7. Daily report script (daily_report_to_proxy.py) — [x] Completed

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

## What was done (Step 2)

- **frigate_buffer/services/ai_analyzer.py:** Smart seeking: `__init__` reads `EXPORT_BUFFER_BEFORE` (buffer_offset), `MAX_MULTI_CAM_FRAMES_SEC`, `MAX_MULTI_CAM_FRAMES_MIN`. `analyze_clip(event_id, clip_path, event_start_ts=0, event_end_ts=0)`; when `event_start_ts > 0`, `_extract_frames` seeks to `fps * buffer_offset`, limits to event duration, and samples at `max_frames_sec` (stride). Paths unchanged; analyzer trusts path from caller.
- **frigate_buffer/orchestrator.py:** In `_on_clip_ready_for_analysis(event_id, clip_path)`, the orchestrator looks up `event = self.state_manager.get_event(event_id)` and passes `event_start_ts`/`event_end_ts` (from `event.created_at`, `event.end_time or event.created_at`) to `analyze_clip`. Lifecycle callback signature remains `(event_id, clip_path)`; no changes to **lifecycle.py**.

---

## What was done (Step 3)

- **frigate_buffer/services/ai_analyzer.py:** Motion-aware selection via **grayscale** frame differencing (no MQTT); `MOTION_THRESHOLD_PX` wired; first and last frames always kept for context. Center crop using `CROP_WIDTH`/`CROP_HEIGHT`. Flat config keys used for proxy URL, model, and tuning (`GEMINI_PROXY_*`). Prompt loaded from `MULTI_CAM_SYSTEM_PROMPT_FILE` when set and valid file; else built-in path or hardcoded default. Proxy request includes `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`.
- **frigate_buffer/orchestrator.py:** Analyzer enable check updated to accept `GEMINI_PROXY_URL` (flat) or `gemini.proxy_url`; API key from config or `GEMINI_API_KEY` env. Single API key; `GEMINI_API_KEY` env overrides config.yaml for security (enforced in config load order).

---

## What was done (Step 4)

- **MQTT:** The app already subscribed to `frigate/+/tracked_object_update`. In **frigate_buffer/orchestrator.py**, `_handle_tracked_update` now handles two payload shapes: (1) `type == "description"` — existing Phase 2 AI description path (set_ai_description, timeline log); (2) per-frame payloads with `id` and `after`/`before` containing `frame_time`, `box`, `area`, `score` — stored only for known events via `state_manager.add_frame_metadata`. Camera is parsed from topic `frigate/<camera>/tracked_object_update`.
- **State:** **frigate_buffer/managers/state.py** has a per-frame metadata store keyed by `event_id`. `_normalize_box(box, frame_width, frame_height)` normalizes Frigate box to `[ymin, xmin, ymax, xmax]` normalized 0–1 (handles pixels or normalized, both orderings). `add_frame_metadata`, `get_frame_metadata`, `clear_frame_metadata`; `remove_event` clears frame metadata to avoid leaks. **frigate_buffer/models.py** adds `FrameMetadata` dataclass (frame_time, box, area, score).
- **AI analyzer:** **frigate_buffer/services/ai_analyzer.py** — `analyze_clip(..., frame_metadata=...)`; `_extract_frames(..., frame_metadata=...)` uses metadata when present: prioritizes frames by `score * area`, matches frame time to closest metadata entry, and applies **smart crop** (crop centered on object box with configurable padding) when box is available; otherwise center crop. `_smart_crop(frame, box, target_w, target_h, padding)` with **SMART_CROP_PADDING** config (default 0.15) for visual context around the subject.
- **Orchestrator wiring:** Clip-ready callback gets `frame_metadata = self.state_manager.get_frame_metadata(event_id)` (same event_id as the clip being analyzed — single or CE primary), passes it to `analyze_clip`, and calls `clear_frame_metadata(event_id)` after analysis.
- **Config:** Optional `SMART_CROP_PADDING` (default 0.15) under `multi_cam` in **frigate_buffer/config.py**.

---

## What was done (Step 5/6)

- **Persistence:** **frigate_buffer/services/ai_analyzer.py** — After a successful proxy response, `analyze_clip` calls **\_save_analysis_result(event_id, clip_path, result)**. The result dict is written as **analysis_result.json** in the same directory as `clip.mp4` (i.e. the event folder, or camera subfolder in consolidated layout). Required for Step 7 (Daily Reporter). Verification: `shortSummary`, `title`, and `potential_threat_level` are checked for presence (warning if missing); full dict is saved in all cases.
- **Orchestrator:** **\_handle_analysis_result** receives the dict from the background thread and calls **download_service.post_event_description(event_id, description)** (Frigate API) and **notifier.publish_notification(..., "finalized")** (Home Assistant). No code changes were required; behavior verified by integration tests.
- **Integration tests:** **tests/test_integration_step_5_6.py** — (1) Persistence: mock proxy success, assert `analysis_result.json` created in event folder with expected keys. (2) Orchestrator: assert `post_event_description` and `publish_notification("finalized")` called when `_handle_analysis_result` runs. (3) Error handling: invalid JSON and proxy 500 — no `analysis_result.json` created, no crash. All five tests pass. Full test suite: 101 passed; 1 pre-existing failure in **tests/test_mqtt_auth.py** (test_mqtt_auth_credentials_set), unrelated to Step 5/6.

---

## What was done (Step 7)

- **frigate_buffer/services/ai_analyzer.py:** Added **send_text_prompt(system_prompt, user_prompt)** — OpenAI-style POST with text-only messages (no images); same `GEMINI_PROXY_URL` and `GEMINI_API_KEY`; returns raw response content string (e.g. Markdown) or `None` on failure.
- **frigate_buffer/services/daily_reporter.py:** New **DailyReporterService**. **generate_report(target_date)** scans `STORAGE_PATH` for `analysis_result.json` in event folders (single: `camera/timestamp_eventid/`; consolidated: `events/timestamp_uuid/camera/`), filters by folder-date, aggregates lines `[{time}] {title}: {shortSummary} (Threat: {level})`, loads report prompt template (default `frigate_buffer/services/report_prompt.txt` or `REPORT_PROMPT_FILE`), replaces `{date}`, `{event_list}`, and optional mock-style placeholders, calls **ai_analyzer.send_text_prompt**, writes Markdown to **{STORAGE_PATH}/daily_reports/{target_date}_report.md**.
- **frigate_buffer/services/report_prompt.txt:** Default template with `{date}` and `{event_list}` (and optional extended placeholders for compatibility).
- **frigate_buffer/config.py:** Schema and load_config: **DAILY_REPORT_SCHEDULE_HOUR** (default 1), **REPORT_PROMPT_FILE** (default ""); env override `DAILY_REPORT_SCHEDULE_HOUR`.
- **frigate_buffer/orchestrator.py:** **DailyReporterService** created when `ai_analyzer` is enabled; **\_daily_report_job** runs at **DAILY_REPORT_SCHEDULE_HOUR** (default 1am) for yesterday; schedule registered in **\_run_scheduler**.
- **Tests:** **tests/test_ai_analyzer.py** — send_text_prompt: success returns content, empty/5xx/missing config return None. **tests/test_daily_reporter.py** — scan (single + consolidated folder date), aggregate format, prompt replacement, save to daily_reports, edge (no events, proxy None). No code changes in mock directory except this plan doc.

---

## Notes for future AI (next steps)

- **Step 2 (Clip paths and Clip timing):** Done. Smart seeking and timestamp wiring are in place; orchestrator does the event lookup so lifecycle stays decoupled.
- **Step 3:** Done. Production logic lives in `ai_analyzer.py`. Motion is **grayscale frame differencing**; first and last frames always kept; crop is center or smart crop when per-frame metadata is available.
- **Step 4:** Done. Per-frame data from `frigate/+/tracked_object_update` is stored in state (normalized box [ymin, xmin, ymax, xmax] 0–1); cleanup on `remove_event` and after `analyze_clip`. Smart crop uses `SMART_CROP_PADDING` (default 0.15). Use the **same event_id** as the clip for frame_metadata (do not merge CE members).
- **Step 5/6:** Done. Persistence: `_save_analysis_result` writes `analysis_result.json` in the clip’s directory (event folder). Orchestrator hand-off confirmed. Integration tests in `tests/test_integration_step_5_6.py` (5 tests, all pass). **Next:** Step 8 (docs and cleanup).
- **Step 7:** Done. DailyReporterService in `frigate_buffer/services/daily_reporter.py`; `send_text_prompt` in ai_analyzer; orchestrator schedule at `DAILY_REPORT_SCHEDULE_HOUR`; report output to `{STORAGE_PATH}/daily_reports/{date}_report.md`; config keys `DAILY_REPORT_SCHEDULE_HOUR`, `REPORT_PROMPT_FILE`. No code changes in mock directory except this plan doc.
- **Config keys reference:** Multi-cam: `MAX_MULTI_CAM_FRAMES_MIN`, `MAX_MULTI_CAM_FRAMES_SEC`, `MOTION_THRESHOLD_PX`, `CROP_WIDTH`, `CROP_HEIGHT`, `MULTI_CAM_SYSTEM_PROMPT_FILE`, `SMART_CROP_PADDING`. Gemini proxy: `GEMINI_PROXY_URL`, `GEMINI_PROXY_MODEL`, `GEMINI_PROXY_TEMPERATURE`, `GEMINI_PROXY_TOP_P`, `GEMINI_PROXY_FREQUENCY_PENALTY`, `GEMINI_PROXY_PRESENCE_PENALTY`. Single API key: `config['GEMINI']['api_key']` (env `GEMINI_API_KEY` overrides config.yaml for security).

---

## Order of work

1. ~~Config schema and load_config~~ **Done.** No Google Fallback, Single API Key (`GEMINI_API_KEY` only).
2. ~~Clip paths and clip timing (vid_start) in **ai_analyzer.py**~~ **Done.** Smart seeking (buffer_offset, seek, limit, stride); orchestrator looks up event and passes timestamps; lifecycle unchanged.
3. ~~Move and wire the service~~ **Done.** Motion (grayscale diff), center crop, flat config, prompt file, proxy tuning in `ai_analyzer.py`; orchestrator enable uses `GEMINI_PROXY_URL` and API key from config/env.
4. ~~Add tracked_object_update subscription and feed EventMetadataStore from it only.~~ **Done.** Per-frame metadata in state; smart crop and score-based selection in ai_analyzer; orchestrator wires get/clear frame_metadata.
5. ~~Proxy HTTP call; save analysis_result.json for Reporter~~ **Done.** `_save_analysis_result` in ai_analyzer.py; integration tests verify persistence and pipeline.
6. ~~Orchestrator hook~~ **Done.** Hand-off to Frigate and HA verified. Update README and clean up mock folder when ready.
7. ~~Daily report~~ **Done.** Native **DailyReporterService** in `frigate_buffer/services/daily_reporter.py`; **send_text_prompt** in ai_analyzer; config `DAILY_REPORT_SCHEDULE_HOUR`, `REPORT_PROMPT_FILE`; orchestrator runs report for yesterday at configured hour; output `{STORAGE_PATH}/daily_reports/{date}_report.md`. Tests in test_daily_reporter.py and test_ai_analyzer.py (send_text_prompt).
