# Multi-cam frame extract mock

Standalone mock for the **multi-cam frame extractor** feature. It is not part of the main frigate-event-buffer app yet.

## Features

- **Multi-cam overlap detection:** Subscribes to `frigate/events` and `frigate/+/tracked_object_update`; when overlapping events end, triggers a single recap for the group.
- **Per-frame metadata:** EventMetadataStore keeps per-event frame_time, box, area, score (from tracked_object_update) and event metadata (label, sub_label, entered_zones, camera) from frigate/events for prompt building and clip correlation.
- **Variable-rate frame extraction:** Selects frames by action score and motion; crops and overlays camera name and timestamp; writes to `multi_cam_frames/` inside the main event folder.
- **Prompt templates:** `multi_cam_system_prompt.txt` and `report_prompt.txt` use `{placeholder}` labels; the scripts fill them from MQTT, config, and (for the report) previous day's recaps. Prompts are externalized so you can tweak the AI's rules or tone without restarting the service or touching code.
- **Bridge between Extractor and Reporter:** The Extractor writes **analysis_result.json** (proxy response: title, scene, confidence, threat_level, etc.) into each event's `multi_cam_frames/` folder. The Reporter scans storage for these files to populate `get_previous_day_recaps()` for the given date. No separate pipeline step required.
- **Daily report (1am):** Compiles the previous day's recap responses (from analysis_result.json files), fills `report_prompt.txt` (report date range, known_person_name, list of event JSON objects), and sends to the Gemini proxy for a single AI report (mock: no HTTP yet; writes placeholder to `daily_reports/`).
- **Config:** MQTT, storage path, proxy URL/key, frame limits, prompt file paths, and `known_person_name` (for the report) via config file or env.

## Contents

| File | Purpose |
|------|--------|
| **multi_cam_frame_extracter.py** | Mock script: MQTT overlap detection, frame extraction, and placeholder for a future Gemini proxy call. Run standalone for idea/flow testing. |
| **multi_cam_system_prompt.txt** | System prompt for the Gemini recap request. Edit this file to change the prompt; the script loads it via `load_system_prompt()`. |
| **daily_report_to_proxy.py** | Mock script: at 1am, compile the previous day's multi-cam recap responses and send them to the proxy for a single AI daily report. Prompt from report_prompt.txt; outputs to daily_reports/. |
| **report_prompt.txt** | System prompt for the daily report. Edit this file to change how the report is written. |
| **multi_cam_frame_extractor_input_output.md** | Expected MQTT and Gemini proxy request/response payloads (as sent/received). |
| **PLAN_TO_MERGE_LATER.md** | Step-by-step plan to merge this into the main project (frame extractor, daily report, config, MQTT, clip paths, proxy, scheduling). |
| **README.md** | This file. |

## Running the mock

From the repo root, with deps installed (e.g. `paho-mqtt`, `opencv-python`, `pyyaml`):

```bash
# Optional: set env for broker and storage
set MQTT_BROKER=REDACTED_LOCAL_IP
set STORAGE_PATH=/app/storage

python multi_cam_frame_extract_mock/multi_cam_frame_extracter.py
```

Config can be provided via a `config.yaml` in the current directory or paths used by the script; see CONFIG_FILE and DEFAULT_CONFIG in the script.

### Daily report (mock)

Run once to generate a mock daily report for yesterday (no proxy call yet; writes placeholder to `storage_path/daily_reports/`):

```bash
python multi_cam_frame_extract_mock/daily_report_to_proxy.py
```

Edit **report_prompt.txt** to change how the daily report is written. When merged, this script will run at 1am (configurable), gather the previous day's recaps from storage, send them to the proxy, and save the AI report.

## Merge

Do **not** merge anything from this folder into the main project until you are ready. When you are, follow **PLAN_TO_MERGE_LATER.md**.

**Step 1 (Config) is done:** The main app’s `frigate_buffer/config.py` and `config.example.yaml` now define `multi_cam` and `gemini_proxy` with flat keys (e.g. `MAX_MULTI_CAM_FRAMES_MIN`, `GEMINI_PROXY_URL`). Defaults live there; no Google fallback, single API key (`GEMINI_API_KEY`). The mock still uses its own `DEFAULT_CONFIG`; when the service is moved (Step 3), it will read from the main config.
