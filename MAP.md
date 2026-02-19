# MAP.md — Context Guide for AI Coding Sessions

A short, scannable map of the Frigate Event Buffer codebase so any AI can grasp architecture and conventions without reading every file.

---

## 1. Project Purpose

**Frigate Buffer** listens to Frigate NVR via MQTT, tracks events through a four-phase lifecycle (NEW → DESCRIBED → FINALIZED → SUMMARIZED), and:

- **Multi-cam ingestion**: Consumes MQTT events from multiple cameras; optional consolidated events; zone/exception filtering via SmartZoneFilter.
- **AI analysis**: Optional Gemini proxy integration—motion-aware frame extraction, smart crop, writes `analysis_result.json`; daily report from aggregated results (requires report prompt file present; no fallback). Report event list: from `daily_reports/aggregate_YYYY-MM-DD.jsonl` when present (appended as analyses complete), else scan for `analysis_result.json` under STORAGE_PATH by calendar date (local); aggregate file deleted after successful report.
- **Web serving**: Flask app for `/player`, `/stats-page`, `/daily-review`, and REST API for events, clips, and snapshots; embeddable in Home Assistant.

Outbound: Ring-style notifications to Home Assistant, clip export/transcode, rolling retention (e.g. 3 days), and export watchdog. DownloadService syncs export_id from GET /api/exports into the timeline so the watchdog can DELETE with the id Frigate expects. The export watchdog logs API responses when present, treats 404/422 as already removed, and logs a run summary (succeeded / failed / already removed). The notifier includes **clear_tag** (previous notification tag) in each payload when the current notification is an update for the same event (same event, not yet ended), so the example HA automation can clear the previous notification before sending the new one; when the last-notified event has ended (CE closed, discarded, or standalone event done), the next notification is treated as a new global event and no clear_tag is sent so multiple events can appear on the phone. Events shorter than **minimum_event_seconds** (config, default 5s) are discarded: data deleted, removed from state/CE, and a "discarded" MQTT notification is published so the example Home Assistant automation can clear the matching phone notification (Companion `clear_notification` by tag).

---


## 2. Architecture Overview

- **Python**: Requires **Python 3.12+** (see `pyproject.toml`: `requires-python = ">=3.12"`).
- **Src layout**: Installable package under `src/frigate_buffer/`. Run with `python -m frigate_buffer.main` after `pip install -e .` (see `pyproject.toml`: `where = ["src"]`). At startup, main logs `VERSION = {version.txt contents}`. Version is read from `src/frigate_buffer/version.txt` (in the package); the Dockerfile needs no special copy because `COPY src/` includes it.
- **Separation**:
  - **Logic in `src/`**: Core package lives in `src/frigate_buffer/` (orchestrator, managers, services, config, models). Only `main.py` is the library entry point.
  - **Web assets**: `src/frigate_buffer/web/` holds the Flask app (`server.py`), `templates/`, and `static/`. The server is created by `create_app(orchestrator)` and closes over the orchestrator.
  - **Entrypoints in `scripts/`**: Main process is started via `python -m frigate_buffer.main`. The optional **standalone** script `scripts/multi_cam_recap.py` runs separately with its own MQTT loop and `process_multi_cam_event`; it is not invoked by the orchestrator.
  - **Deployment**: [INSTALL.md](INSTALL.md) — console-only install and run (clone, layout check, NVENC, config, build, `docker run`); update and troubleshooting included.

---

## 3. Core Data Flow

**Main application (orchestrator-centric)**

- **MQTT** → `MqttClientWrapper` → `StateAwareOrchestrator._on_mqtt_message` in `orchestrator.py`.
- **Event creation/updates**: Orchestrator uses SmartZoneFilter, EventStateManager, ConsolidatedEventManager; delegates event creation and event end to `EventLifecycleService`.
- **Minimum event duration**: When an event ends, if duration < `minimum_event_seconds` (config, default 5), lifecycle discards it: deletes event folder (and CE folder if CE becomes empty), removes from state/CE, and publishes to `frigate/custom/notifications` with `status: "discarded"` and the same `tag` so the example HA automation can clear the phone notification via Companion `clear_notification` by tag.
- **When clip is ready**: Lifecycle invokes the orchestrator’s `on_clip_ready` callback (background thread) → `GeminiAnalysisService.analyze_clip` (frame extraction, Gemini proxy) → result returned to orchestrator → `_handle_analysis_result` updates state, writes files, POSTs description to Frigate API, notifies HA via NotificationPublisher.
- **Multi-camera CE**: When CE has 2+ cameras, lifecycle pipelines clip export: for each camera it runs **download only** (`export_and_download_clip`), then submits **transcode** to a bounded thread pool (`max_concurrent_transcodes`, default 2), and starts the next camera’s download immediately so download and transcode overlap. Transcode (NVENC path) runs **ultralytics** (e.g. YOLOv8-nano) on each frame and writes a **detection sidecar** (`detection.json`) per camera folder. When all transcodes are done, lifecycle calls `on_ce_ready_for_analysis` once → `analyze_multi_clip_ce` uses **multi_clip_extractor**: if every camera has a sidecar, it reads sidecars and picks best camera per time step (no detector); otherwise falls back to OpenCV HOG. Variable rate per `MAX_MULTI_CAM_FRAMES_SEC`/`MAX_MULTI_CAM_FRAMES_MIN`. `_handle_ce_analysis_result` writes summary/metadata to CE root, notifies HA. Writes one `ai_frame_analysis/` + manifest at CE root.
- **Video pipeline**: Decode uses **ffmpegcv** `VideoCaptureNV` (NVDEC) in `ai_analyzer.py`, `multi_cam_recap.py`, and `video.py`. ffmpegcv readers (FFmpegReaderNV, etc.) expose `.fps`, `__len__`/`.count`—**not** OpenCV-style `.get(CAP_PROP_*)`; never call `cap.get()` on ffmpegcv readers. Frame extraction uses a single sequential read because ffmpegcv readers do not support frame-index seek. **Multi-clip extraction** (`multi_clip_extractor.py`) also uses ffmpegcv for decode and a **sequential-read, time-sampled** strategy (no seek); when `VideoCaptureNV` fails, falls back to CPU decode with a log distinguishing "GPU not configured" vs "GPU decode attempted but failed". Transcode (exported clips) uses `VideoWriterNV` (h264_nvenc) then ffmpeg mux for audio; for multi-cam CE the NVENC pass also runs **ultralytics** (YOLO) per frame and writes `detection.json` sidecar per camera. **VideoService** NVENC probe: a **pre-flight probe** runs at startup on the main thread (`run_nvenc_preflight_probe(config)` in `main.py`) and caches the result in the video module; workers then use the cache and do not run a subprocess. Probe resolution is configurable via `multi_cam.nvenc_probe_width` / `nvenc_probe_height` (default 1280×720 to match crop and stay safe for NVENC). If preflight was not run (e.g. tests), the in-process probe is serialized with a lock and uses DEVNULL; on failure the app logs the exact command, extended stderr (longer length for diagnosis), and returncode interpretations (signal/errno/AVERROR). If the probe fails, transcodes use ffmpeg libx264. When libx264 is used and `detection_sidecar_path` and model are set (multi-cam CE), the CPU path also runs YOLO per frame and writes `detection.json` per camera, so the full pipeline (frame extraction, AI request, ai_frame_analysis) completes in CPU-only runs. At startup, **log_gpu_status** (in `video.py`) runs nvidia-smi, checks libnvidia-encode.so, and ffmpeg encoders; NVENC is detected by searching **both stderr and stdout** of `ffmpeg -encoders` (with one retry after a short delay if the first run misses NVENC); **ensure_detection_model_ready** checks whether the detection model is downloaded. Image processing (resize, crop, contours, imencode) remains **OpenCV**. **Docker**: Single `Dockerfile` at repo root. Build from repo root: `docker build -t frigate-buffer:latest .` (see [BUILD_NVENC.md](BUILD_NVENC.md) for NVENC). **Install and run:** [INSTALL.md](INSTALL.md) is the canonical console install (clone, config, build, `docker run`). FFmpeg with NVENC is supplied by a multi-stage Docker build from an image that includes FFmpeg+NVENC (e.g. jrottenberg/ffmpeg:7.0-nvidia2204, which uses standard paths `/usr/local/bin` and `/usr/local/lib`). The final app image is Ubuntu 24.04 to match that donor and avoid NVENC/lib path issues. Run the image with NVIDIA Container Toolkit and `deploy.resources.reservations.devices` (NVIDIA GPU); set `NVIDIA_DRIVER_CAPABILITIES=compute,video,utility` so libnvidia-encode is available at runtime (see BUILD_NVENC.md). The image installs OpenCV runtime libraries (`libgl1`, `libglib2.0-0`, `libxcb1`, etc.) so the headless `cv2` import works. Requires NVIDIA Container Toolkit, GPU reservation, and `YOLO_CONFIG_DIR=/tmp/Ultralytics`.
- **Web / HA**: Flask (`web/server.py`) uses `EventQueryService` to read event/timeline data from disk; serves player, stats, daily review, and API. HA examples are in `examples/home-assistant/` (including automation that clears the phone notification when status is `discarded`).

**Standalone multi-cam script**

- **MQTT** → `scripts/multi_cam_recap.py` (own client and subscriptions) → on linked-event message, spawns thread → `process_multi_cam_event(main_event_id, linked_event_ids)` → frame extraction (motion, crop), optional Gemini call, writes stitched frames/zip via FileManager helpers (`write_stitched_frame`, `create_ai_analysis_zip`). Does not go through the orchestrator.

```mermaid
flowchart LR
  subgraph main [Main App]
    MQTT[MQTT] --> Orch[Orchestrator]
    Orch --> Lifecycle[Lifecycle]
    Lifecycle -->|clip ready| AI[AI Analyzer]
    AI -->|result| Handle[_handle_analysis_result]
    Handle --> State[State]
    Handle --> Files[FileManager]
    Handle --> Frigate[Frigate API]
    Handle --> Notifier[Notifier]
    Orch --> Web[Flask Web]
    Query[EventQueryService] --> Web
  end
```

```mermaid
flowchart LR
  subgraph standalone [Standalone]
    MQTT2[MQTT] --> MultiCam[multi_cam_recap]
    MultiCam --> Process[process_multi_cam_event]
    Process --> Disk[Files / FileManager helpers]
  end
```

---

## 4. The "Big 7" (Power Centers)

| File | Responsibility |
|------|----------------|
| `src/frigate_buffer/orchestrator.py` | Central coordinator: MQTT routing (`_on_mqtt_message`), event/CE handling; wires MqttClientWrapper, SmartZoneFilter, TimelineLogger, managers, lifecycle; registers `on_clip_ready` → ai_analyzer; `_handle_analysis_result`; scheduler (cleanup, export watchdog, daily reporter); Flask app creation; HA state fetch for stats. |
| `src/frigate_buffer/services/ai_analyzer.py` | Gemini proxy integration: motion-aware frame extraction, optional center/smart crop from FrameMetadata, system prompt from file; sends frames to OpenAI-compatible proxy; returns analysis dict; writes `analysis_result.json` (and optional ai_frame_analysis); does not publish to MQTT. For multi-cam CE: `analyze_multi_clip_ce` uses `multi_clip_extractor` (target-centric: reads detection sidecars from transcode when present, else HOG). |
| `src/frigate_buffer/web/server.py` | Flask app factory `create_app(orchestrator)`. Routes: `/player`, `/stats-page`, `/daily-review`, `/api/events`, `/api/events/.../snapshot.jpg`, `/api/files`, `/api/daily-review`, `/api/stats`, `/status`; `/events/<camera>/<subdir>/timeline` (timeline page), `/events/<camera>/<subdir>/timeline/download` (merged timeline JSON as attachment). Uses EventQueryService and `read_timeline_merged`; path safety via file_manager. Player preserves AI Analysis block expand state across auto-refresh. Player shows clip selector when `hosted_clips` has multiple clips. |
| `src/frigate_buffer/managers/file.py` | FileManager: storage paths, clip/snapshot download (via DownloadService), export/transcode coordination, cleanup, path validation (realpath/commonpath). Helpers: `write_stitched_frame`, `write_ai_frame_analysis_single_cam`, `write_ai_frame_analysis_multi_cam`, `create_ai_analysis_zip`; `write_ce_summary`, `write_ce_metadata_json` for CE root; `compute_storage_stats` for legacy + consolidated + daily_reports/daily_reviews. |
| `src/frigate_buffer/services/query.py` | EventQueryService: reads event data from filesystem with TTL and per-folder caching; list events (legacy + consolidated), event_by_id, timeline merge (`read_timeline_merged`). Event dicts include `timestamp` (start); optional `end_timestamp`; `hosted_clips` (list of {camera, url}) for consolidated/multi-clip events; `hosted_clip` only when clip exists (no fallback to missing file). Used by Flask for event lists and stats. |
| `src/frigate_buffer/config.py` | Load and validate config: voluptuous CONFIG_SCHEMA (cameras, network, settings, ha, gemini, multi_cam, gemini_proxy); merge YAML + env + defaults; flat keys for app (e.g. MQTT_BROKER, GEMINI_PROXY_URL, MAX_CONCURRENT_TRANSCODES). Invalid config exits with code 1. |
| `scripts/multi_cam_recap.py` | Standalone entrypoint: uses same config as main app (`frigate_buffer.config.load_config`); same config.yaml and env, including `multi_cam` and `gemini_proxy`. Own MQTT loop and EventMetadataStore; on linked-event message runs `process_multi_cam_event` (frame extract, optional Gemini, write stitched/zip). Uses crop_utils and FileManager helpers. Not started by main orchestrator. |

---

## 5. Testing Philosophy

The project has a substantial test suite in `tests/` (pytest; `pythonpath = ["src"]` in `pyproject.toml`). **All new or changed logic in `src/` should have corresponding updates in `tests/`**—new or modified `test_*.py` as appropriate. Any new behavior or critical path should be covered by tests.

---

## 6. Vibe Rules

- **Strict type hinting**: Use type hints on public functions and important internal APIs.
- **Logic in `src/`, UI in `web/`, execution in `scripts/`**: Core logic stays in `src/frigate_buffer/`; Flask and assets in `web/`; runnable entrypoints (e.g. multi_cam_recap) in `scripts/`.
- **Config**: Use the existing voluptuous schema and flat config dict (see `config.py`). Config is YAML + env, validated at load. **Schema-first**: When adding new features, update `CONFIG_SCHEMA` in `config.py` first so the project stays type-safe and validated.
- **Tests**: Add or update tests in `tests/` for new or changed behavior.
- **Update this file**: When making structural or flow changes to the project, update MAP.md so it remains an accurate context guide for AI sessions.
