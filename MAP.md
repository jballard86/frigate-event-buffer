# map.md — Project Holy Grail

**Primary context document for AI agents.** Read this before executing any task. It defines the project's logic, architecture, file locations, naming conventions, and data flows.

---

## 1. Project Overview & Core Logic

### Summary

**Frigate Event Buffer** is a state-aware orchestrator that:

- Listens to **Frigate NVR** via MQTT and tracks events through a **four-phase lifecycle**: NEW → DESCRIBED → FINALIZED → SUMMARIZED.
- Sends **Ring-style sequential notifications** to Home Assistant and maintains a **configurable rolling evidence locker** (default 3 days).
- Optionally runs **AI analysis** (Gemini proxy): motion-aware frame extraction, smart crop, writes `analysis_result.json`; daily report from aggregated results.
- Serves a **Flask web app** for `/player`, `/stats-page`, `/daily-review`, and REST API for events, clips, and snapshots—embeddable in Home Assistant.

**Target audience:** Home automation users running Frigate 0.17+ who want event buffering, HA notifications, optional GenAI review, and a built-in event viewer.

**Core business logic:**

- **Multi-cam ingestion:** Consumes MQTT events from configured cameras; optional consolidated events; zone/exception filtering via SmartZoneFilter.
- **Event lifecycle:** Creation → clip export/download → (optional) AI analysis → state/files/Frigate API/HA notification; short events discarded, long events canceled (no AI, folder `-canceled`).
- **Outbound:** HA notifications (with `clear_tag` for updates), clip export/transcode-free storage, rolling retention, export watchdog (DELETE completed exports from Frigate).

### Primary Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.12+ |
| **Package** | setuptools, src layout (`src/frigate_buffer/`) |
| **Frameworks** | Flask (web), paho-mqtt (MQTT), schedule (cron-like jobs) |
| **Config** | YAML + env, validated with Voluptuous |
| **Data / state** | In-memory (EventStateManager, ConsolidatedEventManager); filesystem for events, clips, timelines, daily reports |
| **Video / AI** | **NeLux** (vendored wheel: NVDEC decode for sidecars and compilation); compilation **encode** via **FFmpeg h264_nvenc only** (GPU; no CPU fallback). Ultralytics YOLO (detection sidecars), subprocess FFmpeg for GIF and compilation encode; OpenAI-compatible Gemini proxy (HTTP) |
| **Testing** | pytest (`pythonpath = ["src"]`) |

### Video & AI pipeline: zero-copy NeLux/PyTorch (mandatory)

The project uses a **zero-copy GPU pipeline** for all video decode and frame processing in the main path:

- **Decode:** **NeLux** (NVDEC) only. Frames are decoded directly into PyTorch CUDA tensors (BCHW). There is **no CPU decode fallback** and **no ffmpegcv** anywhere in the codebase.
- **Detection sidecars:** NeLux decode → batched frames → float32 [0,1] normalization → YOLO → `del` batch + `torch.cuda.empty_cache()` after each batch. No ffmpegcv or subprocess FFmpeg for decode.
- **Frame crops/resize:** All production frame manipulation uses **`crop_utils`** with **PyTorch tensors in BCHW**. NumPy/OpenCV (e.g. `cv2.resize`, `np.ndarray`) are allowed **only at boundaries** (e.g. decoding a single image from the network, or encoding tensor → JPEG for API/disk).
- **Output encoding:** Tensor frames are encoded via `torchvision.io.encode_jpeg` for API (base64) and file writes; no CPU round-trip for the main pipeline.

**Explicit prohibitions (do not reintroduce):**

- **ffmpegcv** — Forbidden. Do not add it to dependencies or use it for decode/capture.
- **CPU-decoding fallbacks** — Forbidden. Do not add fallback paths that decode video on CPU (e.g. OpenCV `VideoCapture`, FFmpeg subprocess for decode, or ffmpegcv) when NeLux is used for the main pipeline. The only remaining FFmpeg use is **GIF generation** (subprocess) and **ffprobe** for metadata.
- **Production frame processing on NumPy in the core path** — Forbidden. New crop/resize logic in the GPU pipeline must use `crop_utils` (BCHW tensors). Legacy NumPy helpers in ai_analyzer (`_center_crop`, `_smart_crop`) are deprecated; production must use `crop_utils`.

---

## 2. Architectural Pattern

- **Design:** **Orchestrator-centric service layer.** One central coordinator (`StateAwareOrchestrator`) owns MQTT routing, event/CE handling, and scheduling; it delegates to managers (state, file, consolidation, zone filter), services (lifecycle, download, notifier, timeline, AI analyzer, daily reporter, export watchdog), and the Flask app.
- **Separation of concerns:**
  - **Logic in `src/`:** Core package is `src/frigate_buffer/` (orchestrator, managers, services, config, models). Only `main.py` is the library entry point; run with `python -m frigate_buffer.main`.
  - **Web:** Flask app, templates, and static assets live under `src/frigate_buffer/web/`. The server is created by `create_app(orchestrator)` and closes over the orchestrator; it does not own business logic.
  - **Entrypoints:** Main process via `python -m frigate_buffer.main`.
  - **API vs UI:** EventQueryService reads from disk; Flask routes call it for event lists, stats, timeline. No API-fetch logic inside templates.

---

## 3. Directory Structure (The Map)

Excludes: `node_modules`, `.git`, `__pycache__`, `.pytest_cache`, build artifacts.

```
frigate-event-buffer/
├── config.yaml
├── config.example.yaml
├── docker-compose.yaml
├── docker-compose.example.yaml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── wheels/
│   ├── Create_NeLux_Wheel.md                      # Build NeLux wheel from source (Docker, FFmpeg 6.1, CUDA)
│   ├── Update_NeLux_Wheel.md                      # Steps to update to a new NeLux version
│   ├── Dockerfile.nelux                            # Builder image for compiling the NeLux wheel
│   ├── NeLux_troubleshooting.md                    # Runtime/Docker troubleshooting: hollow VideoReader, ldd, LD_LIBRARY_PATH, NVDEC
│   └── nelux-0.8.9-cp312-cp312-linux_x86_64.whl   # vendored; do not use PyPI
├── MAP.md
├── README.md
├── INSTALL.md
├── USER_GUIDE.md
├── MULTI_CAM_PLAN.md
│
├── scripts/
│   (reserved for standalone scripts)
│
├── src/
│   └── frigate_buffer/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── models.py
│       ├── logging_utils.py
│       ├── constants.py
│       ├── version.txt
│       ├── orchestrator.py
│       ├── event_test/
│       │   ├── __init__.py
│       │   └── event_test_orchestrator.py
│       ├── managers/
│       │   ├── __init__.py
│       │   ├── file.py
│       │   ├── state.py
│       │   ├── consolidation.py
│       │   └── zone_filter.py
│       ├── services/
│       │   ├── __init__.py
│       │   ├── ai_analyzer.py
│       │   ├── multi_clip_extractor.py
│       │   ├── timeline_ema.py
│       │   ├── video.py
│       │   ├── lifecycle.py
│       │   ├── download.py
│       │   ├── notifier.py
│       │   ├── query.py
│       │   ├── daily_reporter.py
│       │   ├── frigate_export_watchdog.py
│       │   ├── ha_storage_stats.py
│       │   ├── timeline.py
│       │   ├── video_compilation.py
│       │   ├── mqtt_handler.py
│       │   ├── mqtt_client.py
│       │   ├── crop_utils.py
│       │   ├── quick_title_service.py
│       │   ├── report_prompt.txt
│       │   └── ai_analyzer_system_prompt.txt
│       └── web/
│           ├── __init__.py
│           ├── frigate_proxy.py
│           ├── path_helpers.py
│           ├── report_helpers.py
│           ├── server.py
│           ├── routes/
│           │   ├── __init__.py
│           │   ├── api.py
│           │   ├── daily_review.py
│           │   ├── pages.py
│           │   ├── proxy_routes.py
│           │   └── test_routes.py
│           ├── templates/
│           │   ├── player.html
│           │   ├── test_run.html
│           │   ├── timeline.html
│           │   ├── stats.html
│           │   └── daily_review.html
│           └── static/
│               ├── purify.min.js
│               └── marked.min.js
│
├── tests/
│   ├── conftest.py
│   ├── test_ai_analyzer.py
│   ├── test_multi_clip_extractor.py
│   ├── test_timeline_ema.py
│   ├── test_video_service.py
│   ├── test_config_schema.py
│   ├── test_notifier_clear_tag.py
│   ├── test_daily_reporter.py
│   ├── test_lifecycle_service.py
│   ├── test_timeline.py
│   ├── test_ai_frame_analysis_writing.py
│   ├── test_max_event_length.py
│   ├── test_cleanup_test_folders.py
│   ├── test_constants.py
│   ├── test_query_service.py
│   ├── test_quick_title_service.py
│   ├── test_event_test.py
│   ├── test_download_service.py
│   ├── test_crop_utils.py
│   ├── test_web_server_path_safety.py
│   ├── test_frigate_export_watchdog.py
│   ├── test_ha_storage_stats.py
│   ├── test_integration_step_5_6.py
│   ├── test_notification_models.py
│   ├── test_consolidation.py
│   ├── test_file_manager_path_validation.py
│   ├── test_storage_stats.py
│   ├── test_ai_analyzer_proxy_fix.py
│   ├── test_mqtt_auth.py
│   ├── test_mqtt_handler.py
│   ├── test_state_manager.py
│   ├── test_zone_filter.py
│   ├── test_url_masking.py
│   ├── test_query_caching.py
│   ├── test_main_version.py
│   ├── test_optimization_expectations_temp.py
│   ├── bench_post_download_pre_api.py
│   └── verify_gemini_proxy.py
│
└── examples/
    └── home-assistant/
        ├── Home Assistant Notification Automation.yaml
        └── Home Assistant Notification Dashboard.yaml
```

---

## 4. File Registry & Descriptions

### Root

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `config.yaml` | User config: cameras, network, settings, HA, Gemini, multi_cam. | Read by `config.load_config()` at startup. |
| `config.example.yaml` | Example config with all keys and comments. | Reference only. |
| `pyproject.toml` | Package metadata, deps, `where = ["src"]`, pytest `pythonpath = ["src"]`, `requires-python = ">=3.12"`. | Used by `pip install -e .` and pytest. |
| `requirements.txt` | Pip install list; **no ffmpegcv** (forbidden). Includes vendored NeLux wheel from `wheels/` (do not use PyPI). Zero-copy GPU pipeline only. | Referenced by Dockerfile. |
| `wheels/` | Vendored NeLux wheel (`nelux-0.8.9-cp312-cp312-linux_x86_64.whl`) for zero-copy GPU compilation; built against FFmpeg 6.1 / Ubuntu 24.04. Vendored C++ shared libraries (`libyuv.so*`, `libspdlog.so*`) from the NeLux builder are tracked (`.gitignore` exceptions) so clone-and-build works; Dockerfile copies them into `/usr/lib/x86_64-linux-gnu/` so runtime ABI matches the wheel. Python 3.12; torch must be imported before nelux at runtime. Frame count is obtained from the reader when possible; if the reader's `len()` fails (e.g. missing `_decoder`), we fall back to ffprobe metadata (duration × fps). **Wheel build/update:** `wheels/Create_NeLux_Wheel.md` (build from source), `wheels/Update_NeLux_Wheel.md` (update version), `wheels/Dockerfile.nelux` (builder image). **Runtime/Docker issues:** `wheels/NeLux_troubleshooting.md` (hollow VideoReader, ldd, LD_LIBRARY_PATH, NVDEC in Docker). | Copied into image; wheel installed via requirements.txt; .so libs copied to container lib path. |
| `Dockerfile` | Single-stage: base `nvidia/cuda:12.6.0-runtime-ubuntu24.04`; installs Python 3.12, FFmpeg 6.1 from distro; copies vendored libyuv and libspdlog (`wheels/libyuv.so*`, `wheels/libspdlog.so*`) into `/usr/lib/x86_64-linux-gnu/` for NeLux ABI match; sets `LD_LIBRARY_PATH` so NeLux native deps load at runtime (no distro libyuv/libspdlog packages or symlinks). NeLux wheel from `wheels/`; FFmpeg 6.1 and vendored libs required at runtime. Uses BuildKit (`# syntax=docker/dockerfile:1`); final `pip install .` uses `--mount=type=cache,target=/root/.cache/pip` for faster code-only rebuilds. Build arg `USE_GUI_OPENCV` (default `false`): when `false`, uninstalls opencv-python and reinstalls opencv-python-headless (no X11); when `true`, keeps GUI opencv for faster full rebuilds. Runs `python3 -m frigate_buffer.main`. | Build from repo root. |
| `docker-compose.yaml` / `docker-compose.example.yaml` | Compose for local run; GPU, env, mounts. No `YOLO_CONFIG_DIR` needed—app uses storage for Ultralytics config and model cache. | Deployment. |
| `MAP.md` | This file—architecture and context for AI. | Must be updated when structure or flows change. |

### Entry & config

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/main.py` | Entry point: load config, setup logging/signals, set `YOLO_CONFIG_DIR` to `STORAGE_PATH/ultralytics` and create `yolo_models` dir (before any Ultralytics import), GPU check, ensure detection model, create and start `StateAwareOrchestrator`. | Calls `config.load_config()`, `orchestrator.start()`. |
| `src/frigate_buffer/config.py` | Load/validate YAML + env via Voluptuous `CONFIG_SCHEMA`; flat keys (e.g. `MQTT_BROKER`, `GEMINI_PROXY_URL`, `MAX_EVENT_LENGTH_SECONDS`). Frame limits for AI: `multi_cam.max_multi_cam_frames_*` only. `single_camera_ce_close_delay_seconds` (0 = close single-cam CE as soon as event ends). Invalid config exits 1. | Used by `main.py`. |
| `src/frigate_buffer/version.txt` | Version string read at startup; logged in main. | Package data; included by `COPY src/` in Dockerfile. |
| `src/frigate_buffer/logging_utils.py` | `setup_logging()`, `ErrorBuffer` for stats dashboard. | Called from main; ErrorBuffer used by web/server and orchestrator. |
| `src/frigate_buffer/constants.py` | Shared constants: `NON_CAMERA_DIRS`; `HTTP_STREAM_CHUNK_SIZE` (8192), `HTTP_DOWNLOAD_CHUNK_SIZE` (65536); `FRIGATE_PROXY_SNAPSHOT_TIMEOUT` (15), `FRIGATE_PROXY_LATEST_TIMEOUT` (10); `LOG_MAX_RESPONSE_BODY` (2000), `FRAME_MAX_WIDTH` (1280); `DEFAULT_STORAGE_STATS_MAX_AGE_SECONDS` (30 min); `ERROR_BUFFER_MAX_SIZE` (10). Also `is_tensor()` helper for torch.Tensor checks. | Imported by file manager, query, blueprints, frigate_proxy, frigate_export_watchdog, download, ai_analyzer, ha_storage_stats, logging_utils; is_tensor by file, video, crop_utils. |

### Core coordinator & models

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/orchestrator.py` | **StateAwareOrchestrator:** Wires **MqttMessageHandler** (callback to MqttClientWrapper), lifecycle callbacks, `on_ce_ready_for_analysis` → ai_analyzer.analyze_multi_clip_ce, `_handle_analysis_result` (per-event review path) / `_handle_ce_analysis_result`, scheduler (cleanup, export watchdog, daily reporter), `create_app(orchestrator)`. Holds **StorageStatsAndHaHelper** (`stats_helper`); exposes `get_storage_stats()` and `fetch_ha_state()` for the stats page. Post-refactor ~487 LOC. | Wires MqttMessageHandler, MqttClientWrapper, SmartZoneFilter, TimelineLogger, all managers, lifecycle, ai_analyzer, Flask, ha_storage_stats, quick_title_service. |
| `src/frigate_buffer/models.py` | Pydantic/data models: `EventPhase`, `EventState`, `ConsolidatedEvent`, `FrameMetadata`, `NotificationEvent` protocol; helpers for CE IDs and "no concerns". | Used by orchestrator, managers, notifier, query. |

### Managers

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/managers/file.py` | **FileManager:** storage paths, clip/snapshot download (via DownloadService), export coordination (no transcode), cleanup, path validation (realpath/commonpath). `cleanup_old_events`, `rename_event_folder`, `write_canceled_summary`, `compute_storage_stats`, `resolve_clip_in_folder`. **`write_stitched_frame`** accepts numpy HWC BGR or torch.Tensor BCHW/CHW RGB; for tensor uses `torchvision.io.encode_jpeg` and writes bytes (Phase 1 GPU pipeline). | Used by orchestrator, lifecycle, query, download, timeline, event_test. |
| `src/frigate_buffer/managers/state.py` | **EventStateManager:** in-memory event state (phase, metadata), active event tracking. | Orchestrator, lifecycle. |
| `src/frigate_buffer/managers/consolidation.py` | **ConsolidatedEventManager:** CE grouping, `closing` state, `mark_closing`, on_close callback. `schedule_close_timer(ce_id, delay_seconds=None)` — when delay_seconds is set (e.g. 0 for single-camera CE), uses it instead of event_gap_seconds so CE can close immediately. | Orchestrator, lifecycle, timeline_logger, query. |
| `src/frigate_buffer/managers/zone_filter.py` | **SmartZoneFilter:** per-camera zone/exception filters; `should_start_event` uses both **entered_zones** and **current_zones** so events start as soon as the object is in a tracked zone (avoids delayed first notification when Frigate populates zone only in later messages). | Orchestrator (event creation). |

### Services

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/services/ai_analyzer.py` | **GeminiAnalysisService:** system prompt from file; POST to OpenAI-compatible proxy; parses both **native Gemini** and OpenAI-shaped responses; returns analysis dict; writes `analysis_result.json`; rolling frame cap. **Phase 4:** `_frame_to_base64_url` and `send_to_proxy` accept numpy HWC BGR or torch.Tensor BCHW RGB; `generate_quick_title` accepts numpy or tensor. **Legacy NumPy helpers** `_center_crop` and `_smart_crop` are deprecated (logger.warning); production must use crop_utils (BCHW). All analysis via `analyze_multi_clip_ce`; **generate_quick_title** (single image, quick_title_prompt.txt) for 3–6 word title shortly after event start. | Called by orchestrator (`on_ce_ready_for_analysis`) and QuickTitleService (quick-title); uses VideoService, FileManager. |
| `src/frigate_buffer/services/multi_clip_extractor.py` | Target-centric frame extraction for CE; requires detection sidecars (no HOG fallback when any camera lacks sidecar). Camera assignment uses **timeline_ema** (dense grid + EMA + hysteresis + segment merge). **NeLux** decode: one VideoReader per camera, single-threaded **get_batch** per sample time; **ExtractedFrame.frame** is torch.Tensor BCHW RGB; `del` + `torch.cuda.empty_cache()` after each frame. No ffmpegcv/CPU fallback; parallel sidecar JSON load only. Config: CUDA_DEVICE_INDEX, LOG_EXTRACTION_PHASE_TIMING. | Used by ai_analyzer, event_test_orchestrator. |
| `src/frigate_buffer/services/timeline_ema.py` | Core camera-assignment logic for multi-cam: dense time grid, EMA smoothing, hysteresis, and segment merge (including first-segment roll-forward). Sole path for which camera is chosen per sample time. | Used by multi_clip_extractor. |
| `src/frigate_buffer/services/video.py` | **VideoService:** **NeLux** NVDEC decode for detection sidecars (one `VideoReader` per clip, batched `get_batch` with **BATCH_SIZE=16**, float32/255 normalization for YOLO; `del` + `torch.cuda.empty_cache()` after each batch). No ffmpegcv/CPU decode; NeLux open/get_batch failures log and return False. If reader is missing `_decoder`, monkey-patch `reader._decoder = reader` so existing code works. `_nelux_frame_count(reader, fps, duration)` gets frame count from reader or ffprobe fallback when reader's `len()` fails. `get_detection_model_path(config)` (model under `STORAGE_PATH/yolo_models/`), `generate_detection_sidecar`, `generate_detection_sidecars_for_cameras` (shared YOLO + lock), `run_detection_on_image` (numpy or tensor, normalized for YOLO; quick-title), `generate_gif_from_clip` (subprocess FFmpeg—only remaining FFmpeg use). Single `_get_video_metadata` ffprobe per clip. App-level sidecar lock injected by orchestrator. | Used by lifecycle, ai_analyzer, event_test. |
| `src/frigate_buffer/services/lifecycle.py` | **EventLifecycleService:** event creation, event end (discard short, cancel long). On **new** event (is_new): Phase 1 canned title + `write_metadata_json`, initial notification with live frame via **latest.jpg** proxy (no snapshot download); starts quick-title delay thread (calls `on_quick_title_trigger` = QuickTitleService.run_quick_title). At CE close: export clips, sidecars, then `on_ce_ready_for_analysis`. | Orchestrator delegates; calls download, file_manager, video_service, orchestrator callbacks (`on_ce_ready_for_analysis`, `on_quick_title_trigger`). |
| `src/frigate_buffer/services/download.py` | **DownloadService:** Frigate snapshot, export/clip download (dynamic clip names), `post_event_description`. | FileManager, lifecycle, orchestrator. |
| `src/frigate_buffer/services/notifier.py` | **NotificationPublisher:** publish to `frigate/custom/notifications`; `clear_tag` for updates; timeline_callback = TimelineLogger.log_ha. | Orchestrator, lifecycle. |
| `src/frigate_buffer/services/query.py` | **EventQueryService:** read event data from filesystem with TTL and per-folder caching; `resolve_clip_in_folder`; list events, timeline merge. Excludes non-camera dirs (`NON_CAMERA_DIRS`: ultralytics, yolo_models, daily_reports, daily_reviews) from `get_cameras()` and `get_all_events()`. | Flask server (events, stats, player). |
| `src/frigate_buffer/services/quick_title_service.py` | **QuickTitleService:** quick-title pipeline: fetch Frigate latest.jpg, YOLO detection, crop_utils crop, AI title via GeminiAnalysisService.generate_quick_title, state/metadata/CE update, notify. Used as `on_quick_title_trigger` by lifecycle. | Orchestrator instantiates when AI analyzer and QUICK_TITLE_ENABLED; lifecycle calls run_quick_title. |
| `src/frigate_buffer/services/daily_reporter.py` | **DailyReporterService:** scheduled; aggregate analysis_result (or daily_reports aggregate JSONL), report prompt, send_text_prompt, write `daily_reports/YYYY-MM-DD_report.md`; `cleanup_old_reports(retention_days)`. Single source for daily report UI. | Scheduled by orchestrator; web server reads markdown from daily_reports/. |
| `src/frigate_buffer/services/frigate_export_watchdog.py` | Parse timeline for export_id, verify clip exists, DELETE Frigate `/api/export/{id}`; 404/422 = already removed. | Scheduled by orchestrator. |
| `src/frigate_buffer/services/ha_storage_stats.py` | **StorageStatsAndHaHelper:** storage-stats cache (update from FileManager.compute_storage_stats, get with 30 min TTL) and `fetch_ha_state(ha_url, ha_token, entity_id)` for Home Assistant REST API. Used by orchestrator (scheduler) and Flask stats route. | Orchestrator creates it; server calls `orchestrator.get_storage_stats()` and `orchestrator.fetch_ha_state()`. |
| `src/frigate_buffer/services/timeline.py` | **TimelineLogger:** append HA/MQTT/Frigate API entries to `notification_timeline.json` via FileManager. | Orchestrator, notifier (timeline_callback). |
| `src/frigate_buffer/services/mqtt_handler.py` | **MqttMessageHandler:** parses MQTT JSON, routes by topic (frigate/events, tracked_object_update, frigate/reviews); implements _handle_frigate_event, _handle_tracked_update, _handle_review, _fetch_and_store_review_summary. | Orchestrator builds handler and passes handler.on_message to MqttClientWrapper. |
| `src/frigate_buffer/services/mqtt_client.py` | **MqttClientWrapper:** connect, subscribe, message callback (MqttMessageHandler.on_message). | Orchestrator wires handler.on_message as callback. |
| `src/frigate_buffer/services/crop_utils.py` | Crop/resize and motion helpers: accept **PyTorch tensor BCHW only** (GPU pipeline). `center_crop`, `crop_around_center`, `full_frame_resize_to_target`, `crop_around_detections_with_padding`, `motion_crop` use tensor slicing and `torch.nn.functional.interpolate`; motion_crop casts to int16 before subtraction to avoid uint8 underflow; only the 1-bit mask is transferred to CPU for `cv2.findContours`. `draw_timestamp_overlay` accepts tensor or numpy, converts RGB→BGR at OpenCV boundary, returns numpy HWC BGR. | ai_analyzer, quick_title_service, multi_clip_extractor. |
| `src/frigate_buffer/services/video_compilation.py` | Video compilation service. Generates a stitched, cropped, 20fps summary video for a CE lifecycle; uses the **same timeline config** as the frame timeline (MAX_MULTI_CAM_FRAMES_SEC, MAX_MULTI_CAM_FRAMES_MIN) and **one slice per assignment** so camera swapping matches the AI prompt timeline. Crop follows the tracked object with **smooth panning** (tensor crop with 0-based `t/duration` interpolation; optional EMA on crop center). On the **last slice of a camera run** (slice before a camera switch), crop is **held** (`crop_end = crop_start`). **NeLux** (vendored wheel): NVDEC decode per slice into PyTorch CUDA tensors, tensor crop (smooth pan). **Encode:** FFmpeg subprocess with **h264_nvenc only** (GPU; no CPU fallback); rawvideo stdin; on failure, descriptive error logging (command, returncode, stderr). Reader monkey-patch: if `_decoder` missing, set `reader._decoder = reader`. 20fps, no audio, output MP4. **Import order:** `import torch` before `from nelux import VideoReader` (required by NeLux). | Used by lifecycle, orchestrator. |
| `src/frigate_buffer/services/report_prompt.txt` | Default prompt for daily report. | daily_reporter. |
| `src/frigate_buffer/services/ai_analyzer_system_prompt.txt` | System prompt for Gemini proxy (multi-clip CE analysis). | ai_analyzer. |
| `src/frigate_buffer/services/quick_title_prompt.txt` | System prompt for quick-title (single image, 3–6 word title only). | ai_analyzer. |

### Event test (TEST button only)

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/event_test/__init__.py` | Exports `run_test_pipeline`. | Web server calls it for TEST button. |
| `src/frigate_buffer/event_test/event_test_orchestrator.py` | Allocates `events/testN`, copies source, delegates to VideoService (sidecars) and same multi-clip extractor; no YOLO/lock in this module. | Used only for TEST button; production logic in ai_analyzer and multi_clip_extractor. |

### Web

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/web/frigate_proxy.py` | **proxy_snapshot**, **proxy_camera_latest:** stream Frigate snapshot/latest.jpg; return Flask Response or (body, status). Validates camera name and allowed_cameras; 503 when Frigate URL not set, 502 on request failure. | Used by proxy_routes blueprint. |
| `src/frigate_buffer/web/path_helpers.py` | **resolve_under_storage(storage_path, \*path_parts):** returns normalized absolute path iff it lies strictly under the real storage root; otherwise None. Single place for web path-safety checks (no path traversal). | Used by api and test_routes blueprints; by report_helpers for get_report_for_date. |
| `src/frigate_buffer/web/report_helpers.py` | **daily_reports_dir**, **list_report_dates**, **get_report_for_date:** list/read daily report markdown from daily_reports/ (YYYY-MM-DD_report.md). Path safety via resolve_under_storage. | Used by daily_review blueprint. |
| `src/frigate_buffer/web/server.py` | **create_app(orchestrator):** creates Flask app, registers before_request (request count), registers blueprints (pages, proxy, daily_review, api, test), returns app. No route definitions; thin shell. | Imports and registers blueprints from web.routes. |
| `src/frigate_buffer/web/routes/` | **Blueprints:** `pages` (player, stats-page, daily-review, test-multi-cam), `proxy_routes` (snapshot, latest.jpg), `daily_review` (api/daily-review/*), `api` (cameras, events, delete, viewed, timeline, files, stats, status), `test_routes` (api/test-multi-cam/*). Each module exposes create_bp(orchestrator); routes close over orchestrator. | Registered by server.create_app. |
| `src/frigate_buffer/web/templates/*.html` | Jinja2 templates for player, stats, daily report, timeline, test run. Player's AI Analysis tile: first line = event title (canned or quick AI) or "future screenshot analysis"; then scene. | Rendered by Flask. |
| `src/frigate_buffer/web/static/*.js` | DOMPurify, Marked (min). | Served by Flask. |

### Tests

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `tests/conftest.py` | Pytest fixtures. | All tests. |
| `tests/test_*.py` | Unit tests for config, orchestrator, lifecycle, ai_analyzer, video, query, notifier, download, file manager, cleanup, etc. **Phase 5:** tensor mocks in test_ai_frame_analysis_writing (write_ai_frame_analysis_multi_cam with tensor frames), test_integration_step_5_6 (analyze_multi_clip_ce with tensor ExtractedFrame), test_multi_clip_extractor (sequential get_batch: one call per extracted frame). | Run with `pytest tests/`; `pythonpath = ["src"]`. |

---

## 5. Core Flows & Lifecycles

### Main application (orchestrator-centric)

1. **Startup:** `main.py` → `load_config()` → `StateAwareOrchestrator(config)` → `orchestrator.start()` (MQTT connect, Flask, schedule jobs).
2. **MQTT → Event creation/updates:** MQTT → `MqttClientWrapper` → **MqttMessageHandler.on_message** → SmartZoneFilter (should_start) → EventStateManager / ConsolidatedEventManager → **EventLifecycleService** (event creation, event end).
3. **Quick-title (new event, optional):** When a new event starts and quick_title is enabled, lifecycle sets canned title ("Motion Detected on [Camera]"), writes metadata, sends initial notification with **latest.jpg** proxy image (no Frigate snapshot download). A delay thread (3–5s) then calls **QuickTitleService.run_quick_title** (on_quick_title_trigger): fetch Frigate `latest.jpg`, run **VideoService.run_detection_on_image** (under YOLO lock), **crop_around_detections_with_padding** (master bbox over all detections + 10%), **generate_quick_title** → update state/metadata/CE, notify again with same tag (clear_tag) so HA replaces the notification.
4. **Event end (short):** Lifecycle checks duration &lt; `minimum_event_seconds` → discard: delete folder, remove from state/CE, publish MQTT `status: "discarded"` with same tag for HA clear.
5. **Event end (long/canceled):** Duration ≥ `max_event_length_seconds` → no clip export/AI; write canceled summary, notify HA, rename folder `-canceled`; cleanup later by retention.
6. **Consolidated event (all events):** Every event is a CE (single- or multi-camera). At CE close: Lifecycle exports each camera clip, **VideoService.generate_detection_sidecars_for_cameras** (all cameras, including single) → `on_ce_ready_for_analysis` → **analyze_multi_clip_ce** (multi_clip_extractor with timeline_ema as sole camera-assignment logic) → `_handle_ce_analysis_result` → write summary/metadata at CE root, notify HA. Single-camera CE uses same pipeline (camera count 1).
7. **Frigate review path (per-event only):** `_handle_review` used when GenAI data arrives via MQTT `frigate/reviews` (update state, write files, notify). Daily summary is no longer fetched from Frigate; the daily report page is fed only from **DailyReporterService** output (`daily_reports/YYYY-MM-DD_report.md`).
8. **Web:** Flask (create_app registers blueprints) uses **EventQueryService** to list events, stats, timeline; `resolve_clip_in_folder` for dynamic clip URLs; path safety via path_helpers (web) and FileManager. Daily report UI reads markdown from `daily_reports/`; POST `/api/daily-review/generate` triggers on-demand report generation.
9. **Scheduled:** Cleanup (retention), export watchdog (DELETE completed exports), daily reporter (aggregate + report prompt → proxy → markdown; then `cleanup_old_reports`). Config: **quick_title_delay_seconds** (e.g. 4), **quick_title_enabled** (when true and Gemini enabled, run quick-title pipeline).

**Files touched in primary flow:** `orchestrator.py`, `services/mqtt_handler.py`, `services/lifecycle.py`, `services/quick_title_service.py`, `services/ai_analyzer.py`, `services/mqtt_client.py`, `managers/state.py`, `managers/file.py`, `managers/consolidation.py`, `services/notifier.py`, `services/download.py`, `services/query.py`, `web/server.py`.

---

## 6. AI Agent Directives (Rules & Conventions)

### Zero-copy GPU pipeline (mandatory)

- **Do not** add or use **ffmpegcv** for video decode or capture.
- **Do not** add **CPU-decoding fallbacks** (e.g. OpenCV VideoCapture or FFmpeg subprocess for decode) for the main pipeline; NeLux (NVDEC) is the only decode path. FFmpeg is allowed only for GIF generation and ffprobe metadata.
- **Production frame crops/resize** must use **`crop_utils`** with **BCHW tensors**. Do not add new NumPy/OpenCV-based crop or resize logic in the core frame path; use the existing tensor helpers in `crop_utils.py`. Legacy NumPy helpers in ai_analyzer (`_center_crop`, `_smart_crop`) are deprecated and must not be used for new production code.

### Wheel build and update

When there are issues with the **NeLux wheel** (making, updating, or rebuilding it), use the docs in **`wheels/`**:

| Need | File |
|------|------|
| Build wheel from source (first time or clean build) | `wheels/Create_NeLux_Wheel.md` |
| Update to a new NeLux version | `wheels/Update_NeLux_Wheel.md` |
| Builder image definition | `wheels/Dockerfile.nelux` |
| Runtime/Docker troubleshooting (hollow VideoReader, ldd, LD_LIBRARY_PATH, NVDEC) | `wheels/NeLux_troubleshooting.md` |

Do not use PyPI for NeLux; the project vendors the wheel from `wheels/`.

### File placement rules

| Need | Location |
|------|----------|
| New **UI component / page** | Add route in the appropriate blueprint under `src/frigate_buffer/web/routes/` (e.g. pages.py); template in `src/frigate_buffer/web/templates/`; static assets in `src/frigate_buffer/web/static/`. |
| New **business logic / service** | `src/frigate_buffer/services/` (or `managers/` if it is state/aggregation). Register and call from `orchestrator.py` (or from an existing service) as appropriate. |
| New **utility function** (generic, no I/O) | `src/frigate_buffer/services/` (e.g. `crop_utils.py`) or a new module under `services/` if it fits a clear domain. |
| New **API route** (REST) | Add in the appropriate blueprint under `src/frigate_buffer/web/routes/` (e.g. api.py or daily_review.py); use EventQueryService or FileManager for data; never put business logic in route handlers beyond delegation. |
| New **config key** | Add to **CONFIG_SCHEMA** in `src/frigate_buffer/config.py` first; then add flat key in config merge; use in code via `config.get('KEY', default)`. |
| New **standalone script** | `scripts/` at repo root. |
| **Tests** | `tests/test_<module_or_feature>.py`; mirror structure under `src/frigate_buffer/` where it helps. |

### Naming conventions

| Element | Convention | Example |
|---------|-------------|---------|
| **Python modules** | `snake_case.py` | `ai_analyzer.py`, `multi_clip_extractor.py` |
| **Classes** | PascalCase | `StateAwareOrchestrator`, `EventLifecycleService` |
| **Functions / methods** | snake_case | `_on_mqtt_message`, `resolve_clip_in_folder` |
| **Constants / config keys** | UPPER_SNAKE_CASE | `STORAGE_PATH`, `MAX_EVENT_LENGTH_SECONDS` |
| **Templates** | lowercase with underscores | `player.html`, `daily_review.html` |
| **Test files** | `test_<name>.py` | `test_ai_analyzer.py`, `test_lifecycle_service.py` |

### Coding standards

- **Type hints:** Use type hints on all public function signatures and important internal APIs. Prefer Python 3.10+ syntax (e.g. `str | None`).
- **Config:** Always extend **CONFIG_SCHEMA** in `config.py` for new options; invalid config must exit with code 1.
- **State / side effects:** Core state lives in EventStateManager and ConsolidatedEventManager; services and FileManager are stateless or hold minimal caches (e.g. EventQueryService TTL cache).
- **Error handling:** Validate paths with FileManager (realpath/commonpath); log and handle Frigate/HTTP errors; do not crash the orchestrator on single-event failures.
- **Comments / docstrings:** Docstrings should explain *why* for non-obvious logic; use a consistent style (e.g. Google or NumPy). Prefer early returns over deep nesting.
- **Tests:** All new or changed logic in `src/` must have corresponding tests in `tests/`; keep tests simple (Setup → Execute → Verify).
- **Map maintenance:** When you add/remove files, change core flows, or rename important components, **update MAP.md** so it stays the single source of truth for AI context.

---

*End of map.md*
