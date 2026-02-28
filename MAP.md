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
| **Frameworks** | Flask (web), **Gunicorn** (WSGI; single worker only), paho-mqtt (MQTT), schedule (cron-like jobs) |
| **Config** | YAML + env, validated with Voluptuous |
| **Data / state** | In-memory (EventStateManager, ConsolidatedEventManager); filesystem for events, clips, timelines, daily reports |
| **Video / AI** | **PyNvVideoCodec** (gpu_decoder: NVDEC decode for sidecars, multi-clip extraction, and compilation); compilation **encode** via **FFmpeg h264_nvenc only** (GPU; no CPU fallback). Ultralytics YOLO (detection sidecars), subprocess FFmpeg for GIF and compilation encode; OpenAI-compatible Gemini proxy (HTTP) |
| **Testing** | pytest (`pythonpath = ["src"]`) |

### Video & AI pipeline: zero-copy PyNvVideoCodec/PyTorch (mandatory)

The project uses a **zero-copy GPU pipeline** for all video decode and frame processing in the main path:

- **Decode:** **PyNvVideoCodec** (gpu_decoder) only. Clips are opened with **create_decoder(clip_path, gpu_id)**; frames are decoded directly into PyTorch CUDA tensors (BCHW). There is **no CPU decode fallback** and **no ffmpegcv** anywhere in the codebase. Decoder access is **single-threaded**: the app-wide **GPU_LOCK** (in video.py) is the only lock; it serializes **create_decoder** and every **get_frames** call across video, multi_clip_extractor, and video_compilation.
- **Detection sidecars:** gpu_decoder decode → batched frames via **get_frames(indices)** (batch size 4 to limit VRAM) → float32 [0,1] normalization via in-place `div_(255.0)` → **GPU resize** in _run_detection_on_batch to **DETECTION_IMGSZ** (aspect ratio, multiples of 32) → YOLO → bboxes scaled back to decoder resolution → `_scale_detections_to_native` → `del` batch + aggressive `torch.cuda.empty_cache()` after each batch. No ffmpegcv or subprocess FFmpeg for decode.
- **Frame crops/resize:** All production frame manipulation uses **`crop_utils`** with **PyTorch tensors in BCHW**. NumPy/OpenCV (e.g. `cv2.resize`, `np.ndarray`) are allowed **only at boundaries** (e.g. decoding a single image from the network, or encoding tensor → JPEG for API/disk, or tensor → numpy for FFmpeg rawvideo stdin).
- **Output encoding:** Tensor frames are encoded via `torchvision.io.encode_jpeg` for API (base64) and file writes; compilation **streams** HWC numpy frames to FFmpeg h264_nvenc rawvideo stdin (no in-memory frame list) to avoid RAM spikes.

**Explicit prohibitions (do not reintroduce):**

- **ffmpegcv** — Forbidden. Do not add it to dependencies or use it for decode/capture.
- **CPU-decoding fallbacks** — Forbidden. Do not add fallback paths that decode video on CPU (e.g. OpenCV `VideoCapture`, FFmpeg subprocess for decode, or ffmpegcv). The only remaining FFmpeg use is **GIF generation** (subprocess) and **ffprobe** for metadata.
- **Production frame processing on NumPy in the core path** — Forbidden. New crop/resize logic in the GPU pipeline must use `crop_utils` (BCHW tensors). Legacy NumPy crop helpers were removed from ai_analyzer; production uses `crop_utils`.

---

## 2. Architectural Pattern

- **Design:** **Orchestrator-centric service layer.** One central coordinator (`StateAwareOrchestrator`) owns MQTT routing, event/CE handling, and scheduling; it delegates to managers (state, file, consolidation, zone filter), services (lifecycle, download, notifications, timeline, AI analyzer, daily reporter, export watchdog), and the Flask app.
- **Separation of concerns:**
  - **Logic in `src/`:** Core package is `src/frigate_buffer/` (orchestrator, managers, services, config, models). **Entry:** `run_server.py` at repo root starts Gunicorn (single worker, multi-thread); `main.py` provides `bootstrap()` for the WSGI worker; no Flask built-in server.
  - **Web:** Flask app, templates, and static assets live under `src/frigate_buffer/web/`. The server is created by `create_app(orchestrator)` and closes over the orchestrator; it does not own business logic.
  - **Entrypoints:** Production: `python run_server.py` (loads config for FLASK_HOST/FLASK_PORT, execs Gunicorn with `-w 1 --threads 4`). WSGI app: `frigate_buffer.wsgi:application` (bootstrap, `start_services()`, graceful shutdown on SIGTERM/SIGINT).
  - **API vs UI:** EventQueryService reads from disk; Flask routes call it for event lists, stats, timeline. No API-fetch logic inside templates.
- **Architecture & Performance:** GPU pipeline audit and final verification: see `gpu_pipeline_audit_report.md` (findings and recommendations) and `performance_final_verification.md` (verification status, health grade).

---

## 3. Directory Structure (The Map)

Excludes: `node_modules`, `.git`, `__pycache__`, `.pytest_cache`, build artifacts.

```
frigate-event-buffer/
├── config.yaml
├── config.example.yaml
├── .env.example              # Example env vars (e.g. GOOGLE_APPLICATION_CREDENTIALS for FCM); Docker uses env natively
├── docker-compose.yaml
├── docker-compose.example.yaml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── run_server.py              # Gunicorn launcher: reads config, execvp gunicorn (-w 1, --threads 4)
├── RULE.md
├── map.md
├── MOBILE_API_CONTRACT.md
├── README.md
├── INSTALL.md
├── USER_GUIDE.md
├── MULTI_CAM_PLAN.md
├── DIAGNOSTIC_SIDECAR_TIMELINE_COMPILATION.md  # Diagnostic: sidecar write, timeline EMA, compilation fallback
├── gpu_pipeline_audit_report.md               # GPU performance audit: CPU/GPU boundary, memory, I/O, lock contention
├── performance_final_verification.md          # Final GPU pipeline verification, status table 1.1–4.4, architectural health grade
├── .cursor/
│   └── rules/                                 # Cursor rule files (.mdc); see RULE.md
│
├── scripts/
│   ├── bench_post_download_pre_api.py          # Benchmark: post-download pre-API segment (run with pytest)
│   ├── verify_gemini_proxy.py                  # Manual verification: Gemini proxy connectivity
│   ├── README.md
│   └── scripts_readme.md
│
├── src/
│   └── frigate_buffer/
│       ├── __init__.py
│       ├── main.py            # bootstrap() for wsgi; main() directs to run_server.py
│       ├── wsgi.py             # WSGI entry: bootstrap, start_services(), shutdown hooks, application
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
│       │   ├── zone_filter.py
│       │   └── preferences.py
│       ├── services/
│       │   ├── __init__.py
│       │   ├── ai_analyzer.py
│       │   ├── gemini_proxy_client.py
│       │   ├── gpu_decoder.py
│       │   ├── multi_clip_extractor.py
│       │   ├── timeline_ema.py
│       │   ├── compilation_math.py
│       │   ├── video.py
│       │   ├── lifecycle.py
│       │   ├── download.py
│       │   ├── notifications/
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   ├── dispatcher.py
│       │   │   ├── ADDING_PROVIDERS.md
│       │   │   ├── NOTIFICATION_TIMELINE.md   # When each notification is sent, what it includes, where data comes from
│       │   │   ├── Pushover_Setup.md          # Pushover provider setup and config options
│       │   │   └── providers/
│       │   │       ├── __init__.py
│       │   │       ├── ha_mqtt.py
│       │   │       └── pushover.py            # Pushover API provider (phase filter, priority, attachments)
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
│           │   ├── base.html
│           │   ├── player.html
│           │   ├── test_run.html
│           │   ├── timeline.html
│           │   ├── stats.html
│           │   ├── daily_review.html
│           │   └── test.md              # How to add user-defined tests (sidebar button + display: Log / new bar)
│           └── static/
│               ├── purify.min.js
│               └── marked.min.js
│
├── tests/
│   ├── conftest.py
│   ├── test_ai_analyzer.py
│   ├── test_ai_analyzer_integration.py
│   ├── test_config_schema.py
│   ├── test_constants.py
│   ├── test_consolidation.py
│   ├── test_bootstrap_firebase.py   # Bootstrap Firebase init (mobile_app enabled/disabled, init failure)
│   ├── test_crop_utils.py
│   ├── test_daily_reporter.py
│   ├── test_download_service.py
│   ├── test_event_test.py
│   ├── test_file_manager.py
│   ├── test_frigate_export_watchdog.py
│   ├── test_frigate_proxy.py
│   ├── test_ha_storage_stats.py
│   ├── test_lifecycle_service.py
│   ├── test_logging_utils.py
│   ├── test_main_version.py
│   ├── test_mqtt_auth.py
│   ├── test_mqtt_handler.py
│   ├── test_multi_clip_extractor.py
│   ├── test_notifications.py
│   ├── test_path_helpers.py
│   ├── test_preferences_manager.py   # PreferencesManager + POST /api/mobile/register
│   ├── test_quick_title_service.py
│   ├── test_query_service.py
│   ├── test_report_helpers.py
│   ├── test_state_manager.py
│   ├── test_timeline.py
│   ├── test_timeline_ema.py
│   ├── test_url_masking.py
│   ├── test_video_compilation.py
│   ├── test_video_service.py
│   ├── test_web_server_path_safety.py
│   └── test_zone_filter.py
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
| `config.example.yaml` | Example config with all keys and comments; includes optional **notifications.mobile_app** (enabled, credentials_path) and GOOGLE_APPLICATION_CREDENTIALS note. | Reference only. |
| `.env.example` | Example environment variable names (e.g. GOOGLE_APPLICATION_CREDENTIALS for FCM); Docker uses env natively; copy to .env or set in Compose. | Reference / deployment. |
| `pyproject.toml` | Package metadata, deps, `where = ["src"]`, pytest `pythonpath = ["src"]`, `requires-python = ">=3.12"`. Optional dev deps: pytest, ruff. Tool config: Ruff (lint + format) for `src` and `tests`. E501 (line-too-long) is re-enabled; per-file-ignores list files to fix gradually—remove a file from the list once under 88 chars. | Used by `pip install -e .` and pytest; run `ruff check src tests` and `ruff format src tests` after `pip install -e ".[dev]"`. |
| `requirements.txt` | Pip install list; **no ffmpegcv** (forbidden). **PyNvVideoCodec** from PyPI for zero-copy GPU decode; no vendored wheels. **firebase-admin** for FCM (mobile_app) when notifications.mobile_app enabled. | Referenced by Dockerfile. |
| `Dockerfile` | Single-stage: base `nvidia/cuda:12.6.0-runtime-ubuntu24.04`; installs Python 3.12, FFmpeg from distro (GIF, ffprobe, h264_nvenc encode). **PyNvVideoCodec** from requirements.txt for NVDEC decode; no NeLux or vendored libs. Uses BuildKit (`# syntax=docker/dockerfile:1`); final `pip install .` uses `--mount=type=cache,target=/root/.cache/pip` for faster code-only rebuilds. Build arg `USE_GUI_OPENCV` (default `false`): when `false`, uninstalls opencv-python and reinstalls opencv-python-headless (no X11); when `true`, keeps GUI opencv for faster full rebuilds. **CMD runs `python3 run_server.py`** (Gunicorn launcher; no Flask built-in server). | Build from repo root. |
| `run_server.py` | **Gunicorn launcher** at repo root: loads config via `load_config()`, reads `FLASK_HOST` (default `0.0.0.0`) and `FLASK_PORT`, sets `FRIGATE_BUFFER_SINGLE_WORKER=1`, then `os.execvp` Gunicorn with `-w 1 --threads 4 --capture-output --enable-stdio-inheritance`. Ensures Docker signals go to Gunicorn (PID 1). | Docker CMD; optional local run. |
| `docker-compose.yaml` / `docker-compose.example.yaml` | Compose for local run; GPU, env, mounts. No `YOLO_CONFIG_DIR` needed—app uses storage for Ultralytics config and model cache. | Deployment. |
| `MAP.md` | This file—architecture and context for AI. | Must be updated when structure or flows change. |
| `MOBILE_API_CONTRACT.md` | Source of truth for the native Android (Jetpack Compose + Retrofit) mobile client: all REST endpoints, request/response shapes, query/path params, and media URL construction (clips, snapshots, proxy). | Reference for building the mobile app; do not change without updating the contract. |
| `RULE.md` | Project rules index: points to map.md as source of truth and to .cursor/rules/ for Cursor-specific rules. | Reference only; no code dependencies. |
| `DIAGNOSTIC_SIDECAR_TIMELINE_COMPILATION.md` | Diagnostic: sidecar writing (video.py), timeline_ema usage, and compilation fallback conditions. | Reference for debugging frame extraction and static compilation output. |
| `gpu_pipeline_audit_report.md` | GPU performance audit (no code changes): CPU/GPU boundary, memory, redundant I/O, GPU_LOCK contention. | Reference for optimization and lock refactors. |
| `performance_final_verification.md` | Final GPU pipeline verification: status of audit items 1.1–4.4, silent performance-leak audit, architectural health grade (A-). | Reference for 100% GPU pipeline and performance. |

### Entry & config

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/main.py` | **Bootstrap entry:** Provides `bootstrap() -> (config, StateAwareOrchestrator)` used by `wsgi.py` (load config, setup logging, YOLO dir, GPU check, detection model, create orchestrator). When **notifications.mobile_app** enabled, initializes Firebase Admin SDK (GOOGLE_APPLICATION_CREDENTIALS from config path or env); on missing/invalid credentials logs warning and disables mobile provider. `main()` exits with message to use `run_server.py`; no Flask server run here. | Called by wsgi.py; run_server.py loads config only. |
| `src/frigate_buffer/wsgi.py` | **WSGI entry for Gunicorn:** Requires `FRIGATE_BUFFER_SINGLE_WORKER=1` (set by run_server.py); calls `bootstrap()`, `orchestrator.start_services()`, registers SIGTERM/SIGINT to call `orchestrator.stop()`, exposes `application = orchestrator.flask_app`. Single-worker guardrail raises if env not set. | Loaded by Gunicorn worker; run_server.py execs Gunicorn. | |
| `src/frigate_buffer/config.py` | Load/validate YAML + env via Voluptuous `CONFIG_SCHEMA`; flat keys (e.g. `MQTT_BROKER`, `GEMINI_PROXY_URL`, `MAX_EVENT_LENGTH_SECONDS`). Optional **notifications.mobile_app** (enabled, credentials_path) → `NOTIFICATIONS_MOBILE_APP_ENABLED`, `MOBILE_APP_GOOGLE_APPLICATION_CREDENTIALS`; env `GOOGLE_APPLICATION_CREDENTIALS` overrides path. Frame limits for AI: `multi_cam.max_multi_cam_frames_*` only. Invalid config exits 1. | Used by `main.py`. |
| `src/frigate_buffer/version.txt` | Version string read at startup; logged in main. | Package data; included by `COPY src/` in Dockerfile. |
| `src/frigate_buffer/logging_utils.py` | `setup_logging()`, `ErrorBuffer` for stats dashboard. **`set_suppress_review_debug_logs`** / **`should_suppress_review_debug_logs`**: thread-safe flag so TEST button stream suppresses three MQTT review DEBUG logs (Processing review, Review for … N/A, Skipping finalization). **`StreamCaptureHandler`**: captures log records into a list for the test-run SSE stream; skips records from mqtt_handler/mqtt_client so the test page shows all non-MQTT logs. | Called from main; ErrorBuffer used by web/server and orchestrator; suppress flag set by test_routes, read by mqtt_handler; StreamCaptureHandler used by event_test_orchestrator during run_test_pipeline. |
| `src/frigate_buffer/constants.py` | Shared constants: `NON_CAMERA_DIRS` (includes **saved**—user-kept events folder, excluded from cleanup and normal listing); `HTTP_STREAM_CHUNK_SIZE` (8192), `HTTP_DOWNLOAD_CHUNK_SIZE` (65536); `FRIGATE_PROXY_SNAPSHOT_TIMEOUT` (15), `FRIGATE_PROXY_LATEST_TIMEOUT` (10); `GEMINI_PROXY_QUICK_TITLE_TIMEOUT` (30), `GEMINI_PROXY_ANALYSIS_TIMEOUT` (60); `LOG_MAX_RESPONSE_BODY` (2000), `FRAME_MAX_WIDTH` (1280); `DEFAULT_STORAGE_STATS_MAX_AGE_SECONDS` (30 min); `ERROR_BUFFER_MAX_SIZE` (10); **`NVDEC_INIT_FAILURE_PREFIX`** (log prefix when NVDEC/decoder init fails so crash-loop logs are searchable). **`ZOOM_MIN_FRAME_FRACTION`** (0.4) and **`ZOOM_CONTENT_PADDING`** (0.10) for video compilation dynamic zoom (min crop size and bbox padding). **`COMPILATION_DEFAULT_NATIVE_WIDTH`** (1920), **`COMPILATION_DEFAULT_NATIVE_HEIGHT`** (1080), **`HOLD_CROP_MAX_DISTANCE_SEC`** (5.0), **`ACTION_PREROLL_SEC`** (3.0), **`ACTION_POSTROLL_SEC`** (3.0) for compilation sidecar fallback and slice trimming. Also `is_tensor()` helper for torch.Tensor checks. | Imported by file manager, query, blueprints, frigate_proxy, frigate_export_watchdog, download, ai_analyzer, ha_storage_stats, logging_utils, video, video_compilation, multi_clip_extractor, compilation_math, timeline_ema, gpu_decoder; is_tensor by file, video, crop_utils. |

### Core coordinator & models

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/orchestrator.py` | **StateAwareOrchestrator:** Wires **MqttMessageHandler** (callback to MqttClientWrapper), lifecycle callbacks, `on_ce_ready_for_analysis` → ai_analyzer.analyze_multi_clip_ce, `_handle_analysis_result` (per-event review path) / `_handle_ce_analysis_result`, scheduler (cleanup, export watchdog, daily reporter), `create_app(orchestrator)`. Holds **StorageStatsAndHaHelper** (`stats_helper`) and **EventQueryService** (`query_service`, shared so test_routes can evict test_events cache); exposes `get_storage_stats()` and `fetch_ha_state()` for the stats page. Post-refactor ~487 LOC. | Wires MqttMessageHandler, MqttClientWrapper, SmartZoneFilter, TimelineLogger, all managers, lifecycle, ai_analyzer, Flask, ha_storage_stats, quick_title_service, query_service. |
| `src/frigate_buffer/models.py` | Pydantic/data models: `EventPhase`, `EventState`, `ConsolidatedEvent`, `FrameMetadata`, `NotificationEvent` protocol; helpers for CE IDs and "no concerns". | Used by orchestrator, managers, notifications, query. |

### Managers

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/managers/file.py` | **FileManager:** storage paths, clip/snapshot download (via DownloadService), export coordination (no transcode), cleanup, path validation (realpath/commonpath). `cleanup_old_events`, `rename_event_folder`, `write_canceled_summary`, `compute_storage_stats`, `resolve_clip_in_folder`. **`write_stitched_frame`** accepts numpy HWC BGR or torch.Tensor BCHW/CHW RGB; for tensor uses `torchvision.io.encode_jpeg` and writes bytes (Phase 1 GPU pipeline). | Used by orchestrator, lifecycle, query, download, timeline, event_test. |
| `src/frigate_buffer/managers/state.py` | **EventStateManager:** in-memory event state (phase, metadata), active event tracking. | Orchestrator, lifecycle. |
| `src/frigate_buffer/managers/consolidation.py` | **ConsolidatedEventManager:** CE grouping, `closing` state, `mark_closing`, on_close callback. `schedule_close_timer(ce_id, delay_seconds=None)` — when delay_seconds is set (e.g. 0 for 1-camera CE), uses it instead of event_gap_seconds so CE can close immediately. All events go through CE pipeline. | Orchestrator, lifecycle, timeline_logger, query. |
| `src/frigate_buffer/managers/zone_filter.py` | **SmartZoneFilter:** per-camera zone/exception filters; `should_start_event` uses both **entered_zones** and **current_zones** so events start as soon as the object is in a tracked zone (avoids delayed first notification when Frigate populates zone only in later messages). | Orchestrator (event creation). |
| `src/frigate_buffer/managers/preferences.py` | **PreferencesManager:** thread-safe read/write of `mobile_preferences.json` under storage_path; `get_fcm_token()` / `set_fcm_token(token)` for FCM device token; used by mobile app registration endpoint. | Orchestrator (instantiated at startup); API route POST /api/mobile/register. |

### Services

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/services/ai_analyzer.py` | **GeminiAnalysisService:** uses **GeminiProxyClient** for HTTP; system prompt from file; parses native Gemini and OpenAI responses; returns analysis dict; writes `analysis_result.json`; rolling frame cap. **generate_quick_title** (single image, quick_title_prompt.txt) returns a **dict** with keys `title` and `description` (raw JSON from proxy); used by QuickTitleService for **external_api** snapshot_ready. All analysis via `analyze_multi_clip_ce` when **ai_mode** is **external_api**. | Called by orchestrator, QuickTitleService; uses GeminiProxyClient, VideoService, FileManager. |
| `src/frigate_buffer/services/gemini_proxy_client.py` | **GeminiProxyClient:** HTTP client for Gemini proxy (OpenAI-compatible `/v1/chat/completions`). POST with 2-attempt retry (second attempt: Accept-Encoding identity, Connection close). Used by GeminiAnalysisService for all proxy requests; no response parsing. | Used by ai_analyzer. |
| `src/frigate_buffer/services/gpu_decoder.py` | **GPU decoder service (PyNvVideoCodec):** Sole module that imports PyNvVideoCodec. Wraps **SimpleDecoder** with `use_device_memory=True`, `max_width=4096`, `OutputColorType.RGBP`. Context manager **create_decoder(clip_path, gpu_id)** yields **DecoderContext** with `frame_count`, **get_frames(indices)** → BCHW uint8 RGB tensor (zero-copy via DLPack), **get_frame_at_index**, **get_batch_frames**, **seek_to_index**, **get_index_from_time_in_seconds**. When the library emits UserWarning "Duplicates are not supported" (e.g. stutter in source), it is suppressed from default output and re-logged at DEBUG. On init failure logs **NVDEC_INIT_FAILURE_PREFIX** and re-raises. Callers (video, multi_clip_extractor, video_compilation) must hold **GPU_LOCK** when using the decoder. | Used by video.py, multi_clip_extractor.py, video_compilation.py (Phase 2). |
| `src/frigate_buffer/services/multi_clip_extractor.py` | Target-centric frame extraction for CE; requires detection sidecars (no HOG fallback when any camera lacks sidecar). Camera assignment uses **timeline_ema** (dense grid + EMA + hysteresis + segment merge). **gpu_decoder** (PyNvVideoCodec): one **create_decoder** per camera under **GPU_LOCK**, **get_frames([frame_idx])** per sample time; **ExtractedFrame.frame** is torch.Tensor BCHW RGB; `del` + `torch.cuda.empty_cache()` after each frame. Uses **GPU_LOCK** from video around decoder open and each get_frames. Open failure logs **NVDEC_INIT_FAILURE_PREFIX** and returns []. Parallel sidecar JSON load only. Config: CUDA_DEVICE_INDEX, LOG_EXTRACTION_PHASE_TIMING. | Used by ai_analyzer, event_test_orchestrator. |
| `src/frigate_buffer/services/timeline_ema.py` | Core camera-assignment logic for multi-cam: dense time grid, EMA smoothing, hysteresis, and segment merge (including first-segment roll-forward). Sole path for which camera is chosen per sample time. Also provides **timeline/slice building for compilation**: `convert_timeline_to_segments`, `assignments_to_slices`, `_trim_slices_to_action_window` (uses `ACTION_PREROLL_SEC`, `ACTION_POSTROLL_SEC` from constants). | Used by multi_clip_extractor, video_compilation. |
| `src/frigate_buffer/services/video.py` | **VideoService:** **PyNvVideoCodec decode** via **gpu_decoder** (no sanitizer). `generate_detection_sidecar` opens clip with **create_decoder(clip_path, gpu_id)** under **GPU_LOCK**, gets frame count from `len(ctx)` or ffprobe fallback, batches with **BATCH_SIZE=4** via **ctx.get_frames(indices)** (BCHW uint8), float32/255 then **_run_detection_on_batch** (GPU resize to DETECTION_IMGSZ, aspect ratio, 32-multiple; YOLO; bbox scale-back to decoder space); `del` + `torch.cuda.empty_cache()` after each batch. **GPU_LOCK** serializes decoder open and get_frames app-wide (video_compilation, multi_clip_extractor use it). Decoder open/decode failures log **NVDEC_INIT_FAILURE_PREFIX** and return False. `_decoder_frame_count` and `_decoder_reader_ready` support DecoderContext (used by video service). `get_detection_model_path(config)`, `generate_detection_sidecar`, `generate_detection_sidecars_for_cameras` (shared YOLO + lock), `run_detection_on_image`, `generate_gif_from_clip` (subprocess FFmpeg). Single `_get_video_metadata` ffprobe per clip; results cached in _METADATA_CACHE by path. App-level sidecar lock injected by orchestrator. | Used by lifecycle, ai_analyzer, event_test. |
| `src/frigate_buffer/services/lifecycle.py` | **EventLifecycleService:** event creation, event end (discard short, cancel long). On **new** event (is_new): Phase 1 canned title + `write_metadata_json`, initial notification with live frame via **latest.jpg** proxy; starts quick-title delay thread only when **on_quick_title_trigger** is set (orchestrator sets it only when **ai_mode** is **external_api**). At CE close: export clips, sidecars; when **external_api**, **on_ce_ready_for_analysis** is invoked; when **frigate**, **fetch_review_summary** and **finalized**/**summarized** at CE close; when **external_api**, only **clip_ready** at CE close. | Orchestrator delegates; callbacks gated by config **AI_MODE**. |
| `src/frigate_buffer/services/download.py` | **DownloadService:** Frigate snapshot, export/clip download (dynamic clip names), `post_event_description`. Export flow: match GET /api/exports list by **name** (or path) so `frigate_response["export_id"]` is the list **id** for watchdog DELETE. Export/error logs include file_name_requested and frigate_response. | FileManager, lifecycle, orchestrator. |
| `src/frigate_buffer/services/notifications/` | **Notification system:** **base.py** defines `BaseNotificationProvider`, `NotificationResult`; **dispatcher.py** provides **NotificationDispatcher** (rate limit, queue, `publish_notification`, `send_overflow`, calls `TimelineLogger.log_dispatch_results`); **providers/ha_mqtt.py** provides **HomeAssistantMqttProvider** (HA payload, MQTT publish, clear_tag, `mark_last_event_ended`); **providers/pushover.py** provides **PushoverProvider** (Pushover API, phase filter snapshot_ready/clip_ready/finalized, priority and attachments). **Pushover_Setup.md** documents Pushover config options. Config: `NOTIFICATIONS_HOME_ASSISTANT_ENABLED`, optional **notifications.pushover** block (enabled, user_key, api_token, device, default_sound, html); **ai_mode** (settings.ai_mode / AI_MODE) for two-path behavior. **ADDING_PROVIDERS.md** is the canonical guide for adding new providers. **NOTIFICATION_TIMELINE.md** documents the two **mutually exclusive** paths (Frigate GenAI vs External API), when each notification is sent, what each includes, where data is acquired, and mermaid flowcharts per path. | Orchestrator builds dispatcher via _create_notifier(); lifecycle, mqtt_handler, quick_title_service call notifier.publish_notification. |
| `src/frigate_buffer/services/query.py` | **EventQueryService:** read event data from filesystem with TTL and per-folder caching; `resolve_clip_in_folder`; list events, timeline merge; **`evict_cache(key)`** to invalidate list cache (e.g. `test_events` after Send prompt to AI). **`get_saved_events(camera=None)`** lists events under `saved/` (same dict shape, `saved: True`, file URLs under `/files/saved/...`). **`get_test_events()`** lists only test run events (`events/test1`, `events/test2`, ...), **sorted by folder mtime descending** (newest first); test event **timestamp** = folder content_mtime so player shows correct date; **`_extract_end_timestamp_from_timeline`** returns end_time from Frigate `payload.after.end_time` or from **test-event-only** entries with `source == "test_ai_prompt"` and `data.end_time`. For consolidated events, when `{ce_id}_summary.mp4` exists in the CE root, adds a "Summary video" entry to `hosted_clips` and sets `hosted_clip` to that URL. Excludes non-camera dirs (`NON_CAMERA_DIRS`: ultralytics, yolo_models, daily_reports, daily_reviews, **saved**) from `get_cameras()` and `get_all_events()`. | Flask server (events, stats, player); orchestrator holds shared instance for cache eviction. |
| `src/frigate_buffer/services/quick_title_service.py` | **QuickTitleService:** quick-title pipeline (external_api only): fetch Frigate latest.jpg, YOLO detection, crop_utils crop, **generate_quick_title** (returns dict with title + description), **write_summary** and **write_metadata_json** (Event Viewer), state/CE update, **snapshot_ready** notification with title and description. Used as **on_quick_title_trigger** by lifecycle when orchestrator wires it (AI_MODE == external_api). | Orchestrator instantiates when AI analyzer, QUICK_TITLE_ENABLED, and **AI_MODE == external_api**; lifecycle calls run_quick_title. |
| `src/frigate_buffer/services/daily_reporter.py` | **DailyReporterService:** scheduled; aggregate analysis_result (or daily_reports aggregate JSONL), report prompt, send_text_prompt, write `daily_reports/YYYY-MM-DD_report.md`; `cleanup_old_reports(retention_days)`. Single source for daily report UI. | Scheduled by orchestrator; web server reads markdown from daily_reports/. |
| `src/frigate_buffer/services/frigate_export_watchdog.py` | Parse timeline (base or **append-only**) for export_id, verify clip exists, DELETE Frigate `/api/export/{id}` (uses list **id**, not name); 404/422 = already removed. Run summary when no deletes: folders_with_timeline, export_entries_found, skip reason. All error logs include file_name_requested and frigate_response. | Scheduled by orchestrator. |
| `src/frigate_buffer/services/ha_storage_stats.py` | **StorageStatsAndHaHelper:** storage-stats cache (update from FileManager.compute_storage_stats, get with 30 min TTL) and `fetch_ha_state(ha_url, ha_token, entity_id)` for Home Assistant REST API. Used by orchestrator (scheduler) and Flask stats route. | Orchestrator creates it; server calls `orchestrator.get_storage_stats()` and `orchestrator.fetch_ha_state()`. |
| `src/frigate_buffer/services/timeline.py` | **TimelineLogger:** append HA/MQTT/Frigate API and **notification_dispatch** entries to `notification_timeline.json` via FileManager. **log_dispatch_results(event, status, results)** used by NotificationDispatcher. | Orchestrator, notifications dispatcher. |
| `src/frigate_buffer/services/mqtt_handler.py` | **MqttMessageHandler:** parses MQTT JSON, routes by topic (frigate/events, tracked_object_update, frigate/reviews); implements _handle_frigate_event, _handle_tracked_update, _handle_review. When **AI_MODE** is **external_api**, skips description path in _handle_tracked_update (no set_ai_description, no **described** notification) and returns early from _handle_review (no Frigate GenAI processing). When **should_suppress_review_debug_logs()** is True (TEST stream active), skips three DEBUG logs. Per-event **summarized** (background fetch of review summary) was removed. | Orchestrator builds handler and passes handler.on_message to MqttClientWrapper. |
| `src/frigate_buffer/services/mqtt_client.py` | **MqttClientWrapper:** connect, subscribe, message callback (MqttMessageHandler.on_message). TLS/SSL configured automatically when port is 8883 (cert required, TLS 1.2). | Orchestrator wires handler.on_message as callback. |
| `src/frigate_buffer/services/crop_utils.py` | Crop/resize and motion helpers: accept **PyTorch tensor BCHW only** (GPU pipeline). `center_crop`, `crop_around_center`, **`crop_around_center_to_size`** (crop at variable size then bicubic resize to fixed output; used by video_compilation for dynamic zoom), `full_frame_resize_to_target`, `crop_around_detections_with_padding`, `motion_crop` use tensor slicing and `torch.nn.functional.interpolate`; motion_crop casts to int16 before subtraction to avoid uint8 underflow; only the 1-bit mask is transferred to CPU for `cv2.findContours`. `draw_timestamp_overlay` accepts tensor or numpy, converts RGB→BGR at OpenCV boundary, returns numpy HWC BGR. | ai_analyzer, quick_title_service, multi_clip_extractor, video_compilation. |
| `src/frigate_buffer/services/compilation_math.py` | Pure crop, zoom, and EMA math for video compilation. Detection bbox/center lookup (`_nearest_entry_at_t`, `_nearest_entry_with_detections_at_t`), content area and weighted center (`_content_area_and_center`, `_weighted_center_from_detections`), zoom crop size (`_zoom_crop_size`), `calculate_crop_at_time`, `calculate_segment_crop`, `smooth_zoom_ema`, `smooth_crop_centers_ema`. No I/O, no video_compilation import. Uses constants (ZOOM_*, HOLD_CROP_MAX_DISTANCE_SEC). | Used by video_compilation (imports and re-exports for tests). |
| `src/frigate_buffer/services/video_compilation.py` | Video compilation orchestrator. Imports crop/zoom/EMA from **compilation_math** and timeline/slice building from **timeline_ema**; re-exports for tests. **compile_ce_video** scans CE dir, calls **_load_sidecars_for_cameras** (unified sidecar loader with COMPILATION_DEFAULT_* fallback), uses timeline_ema for build_dense_times, build_phase1_assignments, assignments_to_slices, _trim_slices_to_action_window, then **generate_compilation_video**. Generates a stitched, cropped, 20fps summary video; uses the **same timeline config** as the frame timeline. Slices are **trimmed to an action window** (first/last detection ± pre/post roll). **Dynamic zoom** when TRACKING_TARGET_FRAME_PERCENT > 0; **smooth_zoom_ema** and **smooth_crop_centers_ema** from compilation_math. **PyNvVideoCodec decode** per slice via gpu_decoder under **GPU_LOCK**; **_run_pynv_compilation** streams to FFmpeg h264_nvenc stdin. **_open_compilation_ffmpeg_process** and **_close_compilation_ffmpeg_and_check** shared by list-frames and streaming encode paths. **_log_stutter_once** for one INFO per camera on stutter/missing frames. 20fps, no audio, output MP4. | Used by lifecycle, orchestrator. |
| `src/frigate_buffer/services/report_prompt.txt` | Default prompt for daily report. | daily_reporter. |
| `src/frigate_buffer/services/ai_analyzer_system_prompt.txt` | System prompt for Gemini proxy (multi-clip CE analysis). | ai_analyzer. |
| `src/frigate_buffer/services/quick_title_prompt.txt` | System prompt for quick-title (single image; expects raw JSON with "title" and "description"). | ai_analyzer. |

### Event test (TEST button only)

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/event_test/__init__.py` | Exports `run_test_pipeline`. | Web server calls it for TEST button. |
| `src/frigate_buffer/event_test/event_test_orchestrator.py` | **prepare_test_folder**: allocates `events/testN`, copies source only (no sidecars). **run_test_pipeline** (full) and **run_test_pipeline_from_folder** (post-copy only on existing test folder) delegate to VideoService (sidecars) and same multi-clip extractor. **get_export_time_range_from_folder**: derives (start, end) from folder timeline for Frigate export (video-request). Attaches **StreamCaptureHandler** during runs so test-run page streams non-MQTT logs. | Used only for TEST button; production logic in ai_analyzer and multi_clip_extractor. |

### Web

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/web/frigate_proxy.py` | **proxy_snapshot**, **proxy_camera_latest:** stream Frigate snapshot/latest.jpg; return Flask Response or (body, status). Validates camera name and allowed_cameras; 503 when Frigate URL not set, 502 on request failure. | Used by proxy_routes blueprint. |
| `src/frigate_buffer/web/path_helpers.py` | **resolve_under_storage(storage_path, \*path_parts):** returns normalized absolute path iff it lies strictly under the real storage root; otherwise None. Single place for web path-safety checks (no path traversal). | Used by api and test_routes blueprints; by report_helpers for get_report_for_date. |
| `src/frigate_buffer/web/report_helpers.py` | **daily_reports_dir**, **list_report_dates**, **get_report_for_date:** list/read daily report markdown from daily_reports/ (YYYY-MM-DD_report.md). Path safety via resolve_under_storage. | Used by daily_review blueprint. |
| `src/frigate_buffer/web/server.py` | **create_app(orchestrator):** creates Flask app, registers before_request (request count), registers blueprints (pages, proxy, daily_review, api, test), returns app. No route definitions; thin shell. | Imports and registers blueprints from web.routes. |
| `src/frigate_buffer/web/routes/` | **Blueprints:** `pages` (player, stats-page, daily-review, test-multi-cam), `proxy_routes` (snapshot, latest.jpg), `daily_review` (api/daily-review/*), **api** (cameras, events with **filter=saved** / **filter=test_events**, **POST /keep/<path:event_path>**, delete, **viewed/<path:event_path>**, **events/<path:event_path>/timeline**, files, stats, status, **POST /api/mobile/register**; uses **orchestrator.query_service**, **orchestrator.preferences_manager**), `test_routes` (api/test-multi-cam/**prepare**, **event-data** (GET, returns timeline + event_files for test run), **ai-payload** (GET, prompt + image_urls), **stream**, **video-request**, **send** POST; uses path_helpers, event_test_orchestrator, query.read_timeline_merged and resolve_clip_in_folder). Each module exposes create_bp(orchestrator); routes close over orchestrator. | Registered by server.create_app. |
| `src/frigate_buffer/web/templates/*.html` | **Templates:** **base.html** (shared layout; sidebar, nav, blocks title/head/content/scripts), **player.html** (Events Player), **test_run.html** (Test multi-cam page), **timeline.html** (event timeline), **stats.html** (Stats Dashboard), **daily_review.html** (Daily Review). Jinja2 templates extend base.html. **base.html:** Two-line sidebar title "Frigate Event / Viewer" (left-aligned); nav has Events Player (nested sidebarPlayerExtras on /player), Stats, Daily Review, Test (nested sidebarTestExtras on /test-multi-cam). **Player:** Filter toolbar includes Unreviewed, All Events, Reviewed, Saved, Test events (shown when current event is consolidated). Event page: video, nav/clip selector, **Mark Reviewed bar** (Click to Review left, Mark All Reviewed right), **action bar under video** (Timeline, Delete, Keep, Download—even stretch; Keep only if not saved, Download only if has_clip). No TEST button or action buttons in sidebar; sidebar player extras empty. Timeline link label is "Timeline" (not "View Timeline"). AI Analysis and Event Details cards below. **Test run:** Page supports URL params `subdir` (from player TEST link) and `test_run` (persisted after prepare). When no test_run/subdir: shows **"Select an event to Test"** grid (last 8 consolidated events from /events?filter=all); click runs prepare then shows test view and sets URL to ?test_run=. When ?test_run= in URL: restores that test (GET event-data, no prepare). **Reset Test** sidebar button clears URL and shows grid. Log header has no three-dot decoration. After event selected: video block, clip selector tabs, log, sidebar (Start the Test, Video Request, View AI Request, Send prompt to AI when AI payload present). **Full-width collapsible bars** below log (order: Timeline, Event files, Prompt, Images, Return prompt); all start collapsed; text bodies 4-line max-height with scroll, Images 1-row grid with scroll. GET **/api/test-multi-cam/event-data?test_run=** returns timeline + event_files for bars. **Timeline** template shows entries including test_ai_prompt for test runs. Event paths: saved uses saved/camera/subdir for delete, viewed, timeline. | Rendered by Flask. |
| `src/frigate_buffer/web/templates/test.md` | **Reference for adding user-defined tests** to the Test page. Documents: sidebar placement (nested under Test, above Reset Test), current schema and patterns for nested buttons (#sidebarTestExtras, btnClass, button order), how the user specifies display (Log, new horizontal bar, or both), and how to add a new full-width collapsible bar. Includes implementation notes for a future AI (backend route, frontend renderSidebar, persistence, Reset Test). **Use this file when the user wants to add a test to the test button or Test page.** | See test_run.html, base.html, test_routes. |
| `src/frigate_buffer/web/static/*.js` | DOMPurify, Marked (min). | Served by Flask. |

### Tests

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `tests/conftest.py` | Pytest fixtures. | All tests. |
| `tests/test_*.py` | Unit tests for config, lifecycle, ai_analyzer (including proxy retry and frame analysis writing), ai_analyzer_integration (persistence, orchestrator handoff, error handling), **bootstrap Firebase** (mobile_app enabled/disabled, init failure, env from config), video, video_compilation, multi_clip_extractor, gpu_decoder, query (including caching and read_timeline_merged), **notifications** (dispatcher, HA provider, event compliance), download, file manager (path validation, cleanup, max event length, storage stats, timeline append), crop_utils, consolidation, daily_reporter, frigate_export_watchdog, ha_storage_stats, timeline, mqtt, state, zone_filter, etc. Tensor mocks in test_ai_analyzer and test_ai_analyzer_integration for write_ai_frame_analysis_multi_cam and analyze_multi_clip_ce with tensor ExtractedFrame; test_multi_clip_extractor for sequential get_frames. | Run with `pytest tests/`; `pythonpath = ["src"]`. Benchmark and verify scripts live in `scripts/`. |

---

## 5. Core Flows & Lifecycles

### Main application (orchestrator-centric)

1. **Startup:** `run_server.py` loads config (FLASK_HOST, FLASK_PORT), sets `FRIGATE_BUFFER_SINGLE_WORKER=1`, then **execvp** Gunicorn with `-w 1 --threads 4`. Gunicorn worker loads `frigate_buffer.wsgi:application` → **bootstrap()** (config, logging, YOLO, GPU, orchestrator) → **orchestrator.start_services()** (MQTT, notifier, scheduler) → **application** = orchestrator.flask_app. SIGTERM/SIGINT in worker call **orchestrator.stop()** for graceful shutdown.
2. **MQTT → Event creation/updates:** MQTT → `MqttClientWrapper` → **MqttMessageHandler.on_message** → SmartZoneFilter (should_start) → EventStateManager / ConsolidatedEventManager → **EventLifecycleService** (event creation, event end).
3. **Quick-title (new event, external_api only):** When **ai_mode** is **external_api** and a new event starts with quick_title enabled, lifecycle sets canned title, writes metadata, sends initial notification with **latest.jpg** proxy. A delay thread then calls **QuickTitleService.run_quick_title**: fetch `latest.jpg`, YOLO, crop, **generate_quick_title** (returns JSON with title + description) → update state/CE, **write_summary** and **write_metadata_json** (Event Viewer), notify **snapshot_ready** with same tag. When **ai_mode** is **frigate**, on_quick_title_trigger is not wired so quick-title does not run.
4. **Event end (short):** Lifecycle checks duration &lt; `minimum_event_seconds` → discard: delete folder, remove from state/CE, publish MQTT `status: "discarded"` with same tag for HA clear.
5. **Event end (long/canceled):** Duration ≥ `max_event_length_seconds` → no clip export/AI; write canceled summary, notify HA, rename folder `-canceled`; cleanup later by retention.
6. **Consolidated event (all events):** Every event is a CE. At CE close: Lifecycle exports clips, **generate_detection_sidecars_for_cameras**. When **ai_mode** is **external_api**, **on_ce_ready_for_analysis** is wired → **analyze_multi_clip_ce** → **\_handle_ce_analysis_result** → write CE summary/metadata, notify **finalized**. When **ai_mode** is **frigate**, on_ce_ready_for_analysis is not wired. Then (Frigate mode only) **fetch_review_summary**, write review summary; build notify target; send **clip_ready**; (Frigate only) if best title/desc, send **finalized**; (Frigate only) if summary and not no concerns, send **summarized**. **external_api** mode sends only **clip_ready** at CE close (no Frigate review summary, no finalized/summarized there).
7. **Frigate review path (frigate mode only):** When **ai_mode** is **frigate**, **\_handle_review** processes MQTT `frigate/reviews` (update state, write files, notify **finalized**). When **external_api**, _handle_review returns immediately. **\_handle_tracked_update** description path (set_ai_description, **described** notification) runs only when not **external_api**. Per-event summarized (background thread fetching review summary) was removed globally.
8. **Web:** Flask (create_app registers blueprints) uses **EventQueryService** to list events, stats, timeline; `resolve_clip_in_folder` for dynamic clip URLs; path safety via path_helpers (web) and FileManager. Daily report UI reads markdown from `daily_reports/`; POST `/api/daily-review/generate` triggers on-demand report generation.
9. **Scheduled:** Cleanup (retention), export watchdog (DELETE completed exports), daily reporter (aggregate + report prompt → proxy → markdown; then `cleanup_old_reports`). Config: **quick_title_delay_seconds**, **quick_title_enabled**, **ai_mode** (`frigate` | `external_api`, default `external_api`). See **NOTIFICATION_TIMELINE.md** for the two-path flows and mermaid diagrams.

**Files touched in primary flow:** `orchestrator.py`, `services/mqtt_handler.py`, `services/lifecycle.py`, `services/quick_title_service.py`, `services/ai_analyzer.py`, `services/mqtt_client.py`, `managers/state.py`, `managers/file.py`, `managers/consolidation.py`, `services/notifications/` (dispatcher, providers/ha_mqtt), `services/download.py`, `services/query.py`, `services/timeline.py`, `web/server.py`.

---

## 6. AI Agent Directives (Rules & Conventions)

### Zero-copy GPU pipeline (mandatory)

- **Do not** add or use **ffmpegcv** for video decode or capture.
- **Do not** add **CPU-decoding fallbacks** (e.g. OpenCV VideoCapture or FFmpeg subprocess for decode) for the main pipeline; PyNvVideoCodec (gpu_decoder) is the only decode path. FFmpeg is allowed only for GIF generation and ffprobe metadata.
- **Production frame crops/resize** must use **`crop_utils`** with **BCHW tensors**. Do not add new NumPy/OpenCV-based crop or resize logic in the core frame path; use the existing tensor helpers in `crop_utils.py`. Legacy NumPy crop helpers were removed from ai_analyzer; use crop_utils for all production crops.

### File placement rules

| Need | Location |
|------|----------|
| New **UI component / page** | Add route in the appropriate blueprint under `src/frigate_buffer/web/routes/` (e.g. pages.py); template in `src/frigate_buffer/web/templates/`; static assets in `src/frigate_buffer/web/static/`. |
| New **user-defined test** (Test page) | See **`src/frigate_buffer/web/templates/test.md`** for sidebar button placement (nested under Test, above Reset Test), display options (Log, new horizontal bar, or both), and patterns. |
| New **business logic / service** | `src/frigate_buffer/services/` (or `managers/` if it is state/aggregation). Register and call from `orchestrator.py` (or from an existing service) as appropriate. |
| New **utility function** (generic, no I/O) | `src/frigate_buffer/services/` (e.g. `crop_utils.py`) or a new module under `services/` if it fits a clear domain. |
| New **API route** (REST) | Add in the appropriate blueprint under `src/frigate_buffer/web/routes/` (e.g. api.py or daily_review.py); use EventQueryService or FileManager for data; never put business logic in route handlers beyond delegation. |
| New **config key** | Add to **CONFIG_SCHEMA** in `src/frigate_buffer/config.py` first; then add flat key in config merge; use in code via `config.get('KEY', default)`. |
| New **notification provider** | Add provider under `src/frigate_buffer/services/notifications/providers/`; implement `BaseNotificationProvider`; register in config and `orchestrator._create_notifier()`. **See `services/notifications/ADDING_PROVIDERS.md`** for the provider contract, flow diagram, and mock examples. |
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

- **Ruff (lint & format):** The project uses **Ruff** for linting and formatting (`pyproject.toml`). When **creating or editing** any file in `src/` or `tests/`, follow Ruff’s rules: run `ruff check src tests` and `ruff format src tests` (or `python -m ruff check/format` if ruff is not on PATH). Line length is 88 characters (E501); fix or wrap long lines. Files listed in `[tool.ruff.lint.per-file-ignores]` are temporarily exempt from E501 until fixed—remove a file from that list once it complies. Do not introduce new lint/format violations.
- **Type hints:** Use type hints on all public function signatures and important internal APIs. Prefer Python 3.10+ syntax (e.g. `str | None`).
- **Config:** Always extend **CONFIG_SCHEMA** in `config.py` for new options; invalid config must exit with code 1.
- **State / side effects:** Core state lives in EventStateManager and ConsolidatedEventManager; services and FileManager are stateless or hold minimal caches (e.g. EventQueryService TTL cache).
- **Error handling:** Validate paths with FileManager (realpath/commonpath); log and handle Frigate/HTTP errors; do not crash the orchestrator on individual event failures.
- **Comments / docstrings:** Docstrings should explain *why* for non-obvious logic; use a consistent style (e.g. Google or NumPy). Prefer early returns over deep nesting.
- **Tests:** All new or changed logic in `src/` must have corresponding tests in `tests/`; keep tests simple (Setup → Execute → Verify).
- **Map maintenance:** When you add/remove files, change core flows, or rename important components, **update MAP.md** so it stays the single source of truth for AI context.
- **Constants**  hard coded values go in Constsnts.py  no magic numbers should be created in code

---

*End of map.md*
