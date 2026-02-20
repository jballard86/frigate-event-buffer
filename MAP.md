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
| **Video / AI** | ffmpegcv (NVDEC decode), Ultralytics YOLO (detection sidecars), OpenAI-compatible Gemini proxy (HTTP) |
| **Testing** | pytest (`pythonpath = ["src"]`) |

---

## 2. Architectural Pattern

- **Design:** **Orchestrator-centric service layer.** One central coordinator (`StateAwareOrchestrator`) owns MQTT routing, event/CE handling, and scheduling; it delegates to managers (state, file, consolidation, zone filter, reviews), services (lifecycle, download, notifier, timeline, AI analyzer, daily reporter, export watchdog), and the Flask app.
- **Separation of concerns:**
  - **Logic in `src/`:** Core package is `src/frigate_buffer/` (orchestrator, managers, services, config, models). Only `main.py` is the library entry point; run with `python -m frigate_buffer.main`.
  - **Web:** Flask app, templates, and static assets live under `src/frigate_buffer/web/`. The server is created by `create_app(orchestrator)` and closes over the orchestrator; it does not own business logic.
  - **Entrypoints:** Main process via `python -m frigate_buffer.main`. Optional **standalone** script `scripts/multi_cam_recap.py` runs separately (own MQTT loop, `process_multi_cam_event`); not invoked by the orchestrator.
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
├── entrypoint.sh
├── pyproject.toml
├── requirements.txt
├── MAP.md
├── README.md
├── INSTALL.md
├── INSTALL_READABLE.md
├── USER_GUIDE.md
├── BUILD_NVENC.md
├── COMPOSE_ENTRYPOINT.md
├── MULTI_CAM_PLAN.md
│
├── scripts/
│   └── multi_cam_recap.py
│
├── src/
│   └── frigate_buffer/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── models.py
│       ├── logging_utils.py
│       ├── version.txt
│       ├── orchestrator.py
│       ├── event_test/
│       │   ├── __init__.py
│       │   └── event_test_orchestrator.py
│       ├── managers/
│       │   ├── __init__.py
│       │   ├── file.py
│       │   ├── state.py
│       │   ├── reviews.py
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
│       │   ├── timeline.py
│       │   ├── mqtt_client.py
│       │   ├── crop_utils.py
│       │   ├── report_prompt.txt
│       │   └── ai_analyzer_system_prompt.txt
│       └── web/
│           ├── __init__.py
│           ├── server.py
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
│   ├── test_query_service.py
│   ├── test_event_test.py
│   ├── test_download_service.py
│   ├── test_crop_utils.py
│   ├── test_web_server_path_safety.py
│   ├── test_multi_cam_recap_config.py
│   ├── test_frigate_export_watchdog.py
│   ├── test_integration_step_5_6.py
│   ├── test_notification_models.py
│   ├── test_consolidation.py
│   ├── test_file_manager_path_validation.py
│   ├── test_storage_stats.py
│   ├── test_ai_analyzer_proxy_fix.py
│   ├── test_mqtt_auth.py
│   ├── test_state_manager.py
│   ├── test_zone_filter.py
│   ├── test_url_masking.py
│   ├── test_query_caching.py
│   ├── test_main_version.py
│   ├── test_optimization_expectations_temp.py
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
| `requirements.txt` | Pip install list. | Referenced by Dockerfile. |
| `Dockerfile` | Multi-stage build; FFmpeg/NVDEC; runs `python -m frigate_buffer.main`. | Build from repo root. |
| `docker-compose.yaml` / `docker-compose.example.yaml` | Compose for local run; GPU, env, mounts. | Deployment. |
| `entrypoint.sh` | Container entrypoint. | Docker run. |
| `MAP.md` | This file—architecture and context for AI. | Must be updated when structure or flows change. |

### Entry & config

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/main.py` | Entry point: load config, setup logging/signals, GPU check, ensure detection model, create and start `StateAwareOrchestrator`. | Calls `config.load_config()`, `orchestrator.start()`. |
| `src/frigate_buffer/config.py` | Load/validate YAML + env via Voluptuous `CONFIG_SCHEMA`; flat keys (e.g. `MQTT_BROKER`, `GEMINI_PROXY_URL`, `MAX_EVENT_LENGTH_SECONDS`). Invalid config exits 1. | Used by `main.py`, `multi_cam_recap.py`. |
| `src/frigate_buffer/version.txt` | Version string read at startup; logged in main. | Package data; included by `COPY src/` in Dockerfile. |
| `src/frigate_buffer/logging_utils.py` | `setup_logging()`, `ErrorBuffer` for stats dashboard. | Called from main; ErrorBuffer used by web/server and orchestrator. |

### Core coordinator & models

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/orchestrator.py` | **StateAwareOrchestrator:** MQTT routing (`_on_mqtt_message`), event/CE handling, lifecycle callbacks, `on_clip_ready` → ai_analyzer, `_handle_analysis_result` / `_handle_ce_analysis_result`, scheduler (cleanup, export watchdog, daily reporter), `create_app(orchestrator)`. | Wires MqttClientWrapper, SmartZoneFilter, TimelineLogger, all managers, lifecycle, ai_analyzer, Flask. |
| `src/frigate_buffer/models.py` | Pydantic/data models: `EventPhase`, `EventState`, `ConsolidatedEvent`, `FrameMetadata`, `NotificationEvent` protocol; helpers for CE IDs and "no concerns". | Used by orchestrator, managers, notifier, query. |

### Managers

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/managers/file.py` | **FileManager:** storage paths, clip/snapshot download (via DownloadService), export coordination (no transcode), cleanup, path validation (realpath/commonpath). `cleanup_old_events`, `rename_event_folder`, `write_canceled_summary`, `compute_storage_stats`, `resolve_clip_in_folder`. | Used by orchestrator, lifecycle, query, download, timeline, event_test. |
| `src/frigate_buffer/managers/state.py` | **EventStateManager:** in-memory event state (phase, metadata), active event tracking. | Orchestrator, lifecycle. |
| `src/frigate_buffer/managers/consolidation.py` | **ConsolidatedEventManager:** CE grouping, `closing` state, `mark_closing`, on_close callback. | Orchestrator, lifecycle, timeline_logger, query. |
| `src/frigate_buffer/managers/reviews.py` | **DailyReviewManager:** Frigate daily review fetch/cache. | Orchestrator, web. |
| `src/frigate_buffer/managers/zone_filter.py` | **SmartZoneFilter:** per-camera zone/exception filters; `should_start_event`. | Orchestrator (event creation). |

### Services

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/services/ai_analyzer.py` | **GeminiAnalysisService:** frame extraction, optional crop, system prompt from file; POST to OpenAI-compatible proxy; returns analysis dict; writes `analysis_result.json`; rolling frame cap. Multi-cam: `analyze_multi_clip_ce` uses multi_clip_extractor. | Called by orchestrator (`on_clip_ready`, `on_ce_ready_for_analysis`); uses VideoService, FileManager helpers. |
| `src/frigate_buffer/services/multi_clip_extractor.py` | Target-centric frame extraction for CE; requires detection sidecars (no HOG fallback when any camera lacks sidecar). | Used by ai_analyzer, event_test_orchestrator. |
| `src/frigate_buffer/services/timeline_ema.py` | EMA-based camera timeline (segment assignment, hysteresis) for multi-cam when `camera_timeline_use_ema_pipeline` true. | Used by multi_clip_extractor / pipeline. |
| `src/frigate_buffer/services/video.py` | **VideoService:** NVDEC decode (ffmpegcv), `generate_detection_sidecar`, `generate_detection_sidecars_for_cameras` (shared YOLO + lock), `generate_gif_from_clip`. App-level sidecar lock injected by orchestrator. | Used by lifecycle, ai_analyzer, event_test. |
| `src/frigate_buffer/services/lifecycle.py` | **EventLifecycleService:** event creation, event end (discard short, cancel long, clip export/download), CE pipeline (download per camera, sidecars, then `on_ce_ready_for_analysis`). | Orchestrator delegates; calls download, file_manager, video_service, orchestrator callbacks. |
| `src/frigate_buffer/services/download.py` | **DownloadService:** Frigate snapshot, export/clip download (dynamic clip names), `post_event_description`. | FileManager, lifecycle, orchestrator. |
| `src/frigate_buffer/services/notifier.py` | **NotificationPublisher:** publish to `frigate/custom/notifications`; `clear_tag` for updates; timeline_callback = TimelineLogger.log_ha. | Orchestrator, lifecycle. |
| `src/frigate_buffer/services/query.py` | **EventQueryService:** read event data from filesystem with TTL and per-folder caching; `resolve_clip_in_folder`; list events, timeline merge. | Flask server (events, stats, daily-review, player). |
| `src/frigate_buffer/services/daily_reporter.py` | **DailyReporterService:** scheduled; aggregate analysis_result (or daily_reports aggregate JSONL), report prompt, send_text_prompt, write `daily_reports/YYYY-MM-DD_report.md`. | Scheduled by orchestrator. |
| `src/frigate_buffer/services/frigate_export_watchdog.py` | Parse timeline for export_id, verify clip exists, DELETE Frigate `/api/export/{id}`; 404/422 = already removed. | Scheduled by orchestrator. |
| `src/frigate_buffer/services/timeline.py` | **TimelineLogger:** append HA/MQTT/Frigate API entries to `notification_timeline.json` via FileManager. | Orchestrator, notifier (timeline_callback). |
| `src/frigate_buffer/services/mqtt_client.py` | **MqttClientWrapper:** connect, subscribe, message callback to orchestrator. | Orchestrator provides `_on_mqtt_message`. |
| `src/frigate_buffer/services/crop_utils.py` | Crop/resize and motion-related image helpers. | ai_analyzer, multi_cam_recap. |
| `src/frigate_buffer/services/report_prompt.txt` | Default prompt for daily report. | daily_reporter. |
| `src/frigate_buffer/services/ai_analyzer_system_prompt.txt` | System prompt for Gemini proxy. | ai_analyzer. |

### Event test (TEST button only)

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/event_test/__init__.py` | Exports `run_test_pipeline`. | Web server calls it for TEST button. |
| `src/frigate_buffer/event_test/event_test_orchestrator.py` | Allocates `events/testN`, copies source, delegates to VideoService (sidecars) and same multi-clip extractor; no YOLO/lock in this module. | Used only for TEST button; production logic in ai_analyzer and multi_clip_extractor. |

### Web

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `src/frigate_buffer/web/server.py` | **create_app(orchestrator):** routes `/player`, `/stats-page`, `/daily-review`, `/test-multi-cam`, `/api/test-multi-cam/*`, `/api/events`, `/api/files`, `/api/daily-review`, `/api/stats`, `/status`, timeline page/download. Uses EventQueryService, path safety via file_manager. | Closes over orchestrator; uses query service, file_manager. |
| `src/frigate_buffer/web/templates/*.html` | Jinja2 templates for player, stats, daily review, timeline, test run. | Rendered by Flask. |
| `src/frigate_buffer/web/static/*.js` | DOMPurify, Marked (min). | Served by Flask. |

### Scripts

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `scripts/multi_cam_recap.py` | Standalone: own MQTT loop, `process_multi_cam_event` (frame extract, optional Gemini, write stitched/zip). Uses same config and FileManager helpers. | Not started by orchestrator. |

### Tests

| Path/Name | Purpose | Dependencies/Interactions |
|-----------|---------|---------------------------|
| `tests/conftest.py` | Pytest fixtures. | All tests. |
| `tests/test_*.py` | Unit tests for config, orchestrator, lifecycle, ai_analyzer, video, query, notifier, download, file manager, cleanup, etc. | Run with `pytest tests/`; `pythonpath = ["src"]`. |

---

## 5. Core Flows & Lifecycles

### Main application (orchestrator-centric)

1. **Startup:** `main.py` → `load_config()` → `StateAwareOrchestrator(config)` → `orchestrator.start()` (MQTT connect, Flask, schedule jobs).
2. **MQTT → Event creation/updates:** MQTT → `MqttClientWrapper` → `StateAwareOrchestrator._on_mqtt_message` → SmartZoneFilter (should_start) → EventStateManager / ConsolidatedEventManager → **EventLifecycleService** (event creation, event end).
3. **Event end (short):** Lifecycle checks duration &lt; `minimum_event_seconds` → discard: delete folder, remove from state/CE, publish MQTT `status: "discarded"` with same tag for HA clear.
4. **Event end (long/canceled):** Duration ≥ `max_event_length_seconds` → no clip export/AI; write canceled summary, notify HA, rename folder `-canceled`; cleanup later by retention.
5. **Clip ready (single-cam):** Lifecycle → orchestrator `on_clip_ready` → **GeminiAnalysisService.analyze_clip** (frame extraction, proxy) → `_handle_analysis_result` → update state, write files, POST description to Frigate, notify HA.
6. **Consolidated event (multi-cam):** Lifecycle downloads each clip to `{camera}-{5_digits}.mp4`, **VideoService.generate_detection_sidecars_for_cameras** (parallel, shared YOLO + lock) → when all sidecars ready → `on_ce_ready_for_analysis` → **analyze_multi_clip_ce** (multi_clip_extractor, optional timeline_ema) → `_handle_ce_analysis_result` → write summary/metadata at CE root, notify HA.
7. **Web:** Flask uses **EventQueryService** to list events, stats, timeline; `resolve_clip_in_folder` for dynamic clip URLs; path safety via FileManager.
8. **Scheduled:** Cleanup (retention), export watchdog (DELETE completed exports), daily reporter (aggregate + report prompt → proxy → markdown).

**Files touched in primary flow:** `orchestrator.py`, `services/lifecycle.py`, `services/ai_analyzer.py`, `services/mqtt_client.py`, `managers/state.py`, `managers/file.py`, `managers/consolidation.py`, `services/notifier.py`, `services/download.py`, `services/query.py`, `web/server.py`.

### Standalone multi-cam script

- MQTT → `scripts/multi_cam_recap.py` (own client) → on linked-event message → `process_multi_cam_event(main_event_id, linked_event_ids)` → frame extraction, optional Gemini, FileManager helpers (`write_stitched_frame`, `create_ai_analysis_zip`). Does not go through the orchestrator.

---

## 6. AI Agent Directives (Rules & Conventions)

### File placement rules

| Need | Location |
|------|----------|
| New **UI component / page** | Add route in `src/frigate_buffer/web/server.py`; template in `src/frigate_buffer/web/templates/`; static assets in `src/frigate_buffer/web/static/`. |
| New **business logic / service** | `src/frigate_buffer/services/` (or `managers/` if it is state/aggregation). Register and call from `orchestrator.py` (or from an existing service) as appropriate. |
| New **utility function** (generic, no I/O) | `src/frigate_buffer/services/` (e.g. `crop_utils.py`) or a new module under `services/` if it fits a clear domain. |
| New **API route** (REST) | Add in `src/frigate_buffer/web/server.py`; use EventQueryService or FileManager for data; never put business logic in server.py beyond delegation. |
| New **config key** | Add to **CONFIG_SCHEMA** in `src/frigate_buffer/config.py` first; then add flat key in config merge; use in code via `config.get('KEY', default)`. |
| New **standalone script** | `scripts/` at repo root (e.g. `scripts/multi_cam_recap.py`). |
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
