# Phase 1: Discovery & Cataloging — Unify Event Processing

**Refactor goal:** "Everything is a Consolidated Event." Single-camera triggers are processed as a CE where camera count equals 1. The dedicated single-camera path is eliminated.

**Status:** Phases 1–4 complete. Phase 4 (test suite) updated: removed/refactored tests for `analyze_clip`, `on_clip_ready_for_analysis`, `_extract_frames`, `write_ai_frame_analysis_single_cam`; integration tests use `analyze_multi_clip_ce` and CE folder; lifecycle and max-event tests assert CE path only.

---

## 1. Files to be completely deleted

**None.** No file is used exclusively for single-cam analysis. All single-cam logic lives inside shared modules (`orchestrator.py`, `lifecycle.py`, `ai_analyzer.py`, `file.py`).

---

## 2. Methods / functions to be removed or changed

### Orchestrator (`src/frigate_buffer/orchestrator.py`)

| Item | Action |
|------|--------|
| `_on_clip_ready_for_analysis` callback (inner `_run()` that calls `analyze_clip`, `_handle_analysis_result`, `clear_frame_metadata`) | **Remove** entirely |
| Passing `on_clip_ready` into `EventLifecycleService` | **Remove** (set `on_clip_ready_for_analysis=None` or remove parameter from lifecycle) |
| `_handle_analysis_result` | **Keep** — still used from `_handle_review` for Frigate review updates |

### Lifecycle (`src/frigate_buffer/services/lifecycle.py`)

| Item | Action |
|------|--------|
| Constructor parameter and field `on_clip_ready_for_analysis` | **Remove** |
| In `process_event_end`: block when `ce` is `None` (lines ~169–222) that exports clip and calls `on_clip_ready_for_analysis(event_id, clip_path)` | **Remove** or reduce to scheduling CE close for edge cases |
| In `finalize_consolidated_event`: single-camera branch (lines ~351–388) that exports clip without generating sidecars and calls `on_clip_ready_for_analysis` | **Remove** |
| Unify `finalize_consolidated_event` | **Change:** For all CEs (including single-camera): (1) same clip-export loop for all `camera_events`, (2) always call `generate_detection_sidecars_for_cameras` for every exported clip, (3) always call `on_ce_ready_for_analysis` (remove condition `len(camera_events) > 1`) |

### AI analyzer (`src/frigate_buffer/services/ai_analyzer.py`)

| Item | Action |
|------|--------|
| Public method `analyze_clip` (and its use of `_extract_frames`, `write_ai_frame_analysis_single_cam`, `_save_analysis_result` for that path) | **Remove** |
| `_extract_frames` | **Remove** if unused after removing `analyze_clip` (or keep if still used by tests/event_test) |
| `analyze_multi_clip_ce` | **Verify** only — ensure it works for single camera (one subdir, one clip, one sidecar); no API change |

### File manager (`src/frigate_buffer/managers/file.py`)

| Item | Action |
|------|--------|
| Function `write_ai_frame_analysis_single_cam` | **Remove** |
| `_write_ai_analysis_internal` and `write_ai_frame_analysis_multi_cam` | **Keep** |

### State / consolidation

No method removals in `state.py` or `consolidation.py`. Phase 2 may add behavioral changes (e.g. instant close for single-cam CE).

---

## 3. Config keys to be removed or merged

| Key | Location | Action |
|-----|----------|--------|
| `final_review_image_count` / `FINAL_REVIEW_IMAGE_COUNT` | `config.py` (CONFIG_SCHEMA `settings`), default dict, settings merge; `config.yaml`; `config.example.yaml` | **Remove** from schema and merge; remove or replace comment in YAML. Document that frame limits for all events are only `multi_cam.max_multi_cam_frames_sec` and `multi_cam.max_multi_cam_frames_min`. |
| Cooldowns | — | No separate single/multi cooldowns; `event_gap_seconds` applies to all. **No change.** |
| `MULTI_CAM_SYSTEM_PROMPT_FILE` / prompt file | — | **No key removal.** Update prompt text in `ai_analyzer_system_prompt.txt` to state model may receive frames from a single angle or multiple angles (Phase 3). |

---

## 4. Tests updated (Phase 4 — done)

| Test file | Change applied |
|-----------|----------------|
| `tests/test_ai_analyzer.py` | Removed classes testing `_extract_frames` / `analyze_clip` / `_save_analysis_result`; kept `analyze_multi_clip_ce` and config/smart_crop tests; renamed frame-metadata class to `TestGeminiAnalysisServiceConfigMisc` |
| `tests/test_lifecycle_service.py` | Removed all `on_clip_ready_for_analysis` setup and asserts; `schedule_close_timer` assert updated to `(ce_id, delay_seconds=None)` |
| `tests/test_max_event_length.py` | Removed class `TestOrchestratorDefenseInDepth`; FileManager/max-length behavior covered by lifecycle tests |
| `tests/test_integration_step_5_6.py` | Persistence and error-handling tests refactored to use `analyze_multi_clip_ce` with mocked `extract_target_centric_frames` and CE dir; `_handle_analysis_result` test unchanged |
| `tests/test_ai_frame_analysis_writing.py` | Removed `write_ai_frame_analysis_single_cam` import and `test_single_cam_writing`; `test_multi_cam_writing` uses real numpy frame |
| `tests/test_optimization_expectations_temp.py` | Removed `TestOpt2FrameMetadataBisect` (depended on `_extract_frames` and `FINAL_REVIEW_IMAGE_COUNT`) |

---

## 5. Post-refactor flow (single-cam CE)

1. Event ends → in CE (camera count 1).
2. CE close timer fires → `finalize_consolidated_event`.
3. Export clip for the one camera; call `generate_detection_sidecars_for_cameras` for that clip (so `detection.json` exists).
4. Call `on_ce_ready_for_analysis(ce_id, ce_folder_path, ...)`.
5. `analyze_multi_clip_ce` → `extract_target_centric_frames` (one camera, one sidecar) → proxy → `write_ai_frame_analysis_multi_cam` → `_handle_ce_analysis_result`.

---

**End of Phase 1 catalog.** Proceed to Phase 2 only after explicit approval.
