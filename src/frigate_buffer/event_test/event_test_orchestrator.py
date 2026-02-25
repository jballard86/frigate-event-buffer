"""
Test-button-only orchestrator: runs the post-download pipeline (generate detection sidecar, frame extraction,
build payload) for a copied event folder without sending to the AI proxy. Yields log events for SSE streaming.

This module is only used for test-button orchestration. No core behavior may depend on it—deleting
src/frigate_buffer/event_test/ would break only the TEST button and event_test tests. All logic here
only delegates to the main codebase (no YOLO, no sidecar lock, no ThreadPoolExecutor); it must stay thin.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import logging
from typing import Any, Iterator

from frigate_buffer.logging_utils import StreamCaptureHandler
from frigate_buffer.services.query import read_timeline_merged

logger = logging.getLogger("frigate-buffer")

# Lock for allocating testN so concurrent TEST clicks get different folders
_alloc_lock = threading.Lock()


def get_ce_start_time_from_folder_path(folder_path: str) -> float:
    """
    Derive ce_start_time from the folder name for use in prompt/overlay.

    CE folder names are {timestamp}_{id} (e.g. 1730000000_abc). We use the timestamp
    so overlay and system prompt show the correct activity time. If the folder name
    does not match (e.g. test1), returns 0.0 and the analyzer will use "now" for the
    activity start string.
    """
    name = os.path.basename(os.path.abspath(folder_path))
    parts = name.split("_", 1)
    if parts and parts[0].isdigit():
        try:
            return float(parts[0])
        except ValueError:
            pass
    return 0.0


def resolve_canonical_camera_name(
    folder_name: str,
    config: dict[str, Any],
    file_manager: Any,
) -> str:
    """
    Resolve a camera folder name (e.g. sanitized "doorbell") to the canonical name
    (e.g. "Doorbell") expected by Frigate's Export API. Uses config ALLOWED_CAMERAS
    and FileManager.sanitize_camera_name so the test video-request uses the same
    name as production (lifecycle uses event.camera; folders use sanitized names).
    Returns folder_name if no match (e.g. camera not in config).
    """
    allowed = config.get("ALLOWED_CAMERAS") or []
    for canonical in allowed:
        if file_manager.sanitize_camera_name(canonical) == folder_name:
            return canonical
    return folder_name


def _yield_log(msg: str) -> dict[str, Any]:
    """Single log event for SSE."""
    return {"type": "log", "message": msg}


def _yield_done(test_run_id: str, ai_request_url: str) -> dict[str, Any]:
    return {"type": "done", "test_run_id": test_run_id, "ai_request_url": ai_request_url}


def _yield_error(msg: str) -> dict[str, Any]:
    return {"type": "error", "message": msg}


def _get_camera_subdirs_with_clip(folder_path: str) -> list[str]:
    """
    List camera subdir names under folder_path that contain a resolvable clip (*.mp4).
    Raises OSError if folder_path is not readable. Used by prepare_test_folder and pipeline runners.
    """
    from frigate_buffer.services.query import resolve_clip_in_folder

    with os.scandir(folder_path) as it:
        all_subdirs = [e.name for e in it if e.is_dir() and not e.name.startswith(".")]
    return [
        cam for cam in all_subdirs
        if resolve_clip_in_folder(os.path.join(folder_path, cam))
    ]


def _allocate_test_folder(events_dir: str) -> tuple[str, str]:
    """
    Allocate next events/testN under events_dir. Holds _alloc_lock.
    Returns (test_run_id, test_folder_path). Raises FileExistsError or OSError on failure.
    """
    with _alloc_lock:
        existing = set()
        try:
            with os.scandir(events_dir) as scan:
                for e in scan:
                    if e.is_dir() and re.match(r"^test\d+$", e.name):
                        existing.add(e.name)
        except OSError:
            pass
        n = 1
        while f"test{n}" in existing:
            n += 1
        test_run_id = f"test{n}"
        test_folder_path = os.path.join(events_dir, test_run_id)
        os.makedirs(test_folder_path, exist_ok=False)
        return (test_run_id, test_folder_path)


def _copy_source_to_test_folder(
    source_folder_path: str,
    test_folder_path: str,
    camera_subdirs: list[str],
) -> None:
    """
    Copy camera subdirs and CE-root files from source to test folder. Raises OSError on failure.
    """
    for cam in camera_subdirs:
        src_cam = os.path.join(source_folder_path, cam)
        dst_cam = os.path.join(test_folder_path, cam)
        shutil.copytree(src_cam, dst_cam)
    for name in os.listdir(source_folder_path):
        if name in camera_subdirs or name.startswith("."):
            continue
        src_f = os.path.join(source_folder_path, name)
        if os.path.isfile(src_f):
            shutil.copy2(src_f, os.path.join(test_folder_path, name))


def prepare_test_folder(
    source_folder_path: str,
    storage_path: str,
    file_manager: Any,
) -> tuple[str, str]:
    """
    Allocate testN and copy source event into it (copy only, no sidecars or pipeline).
    Returns (test_run_id, test_folder_path). Raises ValueError on validation or copy failure.
    """
    events_dir = os.path.join(storage_path, "events")
    if not os.path.isdir(events_dir):
        raise ValueError("Events directory not found")

    try:
        camera_subdirs = _get_camera_subdirs_with_clip(source_folder_path)
    except OSError as e:
        raise ValueError(f"Source folder not readable: {e}") from e
    if not camera_subdirs:
        raise ValueError(
            "Need clip and data from at least one camera. No camera in this event has an .mp4 clip yet."
        )

    try:
        test_run_id, test_folder_path = _allocate_test_folder(events_dir)
    except FileExistsError:
        raise ValueError("Allocation race: test folder already exists") from None
    except OSError as e:
        raise ValueError(f"Could not create test folder: {e}") from e

    try:
        _copy_source_to_test_folder(source_folder_path, test_folder_path, camera_subdirs)
    except OSError as e:
        try:
            shutil.rmtree(test_folder_path, ignore_errors=True)
        except Exception:
            pass
        raise ValueError(f"Copy failed: {e}") from e

    return (test_run_id, test_folder_path)


def get_export_time_range_from_folder(
    folder_path: str,
    config: dict[str, Any],
) -> tuple[float, float]:
    """
    Derive (global_min_start, global_max_end) from folder timeline for Frigate export.
    Returns logical event start/end; caller passes these plus export_before/export_after
    to DownloadService.export_and_download_clip. Falls back to folder mtime if no timeline.
    """
    merged = read_timeline_merged(folder_path)
    entries = merged.get("entries") or []
    start_times: list[float] = []
    end_times: list[float] = []
    for e in entries:
        data = e.get("data") or {}
        payload = data.get("payload") or {}
        after = payload.get("after") or {}
        before = payload.get("before") or {}
        created = after.get("created_at") or before.get("created_at") or after.get("start_time")
        if created is not None:
            try:
                start_times.append(float(created))
            except (TypeError, ValueError):
                pass
        end_time = after.get("end_time")
        if end_time is not None:
            try:
                end_times.append(float(end_time))
            except (TypeError, ValueError):
                pass
        if e.get("source") == "test_ai_prompt":
            et = (e.get("data") or {}).get("end_time")
            st = (e.get("data") or {}).get("start_time")
            if et is not None:
                try:
                    end_times.append(float(et))
                except (TypeError, ValueError):
                    pass
            if st is not None:
                try:
                    start_times.append(float(st))
                except (TypeError, ValueError):
                    pass
    if start_times and end_times:
        return (min(start_times), max(end_times))
    fallback_end = os.path.getmtime(folder_path)
    return (fallback_end - 60, fallback_end)


def _get_start_time_from_timeline_entries(entries: list[Any]) -> float | None:
    """
    Extract minimum start time from timeline entries (same structure as get_export_time_range_from_folder).
    Returns None if no valid start times. Used for test runs when folder name has no timestamp.
    """
    start_times: list[float] = []
    for e in entries:
        data = e.get("data") or {}
        payload = data.get("payload") or {}
        after = payload.get("after") or {}
        before = payload.get("before") or {}
        created = after.get("created_at") or before.get("created_at") or after.get("start_time")
        if created is not None:
            try:
                start_times.append(float(created))
            except (TypeError, ValueError):
                pass
        if e.get("source") == "test_ai_prompt":
            st = (e.get("data") or {}).get("start_time")
            if st is not None:
                try:
                    start_times.append(float(st))
                except (TypeError, ValueError):
                    pass
    return min(start_times) if start_times else None


def _write_system_prompt_and_ai_request_html(
    test_folder_path: str,
    test_run_id: str,
    system_prompt: str,
    frame_relative_paths: list[str],
) -> Iterator[dict[str, Any]]:
    """Write system_prompt.txt and ai_request.html; yield log/error events."""
    prompt_path = os.path.join(test_folder_path, "system_prompt.txt")
    try:
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(system_prompt)
        yield _yield_log("Wrote system_prompt.txt")
    except OSError as e:
        yield _yield_log(f"Warning: could not write system_prompt.txt: {e}")

    base_url = f"/files/events/{test_run_id}"
    lines = [
        "<!DOCTYPE html><html><head><meta charset='UTF-8'><title>AI request – " + test_run_id + "</title>",
        "<style>body{font-family:sans-serif;background:#1a1a2e;color:#e0e0e0;padding:16px;}",
        "pre{background:#0f3460;padding:12px;border-radius:8px;overflow:auto;}",
        "a{color:#e94560;} .frame{margin:12px 0;} .frame img{max-width:100%;}</style></head><body>",
        "<h1>AI request – " + test_run_id + "</h1>",
        "<h2>System prompt</h2>",
        f"<p><a href='{base_url}/system_prompt.txt' download>Download system_prompt.txt</a></p>",
        "<pre>" + _html_escape(system_prompt) + "</pre>",
        "<h2>Frames</h2>",
    ]
    for rel in frame_relative_paths:
        url = f"{base_url}/{rel.replace(os.sep, '/')}"
        fname = os.path.basename(rel)
        lines.append(f"<div class='frame'><a href='{url}' download>{fname}</a><br><img src='{url}' alt='{fname}' /></div>")
    lines.append(
        "<p><button id='sendBtn' type='button'>Send prompt to AI</button> <span id='sendStatus'></span></p>"
    )
    lines.append(
        "<script>document.getElementById('sendBtn').onclick=function(){"
        "var s=document.getElementById('sendStatus'); s.textContent='Sending...';"
        "fetch('/api/test-multi-cam/send?test_run=" + test_run_id + "',{method:'POST'})"
        ".then(r=>r.json()).then(d=>{ s.textContent='OK: '+JSON.stringify(d).slice(0,200); })"
        ".catch(e=>{ s.textContent='Error: '+e.message; }); };</script>"
    )
    lines.append("</body></html>")
    html_path = os.path.join(test_folder_path, "ai_request.html")
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        yield _yield_log("Wrote ai_request.html")
    except OSError as e:
        yield _yield_error(f"Could not write ai_request.html: {e}")


def _run_post_copy_steps(
    test_folder_path: str,
    test_run_id: str,
    video_service: Any,
    ai_analyzer: Any,
    config: dict[str, Any],
    camera_subdirs: list[str],
    ce_start_time: float,
) -> Iterator[dict[str, Any]]:
    """
    Run post-copy pipeline steps: timeline log, remove sidecars, generate sidecars,
    compilation, build payload, write system_prompt.txt and ai_request.html, yield done.
    """
    from frigate_buffer.services.query import resolve_clip_in_folder

    entries: list[Any] = []
    try:
        merged = read_timeline_merged(test_folder_path)
        entries = merged.get("entries") or []
        yield _yield_log(f"Timeline merged: {len(entries)} entries")
    except Exception as e:
        yield _yield_log(f"Timeline read warning: {e}")

    # For test runs (folder name test1, test2, ...) folder has no timestamp; use timeline start if available.
    from_timeline = False
    if ce_start_time <= 0 and entries:
        timeline_start = _get_start_time_from_timeline_entries(entries)
        if timeline_start is not None:
            ce_start_time = timeline_start
            from_timeline = True
    if ce_start_time > 0:
        yield _yield_log(f"ce_start_time from {'timeline' if from_timeline else 'folder'}: {ce_start_time}")
    else:
        yield _yield_log("Test run: ce_start_time 0 (prompt will use current time)")

    for cam in camera_subdirs:
        sidecar = os.path.join(test_folder_path, cam, "detection.json")
        if os.path.isfile(sidecar):
            try:
                os.remove(sidecar)
                yield _yield_log(f"Removed existing detection.json for {cam}")
            except OSError:
                pass

    sidecar_tasks: list[tuple[str, str, str]] = []
    for cam in camera_subdirs:
        camera_folder = os.path.join(test_folder_path, cam)
        clip_basename = resolve_clip_in_folder(camera_folder)
        if not clip_basename:
            yield _yield_log(f"Skipping {cam}: no clip found")
            continue
        clip_path = os.path.join(camera_folder, clip_basename)
        sidecar_path = os.path.join(camera_folder, "detection.json")
        sidecar_tasks.append((cam, clip_path, sidecar_path))

    for cam, _clip, _sidecar in sidecar_tasks:
        yield _yield_log(f"Generating detection sidecar for {cam}...")
    if sidecar_tasks:
        results = video_service.generate_detection_sidecars_for_cameras(sidecar_tasks, config)
        for cam, ok in results:
            yield _yield_log(f"Sidecar {cam}: {'OK' if ok else 'failed'}.")

    yield _yield_log("Generating hardware-accelerated compilation video...")
    try:
        from frigate_buffer.services.video_compilation import compile_ce_video
        events_end_ts_list = [
            e.get("data", {}).get("after", {}).get("end_time")
            for e in entries
            if e.get("data", {}).get("after", {}).get("end_time")
        ]
        fallback_end = (max(events_end_ts_list) - ce_start_time) if events_end_ts_list and ce_start_time else 60.0
        comp_output = compile_ce_video(test_folder_path, fallback_end, config)
        if comp_output:
            yield _yield_log(f"Compilation finished at: {comp_output}")
        else:
            yield _yield_log("Compilation failed (see generic console logs).")
    except Exception as e:
        yield _yield_log(f"Compilation hook error: {e}")

    payload_logs: list[str] = []
    result, payload_error = ai_analyzer.build_multi_cam_payload_for_preview(
        test_folder_path, ce_start_time, log_messages=payload_logs
    )
    for msg in payload_logs:
        yield _yield_log(msg)
    if payload_error:
        yield _yield_error(payload_error)
        return
    if not result:
        yield _yield_error("Failed to build payload (no frames or analyzer error)")
        return
    system_prompt, _user_content, frame_relative_paths = result
    yield _yield_log(f"Payload built: {len(frame_relative_paths)} frame files")

    for ev in _write_system_prompt_and_ai_request_html(
        test_folder_path, test_run_id, system_prompt, frame_relative_paths
    ):
        yield ev
        if ev.get("type") == "error":
            return
    yield _yield_done(test_run_id, f"/files/events/{test_run_id}/ai_request.html")


def _run_test_pipeline_inner(
    source_folder_path: str,
    storage_path: str,
    file_manager: Any,
    download_service: Any,
    video_service: Any,
    ai_analyzer: Any,
    config: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """
    Inner pipeline: same steps as run_test_pipeline. Yields log/done/error events.
    Used by run_test_pipeline with a capture handler so Python log output is also streamed.
    """
    events_dir = os.path.join(storage_path, "events")
    if not os.path.isdir(events_dir):
        yield _yield_error("Events directory not found")
        return

    try:
        camera_subdirs = _get_camera_subdirs_with_clip(source_folder_path)
    except OSError as e:
        yield _yield_error(f"Source folder not readable: {e}")
        return
    if not camera_subdirs:
        yield _yield_error(
            "Need clip and data from at least one camera. No camera in this event has an .mp4 clip yet."
        )
        return
    yield _yield_log(f"Source validated: {len(camera_subdirs)} camera(s) with clip")

    try:
        test_run_id, test_folder_path = _allocate_test_folder(events_dir)
    except FileExistsError:
        yield _yield_error("Allocation race: test folder already exists")
        return
    except OSError as e:
        yield _yield_error(f"Could not create test folder: {e}")
        return
    yield _yield_log(f"Test run output: events/{test_run_id}")

    try:
        _copy_source_to_test_folder(source_folder_path, test_folder_path, camera_subdirs)
    except OSError as e:
        yield _yield_error(f"Copy failed: {e}")
        try:
            shutil.rmtree(test_folder_path, ignore_errors=True)
        except Exception:
            pass
        return
    yield _yield_log("Copied source to test folder")

    ce_start_time = get_ce_start_time_from_folder_path(source_folder_path)
    yield from _run_post_copy_steps(
        test_folder_path, test_run_id, video_service, ai_analyzer, config, camera_subdirs, ce_start_time
    )


def _run_test_pipeline_post_copy(
    test_folder_path: str,
    test_run_id: str,
    file_manager: Any,
    video_service: Any,
    ai_analyzer: Any,
    config: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """
    Run only post-copy steps on an existing test folder (steps 4-9).
    test_folder_path must already exist with camera subdirs and clips.
    """
    try:
        camera_subdirs = _get_camera_subdirs_with_clip(test_folder_path)
    except OSError as e:
        yield _yield_error(f"Test folder not readable: {e}")
        return
    if not camera_subdirs:
        yield _yield_error("No camera in this test folder has an .mp4 clip.")
        return
    yield _yield_log(f"Test folder validated: {len(camera_subdirs)} camera(s) with clip")

    ce_start_time = get_ce_start_time_from_folder_path(test_folder_path)
    yield from _run_post_copy_steps(
        test_folder_path, test_run_id, video_service, ai_analyzer, config, camera_subdirs, ce_start_time
    )


def run_test_pipeline_from_folder(
    test_folder_path: str,
    file_manager: Any,
    video_service: Any,
    ai_analyzer: Any,
    config: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """
    Run post-copy pipeline on an existing test folder. Yields SSE events.
    Attaches StreamCaptureHandler so Python log output is streamed.
    """
    test_run_id = os.path.basename(os.path.abspath(test_folder_path))
    captured_logs: list[str] = []
    handler = StreamCaptureHandler(captured=captured_logs)
    frigate_logger = logging.getLogger("frigate-buffer")
    frigate_logger.addHandler(handler)
    try:
        for ev in _run_test_pipeline_post_copy(
            test_folder_path,
            test_run_id,
            file_manager,
            video_service,
            ai_analyzer,
            config,
        ):
            while captured_logs:
                yield _yield_log(captured_logs.pop(0))
            yield ev
        while captured_logs:
            yield _yield_log(captured_logs.pop(0))
    finally:
        frigate_logger.removeHandler(handler)


def run_test_pipeline(
    source_folder_path: str,
    storage_path: str,
    file_manager: Any,
    download_service: Any,
    video_service: Any,
    ai_analyzer: Any,
    config: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """
    Run the post-download pipeline on a copy of the source event. Yields events for SSE:
    {"type": "log", "message": "..."} | {"type": "done", "test_run_id", "ai_request_url"} | {"type": "error", "message": "..."}.

    Attaches a logging handler so all non-MQTT Python log output is also streamed to the test run page.
    """
    captured_logs: list[str] = []
    handler = StreamCaptureHandler(captured=captured_logs)
    frigate_logger = logging.getLogger("frigate-buffer")
    frigate_logger.addHandler(handler)
    try:
        for ev in _run_test_pipeline_inner(
            source_folder_path,
            storage_path,
            file_manager,
            download_service,
            video_service,
            ai_analyzer,
            config,
        ):
            while captured_logs:
                yield _yield_log(captured_logs.pop(0))
            yield ev
        while captured_logs:
            yield _yield_log(captured_logs.pop(0))
    finally:
        frigate_logger.removeHandler(handler)


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
