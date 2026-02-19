"""
Mini test orchestrator: runs the post-download pipeline (transcode, frame extraction, build payload)
for a copied event folder without sending to the AI proxy. Yields log events for SSE streaming.

All logic here only delegates to the main codebase; bugs should surface in original modules.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import logging
from typing import Any, Iterator

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


def _yield_log(msg: str) -> dict[str, Any]:
    """Single log event for SSE."""
    return {"type": "log", "message": msg}


def _yield_done(test_run_id: str, ai_request_url: str) -> dict[str, Any]:
    return {"type": "done", "test_run_id": test_run_id, "ai_request_url": ai_request_url}


def _yield_error(msg: str) -> dict[str, Any]:
    return {"type": "error", "message": msg}


def run_test_pipeline(
    source_folder_path: str,
    storage_path: str,
    file_manager: Any,
    download_service: Any,
    ai_analyzer: Any,
    config: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """
    Run the post-download pipeline on a copy of the source event. Yields events for SSE:
    {"type": "log", "message": "..."} | {"type": "done", "test_run_id", "ai_request_url"} | {"type": "error", "message": "..."}.

    Steps: validate source -> allocate testN -> copy source -> delete detection.json -> transcode ->
    build payload (writes frames) -> write system_prompt.txt and ai_request.html.
    """
    events_dir = os.path.join(storage_path, "events")
    if not os.path.isdir(events_dir):
        yield _yield_error("Events directory not found")
        return

    # 1. Validate source: every camera subdir has clip.mp4
    try:
        with os.scandir(source_folder_path) as it:
            camera_subdirs = [e.name for e in it if e.is_dir() and not e.name.startswith(".")]
    except OSError as e:
        yield _yield_error(f"Source folder not readable: {e}")
        return

    missing: list[str] = []
    for cam in camera_subdirs:
        clip_path = os.path.join(source_folder_path, cam, "clip.mp4")
        if not os.path.isfile(clip_path):
            missing.append(cam)
    if missing:
        yield _yield_error(f"Source event incomplete: missing clip for camera(s): {', '.join(missing)}")
        return

    yield _yield_log(f"Source validated: {len(camera_subdirs)} camera(s) with clip.mp4")

    # 2. Allocate testN with lock
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
        try:
            os.makedirs(test_folder_path, exist_ok=False)
        except FileExistsError:
            yield _yield_error(f"Allocation race: {test_run_id} already exists")
            return
        except OSError as e:
            yield _yield_error(f"Could not create test folder: {e}")
            return

    yield _yield_log(f"Test run output: events/{test_run_id}")

    # 3. Copy source into testN
    try:
        for cam in camera_subdirs:
            src_cam = os.path.join(source_folder_path, cam)
            dst_cam = os.path.join(test_folder_path, cam)
            shutil.copytree(src_cam, dst_cam)
        # Copy CE-root files if any (notification_timeline.json, etc.)
        for name in os.listdir(source_folder_path):
            if name in camera_subdirs or name.startswith("."):
                continue
            src_f = os.path.join(source_folder_path, name)
            if os.path.isfile(src_f):
                shutil.copy2(src_f, os.path.join(test_folder_path, name))
    except OSError as e:
        yield _yield_error(f"Copy failed: {e}")
        try:
            shutil.rmtree(test_folder_path, ignore_errors=True)
        except Exception:
            pass
        return

    yield _yield_log("Copied source to test folder")

    # 4. Timeline (for log)
    try:
        merged = read_timeline_merged(test_folder_path)
        entries = merged.get("entries") or []
        yield _yield_log(f"Timeline merged: {len(entries)} entries")
    except Exception as e:
        yield _yield_log(f"Timeline read warning: {e}")

    ce_start_time = get_ce_start_time_from_folder_path(source_folder_path)
    if ce_start_time > 0:
        yield _yield_log(f"ce_start_time from folder: {ce_start_time}")
    else:
        yield _yield_log("ce_start_time: 0 (folder name has no timestamp)")

    # 5. Delete existing detection.json in testN so transcode runs fresh
    for cam in camera_subdirs:
        sidecar = os.path.join(test_folder_path, cam, "detection.json")
        if os.path.isfile(sidecar):
            try:
                os.remove(sidecar)
                yield _yield_log(f"Removed existing detection.json for {cam}")
            except OSError:
                pass

    # 6. Transcode each camera in testN
    det_model = config.get("DETECTION_MODEL") or None
    det_device = (config.get("DETECTION_DEVICE") or "").strip() or None
    for cam in camera_subdirs:
        camera_folder = os.path.join(test_folder_path, cam)
        sidecar_path = os.path.join(camera_folder, "detection.json")
        yield _yield_log(f"Transcoding {cam}...")
        ok = download_service.transcode_existing_clip(
            "test",
            camera_folder,
            detection_sidecar_path=sidecar_path,
            detection_model=det_model,
            detection_device=det_device,
        )
        if ok:
            yield _yield_log(f"Transcode OK: {cam}")
        else:
            yield _yield_log(f"Transcode failed or skipped: {cam}")

    # 7. Build payload (delegate to analyzer; writes frames and returns prompt + paths)
    yield _yield_log("Extracting frames and building payload...")
    result = ai_analyzer.build_multi_cam_payload_for_preview(test_folder_path, ce_start_time)
    if not result:
        yield _yield_error("Failed to build payload (no frames or analyzer error)")
        return
    system_prompt, _user_content, frame_relative_paths = result
    yield _yield_log(f"Payload built: {len(frame_relative_paths)} frame files")

    # 8. Write system_prompt.txt
    prompt_path = os.path.join(test_folder_path, "system_prompt.txt")
    try:
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(system_prompt)
        yield _yield_log("Wrote system_prompt.txt")
    except OSError as e:
        yield _yield_log(f"Warning: could not write system_prompt.txt: {e}")

    # 9. Generate ai_request.html with download links (system_prompt.txt + frame paths)
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
        return

    yield _yield_done(test_run_id, f"{base_url}/ai_request.html")


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
