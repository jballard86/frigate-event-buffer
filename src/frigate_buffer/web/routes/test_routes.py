"""Test blueprint: /api/test-multi-cam/prepare, stream, video-request, send."""

import json
import os
import re
import time

from flask import Blueprint, Response, jsonify, request

from frigate_buffer.event_test.event_test_orchestrator import (
    get_export_time_range_from_folder,
    prepare_test_folder,
    resolve_canonical_camera_name,
    run_test_pipeline,
    run_test_pipeline_from_folder,
)
from frigate_buffer.logging_utils import set_suppress_review_debug_logs
from frigate_buffer.services.query import read_timeline_merged, resolve_clip_in_folder
from frigate_buffer.web.path_helpers import resolve_under_storage


def _resolve_source_path(storage_path: str, subdir: str) -> str | None:
    """Resolve subdir to absolute source path (supports saved/events/ce_id or
    events/ce_id)."""
    if "/" in subdir:
        path_parts = subdir.split("/")
        return resolve_under_storage(storage_path, *path_parts)
    return resolve_under_storage(storage_path, "events", subdir)


def create_bp(orchestrator):
    """Create test blueprint with routes closed over orchestrator."""
    bp = Blueprint("test", __name__, url_prefix="/api/test-multi-cam")
    storage_path = orchestrator.config["STORAGE_PATH"]
    file_manager = orchestrator.file_manager
    config = orchestrator.config
    download_service = orchestrator.download_service

    @bp.route("/prepare")
    def test_multi_cam_prepare():
        """Copy source event into events/testN (copy only). Returns test_run_id
        or error."""
        subdir = request.args.get("subdir", "").strip()
        if not subdir:
            return jsonify({"error": "Missing subdir"}), 400
        source_path = _resolve_source_path(storage_path, subdir)
        if source_path is None or not os.path.isdir(source_path):
            return jsonify({"error": "Invalid or missing event folder"}), 404
        try:
            test_run_id, _ = prepare_test_folder(
                source_path, storage_path, file_manager
            )
            return jsonify({"test_run_id": test_run_id})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @bp.route("/event-data")
    def test_multi_cam_event_data():
        """
        Return timeline and event_files for a test run (for test page collapsible
        bars). Resolves events/<test_run>, uses read_timeline_merged and same
        event_files scan as api.event_timeline.
        """
        test_run = request.args.get("test_run", "").strip()
        if not test_run or not re.match(r"^test\d+$", test_run):
            return jsonify({"error": "Invalid test_run"}), 400
        folder_path = resolve_under_storage(storage_path, "events", test_run)
        if folder_path is None or not os.path.isdir(folder_path):
            return jsonify({"error": "Test run folder not found"}), 404
        try:
            timeline_data = read_timeline_merged(folder_path)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        event_files: list[dict[str, str]] = []
        try:
            for f in os.listdir(folder_path):
                fp = os.path.join(folder_path, f)
                if os.path.isfile(fp):
                    event_files.append(
                        {
                            "path": f,
                            "url": f"/files/events/{test_run}/{f}",
                        }
                    )
            for sub in os.listdir(folder_path):
                sub_fp = os.path.join(folder_path, sub)
                if os.path.isdir(sub_fp) and not sub.startswith("."):
                    clip_basename = resolve_clip_in_folder(sub_fp)
                    if clip_basename:
                        url = f"/files/events/{test_run}/{sub}/{clip_basename}"
                        event_files.append(
                            {
                                "path": f"{sub}/{clip_basename}",
                                "url": url,
                            }
                        )
                    for sf in (
                        "snapshot.jpg",
                        "metadata.json",
                        "summary.txt",
                        "review_summary.md",
                        "ai_analysis_debug.zip",
                    ):
                        if os.path.isfile(os.path.join(sub_fp, sf)):
                            event_files.append(
                                {
                                    "path": f"{sub}/{sf}",
                                    "url": f"/files/events/{test_run}/{sub}/{sf}",
                                }
                            )
            event_files.sort(key=lambda x: x["path"])
        except OSError:
            pass
        return jsonify({"timeline": timeline_data, "event_files": event_files})

    @bp.route("/ai-payload")
    def test_multi_cam_ai_payload():
        """Return prompt text and image URLs for the test run (for inline AI
        request UI)."""
        test_run = request.args.get("test_run", "").strip()
        if not test_run or not re.match(r"^test\d+$", test_run):
            return jsonify({"error": "Invalid test_run"}), 400
        test_path = resolve_under_storage(storage_path, "events", test_run)
        if test_path is None or not os.path.isdir(test_path):
            return jsonify({"error": "Test run folder not found"}), 404
        prompt_path = os.path.join(test_path, "system_prompt.txt")
        if not os.path.isfile(prompt_path):
            return jsonify({"error": "system_prompt.txt not found"}), 404
        try:
            with open(prompt_path, encoding="utf-8") as f:
                prompt = f.read()
        except OSError as e:
            return jsonify({"error": str(e)}), 500
        frames_dir = os.path.join(test_path, "ai_frame_analysis", "frames")
        image_urls: list[str] = []
        if os.path.isdir(frames_dir):
            base = f"/files/events/{test_run}/ai_frame_analysis/frames"
            for name in sorted(os.listdir(frames_dir)):
                if name.startswith("frame_") and name.endswith(".jpg"):
                    image_urls.append(f"{base}/{name}")
        return jsonify({"prompt": prompt, "image_urls": image_urls})

    @bp.route("/stream")
    def test_multi_cam_stream():
        test_run = request.args.get("test_run", "").strip()
        subdir = request.args.get("subdir", "").strip()
        if test_run:
            if not re.match(r"^test\d+$", test_run):
                return jsonify({"error": "Invalid test_run"}), 400
            test_path = resolve_under_storage(storage_path, "events", test_run)
            if test_path is None or not os.path.isdir(test_path):
                return jsonify({"error": "Test run folder not found"}), 404
            source_path = test_path
            run_from_folder = True
        elif subdir:
            source_path = _resolve_source_path(storage_path, subdir)
            if source_path is None or not os.path.isdir(source_path):
                return jsonify({"error": "Invalid or missing event folder"}), 404
            try:
                with os.scandir(source_path) as it:
                    if not any(e.is_dir() for e in it):
                        return jsonify(
                            {
                                "error": "Event folder has no camera subdirs",
                            }
                        ), 400
            except OSError:
                return jsonify({"error": "Cannot read event folder"}), 400
            run_from_folder = False
        else:
            return jsonify({"error": "Missing subdir or test_run"}), 400

        def generate():
            set_suppress_review_debug_logs(True)
            try:
                if run_from_folder:
                    stream = run_test_pipeline_from_folder(
                        source_path,
                        file_manager,
                        orchestrator.video_service,
                        orchestrator.ai_analyzer,
                        config,
                    )
                else:
                    stream = run_test_pipeline(
                        source_path,
                        storage_path,
                        file_manager,
                        download_service,
                        orchestrator.video_service,
                        orchestrator.ai_analyzer,
                        config,
                    )
                for ev in stream:
                    yield f"data: {json.dumps(ev)}\n\n"
                    if ev.get("type") in ("done", "error"):
                        break
            finally:
                set_suppress_review_debug_logs(False)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @bp.route("/video-request")
    def test_multi_cam_video_request():
        """Delete clips in test folder and request new ones via Frigate Export
        API. Streams SSE."""
        test_run = request.args.get("test_run", "").strip()
        if not test_run or not re.match(r"^test\d+$", test_run):
            return jsonify({"error": "Invalid test_run"}), 400
        test_path = resolve_under_storage(storage_path, "events", test_run)
        if test_path is None or not os.path.isdir(test_path):
            return jsonify({"error": "Test run folder not found"}), 404
        if not config.get("FRIGATE_URL"):
            return jsonify({"error": "FRIGATE_URL not configured"}), 503
        export_before = config.get("EXPORT_BUFFER_BEFORE", 5)
        export_after = config.get("EXPORT_BUFFER_AFTER", 30)

        try:
            global_start, global_end = get_export_time_range_from_folder(
                test_path, config
            )
        except Exception as e:
            return jsonify({"error": f"Could not get export time range: {e}"}), 400

        try:
            with os.scandir(test_path) as it:
                all_subdirs = [
                    e.name for e in it if e.is_dir() and not e.name.startswith(".")
                ]
        except OSError:
            return jsonify({"error": "Cannot read test folder"}), 400
        camera_subdirs = list(all_subdirs)

        def generate():
            set_suppress_review_debug_logs(True)
            try:
                for cam in camera_subdirs:
                    cam_folder = os.path.join(test_path, cam)
                    if not os.path.isdir(cam_folder):
                        continue
                    for name in os.listdir(cam_folder):
                        if name.lower().endswith(".mp4"):
                            try:
                                os.remove(os.path.join(cam_folder, name))
                                payload = {
                                    "type": "log",
                                    "message": f"Deleted clip for {cam}",
                                }
                                yield f"data: {json.dumps(payload)}\n\n"
                            except OSError:
                                pass
                    sidecar = os.path.join(cam_folder, "detection.json")
                    if os.path.isfile(sidecar):
                        try:
                            os.remove(sidecar)
                        except OSError:
                            pass
                rep_id = test_run
                for cam in camera_subdirs:
                    cam_folder = os.path.join(test_path, cam)
                    canonical_camera = resolve_canonical_camera_name(
                        cam, config, file_manager
                    )
                    log_msg = f"Clip export for {cam} (video request)..."
                    yield f"data: {json.dumps({'type': 'log', 'message': log_msg})}\n\n"
                    result = download_service.export_and_download_clip(
                        rep_id,
                        cam_folder,
                        canonical_camera,
                        global_start,
                        global_end,
                        export_before,
                        export_after,
                    )
                    ok = result.get("success", False)
                    status = "success" if ok else "failed"
                    msg = f"Clip export response for {cam}: {status}"
                    yield f"data: {json.dumps({'type': 'log', 'message': msg})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            finally:
                set_suppress_review_debug_logs(False)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @bp.route("/send", methods=["POST"])
    def test_multi_cam_send():
        test_run = request.args.get("test_run", "").strip()
        if not test_run or not re.match(r"^test\d+$", test_run):
            return jsonify({"error": "Invalid test_run"}), 400
        test_path = resolve_under_storage(storage_path, "events", test_run)
        if test_path is None or not os.path.isdir(test_path):
            return jsonify({"error": "Test run folder not found"}), 404
        prompt_path = os.path.join(test_path, "system_prompt.txt")
        if not os.path.isfile(prompt_path):
            return jsonify({"error": "system_prompt.txt not found"}), 404
        try:
            with open(prompt_path, encoding="utf-8") as f:
                system_prompt = f.read()
        except OSError as e:
            return jsonify({"error": str(e)}), 500
        frames_dir = os.path.join(test_path, "ai_frame_analysis", "frames")
        frame_paths = []
        if os.path.isdir(frames_dir):
            for name in sorted(os.listdir(frames_dir)):
                if name.startswith("frame_") and name.endswith(".jpg"):
                    frame_paths.append(os.path.join(frames_dir, name))
        if not frame_paths:
            return jsonify({"error": "No frame images found"}), 404
        try:
            import cv2

            frames = []
            for p in frame_paths:
                img = cv2.imread(p)
                if img is not None:
                    frames.append(img)
            if not frames:
                return jsonify({"error": "Could not load any frame"}), 500
        except Exception as e:
            return jsonify({"error": f"Loading frames: {e}"}), 500
        result = orchestrator.ai_analyzer.send_to_proxy(system_prompt, frames)
        if result is None:
            return jsonify({"error": "Proxy request failed"}), 502

        # Persist to test event only: timeline entry, summary, metadata
        # (regular events unchanged)
        title = result.get("title") or ""
        description = result.get("shortSummary") or result.get("description") or ""
        scene = result.get("scene") or ""
        threat_level = int(result.get("potential_threat_level", 0))
        label = "unknown"
        start_time = os.path.getmtime(test_path)
        end_time = time.time()
        file_manager.append_timeline_entry(
            test_path,
            {
                "source": "test_ai_prompt",
                "direction": "out",
                "label": "Send prompt to AI (test)",
                "data": {
                    "title": title,
                    "shortSummary": description,
                    "scene": scene,
                    "end_time": end_time,
                    "start_time": start_time,
                },
            },
        )
        if title or description:
            file_manager.write_ce_summary(
                test_path,
                test_run,
                title,
                description,
                scene=scene,
                threat_level=threat_level,
                label=label,
                start_time=start_time,
            )
            file_manager.write_ce_metadata_json(
                test_path,
                test_run,
                title,
                description,
                scene=scene,
                threat_level=threat_level,
                label=label,
                camera="events",
                start_time=start_time,
                end_time=end_time,
            )
        orchestrator.query_service.evict_cache("test_events")

        return jsonify(result)

    return bp
