"""Test blueprint: /api/test-multi-cam/stream and /api/test-multi-cam/send."""

import json
import os
import re

from flask import Blueprint, Response, jsonify, request

from frigate_buffer.event_test.event_test_orchestrator import run_test_pipeline
from frigate_buffer.web.path_helpers import resolve_under_storage


def create_bp(orchestrator):
    """Create test blueprint with routes closed over orchestrator."""
    bp = Blueprint("test", __name__, url_prefix="/api/test-multi-cam")
    storage_path = orchestrator.config["STORAGE_PATH"]
    file_manager = orchestrator.file_manager

    @bp.route("/stream")
    def test_multi_cam_stream():
        subdir = request.args.get("subdir", "").strip()
        if not subdir:
            return jsonify({"error": "Missing subdir"}), 400
        source_path = resolve_under_storage(storage_path, "events", subdir)
        if source_path is None or not os.path.isdir(source_path):
            return jsonify({"error": "Invalid or missing event folder"}), 404
        try:
            with os.scandir(source_path) as it:
                has_subdir = any(e.is_dir() for e in it)
            if not has_subdir:
                return jsonify({"error": "Event folder has no camera subdirs"}), 400
        except OSError:
            return jsonify({"error": "Cannot read event folder"}), 400

        def generate():
            for ev in run_test_pipeline(
                source_path,
                storage_path,
                file_manager,
                orchestrator.download_service,
                orchestrator.video_service,
                orchestrator.ai_analyzer,
                orchestrator.config,
            ):
                yield f"data: {json.dumps(ev)}\n\n"
                if ev.get("type") in ("done", "error"):
                    break

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
            with open(prompt_path, "r", encoding="utf-8") as f:
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
        return jsonify(result)

    return bp
