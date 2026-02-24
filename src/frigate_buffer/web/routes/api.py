"""API blueprint: cameras, events, delete, viewed, timeline, files, stats, status."""

import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timedelta

from flask import Blueprint, Response, jsonify, render_template, request, send_from_directory

from frigate_buffer.constants import NON_CAMERA_DIRS
from frigate_buffer.logging_utils import error_buffer
from frigate_buffer.services.query import EventQueryService, read_timeline_merged, resolve_clip_in_folder
from frigate_buffer.web.path_helpers import resolve_under_storage

logger = logging.getLogger("frigate-buffer")


def create_bp(orchestrator):
    """Create API blueprint with routes closed over orchestrator."""
    bp = Blueprint("api", __name__)
    storage_path = orchestrator.config["STORAGE_PATH"]
    allowed_cameras = orchestrator.config.get("ALLOWED_CAMERAS", [])
    state_manager = orchestrator.state_manager
    file_manager = orchestrator.file_manager
    query_service = EventQueryService(storage_path)

    def _filter_events(events: list) -> list:
        f = request.args.get("filter", "unreviewed")
        if f == "reviewed":
            return [e for e in events if e.get("viewed")]
        if f == "all":
            return events
        if f == "saved":
            return events  # Already the saved list from get_saved_events
        if f == "test_events":
            return events  # Already the test list from get_test_events
        return [e for e in events if not e.get("viewed")]

    def _maybe_cleanup():
        now = time.time()
        last = orchestrator._last_cleanup_time
        if last is not None and (now - last) < 60:
            return
        active_ids = state_manager.get_active_event_ids()
        active_ce_folders = list(orchestrator.consolidated_manager.get_active_ce_folders())
        deleted = file_manager.cleanup_old_events(active_ids, active_ce_folders)
        orchestrator._last_cleanup_time = now
        orchestrator._last_cleanup_deleted = deleted

    @bp.route("/cameras")
    def list_cameras():
        active_cameras = query_service.get_cameras()
        all_cameras = list(
            set(active_cameras + [file_manager.sanitize_camera_name(c) for c in allowed_cameras])
        )
        all_cameras.sort()
        return jsonify({"cameras": all_cameras, "default": all_cameras[0] if all_cameras else None})

    @bp.route("/events/<camera>")
    def list_camera_events(camera):
        _maybe_cleanup()
        f = request.args.get("filter", "unreviewed")
        if f == "saved":
            sanitized = file_manager.sanitize_camera_name(camera)
            events = query_service.get_saved_events(camera=sanitized)
            cameras_found = list({e.get("camera") for e in events}) if events else [sanitized]
            return jsonify({"camera": sanitized, "events": events, "cameras": sorted(cameras_found), "total_count": len(events)})
        if f == "test_events":
            events = query_service.get_test_events()
            events = [e for e in events if e.get("camera") == camera]
            return jsonify({"camera": camera, "events": events, "cameras": [camera], "total_count": len(events)})
        sanitized = file_manager.sanitize_camera_name(camera)
        events = _filter_events(query_service.get_events(sanitized))
        return jsonify({"camera": sanitized, "events": events})

    @bp.route("/events")
    def list_events():
        _maybe_cleanup()
        f = request.args.get("filter", "unreviewed")
        if f == "saved":
            try:
                events = query_service.get_saved_events(camera=None)
                cameras_found = sorted({e.get("camera") for e in events if e.get("camera")})
            except Exception as e:
                logger.error("Error listing saved events: %s", e)
                return jsonify({"error": str(e)}), 500
            return jsonify({
                "cameras": cameras_found,
                "total_count": len(events),
                "events": events,
            })
        if f == "test_events":
            try:
                events = query_service.get_test_events()
                cameras_found = ["events"] if events else []
            except Exception as e:
                logger.error("Error listing test events: %s", e)
                return jsonify({"error": str(e)}), 500
            return jsonify({
                "cameras": cameras_found,
                "total_count": len(events),
                "events": events,
            })
        try:
            all_events, cameras_found = query_service.get_all_events()
        except Exception as e:
            logger.error("Error listing events: %s", e)
            return jsonify({"error": str(e)}), 500
        filtered = _filter_events(all_events)
        return jsonify({
            "cameras": sorted(cameras_found),
            "total_count": len(filtered),
            "events": filtered,
        })

    @bp.route("/keep/<path:event_path>", methods=["POST"])
    def keep_event(event_path):
        """Move event folder to saved/ (excluded from retention). Source must not be under saved/."""
        path_parts = event_path.split("/")
        if not path_parts:
            return jsonify({"status": "error", "message": "Invalid path"}), 400
        if path_parts[0] == "saved":
            return jsonify({"status": "error", "message": "Event is already saved"}), 400
        source_path = resolve_under_storage(storage_path, *path_parts)
        if source_path is None or not os.path.isdir(source_path):
            return jsonify({"status": "error", "message": "Event not found or invalid path"}), 404
        # Destination: saved/<camera>/<subdir> or saved/events/<ce_id>
        dest_path = resolve_under_storage(storage_path, "saved", *path_parts)
        if dest_path is None:
            return jsonify({"status": "error", "message": "Invalid destination path"}), 400
        if os.path.exists(dest_path):
            return jsonify({"status": "error", "message": "Saved event already exists"}), 409
        try:
            dest_parent = os.path.dirname(dest_path)
            os.makedirs(dest_parent, exist_ok=True)
            shutil.move(source_path, dest_path)
            logger.info("Kept event: %s -> saved/%s", event_path, event_path)
            return jsonify({"status": "success", "message": f"Moved to saved/{event_path}"}), 200
        except Exception as e:
            logger.error("Error keeping event %s: %s", event_path, e)
            return jsonify({"status": "error", "message": str(e)}), 500

    @bp.route("/delete/<path:subdir>", methods=["POST"])
    def delete_event(subdir):
        # Support multi-segment paths (e.g. saved/camera/subdir) for cross-platform resolve.
        path_parts = subdir.split("/")
        folder_path = resolve_under_storage(storage_path, *path_parts) if path_parts else None
        if folder_path is not None:
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                try:
                    shutil.rmtree(folder_path)
                    logger.info("User manually deleted: %s", subdir)
                    return jsonify({"status": "success", "message": f"Deleted folder: {subdir}"}), 200
                except Exception as e:
                    logger.error("Error deleting %s: %s", subdir, e)
                    return jsonify({"status": "error", "message": str(e)}), 500
            return jsonify({"status": "error", "message": "Folder not found"}), 404
        return jsonify({"status": "error", "message": "Invalid folder or path"}), 400

    @bp.route("/viewed/<path:event_path>", methods=["POST"])
    def mark_viewed(event_path):
        path_parts = event_path.split("/")
        folder_path = resolve_under_storage(storage_path, *path_parts) if path_parts else None
        if folder_path is None:
            return jsonify({"status": "error", "message": "Invalid path"}), 400
        if not os.path.isdir(folder_path):
            return jsonify({"status": "error", "message": "Event not found"}), 404
        viewed_path = os.path.join(folder_path, ".viewed")
        try:
            open(viewed_path, "a").close()
            return jsonify({"status": "success"}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @bp.route("/viewed/<path:event_path>", methods=["DELETE"])
    def unmark_viewed(event_path):
        path_parts = event_path.split("/")
        folder_path = resolve_under_storage(storage_path, *path_parts) if path_parts else None
        if folder_path is None:
            return jsonify({"status": "error", "message": "Invalid path"}), 400
        viewed_path = os.path.join(folder_path, ".viewed")
        if os.path.exists(viewed_path):
            os.remove(viewed_path)
        return jsonify({"status": "success"}), 200

    @bp.route("/viewed/all", methods=["POST"])
    def mark_all_viewed():
        count = 0
        try:
            with os.scandir(storage_path) as it:
                for camera_entry in it:
                    if not camera_entry.is_dir():
                        continue
                    camera_dir = camera_entry.name
                    if camera_dir.split("_")[0].isdigit():
                        continue
                    if camera_dir in NON_CAMERA_DIRS:
                        continue
                    with os.scandir(camera_entry.path) as it_events:
                        for event_entry in it_events:
                            if event_entry.is_dir():
                                viewed_path = os.path.join(event_entry.path, ".viewed")
                                if not os.path.exists(viewed_path):
                                    open(viewed_path, "a").close()
                                    count += 1
        except Exception as e:
            logger.error("Error marking all viewed: %s", e)
            return jsonify({"status": "error", "message": str(e)}), 500
        return jsonify({"status": "success", "marked": count}), 200

    @bp.route("/events/<path:event_path>/timeline")
    def event_timeline(event_path):
        path_parts = event_path.split("/")
        folder_path = resolve_under_storage(storage_path, *path_parts) if path_parts else None
        camera = path_parts[0] if len(path_parts) >= 1 else ""
        subdir = "/".join(path_parts[1:]) if len(path_parts) > 1 else (path_parts[0] if path_parts else "")
        if folder_path is None:
            return "Invalid path", 400
        if not os.path.isdir(folder_path):
            return "Event not found", 404
        try:
            timeline_data = read_timeline_merged(folder_path)
        except Exception as e:
            logger.debug("Error reading timeline: %s", e)
            timeline_data = {"event_id": None, "entries": []}
        entries = timeline_data.get("entries", [])
        entries.sort(key=lambda e: e.get("ts", ""), reverse=True)
        event_files = []
        try:
            for f in os.listdir(folder_path):
                fp = os.path.join(folder_path, f)
                if os.path.isfile(fp):
                    event_files.append(f)
            for sub in os.listdir(folder_path):
                sub_fp = os.path.join(folder_path, sub)
                if os.path.isdir(sub_fp) and not sub.startswith("."):
                    clip_basename = resolve_clip_in_folder(sub_fp)
                    if clip_basename:
                        event_files.append(f"{sub}/{clip_basename}")
                    for sf in (
                        "snapshot.jpg",
                        "metadata.json",
                        "summary.txt",
                        "review_summary.md",
                        "ai_analysis_debug.zip",
                    ):
                        if os.path.isfile(os.path.join(sub_fp, sf)):
                            event_files.append(f"{sub}/{sf}")
            event_files.sort()
        except OSError:
            pass
        zip_entries = [
            f for f in event_files
            if f == "ai_analysis_debug.zip" or f.endswith("/ai_analysis_debug.zip")
        ]
        has_ai_analysis_zip = bool(zip_entries)
        first_ai_analysis_zip_path = zip_entries[0] if zip_entries else None
        export_duration_seconds = None
        export_file_list = [f for f in event_files if f.endswith(".mp4")]
        request_entries = [
            e for e in entries
            if e.get("source") == "frigate_api"
            and e.get("direction") == "out"
            and "Clip export request" in (e.get("label") or "")
        ]
        response_entries = [
            e for e in entries
            if e.get("source") == "frigate_api"
            and e.get("direction") == "in"
            and "Clip export response" in (e.get("label") or "")
        ]
        if request_entries and response_entries:
            try:
                ts_fmt = "%Y-%m-%dT%H:%M:%S"
                first_request_ts = min(e.get("ts", "") for e in request_entries)
                last_response_ts = max(e.get("ts", "") for e in response_entries)
                if first_request_ts and last_response_ts:
                    t0 = datetime.strptime(first_request_ts[:19], ts_fmt)
                    t1 = datetime.strptime(last_response_ts[:19], ts_fmt)
                    export_duration_seconds = (t1 - t0).total_seconds()
            except (ValueError, TypeError):
                pass
        return render_template(
            "timeline.html",
            event_id=timeline_data.get("event_id", subdir),
            camera=camera,
            subdir=subdir,
            entries=entries,
            event_files=event_files,
            export_duration_seconds=export_duration_seconds,
            export_file_list=export_file_list,
            has_ai_analysis_zip=has_ai_analysis_zip,
            first_ai_analysis_zip_path=first_ai_analysis_zip_path,
        )

    @bp.route("/events/<path:event_path>/timeline/download")
    def event_timeline_download(event_path):
        path_parts = event_path.split("/")
        folder_path = resolve_under_storage(storage_path, *path_parts) if path_parts else None
        subdir = path_parts[-1] if len(path_parts) >= 2 else (path_parts[0] if path_parts else "")
        if folder_path is None:
            return "Invalid path", 400
        if not os.path.isdir(folder_path):
            return "Event not found", 404
        try:
            timeline_data = read_timeline_merged(folder_path)
        except Exception as e:
            logger.debug("Error reading timeline for download: %s", e)
            return "Error reading timeline", 500
        body = json.dumps(timeline_data, indent=2)
        return Response(
            body,
            mimetype="application/json",
            headers={"Content-Disposition": 'attachment; filename="notification_timeline.json"'},
        )

    @bp.route("/files/<path:filename>")
    def serve_file(filename):
        path_parts = filename.split("/")
        safe_path = resolve_under_storage(storage_path, *path_parts) if path_parts else None
        if safe_path is None:
            return "File not found", 404
        return send_from_directory(os.path.realpath(storage_path), filename)

    @bp.route("/stats")
    def stats():
        now = time.time()
        day_start = now - 86400
        week_start = now - 604800
        month_start = now - 2592000
        events_today = events_week = events_month = 0
        total_reviewed = total_unreviewed = 0
        by_camera = {}
        most_recent = None
        try:
            with os.scandir(storage_path) as it:
                for camera_entry in it:
                    if not camera_entry.is_dir():
                        continue
                    camera_dir = camera_entry.name
                    if camera_dir.split("_")[0].isdigit():
                        continue
                    if camera_dir in NON_CAMERA_DIRS:
                        continue
                    count = 0
                    with os.scandir(camera_entry.path) as it_events:
                        for event_entry in it_events:
                            if not event_entry.is_dir():
                                continue
                            event_dir = event_entry.name
                            try:
                                parts = event_dir.split("_", 1)
                                ts = float(parts[0])
                            except (ValueError, IndexError):
                                continue
                            viewed = os.path.exists(os.path.join(event_entry.path, ".viewed"))
                            if viewed:
                                total_reviewed += 1
                            else:
                                total_unreviewed += 1
                            count += 1
                            if ts >= day_start:
                                events_today += 1
                            if ts >= week_start:
                                events_week += 1
                            if ts >= month_start:
                                events_month += 1
                            if most_recent is None or ts > most_recent["timestamp"]:
                                most_recent = {
                                    "event_id": parts[1] if len(parts) > 1 else event_dir,
                                    "camera": camera_dir,
                                    "subdir": event_dir,
                                    "timestamp": ts,
                                }
                    by_camera[camera_dir] = count
        except Exception as e:
            logger.error("Error scanning events for stats: %s", e)
        storage_raw = orchestrator.get_storage_stats()
        kb, mb = 1024, 1024 * 1024
        gb = 1024 * mb

        def fmt_size(b):
            if b <= 0:
                return {"value": 0, "unit": "KB"}
            if b < mb:
                return {"value": round(b / kb, 2), "unit": "KB"}
            if b < gb:
                return {"value": round(b / mb, 2), "unit": "MB"}
            return {"value": round(b / gb, 2), "unit": "GB"}

        by_camera_storage = {}
        for cam, data in storage_raw.get("by_camera", {}).items():
            by_camera_storage[cam] = fmt_size(data["total"])
        total_bytes = storage_raw.get("total", 0)
        breakdown = {
            "clips": fmt_size(storage_raw.get("clips", 0)),
            "snapshots": fmt_size(storage_raw.get("snapshots", 0)),
            "descriptions": fmt_size(storage_raw.get("descriptions", 0)),
        }
        most_recent_out = None
        if most_recent:
            most_recent_out = {
                "event_id": most_recent["event_id"],
                "camera": most_recent["camera"],
                "url": "/player?filter=all",
                "timestamp": most_recent["timestamp"],
            }
        last_cleanup = None
        if orchestrator._last_cleanup_time is not None:
            last_cleanup = {
                "at": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(orchestrator._last_cleanup_time)
                ),
                "deleted": orchestrator._last_cleanup_deleted,
            }
        ha_helpers = None
        ha_url = orchestrator.config.get("HA_URL", "").rstrip("/")
        ha_token = orchestrator.config.get("HA_TOKEN", "")
        if ha_url and ha_token:
            try:
                cost_entity = orchestrator.config.get(
                    "HA_GEMINI_COST_ENTITY", "input_number.gemini_daily_cost"
                )
                tokens_entity = orchestrator.config.get(
                    "HA_GEMINI_TOKENS_ENTITY", "input_number.gemini_total_tokens"
                )
                cost_val = orchestrator.fetch_ha_state(ha_url, ha_token, cost_entity)
                tokens_val = orchestrator.fetch_ha_state(ha_url, ha_token, tokens_entity)
                gemini_cost = None
                if cost_val is not None:
                    try:
                        gemini_cost = float(cost_val)
                    except (TypeError, ValueError):
                        pass
                gemini_tokens = None
                if tokens_val is not None:
                    try:
                        gemini_tokens = int(float(tokens_val))
                    except (TypeError, ValueError):
                        pass
                if gemini_cost is not None or gemini_tokens is not None:
                    ha_helpers = {
                        "gemini_month_cost": gemini_cost,
                        "gemini_month_tokens": gemini_tokens,
                    }
            except Exception as e:
                logger.debug("Failed to fetch HA helpers for stats: %s", e)
        response_data = {
            "events": {
                "today": events_today,
                "this_week": events_week,
                "this_month": events_month,
                "total_reviewed": total_reviewed,
                "total_unreviewed": total_unreviewed,
                "by_camera": by_camera,
            },
            "storage": {
                "total_display": fmt_size(total_bytes),
                "by_camera": by_camera_storage,
                "breakdown": breakdown,
            },
            "errors": error_buffer.get_all(),
            "last_cleanup": last_cleanup,
            "most_recent": most_recent_out,
            "system": {
                "uptime_seconds": int(time.time() - orchestrator._start_time),
                "mqtt_connected": orchestrator.mqtt_wrapper.mqtt_connected,
                "active_events": len(orchestrator.state_manager.get_active_event_ids()),
                "retention_days": orchestrator.config["RETENTION_DAYS"],
                "cleanup_interval_hours": orchestrator.config.get("CLEANUP_INTERVAL_HOURS", 1),
                "storage_path": orchestrator.config["STORAGE_PATH"],
                "stats_refresh_seconds": orchestrator.config.get("STATS_REFRESH_SECONDS", 60),
            },
        }
        if ha_helpers is not None:
            response_data["ha_helpers"] = ha_helpers
        return jsonify(response_data)

    @bp.route("/status")
    def status():
        uptime_seconds = time.time() - orchestrator._start_time
        uptime_str = str(timedelta(seconds=int(uptime_seconds)))
        active_ce = []
        try:
            for ce in orchestrator.consolidated_manager.get_all():
                state = "active"
                if ce.closing:
                    state = "closing"
                if ce.closed:
                    state = "closed"
                active_ce.append({
                    "id": ce.consolidated_id,
                    "state": state,
                    "cameras": ce.cameras,
                    "start_time": ce.start_time,
                })
        except Exception as e:
            logger.error("Error gathering consolidated events for status: %s", e)
        metrics = {
            "notification_queue_size": getattr(orchestrator.notifier, "queue_size", 0),
            "active_threads": threading.active_count(),
            "active_consolidated_events": active_ce,
            "recent_errors": error_buffer.get_all()[:5],
        }
        return jsonify({
            "online": True,
            "mqtt_connected": orchestrator.mqtt_wrapper.mqtt_connected,
            "uptime_seconds": uptime_seconds,
            "uptime": uptime_str,
            "started_at": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(orchestrator._start_time)
            ),
            "active_events": state_manager.get_stats(),
            "metrics": metrics,
            "config": {
                "retention_days": orchestrator.config["RETENTION_DAYS"],
                "log_level": orchestrator.config.get("LOG_LEVEL", "INFO"),
                "ffmpeg_timeout": orchestrator.config.get("FFMPEG_TIMEOUT", 60),
                "summary_padding_before": orchestrator.config.get("SUMMARY_PADDING_BEFORE", 15),
                "summary_padding_after": orchestrator.config.get("SUMMARY_PADDING_AFTER", 15),
            },
        })

    return bp
