"""Flask app and routes for the Frigate Event Buffer web interface."""

import os
import json
import time
import shutil
import logging
import threading
from datetime import date, datetime, timedelta

import requests
from flask import Flask, Response, send_from_directory, jsonify, render_template, request, redirect

from frigate_buffer.logging_utils import error_buffer
from frigate_buffer.services.query import EventQueryService, read_timeline_merged

logger = logging.getLogger('frigate-buffer')


def create_app(orchestrator):
    """Create Flask app with all endpoints. Routes close over orchestrator."""
    _template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    _static_dir = os.path.join(os.path.dirname(__file__), 'static')
    app = Flask(__name__, template_folder=_template_dir, static_folder=_static_dir)

    storage_path = orchestrator.config['STORAGE_PATH']
    allowed_cameras = orchestrator.config.get('ALLOWED_CAMERAS', [])
    state_manager = orchestrator.state_manager
    file_manager = orchestrator.file_manager

    query_service = EventQueryService(storage_path)

    @app.before_request
    def _count_request():
        with orchestrator._request_count_lock:
            orchestrator._request_count += 1

    @app.route('/player')
    def player():
        """Serve the event viewer page."""
        if request.args.get('filter') == 'stats':
            return redirect('/stats-page')
        return render_template('player.html',
            stats_refresh_seconds=orchestrator.config.get('STATS_REFRESH_SECONDS', 60))

    @app.route('/api/events/<event_id>/snapshot.jpg')
    def proxy_snapshot(event_id):
        """Proxy snapshot from Frigate so image_url is always buffer-based (Companion app reachability)."""
        frigate_url = orchestrator.config.get('FRIGATE_URL', '').rstrip('/')
        if not frigate_url:
            return "Frigate URL not configured", 503
        url = f"{frigate_url}/api/events/{event_id}/snapshot.jpg"
        try:
            resp = requests.get(url, timeout=15, stream=True)
            resp.raise_for_status()
            return Response(
                resp.iter_content(chunk_size=8192),
                content_type=resp.headers.get('Content-Type', 'image/jpeg'),
                status=resp.status_code
            )
        except requests.RequestException as e:
            logger.debug(f"Snapshot proxy error for {event_id}: {e}")
            return "Snapshot unavailable", 502

    @app.route('/stats-page')
    def stats_page():
        """Serve the standalone stats dashboard page."""
        return render_template('stats.html',
            stats_refresh_seconds=orchestrator.config.get('STATS_REFRESH_SECONDS', 60))

    @app.route('/daily-review')
    def daily_review_page():
        """Serve the daily review page."""
        return render_template('daily_review.html')

    @app.route('/api/daily-review/dates')
    def daily_review_dates():
        """List available daily review dates."""
        dates = orchestrator.daily_review_manager.list_dates()
        return jsonify({"dates": dates})

    @app.route('/api/daily-review/current')
    def daily_review_current():
        """Fetch current day review (midnight to now)."""
        today = date.today()
        data = orchestrator.daily_review_manager.fetch_and_save(today, end_now=True)
        if data:
            return jsonify(data)
        return jsonify({"error": "Failed to fetch current day review"}), 503

    @app.route('/api/daily-review/<date_str>')
    def daily_review_get(date_str):
        """Get cached review for date, or fetch if missing. date_str: YYYY-MM-DD."""
        try:
            d = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({"error": "Invalid date format"}), 400
        force = request.args.get('force') == '1'
        data = orchestrator.daily_review_manager.get_or_fetch(d, force_refresh=force)
        if data:
            return jsonify(data)
        return jsonify({"error": "Failed to fetch review"}), 503

    @app.route('/cameras')
    def list_cameras():
        """List available cameras from config."""
        active_cameras = query_service.get_cameras()

        all_cameras = list(set(active_cameras + [
            file_manager.sanitize_camera_name(c) for c in allowed_cameras
        ]))
        all_cameras.sort()

        return jsonify({
            "cameras": all_cameras,
            "default": all_cameras[0] if all_cameras else None
        })

    def _filter_events(events: list) -> list:
        """Apply ?filter= query param: unreviewed (default), reviewed, all."""
        f = request.args.get('filter', 'unreviewed')
        if f == 'reviewed':
            return [e for e in events if e.get('viewed')]
        elif f == 'all':
            return events
        else:
            return [e for e in events if not e.get('viewed')]

    def _maybe_cleanup():
        """Run cleanup at most once per 60 seconds to avoid full storage scan on every list request."""
        now = time.time()
        last = orchestrator._last_cleanup_time
        if last is not None and (now - last) < 60:
            return
        active_ids = state_manager.get_active_event_ids()
        active_ce_folders = list(orchestrator.consolidated_manager.get_active_ce_folders())
        deleted = file_manager.cleanup_old_events(active_ids, active_ce_folders)
        orchestrator._last_cleanup_time = now
        orchestrator._last_cleanup_deleted = deleted

    @app.route('/events/<camera>')
    def list_camera_events(camera):
        """List events for a specific camera."""
        _maybe_cleanup()

        sanitized = file_manager.sanitize_camera_name(camera)
        events = _filter_events(query_service.get_events(sanitized))

        return jsonify({
            "camera": sanitized,
            "events": events
        })

    @app.route('/events')
    def list_events():
        """List all events across all cameras (global view)."""
        _maybe_cleanup()

        try:
            all_events, cameras_found = query_service.get_all_events()
        except Exception as e:
            logger.error(f"Error listing events: {e}")
            return jsonify({"error": str(e)}), 500

        filtered = _filter_events(all_events)
        return jsonify({
            "cameras": sorted(cameras_found),
            "total_count": len(filtered),
            "events": filtered
        })

    @app.route('/delete/<path:subdir>', methods=['POST'])
    def delete_event(subdir):
        """Delete a specific event folder."""
        base_dir = os.path.realpath(storage_path)
        folder_path = os.path.realpath(os.path.join(base_dir, subdir))

        if folder_path.startswith(base_dir) and folder_path != base_dir:
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                try:
                    shutil.rmtree(folder_path)
                    logger.info(f"User manually deleted: {subdir}")
                    return jsonify({
                        "status": "success",
                        "message": f"Deleted folder: {subdir}"
                    }), 200
                except Exception as e:
                    logger.error(f"Error deleting {subdir}: {e}")
                    return jsonify({
                        "status": "error",
                        "message": str(e)
                    }), 500
            else:
                return jsonify({"status": "error", "message": "Folder not found"}), 404

        return jsonify({
            "status": "error",
            "message": "Invalid folder or path"
        }), 400

    @app.route('/viewed/<camera>/<subdir>', methods=['POST'])
    def mark_viewed(camera, subdir):
        """Mark a specific event as viewed."""
        base_dir = os.path.realpath(storage_path)
        folder_path = os.path.realpath(os.path.join(base_dir, camera, subdir))

        if not folder_path.startswith(base_dir) or folder_path == base_dir:
            return jsonify({"status": "error", "message": "Invalid path"}), 400

        if not os.path.isdir(folder_path):
            return jsonify({"status": "error", "message": "Event not found"}), 404

        viewed_path = os.path.join(folder_path, '.viewed')
        try:
            open(viewed_path, 'a').close()
            return jsonify({"status": "success"}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/viewed/<camera>/<subdir>', methods=['DELETE'])
    def unmark_viewed(camera, subdir):
        """Remove viewed marker from a specific event."""
        base_dir = os.path.realpath(storage_path)
        folder_path = os.path.realpath(os.path.join(base_dir, camera, subdir))

        if not folder_path.startswith(base_dir) or folder_path == base_dir:
            return jsonify({"status": "error", "message": "Invalid path"}), 400

        viewed_path = os.path.join(folder_path, '.viewed')
        if os.path.exists(viewed_path):
            os.remove(viewed_path)
        return jsonify({"status": "success"}), 200

    @app.route('/viewed/all', methods=['POST'])
    def mark_all_viewed():
        """Mark ALL events across all cameras as viewed."""
        count = 0
        try:
            with os.scandir(storage_path) as it:
                for camera_entry in it:
                    if not camera_entry.is_dir():
                        continue
                    camera_dir = camera_entry.name
                    if camera_dir.split('_')[0].isdigit():
                        continue

                    with os.scandir(camera_entry.path) as it_events:
                        for event_entry in it_events:
                            if event_entry.is_dir():
                                viewed_path = os.path.join(event_entry.path, '.viewed')
                                if not os.path.exists(viewed_path):
                                    open(viewed_path, 'a').close()
                                    count += 1
        except Exception as e:
            logger.error(f"Error marking all viewed: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

        return jsonify({"status": "success", "marked": count}), 200

    @app.route('/events/<camera>/<subdir>/timeline')
    def event_timeline(camera, subdir):
        """Serve the notification timeline page for an event (most recent first)."""
        base_dir = os.path.realpath(storage_path)
        folder_path = os.path.realpath(os.path.join(base_dir, camera, subdir))

        if not folder_path.startswith(base_dir) or folder_path == base_dir:
            return "Invalid path", 400

        if not os.path.isdir(folder_path):
            return "Event not found", 404

        try:
            timeline_data = read_timeline_merged(folder_path)
        except Exception as e:
            logger.debug(f"Error reading timeline: {e}")
            timeline_data = {"event_id": None, "entries": []}

        entries = timeline_data.get("entries", [])
        entries.sort(key=lambda e: e.get("ts", ""), reverse=True)

        event_files = []
        try:
            for f in os.listdir(folder_path):
                fp = os.path.join(folder_path, f)
                if os.path.isfile(fp):
                    event_files.append(f)
            # Include files from camera subdirs (consolidated: Carport/clip.mp4, etc.)
            for sub in os.listdir(folder_path):
                sub_fp = os.path.join(folder_path, sub)
                if os.path.isdir(sub_fp) and not sub.startswith('.'):
                    for sf in ('clip.mp4', 'snapshot.jpg', 'metadata.json', 'summary.txt', 'review_summary.md', 'ai_analysis_debug.zip'):
                        if os.path.isfile(os.path.join(sub_fp, sf)):
                            event_files.append(f"{sub}/{sf}")
            event_files.sort()
        except OSError:
            pass
        zip_entries = [f for f in event_files if f == 'ai_analysis_debug.zip' or f.endswith('/ai_analysis_debug.zip')]
        has_ai_analysis_zip = bool(zip_entries)
        first_ai_analysis_zip_path = zip_entries[0] if zip_entries else None

        # Export duration and clip files for intro line
        export_duration_seconds = None
        export_file_list = [f for f in event_files if f == 'clip.mp4' or f.endswith('/clip.mp4')]
        request_entries = [e for e in entries if e.get("source") == "frigate_api" and e.get("direction") == "out" and "Clip export request" in (e.get("label") or "")]
        response_entries = [e for e in entries if e.get("source") == "frigate_api" and e.get("direction") == "in" and "Clip export response" in (e.get("label") or "")]
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
            'timeline.html',
            event_id=timeline_data.get("event_id", subdir),
            camera=camera,
            subdir=subdir,
            entries=entries,
            event_files=event_files,
            export_duration_seconds=export_duration_seconds,
            export_file_list=export_file_list,
            has_ai_analysis_zip=has_ai_analysis_zip,
            first_ai_analysis_zip_path=first_ai_analysis_zip_path
        )

    @app.route('/files/<path:filename>')
    def serve_file(filename):
        """Serve stored files (clips are already transcoded to H.264)."""
        directory = os.path.realpath(storage_path)
        safe_path = os.path.realpath(os.path.join(directory, filename))

        if not safe_path.startswith(directory):
            return "File not found", 404

        return send_from_directory(directory, filename)

    @app.route('/stats')
    def stats():
        """Return stats for the player dashboard (events, storage, errors, system)."""
        now = time.time()
        day_start = now - 86400
        week_start = now - 604800
        month_start = now - 2592000

        events_today = events_week = events_month = 0
        total_reviewed = total_unreviewed = 0
        by_camera = {}
        most_recent = None

        try:
            # Top-level dirs: legacy cameras (e.g. carport) and "events" for consolidated events.
            # For "events", each direct child is a CE folder (events/ce_id/); we count one event per CE.
            with os.scandir(storage_path) as it:
                for camera_entry in it:
                    if not camera_entry.is_dir():
                        continue
                    camera_dir = camera_entry.name
                    if camera_dir.split('_')[0].isdigit():
                        continue

                    count = 0
                    with os.scandir(camera_entry.path) as it_events:
                        for event_entry in it_events:
                            if not event_entry.is_dir():
                                continue

                            event_dir = event_entry.name
                            try:
                                parts = event_dir.split('_', 1)
                                ts = float(parts[0])
                            except (ValueError, IndexError):
                                continue

                            viewed = os.path.exists(os.path.join(event_entry.path, '.viewed'))
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

                            if most_recent is None or ts > most_recent['timestamp']:
                                most_recent = {
                                    'event_id': parts[1] if len(parts) > 1 else event_dir,
                                    'camera': camera_dir,
                                    'subdir': event_dir,
                                    'timestamp': ts
                                }

                    by_camera[camera_dir] = count
        except Exception as e:
            logger.error(f"Error scanning events for stats: {e}")

        storage_raw = orchestrator._cached_storage_stats
        kb, mb = 1024, 1024 * 1024
        gb = 1024 * mb

        def fmt_size(b):
            """Return {value, unit} with unit KB/MB/GB for human-readable display (e.g. 500 KB not 0.5 MB)."""
            if b <= 0:
                return {'value': 0, 'unit': 'KB'}
            if b < mb:
                return {'value': round(b / kb, 2), 'unit': 'KB'}
            if b < gb:
                return {'value': round(b / mb, 2), 'unit': 'MB'}
            return {'value': round(b / gb, 2), 'unit': 'GB'}

        by_camera_storage = {}
        for cam, data in storage_raw.get('by_camera', {}).items():
            by_camera_storage[cam] = fmt_size(data['total'])

        total_bytes = storage_raw.get('total', 0)
        breakdown = {
            'clips': fmt_size(storage_raw.get('clips', 0)),
            'snapshots': fmt_size(storage_raw.get('snapshots', 0)),
            'descriptions': fmt_size(storage_raw.get('descriptions', 0))
        }

        most_recent_out = None
        if most_recent:
            most_recent_out = {
                'event_id': most_recent['event_id'],
                'camera': most_recent['camera'],
                'url': '/player?filter=all',
                'timestamp': most_recent['timestamp']
            }

        last_cleanup = None
        if orchestrator._last_cleanup_time is not None:
            last_cleanup = {
                'at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(orchestrator._last_cleanup_time)),
                'deleted': orchestrator._last_cleanup_deleted
            }

        ha_helpers = None
        ha_url = orchestrator.config.get('HA_URL', '').rstrip('/')
        ha_token = orchestrator.config.get('HA_TOKEN', '')
        if ha_url and ha_token:
            try:
                cost_entity = orchestrator.config.get('HA_GEMINI_COST_ENTITY', 'input_number.gemini_daily_cost')
                tokens_entity = orchestrator.config.get('HA_GEMINI_TOKENS_ENTITY', 'input_number.gemini_total_tokens')
                cost_val = orchestrator._fetch_ha_state(ha_url, ha_token, cost_entity)
                tokens_val = orchestrator._fetch_ha_state(ha_url, ha_token, tokens_entity)
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
                        'gemini_month_cost': gemini_cost,
                        'gemini_month_tokens': gemini_tokens
                    }
            except Exception as e:
                logger.debug(f"Failed to fetch HA helpers for stats: {e}")

        response_data = {
            'events': {
                'today': events_today,
                'this_week': events_week,
                'this_month': events_month,
                'total_reviewed': total_reviewed,
                'total_unreviewed': total_unreviewed,
                'by_camera': by_camera
            },
            'storage': {
                'total_display': fmt_size(total_bytes),
                'by_camera': by_camera_storage,
                'breakdown': breakdown
            },
            'errors': error_buffer.get_all(),
            'last_cleanup': last_cleanup,
            'most_recent': most_recent_out,
            'system': {
                'uptime_seconds': int(time.time() - orchestrator._start_time),
                'mqtt_connected': orchestrator.mqtt_wrapper.mqtt_connected,
                'active_events': len(orchestrator.state_manager.get_active_event_ids()),
                'retention_days': orchestrator.config['RETENTION_DAYS'],
                'cleanup_interval_hours': orchestrator.config.get('CLEANUP_INTERVAL_HOURS', 1),
                'storage_path': orchestrator.config['STORAGE_PATH'],
                'stats_refresh_seconds': orchestrator.config.get('STATS_REFRESH_SECONDS', 60)
            }
        }
        if ha_helpers is not None:
            response_data['ha_helpers'] = ha_helpers
        return jsonify(response_data)

    @app.route('/status')
    def status():
        """Return orchestrator status for monitoring."""
        uptime_seconds = time.time() - orchestrator._start_time
        uptime_str = str(timedelta(seconds=int(uptime_seconds)))

        # Gather active consolidated events
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
                    "start_time": ce.start_time
                })
        except Exception as e:
            logger.error(f"Error gathering consolidated events for status: {e}")

        # Gather metrics
        metrics = {
            "notification_queue_size": getattr(orchestrator.notifier, 'queue_size', 0),
            "active_threads": threading.active_count(),
            "active_consolidated_events": active_ce,
            "recent_errors": error_buffer.get_all()[:5]
        }

        return jsonify({
            "online": True,
            "mqtt_connected": orchestrator.mqtt_wrapper.mqtt_connected,
            "uptime_seconds": uptime_seconds,
            "uptime": uptime_str,
            "started_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(orchestrator._start_time)),
            "active_events": state_manager.get_stats(),
            "metrics": metrics,
            "config": {
                "retention_days": orchestrator.config['RETENTION_DAYS'],
                "log_level": orchestrator.config.get('LOG_LEVEL', 'INFO'),
                "ffmpeg_timeout": orchestrator.config.get('FFMPEG_TIMEOUT', 60),
                "summary_padding_before": orchestrator.config.get('SUMMARY_PADDING_BEFORE', 15),
                "summary_padding_after": orchestrator.config.get('SUMMARY_PADDING_AFTER', 15)
            }
        })

    return app
