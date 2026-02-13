"""Flask app and routes for the Frigate Event Buffer web interface."""

import os
import json
import time
import shutil
import logging
from datetime import date, datetime, timedelta

import requests
from flask import Flask, Response, send_from_directory, jsonify, render_template, request, redirect

from frigate_buffer.logging_utils import error_buffer

logger = logging.getLogger('frigate-buffer')


def create_app(orchestrator):
    """Create Flask app with all endpoints. Routes close over orchestrator."""
    _template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
    _static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
    app = Flask(__name__, template_folder=_template_dir, static_folder=_static_dir)

    storage_path = orchestrator.config['STORAGE_PATH']
    allowed_cameras = orchestrator.config.get('ALLOWED_CAMERAS', [])
    state_manager = orchestrator.state_manager
    file_manager = orchestrator.file_manager

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

    def _parse_summary(summary_text: str) -> dict:
        """Parse key-value pairs from summary.txt format."""
        parsed = {}
        for line in summary_text.split('\n'):
            if ':' in line:
                key, _, value = line.partition(':')
                parsed[key.strip()] = value.strip()
        return parsed

    def _extract_genai_entries(folder_path: str) -> list:
        """Extract GenAI metadata entries from notification_timeline.json."""
        entries = []
        timeline_path = os.path.join(folder_path, 'notification_timeline.json')
        if not os.path.exists(timeline_path):
            return entries
        try:
            with open(timeline_path, 'r') as f:
                data = json.load(f)
        except Exception:
            return entries
        for e in data.get('entries', []):
            payload = (e.get('data') or {}).get('payload') or {}
            if payload.get('type') != 'genai':
                continue
            after = payload.get('after') or {}
            meta = (after.get('data') or {}).get('metadata')
            if not meta:
                continue
            title = meta.get('title') or ''
            scene = meta.get('scene') or ''
            short_summary = meta.get('shortSummary') or meta.get('description') or ''
            # Skip boilerplate "no concerns/activity" entries to avoid noise
            lower = (title + ' ' + short_summary + ' ' + scene).lower()
            if 'no concerns' in lower or 'no activity' in lower:
                if not title and not scene and len(short_summary) < 80:
                    continue
            entries.append({
                'title': title,
                'scene': scene,
                'shortSummary': short_summary,
                'time': meta.get('time'),
                'potential_threat_level': meta.get('potential_threat_level', 0),
            })
        return entries

    def _get_events_from_consolidated() -> list:
        """Get consolidated events from events/{ce_id}/{camera}/ structure."""
        events_dir = os.path.join(storage_path, "events")
        events_list = []
        if not os.path.isdir(events_dir):
            return events_list

        for ce_id in sorted(os.listdir(events_dir), reverse=True):
            ce_path = os.path.join(events_dir, ce_id)
            if not os.path.isdir(ce_path):
                continue
            if '.' in ce_id and not ce_id.split('_')[0].isdigit():
                continue

            parts = ce_id.split('_', 1)
            ts = parts[0] if len(parts) > 0 else "0"
            summary_path = os.path.join(ce_path, 'summary.txt')
            metadata_path = os.path.join(ce_path, 'metadata.json')
            review_summary_path = os.path.join(ce_path, 'review_summary.md')
            viewed_path = os.path.join(ce_path, '.viewed')

            summary_text = "Analysis pending..."
            parsed = {}
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary_text = f.read().strip()
                parsed = _parse_summary(summary_text)

            cameras = [d for d in os.listdir(ce_path)
                      if os.path.isdir(os.path.join(ce_path, d)) and not d.startswith('.')]

            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except Exception:
                    pass
            # Consolidated: metadata may live in primary camera folder
            if not metadata and cameras:
                for cam in cameras:
                    cam_meta = os.path.join(ce_path, cam, 'metadata.json')
                    if os.path.exists(cam_meta):
                        try:
                            with open(cam_meta, 'r') as f:
                                metadata = json.load(f)
                            break
                        except Exception:
                            pass

            genai_entries = _extract_genai_entries(ce_path)

            review_summary = None
            if os.path.exists(review_summary_path):
                with open(review_summary_path, 'r') as f:
                    review_summary = f.read().strip()
            primary_clip = primary_snapshot = None
            for cam in cameras:
                cam_path = os.path.join(ce_path, cam)
                if os.path.exists(os.path.join(cam_path, 'clip.mp4')):
                    primary_clip = f"/files/events/{ce_id}/{cam}/clip.mp4"
                    break
            for cam in cameras:
                cam_path = os.path.join(ce_path, cam)
                if os.path.exists(os.path.join(cam_path, 'snapshot.jpg')):
                    primary_snapshot = f"/files/events/{ce_id}/{cam}/snapshot.jpg"
                    break

            has_clip = any(os.path.exists(os.path.join(ce_path, cam, 'clip.mp4')) for cam in cameras)
            has_snapshot = any(os.path.exists(os.path.join(ce_path, cam, 'snapshot.jpg')) for cam in cameras)

            # Time-based "not ongoing": events older than 90 min are treated as finalized
            try:
                ts_float = float(ts)
                age_seconds = time.time() - ts_float
                ongoing = not has_clip and age_seconds < 5400  # 90 min
            except (ValueError, TypeError):
                ongoing = not has_clip

            events_list.append({
                "event_id": ce_id,
                "camera": "events",
                "subdir": ce_id,
                "timestamp": ts,
                "summary": summary_text,
                "title": metadata.get("genai_title") or parsed.get("Title"),
                "description": metadata.get("genai_description") or parsed.get("Description") or parsed.get("AI Description"),
                "scene": metadata.get("genai_scene") or parsed.get("Scene"),
                "label": metadata.get("label") or parsed.get("Label", "unknown"),
                "severity": metadata.get("severity") or parsed.get("Severity"),
                "threat_level": metadata.get("threat_level", 0),
                "review_summary": review_summary,
                "has_clip": has_clip,
                "has_snapshot": has_snapshot,
                "viewed": os.path.exists(viewed_path),
                "hosted_clip": primary_clip or (f"/files/events/{ce_id}/{cameras[0]}/clip.mp4" if cameras else None),
                "hosted_snapshot": primary_snapshot or (f"/files/events/{ce_id}/{cameras[0]}/snapshot.jpg" if cameras else None),
                "cameras": cameras,
                "consolidated": True,
                "ongoing": ongoing,
                "genai_entries": genai_entries
            })

        return events_list

    def _get_events_for_camera(camera_name: str) -> list:
        """Helper to get events for a specific camera. Special handling for 'events' (consolidated)."""
        if camera_name == "events":
            return _get_events_from_consolidated()

        camera_path = os.path.join(storage_path, camera_name)
        events = []

        if not os.path.isdir(camera_path):
            return events

        subdirs = sorted(
            [d for d in os.listdir(camera_path)
             if os.path.isdir(os.path.join(camera_path, d))],
            reverse=True
        )

        for subdir in subdirs:
            folder_path = os.path.join(camera_path, subdir)
            summary_path = os.path.join(folder_path, 'summary.txt')

            parts = subdir.split('_', 1)
            ts = parts[0] if len(parts) > 0 else "0"
            eid = parts[1] if len(parts) > 1 else subdir

            summary_text = "Analysis pending..."
            parsed = {}
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary_text = f.read().strip()
                parsed = _parse_summary(summary_text)

            metadata_path = os.path.join(folder_path, 'metadata.json')
            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except Exception:
                    pass

            review_summary_path = os.path.join(folder_path, 'review_summary.md')
            review_summary = None
            if os.path.exists(review_summary_path):
                with open(review_summary_path, 'r') as f:
                    review_summary = f.read().strip()

            clip_path = os.path.join(folder_path, 'clip.mp4')
            snapshot_path = os.path.join(folder_path, 'snapshot.jpg')
            viewed_path = os.path.join(folder_path, '.viewed')
            has_clip = os.path.exists(clip_path)
            try:
                ts_float = float(ts)
                age_seconds = time.time() - ts_float
                ongoing = not has_clip and age_seconds < 5400
            except (ValueError, TypeError):
                ongoing = not has_clip
            genai_entries = _extract_genai_entries(folder_path)

            events.append({
                "event_id": eid,
                "camera": camera_name,
                "subdir": subdir,
                "timestamp": ts,
                "summary": summary_text,
                "title": metadata.get("genai_title") or parsed.get("Title"),
                "description": metadata.get("genai_description") or parsed.get("Description") or parsed.get("AI Description"),
                "scene": metadata.get("genai_scene") or parsed.get("Scene"),
                "label": metadata.get("label") or parsed.get("Label", "unknown"),
                "severity": metadata.get("severity") or parsed.get("Severity"),
                "threat_level": metadata.get("threat_level", 0),
                "review_summary": review_summary,
                "has_clip": has_clip,
                "has_snapshot": os.path.exists(snapshot_path),
                "viewed": os.path.exists(viewed_path),
                "hosted_clip": f"/files/{camera_name}/{subdir}/clip.mp4",
                "hosted_snapshot": f"/files/{camera_name}/{subdir}/snapshot.jpg",
                "ongoing": ongoing,
                "genai_entries": genai_entries
            })

        return events

    @app.route('/cameras')
    def list_cameras():
        """List available cameras from config."""
        active_cameras = []
        try:
            for item in os.listdir(storage_path):
                item_path = os.path.join(storage_path, item)
                if os.path.isdir(item_path):
                    if not item.split('_')[0].isdigit():
                        active_cameras.append(item)
        except Exception as e:
            logger.error(f"Error listing cameras: {e}")

        if os.path.isdir(os.path.join(storage_path, "events")) and "events" not in active_cameras:
            active_cameras.append("events")
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

    @app.route('/events/<camera>')
    def list_camera_events(camera):
        """List events for a specific camera."""
        active_ids = state_manager.get_active_event_ids()
        active_ce_folders = [ce.folder_name for ce in orchestrator.consolidated_manager.get_all()]
        deleted = file_manager.cleanup_old_events(active_ids, active_ce_folders)
        orchestrator._last_cleanup_time = time.time()
        orchestrator._last_cleanup_deleted = deleted

        sanitized = file_manager.sanitize_camera_name(camera)
        events = _filter_events(_get_events_for_camera(sanitized))

        return jsonify({
            "camera": sanitized,
            "events": events
        })

    @app.route('/events')
    def list_events():
        """List all events across all cameras (global view)."""
        active_ids = state_manager.get_active_event_ids()
        active_ce_folders = [ce.folder_name for ce in orchestrator.consolidated_manager.get_all()]
        deleted = file_manager.cleanup_old_events(active_ids, active_ce_folders)
        orchestrator._last_cleanup_time = time.time()
        orchestrator._last_cleanup_deleted = deleted

        all_events = []
        cameras_found = []

        try:
            for camera_dir in os.listdir(storage_path):
                camera_path = os.path.join(storage_path, camera_dir)
                if not os.path.isdir(camera_path):
                    continue
                if camera_dir.split('_')[0].isdigit():
                    continue

                cameras_found.append(camera_dir)
                events = _get_events_for_camera(camera_dir)
                all_events.extend(events)

            all_events.sort(key=lambda x: x['timestamp'], reverse=True)

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
            for camera_dir in os.listdir(storage_path):
                camera_path = os.path.join(storage_path, camera_dir)
                if not os.path.isdir(camera_path) or camera_dir.split('_')[0].isdigit():
                    continue
                for subdir in os.listdir(camera_path):
                    folder_path = os.path.join(camera_path, subdir)
                    if os.path.isdir(folder_path):
                        viewed_path = os.path.join(folder_path, '.viewed')
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

        timeline_path = os.path.join(folder_path, "notification_timeline.json")
        timeline_data = {"event_id": None, "entries": []}
        if os.path.exists(timeline_path):
            try:
                with open(timeline_path, 'r') as f:
                    timeline_data = json.load(f)
            except Exception as e:
                logger.debug(f"Error reading timeline: {e}")

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
                    for sf in ('clip.mp4', 'snapshot.jpg', 'metadata.json', 'summary.txt', 'review_summary.md'):
                        if os.path.isfile(os.path.join(sub_fp, sf)):
                            event_files.append(f"{sub}/{sf}")
            event_files.sort()
        except OSError:
            pass

        return render_template(
            'timeline.html',
            event_id=timeline_data.get("event_id", subdir),
            camera=camera,
            subdir=subdir,
            entries=entries,
            event_files=event_files
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
            for camera_dir in os.listdir(storage_path):
                camera_path = os.path.join(storage_path, camera_dir)
                if not os.path.isdir(camera_path) or camera_dir.split('_')[0].isdigit():
                    continue

                count = 0
                for event_dir in os.listdir(camera_path):
                    event_path = os.path.join(camera_path, event_dir)
                    if not os.path.isdir(event_path):
                        continue
                    try:
                        parts = event_dir.split('_', 1)
                        ts = float(parts[0])
                    except (ValueError, IndexError):
                        continue

                    viewed = os.path.exists(os.path.join(event_path, '.viewed'))
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

        storage_raw = file_manager.compute_storage_stats()
        mb = 1024 * 1024

        def fmt_size(b):
            if b >= 1024 * mb:
                return {'gb': round(b / (1024 * mb), 2), 'mb': None}
            return {'mb': round(b / mb, 2), 'gb': None}

        by_camera_storage = {}
        for cam, data in storage_raw.get('by_camera', {}).items():
            total = data['total']
            by_camera_storage[cam] = fmt_size(total)

        total_bytes = storage_raw.get('total', 0)
        breakdown = {
            'clips_mb': round(storage_raw.get('clips', 0) / mb, 2),
            'snapshots_mb': round(storage_raw.get('snapshots', 0) / mb, 2),
            'descriptions_mb': round(storage_raw.get('descriptions', 0) / mb, 2)
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
                'total_mb': round(total_bytes / mb, 2),
                'total_gb': round(total_bytes / (1024 * mb), 2) if total_bytes >= 1024 * mb else None,
                'by_camera': by_camera_storage,
                'breakdown': breakdown
            },
            'errors': error_buffer.get_all(),
            'last_cleanup': last_cleanup,
            'most_recent': most_recent_out,
            'system': {
                'uptime_seconds': int(time.time() - orchestrator._start_time),
                'mqtt_connected': orchestrator.mqtt_connected,
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

        return jsonify({
            "online": True,
            "mqtt_connected": orchestrator.mqtt_connected,
            "uptime_seconds": uptime_seconds,
            "uptime": uptime_str,
            "started_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(orchestrator._start_time)),
            "active_events": state_manager.get_stats(),
            "config": {
                "retention_days": orchestrator.config['RETENTION_DAYS'],
                "log_level": orchestrator.config.get('LOG_LEVEL', 'INFO'),
                "ffmpeg_timeout": orchestrator.config.get('FFMPEG_TIMEOUT', 60),
                "summary_padding_before": orchestrator.config.get('SUMMARY_PADDING_BEFORE', 15),
                "summary_padding_after": orchestrator.config.get('SUMMARY_PADDING_AFTER', 15)
            }
        })

    return app
