#!/usr/bin/env python3
# ==============================================================================
# Multi-Cam Recap - Standalone entrypoint (run with package installed: pip install -e .)
# Usage: python scripts/multi_cam_recap.py   OR   python -m scripts.multi_cam_recap
#
# TODO / ARCHITECTURE NOTES:
# 1. TEMPLATE STATUS: This script is currently a standalone microservice template.
#    It needs to be integrated with the actual Frigate volume paths and ENV vars.
#
# 2. DATA FORMATTING: Before sending to Gemini, we must:
#    - Define the System Prompt (Context + Instructions).
#    - Read the cropped images from disk.
#    - Encode images to Base64 strings.
#    - Pack everything into a JSON payload compatible with the Gemini API.
#
# 3. PROXY ROUTING: Requests must be POSTed to the 'gemini-proxy' container
#    (e.g., REDACTED_LOCAL_IP:5050..) using the internal Docker network.
#
# 4. ASYNC HANDLING: This network call MUST happen inside the background thread
#    (process_multi_cam_event). If placed in the main MQTT loop, we will drop
#    incoming Frigate events while waiting for the AI.
# ==============================================================================

import os
import json
import time
import yaml
import logging
import threading
import cv2
import ffmpegcv
import numpy as np
import paho.mqtt.client as mqtt
from datetime import datetime
from collections import defaultdict

from frigate_buffer.services import crop_utils
from frigate_buffer.managers.file import write_stitched_frame, create_ai_analysis_zip

# --- Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MultiCamService")

# Load Config from Env or Defaults
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
STORAGE_PATH = os.getenv("STORAGE_PATH", "/media/frigate")
CONFIG_FILE = os.getenv("CONFIG_FILE", "config.yaml")

# Default limits
DEFAULT_CONFIG = {
    "max_multi_cam_frames_min": 45,
    "max_multi_cam_frames_sec": 2,
    "motion_threshold_px": 50,
    "crop_width": 1280,
    "crop_height": 720,
    "motion_crop_min_area_fraction": 0.001,
    "motion_crop_min_px": 500,
    "gemini_proxy_url": "REDACTED_LOCAL_IP:5050",
    "gemini_proxy_api_key": "YOUR_GEMINI_PROXY_API_KEY",
    "gemini_proxy_model": "gemini-2.5-flash-lite",
    "gemini_proxy_temperature": 0.3,
    "gemini_proxy_top_p": 1,
    "gemini_proxy_frequency_penalty": 0,
    "gemini_proxy_presence_penalty": 0,
    "save_ai_frames": True,
    "create_ai_analysis_zip": True,
}

def load_config():
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    config.update(user_config)
        except Exception as e:
            logger.error(f"Failed to load config.yaml: {e}")
    return config

CONF = load_config()

# --- State Management ---
class EventMetadataStore:
    """
    Independent state manager. Stores high-frequency metadata (boxes, scores)
    directly from MQTT 'after' payloads to build a granular timeline
    that might be missing from the standard database export.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.data = defaultdict(list)
        self.active_events = {}

    def add_frame_data(self, event_id, payload):
        """Ingest 'after' payload to track object movement."""
        with self._lock:
            self.data[event_id].append({
                "timestamp": payload.get("frame_time", time.time()),
                "box": payload.get("box", []),
                "area": payload.get("area", 0),
                "score": payload.get("score", 0)
            })

    def update_status(self, camera, event_id, start_time, status):
        with self._lock:
            if status == "end":
                if camera in self.active_events and self.active_events[camera]['id'] == event_id:
                    del self.active_events[camera]
            else:
                self.active_events[camera] = {"id": event_id, "start": start_time}

    def get_overlaps(self, target_event_id):
        """Find other events that overlap in time with the target event."""
        if target_event_id not in self.data:
            return []
        target_frames = self.data[target_event_id]
        if not target_frames:
            return []
        t_start = target_frames[0]['timestamp']
        t_end = target_frames[-1]['timestamp']
        overlaps = []
        with self._lock:
            for eid, frames in self.data.items():
                if eid == target_event_id or not frames:
                    continue
                e_start = frames[0]['timestamp']
                e_end = frames[-1]['timestamp']
                if (t_start < e_end) and (t_end > e_start):
                    overlaps.append(eid)
        return overlaps

    def get_data(self, event_id):
        with self._lock:
            return self.data.get(event_id, [])

    def cleanup(self, event_ids):
        with self._lock:
            for eid in event_ids:
                if eid in self.data:
                    del self.data[eid]

metadata_store = EventMetadataStore()

# --- Processing Logic (crop via shared crop_utils) ---

def wait_for_video_file(event_id, timeout=60):
    """Poll storage path until clip.mp4 appears or timeout. Returns path or None."""
    start = time.time()
    while time.time() - start < timeout:
        for root, _, files in os.walk(STORAGE_PATH):
            if event_id in root and "clip.mp4" in files:
                return os.path.join(root, "clip.mp4")
        time.sleep(2)
    return None

def process_multi_cam_event(main_event_id, linked_event_ids):
    logger.info(f"Generating Multi-Cam Recap for {main_event_id} + {linked_event_ids}")
    all_events = [main_event_id] + linked_event_ids
    video_paths = {}
    event_cameras = {}

    for eid in all_events:
        path = wait_for_video_file(eid)
        if path:
            video_paths[eid] = path
            try:
                event_cameras[eid] = path.split(os.sep)[-3]
            except Exception:
                event_cameras[eid] = "unknown"
        else:
            logger.warning(f"Clip for {eid} not found after timeout.")

    if len(video_paths) < 1:
        logger.error("No video files found. Aborting.")
        metadata_store.cleanup(all_events)
        return

    timestamps = []
    for eid in video_paths:
        data = metadata_store.get_data(eid)
        if data:
            timestamps.extend([d['timestamp'] for d in data])
    if not timestamps:
        return
    start_time = min(timestamps)
    end_time = max(timestamps)

    main_clip_path = video_paths.get(main_event_id)
    if main_clip_path:
        base_dir = os.path.dirname(main_clip_path)
        frames_dir = os.path.join(base_dir, "ai_frame_analysis", "frames")
        os.makedirs(frames_dir, exist_ok=True)
    else:
        logger.error("Main event clip missing, cannot determine output folder.")
        return

    caps = {eid: ffmpegcv.VideoCaptureNV(path) for eid, path in video_paths.items()}
    current_time = start_time
    step = 1.0 / CONF['max_multi_cam_frames_sec']
    last_centers = {eid: (0, 0) for eid in video_paths}
    prev_grays = {eid: None for eid in video_paths}
    min_area_frac = float(CONF.get('motion_crop_min_area_fraction', 0.001))
    min_area_px = int(CONF.get('motion_crop_min_px', 500))
    collected = []

    while current_time <= end_time:
        candidates = []
        for eid in video_paths:
            frames = metadata_store.get_data(eid)
            if not frames:
                continue
            closest = min(frames, key=lambda x: abs(x['timestamp'] - current_time))
            if abs(closest['timestamp'] - current_time) > 1.0:
                continue
            action_score = closest['area'] * closest['score']
            candidates.append({
                "eid": eid, "score": action_score, "meta": closest,
                "camera": event_cameras[eid]
            })

        if not candidates:
            current_time += step
            continue

        candidates.sort(key=lambda x: x['score'], reverse=True)
        winner = candidates[0]
        selected = [winner]
        for contender in candidates[1:]:
            if winner['score'] < (1.5 * contender['score']):
                selected.append(contender)

        final_selection = []
        for item in selected:
            eid = item['eid']
            box = item['meta'].get('box') or []
            cy = (box[0] + box[2]) / 2 if len(box) >= 4 else 0.5
            cx = (box[1] + box[3]) / 2 if len(box) >= 4 else 0.5
            py, px = cy * 1080, cx * 1920
            last_py, last_px = last_centers[eid]
            dist = abs(py - last_py) + abs(px - last_px)
            if dist > CONF['motion_threshold_px'] or last_centers[eid] == (0, 0):
                final_selection.append(item)
                last_centers[eid] = (py, px)

        for item in final_selection:
            eid = item['eid']
            cap = caps[eid]
            vid_start = metadata_store.get_data(eid)[0]['timestamp']
            offset_sec = current_time - vid_start
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0, offset_sec * 1000))
            ret, frame = cap.read()
            if ret:
                processed, next_gray = crop_utils.motion_crop(
                    frame,
                    prev_grays[eid],
                    CONF['crop_width'],
                    CONF['crop_height'],
                    min_area_fraction=min_area_frac,
                    min_area_px=min_area_px,
                )
                prev_grays[eid] = next_gray
                collected.append((current_time, item['camera'], processed))

        current_time += step

    collected.sort(key=lambda x: (x[0], x[1]))
    total = len(collected)
    save_frames = bool(CONF.get("save_ai_frames", True))
    create_zip = bool(CONF.get("create_ai_analysis_zip", True))
    manifest_entries = []
    for seq, (ts, camera_name, frame) in enumerate(collected, start=1):
        time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        crop_utils.draw_timestamp_overlay(frame, time_str, camera_name, seq, total)
        time_part = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        fname = f"frame_{seq:03d}_{time_part}_{camera_name}.jpg"
        out_path = os.path.join(frames_dir, fname)
        if save_frames and write_stitched_frame(frame, out_path):
            manifest_entries.append({
                "filename": fname,
                "timestamp_sec": ts,
                "camera": camera_name,
                "seq": seq,
                "total": total,
            })
    if manifest_entries:
        manifest_path = os.path.join(base_dir, "ai_frame_analysis", "manifest.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(manifest_entries, mf, indent=2)
        except OSError as e:
            logger.warning("Failed to write ai_frame_analysis manifest: %s", e)
    if create_zip:
        create_ai_analysis_zip(base_dir)

    for cap in caps.values():
        cap.release()
    metadata_store.cleanup(all_events)
    logger.info("Multi-Cam Recap Complete. Saved %d frames to %s", total, frames_dir)

# --- MQTT Loop ---

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        topic = msg.topic
        if topic == "frigate/events":
            evt_type = payload.get('type')
            after = payload.get('after', {})
            eid = after.get('id')
            cam = after.get('camera')
            if not eid:
                return
            metadata_store.add_frame_data(eid, after)
            if evt_type == 'new':
                metadata_store.update_status(cam, eid, after.get('start_time'), "active")
            elif evt_type == 'end':
                metadata_store.update_status(cam, eid, after.get('start_time'), "end")
                overlaps = metadata_store.get_overlaps(eid)
                if overlaps:
                    threading.Thread(target=process_multi_cam_event,
                                     args=(eid, overlaps)).start()
                else:
                    threading.Timer(60, metadata_store.cleanup, args=([eid],)).start()
    except Exception as e:
        logger.error(f"MQTT Error: {e}")

if __name__ == "__main__":
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe("frigate/events")
    logger.info("Multi-Cam Service Started. Listening for overlap events...")
    client.loop_forever()
