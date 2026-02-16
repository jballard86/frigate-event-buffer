# Lives in multi_cam_frame_extract_mock/ for standalone development; plan to merge into frigate_buffer later (see PLAN_TO_MERGE_LATER.md).

# ==============================================================================
#  ARCHITECTURE NOTES:
# 1. TEMPLATE STATUS: This script is currently a standalone microservice template.
#    It needs to be integrated with the actual Frigate volume paths and ENV vars.
#
# 2. DATA FORMATTING: Before sending to Gemini, we must:
#    - Load the System Prompt from a separate file (multi_cam_system_prompt.txt) for easy editing.
#    - Read the cropped images from disk.
#    - Encode images to Base64 strings.
#    - Pack everything into a JSON payload compatible with the Gemini API.
#
# 3. PROXY ROUTING: Requests must be POSTed to the 'gemini-proxy' container 
#    (e.g., 192.168.21.189:5050..) using the internal Docker network.  should be configurable in config.yaml/config.example.yaml
#    Do not send directly to Google; let the proxy handle rate limits/billing.
#
# 4. ASYNC HANDLING: We must wait for the proxy's JSON reply.
#    CRITICAL: This network call MUST happen inside the background thread 
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
import numpy as np
import paho.mqtt.client as mqtt
from datetime import datetime
from collections import defaultdict

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
DEFAULT_CONFIG = {   # should be configurable in config.yaml/config.example.yaml    
    "max_multi_cam_frames_min": 45,
    "max_multi_cam_frames_sec": 2,    # max Target capture rate (approx 0.5 fps)  variable rates should occur where more motion is detected
    "motion_threshold_px": 50,        # Min pixels movement to trigger "high rate" capture
    "crop_width": 1280,
    "crop_height": 720,
    "gemini_proxy_url": "192.168.21.189:5050",
    "gemini_proxy_api_key": "YOUR_GEMINI_PROXY_API_KEY",    # should be configurable in config.yaml/config.example.yaml
    "gemini_proxy_model": "gemini-2.5-flash-lite",
    "gemini_proxy_temperature": 0.3,
    "gemini_proxy_top_p": 1,
    "gemini_proxy_frequency_penalty": 0,
    "gemini_proxy_presence_penalty": 0,
    "multi_cam_system_prompt_file": "",  # default: same dir as this script, multi_cam_system_prompt.txt
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

def load_system_prompt_template():
    """Load raw template from multi_cam_system_prompt.txt (contains {placeholders})."""
    prompt_file = CONF.get("multi_cam_system_prompt_file") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "multi_cam_system_prompt.txt"
    )
    if os.path.isfile(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not read prompt file {prompt_file}: {e}")
    return "You are a security recap assistant. Describe what happens across the provided camera frames in a short summary."


def build_multi_cam_system_prompt(
    image_count,
    global_event_camera_list,
    first_image_number,
    last_image_number,
    activity_start_datetime_str,
    duration_str,
    zones_list_str,
    labels_sublabels_str,
):
    """
    Fill multi_cam_system_prompt.txt placeholders. Data comes from EventMetadataStore and process_multi_cam_event.
    Placeholders: {image_count}, {global_event_camera_list}, {first_image_number}, {last_image_number},
    {current day and time}, {duration of the event}, {list of zones in global event, dont repeat zones}, {list of labels and sub_labels tracked in scene}.
    """
    template = load_system_prompt_template()
    return template.replace("{image_count}", str(image_count)).replace(
        "{global_event_camera_list}", global_event_camera_list
    ).replace("{first_image_number}", str(first_image_number)).replace(
        "{last_image_number}", str(last_image_number)
    ).replace("{current day and time}", activity_start_datetime_str).replace(
        "{duration of the event}", duration_str
    ).replace(
        "{list of zones in global event, dont repeat zones}", zones_list_str
    ).replace(
        "{list of labels and sub_labels tracked in scene}", labels_sublabels_str
    )


def load_system_prompt():
    """Return system prompt for Gemini; when no event context, returns template as-is (placeholders unfilled)."""
    return load_system_prompt_template().strip()

# --- State Management ---
class EventMetadataStore:
    """
    Independent state manager. Stores high-frequency metadata (boxes, scores) 
    directly from MQTT 'after' payloads to build a granular timeline 
    that might be missing from the standard database export.
    """
    def __init__(self):
        self._lock = threading.Lock()
        # { event_id: [ {timestamp, box, score, area}, ... ] }
        self.data = defaultdict(list)
        # { camera_name: {id: event_id, start: timestamp} }
        self.active_events = {}
        # { event_id: camera_name } — from MQTT frigate/events so we don't infer from path
        self.event_camera = {}
        # { event_id: { label, sub_label, entered_zones, start_time, end_time } } — from frigate/events for prompt placeholders
        self.event_info = {}

    def set_event_info(self, event_id, label=None, sub_label=None, entered_zones=None, start_time=None, end_time=None):
        """Store event metadata from frigate/events for prompt building (label, sub_label, zones, times)."""
        with self._lock:
            if event_id not in self.event_info:
                self.event_info[event_id] = {"label": "", "sub_label": "", "entered_zones": [], "start_time": None, "end_time": None}
            e = self.event_info[event_id]
            if label is not None:
                e["label"] = label or ""
            if sub_label is not None:
                e["sub_label"] = sub_label if isinstance(sub_label, str) else (sub_label[0] if isinstance(sub_label, (list, tuple)) and sub_label else "")
            if entered_zones is not None:
                e["entered_zones"] = list(entered_zones) if entered_zones else []
            if start_time is not None:
                e["start_time"] = start_time
            if end_time is not None:
                e["end_time"] = end_time

    def get_event_info(self, event_id):
        """Return { label, sub_label, entered_zones, start_time, end_time } or None."""
        with self._lock:
            return self.event_info.get(event_id)

    def add_frame_data(self, event_id, payload):
        """Ingest 'after' payload to track object movement."""
        with self._lock:
            self.data[event_id].append({
                "timestamp": payload.get("frame_time", time.time()),
                "box": payload.get("box", []), # [ymin, xmin, ymax, xmax]
                "area": payload.get("area", 0),
                "score": payload.get("score", 0)
            })

    def set_event_camera(self, event_id, camera):
        """Store event_id -> camera_name from frigate/events (use in process_multi_cam_event)."""
        with self._lock:
            if event_id and camera:
                self.event_camera[event_id] = camera

    def get_camera(self, event_id):
        """Look up camera name for event_id; returns None if unknown."""
        with self._lock:
            return self.event_camera.get(event_id)

    def update_status(self, camera, event_id, start_time, status):
        with self._lock:
            if status == "end":
                if camera in self.active_events and self.active_events[camera]['id'] == event_id:
                    del self.active_events[camera]
            else:
                self.active_events[camera] = {"id": event_id, "start": start_time}

    def get_overlaps(self, target_event_id):
        """
        Find other events that overlap in time with the target event.
        Returns a list of overlapping event_ids.
        """
        if target_event_id not in self.data: return []
        
        target_frames = self.data[target_event_id]
        if not target_frames: return []
        
        # Define the target's time window
        t_start = target_frames[0]['timestamp']
        t_end = target_frames[-1]['timestamp']
        
        overlaps = []
        with self._lock:
            for eid, frames in self.data.items():
                if eid == target_event_id or not frames: continue
                
                e_start = frames[0]['timestamp']
                e_end = frames[-1]['timestamp']
                
                # Check for temporal intersection
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
                if eid in self.event_camera:
                    del self.event_camera[eid]
                if eid in self.event_info:
                    del self.event_info[eid]

metadata_store = EventMetadataStore()

# --- Processing Logic ---

def smart_crop(frame, box, target_w, target_h):
    """
    Crops the frame centered on the box.
    Handles boundary clamping and upscaling if the crop is too small.
    """
    h, w, _ = frame.shape
    if not box: return cv2.resize(frame, (target_w, target_h))

    # Unpack Normalized Coordinates (Frigate MQTT uses 0-1)
    y_min, x_min, y_max, x_max = box
    
    # Convert to Pixels
    py_min, px_min = int(y_min * h), int(x_min * w)
    py_max, px_max = int(y_max * h), int(x_max * w)
    
    center_y = (py_min + py_max) // 2
    center_x = (px_min + px_max) // 2
    
    # Calculate Crop Window
    x1 = center_x - (target_w // 2)
    y1 = center_y - (target_h // 2)
    x2 = x1 + target_w
    y2 = y1 + target_h
    
    # Clamp to Image Bounds
    if x1 < 0: x2 += abs(x1); x1 = 0
    if y1 < 0: y2 += abs(y1); y1 = 0
    if x2 > w: x1 -= (x2 - w); x2 = w
    if y2 > h: y1 -= (y2 - h); y2 = h
    
    # Final Safety Clamp
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    crop = frame[y1:y2, x1:x2]
    
    # Resize/Pad if dimensions don't match target
    if crop.shape[1] != target_w or crop.shape[0] != target_h:
        crop = cv2.resize(crop, (target_w, target_h))
        
    return crop

def wait_for_video_file(event_id, timeout=60):
    """
    Polls the storage path until the clip.mp4 appears or timeout reached.
    Returns path or None.
    """
    start = time.time()
    # Search logic: We don't know the exact camera folder, so we walk the specific structure
    # Optimization: Metadata store *could* track camera name, but let's scan safely.
    while time.time() - start < timeout:
        for root, _, files in os.walk(STORAGE_PATH):
            if event_id in root and "clip.mp4" in files:
                return os.path.join(root, "clip.mp4")
        time.sleep(2)
    return None


def send_to_proxy(system_prompt, image_paths, output_dir, main_event_id, primary_camera, start_time):
    """
    POST system prompt + base64-encoded images to Gemini proxy; parse response and save as analysis_result.json.
    This file is the bridge: the Reporter (daily_report_to_proxy.py) scans event folders for analysis_result.json
    to populate get_previous_day_recaps().
    When implementing for real: use requests.post to CONF['gemini_proxy_url'], send OpenAI-format request with
    system message = system_prompt and user content = image parts (base64) + optional text; parse
    choices[0].message.content as JSON (title, scene, shortSummary, confidence, potential_threat_level) and merge
    with event_id, camera, start_time below before writing.
    """
    # TODO: requests.post(CONF["gemini_proxy_url"] + "/v1/chat/completions", json={...}, headers={"Authorization": "Bearer " + CONF["gemini_proxy_api_key"]})
    # Mock: write placeholder so Reporter can find and load it when scanning for previous day's recaps
    analysis = {
        "event_id": main_event_id,
        "camera": primary_camera,
        "start_time": start_time,
        "title": "[Mock] Multi-cam recap title",
        "scene": "[Mock] Scene description from proxy would appear here.",
        "shortSummary": "[Mock] Short summary for notifications.",
        "confidence": 0.0,
        "potential_threat_level": 0,
    }
    result_path = os.path.join(output_dir, "analysis_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Wrote {result_path} (mock proxy response; replace with real POST + parse when testing proxy)")


def process_multi_cam_event(main_event_id, linked_event_ids):
    logger.info(f"Generating Multi-Cam Recap for {main_event_id} + {linked_event_ids}")
    
    all_events = [main_event_id] + linked_event_ids
    video_paths = {}
    event_cameras = {}
    
    # 1. Acquire Video Sources
    for eid in all_events:
        path = wait_for_video_file(eid)
        if path:
            video_paths[eid] = path
            # Prefer camera from MQTT (frigate/events); fallback to path parsing
            cam = metadata_store.get_camera(eid)
            if cam:
                event_cameras[eid] = cam
            else:
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

    # 2. Determine Time Range
    timestamps = []
    for eid in video_paths:
        data = metadata_store.get_data(eid)
        if data: timestamps.extend([d['timestamp'] for d in data])
        
    if not timestamps: return
    start_time = min(timestamps)
    end_time = max(timestamps)
    
    # 3. Setup Output
    # Place folder INSIDE the main event's directory as requested
    main_clip_path = video_paths.get(main_event_id)
    if main_clip_path:
        base_dir = os.path.dirname(main_clip_path)
        output_dir = os.path.join(base_dir, "multi_cam_frames")
        os.makedirs(output_dir, exist_ok=True)
    else:
        logger.error("Main event clip missing, cannot determine output folder.")
        return

    # 4. The Loop (Variable Rate + Smart Selection)
    caps = {eid: cv2.VideoCapture(path) for eid, path in video_paths.items()}
    current_time = start_time
    step = 1.0 / CONF['max_multi_cam_frames_sec'] # Base step (e.g. 0.5s)
    saved_frame_count = 0
    saved_frame_paths = []
    last_centers = {eid: (0,0) for eid in video_paths} # For motion delta

    while current_time <= end_time:
        candidates = []
        
        # A. Gather Candidates
        for eid in video_paths:
            # Get frame data closest to current_time
            frames = metadata_store.get_data(eid)
            # Find closest metadata entry
            # Simple linear search (can be optimized)
            closest = min(frames, key=lambda x: abs(x['timestamp'] - current_time))
            
            # Validity check: is this metadata actually from this second?
            if abs(closest['timestamp'] - current_time) > 1.0: continue
            
            # Action Score = Area * Score
            action_score = closest['area'] * closest['score']
            candidates.append({
                "eid": eid, "score": action_score, "meta": closest, 
                "camera": event_cameras[eid]
            })
            
        if not candidates:
            current_time += step
            continue

        # B. Selection Algorithm (Mixed Strategy)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        winner = candidates[0]
        selected = [winner]
        
        # Scenario B: Complex Scene (others are close to winner)
        for contender in candidates[1:]:
            if winner['score'] < (1.5 * contender['score']):
                selected.append(contender)

        # C. Variable Rate Filter (Motion Check)
        final_selection = []
        for item in selected:
            eid = item['eid']
            box = item['meta']['box'] # [ymin, xmin, ymax, xmax]
            
            # Calculate Center in Normalized Coords
            cy = (box[0] + box[2]) / 2
            cx = (box[1] + box[3]) / 2
            
            # Convert to approx pixels for threshold check (assuming 1080p source relative)
            py, px = cy * 1080, cx * 1920 
            last_py, last_px = last_centers[eid]
            
            dist = abs(py - last_py) + abs(px - last_px)
            
            # Save if moved enough OR first frame
            if dist > CONF['motion_threshold_px'] or last_centers[eid] == (0,0):
                final_selection.append(item)
                last_centers[eid] = (py, px)

        # D. Extract & Save
        for item in final_selection:
            eid = item['eid']
            cap = caps[eid]
            
            # Seek video
            # Frigate clips often include pre-capture buffer; first metadata timestamp may be after real clip start.
            # When merging, use clip start from export (export_buffer_before) to avoid seeking past the action.
            vid_start = metadata_store.get_data(eid)[0]['timestamp']
            offset_sec = current_time - vid_start
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0, offset_sec * 1000))
            
            ret, frame = cap.read()
            if ret:
                # Crop
                processed = smart_crop(frame, item['meta']['box'], 
                                     CONF['crop_width'], CONF['crop_height'])
                
                # Overlay
                label = f"{item['camera']} | {datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')}"
                cv2.putText(processed, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 255, 255), 2) # White
                cv2.putText(processed, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 0, 0), 1)       # Black Border
                
                # Save
                fname = f"frame_{int(current_time)}_{item['camera']}.jpg"
                out_path = os.path.join(output_dir, fname)
                cv2.imwrite(out_path, processed)
                saved_frame_count += 1
                saved_frame_paths.append(out_path)

        current_time += step

    # 5. Build filled system prompt from template (placeholders from MQTT event_info)
    cameras_list = ", ".join(sorted(set(event_cameras.values())))
    zones_seen = set()
    labels_sublabels_lines = []
    for eid in all_events:
        info = metadata_store.get_event_info(eid)
        if info:
            for z in (info.get("entered_zones") or []):
                zones_seen.add(z)
            label, sub = info.get("label") or "", info.get("sub_label") or ""
            if sub and label:
                line = f"{sub} ({label})"
            elif label:
                line = label
            else:
                line = "unknown"
            if line and line not in labels_sublabels_lines:
                labels_sublabels_lines.append(line)
    zones_str = ", ".join(sorted(zones_seen)) if zones_seen else "none recorded"
    labels_str = "\n- ".join(labels_sublabels_lines) if labels_sublabels_lines else "(none recorded)"

    activity_start_datetime_str = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
    duration_sec = max(0, end_time - start_time)
    duration_str = f"{int(duration_sec)} seconds" if duration_sec < 60 else f"{duration_sec / 60:.1f} minutes"

    image_count = max(1, saved_frame_count)
    final_prompt = build_multi_cam_system_prompt(
        image_count=image_count,
        global_event_camera_list=cameras_list,
        first_image_number=1,
        last_image_number=image_count,
        activity_start_datetime_str=activity_start_datetime_str,
        duration_str=duration_str,
        zones_list_str=zones_str,
        labels_sublabels_str=labels_str.strip(),
    )
    prompt_path = os.path.join(output_dir, "system_prompt_filled.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(final_prompt)
    logger.info(f"Wrote filled system prompt to {prompt_path} (for proxy request when implemented)")

    # 6. Send to proxy and save response as analysis_result.json (bridge to Reporter)
    primary_camera = event_cameras.get(main_event_id, list(event_cameras.values())[0] if event_cameras else "unknown")
    send_to_proxy(
        system_prompt=final_prompt,
        image_paths=saved_frame_paths,
        output_dir=output_dir,
        main_event_id=main_event_id,
        primary_camera=primary_camera,
        start_time=start_time,
    )

    # Cleanup
    for cap in caps.values(): cap.release()
    metadata_store.cleanup(all_events)
    logger.info(f"Multi-Cam Recap Complete. Saved to {output_dir}")

# --- MQTT Loop ---

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        topic = msg.topic

        # frigate/events: lifecycle (new/end) and overlap trigger; also store event_id -> camera
        if topic == "frigate/events":
            evt_type = payload.get('type')
            after = payload.get('after', {})
            eid = after.get('id')
            cam = after.get('camera')

            if not eid:
                return

            # Store camera and event metadata for prompt placeholders (label, sub_label, zones, times)
            metadata_store.set_event_camera(eid, cam)
            metadata_store.set_event_info(
                eid,
                label=after.get("label"),
                sub_label=after.get("sub_label"),
                entered_zones=after.get("entered_zones") or [],
                start_time=after.get("start_time"),
                end_time=after.get("end_time") if evt_type == "end" else None,
            )

            # Per-frame data comes from tracked_object_update only; do not add_frame_data here

            if evt_type == 'new':
                metadata_store.update_status(cam, eid, after.get('start_time'), "active")

            elif evt_type == 'end':
                metadata_store.update_status(cam, eid, after.get('start_time'), "end")

                # Check for overlaps (The "Director")
                overlaps = metadata_store.get_overlaps(eid)
                if overlaps:
                    threading.Thread(target=process_multi_cam_event, args=(eid, overlaps)).start()
                else:
                    threading.Timer(60, metadata_store.cleanup, args=([eid],)).start()
            return

        # frigate/<camera>/tracked_object_update: high-frequency per-frame data for selection algorithm
        if "tracked_object_update" in topic:
            parts = topic.split("/")
            camera = parts[1] if len(parts) >= 2 else "unknown"
            after = payload.get("after") or payload.get("before") or {}
            eid = after.get("id")
            if not eid:
                return
            # Ensure we have a camera for this event (tracked_object_update may arrive before frigate/events)
            metadata_store.set_event_camera(eid, camera)
            frame_payload = {
                "frame_time": after.get("frame_time", time.time()),
                "box": after.get("box", []),
                "area": after.get("area", 0),
                "score": after.get("score", 0),
            }
            metadata_store.add_frame_data(eid, frame_payload)

    except Exception as e:
        logger.error(f"MQTT Error: {e}")

if __name__ == "__main__":
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe("frigate/events")
    client.subscribe("frigate/+/tracked_object_update")
    logger.info("Multi-Cam Service Started. Listening for frigate/events and frigate/+/tracked_object_update...")
    client.loop_forever()
