#       this is mostly mock code to be removed when we have a real multi-cam frame extracter  
#       it can be used as a template for the real multi-cam frame extracter, and adjusted for how we are processing the rest of the workload
               

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
DEFAULT_CONFIG = {
    "max_multi_cam_frames_min": 45,
    "max_multi_cam_frames_sec": 2,    # Target capture rate (approx 0.5 fps)
    "motion_threshold_px": 50,        # Min pixels movement to trigger "high rate" capture
    "crop_width": 1280,
    "crop_height": 720
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
        # { event_id: [ {timestamp, box, score, area}, ... ] }
        self.data = defaultdict(list)
        # { camera_name: {id: event_id, start: timestamp} }
        self.active_events = {}

    def add_frame_data(self, event_id, payload):
        """Ingest 'after' payload to track object movement."""
        with self._lock:
            self.data[event_id].append({
                "timestamp": payload.get("frame_time", time.time()),
                "box": payload.get("box", []), # [ymin, xmin, ymax, xmax]
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
                if eid in self.data: del self.data[eid]

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
            # Extract camera name from folder structure: .../camera_name/event_id/clip.mp4
            try:
                event_cameras[eid] = path.split(os.sep)[-3]
            except:
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
            # Video start time is needed. Assuming first metadata timestamp is start.
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
                cv2.imwrite(os.path.join(output_dir, fname), processed)

        current_time += step

    # Cleanup
    for cap in caps.values(): cap.release()
    metadata_store.cleanup(all_events)
    logger.info(f"Multi-Cam Recap Complete. Saved to {output_dir}")

# --- MQTT Loop ---

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        topic = msg.topic
        
        # Handle 'frigate/events'
        if topic == "frigate/events":
            evt_type = payload.get('type')
            after = payload.get('after', {})
            eid = after.get('id')
            cam = after.get('camera')
            
            if not eid: return

            # Record Metadata
            metadata_store.add_frame_data(eid, after)

            if evt_type == 'new':
                metadata_store.update_status(cam, eid, after.get('start_time'), "active")
            
            elif evt_type == 'end':
                metadata_store.update_status(cam, eid, after.get('start_time'), "end")
                
                # Check for overlaps (The "Director")
                overlaps = metadata_store.get_overlaps(eid)
                if overlaps:
                    # Spawn thread to process after delay (waiting for video writes)
                    threading.Thread(target=process_multi_cam_event, 
                                   args=(eid, overlaps)).start()
                else:
                    # Cleanup metadata for single events after a safety delay
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