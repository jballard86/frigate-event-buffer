import os
import json
import time
import threading
import schedule
import requests
import paho.mqtt.client as mqtt
from flask import Flask, jsonify, send_from_directory

# --- CONFIGURATION ---
FRIGATE_URL = os.getenv('FRIGATE_URL', 'http://REDACTED_LOCAL_IP:5000')
MQTT_BROKER = os.getenv('MQTT_BROKER', 'REDACTED_LOCAL_IP')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
STORAGE_DIR = "/data"
RETENTION_DAYS = int(os.getenv('RETENTION_DAYS', 3))

# We now listen to Frigate's native topics
MQTT_TOPIC = "frigate/events/#"

print(f"Starting Native Frigate Listener...")
print(f" - Frigate: {FRIGATE_URL}")
print(f" - MQTT: {MQTT_BROKER} (Listening to {MQTT_TOPIC})")

app = Flask(__name__)

# --- CORS HEADER ---
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# --- SAVE EVENT LOGIC ---
def save_event(payload):
    try:
        # Frigate Native Payload parsing
        # We only care when the event ends (so we get the full clip and final description)
        if payload.get('type') != 'end':
            return

        event_data = payload.get('after', {})
        event_id = event_data.get('id')
        camera = event_data.get('camera')
        
        # Only process Doorbell if that's your preference, or all cameras
        # if camera != "Doorbell": return 

        if not event_id: return
        
        print(f"Processing Event: {event_id} ({camera})")
        
        # Try to find the GenAI description in the payload
        # Frigate usually puts this in 'sub_label' or 'data' depending on version
        summary = event_data.get('sub_label', '')
        if not summary:
            # Fallback: Check if it's in the 'data' dict
            summary = event_data.get('data', {}).get('description', '')
        
        # Fallback: Just use the label (e.g., "Person") if AI didn't run
        if not summary:
            summary = f"{event_data.get('label', 'Unknown Event')} detected."

        timestamp = int(time.time())
        event_dir = os.path.join(STORAGE_DIR, f"{timestamp}_{event_id}")
        os.makedirs(event_dir, exist_ok=True)

        # 1. Save JSON (Construct our own simple metadata)
        meta = {
            "event_id": event_id,
            "title": f"{camera} Alert",
            "summary": summary,
            "timestamp": timestamp
        }
        with open(os.path.join(event_dir, "data.json"), 'w') as f:
            json.dump(meta, f)

        # 2. Download Clean Snapshot
        snap_url = f"{FRIGATE_URL}/api/events/{event_id}/snapshot.jpg?bbox=0"
        r_snap = requests.get(snap_url, timeout=15)
        if r_snap.status_code == 200:
            with open(os.path.join(event_dir, "snapshot.jpg"), 'wb') as f:
                f.write(r_snap.content)

        # 3. Download Clip
        clip_url = f"{FRIGATE_URL}/api/events/{event_id}/clip.mp4"
        r_clip = requests.get(clip_url, timeout=45)
        if r_clip.status_code == 200:
            with open(os.path.join(event_dir, "clip.mp4"), 'wb') as f:
                f.write(r_clip.content)
            print(f"Saved event {event_id}")

    except Exception as e:
        print(f"Error saving event: {e}")

# --- CLEANUP LOGIC ---
def cleanup_old_events():
    cutoff = time.time() - (RETENTION_DAYS * 86400)
    if not os.path.exists(STORAGE_DIR): return
    for folder in os.listdir(STORAGE_DIR):
        try:
            if float(folder.split('_')[0]) < cutoff:
                import shutil
                shutil.rmtree(os.path.join(STORAGE_DIR, folder))
        except: pass

# --- MQTT WORKER ---
def start_mqtt():
    def on_msg(c, u, m):
        try:
            threading.Thread(target=save_event, args=(json.loads(m.payload.decode()),)).start()
        except: pass
    
    client = mqtt.Client()
    client.on_connect = lambda c, u, f, rc: c.subscribe(MQTT_TOPIC)
    client.on_message = on_msg
    
    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            client.loop_forever()
        except:
            time.sleep(10)

# --- FLASK API ---
@app.route('/events')
def get_events():
    events_list = []
    if os.path.exists(STORAGE_DIR):
        for folder in sorted(os.listdir(STORAGE_DIR), reverse=True):
            try:
                with open(os.path.join(STORAGE_DIR, folder, "data.json"), 'r') as f:
                    d = json.load(f)
                    d['hosted_snapshot'] = f"/files/{folder}/snapshot.jpg"
                    d['hosted_clip'] = f"/files/{folder}/clip.mp4"
                    events_list.append(d)
            except: pass
    return jsonify({"events": events_list})

@app.route('/files/<path:filepath>')
def serve_file(filepath):
    return send_from_directory(STORAGE_DIR, filepath)

if __name__ == "__main__":
    threading.Thread(target=start_mqtt, daemon=True).start()
    schedule.every().hour.do(cleanup_old_events)
    threading.Thread(target=lambda: [schedule.run_pending(), time.sleep(60)]).start()
    app.run(host='0.0.0.0', port=5050)