import os
import json
import time
import threading
import schedule
import requests
import paho.mqtt.client as mqtt
from flask import Flask, jsonify, send_from_directory

# --- CONFIGURATION ---
FRIGATE_URL = os.getenv('FRIGATE_URL', 'http://192.168.21.189:5000')
MQTT_BROKER = os.getenv('MQTT_BROKER', '192.168.21.189')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_TOPIC = "frigate/events/Doorbell"  # Listen only to Doorbell for now
STORAGE_DIR = "/data"
RETENTION_DAYS = int(os.getenv('RETENTION_DAYS', 3))

app = Flask(__name__)

# --- CORS (Fixes Video Playback) ---
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# --- ARCHIVE LOGIC ---
def save_event(payload):
    try:
        # We only save the clip when the event ends
        if payload.get('type') != 'end': return

        data = payload.get('after', {})
        event_id = data.get('id')
        description = data.get('sub_label', 'Doorbell Activity') # Get the Gemini text
        
        if not event_id: return

        print(f"Archiving Event: {event_id} - {description}")
        timestamp = int(time.time())
        event_dir = os.path.join(STORAGE_DIR, f"{timestamp}_{event_id}")
        os.makedirs(event_dir, exist_ok=True)

        # 1. Save Metadata
        meta = {
            "event_id": event_id,
            "title": "Doorbell Alert",
            "summary": description,
            "timestamp": timestamp
        }
        with open(os.path.join(event_dir, "data.json"), 'w') as f:
            json.dump(meta, f)

        # 2. Save Clean Snapshot
        r_snap = requests.get(f"{FRIGATE_URL}/api/events/{event_id}/snapshot.jpg?bbox=0")
        if r_snap.status_code == 200:
            with open(os.path.join(event_dir, "snapshot.jpg"), 'wb') as f:
                f.write(r_snap.content)

        # 3. Save Clip
        r_clip = requests.get(f"{FRIGATE_URL}/api/events/{event_id}/clip.mp4")
        if r_clip.status_code == 200:
            with open(os.path.join(event_dir, "clip.mp4"), 'wb') as f:
                f.write(r_clip.content)

    except Exception as e:
        print(f"Error: {e}")

# --- MQTT LISTENER ---
def start_mqtt():
    def on_msg(c, u, m):
        threading.Thread(target=save_event, args=(json.loads(m.payload.decode()),)).start()
    
    client = mqtt.Client()
    client.on_connect = lambda c, u, f, rc: c.subscribe(MQTT_TOPIC)
    client.on_message = on_msg
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()

# --- WEB SERVER ---
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
    app.run(host='0.0.0.0', port=5050) # Dockge maps this to 5055 externally