import os
import json
import time
import threading
import schedule
import requests
import paho.mqtt.client as mqtt
from flask import Flask, jsonify, send_from_directory

# --- CONFIGURATION (LOADED FROM ENVIRONMENT) ---
# We use os.getenv to read settings from Unraid/Docker
FRIGATE_URL = os.getenv('FRIGATE_URL', 'http://localhost:5000')
MQTT_BROKER = os.getenv('MQTT_BROKER', 'localhost')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_TOPIC = os.getenv('MQTT_TOPIC', 'frigate/custom/dashboard')
STORAGE_DIR = "/data"
RETENTION_DAYS = int(os.getenv('RETENTION_DAYS', 3))

print(f"Starting Buffer Service...")
print(f" - Frigate: {FRIGATE_URL}")
print(f" - MQTT: {MQTT_BROKER}:{MQTT_PORT}")
print(f" - Retention: {RETENTION_DAYS} days")

app = Flask(__name__)

# --- SAVE EVENT LOGIC ---
def save_event(payload):
    try:
        event_id = payload.get('event_id')
        if not event_id:
            print("Received payload without event_id, skipping.")
            return
        
        print(f"Processing Event: {event_id}")

        # Create a folder name like: 1707505000_123456-abcdef
        timestamp = int(time.time())
        event_dir = os.path.join(STORAGE_DIR, f"{timestamp}_{event_id}")
        os.makedirs(event_dir, exist_ok=True)

        # 1. Save the JSON Data
        with open(os.path.join(event_dir, "data.json"), 'w') as f:
            json.dump(payload, f)

        # 2. Download the Snapshot
        snap_url = f"{FRIGATE_URL}/api/events/{event_id}/snapshot.jpg"
        r_snap = requests.get(snap_url, timeout=10)
        if r_snap.status_code == 200:
            with open(os.path.join(event_dir, "snapshot.jpg"), 'wb') as f:
                f.write(r_snap.content)
        else:
            print(f"Failed to fetch snapshot: {r_snap.status_code}")

        # 3. Download the Clip
        clip_url = f"{FRIGATE_URL}/api/events/{event_id}/clip.mp4"
        r_clip = requests.get(clip_url, timeout=30)
        if r_clip.status_code == 200:
            with open(os.path.join(event_dir, "clip.mp4"), 'wb') as f:
                f.write(r_clip.content)
        else:
            print(f"Failed to fetch clip: {r_clip.status_code}")
            
        print(f"Saved event {event_id} successfully.")

    except Exception as e:
        print(f"Error saving event: {e}")

# --- CLEANUP LOGIC ---
def cleanup_old_events():
    print("Running cleanup...")
    cutoff = time.time() - (RETENTION_DAYS * 86400)
    
    if not os.path.exists(STORAGE_DIR):
        return

    for folder in os.listdir(STORAGE_DIR):
        folder_path = os.path.join(STORAGE_DIR, folder)
        try:
            # Check if folder starts with a timestamp
            timestamp_str = folder.split('_')[0]
            if float(timestamp_str) < cutoff:
                print(f"Deleting old event: {folder}")
                # Delete files inside then the folder
                for file in os.listdir(folder_path):
                    os.remove(os.path.join(folder_path, file))
                os.rmdir(folder_path)
        except (ValueError, IndexError):
            continue # Skip folders that don't match our format
        except Exception as e:
            print(f"Error cleaning {folder}: {e}")

# --- MQTT WORKER ---
def start_mqtt():
    def on_connect(client, userdata, flags, rc):
        print(f"Connected to MQTT Broker (Code: {rc})")
        client.subscribe(MQTT_TOPIC)

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            # Run save in a separate thread to not block MQTT loop
            threading.Thread(target=save_event, args=(payload,)).start()
        except Exception as e:
            print(f"MQTT Payload Error: {e}")

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            client.loop_forever()
        except Exception as e:
            print(f"MQTT Connection Failed: {e}. Retrying in 10s...")
            time.sleep(10)

# --- FLASK API ---
@app.route('/events', methods=['GET'])
def get_events():
    events_list = []
    if os.path.exists(STORAGE_DIR):
        # Sort folders by name (which starts with timestamp), newest first
        folders = sorted(os.listdir(STORAGE_DIR), reverse=True)
        for folder in folders:
            folder_path = os.path.join(STORAGE_DIR, folder)
            json_path = os.path.join(folder_path, "data.json")
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        # Add the "hosted" URLs that HA will use
                        data['hosted_snapshot'] = f"/files/{folder}/snapshot.jpg"
                        data['hosted_clip'] = f"/files/{folder}/clip.mp4"
                        data['timestamp'] = folder.split('_')[0]
                        events_list.append(data)
                except:
                    continue
    return jsonify({"events": events_list})

@app.route('/files/<path:filepath>')
def serve_file(filepath):
    return send_from_directory(STORAGE_DIR, filepath)

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    # 1. Start MQTT in background
    t_mqtt = threading.Thread(target=start_mqtt)
    t_mqtt.daemon = True
    t_mqtt.start()

    # 2. Schedule Cleanup (Run once an hour)
    schedule.every().hour.do(cleanup_old_events)
    def run_schedule():
        while True:
            schedule.run_pending()
            time.sleep(60)
    t_sched = threading.Thread(target=run_schedule)
    t_sched.daemon = True
    t_sched.start()

    # 3. Start Web Server
    app.run(host='0.0.0.0', port=5050)