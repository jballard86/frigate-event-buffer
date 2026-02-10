import os
import subprocess
import time
import shutil
from flask import Flask, Response, send_from_directory, jsonify
import shutil

app = Flask(__name__)

# --- CONFIGURATION ---
STORAGE_PATH = '/app/storage'
RETENTION_DAYS = 3  # Matches your "Last 3 Days" dashboard title

def cleanup_old_events():
    """Deletes folders older than RETENTION_DAYS."""
    now = time.time()
    cutoff = now - (RETENTION_DAYS * 86400)
    
    try:
        for subdir in os.listdir(STORAGE_PATH):
            folder_path = os.path.join(STORAGE_PATH, subdir)
            if os.path.isdir(folder_path):
                # Folders are named 'timestamp_eventid'
                try:
                    ts = float(subdir.split('_')[0])
                    if ts < cutoff:
                        print(f"Cleaning up old event: {subdir}")
                        shutil.rmtree(folder_path)
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Cleanup error: {e}")

@app.route('/events')
def list_events():
    # Run cleanup every time the list is requested
    cleanup_old_events()
    
    events = []
    try:
        subdirs = sorted([d for d in os.listdir(STORAGE_PATH) if os.path.isdir(os.path.join(STORAGE_PATH, d))], reverse=True)
        for subdir in subdirs:
            folder_path = os.path.join(STORAGE_PATH, subdir)
            summary_path = os.path.join(folder_path, 'summary.txt')
            
            try:
                ts, eid = subdir.split('_')
            except ValueError:
                continue

            summary_text = "Analysis pending..."
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary_text = f.read().strip()

            events.append({
                "event_id": eid,
                "timestamp": ts,
                "summary": summary_text,
                "hosted_clip": f"/files/{subdir}/clip.mp4",
                "hosted_snapshot": f"/files/{subdir}/snapshot.jpg"
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"events": events})

@app.route('/delete/<subdir>', methods=['POST'])
def delete_event(subdir):
    """Deletes a specific event folder manually from the dashboard."""
    folder_path = os.path.join(STORAGE_PATH, subdir)
    
    # Security check: Ensure we are only deleting folders within our storage path
    if os.path.abspath(folder_path).startswith(os.path.abspath(STORAGE_PATH)):
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"User manually deleted: {subdir}")
                return jsonify({"status": "success", "message": f"Deleted {subdir}"}), 200
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 500
    
    return jsonify({"status": "error", "message": "Invalid folder or path"}), 400

@app.route('/files/<path:filename>')
def serve_file(filename):
    file_path = os.path.join(STORAGE_PATH, filename)
    if not os.path.exists(file_path):
        return "File not found", 404

    # On-the-fly transcoding for H.264 compatibility
    if filename.endswith('.mp4'):
        command = [
            'ffmpeg',
            '-i', file_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '24',
            '-c:a', 'copy',
            '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
            '-f', 'mp4',
            '-'
        ]
        
        def generate():
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                while True:
                    data = process.stdout.read(1024 * 128)
                    if not data:
                        break
                    yield data
            finally:
                process.kill()
        return Response(generate(), mimetype='video/mp4')

    return send_from_directory(STORAGE_PATH, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5055, threaded=True)