# Running frigate-buffer on Windows (after build)

After `docker build -t frigate-buffer:latest .` you can start the container as below.

Storage is under your **Documents** folder. Put **config.yaml**, **.env**, and the **Firebase key JSON** in that folder so everything stays in one place.

## One-time setup

1. **Create storage folder in Documents**:
   ```powershell
   $storage = "$env:USERPROFILE\Documents\frigate_buffer_storage"
   New-Item -ItemType Directory -Force -Path $storage
   ```

2. **Put these files in that folder** (`Documents\frigate_buffer_storage`):
   - **config.yaml** — Copy from repo: `Copy-Item examples\config.example.yaml $storage\config.yaml`, then edit (cameras, network.mqtt_broker, frigate_url, buffer_ip, etc.).
   - **.env** — Optional. Copy from `examples\.env.example` if you use one; set MQTT_BROKER, FRIGATE_URL, BUFFER_IP, etc. The run command below uses `--env-file` so the container will load this file.
   - **Firebase key** — Your FCM service-account JSON (e.g. `firebase-credentials.json`). In **config.yaml** under `notifications.mobile_app` set:
     ```yaml
     credentials_path: "/app/storage/firebase-credentials.json"
     ```
     (Use the actual filename you put in the storage folder; the path is inside the container, so `/app/storage/` + filename.)

The app loads config from `/app/storage/config.yaml` automatically when you mount the storage folder.

## Start the container

**Important:** If you use `$storage` in the command, set it in the **same** PowerShell session first, or the mount will be invalid (`empty section between colons`).

```powershell
$storage = "$env:USERPROFILE\Documents\frigate_buffer_storage"
docker run -d `
  --name frigate_buffer `
  --restart unless-stopped `
  --shm-size=1g `
  -p 5055:5055 `
  -v "${storage}:/app/storage" `
  --env-file "${storage}\.env" `
  -e DOCKER_ENV=true `
  -e LOG_LEVEL=INFO `
  --gpus all `
  -e NVIDIA_VISIBLE_DEVICES=all `
  -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility `
  frigate-buffer:latest
```

**Alternative (no variable — copy-paste and run):** Use the path directly so it works even in a new window. Includes `--env-file` for your `.env` in the storage folder.

```powershell
docker run -d --name frigate_buffer --restart unless-stopped --shm-size=1g -p 5055:5055 -v "$env:USERPROFILE\Documents\frigate_buffer_storage:/app/storage" --env-file "$env:USERPROFILE\Documents\frigate_buffer_storage\.env" -e DOCKER_ENV=true -e LOG_LEVEL=INFO --gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility frigate-buffer:latest
```

If you don't use a `.env` file, remove the `--env-file` line and set MQTT/URLs in `config.yaml` or add `-e MQTT_BROKER=...` etc. to the command. If the file doesn't exist yet, remove the line or create an empty `.env` in the storage folder.

**Without GPU** (e.g. to test UI/API only), omit the GPU flags:

```powershell
$storage = "$env:USERPROFILE\Documents\frigate_buffer_storage"
docker run -d `
  --name frigate_buffer `
  --restart unless-stopped `
  --shm-size=1g `
  -p 5055:5055 `
  -v "${storage}:/app/storage" `
  --env-file "${storage}\.env" `
  -e DOCKER_ENV=true `
  -e LOG_LEVEL=INFO `
  frigate-buffer:latest
```

## MQTT / Frigate

Set broker and URLs via env or in `config.yaml`. To pass via env, add before the image name:

```powershell
  -e MQTT_BROKER=192.168.1.100 `
  -e FRIGATE_URL=http://192.168.1.101:5000 `
  -e BUFFER_IP=192.168.1.102 `
```

(Replace with your actual IPs.)

## Open the app

- **Event viewer (main UI):** http://localhost:5055/player  
- **From another device on your network:** http://<your-pc-ip>:5055/player (replace `<your-pc-ip>` with your Windows PC’s local IP).
- **Status:** http://localhost:5055/status

## Useful commands

- **Logs:** `docker logs -f frigate_buffer`
- **Stop:** `docker stop frigate_buffer`
- **Remove:** `docker rm -f frigate_buffer` (then you can run again with the same `docker run`)
