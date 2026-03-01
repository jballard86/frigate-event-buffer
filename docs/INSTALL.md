# Frigate Event Buffer — Console install

Install and run via the command line only. You can use **plain Docker** (`docker build` + `docker run`) or **Docker Compose** (`docker-compose up -d --build`). You need **Docker**; for video features you need an **NVIDIA GPU**. Frigate and MQTT are required; Home Assistant and Gemini API are optional.

Clone the repo so the **repo root** is the directory that contains `Dockerfile` and `src/`. All commands below are run from that directory. **In the steps below, replace `/mnt/user/appdata/frigate-buffer` with your repo root path if you cloned somewhere else.**

**Layout:** The repo must have `src/` (with `src/frigate_buffer/` inside). If `ls` shows `frigate_buffer/` at root instead of `src/`, run `git checkout main` and `git pull`, then confirm you have `Dockerfile` and `src/`.

## Prerequisites

- **Docker** — Use `docker build` and `docker run`, or Docker Compose (`docker-compose up -d --build`).
- **For GPU (video decode/summary):** NVIDIA GPU, driver, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Use `docker run --gpus all` and set **`NVIDIA_DRIVER_CAPABILITIES=compute,video,utility`** — this is **mandatory** for GPU decode no matter how you start the container (plain `docker run` or Docker Compose). Video uses GPU only (PyNvVideoCodec + FFmpeg); there is no CPU decode fallback.

---

## 1. Clean (if you had a previous install)

From the directory where you will clone or already have the repo:

```bash
docker stop frigate_buffer 2>/dev/null
docker rm frigate_buffer 2>/dev/null
docker rmi frigate-buffer:latest 2>/dev/null || true
```

---

## 2. Clone the repo

**Into a new folder** (replace `/mnt/user/appdata` with any parent path you want):

```bash
cd /mnt/user/appdata
git clone https://github.com/jballard86/frigate-event-buffer.git frigate-buffer
cd frigate-buffer
```

You are now in the repo root. **Check layout:** run `ls` and ensure you see `Dockerfile` and `src/`. If you see `frigate_buffer/` at root instead of `src/`, run `git checkout main` and `git pull`, then continue.

**If the folder already exists or you want to clone into an existing directory:** If it already has a `.git` folder, `cd` into it and run `git pull`, then continue from step 3. Otherwise use a different folder name, or to clone into the current directory run `git clone https://github.com/jballard86/frigate-event-buffer.git .` (only if the directory is empty or already a git repo).

---

## 3. Config

Copy the example config and put it where you want (e.g. repo root or a separate folder). The `docker run` in step 5 will mount it into the container; the path in `-v` must match where you put it.

**Example: config in repo root**

```bash
cd /mnt/user/appdata/frigate-buffer
cp examples/config.example.yaml config.yaml
nano config.yaml
```

Edit `config.yaml`: set your **cameras**, and under **network:** set `mqtt_broker`, `frigate_url`, and `buffer_ip`. You can also set these via environment variables in step 5 instead. If you use another path for config.yaml, remember it for the `-v` mount in step 5.

**Optional — use a .env file:** Copy `examples/.env.example` to `.env`, fill in the required variables (MQTT_BROKER, FRIGATE_URL, BUFFER_IP), and in step 5 add `--env-file .env` to the `docker run` command.

---

## 4. Build

From repo root:

```bash
cd /mnt/user/appdata/frigate-buffer
docker build -t frigate-buffer:latest .
```

**Fast build (development only):** The default build swaps to opencv-python-headless and can take 20+ minutes. For quicker iteration, use:

```bash
docker build -t frigate-buffer:latest --build-arg USE_GUI_OPENCV=true .
```

This produces a larger image with X11 dependencies; use the default (headless) for production.  This builds first build will still be high.

---

## 5. Run

The app needs **MQTT_BROKER**, **FRIGATE_URL**, and **BUFFER_IP** (or **HA_IP**). Set them in `config.yaml` under `network:` or here with `-e`. Replace the placeholder paths and IPs with your own.

If a container named `frigate_buffer` already exists, the commands below stop and remove it first.

**With NVIDIA GPU:** You **must** pass `-e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility` (the `video` capability is required for GPU decode; some hosts e.g. Unraid need this). This is mandatory regardless of how you build/start the container.

```bash
cd /mnt/user/appdata/frigate-buffer
docker stop frigate_buffer 2>/dev/null; docker rm frigate_buffer 2>/dev/null || true
docker run -d \
  --name frigate_buffer \
  --restart unless-stopped \
  --network bridge \
  --shm-size=1g \
  -p 5055:5055 \
  -v /path/to/your/storage:/app/storage \
  -v /path/to/your/config.yaml:/app/config.yaml:ro \
  -v /etc/localtime:/etc/localtime:ro \
  -e BUFFER_IP=YOUR_BUFFER_IP \
  -e FRIGATE_URL=http://YOUR_FRIGATE_IP:5000 \
  -e MQTT_BROKER=YOUR_MQTT_BROKER_IP \
  -e MQTT_PORT=1883 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
  --runtime=nvidia \
  --gpus all \
  frigate-buffer:latest
```

You can use `HA_IP` instead of `BUFFER_IP`; both set the same value. To pass variables from a file, add `--env-file .env` before the image name.

The app uses the storage volume for Ultralytics config and the YOLO model cache (`storage/ultralytics/` and `storage/yolo_models/`). These persist across restarts. You do not need to set `YOLO_CONFIG_DIR` unless you want to override the default.

---

## 5b. Run with Docker Compose

If you have a `docker-compose.yml` in the repo root (or copied from `docker-compose.example.yaml` if present), you can build and start the app with:

```bash
cd /mnt/user/Drive/frigate-event-buffer
docker-compose up -d --build
```

- `**-d**` — Run in the background (detached).
- `**--build**` — Build the image before starting (so you get the latest code).

**Before running:** Edit `docker-compose.yml` so that:

- **volumes** point to your storage directory and your `config.yaml` path (e.g. `/mnt/user/Drive/frigate-event-buffer/config.yaml` if config is in the repo).
- **env_file** points to your `.env` file, or add the same variables under **environment** in the compose file. The app still needs **MQTT_BROKER**, **FRIGATE_URL**, and **BUFFER_IP** (or **HA_IP**), either in `config.yaml` or via env.
- **For GPU:** Set **`NVIDIA_DRIVER_CAPABILITIES=compute,video,utility`** in **environment** — this is **mandatory** for GPU decode no matter how you build/start (same as with plain `docker run`).

**Useful commands:**

- **View logs:** `docker-compose logs -f`
- **Stop:** `docker-compose down`
- **Restart after config change:** `docker-compose up -d` (no `--build` unless you changed code)

---

## 6. Update (after code changes)

From repo root: `git pull`, then rebuild (step 4), then `docker restart frigate_buffer`. Code-only rebuilds are usually fast (~1–2 min) because Docker caches the dependency layer; when you change `requirements.txt`, the first rebuild is slower (use the fast build with `USE_GUI_OPENCV=true` if you want to avoid the long opencv swap).

**Switching branches:** Run `git fetch origin`, then `git checkout main` (or the branch you want). If you have uncommitted changes, commit or stash them first. After switching, rebuild (step 4) and restart the container (step 5) if the code changed.

**If `git pull` says local changes would be overwritten:** Run `git checkout -- <file>` (or `git restore <file>`) for each file listed, then `git pull` again and rebuild/restart.

---

## Troubleshooting

- **FFmpeg / GPU decode issues** — Video decode is GPU-only (PyNvVideoCodec). Rebuild with `docker build -t frigate-buffer:latest .` and run with `--gpus all` and `NVIDIA_DRIVER_CAPABILITIES=compute,video,utility`. Check logs for `NVDEC hardware initialization failed`; inside the container run `nvidia-smi` to confirm the GPU is visible.
- **Build fails with "frigate_buffer" or "examples/config.example.yaml" not found** — You are not in the repo root. The Dockerfile expects the repo root as build context (the directory that contains `Dockerfile` and `src/`). `cd` to that directory and run the build again.
- **Decoder / get_frames errors** — Decode is GPU-only. Check GPU memory and driver; reduce concurrent load or clip length if needed. Ensure `NVIDIA_DRIVER_CAPABILITIES` includes `video`.

- **"current commit information was not captured" / git rev-parse warning** — Harmless. It appears when the build directory is not a git work tree (e.g. copied folder without `.git`). Clone the repo with `git clone` so `.git` exists, or ignore the warning.
- **"failed to solve: frontend grpc server closed unexpectedly"** — BuildKit can crash on some hosts (e.g. Unraid). Use the legacy builder: `DOCKER_BUILDKIT=0 docker-compose up -d --build`. The Dockerfile is written to work without BuildKit-only features so the build should succeed with the legacy builder. On Unraid you can also try increasing Docker memory (Settings → Docker → advanced) and retrying with BuildKit enabled.

