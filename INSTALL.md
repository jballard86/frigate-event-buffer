# Frigate Event Buffer — Console install

Install and run via the command line only (no Dockge). Clone the repo so the **repo root** is the directory that contains `Dockerfile` and `src/`. All commands below are run from that directory or use its path.

**Required layout:** This guide assumes the repo has **`src/`** (with `src/frigate_buffer/` inside). If your `ls` shows `frigate_buffer/` at root instead of `src/`, you may be on a branch with a different layout. Switch to **`main`** (or the branch that has `src/`): run `git checkout main` then `git pull`, and confirm with `ls` that you have `Dockerfile` and `src/`.

## Prerequisites

- Docker (no compose required; we use `docker build` and `docker run` only).
- For GPU/NVENC: The image includes FFmpeg with NVENC and is based on Ubuntu 24.04 to match the FFmpeg donor for reliable GPU encoding. At runtime you need an NVIDIA GPU, driver, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (so `docker run --gpus all` works). See [BUILD_NVENC.md](BUILD_NVENC.md).

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

You are now in the repo root. Your prompt should show that path (e.g. `root@Tower:/mnt/user/appdata/frigate-buffer#`). **Check layout:** run `ls` and ensure you see **`Dockerfile`** and **`src/`**. If you see `frigate_buffer/` at root instead of `src/`, run `git checkout main` and `git pull`, then continue. If you cloned somewhere else, replace `/mnt/user/appdata/frigate-buffer` in the `cd` commands below with your repo root path.

**If Git prompts for username/password or says "Password authentication is not supported":** GitHub no longer accepts account passwords for HTTPS. Use either:
- **SSH** (if you have SSH keys added to GitHub): `git clone git@github.com:jballard86/frigate-event-buffer.git frigate-buffer`
- **Personal Access Token (PAT):** At GitHub → Settings → Developer settings → Personal access tokens, create a token (repo scope). When prompted for password, paste the token instead of your account password.

**Avoid being prompted every time:** After the first successful login you can have Git remember credentials.
- **Option A — Store (saves on disk):** From the repo root run `git config --global credential.helper store`. The next time you enter username and PAT, Git will save them (in `~/.git-credentials`) and reuse them for future pulls/fetches.
- **Option B — Cache (memory only, for a while):** Run `git config --global credential.helper 'cache --timeout=86400'` to cache for 24 hours so you’re not asked again in that period.
- **Option C — Use SSH:** If you use SSH to clone (or switch the remote to SSH), you won’t be prompted for a password as long as your SSH key is loaded. To switch an existing clone to SSH: `cd /mnt/user/appdata/frigate-buffer` then `git remote set-url origin git@github.com:jballard86/frigate-event-buffer.git`. Then `git pull` and future pulls use SSH (no user/pass).

**If you get `fatal: destination path 'frigate-buffer' already exists and is not empty`:**
- If that folder **has a git repo** (contains `.git`): `cd frigate-buffer` then `git pull`, then continue from step 3.
- Otherwise use a different name: `git clone ... frigate-buffer-app` and `cd frigate-buffer-app`.

**If you prefer to clone into an existing empty directory** (e.g. you already created it):

```bash
cd /mnt/user/appdata/frigate-buffer
git clone https://github.com/jballard86/frigate-event-buffer.git .
```

**If you get `fatal: destination path '.' already exists and is not empty`:**

- If it **already has a clone** (contains `.git`): `git pull` and continue from step 3.
- If it's **not a git repo**: clone into a new folder instead (see first clone block above; use a different folder name if needed).

You must have `Dockerfile` and `src/` in the current directory (repo root).

---

## 3. Config

Create and edit your config. The paths below match the `docker run` in step 6; change them if you use different paths.

```bash
cd /mnt/user/appdata/frigate-buffer
mkdir -p /mnt/user/appdata/frigate_buffer
cp config.example.yaml /mnt/user/appdata/frigate_buffer/config.yaml
# Edit config.yaml: cameras, FRIGATE_URL, MQTT_BROKER, etc.
nano /mnt/user/appdata/frigate_buffer/config.yaml
```

---

## 4. Build

From repo root (prompt should show `.../frigate-buffer#`):

```bash
cd /mnt/user/appdata/frigate-buffer
docker build -t frigate-buffer:latest .
```

---

## 5. Run

Run from repo root so `$(pwd)/entrypoint.sh` resolves correctly. **Use the same env vars and volume paths as in your `docker-compose.yaml`**; the command below mirrors that file. The only value you must change is **`HA_IP`** — replace `YOUR_HOME_ASSISTANT_IP` with your Home Assistant IP (or leave it if you set it in the compose). If you use different paths or IPs in `docker-compose.yaml`, use those here instead.

If a container named `frigate_buffer` already exists, stop and remove it first: `docker stop frigate_buffer` then `docker rm frigate_buffer`. The commands below do that automatically.

**With NVIDIA GPU:** Use `NVIDIA_DRIVER_CAPABILITIES=compute,video,utility` (the `video` capability is required for NVENC; some hosts e.g. Unraid need this explicit list).

```bash
cd /mnt/user/appdata/frigate-buffer
chmod +x entrypoint.sh
docker stop frigate_buffer 2>/dev/null; docker rm frigate_buffer 2>/dev/null || true
docker run -d --name frigate_buffer --restart unless-stopped \
  --network bridge \
  -p 5055:5055 \
  -v "$(pwd)/entrypoint.sh:/entrypoint.sh:ro" \
  -v /mnt/user/appdata/frigate_buffer:/app/storage \
  -v /mnt/user/appdata/frigate_buffer/config.yaml:/app/config.yaml:ro \
  -v /etc/localtime:/etc/localtime:ro \
  -e HA_IP=YOUR_HOME_ASSISTANT_IP \
  -e FRIGATE_URL=http://REDACTED_LOCAL_IP:5000 \
  -e MQTT_BROKER=REDACTED_LOCAL_IP \
  -e MQTT_PORT=1883 \
  -e RETENTION_DAYS=3 \
  -e LOG_LEVEL=INFO \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
  -e ENABLE_NVENC=true \
  -e YOLO_CONFIG_DIR=/tmp/Ultralytics \
  --gpus all \
  --entrypoint /entrypoint.sh \
  frigate-buffer:latest python -m frigate_buffer.main
```
## 6. Update (after code changes)

From repo root:

```bash
cd /mnt/user/appdata/frigate-buffer
git pull
docker build -t frigate-buffer:latest .
# Then run the same docker run command from step 5 again.
```

## Switching branches

To use a different branch (e.g. a feature branch or `main`):

```bash
cd /mnt/user/appdata/frigate-buffer   # or your repo root
git fetch origin
git checkout main
git pull
```

Replace `<branch-name>` with the branch you want (e.g. `main`, `Multi_Cam_Review`). If you have uncommitted changes, either commit or stash them first, or Git may refuse to switch. After switching, rebuild the image (step 4) and rerun the container (step 5) if the code changed.

---



If `git pull` reports that local changes would be overwritten by merge (e.g. to a file that was removed upstream), discard those changes then pull: run `git checkout -- <file>` for each file listed (or `git restore <file>`), then run `git pull` again and continue with the build and restart steps.

---

## Troubleshooting

- **FFmpeg does not report NVENC encoders** — Rebuild the image from this repo (`docker build -t frigate-buffer:latest .`) and run the container with GPU access (`--gpus all` and NVIDIA env vars). Set **`NVIDIA_DRIVER_CAPABILITIES=compute,video,utility`** (the `video` capability is required for NVENC). Check startup logs for the FFmpeg stderr snippet (e.g. "Cannot load libnvidia-encode.so.1") and inside the container run `find /usr -name 'libnvidia-encode*'` to confirm encode libs are mounted. See [BUILD_NVENC.md](BUILD_NVENC.md). If this does not fix it, next steps are (4) LD_LIBRARY_PATH and (5) trying Frigate's FFmpeg image as donor — see BUILD_NVENC.md.
- **Build fails with "frigate_buffer" or "config.example" not found** — You are not in the repo root. `cd` to the directory that contains `Dockerfile` and `src/frigate_buffer/`.
- **`exec: "/entrypoint.sh": permission denied`** — The mounted entrypoint script must be executable on the host. From repo root run: `chmod +x entrypoint.sh`, then start the container again (remove it first: `docker rm frigate_buffer`).
- **Entrypoint or OpenCV errors at runtime** — Ensure `entrypoint.sh` exists in repo root and is executable: `chmod +x entrypoint.sh`. See [COMPOSE_ENTRYPOINT.md](COMPOSE_ENTRYPOINT.md) if you need the optional entrypoint for fixing libs without rebuilding.
- **"Reader process died for camera … read of closed file"** — This occurs when two or more clips are decoded with NVDEC (GPU) at once and the driver hits contention. The default is now `multi_cam.decode_second_camera_cpu_only: true`, so the second and later cameras use CPU decode and this should not happen. If you set it to `false`, set it back to `true` (or remove the override) to avoid dual-NVDEC.
