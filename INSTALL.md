# Frigate Event Buffer — Clean install

One layout only: **clone the repo at the stack root** so that the stack directory is the repo root (you have `Dockerfile`, `src/`, `scripts/`, `docker-compose.yaml` as siblings). This doc covers a clean server and first-time install; update steps if you hit issues so others can reproduce.

## Prerequisites

- Docker (and Docker Compose).
- For GPU/NVENC: NVIDIA GPU, driver, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (so `docker run --gpus all` works). See [BUILD_NVENC.md](BUILD_NVENC.md).

---

## Option A: Console (copy-paste from repo root)

Use the same directory as both **stack root** and **repo root** (e.g. `/mnt/user/appdata/dockge/stacks/frigate-buffer`).

### 1. Clean (recommended if you had a previous install)

```bash
cd /mnt/user/appdata/dockge/stacks/frigate-buffer
docker compose down
docker rmi frigate-buffer:latest 2>/dev/null || true
```

If this directory used to have the repo inside a `src/` subfolder, remove it and re-clone so the **stack root is the repo root** (step 2).

### 2. Clone at stack root

**Option 2a — clone into existing stack dir (e.g. Dockge already created the folder):**

```bash
cd /mnt/user/appdata/dockge/stacks/frigate-buffer
git clone https://github.com/jballard86/frigate-event-buffer.git .
```

**Option 2b — clone into a new folder, then use that as stack root:**

```bash
cd /mnt/user/appdata/dockge/stacks
git clone https://github.com/jballard86/frigate-event-buffer.git frigate-buffer
cd frigate-buffer
```

You must have `Dockerfile`, `src/`, `scripts/`, and `docker-compose.yaml` in the same directory.

### 3. One-time: FFmpeg NVENC (GPU hosts only)

If you want GPU-accelerated transcoding, run once from repo root:

```bash
chmod +x scripts/build-ffmpeg-nvenc.sh
./scripts/build-ffmpeg-nvenc.sh
```

Takes 15–30 minutes the first time. See [BUILD_NVENC.md](BUILD_NVENC.md) if it fails. The image build expects this directory to exist; if you don’t have an NVIDIA GPU, run the script on a host that does and copy `src/ffmpeg-nvenc-artifacts/` to this repo, or see BUILD_NVENC.md for options.

### 4. Config

Create and edit your config (e.g. where the compose file mounts it):

```bash
mkdir -p /mnt/user/appdata/frigate_buffer
cp config.example.yaml /mnt/user/appdata/frigate_buffer/config.yaml
# Edit config.yaml with your cameras, Frigate URL, MQTT broker, etc.
```

Adjust the path if your `docker-compose.yaml` mounts a different location for `config.yaml`.

### 5. Build

From the **repo root** (same directory as `Dockerfile`):

```bash
docker build -t frigate-buffer:latest .
```

### 6. Run

```bash
docker compose up -d
```

Or use Dockge: set the stack path to this directory and use “Start” (after building in step 5 or via Dockge “Build”).

---

## Option B: Dockge

1. **Stack path** = directory that contains `docker-compose.yaml` and the repo (i.e. repo root). Example: `/mnt/user/appdata/dockge/stacks/frigate-buffer` — and that directory must contain `Dockerfile`, `src/`, `scripts/`, not a subfolder named `src` that contains the repo.
2. **First time:** In the stack terminal, run the NVENC script once if you use GPU (see Option A, step 3). Then in Dockge use **Build** (or run `docker compose build` in the stack terminal), then **Start**.
3. **Config:** Ensure the config file path in `docker-compose.yaml` (e.g. `/mnt/user/appdata/frigate_buffer/config.yaml`) exists and is edited (Option A, step 4).
4. **Updates:** From the stack terminal run `git pull`, then in Dockge use **Build** and **Restart**.

---

## Troubleshooting

- **Build fails with "no such file or directory" for ffmpeg-nvenc-artifacts** — Run `./scripts/build-ffmpeg-nvenc.sh` once from repo root (Option A, step 3), or see [BUILD_NVENC.md](BUILD_NVENC.md).
- **Build fails with "frigate_buffer" or "config.example" not found** — You are not in the repo root, or the repo was cloned inside `src/`. Re-clone so that the directory where you run `docker build .` contains `Dockerfile`, `src/frigate_buffer/`, and `config.example.yaml`.
- **Entrypoint or OpenCV errors at runtime** — The compose file mounts `./entrypoint.sh` from repo root; ensure that file exists and is executable. See [COMPOSE_ENTRYPOINT.md](COMPOSE_ENTRYPOINT.md) if you need the optional entrypoint for fixing libs without rebuilding.
