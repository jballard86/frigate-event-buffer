# Building the image with NVENC (GPU) support

The app image includes FFmpeg built with NVIDIA NVENC so transcoding uses the GPU. Because the build cannot access host driver libs during `docker build`, FFmpeg is built in a **one-off container** that has GPU access; the resulting binaries are then included in the image via the build context.

## Prerequisites

- Host with an **NVIDIA GPU** and the **NVIDIA driver** installed (e.g. Unraid Tower with the Nvidia driver plugin).
- **Docker** with **NVIDIA Container Toolkit** (so `docker run --gpus all` works).

## Order: pull first, then run the script, then build

The script lives in the repo (`scripts/build-ffmpeg-nvenc.sh`). So: **pull** to get the latest code (including the script), **then** run the script once on the build host, **then** build the image. You do **not** run the script before every build—only when the artifacts are missing (e.g. first time on a host, or after deleting them).

## Step 1: Build FFmpeg with NVENC (once per host / driver or FFmpeg version)

From the **repo root** (e.g. `/mnt/user/appdata/dockge/stacks/frigate-buffer`). Pull first so you have the script, then run it:

```bash
cd /mnt/user/appdata/dockge/stacks/frigate-buffer
cd src && git pull && cd ..
chmod +x scripts/build-ffmpeg-nvenc.sh
./scripts/build-ffmpeg-nvenc.sh
```

- The script is in the repo; after `git pull` you have `scripts/build-ffmpeg-nvenc.sh`.
- It runs a container with `--gpus all`, builds FFmpeg with NVENC and libx264, and writes artifacts into **`src/ffmpeg-nvenc-artifacts/`** so `docker build -f src/Dockerfile src` finds them.
- First run can take 15–30 minutes. You do **not** need to run it before every build—only when artifacts are missing or you want to refresh (e.g. new driver or FFmpeg version).
- If the script fails (e.g. “NVENC not in build”), ensure the host has the NVIDIA driver and that `docker run --gpus all` works.

## Step 2: Build the app image

From the **repo root** (e.g. `/mnt/user/appdata/dockge/stacks/frigate-buffer`):

```bash
docker build -t frigate-buffer:latest -f src/Dockerfile src
```

- **Context is `src`**: the script wrote `ffmpeg-nvenc-artifacts/` into `src/`, so the context includes it. The Dockerfile is at `src/Dockerfile`.
- If you see **“no such file or directory”** for `ffmpeg-nvenc-artifacts/`, run Step 1 first from the repo root.
- **Alternative (context = repo root):** build with `docker build -t frigate-buffer:latest -f Dockerfile .` and run the script with `OUT_DIR="$REPO_ROOT/ffmpeg-nvenc-artifacts"` so artifacts are at repo root.

## Dockge

1. On Tower, from the stack directory (e.g. `/mnt/user/appdata/dockge/stacks/frigate-buffer`), run the script once so `src/ffmpeg-nvenc-artifacts/` exists.
2. Build as usual with context `src` and Dockerfile `src/Dockerfile` (e.g. `docker build -t frigate-buffer:latest -f src/Dockerfile src`). Dockge may run this from the stack directory after `git pull`; ensure the script has been run so artifacts are present.

## Optional: script env vars

- `FFMPEG_VERSION` – FFmpeg version to build (default: `7.0.2`).
- `CUDA_IMAGE` – Base image for the build container (default: `nvidia/cuda:12.2.0-devel-ubuntu22.04`).

Example:

```bash
FFMPEG_VERSION=7.0.2 ./scripts/build-ffmpeg-nvenc.sh
```

## When to re-run the script

- After updating the **NVIDIA driver** on the host (different ABI).
- When you want a **new FFmpeg version** (set `FFMPEG_VERSION` and re-run).
- If you delete `src/ffmpeg-nvenc-artifacts/` or build on another machine that doesn’t have the artifacts.

Artifacts are in `.gitignore`; they are not committed. Each host (or each place you build the image) that needs NVENC must run the script once (or copy the artifacts from a host that has already run it).
