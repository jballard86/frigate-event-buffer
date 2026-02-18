# Building the image with NVENC (GPU) support

The app image includes FFmpeg built with NVIDIA NVENC so transcoding uses the GPU. Because the build cannot access host driver libs during `docker build`, FFmpeg is built in a **one-off container** that has GPU access; the resulting binaries are then included in the image via the build context.

## Prerequisites

- Host with an **NVIDIA GPU** and the **NVIDIA driver** installed (e.g. Unraid Tower with the Nvidia driver plugin).
- **Docker** with **NVIDIA Container Toolkit** (so `docker run --gpus all` works).

## Order: pull first, then run the script, then build

The script lives in the repo (`scripts/build-ffmpeg-nvenc.sh`). So: **pull** to get the latest code (including the script), **then** run the script once on the build host, **then** build the image. You do **not** run the script before every build—only when the artifacts are missing (e.g. first time on a host, or after deleting them).

## Step 1: Build FFmpeg with NVENC (once per host / driver or FFmpeg version)

Pull first, then run the script. **Where the script is** depends on where the repo is cloned:

- **If the repo is inside `src/`** (e.g. Dockge: stack dir is `frigate-buffer`, you run `cd src && git pull`): the script is at **`src/scripts/build-ffmpeg-nvenc.sh`**. From the stack directory:

```bash
cd /mnt/user/appdata/dockge/stacks/frigate-buffer
cd src && git pull && cd ..
chmod +x src/scripts/build-ffmpeg-nvenc.sh
./src/scripts/build-ffmpeg-nvenc.sh
```

- **If the repo root is the stack directory** (you have `scripts/` and `src/` as siblings): the script is at **`scripts/build-ffmpeg-nvenc.sh`**:

```bash
cd /mnt/user/appdata/dockge/stacks/frigate-buffer
git pull
chmod +x scripts/build-ffmpeg-nvenc.sh
./scripts/build-ffmpeg-nvenc.sh
```

- The script writes artifacts into the directory that will be the **build context** (so `docker build -f src/Dockerfile src` finds them). That is either `src/ffmpeg-nvenc-artifacts/` (when repo root is the parent of `src/`) or `src/ffmpeg-nvenc-artifacts/` inside the clone (when the repo is in `src/`).
- First run can take 15–30 minutes. You do **not** need to run it before every build—only when artifacts are missing or you want to refresh (e.g. new driver or FFmpeg version).
- **Script not found?** If you see "No such file or directory" for the script, you're likely in the stack directory with the repo inside `src/`. Use `src/scripts/build-ffmpeg-nvenc.sh` (or `cd src` then `./scripts/build-ffmpeg-nvenc.sh`).
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

Example (use the path that matches your layout—`scripts/` or `src/scripts/`):

```bash
FFMPEG_VERSION=7.0.2 ./src/scripts/build-ffmpeg-nvenc.sh
```

## When to re-run the script

- After updating the **NVIDIA driver** on the host (different ABI).
- When you want a **new FFmpeg version** (set `FFMPEG_VERSION` and re-run).
- If you delete `src/ffmpeg-nvenc-artifacts/` or build on another machine that doesn’t have the artifacts.

Artifacts are in `.gitignore`; they are not committed. Each host (or each place you build the image) that needs NVENC must run the script once (or copy the artifacts from a host that has already run it).
