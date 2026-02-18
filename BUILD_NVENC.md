# Building the image with NVENC (GPU) support

The app image includes FFmpeg built with NVIDIA NVENC so transcoding uses the GPU. Because the build cannot access host driver libs during `docker build`, FFmpeg is built in a **one-off container** that has GPU access; the resulting binaries are then included in the image via the build context.

**Layout:** Clone the repo at stack root (so you have `Dockerfile`, `src/`, `scripts/` as siblings). See [INSTALL.md](INSTALL.md) for full clean-install steps.

## Prerequisites

- Host with an **NVIDIA GPU** and the **NVIDIA driver** installed (e.g. Unraid Tower with the Nvidia driver plugin).
- **Docker** with **NVIDIA Container Toolkit** (so `docker run --gpus all` works).

## Order: pull first, then run the script, then build

The script lives at `scripts/build-ffmpeg-nvenc.sh`. **Pull** to get the latest code, **then** run the script once on the build host, **then** build the image. You do **not** run the script before every build—only when the artifacts are missing (e.g. first time on a host, or after deleting them).

## Step 1: Build FFmpeg with NVENC (once per host / driver or FFmpeg version)

From the **repo root** (stack root):

```bash
cd /mnt/user/appdata/dockge/stacks/frigate-buffer   # or your stack root
git pull
chmod +x scripts/build-ffmpeg-nvenc.sh
./scripts/build-ffmpeg-nvenc.sh
```

- The script writes artifacts to **`src/ffmpeg-nvenc-artifacts/`** so `docker build .` finds them.
- **"NVENC was enabled at configure time; runtime -encoders did not list it (no GPU in build container). Treating as success."** — Normal. The build container has no GPU, so the runtime check is skipped; the artifacts are still valid and are copied out.
- First run can take 15–30 minutes. You do **not** need to run it before every build—only when artifacts are missing or you want to refresh (e.g. new driver or FFmpeg version).
- If the script fails (e.g. "NVENC not in build"), ensure the host has the NVIDIA driver and that `docker run --gpus all` works. See Troubleshooting below.

## Step 2: Build the app image

From the **repo root**:

```bash
docker build -t frigate-buffer:latest .
```

- If you see **"no such file or directory"** for `ffmpeg-nvenc-artifacts/`, run Step 1 first so artifacts exist in `src/ffmpeg-nvenc-artifacts/`.

## Dockge

1. Stack path must be the **repo root** (directory that contains `Dockerfile`, `src/`, `scripts/`).
2. From the stack terminal, run the NVENC script once so `src/ffmpeg-nvenc-artifacts/` exists (Step 1).
3. Use "Build" in Dockge or run `docker compose build` from the stack directory. Then Start.

## Optional: script env vars

- `FFMPEG_VERSION` – FFmpeg version to build (default: `7.0.2`).
- `CUDA_IMAGE` – Base image for the build container (default: `nvidia/cuda:12.2.0-devel-ubuntu22.04`).

Example:

```bash
FFMPEG_VERSION=7.0.2 ./scripts/build-ffmpeg-nvenc.sh
```

## Troubleshooting

The script checks that the built FFmpeg lists `h264_nvenc` among encoders. If that check fails, FFmpeg's configure didn't enable NVENC. The script sets `PKG_CONFIG_PATH` so configure can find `ffnvcodec.pc` (from nv-codec-headers) and passes `-I/usr/local/include` and `-L/usr/local/lib`. If the full check fails, the script prints the last 60 lines of configure output and nvenc-related lines. Ensure the host has the NVIDIA driver and that `docker run --rm --gpus all nvidia/cuda:12.2.0-devel-ubuntu22.04 nvidia-smi` works.

**Artifacts in `src/src/ffmpeg-nvenc-artifacts` instead of `src/ffmpeg-nvenc-artifacts`** — If the script reported output under `src/src/`, move them: `mv src/src/ffmpeg-nvenc-artifacts src/ffmpeg-nvenc-artifacts` (from the repo root), then `rmdir src/src` if empty. Run the script from the repo root so future runs write to `src/ffmpeg-nvenc-artifacts/`.

## When to re-run the script

- After updating the **NVIDIA driver** on the host (different ABI).
- When you want a **new FFmpeg version** (set `FFMPEG_VERSION` and re-run).
- If you delete `src/ffmpeg-nvenc-artifacts/` or build on another machine that doesn't have the artifacts.

Artifacts are in `.gitignore`; they are not committed. Each host (or each place you build the image) that needs NVENC must run the script once (or copy the artifacts from a host that has already run it).
