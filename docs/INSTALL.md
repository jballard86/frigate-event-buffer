# Frigate Event Buffer — Console install

Install and run via the command line only. You can use **plain Docker** (`docker build` + `docker run`) or **Docker Compose** (`docker compose up -d --build`). You need **Docker**; for video features use an **NVIDIA GPU** (default **`Dockerfile`**) or **Intel Arc** (**`Dockerfile.intel`** + DRI) or **AMD ROCm** (**`Dockerfile.rocm`** + **`/dev/kfd`** and DRI). Frigate and MQTT are required; Home Assistant and Gemini API are optional.

Clone the repo so the **repo root** is the directory that contains `Dockerfile` and `src/`. All commands below are run from that directory. **In the steps below, replace `/mnt/user/appdata/frigate-buffer` with your repo root path if you cloned somewhere else.**

**Layout:** The repo must have `src/` (with `src/frigate_buffer/` inside). If `ls` shows `frigate_buffer/` at root instead of `src/`, run `git checkout main` and `git pull`, then confirm you have `Dockerfile` and `src/`.

## Prerequisites

- **Docker** — Use `docker build` and `docker run`, or Docker Compose (`docker-compose up -d --build`).
- **For GPU (NVIDIA):** Driver and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Use `docker run --gpus all` and set **`NVIDIA_DRIVER_CAPABILITIES=compute,video,utility`**. Video uses PyNvVideoCodec + FFmpeg; there is no CPU decode fallback on that path.
- **For GPU (Intel):** Use **`Dockerfile.intel`** and pass **`/dev/dri/renderD*`** into the container (see below). Decode uses **`frigate_intel_decode`** (QSV when available). Host Intel / VAAPI drivers apply.

### GPU vendor and adapter index (config)

The app selects the decode/runtime bundle via **`GPU_VENDOR`** (flat config; default **`nvidia`**). **`nvidia`** uses PyNvVideoCodec (CUDA). **`intel`** uses the **`frigate_intel_decode`** native extension (FFmpeg QSV when available; build from `native/intel_decode/`) plus Intel QSV FFmpeg helpers; see `docs/Multi_GPU_Support_Integration_Plan/gpu-02-intel-arc.md`. **`amd`** selects **`services/gpu_backends/amd/`** plus native **`frigate_amd_decode`** (build from **`native/amd_decode/`**, **`scripts/build_amd_decode.sh`**; see **`native/amd_decode/README.md`** and `docs/Multi_GPU_Support_Integration_Plan/gpu-03-amd-rocm.md`). Other vendors fail at startup.

**Which physical GPU** uses **`GPU_DEVICE_INDEX`** (default **`0`**): for NVIDIA and AMD (ROCm), `cuda:N`; for Intel decode, the native session’s adapter index; YOLO defaults follow **`DETECTION_DEVICE`** or vendor runtime (`xpu:N` when Intel Extension for PyTorch is available; `cuda:N` when ROCm torch sees a GPU). Legacy **`CUDA_DEVICE_INDEX`** is still merged; prefer **`GPU_DEVICE_INDEX`**.

Set them in **`config.yaml`** under **`multi_cam:`** as **`gpu_vendor`**, **`gpu_device_index`** (preferred), or deprecated **`cuda_device_index`**. You can also set **`GPU_VENDOR`**, **`GPU_DEVICE_INDEX`**, or **`CUDA_DEVICE_INDEX`** in **`.env`** / Docker **`environment`** — they override YAML after merge.

**Docker note:** `NVIDIA_VISIBLE_DEVICES` restricts which GPUs the container sees; the in-app index is then **relative to that visible set** (e.g. one GPU exposed → use index `0`). See **`examples/config.example.yaml`** and **`examples/.env.example`**.

### Intel GPU Docker (multi-stage)

For **`GPU_VENDOR=intel`**, build the Intel image (compiles **`native/intel_decode`** in stage 1; runtime has **no** compiler toolchain):

```bash
docker build -f Dockerfile.intel -t frigate-buffer:intel .
```

**CI:** On push/PR to **`main`** / **`master`**, **`.github/workflows/ci.yml`** runs **Ruff**, **pytest**, and **`docker build -f Dockerfile.intel`** so the native extension and image stay buildable.

- **Runtime** installs **`requirements-intel.txt`** (same as **`requirements.txt`** but **without** `PyNvVideoCodec`) and **`pip install --no-deps .`** so the metadata dependency on PyNv is not pulled.
- **Torch / torchvision** are **pinned** in **`requirements-intel.txt`**; the **builder** installs only those `torch==` / `torchvision==` lines (CPU wheels via **`TORCH_INDEX_URL`**, default PyTorch CPU) so **`frigate_intel_decode`** matches runtime libtorch. Bump both pins together when upgrading.
- **Smoke (optional):** after build, `docker run --rm … frigate-buffer:intel python3 scripts/smoke_intel_gpu_path.py --strict` checks **`import frigate_intel_decode`** and torch. Add **`--vainfo`** for VA-API output; **`--strict-dri`** with **`--vainfo`** fails if **`vainfo`** is missing or errors (use on Arc hosts with **`/dev/dri`** passed through). Add a bind-mounted test clip to run SW decode: `… python3 scripts/smoke_intel_gpu_path.py --strict /path/in/container/clip.mp4`.
- **Hardware verify (Phase 8):** on a Linux host with Intel GPU + Docker, **`./scripts/run_intel_arc_docker_smoke.sh --strict --vainfo`** (see **`docs/Multi_GPU_Support_Integration_Plan/intel-arc-hardware-smoke.md`**). Manual self-hosted CI: **`.github/workflows/intel_arc_smoke.yml`**.
- **Entrypoint** **`scripts/docker_entrypoint_intel.sh`** prepends **`torch/lib`** to **`LD_LIBRARY_PATH`** so **`import frigate_intel_decode`** resolves **`libtorch`**.
- **Compose:** use **`docker-compose.intel.example.yml`** as a template: **`devices:`** `/dev/dri/renderD128` (change if your Arc node differs), **`group_add:`** `video` and `render` (or numeric GIDs from the host if group names are missing).
- **XPU / IPEX:** build with **`--build-arg INSTALL_IPEX=1`** (or Compose **`INSTALL_IPEX: "1"`**) to **`pip install intel-extension-for-pytorch`** after the main deps. This can change the effective **`torch`** build; confirm versions against [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/). For bare metal, install IPEX per Intel’s docs and set **`DETECTION_DEVICE`** / rely on **`default_detection_device`** when **`torch.xpu`** is available.
- **QSV compilation encode:** optional **`multi_cam.intel.qsv_encode_preset`** and **`qsv_encode_global_quality`** in **`config.yaml`** tune **`h264_qsv`** for summary MP4s; env **`INTEL_QSV_ENCODE_PRESET`** and **`INTEL_QSV_ENCODE_GLOBAL_QUALITY`** override YAML (see **`examples/config.example.yaml`**).

### AMD ROCm (GPU_VENDOR=amd; gpu-03)

- **Dependencies:** **`requirements-rocm.txt`** mirrors **`requirements-intel.txt`** (no **`PyNvVideoCodec`**) but pins **`torch`** / **`torchvision`** for a **ROCm** wheel. Install with a PyTorch ROCm index URL that matches your driver, for example:
  `pip install -r requirements-rocm.txt --extra-index-url https://download.pytorch.org/whl/rocm6.2`
  (see [PyTorch Get Started](https://pytorch.org/get-started/locally/) for current indices). **`Dockerfile.rocm`** defaults to **`rocm/pytorch:…pytorch_release_2.6.0`** and **`rocm6.4`** pip index — keep **`ROCM_PYTORCH_IMAGE`** and **`PYTORCH_ROCM_INDEX`** build-args aligned when you override them.
- **AMD GPU Docker (multi-stage):** **`docker build -f Dockerfile.rocm -t frigate-buffer:rocm .`** — stage 1 compiles **`native/amd_decode`** against the base image’s libtorch; stage 2 installs **`requirements-rocm.txt`**, copies **`frigate_amd_decode*.so`** to **`/usr/local/lib/frigate_amd_decode`**, sets **`PYTHONPATH`** and **`GPU_VENDOR=amd`**, and uses **`scripts/docker_entrypoint_rocm.sh`** for **`LD_LIBRARY_PATH`**. Runtime has **no** compiler toolchain. **Compose:** **`docker-compose.rocm.example.yml`** — pass **`/dev/kfd`**, **`/dev/dri/renderD128`** (or your render node), and **`group_add:`** **`video`** / **`render`**. The base image is **very large**; CI does not build it on every push — use **`.github/workflows/rocm_docker_build.yml`** (**workflow_dispatch**) when you need a registry build.
- **Native decode (bare metal):** on Linux, run **`./scripts/build_amd_decode.sh`** after installing FFmpeg **-dev** packages (see **`native/amd_decode/README.md`**). Add **`native/amd_decode/build`** to **`PYTHONPATH`** (or install the ``.so`` on the module search path) so **`import frigate_amd_decode`** succeeds.
- **Smoke:** **`python scripts/smoke_amd_rocm_torch.py`** — allocates a **`cuda:0`** tensor when **`torch.version.hip`** is set (ROCm build). Without ROCm, it prints **`skip`** and exits **`0`**; **`--strict`** exits **`2`** if the ROCm GPU path is missing; **`--strict-native`** exits **`2`** if **`frigate_amd_decode`** cannot be imported. Optional **clip path** decodes the first frame via native when built. **`--yolo`** imports **`ultralytics`** and prints its version (no model download). On a host with **`/dev/kfd`** + DRI, use **`./scripts/run_amd_rocm_docker_smoke.sh`** (**`docs/Multi_GPU_Support_Integration_Plan/amd-rocm-hardware-smoke.md`**).
- **pytest:** tests marked **`@pytest.mark.amd_gpu`** are **skipped** unless **`RUN_AMD_GPU_TESTS=1`** (see **`tests/test_smoke_amd_rocm_path.py`**).

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
- **env_file** points to your `.env` file, or add the same variables under **environment** in the compose file. The app still needs **MQTT_BROKER**, **FRIGATE_URL**, and **BUFFER_IP** (or **HA_IP**), either in `config.yaml` or via env. The default is `.env` in the **project root**. If you see "env file ... not found", copy `examples/.env.example` to `.env` in the repo root (e.g. `cp examples/.env.example .env` or on Windows `copy examples\.env.example .env`) and set your values, or remove the **env_file** block and set variables under **environment** or in your mounted config.
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
- **Decoder / get_frames errors** — Decode is GPU-only. Check GPU memory and driver; reduce concurrent load or clip length if needed. Ensure `NVIDIA_DRIVER_CAPABILITIES` includes `video`. If you use multiple GPUs, confirm **`GPU_DEVICE_INDEX`** (or **`multi_cam.gpu_device_index`**) matches the adapter you intend, and that **`NVIDIA_VISIBLE_DEVICES`** (if set) exposes that GPU to the container.
- **"current commit information was not captured" / git rev-parse warning** — Harmless. It appears when the build directory is not a git work tree (e.g. copied folder without `.git`). Clone the repo with `git clone` so `.git` exists, or ignore the warning.
- **"failed to solve: frontend grpc server closed unexpectedly"** — BuildKit can crash on some hosts (e.g. Unraid). Use the legacy builder: `DOCKER_BUILDKIT=0 docker-compose up -d --build`. The Dockerfile is written to work without BuildKit-only features so the build should succeed with the legacy builder. On Unraid you can also try increasing Docker memory (Settings → Docker → advanced) and retrying with BuildKit enabled.

