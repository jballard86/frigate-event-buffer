# GPU / NVENC support

The app image is built with FFmpeg that includes NVENC (multi-stage build from an image that provides FFmpeg+NVENC). **No script or artifact folder is required.**

The base image (e.g. `jrottenberg/ffmpeg:7.0-nvidia2204`) uses standard Linux paths: FFmpeg and ffprobe are in `/usr/local/bin`, and shared libs in `/usr/local/lib` (and sometimes `/usr/local/lib64` or `/usr/local/cuda/lib64` for NPP/CUDA libs like `libnppig.so.12`). The Dockerfile gathers libs from all those locations into the final image so FFmpeg can load them at runtime. The **app image is based on Ubuntu 24.04** (Python 3.12 from default repos); the Ubuntu base matches the FFmpeg donor and avoids distro/ABI mismatch (e.g. with Debian) that can cause FFmpeg to silently omit NVENC when NVIDIA driver libs are injected at runtime.

- **Build** (from repo root): `docker build -t frigate-buffer:latest .`
- **Runtime:** For GPU transcode you need an NVIDIA GPU, driver, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Run the container with the GPU reserved (e.g. `--gpus all` or compose `deploy.resources.reservations.devices`). Set **`NVIDIA_DRIVER_CAPABILITIES=compute,video,utility`** (as Frigate does); the `video` capability is required so the toolkit mounts encode libs (e.g. `libnvidia-encode.so.1`). See [INSTALL.md](INSTALL.md) for full install and run steps.

At startup the app logs GPU and NVENC status (nvidia-smi, libnvidia-encode check, FFmpeg encoders). The NVENC check reads both stderr and stdout of `ffmpeg -encoders` and may retry once after a short delay to work around startup timing on some hosts. If FFmpeg does not report NVENC encoders, check the logged **FFmpeg -encoders stderr snippet**. A missing **libnvidia-encode** means the container needs `NVIDIA_DRIVER_CAPABILITIES=compute,video,utility` and the NVIDIA Container Toolkit. A missing **libnppig.so.12** (or similar NPP lib) means the image was built without the NPP gather step — rebuild from this repo so the Dockerfile copies libs from the donor’s lib, lib64, and cuda/lib64.

### If NVENC still unavailable

If you have set `NVIDIA_DRIVER_CAPABILITIES=compute,video,utility`, rebuilt the image, and confirmed the container has GPU access but FFmpeg still does not list NVENC encoders, the next steps are:

- **(4) LD_LIBRARY_PATH** — Try running the container with `LD_LIBRARY_PATH` set so the loader can find the injected lib (e.g. `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}`). Use `find /usr -name 'libnvidia-encode*'` inside the container to see where the host mounted it.
- **(5) Alternative FFmpeg donor** — Try building with Frigate's FFmpeg image as the multi-stage donor (e.g. `blakeblackshear/frigate-ffmpeg`) instead of jrottenberg; that image is built on an NVIDIA CUDA base for the Frigate ecosystem. See the plan or repo for Dockerfile build-arg and paths.

### Pre-flight NVENC probe (main-thread init)

At startup, the app runs a **pre-flight NVENC probe on the main thread** (immediately after `log_gpu_status`). That establishes the CUDA/NVENC context before any worker threads run, which avoids returncode 234 when the first GPU use would otherwise happen in a ThreadPoolExecutor worker. The result is cached so workers do not run the probe again.

### Probe fails with returncode 234 after startup

If startup shows **NVENC success** but the probe (preflight or in-process) still fails with returncode 234, the app now:

- Logs the **exact FFmpeg command** used for the probe.
- Logs **extended stderr** (up to 2000 chars) when returncode is 234.
- Logs **returncode interpretations** (e.g. 256-22=SIGPIPE, possible EINVAL/EBUSY, AVERROR/NVENC hints) via `decode_nvenc_returncode()` in `video.py`.

Use these logs to see if 234 correlates with buffer/surface/CUDA messages. If 234 persists after preflight, consider GPU contention from other processes or driver/session limits.
