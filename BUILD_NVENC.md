# GPU / NVENC support

The app image is built with FFmpeg that includes NVENC (multi-stage build from an image that provides FFmpeg+NVENC). **No script or artifact folder is required.**

The base image (e.g. `jrottenberg/ffmpeg:7.0-nvidia2204`) uses standard Linux paths: FFmpeg and ffprobe are in `/usr/local/bin`, and shared libs in `/usr/local/lib`. The Dockerfile copies from those locations into the final image (not from a custom `/opt/ffmpeg` layout). The **app image is based on Ubuntu 24.04** (Python 3.12 from default repos); the Ubuntu base matches the FFmpeg donor and avoids distro/ABI mismatch (e.g. with Debian) that can cause FFmpeg to silently omit NVENC when NVIDIA driver libs are injected at runtime.

- **Build** (from repo root): `docker build -t frigate-buffer:latest .`
- **Runtime:** For GPU transcode you need an NVIDIA GPU, driver, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Run the container with the GPU reserved (e.g. `--gpus all` or compose `deploy.resources.reservations.devices`). Set **`NVIDIA_DRIVER_CAPABILITIES=compute,video,utility`** (as Frigate does); the `video` capability is required so the toolkit mounts encode libs (e.g. `libnvidia-encode.so.1`). See [INSTALL.md](INSTALL.md) for full install and run steps.

At startup the app logs GPU and NVENC status (nvidia-smi, libnvidia-encode check, FFmpeg encoders). If FFmpeg does not report NVENC encoders, check the logged **FFmpeg -encoders stderr snippet** (e.g. "Cannot load libnvidia-encode.so.1"). Inside the container run `find /usr -name 'libnvidia-encode*'` to confirm encode libs are mounted; if missing, fix NVIDIA_DRIVER_CAPABILITIES and Container Toolkit.

### If NVENC still unavailable

If you have set `NVIDIA_DRIVER_CAPABILITIES=compute,video,utility`, rebuilt the image, and confirmed the container has GPU access but FFmpeg still does not list NVENC encoders, the next steps are:

- **(4) LD_LIBRARY_PATH** — Try running the container with `LD_LIBRARY_PATH` set so the loader can find the injected lib (e.g. `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}`). Use `find /usr -name 'libnvidia-encode*'` inside the container to see where the host mounted it.
- **(5) Alternative FFmpeg donor** — Try building with Frigate's FFmpeg image as the multi-stage donor (e.g. `blakeblackshear/frigate-ffmpeg`) instead of jrottenberg; that image is built on an NVIDIA CUDA base for the Frigate ecosystem. See the plan or repo for Dockerfile build-arg and paths.
