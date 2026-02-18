# GPU / NVENC support

The app image is built with FFmpeg that includes NVENC (multi-stage build from an image that provides FFmpeg+NVENC). **No script or artifact folder is required.**

The base image (e.g. `jrottenberg/ffmpeg:7.0-nvidia2204`) uses standard Linux paths: FFmpeg and ffprobe are in `/usr/local/bin`, and shared libs in `/usr/local/lib`. The Dockerfile copies from those locations into the final image (not from a custom `/opt/ffmpeg` layout). The **app image is based on Ubuntu 24.04** (Python 3.12 from default repos); the Ubuntu base matches the FFmpeg donor and avoids distro/ABI mismatch (e.g. with Debian) that can cause FFmpeg to silently omit NVENC when NVIDIA driver libs are injected at runtime.

- **Build** (from repo root): `docker build -t frigate-buffer:latest .`
- **Runtime:** For GPU transcode you need an NVIDIA GPU, driver, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Run the container with the GPU reserved (e.g. `--gpus all` or compose `deploy.resources.reservations.devices`). See [INSTALL.md](INSTALL.md) for full install and run steps.

At startup the app logs GPU and NVENC status (nvidia-smi, FFmpeg encoders). If FFmpeg does not report NVENC encoders, rebuild the image from this repo and ensure the container is run with GPU access.
