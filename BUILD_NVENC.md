# GPU / NVENC support

The app image is built with FFmpeg that includes NVENC (multi-stage build from an image that provides FFmpeg+NVENC). **No script or artifact folder is required.**

- **Build** (from repo root): `docker build -t frigate-buffer:latest .`
- **Runtime:** For GPU transcode you need an NVIDIA GPU, driver, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Run the container with the GPU reserved (e.g. `--gpus all` or compose `deploy.resources.reservations.devices`). See [INSTALL.md](INSTALL.md) for full install and run steps.

At startup the app logs GPU and NVENC status (nvidia-smi, FFmpeg encoders). If FFmpeg does not report NVENC encoders, rebuild the image from this repo and ensure the container is run with GPU access.
