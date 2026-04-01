# AMD ROCm hardware smoke (gpu-03 Phase 6)

Use this after **`Dockerfile.rocm`** and **`rocm_docker_build.yml`** (CPU-only image smoke). Phase 6 checks **ROCm + DRI inside a container** on a real AMD GPU host.

## Prerequisites

- Linux host with **AMD GPU**, **`/dev/kfd`**, and **`/dev/dri/renderD*`**.
- Docker. User in **`video`** and **`render`** groups (or use **`docker run --group-add`** as in the script).
- Repo cloned; first **`Dockerfile.rocm`** build is large ( **`rocm/pytorch`** base).

## One-shot script (recommended)

From the repo root:

```bash
chmod +x scripts/run_amd_rocm_docker_smoke.sh
./scripts/run_amd_rocm_docker_smoke.sh --strict --strict-native
```

Environment:

| Variable       | Default                 | Meaning                            |
|----------------|-------------------------|------------------------------------|
| `KFD`          | `/dev/kfd`              | Kernel fusion device               |
| `RENDER_NODE`  | `/dev/dri/renderD128`   | DRM render node to pass through    |
| `IMAGE`        | `frigate-buffer:rocm`   | Image tag for `docker build`/`run` |
| `BUILD`        | `1`                     | Set `0` to skip `docker build`     |

Decode a clip **inside** the container (bind-mount host file):

```bash
docker build -f Dockerfile.rocm -t frigate-buffer:rocm .
docker run --rm \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  --group-add video --group-add render \
  -v /path/on/host/clip.mp4:/tmp/clip.mp4:ro \
  frigate-buffer:rocm \
  python3 scripts/smoke_amd_rocm_torch.py --strict --strict-native /tmp/clip.mp4
```

## Smoke flags

See **`scripts/smoke_amd_rocm_torch.py --help`**. Notable:

- **`--strict`** — exit 2 if ROCm PyTorch path is not usable (**`torch.cuda.is_available()`** and **`torch.version.hip`** set).
- **`--strict-native`** — exit 2 if **`frigate_amd_decode`** cannot be imported (image must include the built **`.so`**). Runs even when ROCm is not visible (e.g. image sanity without **`/dev/kfd`**), so **`--strict`** is not required for that check alone.
- Optional **clip path** — first argument after flags: runs **`decode_first_frame_bchw_rgb`** when native import succeeds.

## GitHub Actions (self-hosted)

**`.github/workflows/amd_rocm_smoke.yml`** is **manual** (**`workflow_dispatch`**) and targets **`runs-on: [self-hosted, amd-rocm]`**. Add a runner with that label, or edit the workflow labels to match your fleet.

## Compose

**`docker-compose.rocm.example.yml`** uses the same **`devices:`** / **`group_add:`** pattern. The script above is the shortest path for a one-off verify.
