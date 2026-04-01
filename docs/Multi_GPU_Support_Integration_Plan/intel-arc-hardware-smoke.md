# Intel Arc hardware smoke (gpu-02 Phase 8)

Use this after **Phase 7** CI (CPU-only image build + import smoke). Phase 8 checks **DRI inside a container** on a real Intel GPU host.

## Prerequisites

- Linux host with **Intel Arc** (or compatible) and **`/dev/dri/renderD*`**.
- Docker. User in `video` and `render` groups (or rely on `docker run --group-add` as in the script).
- Repo cloned; network for first **Dockerfile.intel** build if needed.

## One-shot script (recommended)

From the repo root:

```bash
chmod +x scripts/run_intel_arc_docker_smoke.sh
./scripts/run_intel_arc_docker_smoke.sh --strict --vainfo
```

Environment:

| Variable       | Default                 | Meaning                          |
|----------------|-------------------------|----------------------------------|
| `RENDER_NODE`  | `/dev/dri/renderD128`   | Host render node to pass through |
| `IMAGE`        | `frigate-buffer:intel`  | Image tag for `docker build`/`run` |
| `BUILD`        | `1`                     | Set `0` to skip `docker build`   |

Decode a clip **inside** the container (bind-mount host file):

```bash
docker build -f Dockerfile.intel -t frigate-buffer:intel .
docker run --rm \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  --group-add video --group-add render \
  -v /path/on/host/clip.mp4:/tmp/clip.mp4:ro \
  frigate-buffer:intel \
  python3 scripts/smoke_intel_gpu_path.py --strict --vainfo /tmp/clip.mp4
```

## Smoke flags

See **`scripts/smoke_intel_gpu_path.py --help`**. Notable:

- **`--strict`** — exit 2 if **`frigate_intel_decode`** cannot be imported.
- **`--vainfo`** — run **`vainfo --display drm`** (requires **`vainfo`** in the image; **Dockerfile.intel** installs it).
- **`--strict-dri`** — with **`--vainfo`**, exit 3 if **`vainfo`** is missing or fails (use on hardware hosts).

## GitHub Actions (self-hosted)

**`.github/workflows/intel_arc_smoke.yml`** is **manual** (`workflow_dispatch`) and targets **`runs-on: [self-hosted, intel-arc]`**. Add a self-hosted runner with that label in your org/repo, or edit the workflow labels to match your fleet.

## Compose

**`docker-compose.intel.example.yml`** is unchanged; use the same **`devices:`** / **`group_add:`** pattern. The script above is the shortest path for a one-off verify.
