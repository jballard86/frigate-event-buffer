# GPU multi-vendor implementation plans

**Architecture:** NVIDIA = PyNvVideoCodec. **Intel / AMD decode** = **C++ pybind11** (FFmpeg + vendor HW → libtorch → Python), thin wrappers in `gpu_backends/` implementing **`DecoderContextProto`** (gpu-01). **Multi-stage Docker** builds the `.so` then ships a slim runtime image.

| Doc | Purpose |
|-----|---------|
| **[gpu-00-primary-multi-vendor-gpu-plan.md](./gpu-00-primary-multi-vendor-gpu-plan.md)** | **Primary plan** — analysis, architecture, summary (git-versioned). |

Executable sub-plans for multi-vendor GPU support and an **optional CPU-only** mode that **waives** normal GPU pipeline rules.

| Order | Document | Purpose |
|-------|----------|---------|
| 1 | [gpu-01-nvidia-refactor-and-prep.md](./gpu-01-nvidia-refactor-and-prep.md) | Refactor NVIDIA code into `gpu_backends/`, protocols, registry, config prep — **no** Intel/AMD/CPU implementation yet. |
| 2 | [gpu-02-intel-arc.md](./gpu-02-intel-arc.md) | Intel: **pybind11** + FFmpeg/oneVPL/QSV → libtorch/XPU; multi-stage Docker; QSV encode/GIF. |
| 3 | [gpu-03-amd-rocm.md](./gpu-03-amd-rocm.md) | AMD: **pybind11** + FFmpeg/AMF/VAAPI → libtorch ROCm; multi-stage Docker; AMF encode/GIF. |
| 4 | [gpu-04-cpu-only-path.md](./gpu-04-cpu-only-path.md) | **CPU-only (opt-in):** CPU decode, libx264, CPU GIF — **explicit MAP/PROCESSING exceptions**; dev/CI/no-GPU use. |

**Run order:** **1 → 2**, **1 → 3**, **1 → 4** (2, 3, 4 can proceed in parallel after 1).

The same primary plan may exist under `.cursor/plans/amd_intel_gpu_support_d5add60b.plan.md` for Cursor UI; **prefer [gpu-00-primary-multi-vendor-gpu-plan.md](./gpu-00-primary-multi-vendor-gpu-plan.md)** under `docs/Multi_GPU_Support_Integration_Plan/` for links and reviews.
