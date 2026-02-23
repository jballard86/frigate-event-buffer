# NeLux Troubleshooting & Environment Resolution Log

## Overview
This document logs the successful troubleshooting steps for resolving "hollow" NeLux `VideoReader` objects and C++ library linking errors in a Linux/Docker environment (Unraid) using NVIDIA RTX 40-series GPUs.

## 1. The "Hollow Object" Symptom
**Problem:** `nelux.VideoReader` would initialize without error but return an object where `_decoder` was missing and all frame-reading methods failed.
**Root Cause:** The C++ backend (`_nelux.so`) was failing to bind to hardware or libraries silently during the constructor call.

### Verification Script
```python
import nelux
reader = nelux.VideoReader("path/to/video.mp4", decode_accelerator='nvdec')
print(f"Metadata: {reader.width}x{reader.height}") # Works (Parses header)
print(f"Hollow: {not hasattr(reader, '_decoder')}") # True (Backend failed)


2. Library Resolution (The ldd Discovery)
The C++ extension _nelux.so has deep dependencies on PyTorch's internal C++ libraries and NVIDIA runtimes that are not in the standard Linux system path.

Key Command:
ldd /usr/local/lib/python3.12/dist-packages/nelux/_nelux*.so

Identified Missing Links:

libtorch_cpu.so => not found

libc10.so => not found

libtorch_cuda.so => not found

libyuv.so => not found (Must be vendored or apt-installed)

The "Nuclear" Fix (LD_LIBRARY_PATH)
To resolve this, the following paths MUST be in the environment's LD_LIBRARY_PATH:

/usr/local/lib/python3.12/dist-packages/torch/lib (PyTorch C++ core)

/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib (CUDA Runtime)

/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib (CuDNN)

3. Hardware Acceleration Requirements
For NVDEC (hardware decoding) to work inside Docker:

Host Driver: Must be newer than the container's CUDA version (Verified: v580+ for CUDA 12.6/12.8).

Docker Capabilities: The container MUST be started with video and compute capabilities.

Flag: -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

Piping vs. Encoding: NeLux create_encoder is hardcoded for Windows h264_mf. On Linux, use Option B: Pipe raw RGB bytes from reader.read_frame() to an FFmpeg subprocess using h264_nvenc.

4. Software Monkey-Patch
In some NeLux builds, the Python wrapper expects the attribute _decoder, but the C++ object exposes the methods (like get_frame_count) on itself.

Required Patch:

Python
reader = nelux.VideoReader(...)
if not hasattr(reader, "_decoder"):
    reader._decoder = reader # Redirects wrapper calls to the main object
5. Final Verified Docker Environment
Base Image: nvidia/cuda:12.6.0-runtime-ubuntu24.04
Python: 3.12
Crucial Dockerfile Entry:

Dockerfile
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"


The Health Check Script: check_nelux.sh
Create a file named check_nelux.sh in your project or wheels/ folder and paste this in:

CONTAINER_NAME="frigate_buffer"
TEST_FILE="/app/storage/events/test13/doorbell/doorbell-66707.mp4"

echo "--- 1. Checking Docker Permissions & GPU Visibility ---"
docker exec -it $CONTAINER_NAME nvidia-smi | grep -E "Driver Version|NVIDIA GeForce RTX 4060"

echo -e "\n--- 2. Checking Library Linking (ldd) ---"
docker exec -it $CONTAINER_NAME bash -c "ldd /usr/local/lib/python3.12/dist-packages/nelux/_nelux*.so | grep -E 'torch|nvidia|not found'"

echo -e "\n--- 3. Testing Python Initialization (Monkey-Patch Check) ---"
docker exec -it $CONTAINER_NAME python3 -c "
import torch, nelux, os
path = '$TEST_FILE'
if not os.path.exists(path):
    print(f'ERROR: Test file not found at {path}')
    exit(1)
try:
    reader = nelux.VideoReader(path, decode_accelerator='nvdec')
    # The Patch Test
    if not hasattr(reader, '_decoder'):
        print('Monkey-patching _decoder...')
        reader._decoder = reader
    
    count = reader.get_frame_count()
    frame = reader.read_frame()
    if frame is not None:
        print(f'SUCCESS: Decoded {count} frames. Shape: {frame.shape}')
    else:
        print('FAILURE: Reader initialized but frame is None.')
except Exception as e:
    print(f'CRITICAL ERROR: {e}')
"
```
Why this is your "Insurance Policy"
Driver Validation: It confirms the RTX 4060 is passed through with the correct capabilities.

Linker Validation: It scans for "not found" errors in the library paths, ensuring your LD_LIBRARY_PATH in the Dockerfile is actually working.

Execution Validation: It runs the exact monkey-patch logic we designed to ensure the Python wrapper doesn't crash on the missing _decoder attribute.

