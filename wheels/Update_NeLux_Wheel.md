# Updating the NeLux Wheel

When a new NeLux version is released, rebuild the wheel by running this single command from the directory containing your existing wheel (e.g. `/mnt/user/Drive`):

```bash
cd /mnt/user/Drive
```

```bash
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  nelux-builder bash -c " \
    export PATH=/opt/build-env/bin:\$PATH && \
    git clone https://github.com/NevermindNilas/NeLux.git /tmp/NeLux && \
    cd /tmp/NeLux && \
    mkdir -p external/ffmpeg/include external/ffmpeg/lib && \
    cp -rL /usr/include/x86_64-linux-gnu/libavcodec /usr/include/x86_64-linux-gnu/libavformat \
       /usr/include/x86_64-linux-gnu/libavutil /usr/include/x86_64-linux-gnu/libavfilter \
       /usr/include/x86_64-linux-gnu/libswscale /usr/include/x86_64-linux-gnu/libswresample \
       /usr/include/x86_64-linux-gnu/libavdevice external/ffmpeg/include/ && \
    for lib in avcodec avformat avutil avfilter swscale swresample avdevice; do \
      cp -L /usr/lib/x86_64-linux-gnu/lib\${lib}.so external/ffmpeg/lib/; \
    done && \
    sed -i 's/find_package(Python3 3.13/find_package(Python3 3.12/' CMakeLists.txt && \
    sed -i 's/SPDLOG_FMT_EXTERNAL/SPDLOG_FMT_BUNDLED/' CMakeLists.txt && \
    sed -i '/FMT_HEADER_ONLY/d' CMakeLists.txt && \
    sed -i 's/requires-python = \">=3.13\"/requires-python = \">=3.12\"/' pyproject.toml && \
    python -m build --wheel --no-isolation && \
    cp dist/*.whl /workspace/ \
"
```

The new `.whl` file will appear in your current directory. Replace the old wheel in your project with the new one.

---

## Notes

- The `nelux-builder` Docker image must already exist. If it doesn't, see `Create_NeLux_Wheel.md` for the full setup.
- Build takes 8–18 minutes. No base image rebuild needed.
- If NeLux changes its Python version requirement (e.g. `3.14`), update the two `sed` lines that reference `3.13` accordingly.
- If the build fails, check the troubleshooting section in `Create_NeLux_Wheel.md`.
