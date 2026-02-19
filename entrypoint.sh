#!/bin/sh
# Install OpenCV runtime deps at startup when the image was built without them.
# Remove this once the image is rebuilt with libglib2.0-0 (and related) in the Dockerfile.
set -e
if ! python -c "import cv2" 2>/dev/null; then
  apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    libglib2.0-0 libgl1 libxcb1 libxcb-shm0 libxext6 libxrender1 libsm6 \
    && rm -rf /var/lib/apt/lists/*
fi
exec "$@"
