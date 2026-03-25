"""App-wide lock serializing GPU decode (create_decoder and frame pulls).

All vendors share this single lock so decode and tensor handoff stay ordered.
"""

from __future__ import annotations

import threading

# Serialize decoder access across video, multi_clip_extractor, and compilation.
GPU_LOCK = threading.Lock()
