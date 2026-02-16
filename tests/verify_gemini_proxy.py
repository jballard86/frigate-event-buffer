#!/usr/bin/env python3
"""
Manual verification script for Gemini proxy connectivity.

Run from project root:
  python tests/verify_gemini_proxy.py

Loads real config.yaml, instantiates GeminiAnalysisService, sends a single
dummy image to the proxy, and prints request payload and response.
Asserts that the response contains expected JSON keys (title, potential_threat_level, etc.).
"""

import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from frigate_buffer.config import load_config
from frigate_buffer.services.ai_analyzer import GeminiAnalysisService


REQUIRED_KEYS = ("title", "potential_threat_level")
OPTIONAL_KEYS = ("shortSummary", "scene", "description", "confidence")


def main():
    print("Loading config...")
    try:
        config = load_config()
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        sys.exit(1)

    gemini = config.get("GEMINI") or {}
    if not gemini.get("enabled"):
        print("WARNING: gemini.enabled is false in config.")
    if not gemini.get("proxy_url") or not gemini.get("api_key"):
        print("ERROR: gemini.proxy_url and gemini.api_key must be set (or GEMINI_API_KEY env).")
        sys.exit(1)

    print("Creating GeminiAnalysisService...")
    service = GeminiAnalysisService(config)

    # Dummy image: 320x240 solid color (BGR)
    dummy_frame = cv2.imencode(".jpg", __dummy_image())[1].tobytes()
    import base64
    b64 = base64.b64encode(dummy_frame).decode("ascii")
    dummy_url = f"data:image/jpeg;base64,{b64}"
    # Build minimal payload for printing (same shape as service uses)
    system_prompt = "You are a security recap assistant. Describe the image. Output JSON with: title, shortSummary, scene, confidence, potential_threat_level."
    payload = {
        "model": service._model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze this security camera frame."},
                {"type": "image_url", "image_url": {"url": dummy_url}},
            ]},
        ],
    }
    print("\n--- Request payload (model, messages; image truncated) ---")
    # Print payload with truncated base64 for readability
    payload_debug = {
        "model": payload["model"],
        "messages": [
            payload["messages"][0],
            {
                "role": payload["messages"][1]["role"],
                "content": [
                    payload["messages"][1]["content"][0],
                    {"type": "image_url", "image_url": {"url": dummy_url[:50] + "..."}},
                ],
            },
        ],
    }
    print(json.dumps(payload_debug, indent=2))
    print("--- End request payload ---\n")

    # Create one numpy-style frame for send_to_proxy (it expects BGR arrays)
    dummy_np = __dummy_image()
    print("Calling proxy...")
    result = service.send_to_proxy(system_prompt, [dummy_np])

    if result is None:
        print("ERROR: send_to_proxy returned None (check logs for proxy/network errors).")
        sys.exit(1)

    print("\n--- Raw response (parsed JSON) ---")
    print(json.dumps(result, indent=2))
    print("--- End response ---\n")

    for key in REQUIRED_KEYS:
        if key not in result:
            print(f"ERROR: Response missing required key: {key}")
            sys.exit(1)
    print(f"OK: Required keys present: {REQUIRED_KEYS}")
    for key in OPTIONAL_KEYS:
        if key in result:
            print(f"  Optional key present: {key}")
    print("\nVerification passed.")
    sys.exit(0)


def __dummy_image():
    """Return a small BGR image (numpy array) for testing."""
    import numpy as np
    h, w = 240, 320
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (40, 40, 60)  # Dark blue-gray
    cv2.putText(img, "Test frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


if __name__ == "__main__":
    main()
