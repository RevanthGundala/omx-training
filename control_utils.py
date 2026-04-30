"""Control loop and misc helpers for OMX scripts."""

import os
import time
from pathlib import Path

import cv2

from config import CAMERA_HEIGHT, CAMERA_WIDTH


def maintain_fps(loop_start: float, fps: int) -> None:
    """Sleep to maintain the target FPS."""
    elapsed = time.perf_counter() - loop_start
    sleep_time = (1.0 / fps) - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)


def ensure_camera_size(
    observation: dict,
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
    key: str = "wrist",
) -> dict:
    """Resize the camera frame in an observation dict if needed."""
    if key not in observation:
        return observation
    img = observation[key]
    h, w = img.shape[:2]
    if (w, h) != (width, height):
        observation[key] = cv2.resize(img, (width, height))
    return observation


def get_hf_token() -> str:
    """Resolve a HuggingFace token from env or local cache."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return ""
