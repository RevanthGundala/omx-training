"""Hardware preflight + request payload assembly."""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np

from utils.config import CAMERAS, TASK_NAME
from utils.dynamixel_errors import read_hw_errors
from utils.robot_utils import create_follower


def collect_hw_preflight(robot, *, label: str, feedback_motors: list[str] | None = None) -> dict:
    bus = robot.bus
    hw_errors = read_hw_errors(bus)
    present = {motor: float(value) for motor, value in bus.sync_read("Present_Position").items()}
    result = {
        "label": label,
        "motors": list(bus.motors),
        "feedback_motors": feedback_motors or [],
        "present_position": present,
        "hardware_errors": hw_errors,
        "ok": not any(entry["value"] for entry in hw_errors.values()),
    }
    if not result["ok"]:
        faulted = {m: e for m, e in hw_errors.items() if e["value"]}
        raise RuntimeError(f"{label} motor hardware errors: {faulted}")
    return result


def build_follower():
    return create_follower(camera=True)


def build_payload(
    observation_frame,
    observation,
    inference_delay: int,
    prev_steps_consumed: int,
) -> bytes:
    state = np.asarray(observation_frame["observation.state"]).tolist()
    payload = {
        "op": "predict",
        "state": state,
        "task": TASK_NAME,
        "robot_type": "omx_follower",
        # Keep steps_executed for older servers; newer servers distinguish
        # chunk alignment from future execution delay.
        "steps_executed": inference_delay,
        "inference_delay": inference_delay,
        "prev_steps_consumed": prev_steps_consumed,
    }
    for cam_name in CAMERAS:
        if cam_name in observation:
            img = observation[cam_name]
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            payload[f"image_{cam_name}"] = base64.b64encode(buf.tobytes()).decode("ascii")
    return json.dumps(payload).encode("utf-8")
