"""
eval_pi0.py — Run a Pi0.5 policy live on the OMX follower arm via remote inference.

Uses a Modal GPU server (deploy serve_pi0_modal.py first).
Sends observations over HTTP and executes returned action chunks locally.
The control loop never stalls: a background thread continuously fetches
new chunks while the main loop pops one action per tick at FPS.

Press Ctrl+C to stop.
"""

import argparse
import base64
import time
import threading
from collections import deque

import numpy as np
import rerun as rr
import requests

from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features

from utils.config import CAMERAS, FPS, JOINT_NAMES, TASK_NAME
from utils.control_utils import ensure_camera_size, maintain_fps
from utils.rerun_utils import init_rerun
from utils.robot_utils import create_follower, safe_disconnect

# ──────────────────────────────────────────────
# Eval-specific configuration
# ──────────────────────────────────────────────
START_DELAY_S = 3


def _build_follower():
    return create_follower(camera=True)


# ──────────────────────────────────────────────
# Thread-safe action queue for RTC
# ──────────────────────────────────────────────
class RTCActionQueue:
    """Thread-safe action queue that supports RTC-style replacement.

    The main loop pops one action per tick via get().  When the background
    inference thread delivers a new chunk it calls replace(), which swaps
    the remaining queue contents with the fresh actions (skipping the first
    ``skip`` entries that correspond to steps already executed during the
    inference round-trip).

    If the queue is empty, get() returns a clone of the last action sent
    so the robot never stalls.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._queue: deque[np.ndarray] = deque()
        self._last_action: np.ndarray | None = None
        self._steps_since_replace = 0

    def replace(self, actions: np.ndarray, skip: int = 0):
        """Replace the queue with *actions*, dropping the first *skip* rows."""
        with self._lock:
            self._queue.clear()
            clamped_skip = max(0, min(skip, len(actions)))
            for row in actions[clamped_skip:]:
                self._queue.append(row)
            self._steps_since_replace = 0

    def replace_atomic(self, actions: np.ndarray):
        """Atomically read steps_since_replace and replace the queue.

        Avoids a race where the main loop's get() increments the counter
        between a separate read and a subsequent replace() call.
        """
        with self._lock:
            skip = max(0, min(self._steps_since_replace, len(actions)))
            self._queue.clear()
            for row in actions[skip:]:
                self._queue.append(row)
            self._steps_since_replace = 0

    def get(self) -> np.ndarray | None:
        """Pop the next action.  Returns last-sent action if empty."""
        with self._lock:
            if self._queue:
                action = self._queue.popleft()
                self._last_action = action
            else:
                action = self._last_action  # hold position
            self._steps_since_replace += 1
            return action

    def qsize(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def steps_since_replace(self) -> int:
        with self._lock:
            return self._steps_since_replace

    @property
    def empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0


# ──────────────────────────────────────────────
# Remote inference (Modal)
# ──────────────────────────────────────────────
def _predict_remote(observation_frame, server_url, observation, steps_executed: int):
    import cv2
    state = np.asarray(observation_frame["observation.state"]).tolist()
    payload = {
        "state": state,
        "task": TASK_NAME,
        "robot_type": "omx_follower",
        "steps_executed": steps_executed,
    }

    for cam_name in CAMERAS:
        if cam_name in observation:
            img = observation[cam_name]
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            payload[f"image_{cam_name}"] = base64.b64encode(buf.tobytes()).decode("ascii")

    resp = requests.post(server_url, json=payload, timeout=30)
    resp.raise_for_status()
    return np.array(resp.json()["actions"], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Pi0.5 eval on OMX follower (remote inference)")
    parser.add_argument("--remote", type=str, required=True, metavar="URL",
                        help="Modal GPU server predict URL")
    args = parser.parse_args()

    server_url = args.remote.rstrip("/")

    print(f"Using remote server: {server_url}")
    print("Warming up server (cold start can take ~60s)...")
    health_url = server_url.replace("-predict", "-health", 1)
    try:
        health = requests.get(health_url, timeout=120)
        health.raise_for_status()
        print(f"Remote server is ready: {health.json()}")
    except Exception as e:
        print(f"Health check failed ({e}), proceeding anyway...")

    follower = _build_follower()
    dataset_features = {
        **hw_to_dataset_features(follower.action_features, "action", use_video=False),
        **hw_to_dataset_features(follower.observation_features, "observation", use_video=False),
    }

    print("Connecting follower arm...")
    for attempt in range(1, 4):
        try:
            follower.connect(calibrate=False)
            break
        except (TimeoutError, RuntimeError) as e:
            print(f"  Camera connect attempt {attempt}/3 failed: {e}")
            if attempt == 3:
                raise
            follower.disconnect() if hasattr(follower, 'disconnect') else None
            follower = _build_follower()
            dataset_features = {
                **hw_to_dataset_features(follower.action_features, "action", use_video=False),
                **hw_to_dataset_features(follower.observation_features, "observation", use_video=False),
            }
            time.sleep(2)

    # ── Rerun setup ──
    init_rerun("omx_eval_pi05", save_rrd=True)

    # Reset server state (clears stale RTC context)
    reset_url = server_url.replace("-predict", "-reset", 1)
    try:
        requests.post(reset_url, json={}, timeout=10)
    except Exception:
        pass

    action_queue = RTCActionQueue()

    # ── Shared state for background inference thread ──
    obs_lock = threading.Lock()
    latest_observation = None        # freshest observation snapshot
    latest_observation_frame = None  # freshest dataset frame
    inference_running = threading.Event()
    stop_event = threading.Event()

    def _inference_loop():
        """Background thread: continuously fetch chunks from the server."""
        # Wait for the first observation to be available
        while not stop_event.is_set():
            with obs_lock:
                obs = latest_observation
                obs_frame = latest_observation_frame
            if obs is not None and obs_frame is not None:
                break
            time.sleep(0.01)

        while not stop_event.is_set():
            # Snapshot the freshest observation and steps consumed
            with obs_lock:
                obs_snapshot = {k: v for k, v in latest_observation.items()}
                frame_snapshot = latest_observation_frame
            steps_executed = action_queue.steps_since_replace

            inference_running.set()
            try:
                actions = _predict_remote(
                    frame_snapshot, server_url, obs_snapshot, steps_executed,
                )
                # Atomically read consumed steps and swap the queue
                action_queue.replace_atomic(actions)
            except Exception as e:
                print(f"\n⚠ Inference error: {e}")
            finally:
                inference_running.clear()

    inference_thread = threading.Thread(target=_inference_loop, daemon=True)
    inference_thread.start()

    try:
        print(f"Starting Pi0 eval in {START_DELAY_S}s. Press Ctrl+C to stop.")
        for remaining in range(START_DELAY_S, 0, -1):
            print(f"  {remaining}...", end="\r")
            time.sleep(1)
        print(" " * 40, end="\r")

        run_start = time.perf_counter()
        step = 0
        frozen_joint_targets: dict[str, float] = {}
        observation_frame = None

        while True:
            loop_start = time.perf_counter()

            observation = follower.get_observation()

            for cam_name in CAMERAS:
                if cam_name in observation:
                    ensure_camera_size(observation, key=cam_name)

            # Update shared observation for the inference thread
            observation_frame = build_dataset_frame(
                dataset_features, observation, prefix="observation",
            )
            with obs_lock:
                latest_observation = observation
                latest_observation_frame = observation_frame

            # Pop one action (never blocks; repeats last if empty)
            action_values = action_queue.get()
            if action_values is None:
                # Very first tick — no actions yet; skip actuation
                maintain_fps(loop_start, FPS)
                continue

            action = {
                key: float(action_values[i])
                for i, key in enumerate(follower.action_features)
            }
            for name, target in frozen_joint_targets.items():
                action[f"{name}.pos"] = target

            sent_action = follower.send_action(action)

            loop_dt = time.perf_counter() - loop_start
            hz = 1.0 / loop_dt if loop_dt > 0 else float("inf")

            # ── Rerun logging ──
            rr.set_time_sequence("step", step)
            rr.set_time_seconds("time", time.perf_counter() - run_start)
            rr.log("metrics/loop_hz", rr.Scalars(hz))

            for cam_name in CAMERAS:
                if cam_name in observation:
                    rr.log(f"camera/{cam_name}", rr.Image(observation[cam_name]))

            if observation_frame is not None:
                for i, name in enumerate(JOINT_NAMES):
                    if "observation.state" in observation_frame:
                        state_val = observation_frame["observation.state"][i].item()
                        rr.log(f"joints/{name}/state", rr.Scalars(state_val))
                    rr.log(f"joints/{name}/policy_action", rr.Scalars(float(action_values[i])))
                    rr.log(f"joints/{name}/sent_action", rr.Scalars(sent_action[f"{name}.pos"]))
                    if name in frozen_joint_targets:
                        rr.log(f"joints/{name}/frozen_target", rr.Scalars(frozen_joint_targets[name]))

            step += 1
            queue_len = action_queue.qsize()
            inf_sym = "↻" if inference_running.is_set() else " "
            action_preview = "  |  ".join(
                f"{name}: {value:7.2f}" for name, value in sent_action.items()
            )
            print(f"Step {step:05d} | {hz:5.1f} Hz | Q:{queue_len:2d} {inf_sym}| {action_preview}", end="\r")

            maintain_fps(loop_start, FPS)

    except KeyboardInterrupt:
        print("\n\nStopping Pi0 eval...")
    finally:
        stop_event.set()
        inference_thread.join(timeout=5)
        safe_disconnect(follower)
        print("Disconnected. Done!")


if __name__ == "__main__":
    main()
