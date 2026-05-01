"""
eval_pi0.py — Run a Pi0.5 policy live on the OMX follower arm via remote inference.

Uses a Modal GPU server (deploy serve_pi0_modal.py first).
Sends observations over HTTP and executes returned action chunks locally,
prefetching the next chunk in a background thread for smooth motion.

Press Ctrl+C to stop.
"""

import argparse
import base64
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

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
FROZEN_JOINTS: set[str] = set()


def _build_follower():
    return create_follower(camera=True)


# ──────────────────────────────────────────────
# Remote inference (Modal)
# ──────────────────────────────────────────────
def _predict_remote(observation_frame, server_url, observation):
    import cv2
    state = np.asarray(observation_frame["observation.state"]).tolist()
    payload = {
        "state": state,
        "task": TASK_NAME,
        "robot_type": "omx_follower",
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

    executor = ThreadPoolExecutor(max_workers=1)

    try:
        print(f"Starting Pi0 eval in {START_DELAY_S}s. Press Ctrl+C to stop.")
        for remaining in range(START_DELAY_S, 0, -1):
            print(f"  {remaining}...", end="\r")
            time.sleep(1)
        print(" " * 40, end="\r")

        run_start = time.perf_counter()
        step = 0
        joint_indices = {name: idx for idx, name in enumerate(JOINT_NAMES)}
        frozen_joint_targets: dict[str, float] = {}
        action_queue: deque[np.ndarray] = deque()
        prefetch_future = None
        observation_frame = None

        def _enqueue(actions_np):
            if actions_np.ndim == 1:
                action_queue.append(actions_np)
            else:
                for row in actions_np:
                    action_queue.append(row)

        while True:
            loop_start = time.perf_counter()

            observation = follower.get_observation()

            for cam_name in CAMERAS:
                if cam_name in observation:
                    ensure_camera_size(observation, key=cam_name)

            # If queue is empty, wait for the prefetch result
            if not action_queue:
                if prefetch_future is not None:
                    _enqueue(prefetch_future.result())
                    prefetch_future = None
                else:
                    # First step — no prefetch yet, fetch synchronously
                    observation_frame = build_dataset_frame(
                        dataset_features, observation, prefix="observation",
                    )
                    _enqueue(_predict_remote(observation_frame, server_url, observation))

                # Immediately kick off prefetch for the NEXT chunk
                # using the freshest observation available
                observation_frame = build_dataset_frame(
                    dataset_features, observation, prefix="observation",
                )
                for name in FROZEN_JOINTS:
                    if name not in frozen_joint_targets:
                        frozen_joint_targets[name] = observation_frame[
                            "observation.state"
                        ][joint_indices[name]].item()
                        print(f"\nFreezing {name} at {frozen_joint_targets[name]:.2f}")

                obs_snapshot = {k: v for k, v in observation.items()}
                prefetch_future = executor.submit(
                    _predict_remote, observation_frame, server_url, obs_snapshot,
                )

            action_values = action_queue.popleft()

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
            queue_len = len(action_queue)
            pf = "↻" if prefetch_future is not None else " "
            action_preview = "  |  ".join(
                f"{name}: {value:7.2f}" for name, value in sent_action.items()
            )
            print(f"Step {step:05d} | {hz:5.1f} Hz | Q:{queue_len:2d} {pf}| {action_preview}", end="\r")

            maintain_fps(loop_start, FPS)

    except KeyboardInterrupt:
        print("\n\nStopping Pi0 eval...")
    finally:
        executor.shutdown(wait=False)
        safe_disconnect(follower)
        print("Disconnected. Done!")


if __name__ == "__main__":
    main()
