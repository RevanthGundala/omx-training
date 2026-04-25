"""
eval_pi0.py — Run a Pi0 policy live on the OMX follower arm.

Supports two modes:
  --local    Load Pi0 on this machine (slow on CPU/MPS)
  --remote   Use a Modal GPU server (deploy serve_pi0_modal.py first)

The remote mode sends observations over HTTP and executes returned action
chunks locally, re-fetching when the chunk is exhausted.

Press Ctrl+C to stop.
"""

import argparse
import base64
import time
from collections import deque

import numpy as np
import rerun as rr
import requests

from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features

from config import FPS, JOINT_NAMES, PI0_MODEL_REPO_ID as HF_REPO_ID, TASK_NAME
from control_utils import ensure_camera_size, maintain_fps
from rerun_utils import init_rerun
from robot_utils import create_follower, safe_disconnect

# ──────────────────────────────────────────────
# Eval-specific configuration
# ──────────────────────────────────────────────
START_DELAY_S = 3
FROZEN_JOINTS: set[str] = set()


def _build_follower():
    return create_follower(camera=True)


# ──────────────────────────────────────────────
# Local inference
# ──────────────────────────────────────────────
def _load_local_policy():
    import torch
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.utils.utils import get_safe_torch_device

    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    device = get_safe_torch_device(dev, log=True)

    print(f"Loading Pi0 locally from {HF_REPO_ID}...")

    NUM_JOINTS = 6
    NUM_CHANNELS = 3
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(NUM_JOINTS),
            "std": torch.ones(NUM_JOINTS),
        },
        "observation.images.front": {
            "mean": torch.zeros(NUM_CHANNELS, 1, 1),
            "std": torch.ones(NUM_CHANNELS, 1, 1),
        },
        "action": {
            "mean": torch.zeros(NUM_JOINTS),
            "std": torch.ones(NUM_JOINTS),
        },
    }

    policy = PI0Policy.from_pretrained(HF_REPO_ID, dataset_stats=dataset_stats)
    policy.config.device = device.type
    policy.to(device)
    policy.eval()
    policy.reset()
    return policy, device


def _predict_local(observation_frame, policy, device):
    from lerobot.utils.control_utils import predict_action

    action_values = predict_action(
        observation_frame,
        policy,
        device,
        policy.config.use_amp,
        task=TASK_NAME,
        robot_type="omx_follower",
    )
    return action_values.cpu().numpy()


# ──────────────────────────────────────────────
# Remote inference (Modal)
# ──────────────────────────────────────────────
def _predict_remote(observation_frame, server_url, observation):
    state = np.asarray(observation_frame["observation.state"]).tolist()
    payload = {
        "state": state,
        "task": TASK_NAME,
        "robot_type": "omx_follower",
    }

    if "front" in observation:
        img = observation["front"]
        payload["image"] = base64.b64encode(img.tobytes()).decode("ascii")
        payload["image_shape"] = list(img.shape)

    resp = requests.post(server_url, json=payload, timeout=30)
    resp.raise_for_status()
    return np.array(resp.json()["actions"], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Pi0 eval on OMX follower")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true", help="Run Pi0 locally")
    mode.add_argument("--remote", type=str, metavar="URL",
                      help="Modal server URL (e.g. https://your-app--pi0server-predict.modal.run)")
    args = parser.parse_args()

    policy = device = server_url = None
    if args.local:
        policy, device = _load_local_policy()

    if args.remote:
        server_url = args.remote.rstrip("/")
        print(f"Using remote server: {server_url}")
        print("Warming up server (cold start can take ~60s)...")
        health_url = server_url.replace("-predict.", "-health.")
        health = requests.get(health_url, timeout=120)
        health.raise_for_status()
        print("Remote server is ready.")

    follower = _build_follower()
    dataset_features = {
        **hw_to_dataset_features(follower.action_features, "action", use_video=False),
        **hw_to_dataset_features(follower.observation_features, "observation", use_video=False),
    }

    print("Connecting follower arm...")
    follower.connect(calibrate=False)

    # ── Rerun setup ──
    init_rerun("omx_eval_pi0")

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

        while True:
            loop_start = time.perf_counter()

            observation = follower.get_observation()

            cam_key = "front"
            if cam_key in observation:
                ensure_camera_size(observation)

            # Get next action — from queue or fresh prediction
            if not action_queue:
                observation_frame = build_dataset_frame(
                    dataset_features, observation, prefix="observation",
                )

                for name in FROZEN_JOINTS:
                    if name not in frozen_joint_targets:
                        frozen_joint_targets[name] = observation_frame[
                            "observation.state"
                        ][joint_indices[name]].item()
                        print(f"\nFreezing {name} at {frozen_joint_targets[name]:.2f}")

                if args.local:
                    actions_np = _predict_local(observation_frame, policy, device)
                else:
                    actions_np = _predict_remote(observation_frame, server_url, observation)

                # Actions may be a single step or a chunk
                if actions_np.ndim == 1:
                    action_queue.append(actions_np)
                else:
                    for row in actions_np:
                        action_queue.append(row)

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
            rr.log("metrics/loop_hz", rr.Scalar(hz))

            if cam_key in observation:
                rr.log("camera/front", rr.Image(observation[cam_key]))

            for i, name in enumerate(JOINT_NAMES):
                if "observation.state" in observation_frame:
                    state_val = observation_frame["observation.state"][i].item()
                    rr.log(f"joints/{name}/state", rr.Scalar(state_val))
                rr.log(f"joints/{name}/policy_action", rr.Scalar(float(action_values[i])))
                rr.log(f"joints/{name}/sent_action", rr.Scalar(sent_action[f"{name}.pos"]))
                if name in frozen_joint_targets:
                    rr.log(f"joints/{name}/frozen_target", rr.Scalar(frozen_joint_targets[name]))

            step += 1
            queue_len = len(action_queue)
            action_preview = "  |  ".join(
                f"{name}: {value:7.2f}" for name, value in sent_action.items()
            )
            print(f"Step {step:05d} | {hz:5.1f} Hz | Q:{queue_len:2d} | {action_preview}", end="\r")

            maintain_fps(loop_start, FPS)

    except KeyboardInterrupt:
        print("\n\nStopping Pi0 eval...")
    finally:
        safe_disconnect(follower)
        print("Disconnected. Done!")


if __name__ == "__main__":
    main()
