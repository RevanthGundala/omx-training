"""
eval_pi0.py — Run a Pi0 or PI0.5 policy live on the OMX follower arm.

Supports two modes:
  --local    Load policy on this machine (slow on CPU/MPS)
  --remote   Use a Modal GPU server (deploy serve_pi0_modal.py first)

Use --pi05 to load a PI0.5 checkpoint instead of PI0.
Use --checkpoint to load a finetuned checkpoint from a local path.

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

from config import CAMERAS, FPS, JOINT_NAMES, PI0_MODEL_REPO_ID, PI05_MODEL_REPO_ID, TASK_NAME
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
def _load_local_policy(checkpoint_path=None, use_pi05=False):
    import torch
    try:
        from lerobot.utils.device_utils import get_safe_torch_device
    except ImportError:
        from lerobot.utils.utils import get_safe_torch_device

    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    device = get_safe_torch_device(dev, log=True)

    if checkpoint_path:
        # Load finetuned checkpoint — auto-detect pi0 vs pi0.5 from config
        from safetensors.torch import load_file
        import json
        from pathlib import Path

        ckpt_dir = Path(checkpoint_path)
        config_path = ckpt_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json in {ckpt_dir}")

        with open(config_path) as f:
            config_data = json.load(f)

        model_type = config_data.get("model_type", "")
        if model_type == "pi05" or use_pi05:
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            print(f"Loading PI0.5 checkpoint from {checkpoint_path}...")
            policy = PI05Policy.from_pretrained(checkpoint_path)
        else:
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            print(f"Loading PI0 checkpoint from {checkpoint_path}...")
            policy = PI0Policy.from_pretrained(checkpoint_path)
    elif use_pi05:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        repo_id = PI05_MODEL_REPO_ID
        print(f"Loading PI0.5 base model from {repo_id}...")
        policy = PI05Policy.from_pretrained(repo_id)
    else:
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        repo_id = PI0_MODEL_REPO_ID
        print(f"Loading PI0 base model from {repo_id}...")

        NUM_JOINTS = 6
        NUM_CHANNELS = 3
        dataset_stats = {
            "observation.state": {
                "mean": torch.zeros(NUM_JOINTS),
                "std": torch.ones(NUM_JOINTS),
            },
        }
        for cam_name in CAMERAS:
            dataset_stats[f"observation.images.{cam_name}"] = {
                "mean": torch.zeros(NUM_CHANNELS, 1, 1),
                "std": torch.ones(NUM_CHANNELS, 1, 1),
            }
        dataset_stats["action"] = {
            "mean": torch.zeros(NUM_JOINTS),
            "std": torch.ones(NUM_JOINTS),
        }
        policy = PI0Policy.from_pretrained(repo_id, dataset_stats=dataset_stats)

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
    import cv2
    state = np.asarray(observation_frame["observation.state"]).tolist()
    payload = {
        "state": state,
        "task": TASK_NAME,
        "robot_type": "omx_follower",
    }

    # Send camera images as JPEG (much smaller than raw bytes)
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
    parser = argparse.ArgumentParser(description="Pi0/Pi0.5 eval on OMX follower")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true", help="Run policy locally")
    mode.add_argument("--remote", type=str, nargs="?", const="auto", metavar="URL",
                      help="Use Modal GPU server. Omit URL to auto-deploy.")
    parser.add_argument("--pi05", action="store_true", help="Use PI0.5 instead of PI0")
    parser.add_argument("--checkpoint", type=str, metavar="PATH",
                        help="Local checkpoint path (overrides base model)")
    args = parser.parse_args()

    policy = device = server_url = None
    if args.local:
        policy, device = _load_local_policy(
            checkpoint_path=args.checkpoint,
            use_pi05=args.pi05,
        )

    if args.remote:
        if args.remote == "auto":
            import subprocess
            print("Auto-deploying Modal inference server...")
            result = subprocess.run(
                ["modal", "deploy", "serve_pi0_modal.py"],
                capture_output=True, text=True, timeout=600,
            )
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            if result.returncode != 0:
                print(f"Deploy failed:\n{result.stderr[-500:]}")
                return
            # Extract predict URL from output
            for line in result.stdout.splitlines():
                if "predict =>" in line:
                    server_url = line.split("=>")[-1].strip()
                    break
            if not server_url:
                print("Could not find predict URL in deploy output.")
                return
        else:
            server_url = args.remote.rstrip("/")

        print(f"Using remote server: {server_url}")
        print("Warming up server (cold start can take ~60s)...")
        health_url = server_url.replace("-predict.", "-health.")
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
    model_name = "pi05" if args.pi05 else "pi0"
    init_rerun(f"omx_eval_{model_name}", save_rrd=True)

    # Reset server state (clears stale RTC context)
    if server_url:
        reset_url = server_url.replace("-predict.", "-reset.")
        try:
            requests.post(reset_url, json={}, timeout=10)
        except Exception:
            pass

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

            for cam_name in CAMERAS:
                if cam_name in observation:
                    ensure_camera_size(observation, key=cam_name)

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
            rr.log("metrics/loop_hz", rr.Scalars(hz))

            for cam_name in CAMERAS:
                if cam_name in observation:
                    rr.log(f"camera/{cam_name}", rr.Image(observation[cam_name]))

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
