"""
eval.py — Run a trained ACT checkpoint live on the OMX follower arm.

This script loads a saved ACT checkpoint, reads live observations from the
follower arm (joint state + camera), predicts the next action, and sends that
action back to the robot in a closed loop.

Observations, raw policy actions, sent actions, and camera frames are logged to
Rerun for review.

Press Ctrl+C to stop.
"""

import time
from pathlib import Path

import rerun as rr
import torch

from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device

from config import FPS, JOINT_NAMES, TASK_NAME
from control_utils import ensure_camera_size, maintain_fps
from rerun_utils import init_rerun
from robot_utils import create_follower, safe_disconnect

# ──────────────────────────────────────────────
# Eval-specific configuration
# ──────────────────────────────────────────────
# Camera captures at higher res, then resizes to model input
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MODEL_WIDTH = 640
MODEL_HEIGHT = 480

CHECKPOINT_PATH = Path("outputs/checkpoints/last/pretrained_model")
DEVICE = "auto"  # "auto", "mps", "cuda", or "cpu"
START_DELAY_S = 3

# Hold selected joints at their initial observed positions instead of following
# the policy. Useful for isolating overloaded joints during live eval.
FROZEN_JOINTS = {"shoulder_lift"}


def _resolve_device(requested_device: str) -> torch.device:
    requested_device = requested_device.lower()

    if requested_device == "auto":
        if torch.cuda.is_available():
            requested_device = "cuda"
        elif torch.backends.mps.is_available():
            requested_device = "mps"
        else:
            requested_device = "cpu"

    return get_safe_torch_device(requested_device, log=True)


def _build_follower():
    return create_follower(camera=True, camera_width=CAMERA_WIDTH, camera_height=CAMERA_HEIGHT)


def _load_policy(device: torch.device) -> ACTPolicy:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Set CHECKPOINT_PATH to one of the downloaded pretrained_model directories."
        )

    policy = ACTPolicy.from_pretrained(CHECKPOINT_PATH)
    policy.config.device = device.type
    policy.to(device)
    policy.eval()
    policy.reset()
    return policy


def _build_observation_features(follower: OmxFollower) -> dict[str, dict]:
    action_features = hw_to_dataset_features(follower.action_features, "action", use_video=False)
    obs_features = hw_to_dataset_features(follower.observation_features, "observation", use_video=False)
    return {**action_features, **obs_features}


def _validate_policy_inputs(
    policy: ACTPolicy,
    observation_frame: dict,
    follower: OmxFollower,
) -> None:
    expected_inputs = set(policy.config.input_features.keys())
    missing_inputs = sorted(expected_inputs - set(observation_frame.keys()))
    if missing_inputs:
        raise ValueError(
            "Live observations do not match the checkpoint inputs. "
            f"Missing inputs: {missing_inputs}"
        )

    if any(name.startswith("observation.images.") for name in expected_inputs) and not USE_CAMERA:
        raise ValueError(
            "This checkpoint expects camera inputs, but USE_CAMERA is False."
        )

    if len(follower.action_features) != policy.config.output_features["action"].shape[0]:
        raise ValueError(
            "Checkpoint action dimension does not match follower action features. "
            f"Checkpoint expects {policy.config.output_features['action'].shape[0]} values, "
            f"but the follower exposes {len(follower.action_features)} action features."
        )


def main():
    device = _resolve_device(DEVICE)
    follower = _build_follower()
    policy = _load_policy(device)
    dataset_features = _build_observation_features(follower)

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    print(f"Running policy on device: {device}")
    print("Connecting follower arm...")
    follower.connect(calibrate=False)

    # ── Rerun setup ──
    init_rerun("omx_eval")

    try:
        print(f"Starting live eval in {START_DELAY_S}s. Press Ctrl+C to stop.")
        for remaining in range(START_DELAY_S, 0, -1):
            print(f"  {remaining}...", end="\r")
            time.sleep(1)
        print(" " * 40, end="\r")

        run_start = time.perf_counter()
        step = 0
        validated = False
        joint_indices = {name: idx for idx, name in enumerate(JOINT_NAMES)}
        frozen_joint_targets: dict[str, float] = {}

        while True:
            loop_start = time.perf_counter()

            observation = follower.get_observation()

            # Resize camera frame if needed to match model input
            ensure_camera_size(observation, MODEL_WIDTH, MODEL_HEIGHT)

            observation_frame = build_dataset_frame(
                dataset_features,
                observation,
                prefix="observation",
            )

            if not validated:
                _validate_policy_inputs(policy, observation_frame, follower)
                validated = True

            for name in FROZEN_JOINTS:
                if name not in frozen_joint_targets:
                    frozen_joint_targets[name] = observation_frame["observation.state"][
                        joint_indices[name]
                    ].item()
                    print(f"\nFreezing {name} at {frozen_joint_targets[name]:.2f}")

            action_values = predict_action(
                observation_frame,
                policy,
                device,
                policy.config.use_amp,
                task=TASK_NAME,
                robot_type=follower.robot_type,
            )
            action = {key: action_values[i].item() for i, key in enumerate(follower.action_features)}

            for name, target in frozen_joint_targets.items():
                action[f"{name}.pos"] = target

            sent_action = follower.send_action(action)

            loop_dt = time.perf_counter() - loop_start
            hz = 1.0 / loop_dt if loop_dt > 0 else float("inf")

            # ── Rerun logging ──
            rr.set_time_sequence("step", step)
            rr.set_time_seconds("time", time.perf_counter() - run_start)
            rr.log("metrics/loop_hz", rr.Scalar(hz))

            if "front" in observation:
                rr.log("camera/front", rr.Image(observation["front"]))

            for i, name in enumerate(JOINT_NAMES):
                if "observation.state" in observation_frame:
                    state_val = observation_frame["observation.state"][i].item()
                    rr.log(f"joints/{name}/state", rr.Scalar(state_val))

                policy_action_val = action_values[i].item()
                rr.log(f"joints/{name}/policy_action", rr.Scalar(policy_action_val))
                rr.log(f"joints/{name}/sent_action", rr.Scalar(sent_action[f"{name}.pos"]))

                if name in frozen_joint_targets:
                    rr.log(f"joints/{name}/frozen_target", rr.Scalar(frozen_joint_targets[name]))

            step += 1
            action_preview = "  |  ".join(f"{name}: {value:7.2f}" for name, value in sent_action.items())
            print(f"Step {step:05d} | {hz:5.1f} Hz | {action_preview}", end="\r")

            maintain_fps(loop_start, FPS)

    except KeyboardInterrupt:
        print("\n\nStopping live eval...")
    finally:
        safe_disconnect(follower)
        print("Disconnected. Done!")


if __name__ == "__main__":
    main()
