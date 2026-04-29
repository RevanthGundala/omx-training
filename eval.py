"""
eval.py — Run a trained ACT checkpoint live on the OMX follower arm.

Automatically detects whether the checkpoint needs cameras (and which ones)
from the saved config. Supports state-only, single-camera, and multi-camera
checkpoints. State-only checkpoints can still open the camera for monitoring
and Rerun logging, but camera frames only affect actions if the checkpoint was
trained with image inputs.

Observations, policy actions, and sent actions are logged to Rerun for review.

Press Ctrl+C to stop.
"""

import time
from pathlib import Path

import numpy as np
import rerun as rr
import torch

from lerobot.policies.act.modeling_act import ACTPolicy

try:
    from lerobot.utils.device_utils import get_safe_torch_device
except ImportError:
    from lerobot.utils.utils import get_safe_torch_device

from config import FPS, JOINT_NAMES, TASK_NAME, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERAS
from control_utils import ensure_camera_size, maintain_fps
from rerun_utils import init_rerun
from robot_utils import create_follower, safe_disconnect

# ──────────────────────────────────────────────
# Eval-specific configuration
# ──────────────────────────────────────────────
CHECKPOINT_PATH = Path("outputs/act-run/pretrained_model")
DEVICE = "auto"  # "auto", "mps", "cuda", or "cpu"
START_DELAY_S = 3

# Episode termination. ACT has no learned "done" signal, so eval ends on a
# fixed step or wall-clock budget (whichever comes first), or on Ctrl+C.
MAX_STEPS = 1500          # at FPS=30 this is ~50s of motion
MAX_DURATION_S = 60.0     # hard wall-clock cap
RETURN_HOME_S = 3.0       # seconds to smoothly return to home after eval

# Home position (radians) — upright, gripper open, safe resting pose.
# Adjust these values to match your robot's safe rest position.
HOME_POSITION = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 0.0,
}

# Camera captures at higher res, then resizes to model input
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
MODEL_WIDTH = CAMERA_WIDTH    # resize to match training resolution
MODEL_HEIGHT = CAMERA_HEIGHT
ENABLE_CAMERA_MONITORING = True

# Hold selected joints at their initial observed positions instead of following
# the policy. Useful for isolating overloaded joints during live eval.
FROZEN_JOINTS: set[str] = set()


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


def _detect_camera_features(policy: ACTPolicy) -> list[str]:
    """Return camera names expected by the checkpoint (e.g. ['front', 'wrist'])."""
    camera_names = []
    for key in policy.config.input_features:
        if key.startswith("observation.images."):
            cam_name = key.split("observation.images.")[-1]
            camera_names.append(cam_name)
    return camera_names


def _resolve_capture_camera_names(policy_camera_names: list[str]) -> list[str]:
    """Return camera names to open at runtime."""
    if policy_camera_names:
        return policy_camera_names
    if ENABLE_CAMERA_MONITORING:
        return list(CAMERAS.keys())
    return []


def _detect_uses_env_state(policy: ACTPolicy) -> bool:
    """Check if the checkpoint was trained with observation.state as ENV type."""
    for key, ft in policy.config.input_features.items():
        if key == "observation.state" and ft.type.value == "ENV":
            return True
    return False


def _build_observation(
    follower,
    device: torch.device,
    policy_camera_names: list[str],
    capture_camera_names: list[str],
    uses_env_state: bool,
) -> tuple[dict[str, torch.Tensor], np.ndarray, dict[str, np.ndarray]]:
    """Read follower state (+ cameras) and build the observation dict for the policy.

    Returns tensors matching training format: state as float32, images as
    float32 [0,1] in (C, H, W) layout. Tensors are on CPU — moved to device
    in the inference loop.
    """
    obs = follower.get_observation()

    # Joint state — keep on CPU; preprocessor moves to device
    state = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
    state_tensor = torch.from_numpy(state).unsqueeze(0)

    obs_dict: dict[str, torch.Tensor] = {"observation.state": state_tensor}

    # If checkpoint was trained with state as ENV, add the environment_state alias
    if uses_env_state:
        obs_dict["observation.environment_state"] = state_tensor

    # Camera images
    camera_frames: dict[str, np.ndarray] = {}
    for cam_name in capture_camera_names:
        if cam_name not in obs:
            raise RuntimeError(
                f"Eval requested camera '{cam_name}' but follower doesn't provide it. "
                f"Available keys: {list(obs.keys())}"
            )
        frame = obs[cam_name]
        camera_frames[cam_name] = frame

        if cam_name in policy_camera_names:
            ensure_camera_size(obs, MODEL_WIDTH, MODEL_HEIGHT, key=cam_name)
            frame = obs[cam_name]
            camera_frames[cam_name] = frame
            # (H, W, C) -> (C, H, W), float32 [0,1], batch dim — preprocessor normalizes further
            img_tensor = torch.from_numpy(frame).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            obs_dict[f"observation.images.{cam_name}"] = img_tensor

    return obs_dict, state, camera_frames


def _return_home(follower, duration_s: float = RETURN_HOME_S):
    """Smoothly interpolate from current position to HOME_POSITION."""
    print(f"\nReturning to home position over {duration_s}s...")
    current = follower.get_observation()
    steps = max(1, int(duration_s * FPS))
    for step in range(1, steps + 1):
        loop_start = time.perf_counter()
        alpha = step / steps
        target = {
            k: current[k] + alpha * (HOME_POSITION[k] - current[k])
            for k in HOME_POSITION
        }
        follower.send_action(target)
        maintain_fps(loop_start, FPS)
    print("Home position reached.")


def main():
    device = _resolve_device(DEVICE)
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    policy = _load_policy(device)
    print(f"Running policy on device: {device}")

    # Detect what the checkpoint expects
    policy_camera_names = _detect_camera_features(policy)
    capture_camera_names = _resolve_capture_camera_names(policy_camera_names)
    uses_env_state = _detect_uses_env_state(policy)
    use_camera = len(capture_camera_names) > 0

    action_dim = policy.config.output_features["action"].shape[0]
    assert action_dim == len(JOINT_NAMES), (
        f"Checkpoint expects {action_dim} action dims but robot has {len(JOINT_NAMES)} joints"
    )

    if policy_camera_names:
        print(f"Checkpoint consumes cameras: {policy_camera_names}")
        print("Connecting follower arm (with camera)...")
        follower = create_follower(
            camera=True, camera_width=CAPTURE_WIDTH, camera_height=CAPTURE_HEIGHT,
        )
    elif use_camera:
        print(
            "Checkpoint is state-only; capturing camera for monitoring only: "
            f"{capture_camera_names}"
        )
        print("Connecting follower arm (with camera)...")
        follower = create_follower(
            camera=True, camera_width=CAPTURE_WIDTH, camera_height=CAPTURE_HEIGHT,
        )
    else:
        print("Checkpoint is state-only (camera monitoring disabled)")
        print("Connecting follower arm (no camera)...")
        follower = create_follower(camera=False)

    follower.connect(calibrate=False)
    init_rerun("omx_eval", has_camera=use_camera, camera_names=capture_camera_names)

    try:
        print(f"Starting live eval in {START_DELAY_S}s. Press Ctrl+C to stop.")
        for remaining in range(START_DELAY_S, 0, -1):
            print(f"  {remaining}...", end="\r")
            time.sleep(1)
        print(" " * 40, end="\r")

        step = 0
        joint_indices = {name: idx for idx, name in enumerate(JOINT_NAMES)}
        frozen_joint_targets: dict[str, float] = {}
        episode_start = time.perf_counter()

        while True:
            elapsed = time.perf_counter() - episode_start
            if step >= MAX_STEPS:
                print(f"\nReached MAX_STEPS={MAX_STEPS}. Ending eval.")
                break
            if elapsed >= MAX_DURATION_S:
                print(f"\nReached MAX_DURATION_S={MAX_DURATION_S:.1f}s. Ending eval.")
                break

            loop_start = time.perf_counter()

            obs_dict, state_np, camera_frames = _build_observation(
                follower,
                device,
                policy_camera_names,
                capture_camera_names,
                uses_env_state,
            )

            # Freeze joints on first step
            for name in FROZEN_JOINTS:
                if name not in frozen_joint_targets:
                    frozen_joint_targets[name] = state_np[joint_indices[name]].item()
                    print(f"\nFreezing {name} at {frozen_joint_targets[name]:.2f}")

            with torch.inference_mode():
                # Move tensors to device (matching what train.py does)
                for key in obs_dict:
                    if isinstance(obs_dict[key], torch.Tensor):
                        obs_dict[key] = obs_dict[key].to(device)
                action_values = policy.select_action(obs_dict)
                action_values = action_values.squeeze(0).cpu()

            action = {
                f"{name}.pos": action_values[i].item()
                for i, name in enumerate(JOINT_NAMES)
            }

            for name, target in frozen_joint_targets.items():
                action[f"{name}.pos"] = target

            sent_action = follower.send_action(action)

            loop_dt = time.perf_counter() - loop_start
            hz = 1.0 / loop_dt if loop_dt > 0 else float("inf")

            # ── Rerun logging ──
            rr.set_time("step", sequence=step)
            for i, name in enumerate(JOINT_NAMES):
                rr.log(f"joints/{name}/state", rr.Scalars([state_np[i].item()]))
                rr.log(f"joints/{name}/policy_action", rr.Scalars([action_values[i].item()]))
                rr.log(f"joints/{name}/sent_action", rr.Scalars([sent_action[f"{name}.pos"]]))
            rr.log("metrics/loop_hz", rr.Scalars([hz]))

            for cam_name, frame in camera_frames.items():
                rr.log(f"camera/{cam_name}", rr.Image(frame))

            step += 1
            action_preview = "  |  ".join(
                f"{name}: {value:7.2f}" for name, value in sent_action.items()
            )
            print(f"Step {step:05d} | {hz:5.1f} Hz | {action_preview}", end="\r")

            maintain_fps(loop_start, FPS)

    except KeyboardInterrupt:
        print("\n\nStopping live eval...")
    finally:
        try:
            _return_home(follower)
        except Exception as e:
            print(f"Could not return home: {e}")
        safe_disconnect(follower)
        print("Disconnected. Done!")


if __name__ == "__main__":
    main()
