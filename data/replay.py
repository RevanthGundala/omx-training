"""
replay.py — Replay a recorded OMX episode on the follower arm.

This is the right sanity check before blaming the policy: it replays a known
recorded action sequence from the dataset, compares the live robot state to the
recorded state, and logs everything to Rerun.

If replay works but eval does not, the remaining problem is usually camera
placement, initial-state mismatch, or the policy itself — not low-level motion
execution.
"""

import time

import rerun as rr
import rerun.blueprint as rrb

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from utils.config import TRAIN_DATASET_REPO_ID as DATASET_REPO_ID
from utils.rerun_utils import build_joint_blueprint
from utils.robot_utils import create_follower, safe_disconnect

# ──────────────────────────────────────────────
# Replay-specific configuration
# ──────────────────────────────────────────────
EPISODE_INDEX = 47
FPS_OVERRIDE = None
START_DELAY_S = 3
MOVE_TO_START = True
# Maximum slew rate (pct per second) during the ramp to the recorded start.
# Total ramp duration scales as max_delta_pct / MOVE_TO_START_MAX_SLEW_PCT_PER_S,
# clamped to [MOVE_TO_START_MIN_S, MOVE_TO_START_MAX_S].
MOVE_TO_START_MAX_SLEW_PCT_PER_S = 8.0
MOVE_TO_START_MIN_S = 2.0
MOVE_TO_START_MAX_S = 12.0
TOLERANCE_S = 1e4


def _vector_to_dict(values, names: list[str]) -> dict[str, float]:
    return {name: float(values[i]) for i, name in enumerate(names)}


def _base_joint_name(name: str) -> str:
    return name.removesuffix(".pos")


def _init_rerun(joint_names: list[str], image_keys: list[str]) -> None:
    joint_views = [
        rrb.TimeSeriesView(name=joint_name, contents=[f"joints/{joint_name}/**"])
        for joint_name in joint_names
    ]
    image_views = [
        rrb.Spatial2DView(name=key.split(".")[-1], contents=[f"cameras/{key}"])
        for key in image_keys
    ]
    blueprint = rrb.Horizontal(
        rrb.Vertical(
            rrb.TimeSeriesView(name="Tracking error", contents=["metrics/**"]),
            *image_views,
        ),
        rrb.Vertical(*joint_views),
        column_shares=[1, 3],
    )
    rr.init("omx_replay", spawn=True)
    rr.send_blueprint(blueprint)


def _load_episode_data():
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        episodes=[EPISODE_INDEX],
        tolerance_s=TOLERANCE_S,
        video_backend="pyav",
    )
    action_rows = dataset.hf_dataset.select_columns("action")
    reference_state_rows = dataset.hf_dataset.select_columns("observation.state")
    action_names = dataset.features["action"]["names"]
    state_names = dataset.features["observation.state"].get("names", action_names)
    image_keys = [k for k, v in dataset.features.items() if v.get("dtype") == "video"]
    fps = dataset.fps if FPS_OVERRIDE is None else FPS_OVERRIDE
    return dataset, action_rows, reference_state_rows, action_names, state_names, image_keys, fps


def _move_to_start(
    follower,
    start_state: dict[str, float],
    fps: int,
) -> None:
    current_state = follower.get_observation()
    # Filter to only joint pcts (ignore camera/other obs keys).
    current_joints = {k: float(v) for k, v in current_state.items() if k in start_state}

    max_delta = max(abs(start_state[k] - current_joints[k]) for k in start_state)
    duration_s = max(
        MOVE_TO_START_MIN_S,
        min(MOVE_TO_START_MAX_S, max_delta / MOVE_TO_START_MAX_SLEW_PCT_PER_S),
    )
    steps = max(1, int(duration_s * fps))

    print(
        f"Moving to recorded start state: max joint delta {max_delta:.1f}%, "
        f"ramp over {duration_s:.1f}s ({steps} steps)..."
    )
    print("  per-joint deltas:")
    for k in start_state:
        print(f"    {k:18s} {current_joints[k]:7.2f} -> {start_state[k]:7.2f}  "
              f"(delta={start_state[k] - current_joints[k]:+7.2f})")
    print("Ctrl+C now to abort.")
    time.sleep(1.5)

    for step in range(1, steps + 1):
        loop_start = time.perf_counter()
        alpha = step / steps
        target = {
            key: current_joints[key] + alpha * (start_state[key] - current_joints[key])
            for key in start_state
        }
        follower.send_action(target)
        sleep_t = 1.0 / fps - (time.perf_counter() - loop_start)
        if sleep_t > 0:
            time.sleep(sleep_t)


def main():
    dataset, action_rows, reference_state_rows, action_names, state_names, image_keys, fps = _load_episode_data()
    joint_names = [_base_joint_name(name) for name in action_names]

    follower = create_follower(camera=False)

    print(f"Dataset: {DATASET_REPO_ID}")
    print(f"Episode: {EPISODE_INDEX}")
    print(f"Frames:  {dataset.num_frames}")
    print(f"FPS:     {fps}")
    print(f"Cameras: {image_keys}")
    print("Connecting follower arm...")
    follower.connect(calibrate=False)

    _init_rerun(joint_names, image_keys)

    try:
        if MOVE_TO_START:
            # Ramp to action[0], not observation.state[0]: at record time the
            # leader is slightly ahead of the follower, so action[0] is where
            # we'll command on replay step 0. Ramping there avoids a jerk on
            # the transition from move-to-start into replay.
            start_target = _vector_to_dict(action_rows[0]["action"], action_names)
            # Keys in start_target are *.pos (matching action_names). The
            # follower's get_observation also returns *.pos keys, so they line
            # up directly.
            _move_to_start(follower, start_target, fps)

        print(f"Starting replay in {START_DELAY_S}s. Press Ctrl+C to stop.")
        for remaining in range(START_DELAY_S, 0, -1):
            print(f"  {remaining}...", end="\r")
            time.sleep(1)
        print(" " * 40, end="\r")

        run_start = time.perf_counter()

        for idx in range(dataset.num_frames):
            loop_start = time.perf_counter()

            live_state = follower.get_observation()
            reference_state = _vector_to_dict(reference_state_rows[idx]["observation.state"], state_names)
            replay_action = _vector_to_dict(action_rows[idx]["action"], action_names)
            sent_action = follower.send_action(replay_action)

            # Load and log recorded camera frames for this dataset index
            if image_keys:
                sample = dataset[idx]
                for key in image_keys:
                    img = sample[key]  # tensor [3, H, W] float32 in [0, 1]
                    img_np = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
                    rr.log(f"cameras/{key}", rr.Image(img_np))

            max_abs_error = 0.0
            max_abs_error_joint = joint_names[0]

            rr.set_time("step", sequence=idx)
            rr.set_time("time", timestamp=time.perf_counter() - run_start)

            for action_key, state_key in zip(action_names, state_names, strict=True):
                joint_name = _base_joint_name(action_key)
                live_val = float(live_state[state_key])
                ref_val = float(reference_state[state_key])
                replay_val = float(replay_action[action_key])
                sent_val = float(sent_action[action_key])
                err_val = live_val - ref_val

                rr.log(f"joints/{joint_name}/live_state", rr.Scalars([live_val]))
                rr.log(f"joints/{joint_name}/reference_state", rr.Scalars([ref_val]))
                rr.log(f"joints/{joint_name}/replay_action", rr.Scalars([replay_val]))
                rr.log(f"joints/{joint_name}/sent_action", rr.Scalars([sent_val]))
                rr.log(f"joints/{joint_name}/tracking_error", rr.Scalars([err_val]))

                if abs(err_val) > max_abs_error:
                    max_abs_error = abs(err_val)
                    max_abs_error_joint = joint_name

            rr.log("metrics/max_abs_tracking_error", rr.Scalars([max_abs_error]))

            print(
                f"Step {idx + 1:05d}/{dataset.num_frames:05d} | "
                f"max tracking error: {max_abs_error_joint}={max_abs_error:6.2f}",
                end="\r",
            )

            sleep_t = 1.0 / fps - (time.perf_counter() - loop_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n\nStopping replay...")
    finally:
        safe_disconnect(follower)
        print("Disconnected. Done!")


if __name__ == "__main__":
    main()
