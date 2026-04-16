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
from lerobot.utils.robot_utils import busy_wait

from config import FOLLOWER_PORT, RECORD_DATASET_REPO_ID as DATASET_REPO_ID
from rerun_utils import build_joint_blueprint
from robot_utils import create_follower, safe_disconnect

# ──────────────────────────────────────────────
# Replay-specific configuration
# ──────────────────────────────────────────────
EPISODE_INDEX = 0
FPS_OVERRIDE = None
START_DELAY_S = 3
MOVE_TO_START = True
MOVE_TO_START_DURATION_S = 4.0
TOLERANCE_S = 1e4


def _vector_to_dict(values, names: list[str]) -> dict[str, float]:
    return {name: float(values[i]) for i, name in enumerate(names)}


def _base_joint_name(name: str) -> str:
    return name.removesuffix(".pos")


def _init_rerun(joint_names: list[str]) -> None:
    joint_views = [
        rrb.TimeSeriesView(name=joint_name, contents=[f"joints/{joint_name}/**"])
        for joint_name in joint_names
    ]
    # Replay uses "Tracking error" instead of camera, so custom layout
    blueprint = rrb.Horizontal(
        rrb.TimeSeriesView(name="Tracking error", contents=["metrics/**"]),
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
    )
    action_rows = dataset.hf_dataset.select_columns("action")
    reference_state_rows = dataset.hf_dataset.select_columns("observation.state")
    action_names = dataset.features["action"]["names"]
    state_names = dataset.features["observation.state"].get("names", action_names)
    fps = dataset.fps if FPS_OVERRIDE is None else FPS_OVERRIDE
    return dataset, action_rows, reference_state_rows, action_names, state_names, fps


def _move_to_start(
    follower: OmxFollower,
    start_state: dict[str, float],
    fps: int,
) -> None:
    current_state = follower.get_observation()
    steps = max(1, int(MOVE_TO_START_DURATION_S * fps))

    print(
        f"Moving to recorded start state over {MOVE_TO_START_DURATION_S:.1f}s "
        f"({steps} steps)..."
    )
    for step in range(1, steps + 1):
        loop_start = time.perf_counter()
        alpha = step / steps
        target = {
            key: current_state[key] + alpha * (start_state[key] - current_state[key])
            for key in start_state
        }
        follower.send_action(target)
        busy_wait(max(1.0 / fps - (time.perf_counter() - loop_start), 0.0))


def main():
    dataset, action_rows, reference_state_rows, action_names, state_names, fps = _load_episode_data()
    joint_names = [_base_joint_name(name) for name in action_names]

    follower = create_follower(camera=False)

    print(f"Dataset: {DATASET_REPO_ID}")
    print(f"Episode: {EPISODE_INDEX}")
    print(f"Frames:  {dataset.num_frames}")
    print(f"FPS:     {fps}")
    print("Connecting follower arm...")
    follower.connect(calibrate=False)

    _init_rerun(joint_names)

    try:
        if MOVE_TO_START:
            start_state = _vector_to_dict(reference_state_rows[0]["observation.state"], state_names)
            _move_to_start(follower, start_state, fps)

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

            max_abs_error = 0.0
            max_abs_error_joint = joint_names[0]

            rr.set_time_sequence("step", idx)
            rr.set_time_seconds("time", time.perf_counter() - run_start)

            for action_key, state_key in zip(action_names, state_names, strict=True):
                joint_name = _base_joint_name(action_key)
                live_val = float(live_state[state_key])
                ref_val = float(reference_state[state_key])
                replay_val = float(replay_action[action_key])
                sent_val = float(sent_action[action_key])
                err_val = live_val - ref_val

                rr.log(f"joints/{joint_name}/live_state", rr.Scalar(live_val))
                rr.log(f"joints/{joint_name}/reference_state", rr.Scalar(ref_val))
                rr.log(f"joints/{joint_name}/replay_action", rr.Scalar(replay_val))
                rr.log(f"joints/{joint_name}/sent_action", rr.Scalar(sent_val))
                rr.log(f"joints/{joint_name}/tracking_error", rr.Scalar(err_val))

                if abs(err_val) > max_abs_error:
                    max_abs_error = abs(err_val)
                    max_abs_error_joint = joint_name

            rr.log("metrics/max_abs_tracking_error", rr.Scalar(max_abs_error))

            print(
                f"Step {idx + 1:05d}/{dataset.num_frames:05d} | "
                f"max tracking error: {max_abs_error_joint}={max_abs_error:6.2f}",
                end="\r",
            )

            busy_wait(max(1.0 / fps - (time.perf_counter() - loop_start), 0.0))

    except KeyboardInterrupt:
        print("\n\nStopping replay...")
    finally:
        safe_disconnect(follower)
        print("Disconnected. Done!")


if __name__ == "__main__":
    main()
