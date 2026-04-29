"""
record.py — Record teleoperation episodes to a LeRobot dataset.

Uses the leader arm for teleoperation and the follower arm as the robot.
Records joint positions + camera images to a local dataset.

Controls:
  →  stop recording, enter review
  ←  discard episode immediately

During review (after → stops recording):
  ↑  replay the episode on the follower
  →  save episode
  ←  discard episode

Ctrl+C during reset → stop recording entirely
"""

import time
import threading
import shutil
from pathlib import Path

import numpy as np
import rerun as rr
from huggingface_hub import HfApi
from pynput import keyboard

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features

from config import CAMERAS, FPS, JOINT_NAMES, RECORD_DATASET_REPO_ID as DATASET_REPO_ID, TASK_NAME
from control_utils import maintain_fps
from rerun_utils import init_rerun
from robot_utils import create_follower, create_leader, safe_disconnect

# ──────────────────────────────────────────────
# Replay from buffer helper
# ──────────────────────────────────────────────
MOVE_TO_START_DURATION_S = 4.0


def replay_from_buffer(action_buffer, state_buffer, action_names, state_names, follower, fps):
    """Replay buffered frames on the follower arm."""
    num_frames = len(action_buffer)
    if num_frames == 0:
        print("  No frames to replay.")
        return

    print(f"\n  ▶ Replaying {num_frames} frames...")

    # Move to start position
    start_state = dict(zip(state_names, state_buffer[0]))
    current = follower.get_observation()
    steps = max(1, int(MOVE_TO_START_DURATION_S * fps))
    print(f"  Moving to start pose over {MOVE_TO_START_DURATION_S}s...")
    for step in range(1, steps + 1):
        loop_start = time.perf_counter()
        alpha = step / steps
        target = {k: current[k] + alpha * (start_state[k] - current[k]) for k in start_state}
        follower.send_action(target)
        maintain_fps(loop_start, fps)

    # Play back actions
    print(f"  Playing... (Ctrl+C to skip)")
    try:
        for idx, action_vals in enumerate(action_buffer):
            loop_start = time.perf_counter()
            action = dict(zip(action_names, action_vals))
            follower.send_action(action)

            # Stream live camera to Rerun during replay
            observation = follower.get_observation()
            rr.set_time("replay_step", sequence=idx)
            for cam_name in CAMERAS:
                if cam_name in observation:
                    rr.log(f"camera/{cam_name}", rr.Image(observation[cam_name]))
            for name in JOINT_NAMES:
                key = f"{name}.pos"
                if key in observation:
                    rr.log(f"joints/{name}/replay", rr.Scalars(observation[key]))

            print(f"  Replay frame {idx+1:4d}/{num_frames}", end="\r")
            maintain_fps(loop_start, fps)
    except KeyboardInterrupt:
        print("\n  Replay interrupted.")
    print(f"\n  ▶ Replay complete.")

# ──────────────────────────────────────────────
# Recording-specific configuration
# ──────────────────────────────────────────────
USE_CAMERA = True
NUM_EPISODES = 50
EPISODE_DURATION_S = 90
RESET_DURATION_S = 10
USE_VIDEO = True
PUSH_TO_HUB = False
SOFT_START_DURATION_S = 3.0  # gradually ramp follower to leader position on connect


def soft_start(follower, leader, duration_s=SOFT_START_DURATION_S):
    """Gradually move follower to match leader position to prevent jerk/overload."""
    print(f"  Soft-starting: ramping follower to leader over {duration_s}s...")
    current = follower.get_observation()
    target = leader.get_action()
    steps = max(1, int(duration_s * FPS))

    for step in range(1, steps + 1):
        loop_start = time.perf_counter()
        alpha = step / steps
        blended = {
            key: current[key] + alpha * (target[key] - current[key])
            for key in target
        }
        follower.send_action(blended)
        # Re-read leader in case it moved during ramp
        target = leader.get_action()
        maintain_fps(loop_start, FPS)

    print("  Soft-start complete.")


def record_one_episode(robot, leader, dataset, episode_num, rerun_step=0):
    """Record a single episode. Returns (frame_count, action_buffer, state_buffer, rerun_step).
    frame_count = -1 means discard was pressed during recording."""
    print(f"\n{'='*60}")
    print(f"  RECORDING Episode {episode_num}")
    print(f"  Task: {TASK_NAME}")
    print(f"  Max duration: {EPISODE_DURATION_S}s — → stop & review, ← discard")
    print(f"{'='*60}\n")

    end_episode = threading.Event()
    discard_episode = threading.Event()

    def on_press(key):
        if key == keyboard.Key.right:
            end_episode.set()
        elif key == keyboard.Key.left:
            discard_episode.set()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    frame_count = 0
    action_buffer = []
    state_buffer = []
    start_time = time.perf_counter()

    try:
        while not end_episode.is_set() and not discard_episode.is_set():
            loop_start = time.perf_counter()
            elapsed = loop_start - start_time

            if elapsed >= EPISODE_DURATION_S:
                print(f"\n  Episode time limit reached ({EPISODE_DURATION_S}s).")
                break

            observation = robot.get_observation()
            action = leader.get_action()
            sent_action = robot.send_action(action)

            # Buffer for potential replay
            action_buffer.append([sent_action[k] for k in sorted(sent_action)])
            state_buffer.append([observation[k] for k in sorted(observation)])

            obs_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**obs_frame, **action_frame}
            frame["task"] = TASK_NAME
            dataset.add_frame(frame)
            frame_count += 1

            # ── Rerun: stream camera + joints live ──
            rr.set_time("step", sequence=rerun_step)
            for cam_name in CAMERAS:
                if cam_name in observation:
                    rr.log(f"camera/{cam_name}", rr.Image(observation[cam_name]))
            for i, name in enumerate(JOINT_NAMES):
                obs_key = f"{name}.pos"
                act_key = f"{name}.pos"
                if obs_key in observation:
                    rr.log(f"joints/{name}/state", rr.Scalars(observation[obs_key]))
                if act_key in sent_action:
                    rr.log(f"joints/{name}/action", rr.Scalars(sent_action[act_key]))
            rerun_step += 1

            print(f"  Frame {frame_count:4d} | Time: {elapsed:6.1f}s", end="\r")
            maintain_fps(loop_start, FPS)

    except KeyboardInterrupt:
        print(f"\n  Episode ended early by user.")
    finally:
        listener.stop()

    print(f"  Recorded {frame_count} frames ({frame_count/FPS:.1f}s)")

    action_names = sorted(sent_action.keys()) if frame_count > 0 else []
    state_names = sorted(observation.keys()) if frame_count > 0 else []

    if discard_episode.is_set():
        return -1, [], [], [], [], rerun_step
    return frame_count, action_buffer, state_buffer, action_names, state_names, rerun_step


def main():
    leader = create_leader()
    follower = create_follower(camera=USE_CAMERA)

    # Build dataset feature definitions from robot hardware
    action_features = hw_to_dataset_features(follower.action_features, "action", USE_VIDEO)
    obs_features = hw_to_dataset_features(follower.observation_features, "observation", USE_VIDEO)
    dataset_features = {**action_features, **obs_features}

    print("Dataset features:")
    for name, feat in dataset_features.items():
        print(f"  {name}: shape={feat['shape']}, dtype={feat['dtype']}")

    # Create or resume the dataset
    dataset_path = Path.home() / ".cache/huggingface/lerobot" / DATASET_REPO_ID
    info_path = dataset_path / "meta" / "info.json"

    if info_path.exists():
        import json
        with open(info_path) as f:
            info = json.load(f)
        if info.get("total_episodes", 0) > 0:
            # Has committed episodes — resume using the proper classmethod
            print(f"\nResuming existing dataset at {dataset_path}")
            dataset = LeRobotDataset.resume(
                repo_id=DATASET_REPO_ID,
                root=str(dataset_path),
                image_writer_processes=0,
                image_writer_threads=4,
            )
            print(f"  Existing episodes: {dataset.num_episodes}")
        else:
            # Created but 0 episodes — safe to recreate
            print(f"\n⚠️  Found empty dataset (0 episodes). Re-creating.")
            shutil.rmtree(dataset_path)
            dataset = LeRobotDataset.create(
                repo_id=DATASET_REPO_ID,
                fps=FPS,
                robot_type="omx_follower",
                features=dataset_features,
                use_videos=USE_VIDEO,
                image_writer_processes=0,
                image_writer_threads=4,
            )
    else:
        # No dataset at all — create fresh
        if dataset_path.exists():
            backup_path = dataset_path.with_name(dataset_path.name + f"_backup_{int(time.time())}")
            print(f"\n⚠️  Existing directory has no info.json. Backing up to: {backup_path}")
            shutil.move(str(dataset_path), str(backup_path))
        print(f"\nCreating new dataset: {DATASET_REPO_ID}")
        dataset = LeRobotDataset.create(
            repo_id=DATASET_REPO_ID,
            fps=FPS,
            robot_type="omx_follower",
            features=dataset_features,
            use_videos=USE_VIDEO,
            image_writer_processes=0,
            image_writer_threads=4,
        )

    # Connect hardware
    print("\nConnecting leader arm...")
    leader.connect()
    print("Connecting follower arm...")
    follower.connect()

    # Gradually ramp follower to leader position to prevent jerk/overload
    soft_start(follower, leader)

    # ── Rerun setup (camera POV + joint plots) ──
    init_rerun("omx_record", has_camera=USE_CAMERA, camera_primary=True, save_rrd=False)

    # Record episodes in a loop
    episode = 0
    rerun_step = 0
    try:
        while episode < NUM_EPISODES:
            result = record_one_episode(follower, leader, dataset, dataset.num_episodes, rerun_step)
            frame_count, action_buf, state_buf, action_names, state_names, rerun_step = result

            if frame_count == 0:
                print("  No frames recorded, skipping episode.")
                dataset.clear_episode_buffer()
                continue

            if frame_count < 0:
                print("  ← Episode DISCARDED.")
                dataset.clear_episode_buffer()
                continue

            # ── Review phase: replay/save/discard ──
            print(f"\n  REVIEW: ↑ replay, → save, ← discard")
            save_ep = threading.Event()
            discard_ep = threading.Event()
            replay_ep = threading.Event()

            def on_review_press(key):
                if key == keyboard.Key.right:
                    save_ep.set()
                elif key == keyboard.Key.left:
                    discard_ep.set()
                elif key == keyboard.Key.up:
                    replay_ep.set()

            review_listener = keyboard.Listener(on_press=on_review_press)
            review_listener.start()

            try:
                while not save_ep.is_set() and not discard_ep.is_set():
                    # Keep teleop alive while waiting
                    loop_start = time.perf_counter()
                    action = leader.get_action()
                    follower.send_action(action)

                    if replay_ep.is_set():
                        replay_ep.clear()
                        replay_from_buffer(
                            action_buf, state_buf, action_names, state_names,
                            follower, FPS,
                        )
                        soft_start(follower, leader)
                        print(f"  REVIEW: ↑ replay again, → save, ← discard")

                    maintain_fps(loop_start, FPS)
            except KeyboardInterrupt:
                review_listener.stop()
                print("\n\n  Stopping recording.")
                dataset.clear_episode_buffer()
                break
            finally:
                review_listener.stop()

            if discard_ep.is_set():
                print("  ← Episode DISCARDED.")
                dataset.clear_episode_buffer()
                continue

            # Save the episode
            dataset.save_episode()
            episode += 1
            print(f"  ✓ Episode saved! (Total episodes: {dataset.num_episodes})")

            # Brief reset period
            print(f"\n  Resetting for {RESET_DURATION_S}s... (Ctrl+C to stop)")
            try:
                start = time.perf_counter()
                while time.perf_counter() - start < RESET_DURATION_S:
                    remaining = RESET_DURATION_S - (time.perf_counter() - start)
                    print(f"  Next episode in {remaining:.0f}s...", end="\r")
                    action = leader.get_action()
                    follower.send_action(action)
                    maintain_fps(time.perf_counter(), FPS)
            except KeyboardInterrupt:
                print("\n\n  Stopping recording.")
                break

    except KeyboardInterrupt:
        print("\n\nRecording interrupted.")

    finally:
        safe_disconnect(follower)
        safe_disconnect(leader)

    # Push to hub (optional)
    if PUSH_TO_HUB:
        api = HfApi()
        api.create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", exist_ok=True)
        print("\nPushing dataset to HuggingFace Hub...")
        dataset.push_to_hub()
        print("Done!")

    print(f"\nDataset saved locally: {dataset.root}")
    print(f"Total episodes: {dataset.num_episodes}")


if __name__ == "__main__":
    main()
