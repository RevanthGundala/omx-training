"""
record.py — Record teleoperation episodes to a LeRobot dataset.

Uses the leader arm for teleoperation and the follower arm as the robot.
Records joint positions + camera images to a local dataset.
Press right arrow or Ctrl+C during an episode to finish it early.
Press Ctrl+C during reset to stop recording entirely.
"""

import time
import threading
import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi
from pynput import keyboard

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features

from config import FPS, RECORD_DATASET_REPO_ID as DATASET_REPO_ID, TASK_NAME
from control_utils import maintain_fps
from robot_utils import create_follower, create_leader, safe_disconnect

# ──────────────────────────────────────────────
# Recording-specific configuration
# ──────────────────────────────────────────────
USE_CAMERA = True
NUM_EPISODES = 50
EPISODE_DURATION_S = 90
RESET_DURATION_S = 5
USE_VIDEO = True
PUSH_TO_HUB = False


def record_one_episode(robot, leader, dataset, episode_num):
    """Record a single episode: teleop the robot and save every frame."""
    print(f"\n{'='*60}")
    print(f"  RECORDING Episode {episode_num}")
    print(f"  Task: {TASK_NAME}")
    print(f"  Max duration: {EPISODE_DURATION_S}s — press → or Ctrl+C to finish early")
    print(f"{'='*60}\n")

    # Track right arrow key press to end episode
    end_episode = threading.Event()

    def on_press(key):
        if key == keyboard.Key.right:
            end_episode.set()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    frame_count = 0
    start_time = time.perf_counter()

    try:
        while not end_episode.is_set():
            loop_start = time.perf_counter()
            elapsed = loop_start - start_time

            if elapsed >= EPISODE_DURATION_S:
                print(f"\n  Episode time limit reached ({EPISODE_DURATION_S}s).")
                break

            # Step 1: Read the robot's current observation (joints + cameras)
            observation = robot.get_observation()

            # Step 2: Read the leader arm's joint positions as the action
            action = leader.get_action()

            # Step 3: Send the action to the follower arm
            sent_action = robot.send_action(action)

            # Step 4: Build the dataset frame from raw hardware values
            obs_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**obs_frame, **action_frame}

            # Step 5: Add the frame to the dataset buffer
            dataset.add_frame(frame, task=TASK_NAME)
            frame_count += 1

            # Print progress
            print(f"  Frame {frame_count:4d} | Time: {elapsed:6.1f}s", end="\r")

            # Maintain target FPS
            maintain_fps(loop_start, FPS)

    except KeyboardInterrupt:
        print(f"\n  Episode ended early by user.")
    finally:
        listener.stop()

    print(f"  Recorded {frame_count} frames ({frame_count/FPS:.1f}s)")
    return frame_count


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
    required_meta = ["info.json", "tasks.jsonl", "episodes.jsonl"]
    if all((dataset_path / "meta" / f).exists() for f in required_meta):
        print(f"\nResuming existing dataset at {dataset_path}")
        dataset = LeRobotDataset(DATASET_REPO_ID)
        dataset.start_image_writer(num_processes=0, num_threads=4)
        print(f"  Existing episodes: {dataset.num_episodes}")
    else:
        # Remove stale directory if it exists without valid metadata
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
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

    # Record episodes in a loop (→ ends episode, Ctrl+C stops entirely)
    episode = 0
    try:
        while episode < NUM_EPISODES:
            frame_count = record_one_episode(follower, leader, dataset, dataset.num_episodes)

            if frame_count == 0:
                print("  No frames recorded, skipping episode.")
                dataset.clear_episode_buffer()
                continue

            # Save the episode to disk (parquet + video encoding)
            dataset.save_episode()
            episode += 1
            print(f"  Episode saved! (Total episodes: {dataset.num_episodes})")

            # Brief reset period, then auto-start next episode
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
