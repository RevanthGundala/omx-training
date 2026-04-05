"""
record.py — Record teleoperation episodes to a LeRobot dataset.

Uses the leader arm for teleoperation and the follower arm as the robot.
Records joint positions + camera images to a local dataset.
Press right arrow or Ctrl+C during an episode to finish it early.
Press Ctrl+C during reset to stop recording entirely.
"""

import time
import threading
from pathlib import Path

import numpy as np
from pynput import keyboard

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
from lerobot.robots.omx_follower.omx_follower import OmxFollower
from lerobot.teleoperators.omx_leader.config_omx_leader import OmxLeaderConfig
from lerobot.teleoperators.omx_leader.omx_leader import OmxLeader

# ──────────────────────────────────────────────
# Configuration — edit these to match your setup
# ──────────────────────────────────────────────
LEADER_PORT = "/dev/tty.usbmodem1301"
FOLLOWER_PORT = "/dev/tty.usbmodem1201"
USE_CAMERA = True  # set True when camera is connected
CAMERA_INDEX = 0  # camera device index (try 0, 1, 2 if wrong camera)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

DATASET_REPO_ID = "RevanthGundala/pick_up_packet_test"
TASK_NAME = "Pick up packet"
FPS = 30
NUM_EPISODES = 50  # keep recording until Ctrl+C or this many episodes
EPISODE_DURATION_S = 90  # max seconds per episode (→ to end early)
RESET_DURATION_S = 5  # seconds between episodes to reset the scene
USE_VIDEO = True  # encode camera frames to MP4
PUSH_TO_HUB = False  # set True to upload to HuggingFace after recording


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
            dt = time.perf_counter() - loop_start
            sleep_time = (1.0 / FPS) - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n  Episode ended early by user.")
    finally:
        listener.stop()

    print(f"  Recorded {frame_count} frames ({frame_count/FPS:.1f}s)")
    return frame_count


def main():
    # ── 1. Create hardware configs ──
    # id must match the calibration filename (e.g. "omx_leader_arm" → omx_leader_arm.json)
    leader_config = OmxLeaderConfig(port=LEADER_PORT, id="omx_leader_arm")
    cameras = {}
    if USE_CAMERA:
        cameras["front"] = OpenCVCameraConfig(
            index_or_path=CAMERA_INDEX,
            fps=FPS,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
        )

    follower_config = OmxFollowerConfig(
        port=FOLLOWER_PORT,
        id="omx_follower_arm",
        cameras=cameras,
    )

    # ── 2. Instantiate hardware ──
    leader = OmxLeader(leader_config)
    follower = OmxFollower(follower_config)

    # ── 3. Build dataset feature definitions from robot hardware ──
    action_features = hw_to_dataset_features(follower.action_features, "action", USE_VIDEO)
    obs_features = hw_to_dataset_features(follower.observation_features, "observation", USE_VIDEO)
    dataset_features = {**action_features, **obs_features}

    print("Dataset features:")
    for name, feat in dataset_features.items():
        print(f"  {name}: shape={feat['shape']}, dtype={feat['dtype']}")

    # ── 4. Create or resume the dataset ──
    dataset_path = Path.home() / ".cache/huggingface/lerobot" / DATASET_REPO_ID
    if dataset_path.exists():
        print(f"\nResuming existing dataset at {dataset_path}")
        dataset = LeRobotDataset(DATASET_REPO_ID)
        dataset.start_image_writer(num_processes=0, num_threads=4)
        print(f"  Existing episodes: {dataset.num_episodes}")
    else:
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

    # ── 5. Connect hardware ──
    print("\nConnecting leader arm...")
    leader.connect()
    print("Connecting follower arm...")
    follower.connect()

    # ── 6. Record episodes in a loop (→ ends episode, Ctrl+C stops entirely) ──
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
                    time.sleep(1.0 / FPS)
            except KeyboardInterrupt:
                print("\n\n  Stopping recording.")
                break

    except KeyboardInterrupt:
        print("\n\nRecording interrupted.")

    finally:
        # ── 7. Disconnect hardware ──
        follower.disconnect()
        leader.disconnect()

    # ── 8. Push to hub (optional) ──
    if PUSH_TO_HUB:
        print("\nPushing dataset to HuggingFace Hub...")
        dataset.push_to_hub()
        print("Done!")

    print(f"\nDataset saved locally: {dataset.root}")
    print(f"Total episodes: {dataset.num_episodes}")


if __name__ == "__main__":
    main()
