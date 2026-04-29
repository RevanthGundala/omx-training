"""
visualize.py — Visualize a recorded OMX dataset using Rerun.

Loads the dataset and opens the Rerun viewer showing:
  - Camera feed (front camera)
  - Joint state plots (what the robot actually did)
  - Joint action plots (what was commanded)

Usage: just run this script. It will open the Rerun viewer automatically.
"""

import rerun as rr
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from config import JOINT_NAMES, TRAIN_DATASET_REPO_ID as DATASET_REPO_ID
from rerun_utils import init_rerun

# ──────────────────────────────────────────────
# Visualization-specific configuration
# ──────────────────────────────────────────────
EPISODE_INDEX = 2


def main():
    # ── 1. Load the dataset (using pyav since FFmpeg 8 breaks torchcodec) ──
    print(f"Loading dataset: {DATASET_REPO_ID}")
    dataset = LeRobotDataset(DATASET_REPO_ID, video_backend="pyav")

    print(f"  Total episodes: {dataset.num_episodes}")
    print(f"  Total frames:   {dataset.num_frames}")
    print(f"  FPS:            {dataset.fps}")
    print(f"  Features:       {list(dataset.features.keys())}")

    # ── 2. Get frame range for the chosen episode ──
    from_idx = dataset.episode_data_index["from"][EPISODE_INDEX].item()
    to_idx = dataset.episode_data_index["to"][EPISODE_INDEX].item()
    num_frames = to_idx - from_idx
    print(f"\nVisualizing episode {EPISODE_INDEX}: {num_frames} frames ({num_frames / dataset.fps:.1f}s)")

    # ── 3. Initialize Rerun viewer with a layout blueprint ──
    has_camera = "observation.images.front" in dataset.features
    init_rerun(f"{DATASET_REPO_ID}/episode_{EPISODE_INDEX}", has_camera=has_camera)

    # ── 4. Log each frame ──
    for idx in tqdm.tqdm(range(from_idx, to_idx), desc="Loading frames"):
        item = dataset[idx]

        # Set the timeline position
        rr.set_time_sequence("frame_index", item["frame_index"].item())
        rr.set_time_seconds("timestamp", item["timestamp"].item())

        # Log camera image
        if "observation.images.front" in item:
            img_tensor = item["observation.images.front"]  # [C, H, W] float32 0-1
            img_np = (img_tensor * 255).byte().permute(1, 2, 0).numpy()  # [H, W, C] uint8
            rr.log("camera/front", rr.Image(img_np))

        # Log joint states (actual robot positions)
        if "observation.state" in item:
            for i, name in enumerate(JOINT_NAMES):
                val = item["observation.state"][i].item()
                rr.log(f"joints/{name}/state", rr.Scalars(val))

        # Log joint actions (commanded positions)
        if "action" in item:
            for i, name in enumerate(JOINT_NAMES):
                val = item["action"][i].item()

                # Filter corrupted values (e.g., wrist_roll bug reads ~3.7e8)
                if abs(val) > 10_000:
                    continue

                rr.log(f"joints/{name}/action", rr.Scalars(val))

    print("\nDone! The Rerun viewer should be open.")
    print("Tips:")
    print("  - Use the timeline scrubber at the bottom to navigate")
    print("  - Press spacebar to play/pause")
    print("  - Click items in the left panel to show/hide them")


if __name__ == "__main__":
    main()
