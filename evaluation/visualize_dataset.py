"""Quick Rerun visualizer for a LeRobot dataset episode.

Usage:
    uv run python evaluation/visualize_dataset.py --episode 0
    uv run python evaluation/visualize_dataset.py --episode 5 --repo-id RevanthGundala/003-pour-water
"""

import argparse
import numpy as np
import rerun as rr

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from utils.config import TRAIN_DATASET_REPO_ID, JOINT_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=TRAIN_DATASET_REPO_ID)
    ap.add_argument("--episode", type=int, default=0)
    args = ap.parse_args()

    print(f"Loading {args.repo_id} episode {args.episode}...")
    ds = LeRobotDataset(args.repo_id, episodes=[args.episode])

    rr.init(f"dataset_{args.repo_id.split('/')[-1]}_ep{args.episode}", spawn=True)

    for i in range(len(ds)):
        sample = ds[i]
        rr.set_time("time", duration=sample["timestamp"].item())

        state = sample["observation.state"].numpy()
        action = sample["action"].numpy()
        for j, name in enumerate(JOINT_NAMES):
            rr.log(f"state/{name}", rr.Scalars(float(state[j])))
            rr.log(f"action/{name}", rr.Scalars(float(action[j])))

        for cam in ("wrist", "top"):
            key = f"observation.images.{cam}"
            if key in sample:
                img = sample[key]
                if hasattr(img, "numpy"):
                    img = img.numpy()
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                rr.log(f"camera/{cam}", rr.Image(img))

    print(f"Logged {len(ds)} frames. Rerun viewer should be open.")


if __name__ == "__main__":
    main()
