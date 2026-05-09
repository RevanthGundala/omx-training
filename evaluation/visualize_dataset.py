"""Quick Rerun visualizer for LeRobot dataset episodes.

Usage:
    uv run python evaluation/visualize_dataset.py --episode 0
    uv run python evaluation/visualize_dataset.py --episode 5 --repo-id RevanthGundala/003-pour-water
    uv run python evaluation/visualize_dataset.py --episodes all --visual-only
    uv run python evaluation/visualize_dataset.py --episodes 50:70 --repo-id RevanthGundala/003-pour-water-globalstats
    uv run python evaluation/visualize_dataset.py --episodes all --grid --grid-cols 6 --samples-per-episode 4
    uv run python evaluation/visualize_dataset.py --episodes all --play-grid --stride 10
"""

import argparse
from pathlib import Path
import numpy as np
import rerun as rr

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from utils.config import TRAIN_DATASET_REPO_ID, JOINT_NAMES


def parse_episodes(value: str | None, fallback: int) -> list[int] | None:
    if value is None:
        return [fallback]
    if value == "all":
        return None
    if ":" in value:
        start, end = value.split(":", 1)
        return list(range(int(start), int(end)))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def image_to_uint8_hwc(img):
    if hasattr(img, "numpy"):
        img = img.numpy()
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    return img


def resize_nearest(img, size: tuple[int, int]):
    height, width = size
    ys = np.linspace(0, img.shape[0] - 1, height).astype(np.int64)
    xs = np.linspace(0, img.shape[1] - 1, width).astype(np.int64)
    return img[ys][:, xs]


def log_episode_grid(ds, episodes: list[int], samples_per_episode: int, grid_cols: int, thumb_width: int):
    first_indices = collect_episode_indices(ds)

    episode_ids = episodes if episodes is not None else sorted(first_indices)
    cams = ("top", "wrist")
    thumb_height = int(thumb_width * 0.75)
    cell_width = thumb_width * samples_per_episode
    cell_height = thumb_height

    for cam in cams:
        rows = int(np.ceil(len(episode_ids) / grid_cols))
        sheet = np.zeros((rows * cell_height, grid_cols * cell_width, 3), dtype=np.uint8)
        for epi, episode in enumerate(episode_ids):
            indices = first_indices.get(episode, [])
            if not indices:
                continue
            sample_positions = np.linspace(0, len(indices) - 1, samples_per_episode).astype(int)
            row = epi // grid_cols
            col = epi % grid_cols
            for sample_col, pos in enumerate(sample_positions):
                sample = ds[indices[pos]]
                key = f"observation.images.{cam}"
                if key not in sample:
                    continue
                img = resize_nearest(image_to_uint8_hwc(sample[key]), (thumb_height, thumb_width))
                y0 = row * cell_height
                x0 = col * cell_width + sample_col * thumb_width
                sheet[y0 : y0 + thumb_height, x0 : x0 + thumb_width] = img

        rr.log(f"grid/{cam}", rr.Image(sheet))
        print(f"Logged {cam} grid: {len(episode_ids)} episodes, {samples_per_episode} samples/episode")


def collect_episode_indices(ds) -> dict[int, list[int]]:
    indices: dict[int, list[int]] = {}
    for i in range(len(ds)):
        sample = ds.hf_dataset[i]
        episode = int(sample["episode_index"])
        indices.setdefault(episode, []).append(i)
    return indices


def log_playable_grid(ds, episodes: list[int], grid_cols: int, thumb_width: int, stride: int):
    episode_indices = collect_episode_indices(ds)
    episode_ids = episodes if episodes is not None else sorted(episode_indices)
    cams = ("top", "wrist")
    thumb_height = int(thumb_width * 0.75)
    rows = int(np.ceil(len(episode_ids) / grid_cols))
    max_len = max(len(episode_indices[episode]) for episode in episode_ids)

    for frame in range(0, max_len, stride):
        rr.set_time("relative_frame", sequence=frame)
        for cam in cams:
            sheet = np.full((rows * thumb_height, grid_cols * thumb_width, 3), 32, dtype=np.uint8)
            for epi, episode in enumerate(episode_ids):
                indices = episode_indices.get(episode, [])
                if not indices:
                    continue
                sample = ds[indices[min(frame, len(indices) - 1)]]
                key = f"observation.images.{cam}"
                if key not in sample:
                    continue
                img = resize_nearest(image_to_uint8_hwc(sample[key]), (thumb_height, thumb_width))
                row = epi // grid_cols
                col = epi % grid_cols
                y0 = row * thumb_height
                x0 = col * thumb_width
                sheet[y0 : y0 + thumb_height, x0 : x0 + thumb_width] = img
            rr.log(f"play_grid/{cam}", rr.Image(sheet))

    print(
        f"Logged playable grid: {len(episode_ids)} episodes, "
        f"{max_len // stride + 1} timesteps, stride={stride}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=TRAIN_DATASET_REPO_ID)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--episodes", default=None, help="'all', 'start:end', or comma-separated episode ids")
    ap.add_argument("--visual-only", action="store_true", help="Only log camera images, not state/action scalars")
    ap.add_argument("--stride", type=int, default=1, help="Log every Nth frame")
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--grid", action="store_true", help="Log contact-sheet grids instead of a frame timeline")
    ap.add_argument("--play-grid", action="store_true", help="Log playable contact-sheet grids over relative episode time")
    ap.add_argument("--grid-cols", type=int, default=5, help="Episode cells per grid row")
    ap.add_argument("--samples-per-episode", type=int, default=5, help="Frames sampled across each episode cell")
    ap.add_argument("--thumb-width", type=int, default=128)
    args = ap.parse_args()

    episodes = parse_episodes(args.episodes, args.episode)
    label = "all" if episodes is None else ",".join(map(str, episodes[:5])) + ("..." if len(episodes) > 5 else "")
    print(f"Loading {args.repo_id} episodes {label}...")
    ds_kwargs = {"episodes": episodes, "video_backend": args.video_backend}
    if args.root is not None:
        ds_kwargs["root"] = args.root
    ds = LeRobotDataset(args.repo_id, **ds_kwargs)

    rr.init(f"dataset_{args.repo_id.split('/')[-1]}_eps_{label}", spawn=True)

    if args.grid:
        log_episode_grid(ds, episodes, args.samples_per_episode, args.grid_cols, args.thumb_width)
        print("Rerun grid viewer should be open.")
        return
    if args.play_grid:
        log_playable_grid(ds, episodes, args.grid_cols, args.thumb_width, args.stride)
        print("Rerun playable grid viewer should be open.")
        return

    for i in range(len(ds)):
        if i % args.stride != 0:
            continue
        sample = ds[i]
        episode = int(sample["episode_index"].item())
        frame = int(sample["frame_index"].item())
        rr.set_time("episode", sequence=episode)
        rr.set_time("frame", sequence=frame)
        rr.set_time("time", duration=sample["timestamp"].item())

        if not args.visual_only:
            state = sample["observation.state"].numpy()
            action = sample["action"].numpy()
            for j, name in enumerate(JOINT_NAMES):
                rr.log(f"state/{name}", rr.Scalars(float(state[j])))
                rr.log(f"action/{name}", rr.Scalars(float(action[j])))

        for cam in ("wrist", "top"):
            key = f"observation.images.{cam}"
            if key in sample:
                rr.log(f"camera/{cam}", rr.Image(image_to_uint8_hwc(sample[key])))

    print(f"Logged up to {len(ds)} frames. Rerun viewer should be open.")


if __name__ == "__main__":
    main()
