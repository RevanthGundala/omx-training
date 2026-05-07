"""Overlay every episode's state/action trajectories in Rerun for distribution analysis.

Reads parquet directly (no video decoding) so it works on macOS where torchcodec
often can't find ffmpeg.

Each episode is logged on its own timeline (`ep_{N}`) so you can:
  - Toggle episodes on/off in the viewer
  - See if first-frame poses cluster or spread (open `state/{joint}` plots)
  - Compare your eval rollout (load its .rrd alongside) against training spread

Usage:
    uv run python evaluation/visualize_distribution.py
    uv run python evaluation/visualize_distribution.py --repo-id RevanthGundala/003-pour-water
    uv run python evaluation/visualize_distribution.py --max-episodes 5 --stride 2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rerun as rr
from huggingface_hub import snapshot_download

from utils.config import TRAIN_DATASET_REPO_ID, JOINT_NAMES


def _load_episode_frames(local_root: Path) -> pd.DataFrame:
    """Concatenate every episode parquet under data/chunk-*/file-*.parquet."""
    parquet_files = sorted(local_root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_files:
        # Older v2 layout
        parquet_files = sorted(local_root.glob("data/chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {local_root}/data/")
    frames = []
    for p in parquet_files:
        frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=TRAIN_DATASET_REPO_ID)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-rerun", action="store_true",
                    help="Just print stats; don't spawn the viewer.")
    args = ap.parse_args()

    print(f"Resolving {args.repo_id} (data only, skipping videos)...")
    local_root = Path(snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=["data/**/*.parquet", "meta/**"],
    ))
    print(f"  local cache: {local_root}")

    df = _load_episode_frames(local_root)
    if "episode_index" not in df.columns:
        raise RuntimeError(f"parquet missing episode_index column. Have: {list(df.columns)}")
    eps = sorted(df["episode_index"].unique().tolist())
    if args.max_episodes is not None:
        eps = eps[: args.max_episodes]
    print(f"Loaded {len(df)} frames across {len(eps)} episodes")

    if not args.no_rerun:
        rr.init(f"distribution_{args.repo_id.split('/')[-1]}", spawn=True)

    first_states = []
    all_states = []

    for ep in eps:
        ep_df = df[df["episode_index"] == ep].sort_values("frame_index").reset_index(drop=True)
        timeline_name = f"ep_{ep:03d}"
        states = np.stack(ep_df["observation.state"].to_numpy())
        actions = np.stack(ep_df["action"].to_numpy())
        all_states.append(states)
        first_states.append(states[0])

        if not args.no_rerun:
            for i in range(0, len(ep_df), args.stride):
                t = float(ep_df["timestamp"].iloc[i]) if "timestamp" in ep_df.columns else float(i) / 30.0
                rr.set_time(timeline_name, duration=t)
                for j, name in enumerate(JOINT_NAMES):
                    rr.log(f"state/{name}/ep_{ep:03d}", rr.Scalars(float(states[i, j])))
                    rr.log(f"action/{name}/ep_{ep:03d}", rr.Scalars(float(actions[i, j])))

        print(f"  ep {ep}: {len(ep_df)} frames  first_state={states[0].round(2).tolist()}")

    arr = np.concatenate(all_states, axis=0)
    first = np.array(first_states)
    print("\n=== Per-joint stats ===")
    print(f"{'joint':18s}  {'mean':>7s}  {'std':>6s}  {'min':>7s}  {'max':>7s}  {'first_std':>9s}")
    for j, name in enumerate(JOINT_NAMES):
        print(
            f"{name:18s}  {arr[:, j].mean():7.2f}  {arr[:, j].std():6.2f}  "
            f"{arr[:, j].min():7.2f}  {arr[:, j].max():7.2f}  {first[:, j].std():9.2f}"
        )
    print("\nfirst_std = std across episodes' STARTING poses.")
    print("Low first_std (< ~5) on a joint => all demos start the same → eval will fail OOD.")


if __name__ == "__main__":
    main()
