"""Build a relative-action variant of an OMX LeRobot v3.0 dataset.

For every frame, this script replaces the absolute action with:

    new_action = old_action - observation.state

NOTE: When training and evaluating on this dataset, the eval-time client must
convert the predicted delta back to an absolute target before sending to the
follower:

    target = current_state + policy_delta

See ``evaluation/eval_pi0_quic.py`` for the place to add this transform. NOT
modified yet — do that as a separate task once a model is trained on this
dataset.

The conversion intentionally rewrites only the parquet data column and copies
metadata/videos byte-for-byte where possible. That preserves the original video
assets exactly and avoids requiring video decode/re-encode during conversion.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_SOURCE = "RevanthGundala/003-pour-water"
DEFAULT_TARGET = "RevanthGundala/004-pour-water-relative"
EXPECTED_EPISODES = 50
EXPECTED_FRAMES = 29_449
DEFAULT_TASK = "Pour water from one plastic bottle into another."
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def lerobot_cache_root(repo_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def vector_stats(values: np.ndarray) -> dict:
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).astype(float).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).astype(float).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).astype(float).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).astype(float).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(float).tolist(),
    }


def copy_tree_contents(source: Path, target: Path, *, skip_top_level: set[str] | None = None) -> None:
    skip_top_level = skip_top_level or set()
    for child in source.iterdir():
        if child.name in skip_top_level:
            continue
        destination = target / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=False)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, destination)


def relative_table(table: pa.Table) -> tuple[pa.Table, np.ndarray]:
    old_action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    delta = (old_action - state).astype(np.float32)

    action_index = table.schema.get_field_index("action")
    action_type = table.schema.field(action_index).type
    delta_array = pa.array(delta.tolist(), type=action_type)
    return table.set_column(action_index, "action", delta_array), delta


def rewrite_data_parquets(source_root: Path, target_root: Path) -> np.ndarray:
    deltas: list[np.ndarray] = []
    source_data = source_root / "data"
    target_data = target_root / "data"

    for source_file in sorted(source_data.glob("**/*.parquet")):
        relative_path = source_file.relative_to(source_data)
        target_file = target_data / relative_path
        target_file.parent.mkdir(parents=True, exist_ok=True)

        parquet_file = pq.ParquetFile(source_file)
        with pq.ParquetWriter(target_file, parquet_file.schema_arrow, compression="snappy") as writer:
            for row_group_index in range(parquet_file.num_row_groups):
                row_group = parquet_file.read_row_group(row_group_index)
                converted, delta = relative_table(row_group)
                writer.write_table(converted)
                deltas.append(delta)

    if not deltas:
        raise RuntimeError(f"No parquet data files found under {source_data}")
    return np.concatenate(deltas, axis=0)


def copy_meta_with_updated_action_stats(source_root: Path, target_root: Path, deltas: np.ndarray) -> None:
    source_meta = source_root / "meta"
    target_meta = target_root / "meta"
    shutil.copytree(source_meta, target_meta, dirs_exist_ok=False)

    stats = load_json(target_meta / "stats.json")
    stats["action"] = vector_stats(deltas)
    write_json(target_meta / "stats.json", stats)


def validate_source(source: str, source_root: Path) -> LeRobotDataset:
    dataset = LeRobotDataset(source, root=source_root, video_backend="pyav")
    if dataset.fps != 30:
        raise AssertionError(f"Expected 30 fps, got {dataset.fps}")
    if dataset.num_episodes != EXPECTED_EPISODES:
        raise AssertionError(f"Expected {EXPECTED_EPISODES} episodes, got {dataset.num_episodes}")
    if len(dataset) != EXPECTED_FRAMES:
        raise AssertionError(f"Expected {EXPECTED_FRAMES} frames, got {len(dataset)}")
    if dataset.features["action"]["shape"] != (6,):
        raise AssertionError(f"Expected action shape (6,), got {dataset.features['action']['shape']}")
    if dataset.features["observation.state"]["shape"] != (6,):
        raise AssertionError(
            f"Expected observation.state shape (6,), got {dataset.features['observation.state']['shape']}"
        )
    return dataset


def load_actions_and_states(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    for parquet_path in sorted((root / "data").glob("**/*.parquet")):
        table = pq.read_table(parquet_path, columns=["action", "observation.state", "episode_index"])
        actions.append(np.asarray(table["action"].to_pylist(), dtype=np.float32))
        states.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
        episode_indices.append(np.asarray(table["episode_index"].to_pylist(), dtype=np.int64))
    if not actions:
        raise RuntimeError(f"No parquet data files found under {root / 'data'}")
    return np.concatenate(actions), np.concatenate(states), np.concatenate(episode_indices)


def sanity_check(source_root: Path, target_root: Path, *, sample_size: int = 100) -> dict:
    source_actions, _source_states, source_episode_indices = load_actions_and_states(source_root)
    target_actions, target_states, target_episode_indices = load_actions_and_states(target_root)

    if source_actions.shape[0] != EXPECTED_FRAMES:
        raise AssertionError(f"Source frame count mismatch: {source_actions.shape[0]}")
    if target_actions.shape[0] != EXPECTED_FRAMES:
        raise AssertionError(f"Target frame count mismatch: {target_actions.shape[0]}")
    episode_count = int(np.unique(target_episode_indices).size)
    if episode_count != EXPECTED_EPISODES:
        raise AssertionError(f"Target episode count mismatch: {episode_count}")
    if not np.array_equal(source_episode_indices, target_episode_indices):
        raise AssertionError("Episode indices differ between source and target")

    rng = np.random.default_rng(4)
    sample_indices = rng.choice(target_actions.shape[0], size=min(sample_size, target_actions.shape[0]), replace=False)
    if not np.allclose(target_states[sample_indices] + target_actions[sample_indices], source_actions[sample_indices], atol=1e-4):
        raise AssertionError("Relative-action sanity check failed on sampled frames")

    return {
        "episodes": episode_count,
        "frames": int(target_actions.shape[0]),
        "stats": vector_stats(target_actions),
    }


def print_stats(summary: dict) -> None:
    print(f"Sanity check passed: episodes={summary['episodes']} frames={summary['frames']}")
    print("Delta stats per joint (pct):")
    stats = summary["stats"]
    for index, joint in enumerate(JOINT_NAMES):
        print(
            "  "
            f"{joint:14s} "
            f"mean={stats['mean'][index]: .6f} "
            f"std={stats['std'][index]: .6f} "
            f"min={stats['min'][index]: .6f} "
            f"max={stats['max'][index]: .6f} "
            f"q01={stats['q01'][index]: .6f} "
            f"q99={stats['q99'][index]: .6f}"
        )


def build_dataset(source: str, target: str, *, dry_run: bool) -> Path:
    source_root = lerobot_cache_root(source)
    target_root = lerobot_cache_root(target)
    print(f"Source: {source} ({source_root})")
    print(f"Target: {target} ({target_root})")

    validate_source(source, source_root)
    source_actions, source_states, _ = load_actions_and_states(source_root)
    dry_run_deltas = (source_actions - source_states).astype(np.float32)
    dry_run_summary = {
        "episodes": EXPECTED_EPISODES,
        "frames": int(dry_run_deltas.shape[0]),
        "stats": vector_stats(dry_run_deltas),
    }

    if dry_run:
        print("Dry run: validated source and computed relative-action stats; no files written.")
        print_stats(dry_run_summary)
        return target_root

    if target_root.exists() and any(target_root.iterdir()):
        if (target_root / "meta" / "info.json").exists() and (target_root / "data").exists():
            print("Target dataset already exists; leaving files unchanged and validating existing target.")
            return target_root
        raise FileExistsError(
            f"Target root already exists and does not look complete: {target_root}. "
            "Leaving it in place as requested."
        )

    target_root.mkdir(parents=True, exist_ok=True)
    copy_tree_contents(source_root, target_root, skip_top_level={"data", "meta"})
    deltas = rewrite_data_parquets(source_root, target_root)
    copy_meta_with_updated_action_stats(source_root, target_root, deltas)
    print(f"Wrote relative-action dataset to {target_root}")
    return target_root


def push_dataset(target: str, target_root: Path) -> None:
    dataset = LeRobotDataset(target, root=target_root, video_backend="pyav")
    try:
        dataset.push_to_hub(upload_large_folder=True)
    finally:
        finalize = getattr(dataset, "finalize", None)
        if callable(finalize):
            finalize()
    print(f"Pushed to https://huggingface.co/datasets/{target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Source LeRobot dataset repo id")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target LeRobot dataset repo id")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print stats without writing")
    parser.add_argument("--push", action="store_true", help="Push the target dataset to HuggingFace Hub")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_root = build_dataset(args.source, args.target, dry_run=args.dry_run)

    if not args.dry_run:
        summary = sanity_check(lerobot_cache_root(args.source), target_root)
        print_stats(summary)

    if args.push:
        if args.dry_run:
            print("Skipping push because --dry-run was set.")
        else:
            push_dataset(args.target, target_root)
    else:
        print("skipped push, run again with --push")


if __name__ == "__main__":
    main()
