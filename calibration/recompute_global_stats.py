"""Audit and repair LeRobot dataset action/state stats using true global quantiles.

LeRobot v3.0 aggregates dataset quantiles as a weighted mean of per-episode
quantiles. For multimodal robot joints, that can badly understate the true
dataset q01/q99 tails used by PI0.5's QUANTILES normalization. This script
computes global stats directly from every row in data/chunk-*/file-*.parquet
and can patch meta/stats.json for action and observation.state.

Examples:
    uv run python calibration/recompute_global_stats.py \
        --repo-id RevanthGundala/003-pour-water --dry-run

    uv run python calibration/recompute_global_stats.py \
        --repo-id RevanthGundala/003-pour-water \
        --output-root outputs/datasets/003-pour-water-globalstats \
        --write

    uv run python calibration/recompute_global_stats.py \
        --root outputs/datasets/003-pour-water-globalstats \
        --push --target-repo-id RevanthGundala/003-pour-water-globalstats
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.io_utils import load_info
from lerobot.datasets.io_utils import load_stats, write_stats

from utils.config import JOINT_NAMES

FEATURES_TO_REPAIR = ("action", "observation.state")
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def _resolve_root(args: argparse.Namespace) -> Path:
    if args.root is not None:
        return args.root

    if args.repo_id is None:
        raise ValueError("Either --repo-id or --root is required.")

    if args.output_root is not None:
        if args.output_root.exists():
            if not args.force:
                raise FileExistsError(
                    f"{args.output_root} already exists. Pass --force to replace it."
                )
            shutil.rmtree(args.output_root)
        print(f"Downloading full dataset {args.repo_id} -> {args.output_root}")
        return Path(
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                local_dir=args.output_root,
            )
        )

    print(f"Resolving {args.repo_id} data/meta only...")
    return Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=["data/**/*.parquet", "meta/**"],
        )
    )


def _load_feature(root: Path, feature: str) -> np.ndarray:
    arrays: list[np.ndarray] = []
    parquet_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No data parquets found under {root / 'data'}")

    for path in parquet_files:
        df = pd.read_parquet(path, columns=[feature])
        values = df[feature].to_numpy()
        if len(values) == 0:
            continue
        if hasattr(values[0], "__len__"):
            arrays.append(np.stack(values).astype(np.float64))
        else:
            arrays.append(values.astype(np.float64).reshape(-1, 1))

    if not arrays:
        raise RuntimeError(f"No values loaded for {feature!r}")
    return np.concatenate(arrays, axis=0)


def _global_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.array([values.shape[0]], dtype=np.int64),
        "q01": np.quantile(values, 0.01, axis=0),
        "q10": np.quantile(values, 0.10, axis=0),
        "q50": np.quantile(values, 0.50, axis=0),
        "q90": np.quantile(values, 0.90, axis=0),
        "q99": np.quantile(values, 0.99, axis=0),
    }


def _load_episode_metadata(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata parquets found under {root / 'meta' / 'episodes'}")
    episodes = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    if episodes["episode_index"].duplicated().any():
        dupes = episodes.loc[episodes["episode_index"].duplicated(keep=False), "episode_index"].tolist()
        raise ValueError(f"Duplicate episode metadata rows: {dupes[:20]}")
    return episodes


def _count_data_rows(root: Path) -> int:
    total = 0
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        total += pq.read_metadata(path).num_rows
    return total


def _prune_data_to_episode_metadata(root: Path) -> None:
    """Rewrite data parquets so they contain exactly the episodes listed in metadata.

    Some failed append/repair flows can leave stale duplicate episode rows at the
    end of a data file. LeRobot's reader sees rows in parquet order, so stale rows
    can be trained on even when meta/info.json reports fewer frames. This function
    keeps only the episodes assigned to each data file in meta/episodes and rewrites
    the global `index` column to be contiguous.
    """

    info = load_info(root)
    fps = int(info["fps"])
    episodes = _load_episode_metadata(root)
    expected_total = int(info["total_frames"])
    expected_from_lengths = int(episodes["length"].sum())
    if expected_total != expected_from_lengths:
        raise ValueError(
            f"meta/info.json total_frames={expected_total}, but episode lengths sum to "
            f"{expected_from_lengths}"
        )

    before_rows = _count_data_rows(root)
    if before_rows == expected_total:
        print(f"Data row count already matches metadata: {before_rows}")
    else:
        print(f"Pruning data rows from {before_rows} to metadata total {expected_total}")

    data_dir = root / "data"
    rewritten_rows = 0
    for (chunk_idx, file_idx), file_episodes in episodes.groupby(
        ["data/chunk_index", "data/file_index"], sort=True
    ):
        path = data_dir / f"chunk-{int(chunk_idx):03d}" / f"file-{int(file_idx):03d}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Metadata references missing data file: {path}")

        source = pd.read_parquet(path)
        kept: list[pd.DataFrame] = []
        for _, ep in file_episodes.sort_values("episode_index").iterrows():
            ep_idx = int(ep["episode_index"])
            length = int(ep["length"])
            rows = source[source["episode_index"] == ep_idx].copy()
            if len(rows) != length:
                raise ValueError(
                    f"{path.relative_to(root)} episode {ep_idx} has {len(rows)} rows, "
                    f"metadata expects {length}"
                )

            start = int(ep["dataset_from_index"])
            rows["episode_index"] = ep_idx
            if "frame_index" in rows.columns:
                rows["frame_index"] = np.arange(length, dtype=np.int64)
            if "index" in rows.columns:
                rows["index"] = np.arange(start, start + length, dtype=np.int64)
            if "timestamp" in rows.columns:
                rows["timestamp"] = np.arange(length, dtype=np.float64) / float(fps)
            kept.append(rows)

        out = pd.concat(kept, ignore_index=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        out.to_parquet(tmp, index=False)
        tmp.replace(path)
        rewritten_rows += len(out)
        print(f"  rewrote {path.relative_to(root)} rows={len(out)}")

    after_rows = _count_data_rows(root)
    if after_rows != expected_total or rewritten_rows != expected_total:
        raise AssertionError(
            f"Prune failed: after_rows={after_rows}, rewritten_rows={rewritten_rows}, "
            f"expected={expected_total}"
        )


def _feature_names(feature: str, dim: int) -> list[str]:
    if dim == len(JOINT_NAMES) and feature in FEATURES_TO_REPAIR:
        return JOINT_NAMES
    return [f"dim_{i}" for i in range(dim)]


def _stat_array(stats: dict, feature: str, key: str, dim: int) -> np.ndarray | None:
    if stats is None or feature not in stats or key not in stats[feature]:
        return None
    arr = np.asarray(stats[feature][key], dtype=np.float64).reshape(-1)
    if key == "count":
        return arr
    if arr.size != dim:
        raise ValueError(f"{feature}.{key} has shape {arr.shape}; expected {dim} values")
    return arr


def _print_comparison(feature: str, current: dict | None, repaired: dict[str, np.ndarray]) -> bool:
    dim = int(repaired["mean"].reshape(-1).shape[0])
    names = _feature_names(feature, dim)
    mismatch = False

    print(f"\n=== {feature} ===")
    print(f"global count: {int(repaired['count'][0])}")
    for stat_key in ("q01", "q50", "q99", "min", "max"):
        cur = _stat_array(current, feature, stat_key, dim)
        new = repaired[stat_key].reshape(-1)
        if cur is None:
            print(f"  current {stat_key}: <missing>")
            mismatch = True
            continue
        max_abs = float(np.max(np.abs(new - cur)))
        denom = np.maximum(np.abs(new), 1.0)
        max_rel = float(np.max(np.abs(new - cur) / denom))
        if max_abs > 1e-3:
            mismatch = True
        print(f"\n  {stat_key}: max_abs_delta={max_abs:.3f} max_rel_delta={max_rel:.3f}")
        print(f"    {'joint':18s} {'current':>10s} {'global':>10s} {'delta':>10s}")
        for name, old, val in zip(names, cur, new, strict=False):
            delta = val - old
            marker = " *" if abs(delta) > 1.0 else ""
            print(f"    {name:18s} {old:10.3f} {val:10.3f} {delta:10.3f}{marker}")

    return mismatch


def _repair_stats(root: Path, repaired_blocks: dict[str, dict[str, np.ndarray]]) -> None:
    current = load_stats(root) or {}
    for feature, block in repaired_blocks.items():
        current[feature] = block
    write_stats(current, root)


def _validate_metadata(root: Path, repo_id: str | None) -> None:
    # LeRobotDatasetMetadata wants a repo id for cache-aware paths, but root
    # lets it read the local metadata we just wrote.
    metadata_repo = repo_id or root.name
    meta = LeRobotDatasetMetadata(metadata_repo, root=root)
    print(
        f"\nMetadata validated: episodes={meta.total_episodes} "
        f"frames={meta.total_frames} fps={meta.fps}"
    )


def _push_dataset(root: Path, target_repo_id: str) -> None:
    api = HfApi()
    print(f"\nCreating/verifying dataset repo {target_repo_id}")
    api.create_repo(repo_id=target_repo_id, repo_type="dataset", exist_ok=True)
    print(f"Uploading {root} -> hf://datasets/{target_repo_id}")
    api.upload_folder(
        repo_id=target_repo_id,
        repo_type="dataset",
        folder_path=str(root),
        commit_message="Repair action/state stats with true global quantiles",
    )
    print(f"Pushed repaired dataset: https://huggingface.co/datasets/{target_repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo-id", help="Hugging Face dataset repo id to audit/download.")
    source.add_argument("--root", type=Path, help="Local LeRobot dataset root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Download a full mutable copy here before writing/pushing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Audit only; do not write.")
    parser.add_argument("--write", action="store_true", help="Patch meta/stats.json in the resolved root.")
    parser.add_argument(
        "--prune-to-meta",
        action="store_true",
        help="Before computing stats, rewrite data parquets to exactly match meta/episodes.",
    )
    parser.add_argument("--force", action="store_true", help="Replace --output-root if it exists.")
    parser.add_argument("--push", action="store_true", help="Upload repaired root to Hugging Face.")
    parser.add_argument("--target-repo-id", help="Destination dataset repo id for --push.")
    args = parser.parse_args()

    if args.push and not args.target_repo_id:
        raise ValueError("--push requires --target-repo-id")
    if args.push and not args.write:
        raise ValueError("--push requires --write so uploaded stats are repaired")
    if args.root is not None and args.output_root is not None:
        raise ValueError("--output-root is only valid with --repo-id")

    root = _resolve_root(args).resolve()
    print(f"Dataset root: {root}")

    if args.prune_to_meta:
        if not args.write:
            raise ValueError("--prune-to-meta modifies data files and therefore requires --write")
        _prune_data_to_episode_metadata(root)

    current_stats = load_stats(root)
    if current_stats is None:
        raise FileNotFoundError(f"No meta/stats.json found under {root}")

    repaired: dict[str, dict[str, np.ndarray]] = {}
    any_mismatch = False
    for feature in FEATURES_TO_REPAIR:
        values = _load_feature(root, feature)
        block = _global_stats(values)
        repaired[feature] = block
        any_mismatch = _print_comparison(feature, current_stats, block) or any_mismatch

    if not any_mismatch:
        print("\nNo metadata mismatch detected for action/observation.state.")

    if args.write:
        _repair_stats(root, repaired)
        print(f"\nRewrote {root / 'meta' / 'stats.json'}")
        _validate_metadata(root, args.repo_id)
        # Re-load and compare again so a write bug is caught immediately.
        updated = load_stats(root)
        for feature, block in repaired.items():
            for key in STAT_KEYS:
                got = np.asarray(updated[feature][key]).reshape(-1)
                want = np.asarray(block[key]).reshape(-1)
                if not np.allclose(got, want):
                    raise AssertionError(f"Written stats mismatch for {feature}.{key}")
        print("Written stats verified against raw global stats.")
    else:
        print("\nDry run only; no files were modified.")

    if args.push:
        _push_dataset(root, args.target_repo_id)


if __name__ == "__main__":
    main()
