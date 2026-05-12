"""Create a LeRobot v3 episode subset with contiguous episode indexes.

This is intentionally conservative: it copies only selected data rows into a
fresh parquet, preserves source episode lineage, and copies the referenced
video files without re-encoding them.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata


def _default_lerobot_root(repo_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id


def _parse_episode_spec(spec: str) -> list[int]:
    episodes: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            if end < start:
                raise ValueError(f"Invalid episode range: {part}")
            episodes.update(range(start, end + 1))
        else:
            episodes.add(int(part))
    return sorted(episodes)


def _load_episodes(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata parquets found under {root / 'meta' / 'episodes'}")
    episodes = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    return episodes.sort_values("episode_index").reset_index(drop=True)


def _copy_video_files(source_root: Path, target_root: Path, episode_rows: pd.DataFrame) -> None:
    video_columns = [
        column
        for column in episode_rows.columns
        if column.startswith("videos/") and column.endswith("/chunk_index")
    ]
    for chunk_col in video_columns:
        prefix = chunk_col.removesuffix("/chunk_index")
        file_col = f"{prefix}/file_index"
        if file_col not in episode_rows.columns:
            continue
        video_key = prefix.removeprefix("videos/")
        for _, row in episode_rows[[chunk_col, file_col]].drop_duplicates().iterrows():
            chunk_index = int(row[chunk_col])
            file_index = int(row[file_col])
            rel = (
                Path("videos")
                / video_key
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            src = source_root / rel
            dst = target_root / rel
            if not src.exists():
                raise FileNotFoundError(f"Metadata references missing video file: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)


def _load_selected_data(source_root: Path, source_to_new: dict[int, int], fps: int) -> pd.DataFrame:
    selected = []
    wanted = set(source_to_new)
    for path in sorted((source_root / "data").glob("chunk-*/file-*.parquet")):
        data = pd.read_parquet(path)
        data = data[data["episode_index"].isin(wanted)].copy()
        if not data.empty:
            selected.append(data)
    if not selected:
        raise RuntimeError(f"No data rows found for selected episodes: {sorted(wanted)}")
    data = pd.concat(selected, ignore_index=True)

    remapped = []
    global_index = 0
    for source_episode, new_episode in source_to_new.items():
        rows = data[data["episode_index"] == source_episode].copy().sort_values("frame_index")
        if rows.empty:
            raise RuntimeError(f"No data rows found for source episode {source_episode}")
        length = len(rows)
        rows["episode_index"] = np.full(length, new_episode, dtype=np.int64)
        rows["frame_index"] = np.arange(length, dtype=np.int64)
        rows["index"] = np.arange(global_index, global_index + length, dtype=np.int64)
        rows["timestamp"] = rows["frame_index"].astype(np.float32) / float(fps)
        global_index += length
        remapped.append(rows)

    return pd.concat(remapped, ignore_index=True)


def _build_episode_metadata(
    source_rows: pd.DataFrame,
    subset_data: pd.DataFrame,
    source_to_new: dict[int, int],
) -> pd.DataFrame:
    rows = []
    dataset_from = 0
    for source_episode, new_episode in source_to_new.items():
        source = source_rows[source_rows["episode_index"] == source_episode]
        if len(source) != 1:
            raise RuntimeError(f"Expected one metadata row for episode {source_episode}, found {len(source)}")
        row = source.iloc[0].copy()
        length = int((subset_data["episode_index"] == new_episode).sum())
        row["source_episode_index"] = source_episode
        row["episode_index"] = new_episode
        row["length"] = length
        row["data/chunk_index"] = 0
        row["data/file_index"] = 0
        row["dataset_from_index"] = dataset_from
        row["dataset_to_index"] = dataset_from + length
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        dataset_from += length
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _write_info(source_root: Path, target_root: Path, total_episodes: int, total_frames: int) -> None:
    info = json.loads((source_root / "meta" / "info.json").read_text(encoding="utf-8"))
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{total_episodes}"}
    (target_root / "meta").mkdir(parents=True, exist_ok=True)
    (target_root / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def create_subset(
    source_root: Path,
    output_root: Path,
    episodes: Iterable[int],
    force: bool = False,
) -> None:
    episodes = list(episodes)
    if output_root.exists():
        if not force:
            raise FileExistsError(f"{output_root} already exists; pass --force to replace it")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    source_episode_rows = _load_episodes(source_root)
    available = set(int(v) for v in source_episode_rows["episode_index"].tolist())
    missing = [episode for episode in episodes if episode not in available]
    if missing:
        raise ValueError(f"Requested episodes are missing from metadata: {missing}")

    source_to_new = {source_episode: idx for idx, source_episode in enumerate(episodes)}
    selected_episode_rows = source_episode_rows[source_episode_rows["episode_index"].isin(episodes)].copy()
    source_info = json.loads((source_root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = int(source_info["fps"])

    print(f"Creating subset from {source_root}")
    print(f"  source episodes: {episodes}")
    print(f"  output root: {output_root}")

    subset_data = _load_selected_data(source_root, source_to_new, fps=fps)
    episode_metadata = _build_episode_metadata(selected_episode_rows, subset_data, source_to_new)

    data_path = output_root / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    subset_data.to_parquet(data_path, index=False)

    episodes_path = output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episode_metadata.to_parquet(episodes_path, index=False)

    shutil.copy2(source_root / "meta" / "tasks.parquet", output_root / "meta" / "tasks.parquet")
    shutil.copy2(source_root / "meta" / "stats.json", output_root / "meta" / "stats.json")
    before_stats = source_root / "meta" / "stats.before-absolute-recompute.json"
    if before_stats.exists():
        shutil.copy2(before_stats, output_root / "meta" / before_stats.name)

    _copy_video_files(source_root, output_root, selected_episode_rows)
    _write_info(
        source_root,
        output_root,
        total_episodes=len(episodes),
        total_frames=len(subset_data),
    )
    _validate_subset(output_root, episodes)


def _validate_subset(root: Path, source_episodes: list[int]) -> None:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes = _load_episodes(root)
    row_count = sum(pq.read_metadata(path).num_rows for path in (root / "data").glob("chunk-*/file-*.parquet"))
    if int(info["total_episodes"]) != len(source_episodes):
        raise AssertionError("info.json total_episodes mismatch")
    if len(episodes) != len(source_episodes):
        raise AssertionError("episode metadata row count mismatch")
    if episodes["episode_index"].tolist() != list(range(len(source_episodes))):
        raise AssertionError("subset episode indexes are not contiguous")
    if episodes["source_episode_index"].astype(int).tolist() != source_episodes:
        raise AssertionError("source_episode_index lineage mismatch")
    if int(info["total_frames"]) != int(episodes["length"].sum()) or row_count != int(info["total_frames"]):
        raise AssertionError(
            f"frame count mismatch: info={info['total_frames']} episodes={episodes['length'].sum()} rows={row_count}"
        )
    LeRobotDatasetMetadata(root.name, root=root)
    print(
        f"Subset validated: episodes={info['total_episodes']} frames={info['total_frames']} "
        f"source={source_episodes[0]}-{source_episodes[-1]}"
    )


def _push_dataset(root: Path, repo_id: str) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(root),
        commit_message="Create episode subset dataset",
    )
    print(f"Pushed subset dataset: https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo-id", default="RevanthGundala/003-pour-water")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--episodes", required=True, help="Comma/range spec, e.g. 50-69,75-89")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--target-repo-id")
    args = parser.parse_args()

    if args.push and not args.target_repo_id:
        raise ValueError("--push requires --target-repo-id")

    source_root = args.source_root or _default_lerobot_root(args.source_repo_id)
    episodes = _parse_episode_spec(args.episodes)
    create_subset(source_root.resolve(), args.output_root.resolve(), episodes, force=args.force)
    if args.push:
        _push_dataset(args.output_root.resolve(), args.target_repo_id)


if __name__ == "__main__":
    main()
