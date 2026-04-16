"""Fix the Hugging Face dataset metadata and prune stale episode files.

Usage:
    export HF_TOKEN="your-write-token"
    uv run python fix_metadata.py
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download

from control_utils import get_hf_token


REPO_ID = "RevanthGundala/pick_up_packet_test"

EPISODES = [
    {"episode_index": 0, "tasks": ["Pick up packet"], "length": 597},
    {"episode_index": 1, "tasks": ["Pick up packet"], "length": 103},
    {"episode_index": 2, "tasks": ["Pick up packet"], "length": 164},
    {"episode_index": 3, "tasks": ["Pick up packet"], "length": 719},
    {"episode_index": 4, "tasks": ["Pick up packet"], "length": 121},
    {"episode_index": 5, "tasks": ["Pick up packet"], "length": 182},
    {"episode_index": 6, "tasks": ["Pick up packet"], "length": 173},
    {"episode_index": 7, "tasks": ["Pick up packet"], "length": 123},
    {"episode_index": 8, "tasks": ["Pick up packet"], "length": 155},
    {"episode_index": 9, "tasks": ["Pick up packet"], "length": 53},
    {"episode_index": 10, "tasks": ["Pick up packet"], "length": 74},
    {"episode_index": 11, "tasks": ["Pick up packet"], "length": 57},
    {"episode_index": 12, "tasks": ["Pick up packet"], "length": 265},
    {"episode_index": 13, "tasks": ["Pick up packet"], "length": 281},
    {"episode_index": 14, "tasks": ["Pick up packet"], "length": 318},
    {"episode_index": 15, "tasks": ["Pick up packet"], "length": 176},
    {"episode_index": 16, "tasks": ["Pick up packet"], "length": 848},
    {"episode_index": 17, "tasks": ["Pick up packet"], "length": 333},
    {"episode_index": 18, "tasks": ["Pick up packet"], "length": 65},
    {"episode_index": 19, "tasks": ["Pick up packet"], "length": 240},
    {"episode_index": 20, "tasks": ["Pick up packet"], "length": 45},
    {"episode_index": 21, "tasks": ["Pick up packet"], "length": 57},
    {"episode_index": 22, "tasks": ["Pick up packet"], "length": 62},
    {"episode_index": 23, "tasks": ["Pick up packet"], "length": 36},
    {"episode_index": 24, "tasks": ["Pick up packet"], "length": 109},
    {"episode_index": 25, "tasks": ["Pick up packet"], "length": 796},
    {"episode_index": 26, "tasks": ["Pick up packet"], "length": 81},
    {"episode_index": 27, "tasks": ["Pick up packet"], "length": 233},
    {"episode_index": 28, "tasks": ["Pick up packet"], "length": 438},
    {"episode_index": 29, "tasks": ["Pick up packet"], "length": 90},
    {"episode_index": 30, "tasks": ["Pick up packet"], "length": 597},
]

VALID_EPISODE_INDEXES = {episode["episode_index"] for episode in EPISODES}


def _build_info() -> dict:
    total_frames = sum(episode["length"] for episode in EPISODES)
    return {
        "codebase_version": "v2.1",
        "robot_type": "omx_follower",
        "total_episodes": len(EPISODES),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(EPISODES),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": f"0:{len(EPISODES)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [6],
                "names": [
                    "shoulder_pan.pos",
                    "shoulder_lift.pos",
                    "elbow_flex.pos",
                    "wrist_flex.pos",
                    "wrist_roll.pos",
                    "gripper.pos",
                ],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [6],
                "names": [
                    "shoulder_pan.pos",
                    "shoulder_lift.pos",
                    "elbow_flex.pos",
                    "wrist_flex.pos",
                    "wrist_roll.pos",
                    "gripper.pos",
                ],
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": 30,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def _build_episodes_jsonl() -> str:
    return "".join(json.dumps(episode) + "\n" for episode in EPISODES)


def _episode_index_from_path(path_in_repo: str) -> int | None:
    stem = Path(path_in_repo).stem
    if not stem.startswith("episode_"):
        return None
    suffix = stem.removeprefix("episode_")
    return int(suffix) if suffix.isdigit() else None


def _find_stale_episode_paths(api: HfApi, token: str) -> list[str]:
    stale_paths: list[str] = []
    entries = api.list_repo_tree(repo_id=REPO_ID, repo_type="dataset", recursive=True, token=token)
    for entry in entries:
        path_in_repo = getattr(entry, "path", "")
        is_episode_file = (
            path_in_repo.startswith("data/")
            and path_in_repo.endswith(".parquet")
            or path_in_repo.startswith("videos/")
            and path_in_repo.endswith(".mp4")
        )
        if not is_episode_file:
            continue

        episode_index = _episode_index_from_path(path_in_repo)
        if episode_index is not None and episode_index not in VALID_EPISODE_INDEXES:
            stale_paths.append(path_in_repo)

    return sorted(stale_paths)


def _metadata_matches_repo(desired_episodes_jsonl: str, desired_info: dict) -> bool:
    current_episodes_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename="meta/episodes.jsonl")
    current_info_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename="meta/info.json")

    current_episodes_jsonl = Path(current_episodes_path).read_text()
    current_info = json.loads(Path(current_info_path).read_text())

    return current_episodes_jsonl == desired_episodes_jsonl and current_info == desired_info


def main() -> None:
    token = get_hf_token()
    if not token:
        raise SystemExit(
            "ERROR: HF_TOKEN not set.\n"
            "  export HF_TOKEN='your-write-token'\n"
            "  or run: huggingface-cli login"
        )

    desired_episodes_jsonl = _build_episodes_jsonl()
    desired_info = _build_info()

    api = HfApi(token=token)
    stale_paths = _find_stale_episode_paths(api, token)
    metadata_matches = _metadata_matches_repo(desired_episodes_jsonl, desired_info)

    print(f"Expected episodes: {len(EPISODES)}")
    print(f"Stale remote files to delete: {len(stale_paths)}")

    if metadata_matches and not stale_paths:
        print("Dataset already matches the expected metadata and episode files.")
        return

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        episodes_path = tmpdir_path / "episodes.jsonl"
        info_path = tmpdir_path / "info.json"
        episodes_path.write_text(desired_episodes_jsonl)
        info_path.write_text(json.dumps(desired_info, indent=2))

        operations = [
            CommitOperationAdd(path_in_repo="meta/episodes.jsonl", path_or_fileobj=episodes_path),
            CommitOperationAdd(path_in_repo="meta/info.json", path_or_fileobj=info_path),
            *[CommitOperationDelete(path_in_repo=path) for path in stale_paths],
        ]

        commit_info = api.create_commit(
            repo_id=REPO_ID,
            repo_type="dataset",
            operations=operations,
            commit_message="Fix dataset metadata and prune stale episodes",
            commit_description=(
                f"Keep episodes 0-{len(EPISODES) - 1} in metadata and remove stale remote files: "
                + ", ".join(stale_paths)
            )
            if stale_paths
            else "Refresh dataset metadata to the curated 31-episode split.",
            token=token,
        )

    print("Committed dataset fix.")
    commit_url = getattr(commit_info, "commit_url", None)
    if commit_url:
        print(f"Commit URL: {commit_url}")


if __name__ == "__main__":
    main()
