"""
smoke_test.py — Lightweight preflight for ACT training and Vast.ai launch.

Default checks avoid downloading the full dataset:
  1. Validate the dataset metadata against the Hugging Face repo tree.
  2. Validate the train.py configuration on CPU.
  3. Validate the Vast.ai SDK surface and generated onstart script.

Optional:
  --train-step   Run a full 1-step CPU training smoke test. This downloads the
                 dataset files needed by LeRobot.

Usage:
    uv run python smoke_test.py
    uv run python smoke_test.py --train-step
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import site
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from vastai_sdk import VastAI


REQUIRED_VAST_METHODS = (
    "attach_ssh",
    "destroy_instance",
    "launch_instance",
    "logs",
    "scp_url",
    "show_instance",
)


class SmokeTestFailure(RuntimeError):
    """Raised when the preflight detects a blocker."""


def _status(message: str) -> None:
    print(f"[smoke] {message}")


def _episode_index(path: str) -> int:
    return int(Path(path).stem.split("_")[-1])


def _candidate_macos_ffmpeg_dirs() -> list[Path]:
    candidates: list[Path] = []

    for cellar_root in (Path("/opt/homebrew/Cellar/ffmpeg@7"), Path("/usr/local/Cellar/ffmpeg@7")):
        if not cellar_root.exists():
            continue
        versioned_lib_dirs = sorted(cellar_root.glob("*/lib"), reverse=True)
        for lib_dir in versioned_lib_dirs:
            if any(lib_dir.glob("libavutil.59*.dylib")):
                candidates.append(lib_dir)
                break

    site_packages = []
    try:
        site_packages.extend(site.getsitepackages())
    except AttributeError:
        pass

    user_site = site.getusersitepackages()
    if user_site:
        site_packages.append(user_site)

    for site_dir in site_packages:
        dylib_dir = Path(site_dir) / "cv2" / ".dylibs"
        if any(dylib_dir.glob("libavutil.59*.dylib")):
            candidates.append(dylib_dir)

    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate.exists() and candidate not in unique_candidates:
            unique_candidates.append(candidate)

    return unique_candidates


def validate_dataset_listing(repo_id: str, revision: str | None) -> None:
    api = HfApi()
    entries = list(
        api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            recursive=True,
            expand=True,
            revision=revision,
        )
    )

    parquet_paths = sorted(
        entry.path for entry in entries if getattr(entry, "path", "").endswith(".parquet")
    )
    video_paths = sorted(
        entry.path for entry in entries if getattr(entry, "path", "").endswith(".mp4")
    )
    total_bytes = sum((getattr(entry, "size", 0) or 0) for entry in entries)

    episodes_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="meta/episodes.jsonl",
        revision=revision,
    )
    info_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="meta/info.json",
        revision=revision,
    )

    episodes = [
        json.loads(line) for line in Path(episodes_path).read_text().splitlines() if line.strip()
    ]
    info = json.loads(Path(info_path).read_text())

    expected_indexes = list(range(len(episodes)))
    episode_indexes = [episode["episode_index"] for episode in episodes]
    parquet_indexes = [_episode_index(path) for path in parquet_paths]
    video_indexes = [_episode_index(path) for path in video_paths]
    metadata_total_frames = sum(int(episode["length"]) for episode in episodes)

    problems: list[str] = []

    if episode_indexes != expected_indexes:
        problems.append("episodes.jsonl episode indexes are not contiguous")
    if info.get("total_episodes") != len(episodes):
        problems.append(
            f"info.json says total_episodes={info.get('total_episodes')}, but episodes.jsonl has {len(episodes)} entries"
        )
    if info.get("total_frames") != metadata_total_frames:
        problems.append(
            f"info.json says total_frames={info.get('total_frames')}, but episodes.jsonl sums to {metadata_total_frames}"
        )

    extra_parquet = sorted(set(parquet_indexes) - set(expected_indexes))
    missing_parquet = sorted(set(expected_indexes) - set(parquet_indexes))
    extra_videos = sorted(set(video_indexes) - set(expected_indexes))
    missing_videos = sorted(set(expected_indexes) - set(video_indexes))

    if missing_parquet:
        problems.append(f"missing parquet episodes: {missing_parquet}")
    if extra_parquet:
        problems.append(f"extra parquet episodes still uploaded: {extra_parquet}")
    if video_paths and missing_videos:
        problems.append(f"missing video episodes: {missing_videos}")
    if video_paths and extra_videos:
        problems.append(f"extra video episodes still uploaded: {extra_videos}")

    if problems:
        size_mb = round(total_bytes / (1024 * 1024), 2)
        raise SmokeTestFailure(
            f"Dataset repo is inconsistent ({size_mb} MB total). " + "; ".join(problems) + "."
        )

    _status(
        f"Dataset listing looks consistent for revision {revision or 'main'}: {len(episodes)} episodes, {metadata_total_frames} frames, {round(total_bytes / (1024 * 1024), 2)} MB"
    )


def validate_train_config() -> None:
    train = importlib.import_module("train")
    temp_root = Path(tempfile.mkdtemp(prefix="omx-train-config-"))

    dataset_config = train.DatasetConfig(repo_id=train.DATASET_REPO_ID)
    act_config = train.ACTConfig(
        input_features={},
        output_features={},
        device="cpu",
        chunk_size=train.CHUNK_SIZE,
        n_action_steps=train.CHUNK_SIZE,
        use_vae=train.USE_VAE,
        kl_weight=train.KL_WEIGHT,
        optimizer_lr=train.LEARNING_RATE,
        optimizer_lr_backbone=train.LEARNING_RATE,
        vision_backbone=train.VISION_BACKBONE,
        repo_id="smoke-test-act-policy",
    )
    train_cfg = train.TrainPipelineConfig(
        dataset=dataset_config,
        policy=act_config,
        output_dir=temp_root / "outputs",
        batch_size=1,
        num_workers=0,
        steps=1,
        log_freq=1,
        save_freq=1,
        seed=train.SEED,
        eval_freq=-1,
    )

    try:
        train_cfg.validate()
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    _status("train.py configuration validates on CPU")


def validate_vast_launcher() -> None:
    train_vastai = importlib.import_module("train_vastai")

    missing_methods = [name for name in REQUIRED_VAST_METHODS if not hasattr(VastAI, name)]
    if missing_methods:
        raise SmokeTestFailure(
            f"Installed vastai-sdk is missing methods used by train_vastai.py: {', '.join(missing_methods)}"
        )

    onstart = train_vastai._build_onstart_script("dGVzdA==")
    required_snippets = (
        "=== CUDA PREFLIGHT ===",
        "torch.cuda.is_available()",
        "git clone https://github.com/ROBOTIS-GIT/lerobot.git",
        'pip install --no-cache-dir -e ".[dynamixel]"',
        "export HF_TOKEN=",
        "export OMX_REQUIRE_CUDA=1",
        "base64 -d > train.py",
        "python train.py",
    )
    missing_snippets = [snippet for snippet in required_snippets if snippet not in onstart]
    if missing_snippets:
        raise SmokeTestFailure(
            "train_vastai.py generated an unexpected onstart script. Missing: "
            + ", ".join(repr(snippet) for snippet in missing_snippets)
        )

    original_gpu_name = os.environ.get("OMX_GPU_NAME")
    original_vast_image = os.environ.get("OMX_VAST_IMAGE")
    try:
        os.environ["OMX_GPU_NAME"] = "RTX_3090"
        os.environ["OMX_VAST_IMAGE"] = "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"
        train_vastai = importlib.reload(train_vastai)
        if train_vastai.GPU_NAME != "RTX_3090":
            raise SmokeTestFailure("train_vastai.py did not honor OMX_GPU_NAME")
        if train_vastai.VAST_IMAGE != "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel":
            raise SmokeTestFailure("train_vastai.py did not honor OMX_VAST_IMAGE")
    finally:
        if original_gpu_name is None:
            os.environ.pop("OMX_GPU_NAME", None)
        else:
            os.environ["OMX_GPU_NAME"] = original_gpu_name

        if original_vast_image is None:
            os.environ.pop("OMX_VAST_IMAGE", None)
        else:
            os.environ["OMX_VAST_IMAGE"] = original_vast_image

        importlib.reload(train_vastai)

    _status("Vast launcher looks consistent with the installed SDK")


def _run_train_step_in_process() -> None:
    train = importlib.import_module("train")
    temp_root = Path(tempfile.mkdtemp(prefix="omx-train-step-"))
    output_dir = temp_root / "outputs"
    dataset_root = temp_root / "dataset"
    original_dataset_root = train.DATASET_ROOT

    train.DEVICE = "cpu"
    train.BATCH_SIZE = 1
    train.NUM_WORKERS = 0
    train.TRAINING_STEPS = 1
    train.LOG_FREQ = 1
    train.SAVE_FREQ = 1
    train.OUTPUT_DIR = output_dir
    train.DATASET_ROOT = dataset_root

    try:
        train.main()
    except Exception as exc:
        raise SmokeTestFailure(
            f"1-step train.py smoke test failed on CPU: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        train.DATASET_ROOT = original_dataset_root
        shutil.rmtree(temp_root, ignore_errors=True)

    _status("1-step train.py smoke test passed")


def run_train_step() -> None:
    if platform.system() != "Darwin" or os.environ.get("_OMX_SMOKE_DYLD_READY") == "1":
        _run_train_step_in_process()
        return

    ffmpeg_dirs = _candidate_macos_ffmpeg_dirs()
    if not ffmpeg_dirs:
        _run_train_step_in_process()
        return

    env = os.environ.copy()
    dyld_dirs = [str(path) for path in ffmpeg_dirs]
    existing = env.get("DYLD_FALLBACK_LIBRARY_PATH")
    if existing:
        dyld_dirs.append(existing)
    env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(dyld_dirs)
    env["_OMX_SMOKE_DYLD_READY"] = "1"

    result = subprocess.run(
        [sys.executable, __file__, "--train-step-subprocess"],
        cwd=str(Path(__file__).parent),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise SmokeTestFailure(
            f"1-step train.py smoke test failed in subprocess (exit code {result.returncode})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight ACT training before launching on Vast.ai.")
    parser.add_argument(
        "--train-step",
        action="store_true",
        help="Also run a full 1-step CPU training smoke test (downloads dataset files).",
    )
    parser.add_argument(
        "--train-step-subprocess",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train = importlib.import_module("train")

    if args.train_step_subprocess:
        try:
            _run_train_step_in_process()
        except SmokeTestFailure as exc:
            print(f"[smoke] FAIL: {exc}")
            return 1
        return 0

    checks = [
        (
            "dataset listing",
            lambda: validate_dataset_listing(train.DATASET_REPO_ID, getattr(train, "DATASET_REVISION", None)),
        ),
        ("train config", validate_train_config),
        ("Vast launcher", validate_vast_launcher),
    ]
    if args.train_step:
        checks.append(("train step", run_train_step))

    for label, check in checks:
        _status(f"Running {label} check")
        try:
            check()
        except SmokeTestFailure as exc:
            print(f"[smoke] FAIL: {exc}")
            return 1

    print("[smoke] PASS: preflight succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
