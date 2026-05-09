"""
Run stock LeRobot PI0.5 fine-tuning on Modal.

This intentionally mirrors deploy/train_vastai.py's stock `lerobot-train`
command instead of using the older custom training/train_pi0.py Modal path.

Usage:
    modal run --detach deploy/train_modal_lerobot.py --profile pour_absolute_70_from_3k
    modal run --detach deploy/train_modal_lerobot.py --profile pour_absolute_new20_from_3k
"""

from __future__ import annotations

from dataclasses import asdict
import importlib
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import modal


REMOTE_WORKSPACE = Path("/workspace")
REMOTE_OUTPUTS = REMOTE_WORKSPACE / "outputs"
MODAL_GPU = "A100-80GB"

hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04", add_python="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu124",
    )
)

app = modal.App("omx-pi05-stock-lerobot-training", image=image)


def load_profile(profile: str):
    module = importlib.import_module(f"configs.train.{profile}")
    return module.config


def _bool_cli(value: bool) -> str:
    return str(value).lower()


def build_lerobot_train_command(config_dict: dict[str, Any], output_dir: str) -> list[str]:
    relative_exclude_joints = config_dict["relative_exclude_joints"]
    relative_exclude_joints_cli = "[" + ",".join(repr(j) for j in relative_exclude_joints) + "]"

    return [
        "lerobot-train",
        f"--dataset.repo_id={config_dict['dataset_repo_id']}",
        f"--policy.type={config_dict['policy_type']}",
        f"--output_dir={output_dir}",
        f"--job_name={config_dict['job_name']}",
        f"--policy.pretrained_path={config_dict['policy_pretrained_path']}",
        f"--policy.repo_id={config_dict['policy_repo_id']}",
        f"--policy.compile_model={_bool_cli(config_dict['compile_model'])}",
        f"--policy.gradient_checkpointing={_bool_cli(config_dict['gradient_checkpointing'])}",
        f"--policy.dtype={config_dict['dtype']}",
        f"--policy.freeze_vision_encoder={_bool_cli(config_dict['freeze_vision_encoder'])}",
        f"--policy.train_expert_only={_bool_cli(config_dict['train_expert_only'])}",
        f"--policy.use_relative_actions={_bool_cli(config_dict['use_relative_actions'])}",
        f"--policy.relative_exclude_joints={relative_exclude_joints_cli}",
        f"--steps={config_dict['steps']}",
        f"--batch_size={config_dict['batch_size']}",
        "--policy.device=cuda",
        f"--log_freq={config_dict['log_freq']}",
    ]


def format_command(command: list[str]) -> str:
    return " \\\n    ".join(shlex.quote(part) for part in command)


def preflight_profile(profile: str) -> None:
    import urllib.error
    import urllib.request

    config = load_profile(profile)
    config_dict = asdict(config)
    output_dir = str(REMOTE_OUTPUTS / config.job_name)
    command = build_lerobot_train_command(config_dict, output_dir)

    print(f"PROFILE {profile}")
    print(f"  modal_gpu={MODAL_GPU}")
    print(f"  dataset={config.dataset_repo_id}")
    print(f"  checkpoint={config.policy_pretrained_path}")
    print(f"  output_repo={config.policy_repo_id}")
    print(f"  batch_size={config.batch_size}")
    print(f"  steps={config.steps}")
    print(f"  compile_model={config.compile_model}")
    print(f"  use_relative_actions={config.use_relative_actions}")
    print("  command:")
    print(format_command(command))

    if config.policy_pretrained_path == "lerobot/pi05_base":
        raise ValueError(f"{profile} starts from lerobot/pi05_base; expected the current 3k checkpoint")
    if config.policy_pretrained_path != "RevanthGundala/pi05-pour-water-3k":
        raise ValueError(f"{profile} uses unexpected checkpoint: {config.policy_pretrained_path}")
    if config.use_relative_actions:
        raise ValueError(f"{profile} must be absolute-action, but use_relative_actions=True")
    if config.batch_size != 32:
        raise ValueError(f"{profile} changed batch_size to {config.batch_size}; expected 32")
    if config.compile_model:
        raise ValueError(f"{profile} has compile_model=True; expected False")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("  HF repo checks skipped locally because HF_TOKEN is not set; Modal uses the 'huggingface' secret remotely.")
        return

    for repo_id, repo_type in (
        (config.dataset_repo_id, "datasets"),
        (config.policy_pretrained_path, "models"),
    ):
        url = f"https://huggingface.co/api/{repo_type}/{repo_id}"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
                if response.status != 200:
                    raise RuntimeError(f"Unexpected HTTP {response.status} for {url}")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Hugging Face repo check failed for {repo_id}: HTTP {exc.code}") from exc


@app.function(gpu=MODAL_GPU, timeout=14400, memory=32768, secrets=[hf_secret])
def train_remote(config_dict: dict[str, Any]) -> str:
    import os
    import subprocess
    import sys

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    print("=== SETUP: installing stock LeRobot training environment ===")
    pip_install = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--progress-bar",
        "off",
        "--retries",
        "2",
        "--timeout",
        "120",
        "--no-cache-dir",
    ]
    subprocess.run([*pip_install, "lerobot[pi]>=0.5.1"], check=True)
    subprocess.run(
        [*pip_install, "--force-reinstall", "av>=15,<16"],
        check=True,
    )

    import torch
    from huggingface_hub import hf_hub_download, snapshot_download

    print("=== CUDA PREFLIGHT ===")
    print("Torch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Modal container.")
    print("Detected GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))

    dataset_repo_id = config_dict["dataset_repo_id"]
    dataset_revision = config_dict["dataset_revision"]
    dataset_root_name = f"{dataset_repo_id.replace('/', '__')}__{dataset_revision}"
    dataset_root = Path.home() / ".cache" / "huggingface" / "lerobot" / dataset_root_name

    print("=== DATASET PREFETCH ===")
    snapshot_download(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        revision=dataset_revision,
        local_dir=dataset_root,
    )
    info_path = hf_hub_download(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        revision=dataset_revision,
        filename="meta/info.json",
        local_dir=dataset_root,
    )
    print(f"Dataset snapshot ready at {dataset_root}")
    print(f"Info path: {info_path}")

    output_dir = str(REMOTE_OUTPUTS / config_dict["job_name"])
    command = build_lerobot_train_command(config_dict, output_dir)
    print("=== Starting PI0.5 fine-tuning (stock lerobot-train CLI) ===")
    print(format_command(command))
    subprocess.run(command, check=True)
    print("=== TRAINING COMPLETE ===")
    return (
        f"PI0.5 fine-tuning complete for {config_dict['job_name']}. "
        f"Pushed to {config_dict['policy_repo_id']}."
    )


@app.local_entrypoint()
def main(profile: str = "pour_absolute_70_from_3k", preflight_only: bool = False):
    preflight_profile(profile)
    if preflight_only:
        print("Preflight only; not launching Modal GPU job.")
        return

    config = load_profile(profile)
    function_call = train_remote.spawn(asdict(config))
    print(f"Spawned Modal training function call: {function_call.object_id}")
    print(f"Output repo: {config.policy_repo_id}")
