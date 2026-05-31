"""
Train the custom models/act ACT implementation on Modal.

Usage:
    modal run deploy/train_custom_act_modal.py --profile pour_water
    modal run deploy/train_custom_act_modal.py --profile pour_water_fast --run-name smoke-modal

Download outputs:
    modal volume get omx-custom-act-training-logs / outputs/modal_custom_act
"""

from __future__ import annotations

from dataclasses import asdict, replace
import importlib
from pathlib import Path

import modal


REMOTE_WORKSPACE = Path("/workspace")
REMOTE_OUTPUTS = Path("/outputs")
MODAL_GPU = "A100-80GB"

hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04", add_python="3.12")
    .apt_install("git", "ffmpeg", "linux-libc-dev", "clang")
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "huggingface-hub>=1.0,<2.0",
        "lerobot[dynamixel]>=0.5.1",
        "av>=15,<16",
        "pandas",
        "pyarrow",
        "safetensors",
    )
    .add_local_dir("models", remote_path=str(REMOTE_WORKSPACE / "models"))
    .add_local_dir("configs", remote_path=str(REMOTE_WORKSPACE / "configs"))
    .add_local_dir("utils", remote_path=str(REMOTE_WORKSPACE / "utils"))
)

app = modal.App("omx-custom-act-training", image=image)
vol = modal.Volume.from_name("omx-custom-act-training-logs", create_if_missing=True)


def load_profile(profile: str):
    module = importlib.import_module(f"configs.act.{profile}")
    return module.config


def preflight_profile(profile: str) -> None:
    from huggingface_hub import HfApi

    config = load_profile(profile)
    print(f"PROFILE {profile}")
    print(f"  modal_gpu={MODAL_GPU}")
    print(f"  dataset={config.dataset_repo_id}@{config.dataset_revision}")
    print(f"  cameras={config.camera_names}")
    print(f"  batch_size={config.batch_size}")
    print(f"  chunk_size={config.chunk_size}")
    print(f"  d_model={config.d_model}")
    print(f"  encoder_layers={config.num_encoder_layers}")
    print(f"  decoder_layers={config.num_decoder_layers}")
    print(f"  lr={config.learning_rate}")
    print(f"  kl_weight={config.kl_weight}")
    print(f"  steps={config.num_train_steps}")

    HfApi().dataset_info(config.dataset_repo_id, revision=config.dataset_revision)


@app.function(
    gpu=MODAL_GPU,
    timeout=86_400,
    memory=65_536,
    volumes={REMOTE_OUTPUTS: vol},
    secrets=[hf_secret],
)
def train_remote(profile: str, run_name: str | None = None) -> str:
    import os
    import sys

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    sys.path.insert(0, str(REMOTE_WORKSPACE))
    os.chdir(REMOTE_WORKSPACE)

    import torch

    print("=== CUDA PREFLIGHT ===")
    print("Torch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Modal container.")
    print("Detected GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))

    config_module = importlib.import_module(f"configs.act.{profile}")
    remote_output_root = REMOTE_OUTPUTS / "act_experiments"
    config_module.config = replace(config_module.config, output_root=str(remote_output_root))

    from models.act.train import main as train_main

    print("=== STARTING CUSTOM ACT TRAINING ===")
    print(f"profile={profile}")
    print(f"run_name={run_name}")
    print(f"output_root={remote_output_root}")
    print(f"config={asdict(config_module.config)}")

    train_main(profile=profile, run_name=run_name, dry_run=False)
    vol.commit()
    return f"Custom ACT training complete for profile={profile}; outputs saved in volume omx-custom-act-training-logs."


@app.function(volumes={REMOTE_OUTPUTS: vol})
def list_outputs() -> list[str]:
    import os

    vol.reload()
    results = []
    for root, _dirs, files in os.walk(REMOTE_OUTPUTS):
        for filename in files:
            full = os.path.join(root, filename)
            results.append(os.path.relpath(full, REMOTE_OUTPUTS))
    return sorted(results)


@app.local_entrypoint()
def main(profile: str = "pour_water", run_name: str | None = None, preflight_only: bool = False):
    preflight_profile(profile)
    if preflight_only:
        print("Preflight only; not launching Modal GPU job.")
        return

    function_call = train_remote.spawn(profile, run_name)
    print(f"Spawned Modal custom ACT training call: {function_call.object_id}")
    print("Outputs volume: omx-custom-act-training-logs")
    print("Download later with:")
    print("  modal volume get omx-custom-act-training-logs / outputs/modal_custom_act")
