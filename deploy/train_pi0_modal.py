"""
train_pi0_modal.py — Finetune PI0.5 on Modal GPUs.

Uploads train_pi0.py and config.py into a cloud container, installs LeRobot,
and runs PI0.5 finetuning on a GPU.

Usage:
    # First time: authenticate with Modal
    modal token new

    # Train with defaults (A10G, 10k steps, expert-only):
    modal run train_pi0_modal.py

    # Custom GPU and steps:
    modal run train_pi0_modal.py --gpu a100 --training-steps 20000

    # Download checkpoints after training:
    modal volume get omx-pi0-training-logs outputs/ outputs/
"""

import modal

REMOTE_WORKSPACE = "/workspace"
REMOTE_OUTPUTS = "/workspace/outputs"

hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch",
        "torchvision",
        "lerobot[pi]",
    )
    .pip_install("av")
    .add_local_file("training/train_pi0.py", remote_path=f"{REMOTE_WORKSPACE}/train_pi0.py")
    .add_local_file("utils/config.py", remote_path=f"{REMOTE_WORKSPACE}/config.py")
)

app = modal.App("omx-pi05-training", image=image)

vol = modal.Volume.from_name("omx-pi0-training-logs", create_if_missing=True)


@app.function(
    gpu="A100",
    timeout=14400,  # 4 hours
    memory=32768,
    volumes={REMOTE_OUTPUTS: vol},
    secrets=[hf_secret],
)
def train(
    training_steps: int = 10_000,
    batch_size: int = 2,
    save_freq: int = 5_000,
    log_freq: int = 100,
) -> str:
    import os
    import sys
    import shutil
    from pathlib import Path

    sys.path.insert(0, REMOTE_WORKSPACE)
    os.chdir(REMOTE_WORKSPACE)

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["OMX_REQUIRE_CUDA"] = "1"

    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        raise RuntimeError("CUDA not available — aborting.")

    # Clear previous checkpoints on the volume
    vol_path = Path(REMOTE_OUTPUTS)
    for child in vol_path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    actual_output = vol_path / "run"

    # Patch training hyperparameters before importing main()
    import train_pi0 as train_module

    train_module.TRAINING_STEPS = training_steps
    train_module.BATCH_SIZE = batch_size
    train_module.SAVE_FREQ = save_freq
    train_module.LOG_FREQ = log_freq
    train_module.OUTPUT_DIR = actual_output

    train_module.main()

    vol.commit()
    return f"PI0.5 finetuning complete ({training_steps} steps). Checkpoints at volume 'omx-pi0-training-logs'."


@app.function(volumes={REMOTE_OUTPUTS: vol})
def list_checkpoints() -> list[str]:
    """List available checkpoints on the volume."""
    import os

    vol.reload()
    results = []
    for root, dirs, files in os.walk(REMOTE_OUTPUTS):
        for f in files:
            full = os.path.join(root, f)
            results.append(os.path.relpath(full, REMOTE_OUTPUTS))
    return sorted(results)


@app.local_entrypoint()
def main(
    gpu: str = "A100",
    training_steps: int = 10_000,
    batch_size: int = 2,
    save_freq: int = 5_000,
    log_freq: int = 100,
):
    print(f"🚀 Launching PI0.5 finetuning on Modal ({gpu}) ...")
    print(f"   training_steps={training_steps}, batch_size={batch_size}")
    print(f"   save_freq={save_freq}, log_freq={log_freq}")
    print(f"   train_expert_only=True, gradient_checkpointing=True")
    print()

    result = train.remote(
        training_steps=training_steps,
        batch_size=batch_size,
        save_freq=save_freq,
        log_freq=log_freq,
    )
    print(result)

    # Show available checkpoints
    print("\n📁 Checkpoints on volume:")
    files = list_checkpoints.remote()
    for f in files:
        print(f"   {f}")

    print("\n📥 To download checkpoints locally:")
    print("   modal volume get omx-pi0-training-logs outputs/ outputs/")
