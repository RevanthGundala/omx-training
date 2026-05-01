"""
train_modal.py — Train ACT policy on Modal GPUs.

This is the Modal equivalent of train_vastai.py. It uploads train.py and
config.py into a cloud container, installs the ROBOTIS LeRobot fork, and
runs ACT training on a GPU.

Usage:
    # First time: authenticate with Modal
    modal token new

    # Train with defaults (A10G, 50k steps):
    modal run train_modal.py

    # Custom GPU and steps:
    modal run train_modal.py --gpu a100 --training-steps 100000

    # Download checkpoints after training:
    modal volume get omx-act-training-logs outputs/ outputs/
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
        "lerobot",
        "huggingface-hub<1.0",
    )
    .pip_install("av")  # force-reinstall for av1 codec support
    .add_local_file("training/train.py", remote_path=f"{REMOTE_WORKSPACE}/train.py")
    .add_local_file("utils/config.py", remote_path=f"{REMOTE_WORKSPACE}/config.py")
)

app = modal.App("omx-act-training", image=image)

vol = modal.Volume.from_name("omx-act-training-logs", create_if_missing=True)


@app.function(
    gpu="A10G",
    timeout=14400,  # 4 hours
    memory=32768,
    volumes={REMOTE_OUTPUTS: vol},
    secrets=[hf_secret],
)
def train(
    training_steps: int = 50_000,
    batch_size: int = 8,
    save_freq: int = 10_000,
    log_freq: int = 100,
) -> str:
    import os
    import sys
    import shutil
    from pathlib import Path

    sys.path.insert(0, REMOTE_WORKSPACE)
    os.chdir(REMOTE_WORKSPACE)

    # Make HF token available
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["OMX_REQUIRE_CUDA"] = "1"

    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError("CUDA not available — aborting.")

    # Clear previous checkpoints on the volume
    vol_path = Path(REMOTE_OUTPUTS)
    for child in vol_path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    # Use a subdirectory inside the volume mount so validate() doesn't
    # complain about the mount point itself already existing.
    actual_output = vol_path / "run"

    # Patch training hyperparameters before importing main()
    import train as train_module

    train_module.TRAINING_STEPS = training_steps
    train_module.BATCH_SIZE = batch_size
    train_module.SAVE_FREQ = save_freq
    train_module.LOG_FREQ = log_freq
    train_module.OUTPUT_DIR = actual_output

    train_module.main()

    vol.commit()
    return f"Training complete ({training_steps} steps). Checkpoints at volume 'omx-act-training-logs'."


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
    gpu: str = "A10G",
    training_steps: int = 50_000,
    batch_size: int = 8,
    save_freq: int = 10_000,
    log_freq: int = 100,
):
    print(f"🚀 Launching ACT training on Modal ({gpu}) ...")
    print(f"   training_steps={training_steps}, batch_size={batch_size}")
    print(f"   save_freq={save_freq}, log_freq={log_freq}")
    print()

    # Override GPU type via the function's spec
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
    print("   modal volume get omx-act-training-logs outputs/ outputs/")
