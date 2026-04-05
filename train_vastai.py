"""
train_vastai.py — Launch ACT training on a Vast.ai RTX 4090 from your Mac.

This script:
  1. Finds the cheapest RTX 4090 on Vast.ai
  2. Launches it with a PyTorch image
  3. Automatically clones the repo, installs deps, downloads dataset, and trains
  4. You monitor logs from your Mac
  5. When done, download the checkpoint and destroy the instance

Usage:
  1. Set your VASTAI_API_KEY and HF_TOKEN below
  2. Run: uv run python omx_scripts/train_vastai.py
"""

import json
import time
import sys

from vastai_sdk import VastAI

# ──────────────────────────────────────────────
# Configuration — edit these
# ──────────────────────────────────────────────
VASTAI_API_KEY = "f0bb64e15db27a05b3b32317b77a5e10d4d78476186ea0bfd9be386821355f12"  # Get from https://cloud.vast.ai/account/
HF_TOKEN = "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"  # Get from https://huggingface.co/settings/tokens

DATASET_REPO_ID = "RevanthGundala/pick_up_packet_test"
GPU_NAME = "RTX_4090"
DISK_GB = 40
TRAINING_STEPS = 50_000

# Training script that runs on the remote GPU instance
REMOTE_TRAINING_SCRIPT = r'''#!/bin/bash
set -e

echo "=== OMX ACT Training on Vast.ai ==="

# Install lerobot from ROBOTIS fork (has OMX support)
cd /workspace
git clone https://github.com/ROBOTIS-GIT/lerobot.git
cd lerobot
git checkout feature-omx-devel
pip install -e ".[dynamixel]"

# Clone our training scripts
cd /workspace
git clone https://github.com/RevanthGundala/omx-training.git

# Login to HuggingFace (for private dataset)
huggingface-cli login --token {hf_token}

# Download dataset
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('{dataset_repo_id}')
print(f'Dataset loaded: {{ds.num_episodes}} episodes, {{ds.num_frames}} frames')
"

# Run training
echo "=== Starting ACT training ==="
cd /workspace/omx-training
python train.py

echo "=== TRAINING COMPLETE ==="
echo "Checkpoints saved to /workspace/omx-training/omx_scripts/outputs/"
'''


def main():
    if VASTAI_API_KEY == "YOUR_VASTAI_API_KEY":
        print("ERROR: Set your VASTAI_API_KEY at the top of this script.")
        print("Get it from: https://cloud.vast.ai/account/")
        sys.exit(1)
    if HF_TOKEN == "YOUR_HF_TOKEN":
        print("ERROR: Set your HF_TOKEN at the top of this script.")
        print("Get it from: https://huggingface.co/settings/tokens")
        sys.exit(1)

    vast = VastAI(api_key=VASTAI_API_KEY)

    # ── 1. Launch RTX 4090 instance ──
    print(f"Searching for cheapest {GPU_NAME}...")

    onstart = REMOTE_TRAINING_SCRIPT.format(
        hf_token=HF_TOKEN,
        dataset_repo_id=DATASET_REPO_ID,
        training_steps=TRAINING_STEPS,
    )

    print(f"Launching {GPU_NAME} instance...")
    result = vast.launch_instance(
        gpu_name=GPU_NAME,
        num_gpus="1",
        image="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel",
        disk=DISK_GB,
        onstart_cmd=onstart,
        ssh=True,
        label="omx-act-training",
    )

    # Parse instance ID from result
    result_data = json.loads(result) if isinstance(result, str) else result
    if isinstance(result_data, dict) and "new_contract" in result_data:
        instance_id = result_data["new_contract"]
    else:
        print(f"Launch result: {result}")
        print("Check https://cloud.vast.ai/instances/ for your instance.")
        return

    print(f"\n✅ Instance {instance_id} launched!")
    print(f"Training will start automatically (~50 min for {TRAINING_STEPS} steps)")
    print(f"\nMonitor at: https://cloud.vast.ai/instances/")
    print(f"\nTo check logs:")
    print(f"  uv run python -c \"from vastai_sdk import VastAI; v=VastAI(api_key='{VASTAI_API_KEY}'); print(v.logs({instance_id}))\"")
    print(f"\nWhen training is done, download checkpoints:")
    print(f"  uv run python -c \"from vastai_sdk import VastAI; v=VastAI(api_key='{VASTAI_API_KEY}'); print(v.scp_url({instance_id}))\"")
    print(f"  # Then: scp -r <instance>:/workspace/outputs/ omx_scripts/outputs/")
    print(f"\nTo destroy the instance when done:")
    print(f"  uv run python -c \"from vastai_sdk import VastAI; v=VastAI(api_key='{VASTAI_API_KEY}'); v.destroy_instance(id={instance_id})\"")


if __name__ == "__main__":
    main()
