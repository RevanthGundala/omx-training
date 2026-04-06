"""
train_vastai.py — Launch ACT training on a Vast.ai RTX 4090 from your Mac.

This script:
  1. Finds the cheapest RTX 4090 on Vast.ai
  2. Launches it with a PyTorch image and installs dependencies
  3. Uploads train.py via the SDK (no git clone of this repo needed)
  4. Runs training and streams logs to your terminal
  5. Downloads the checkpoint and destroys the instance

Usage:
  1. Export your API keys:
       export VASTAI_API_KEY="your-key"
       export HF_TOKEN="your-token"
  2. Run: uv run python train_vastai.py
"""

import json
import os
import sys
import time
from pathlib import Path

from vastai_sdk import VastAI

# ──────────────────────────────────────────────
# Configuration — edit these
# ──────────────────────────────────────────────
VASTAI_API_KEY = os.environ.get("VASTAI_API_KEY", "")

# Read HF token from env var, or fall back to local `hf auth login` cache
def _get_hf_token():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return ""

HF_TOKEN = _get_hf_token()

DATASET_REPO_ID = "RevanthGundala/pick_up_packet_test"
GPU_NAME = "RTX_4090"
DISK_GB = 40
TRAINING_STEPS = 50_000

POLL_INTERVAL = 15  # seconds between status checks
SETUP_TIMEOUT = 600  # max seconds to wait for instance setup
LOG_POLL_INTERVAL = 30  # seconds between log checks during training

# Onstart script: install deps + download dataset only (no training, no repo clone)
SETUP_SCRIPT = r'''#!/bin/bash
set -e

echo "=== Setting up OMX training environment ==="

# Install lerobot from ROBOTIS fork (has OMX support)
cd /workspace
git clone https://github.com/ROBOTIS-GIT/lerobot.git
cd lerobot
git checkout feature-omx-devel
pip install -e ".[dynamixel]"

# Login to HuggingFace (for private dataset)
huggingface-cli login --token {hf_token}

# Download dataset
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('{dataset_repo_id}')
print(f'Dataset loaded: {{ds.num_episodes}} episodes, {{ds.num_frames}} frames')
"

echo "=== SETUP COMPLETE ==="
touch /workspace/.setup_done
'''


def _parse_instance_id(result):
    """Extract instance ID from launch_instance response."""
    result_data = json.loads(result) if isinstance(result, str) else result
    if isinstance(result_data, dict) and "new_contract" in result_data:
        return result_data["new_contract"]
    return None


def _wait_for_instance(vast, instance_id, timeout=SETUP_TIMEOUT):
    """Poll until the instance is running and setup is complete."""
    print("⏳ Waiting for instance to be ready...")
    start = time.time()

    # Phase 1: wait for instance status to be "running"
    while time.time() - start < timeout:
        try:
            info = vast.show_instance(id=instance_id)
            data = json.loads(info) if isinstance(info, str) else info
            status = data.get("actual_status", data.get("status_msg", "unknown"))
            print(f"   Status: {status} ({int(time.time() - start)}s elapsed)")
            if status == "running":
                break
        except Exception as e:
            print(f"   Polling error: {e}")
        time.sleep(POLL_INTERVAL)
    else:
        raise TimeoutError(f"Instance not running after {timeout}s")

    # Phase 2: wait for setup script to finish (sentinel file)
    print("⏳ Waiting for dependency setup to finish...")
    while time.time() - start < timeout:
        try:
            logs = vast.logs(INSTANCE_ID=instance_id, tail="20")
            logs_str = logs if isinstance(logs, str) else str(logs)
            if "SETUP COMPLETE" in logs_str:
                print("✅ Setup complete!")
                return
        except Exception as e:
            print(f"   Log check error: {e}")
        time.sleep(POLL_INTERVAL)

    raise TimeoutError(f"Setup did not complete within {timeout}s")


def _stream_logs(vast, instance_id):
    """Stream logs until training completes or fails."""
    print("\n📋 Streaming training logs...\n")
    seen_lines = set()

    while True:
        try:
            logs = vast.logs(INSTANCE_ID=instance_id, tail="50")
            logs_str = logs if isinstance(logs, str) else str(logs)

            for line in logs_str.splitlines():
                if line not in seen_lines:
                    seen_lines.add(line)
                    print(f"  [remote] {line}")

            if "TRAINING COMPLETE" in logs_str:
                print("\n✅ Training finished!")
                return True
        except Exception as e:
            print(f"  Log error: {e}")

        time.sleep(LOG_POLL_INTERVAL)


def main():
    if not VASTAI_API_KEY:
        print("ERROR: VASTAI_API_KEY not set.")
        print("  export VASTAI_API_KEY='your-key'  # from https://cloud.vast.ai/account/")
        sys.exit(1)
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set.")
        print("  export HF_TOKEN='your-token'  # from https://huggingface.co/settings/tokens")
        sys.exit(1)

    vast = VastAI(api_key=VASTAI_API_KEY)
    instance_id = None

    try:
        # ── 1. Launch instance (setup only, no training) ──
        print(f"🔍 Searching for cheapest {GPU_NAME}...")

        onstart = SETUP_SCRIPT.format(
            hf_token=HF_TOKEN,
            dataset_repo_id=DATASET_REPO_ID,
        )

        print(f"🚀 Launching {GPU_NAME} instance...")
        result = vast.launch_instance(
            gpu_name=GPU_NAME,
            num_gpus="1",
            image="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel",
            disk=DISK_GB,
            onstart_cmd=onstart,
            ssh=True,
            label="omx-act-training",
        )

        instance_id = _parse_instance_id(result)
        if not instance_id:
            print(f"❌ Failed to parse instance ID from: {result}")
            sys.exit(1)

        print(f"✅ Instance {instance_id} launched!")
        print(f"   Dashboard: https://cloud.vast.ai/instances/")

        # ── 2. Wait for instance + setup ──
        _wait_for_instance(vast, instance_id)

        # ── 3. Upload train.py ──
        print("📤 Uploading train.py to instance...")
        train_py = Path(__file__).parent / "train.py"
        if not train_py.exists():
            print(f"❌ train.py not found at {train_py}")
            sys.exit(1)

        vast.copy(
            src=str(train_py),
            dst=f"{instance_id}:/workspace/train.py",
        )
        print("✅ train.py uploaded!")

        # ── 4. Start training ──
        print("🏋️ Starting training...")
        vast.execute(
            id=instance_id,
            COMMAND="cd /workspace && nohup python train.py > /workspace/train.log 2>&1 &",
        )

        # ── 5. Stream logs ──
        _stream_logs(vast, instance_id)

        # ── 6. Download checkpoints ──
        print("\n📥 Getting SCP URL for checkpoint download...")
        scp_info = vast.scp_url(id=instance_id)
        print(f"   {scp_info}")

        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        print(f"\n   To download checkpoints, run:")
        print(f"   scp -r <instance>:/workspace/omx_scripts/outputs/ {output_dir}/")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # ── 7. Destroy instance ──
        if instance_id:
            print(f"\n🗑️  Destroying instance {instance_id}...")
            try:
                vast.destroy_instance(id=instance_id)
                print("✅ Instance destroyed.")
            except Exception as e:
                print(f"⚠️  Could not destroy instance: {e}")
                print(f"   Destroy manually at https://cloud.vast.ai/instances/")


if __name__ == "__main__":
    main()
