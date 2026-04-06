"""
train_vastai.py — Launch ACT training on a Vast.ai RTX 4090 from your Mac.

This script:
  1. Finds the cheapest RTX 4090 on Vast.ai
  2. Launches it with a PyTorch image
  3. Installs deps, downloads dataset, and runs training (all in onstart)
  4. Streams all logs to your terminal via vast.logs()
  5. When done, prints download instructions and destroys the instance

Usage:
  1. Export your API keys:
       export VASTAI_API_KEY="your-key"
       export HF_TOKEN="your-token"  (or run: huggingface-cli login)
  2. Run: uv run python train_vastai.py
"""

import base64
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

POLL_INTERVAL = 15  # seconds between status checks
BOOT_TIMEOUT = 1200  # max seconds to wait for instance to start running
LOG_POLL_INTERVAL = 30  # seconds between log checks


def _build_onstart_script(train_py_b64: str) -> str:
    """Build the onstart script that runs setup + training in one shot.

    Everything runs as the main container process, so all output
    naturally appears in vast.logs() and the Vast.ai dashboard.
    """
    return r'''#!/bin/bash
set -e

echo "=== Setting up OMX training environment ==="

# Install lerobot from ROBOTIS fork (has OMX support)
cd /workspace
git clone https://github.com/ROBOTIS-GIT/lerobot.git
cd lerobot
git checkout feature-omx-devel
pip install -e ".[dynamixel]"
pip install "huggingface-hub<1.0"

# Make HF token available to Python libraries
export HF_TOKEN="{hf_token}"

# Download dataset
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('{dataset_repo_id}')
print(f'Dataset loaded: {{ds.num_episodes}} episodes, {{ds.num_frames}} frames')
"

echo "=== SETUP COMPLETE ==="

# Decode and run train.py (embedded as base64, no git clone needed)
cd /workspace
echo "{train_py_b64}" | base64 -d > train.py
echo "=== Starting ACT training ==="
python train.py

echo "=== TRAINING COMPLETE ==="
'''.format(
        hf_token=HF_TOKEN,
        dataset_repo_id=DATASET_REPO_ID,
        train_py_b64=train_py_b64,
    )


def _parse_instance_id(result):
    """Extract instance ID from launch_instance response."""
    result_data = json.loads(result) if isinstance(result, str) else result
    if isinstance(result_data, dict) and "new_contract" in result_data:
        return result_data["new_contract"]
    return None


def _stream_logs(vast, instance_id):
    """Stream container logs until training completes."""
    print("\n📋 Streaming logs (setup + training)...\n")
    seen_lines = set()

    while True:
        try:
            logs = vast.logs(INSTANCE_ID=instance_id, tail="50")
            logs_str = logs if isinstance(logs, str) else str(logs)

            for line in logs_str.splitlines():
                if line and line not in seen_lines:
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
        print("  Run: huggingface-cli login")
        print("  Or:  export HF_TOKEN='your-token'  # from https://huggingface.co/settings/tokens")
        sys.exit(1)

    # Embed train.py as base64
    train_py = Path(__file__).parent / "train.py"
    if not train_py.exists():
        print(f"❌ train.py not found at {train_py}")
        sys.exit(1)
    train_py_b64 = base64.b64encode(train_py.read_bytes()).decode()

    vast = VastAI(api_key=VASTAI_API_KEY)
    instance_id = None

    try:
        # ── 1. Launch instance ──
        print(f"🔍 Searching for cheapest {GPU_NAME}...")

        onstart = _build_onstart_script(train_py_b64)

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

        # ── 2. Attach SSH key ──
        ssh_key_path = Path.home() / ".ssh" / "id_ed25519.pub"
        if not ssh_key_path.exists():
            ssh_key_path = Path.home() / ".ssh" / "id_rsa.pub"
        if ssh_key_path.exists():
            pubkey = ssh_key_path.read_text().strip()
            vast.attach_ssh(instance_id=instance_id, ssh_key=pubkey)
            print("🔑 SSH key attached to instance")

        # ── 3. Wait for boot ──
        print("⏳ Waiting for instance to boot...")
        start = time.time()
        while time.time() - start < BOOT_TIMEOUT:
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
            raise TimeoutError(f"Instance not running after {BOOT_TIMEOUT}s")

        # ── 4. Stream all logs (setup + training) ──
        _stream_logs(vast, instance_id)

        # ── 5. Download checkpoints ──
        print("\n📥 Getting SCP URL for checkpoint download...")
        scp_info = vast.scp_url(id=instance_id)
        print(f"   {scp_info}")

        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        print(f"\n   To download checkpoints, run:")
        print(f"   scp -r <instance>:/workspace/outputs/ {output_dir}/")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        if instance_id:
            print(f"\n💡 Instance {instance_id} is still running.")
            print(f"   Dashboard: https://cloud.vast.ai/instances/")
            print(f"   To destroy: uv run python -c \"from vastai_sdk import VastAI; import os; VastAI(api_key=os.environ['VASTAI_API_KEY']).destroy_instance(id={instance_id})\"")


if __name__ == "__main__":
    main()
