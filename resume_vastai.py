"""
resume_vastai.py — Resume training on an already-running Vast.ai instance.

Skips provisioning. Just fixes the HF token, downloads the dataset,
uploads train.py, runs training, streams logs, and cleans up.

Usage:
    uv run python resume_vastai.py
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

from vastai_sdk import VastAI

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
INSTANCE_ID = 34219860 

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
LOG_POLL_INTERVAL = 30


def _stream_logs(vast, instance_id):
    """Stream training logs until training completes."""
    print("\n📋 Streaming training logs...\n")
    seen_lines = set()

    while True:
        try:
            result = vast.execute(
                id=instance_id,
                COMMAND="tail -n 50 /workspace/train.log 2>/dev/null || echo 'Waiting for training output...'",
            )
            logs_str = result if isinstance(result, str) else str(result)

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
        print("ERROR: export VASTAI_API_KEY='your-key'")
        sys.exit(1)

    vast = VastAI(api_key=VASTAI_API_KEY)

    try:
        # ── 0. Attach SSH key ──
        ssh_key_path = Path.home() / ".ssh" / "id_ed25519.pub"
        if not ssh_key_path.exists():
            ssh_key_path = Path.home() / ".ssh" / "id_rsa.pub"
        if ssh_key_path.exists():
            pubkey = ssh_key_path.read_text().strip()
            vast.attach_ssh(instance_id=INSTANCE_ID, ssh_key=pubkey)
            print("🔑 SSH key attached to instance")

        # ── 1. Fix huggingface-hub version + download dataset ──
        print(f"📦 Fixing deps and downloading dataset on instance {INSTANCE_ID}...")
        result = vast.execute(
            id=INSTANCE_ID,
            COMMAND=f'pip install "huggingface-hub<1.0" -q && export HF_TOKEN="{HF_TOKEN}" && python -c "'
                    f'from lerobot.datasets.lerobot_dataset import LeRobotDataset; '
                    f'ds = LeRobotDataset(\\\"{DATASET_REPO_ID}\\\"); '
                    f'print(f\\\"Dataset: {{ds.num_episodes}} episodes, {{ds.num_frames}} frames\\\")'
                    f'"',
        )
        print(f"  {result}")

        # ── 2. Upload train.py ──
        print("📤 Uploading train.py...")
        train_py = Path(__file__).parent / "train.py"
        encoded = base64.b64encode(train_py.read_bytes()).decode()
        vast.execute(
            id=INSTANCE_ID,
            COMMAND=f'echo "{encoded}" | base64 -d > /workspace/train.py',
        )
        print("✅ train.py uploaded!")

        # ── 3. Start training ──
        print("🏋️ Starting training...")
        vast.execute(
            id=INSTANCE_ID,
            COMMAND='cd /workspace && nohup bash -c \'export HF_TOKEN="' + HF_TOKEN + '" && python train.py 2>&1; echo "=== TRAINING COMPLETE ==="\' | tee /workspace/train.log > /proc/1/fd/1 2>&1 &',
        )

        # ── 4. Stream logs ──
        _stream_logs(vast, INSTANCE_ID)

        # ── 5. Download checkpoints ──
        print("\n📥 Getting SCP URL for checkpoint download...")
        scp_info = vast.scp_url(id=INSTANCE_ID)
        print(f"   {scp_info}")

        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        print(f"\n   To download checkpoints, run:")
        print(f"   scp -r <instance>:/workspace/omx_scripts/outputs/ {output_dir}/")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        print(f"\n💡 Instance {INSTANCE_ID} is still running (not destroyed).")
        print(f"   To destroy: uv run python -c \"from vastai_sdk import VastAI; import os; VastAI(api_key=os.environ['VASTAI_API_KEY']).destroy_instance(id={INSTANCE_ID})\"")


if __name__ == "__main__":
    main()
