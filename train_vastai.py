"""
train_vastai.py — Launch ACT training on a Vast.ai GPU from your Mac.

This script:
  1. Searches multiple matching GPU offers on Vast.ai
  2. Skips recently failed hosts/offers and retries the next candidate
  3. Launches a PyTorch image with setup + training in onstart
  4. Streams all logs to your terminal via vast.logs()
  5. When done, prints download instructions and leaves the instance running

Usage:
  1. Export your API keys:
       export VASTAI_API_KEY="your-key"
       export HF_TOKEN="your-token"  (or run: huggingface-cli login)
       export OMX_GPU_NAME="RTX_4090"  # optional
       export OMX_VAST_IMAGE="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"  # optional
   2. Run: uv run python train_vastai.py
"""

import base64
import ast
import json
import os
import sys
import time
from pathlib import Path

from vastai_sdk import VastAI

from config import TRAIN_DATASET_REPO_ID as DATASET_REPO_ID
from control_utils import get_hf_token

# ──────────────────────────────────────────────
# Configuration — edit these
# ──────────────────────────────────────────────
VASTAI_API_KEY = os.environ.get("VASTAI_API_KEY", "")
HF_TOKEN = get_hf_token()
DATASET_REVISION = "main"
DEFAULT_GPU_NAME = "RTX_4090"
DEFAULT_VAST_IMAGE = "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"
GPU_NAME = os.environ.get("OMX_GPU_NAME", DEFAULT_GPU_NAME)
DISK_GB = 80
INSTANCE_LABEL = "omx-act-training"
VAST_IMAGE = os.environ.get("OMX_VAST_IMAGE", DEFAULT_VAST_IMAGE)

POLL_INTERVAL = 15  # seconds between status checks
BOOT_TIMEOUT = 1200  # max seconds to wait for instance to start running
LOG_POLL_INTERVAL = 30  # seconds between log checks
SEARCH_LIMIT = 20
SEARCH_ORDER = "dph+"
MAX_LAUNCH_ATTEMPTS = 8
MIN_RELIABILITY = 0.98
FAILED_OFFER_CACHE = Path.home() / ".cache" / "omx-training" / "vastai_failed_offers.json"
FAILED_OFFER_TTL_S = 12 * 60 * 60
MAX_FAILED_CACHE_ENTRIES = 100


def _build_onstart_script(train_py_b64: str) -> str:
    """Build the onstart script that runs setup + training in one shot.

    Everything runs as the main container process, so all output
    naturally appears in vast.logs() and the Vast.ai dashboard.
    """
    return r'''#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
trap 'status=$?; if [ $status -ne 0 ]; then echo "=== REMOTE FAILED (exit ${{status}}) ==="; fi' EXIT

echo "=== CUDA PREFLIGHT ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this container.")
print("Detected GPU:", torch.cuda.get_device_name(0))
PY

echo "=== Setting up OMX training environment ==="

# Install FFmpeg (needed by torchcodec for video decoding)
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1

# Install lerobot from ROBOTIS fork (has OMX support)
cd /workspace
git clone https://github.com/ROBOTIS-GIT/lerobot.git
cd lerobot
git checkout feature-omx-devel
pip install --no-cache-dir -e ".[dynamixel]"
pip install --no-cache-dir "huggingface-hub<1.0"

# Ensure av1 video decoding works (dataset uses av1 codec)
pip install --no-cache-dir --force-reinstall av

# Make HF token available to Python libraries
export HF_TOKEN="{hf_token}"

# Pre-download dataset (errors here are OK — train.py re-downloads with correct tolerance)
python -c "
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

root = Path.home() / '.cache' / 'huggingface' / 'lerobot' / '{dataset_root_name}'
snapshot_download(
    repo_id='{dataset_repo_id}',
    repo_type='dataset',
    revision='{dataset_revision}',
    local_dir=root,
)
info_path = hf_hub_download(
    repo_id='{dataset_repo_id}',
    repo_type='dataset',
    revision='{dataset_revision}',
    filename='meta/info.json',
    local_dir=root,
)
print(f'Dataset snapshot ready at {{root}}')
print(f'Info path: {{info_path}}')
" || echo "Dataset pre-download had warnings (train.py will handle it)"

echo "=== SETUP COMPLETE ==="

# Decode and run train.py (embedded as base64, no git clone needed)
cd /workspace
echo "{train_py_b64}" | base64 -d > train.py
echo "=== Starting ACT training ==="
export OMX_REQUIRE_CUDA=1
python train.py

echo "=== TRAINING COMPLETE ==="
'''.format(
        hf_token=HF_TOKEN,
        dataset_repo_id=DATASET_REPO_ID,
        dataset_revision=DATASET_REVISION,
        dataset_root_name=f"{DATASET_REPO_ID.replace('/', '__')}__{DATASET_REVISION}",
        train_py_b64=train_py_b64,
    )


def _coerce_result_data(result):
    if isinstance(result, (dict, list)):
        return result
    if not isinstance(result, str):
        return str(result)

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(result)
        except (ValueError, SyntaxError):
            return result


def _parse_instance_id(result):
    """Extract instance ID from launch_instance response."""
    result_data = _coerce_result_data(result)
    if isinstance(result_data, dict):
        for key in ("new_contract", "instance_id", "contract_id", "id"):
            value = result_data.get(key)
            if isinstance(value, int):
                return value
    return None


def _load_failed_offer_cache():
    now = time.time()
    if not FAILED_OFFER_CACHE.exists():
        return []

    try:
        payload = json.loads(FAILED_OFFER_CACHE.read_text())
    except json.JSONDecodeError:
        return []

    failures = payload if isinstance(payload, list) else payload.get("failures", [])
    return [
        item
        for item in failures
        if isinstance(item, dict) and now - item.get("timestamp", 0) < FAILED_OFFER_TTL_S
    ]


def _save_failed_offer_cache(failures):
    FAILED_OFFER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    trimmed = failures[-MAX_FAILED_CACHE_ENTRIES:]
    FAILED_OFFER_CACHE.write_text(json.dumps(trimmed, indent=2))


def _record_failed_offer(offer, reason):
    failures = _load_failed_offer_cache()
    failures.append(
        {
            "timestamp": time.time(),
            "offer_id": offer.get("id"),
            "host_id": offer.get("host_id"),
            "machine_id": offer.get("machine_id"),
            "reason": reason,
        }
    )
    _save_failed_offer_cache(failures)


def _get_failed_offer_sets():
    failures = _load_failed_offer_cache()
    return (
        {item["offer_id"] for item in failures if item.get("offer_id") is not None},
        {item["host_id"] for item in failures if item.get("host_id") is not None},
        {item["machine_id"] for item in failures if item.get("machine_id") is not None},
    )


def _format_offer(offer):
    location = offer.get("geolocation", "unknown")
    price = offer.get("dph_total", offer.get("dph_base", "?"))
    reliability = offer.get("reliability2", offer.get("reliability", "?"))
    return (
        f"offer={offer.get('id')} host={offer.get('host_id')} machine={offer.get('machine_id')} "
        f"price=${price}/hr reliability={reliability} location={location}"
    )


def _search_candidate_offers(vast, blocked_offer_ids, blocked_host_ids, blocked_machine_ids):
    query = f"gpu_name={GPU_NAME} rentable=True rented=False"
    offers = _coerce_result_data(vast.search_offers(query=query, limit=SEARCH_LIMIT, order=SEARCH_ORDER))
    if not isinstance(offers, list):
        raise RuntimeError(f"Unexpected search_offers result: {offers}")

    filtered = []
    skipped_blocked = 0

    for offer in offers:
        if not isinstance(offer, dict):
            continue
        if int(offer.get("num_gpus", 0)) != 1:
            continue
        if float(offer.get("disk_space", 0)) < DISK_GB:
            continue
        if offer.get("verification") != "verified":
            continue
        if offer.get("is_vm_deverified"):
            continue

        reliability = float(offer.get("reliability2", offer.get("reliability", 0.0)) or 0.0)
        if reliability < MIN_RELIABILITY:
            continue

        offer_id = offer.get("id")
        host_id = offer.get("host_id")
        machine_id = offer.get("machine_id")
        if (
            offer_id in blocked_offer_ids
            or host_id in blocked_host_ids
            or machine_id in blocked_machine_ids
        ):
            skipped_blocked += 1
            continue

        filtered.append(offer)

    if skipped_blocked:
        print(f"↪️  Skipping {skipped_blocked} recently failed offers/hosts")

    return filtered


def _launch_offer(vast, offer, onstart):
    print(f"🚀 Trying {_format_offer(offer)}")
    return vast.create_instance(
        id=offer["id"],
        image=VAST_IMAGE,
        disk=DISK_GB,
        onstart_cmd=onstart,
        ssh=True,
        label=INSTANCE_LABEL,
        cancel_unavail=True,
    )


def _attach_ssh_key(vast, instance_id):
    ssh_key_path = Path.home() / ".ssh" / "id_ed25519.pub"
    if not ssh_key_path.exists():
        ssh_key_path = Path.home() / ".ssh" / "id_rsa.pub"
    if ssh_key_path.exists():
        pubkey = ssh_key_path.read_text().strip()
        vast.attach_ssh(instance_id=instance_id, ssh_key=pubkey)
        print("🔑 SSH key attached to instance")


def _wait_for_instance_running(vast, instance_id):
    print("⏳ Waiting for instance to boot...")
    start = time.time()
    while time.time() - start < BOOT_TIMEOUT:
        info = vast.show_instance(id=instance_id)
        data = _coerce_result_data(info)
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected show_instance result: {data}")

        status = data.get("actual_status", data.get("status_msg", "unknown"))
        print(f"   Status: {status} ({int(time.time() - start)}s elapsed)")
        if status == "running":
            return
        if status in {"offline", "exited", "error", "dead"}:
            raise RuntimeError(f"Instance entered bad status: {status}")
        time.sleep(POLL_INTERVAL)

    raise TimeoutError(f"Instance not running after {BOOT_TIMEOUT}s")


def _destroy_instance_safely(vast, instance_id):
    try:
        vast.destroy_instance(id=instance_id)
        print(f"🧹 Destroyed failed instance {instance_id}")
    except Exception as e:
        print(f"⚠️  Failed to destroy instance {instance_id}: {e}")


def _stream_logs(vast, instance_id):
    """Stream container logs until training completes."""
    print("\n📋 Streaming logs (setup + training)...\n")
    seen_lines = set()

    NOISE_PATTERNS = (
        "kex_exchange_identification", "Connection closed", "Connection to ssh",
        "UTC 202",  # bare timestamp heartbeat lines
    )

    while True:
        try:
            logs = vast.logs(INSTANCE_ID=instance_id, tail="500")
            logs_str = logs if isinstance(logs, str) else str(logs)

            for line in logs_str.splitlines():
                if line and line not in seen_lines:
                    seen_lines.add(line)
                    if not any(p in line for p in NOISE_PATTERNS):
                        print(f"  [remote] {line}")

            if "TRAINING COMPLETE" in logs_str:
                print("\n✅ Training finished!")
                return True
            if "=== REMOTE FAILED" in logs_str:
                raise RuntimeError("Remote setup or training failed. See logs above.")
        except Exception as e:
            print(f"  Log error: {e}")
            if "Remote setup or training failed" in str(e):
                raise

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
        # ── 1. Search offers and launch instance ──
        onstart = _build_onstart_script(train_py_b64)
        blocked_offer_ids, blocked_host_ids, blocked_machine_ids = _get_failed_offer_sets()
        runtime_blocked_offer_ids = set(blocked_offer_ids)
        runtime_blocked_host_ids = set(blocked_host_ids)
        runtime_blocked_machine_ids = set(blocked_machine_ids)

        print(f"🔍 Searching for {GPU_NAME} offers...")
        candidates = _search_candidate_offers(
            vast,
            runtime_blocked_offer_ids,
            runtime_blocked_host_ids,
            runtime_blocked_machine_ids,
        )
        if not candidates and blocked_offer_ids:
            print("↪️  No candidates left after blacklist, retrying without cached failures...")
            candidates = _search_candidate_offers(vast, set(), set(), set())
        if not candidates:
            raise RuntimeError(f"No viable {GPU_NAME} offers found.")

        launch_offer = None
        last_error = None
        for attempt, offer in enumerate(candidates[:MAX_LAUNCH_ATTEMPTS], start=1):
            launch_offer = offer
            try:
                print(f"\n=== Launch attempt {attempt}/{min(len(candidates), MAX_LAUNCH_ATTEMPTS)} ===")
                result = _launch_offer(vast, offer, onstart)
                instance_id = _parse_instance_id(result)
                if not instance_id:
                    raise RuntimeError(f"Failed to parse instance ID from: {result}")

                print(f"✅ Instance {instance_id} launched!")
                print(f"   Dashboard: https://cloud.vast.ai/instances/")

                _attach_ssh_key(vast, instance_id)
                _wait_for_instance_running(vast, instance_id)
                break
            except Exception as e:
                last_error = e
                print(f"⚠️  {_format_offer(offer)} failed: {e}")
                runtime_blocked_offer_ids.add(offer.get("id"))
                runtime_blocked_host_ids.add(offer.get("host_id"))
                runtime_blocked_machine_ids.add(offer.get("machine_id"))
                _record_failed_offer(offer, str(e))
                if instance_id is not None:
                    _destroy_instance_safely(vast, instance_id)
                    instance_id = None
        else:
            raise RuntimeError(
                f"Unable to launch a healthy {GPU_NAME} instance after {min(len(candidates), MAX_LAUNCH_ATTEMPTS)} attempts: {last_error}"
            )

        # ── 2. Stream all logs (setup + training) ──
        _stream_logs(vast, instance_id)

        # ── 3. Download checkpoints ──
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
