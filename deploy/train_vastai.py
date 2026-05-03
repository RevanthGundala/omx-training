"""
train_vastai.py — Launch PI0.5 finetuning on a Vast.ai GPU from your Mac.

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
       export OMX_GPU_NAME="A100_PCIE_80GB"  # optional
       export OMX_VAST_IMAGE="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"  # optional
   2. Run: uv run python deploy/train_vastai.py
"""

import ast
import json
import os
import sys
import time
from pathlib import Path

from vastai_sdk import VastAI

from utils.config import TRAIN_DATASET_REPO_ID as DATASET_REPO_ID
from utils.control_utils import get_hf_token

# ──────────────────────────────────────────────
# Configuration — edit these
# ──────────────────────────────────────────────
VASTAI_API_KEY = os.environ.get("VASTAI_API_KEY", "")
HF_TOKEN = get_hf_token()
DATASET_REVISION = "main"
DEFAULT_GPU_NAME = "A100_PCIE"
MIN_GPU_RAM_MB = 75000  # require 80GB-class GPU (filters out 40GB A100s)
DEFAULT_VAST_IMAGE = "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"
GPU_NAME = os.environ.get("OMX_GPU_NAME", DEFAULT_GPU_NAME)
DISK_GB = 150
INSTANCE_LABEL = "omx-pi05-training"
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


def _verify_image_exists(image: str) -> None:
    """HEAD the Docker registry manifest. Abort with clear error if not 200."""
    import urllib.request
    import urllib.error

    if ":" not in image:
        repo, tag = image, "latest"
    else:
        repo, tag = image.rsplit(":", 1)
    if "/" not in repo:
        repo = "library/" + repo

    try:
        token_url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
        with urllib.request.urlopen(token_url, timeout=10) as r:
            token = json.loads(r.read())["token"]
        manifest_url = f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}"
        req = urllib.request.Request(manifest_url, method="HEAD", headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.index.v1+json",
        })
        with urllib.request.urlopen(req, timeout=10) as r:
            if r.status == 200:
                print(f"✅ Image manifest verified: {image}")
                return
    except urllib.error.HTTPError as e:
        print(f"❌ Image manifest check failed for {image}: HTTP {e.code}")
        print(f"   Docker Hub does not have a published manifest for tag '{tag}'.")
        print(f"   Pick a valid tag from https://hub.docker.com/r/{repo}/tags")
        sys.exit(1)
    except Exception as e:
        print(f"⚠️  Could not verify image (network issue?): {e}. Proceeding anyway.")
        return


def _check_existing_instances(vast) -> None:
    """List existing instances with our label. Auto-destroy stuck ones, abort if healthy.

    Vast SDK's show_instances may return an empty list under the SDK; fall back to direct API.
    """
    import urllib.request

    try:
        req = urllib.request.Request(
            "https://console.vast.ai/api/v0/instances/",
            headers={"Authorization": f"Bearer {VASTAI_API_KEY}"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        instances = data.get("instances", [])
    except Exception as e:
        print(f"⚠️  Could not list existing instances ({e}). Proceeding without check.")
        return

    ours = [i for i in instances if i.get("label") == INSTANCE_LABEL]
    if not ours:
        return

    healthy = [i for i in ours if i.get("actual_status") == "running"]
    if healthy:
        print(f"❌ Found {len(healthy)} healthy '{INSTANCE_LABEL}' instance(s) already running:")
        for i in healthy:
            print(f"     id={i['id']} gpu={i.get('gpu_name')} status={i.get('actual_status')}")
        print(f"   Attach with: vastai logs {healthy[0]['id']}")
        print(f"   Or destroy:  curl -X DELETE -H 'Authorization: Bearer $VASTAI_API_KEY' \\")
        print(f"                  https://console.vast.ai/api/v0/instances/{healthy[0]['id']}/")
        sys.exit(1)

    # All ours are unhealthy (loading/error/offline) — destroy them.
    print(f"🧹 Found {len(ours)} stale '{INSTANCE_LABEL}' instance(s); destroying:")
    for i in ours:
        print(f"   - destroying id={i['id']} (status={i.get('actual_status')})")
        try:
            req = urllib.request.Request(
                f"https://console.vast.ai/api/v0/instances/{i['id']}/",
                method="DELETE",
                headers={"Authorization": f"Bearer {VASTAI_API_KEY}"},
            )
            urllib.request.urlopen(req, timeout=15).read()
        except Exception as e:
            print(f"     ⚠️  destroy failed: {e}")


def _build_onstart_script() -> str:
    """Build the onstart script that runs setup + training in one shot.

    Everything runs as the main container process, so all output
    naturally appears in vast.logs() and the Vast.ai dashboard.
    """
    return r'''#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
trap 'status=$?; if [ $status -ne 0 ]; then echo "=== REMOTE FAILED (exit ${{status}}) ==="; fi' EXIT

echo "=== CUDA PREFLIGHT ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# Ensure /workspace exists (nvidia/cuda image doesn't pre-create it).
mkdir -p /workspace
cd /workspace

# Ubuntu 24.04 base image — has Python 3.12 by default. Install pip + torch.
apt-get update -qq && apt-get install -y -qq python3 python3-pip python3-venv ffmpeg > /dev/null 2>&1
ln -sf /usr/bin/python3 /usr/local/bin/python || true

# Use a venv to avoid PEP 668 externally-managed errors on Ubuntu 24.04.
python3 -m venv /opt/venv
. /opt/venv/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this container.")
print("Detected GPU:", torch.cuda.get_device_name(0))
print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
PY

echo "=== Setting up OMX PI0.5 training environment ==="

# Install stock lerobot from PyPI with PI0.5 extras.
# NOTE: ROBOTIS fork is needed for OMX *hardware*, not training. Stock
# lerobot[pi] has the pi05 module; ROBOTIS fork doesn't.
pip install --no-cache-dir "lerobot[pi]>=0.5.1"

# Ensure av1 video decoding works (dataset uses av1 codec). lerobot 0.5.1 needs av>=15,<16.
pip install --no-cache-dir --force-reinstall "av>=15,<16"

# Make HF token available to Python libraries
export HF_TOKEN="{hf_token}"
export HUGGINGFACE_HUB_TOKEN="{hf_token}"

# Pre-download dataset to populate the HF cache (lerobot-train re-uses it).
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
" || echo "Dataset pre-download had warnings (lerobot-train will re-fetch as needed)"

echo "=== SETUP COMPLETE ==="

cd /workspace
. /opt/venv/bin/activate

# lerobot-train refuses to write into an existing output dir.
# Make sure /workspace/outputs is fresh on every run.
rm -rf /workspace/outputs

echo "=== Starting PI0.5 finetuning (stock lerobot-train CLI) ==="
# Run the stock LeRobot training CLI verbatim per docs/source/pi05.mdx.
# Output is teed to /workspace/train.log so we can `vastai logs` or ssh tail it.
lerobot-train \
    --dataset.repo_id={dataset_repo_id} \
    --policy.type=pi05 \
    --output_dir=/workspace/outputs \
    --job_name=pi05_pour_water \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.repo_id=RevanthGundala/pi05-pour-water-3k \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --steps=3000 \
    --batch_size=32 \
    --policy.device=cuda \
    --log_freq=50 \
    2>&1 | tee /workspace/train.log

echo "=== TRAINING COMPLETE ==="
'''.format(
        hf_token=HF_TOKEN,
        dataset_repo_id=DATASET_REPO_ID,
        dataset_revision=DATASET_REVISION,
        dataset_root_name=f"{DATASET_REPO_ID.replace('/', '__')}__{DATASET_REVISION}",
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
        if float(offer.get("gpu_ram", 0)) < MIN_GPU_RAM_MB:
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
        runtype="ssh_proxy",
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
            logs = vast.logs(instance_id=instance_id, tail="500")
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

    # Pre-flight: verify Docker image manifest exists on the registry.
    _verify_image_exists(VAST_IMAGE)

    # Pre-flight: check for existing labeled instances. Auto-cleans stale, aborts if healthy.
    vast = VastAI(api_key=VASTAI_API_KEY)
    _check_existing_instances(vast)

    instance_id = None

    try:
        # ── 1. Search offers and launch instance ──
        onstart = _build_onstart_script()
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
