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

# Clone repo and install
cd /workspace
git clone https://github.com/ROBOTIS-GIT/lerobot.git
cd lerobot
git checkout feature-omx-devel
pip install -e ".[dynamixel]"

# Login to HuggingFace
huggingface-cli login --token {hf_token}

# Download dataset
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('{dataset_repo_id}')
print(f'Dataset loaded: {{ds.num_episodes}} episodes, {{ds.num_frames}} frames')
"

# Write the training script
cat > /workspace/train_act.py << 'TRAINEOF'
import logging
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.amp import GradScaler

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import save_checkpoint, get_step_checkpoint_dir, update_last_checkpoint
from lerobot.utils.utils import get_safe_torch_device, has_method, format_big_number

DATASET_REPO_ID = "{dataset_repo_id}"
OUTPUT_DIR = Path("/workspace/outputs")
DEVICE = "cuda"
BATCH_SIZE = 8
NUM_WORKERS = 4
TRAINING_STEPS = {training_steps}
LOG_FREQ = 100
SAVE_FREQ = 10_000
SEED = 1000
CHUNK_SIZE = 100
USE_VAE = True
KL_WEIGHT = 10.0
LEARNING_RATE = 1e-5

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    set_seed(SEED)

    dataset_config = DatasetConfig(repo_id=DATASET_REPO_ID)
    act_config = ACTConfig(
        input_features={{}}, output_features={{}},
        device=DEVICE, chunk_size=CHUNK_SIZE, n_action_steps=CHUNK_SIZE,
        use_vae=USE_VAE, kl_weight=KL_WEIGHT,
        optimizer_lr=LEARNING_RATE, optimizer_lr_backbone=LEARNING_RATE,
        vision_backbone="resnet18",
    )
    train_cfg = TrainPipelineConfig(
        dataset=dataset_config, policy=act_config, output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, steps=TRAINING_STEPS,
        log_freq=LOG_FREQ, save_freq=SAVE_FREQ, seed=SEED, eval_freq=-1,
    )
    train_cfg.validate()

    device = get_safe_torch_device(DEVICE, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info(f"Loading dataset: {{DATASET_REPO_ID}}")
    dataset = make_dataset(train_cfg)
    logging.info(f"  Episodes: {{dataset.num_episodes}}, Frames: {{dataset.num_frames}}")

    logging.info("Creating ACT policy")
    policy = make_policy(cfg=train_cfg.policy, ds_meta=dataset.meta)
    policy.train()
    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(f"  Learnable parameters: {{format_big_number(num_params)}}")

    optimizer, lr_scheduler = make_optimizer_and_scheduler(train_cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=act_config.use_amp)

    if hasattr(train_cfg.policy, "drop_n_last_frames"):
        sampler = EpisodeAwareSampler(dataset.episode_data_index, drop_n_last_frames=train_cfg.policy.drop_n_last_frames, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=shuffle, sampler=sampler, pin_memory=True, drop_last=False)
    dl_iter = cycle(dataloader)

    train_metrics = {{
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }}
    train_tracker = MetricsTracker(BATCH_SIZE, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=0)

    logging.info(f"Starting training for {{TRAINING_STEPS}} steps on {{device}}...")
    for step in range(1, TRAINING_STEPS + 1):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        start_time = time.perf_counter()
        with torch.autocast(device_type=device.type) if act_config.use_amp else nullcontext():
            loss, output_dict = policy.forward(batch)
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), train_cfg.optimizer.grad_clip_norm, error_if_nonfinite=False)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if has_method(policy, "update"):
            policy.update()

        train_tracker.loss = loss.item()
        train_tracker.grad_norm = grad_norm.item()
        train_tracker.lr = optimizer.param_groups[0]["lr"]
        train_tracker.update_s = time.perf_counter() - start_time
        train_tracker.step()

        if step % LOG_FREQ == 0:
            logging.info(train_tracker)
            train_tracker.reset_averages()
        if step % SAVE_FREQ == 0 or step == TRAINING_STEPS:
            logging.info(f"Saving checkpoint at step {{step}}")
            checkpoint_dir = get_step_checkpoint_dir(OUTPUT_DIR, TRAINING_STEPS, step)
            save_checkpoint(checkpoint_dir, step, train_cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)

    logging.info(f"Training complete! Checkpoints at {{OUTPUT_DIR}}")

if __name__ == "__main__":
    main()
TRAINEOF

# Run training
cd /workspace/lerobot
echo "=== Starting ACT training ==="
python /workspace/train_act.py

echo "=== TRAINING COMPLETE ==="
echo "Checkpoints saved to /workspace/outputs/"
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
