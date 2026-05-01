"""
train.py — Train an ACT policy on your collected OMX dataset.

This script trains an Action Chunking with Transformers (ACT) policy
using imitation learning on your recorded teleoperation data.

Run on a machine with a CUDA GPU:
    python omx_scripts/train.py
"""

import logging
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from torch.amp import GradScaler

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import save_checkpoint, get_step_checkpoint_dir, update_last_checkpoint
try:
    from lerobot.utils.device_utils import get_safe_torch_device
except ImportError:
    from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.utils import has_method, format_big_number

# ──────────────────────────────────────────────
# Configuration — edit these to match your setup
# ──────────────────────────────────────────────
from utils.config import TRAIN_DATASET_REPO_ID as DATASET_REPO_ID
DATASET_REVISION = "main"
DATASET_ROOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "lerobot"
    / f"{DATASET_REPO_ID.replace('/', '__')}__{DATASET_REVISION}"
)
OUTPUT_DIR = Path("outputs")
DEVICE = "cuda"
REQUIRE_CUDA = os.environ.get("OMX_REQUIRE_CUDA", "0") == "1"

# Training hyperparameters
BATCH_SIZE = 8
NUM_WORKERS = 4
TRAINING_STEPS = 50_000
LOG_FREQ = 100  # print loss every N steps
SAVE_FREQ = 10_000  # save checkpoint every N steps
SEED = 1000

# ACT policy hyperparameters
CHUNK_SIZE = 100  # number of future actions to predict at once
USE_VAE = True  # use variational autoencoder (standard for ACT)
KL_WEIGHT = 10.0  # weight for VAE KL-divergence loss
LEARNING_RATE = 1e-5
VISION_BACKBONE = "resnet18"


def _resolve_training_device(requested_device: str) -> torch.device:
    requested_device = str(requested_device)

    if requested_device == "cuda":
        if torch.cuda.is_available():
            return get_safe_torch_device("cuda", log=True)

        message = "CUDA requested but not available."
        if REQUIRE_CUDA:
            raise RuntimeError(
                f"{message} This run requires an NVIDIA GPU. "
                "Try a different Vast.ai offer or a more compatible CUDA container image."
            )

        logging.warning(f"{message} Falling back to CPU.")
        return get_safe_torch_device("cpu", log=True)

    return get_safe_torch_device(requested_device, log=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    set_seed(SEED)

    # ── 1. Build the training config ──
    dataset_config = DatasetConfig(repo_id=DATASET_REPO_ID, root=str(DATASET_ROOT))

    act_config = ACTConfig(
        output_features={},  # filled in by make_policy from dataset metadata
        device=DEVICE,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        use_vae=USE_VAE,
        kl_weight=KL_WEIGHT,
        optimizer_lr=LEARNING_RATE,
        vision_backbone=VISION_BACKBONE,
        repo_id="RevanthGundala/act-single-fold-tissue",
    )

    train_cfg = TrainPipelineConfig(
        dataset=dataset_config,
        policy=act_config,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        steps=TRAINING_STEPS,
        log_freq=LOG_FREQ,
        save_freq=SAVE_FREQ,
        seed=SEED,
        eval_freq=-1,  # no eval during training (real robot, not sim)
        resume=False,
    )
    # Clear stale output directory from previous runs
    if OUTPUT_DIR.exists() and not OUTPUT_DIR.is_symlink():
        shutil.rmtree(OUTPUT_DIR)
    train_cfg.validate()

    # ── 2. Setup device ──
    device = _resolve_training_device(DEVICE)
    train_cfg.policy.device = device.type
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # ── 3. Load dataset ──
    logging.info(f"Loading dataset: {DATASET_REPO_ID}")
    snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=DATASET_ROOT,
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    ds_meta = LeRobotDatasetMetadata(
        DATASET_REPO_ID,
        root=DATASET_ROOT,
    )
    delta_timestamps = resolve_delta_timestamps(train_cfg.policy, ds_meta)
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
        tolerance_s=1e4,  # recording has uneven frame timing between episodes
    )
    logging.info(f"  Episodes: {dataset.num_episodes}")
    logging.info(f"  Frames:   {dataset.num_frames}")

    # ── 4. Create policy ──
    logging.info("Creating ACT policy")
    policy = make_policy(cfg=train_cfg.policy, ds_meta=dataset.meta)
    policy.train()

    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(f"  Learnable parameters: {format_big_number(num_params)}")

    # ── 5. Create preprocessor/postprocessor for normalization ──
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=train_cfg.policy,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
        },
    )

    # ── 6. Create optimizer and scheduler ──
    optimizer, lr_scheduler = make_optimizer_and_scheduler(train_cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=act_config.use_amp)

    # ── 6. Create dataloader ──
    if hasattr(train_cfg.policy, "drop_n_last_frames"):
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=train_cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    # ── 7. Training loop ──
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        BATCH_SIZE, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=0
    )

    logging.info(f"\nStarting training for {TRAINING_STEPS} steps...")
    logging.info(f"Output dir: {OUTPUT_DIR}")

    for step in range(1, TRAINING_STEPS + 1):
        # Load batch
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

        # Forward + backward
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

        # Track metrics
        train_tracker.loss = loss.item()
        train_tracker.grad_norm = grad_norm.item()
        train_tracker.lr = optimizer.param_groups[0]["lr"]
        train_tracker.update_s = time.perf_counter() - start_time
        train_tracker.step()

        # Log
        if step % LOG_FREQ == 0:
            logging.info(train_tracker)
            train_tracker.reset_averages()

        # Save checkpoint
        if step % SAVE_FREQ == 0 or step == TRAINING_STEPS:
            logging.info(f"Saving checkpoint at step {step}")
            checkpoint_dir = get_step_checkpoint_dir(OUTPUT_DIR, TRAINING_STEPS, step)
            save_checkpoint(checkpoint_dir, step, train_cfg, policy, optimizer, lr_scheduler,
                           preprocessor=preprocessor, postprocessor=postprocessor)
            update_last_checkpoint(checkpoint_dir)

    logging.info(f"\nTraining complete! Final checkpoint saved to {OUTPUT_DIR}")
    logging.info("To run inference, use the checkpoint with your robot.")


if __name__ == "__main__":
    main()
