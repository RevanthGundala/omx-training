"""
train_pi0.py — Finetune a PI0.5 policy on your collected OMX dataset.

This script finetunes PI0.5 (Physical Intelligence's vision-language-action
model) using imitation learning on your recorded teleoperation data.

PI0.5 requires camera images + joint state + text task description.

By default, only the action expert (~300M params) is trained while the
PaliGemma VLM backbone (~2B params) is frozen.

Run on a machine with a CUDA GPU:
    python train_pi0.py
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
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.policies.pi05.configuration_pi05 import PI05Config
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
from utils.config import TRAIN_DATASET_REPO_ID as DATASET_REPO_ID, PI05_MODEL_REPO_ID, TASK_NAME
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
BATCH_SIZE = 2
NUM_WORKERS = 4
TRAINING_STEPS = 10_000
LOG_FREQ = 100
SAVE_FREQ = 5_000
SEED = 1000

# PI0.5 finetuning hyperparameters
CHUNK_SIZE = 100
N_ACTION_STEPS = 50
LEARNING_RATE = 2.5e-5
TRAIN_EXPERT_ONLY = True       # freeze PaliGemma VLM, train only action expert
FREEZE_VISION_ENCODER = True   # freeze vision tower (PaliGemma SigLIP)
GRADIENT_CHECKPOINTING = True  # reduce VRAM usage
DTYPE = "bfloat16"
NUM_JOINTS = 6


def _resolve_training_device(requested_device: str) -> torch.device:
    requested_device = str(requested_device)

    if requested_device == "cuda":
        if torch.cuda.is_available():
            return get_safe_torch_device("cuda", log=True)

        message = "CUDA requested but not available."
        if REQUIRE_CUDA:
            raise RuntimeError(
                f"{message} This run requires an NVIDIA GPU. "
            )

        logging.warning(f"{message} Falling back to CPU.")
        return get_safe_torch_device("cpu", log=True)

    return get_safe_torch_device(requested_device, log=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    set_seed(SEED)

    # ── 1. Build the training config ──
    dataset_config = DatasetConfig(repo_id=DATASET_REPO_ID, root=str(DATASET_ROOT))

    pi05_config = PI05Config(
        pretrained_path=PI05_MODEL_REPO_ID,
        device=DEVICE,
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        optimizer_lr=LEARNING_RATE,
        train_expert_only=TRAIN_EXPERT_ONLY,
        freeze_vision_encoder=FREEZE_VISION_ENCODER,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        dtype=DTYPE,
        repo_id="RevanthGundala/pi05-pour-water",
    )

    train_cfg = TrainPipelineConfig(
        dataset=dataset_config,
        policy=pi05_config,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        steps=TRAINING_STEPS,
        log_freq=LOG_FREQ,
        save_freq=SAVE_FREQ,
        seed=SEED,
        eval_freq=-1,
        resume=False,
    )
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
        tolerance_s=1e4,
    )
    logging.info(f"  Episodes: {dataset.num_episodes}")
    logging.info(f"  Frames:   {dataset.num_frames}")

    # ── 4. Create policy (loads pretrained PI0.5 base) ──
    print(f"Creating PI0.5 policy from {PI05_MODEL_REPO_ID}", flush=True)
    policy = make_policy(cfg=train_cfg.policy, ds_meta=dataset.meta)
    print("Policy created, setting to train mode...", flush=True)
    policy.train()
    print("Policy in train mode.", flush=True)

    num_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in policy.parameters())
    print(f"  Total parameters:     {format_big_number(num_total)}", flush=True)
    print(f"  Learnable parameters: {format_big_number(num_learnable)}", flush=True)

    # ── 5. Move policy to device ──
    print(f"Moving policy to {device}...", flush=True)
    policy.to(device)
    print("Policy on device.", flush=True)

    # ── 6. Create preprocessor/postprocessor pipelines ──
    # Handles: normalization, tokenization of task text, device placement
    print("Creating preprocessor/postprocessor...", flush=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=train_cfg.policy,
        pretrained_path=PI05_MODEL_REPO_ID,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )
    print("Preprocessor/postprocessor created.", flush=True)

    # ── 6. Create optimizer and scheduler ──
    print("Creating optimizer...", flush=True)
    optimizer, lr_scheduler = make_optimizer_and_scheduler(train_cfg, policy)
    print("Optimizer created.", flush=True)

    # ── 7. Create dataloader ──
    print("Creating dataloader...", flush=True)
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
    print("Dataloader created.", flush=True)

    # ── 8. Training loop ──
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

    print(f"\nStarting PI0.5 finetuning for {TRAINING_STEPS} steps...", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)

    grad_clip_norm = train_cfg.optimizer.grad_clip_norm

    for step in range(1, TRAINING_STEPS + 1):
        # Load batch
        start_time = time.perf_counter()
        if step == 1:
            print("Loading first batch...", flush=True)
        batch = next(dl_iter)
        if step == 1:
            print(f"First batch loaded. Keys: {list(batch.keys())}", flush=True)
        # Run through preprocessor (normalizes, tokenizes task text, moves to device)
        batch = preprocessor(batch)
        if step == 1:
            print("First batch preprocessed.", flush=True)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        # Forward + backward
        start_time = time.perf_counter()
        policy.train()
        if step == 1:
            print("Running first forward pass...", flush=True)
        loss, output_dict = policy.forward(batch)
        if step == 1:
            print(f"First forward pass done. Loss: {loss.item():.4f}", flush=True)

        loss.backward()
        if step == 1:
            print("First backward pass done.", flush=True)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), grad_clip_norm, error_if_nonfinite=False
        )
        optimizer.step()
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
            save_checkpoint(checkpoint_dir, step, train_cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)

            # Also save processors for inference
            preprocessor.push_to_hub(pi05_config.repo_id) if step == TRAINING_STEPS else None

    logging.info(f"\nTraining complete! Final checkpoint saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
