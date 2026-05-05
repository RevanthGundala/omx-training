from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    # Dataset
    dataset_repo_id: str
    dataset_revision: str = "main"
    # Policy
    policy_pretrained_path: str = "lerobot/pi05_base"
    policy_repo_id: str = ""              # HF repo to push checkpoint to (must be set per profile)
    policy_type: str = "pi05"
    # Training hparams
    job_name: str = "pi05_pour_water"
    output_dir: str = "/workspace/outputs"
    steps: int = 3000
    batch_size: int = 32
    dtype: str = "bfloat16"
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    gradient_checkpointing: bool = True
    compile_model: bool = True
    log_freq: int = 50
    # Vast.ai instance
    gpu_name: str = "A100_PCIE"
    min_gpu_ram_mb: int = 75000
    disk_gb: int = 150
    vast_image: str = "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"
    instance_label: str = "omx-pi05-training"
