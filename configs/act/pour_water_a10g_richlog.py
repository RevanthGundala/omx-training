from configs.act.base import ACTConfig
from utils.config import TRAIN_DATASET_REPO_ID


config = ACTConfig(
    dataset_repo_id=TRAIN_DATASET_REPO_ID,
    job_name="act_pour_water_wrist_top_a10g_richlog",
    num_train_steps=40_000,
    log_freq=100,
    eval_freq=1000,
    save_freq=1000,
)