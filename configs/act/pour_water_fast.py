from configs.act.base import ACTConfig
from utils.config import TRAIN_DATASET_REPO_ID


config = ACTConfig(
    dataset_repo_id=TRAIN_DATASET_REPO_ID,
    job_name="act_pour_water_fast",
    batch_size=2,
    num_train_steps=2,
    num_workers=0,
    eval_freq=1,
    save_freq=1,
)