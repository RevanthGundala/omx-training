from configs.train.base import TrainConfig

config = TrainConfig(
    dataset_repo_id="RevanthGundala/003-pour-water",
    policy_repo_id="RevanthGundala/pi05-pour-water-3k",
    job_name="pi05_pour_water",
    instance_label="omx-pi05-training",
)
