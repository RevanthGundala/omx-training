from configs.train.base import TrainConfig

config = TrainConfig(
    dataset_repo_id="RevanthGundala/003-pour-water",
    policy_pretrained_path="RevanthGundala/pi05-pour-water-3k",
    policy_repo_id="RevanthGundala/pi05-pour-water-70-from-3k",
    job_name="pi05_pour_water_70_from_3k",
    instance_label="omx-pi05-70-from-3k",
    steps=1000,
    compile_model=False,
)
