from configs.train.base import TrainConfig


config = TrainConfig(
    dataset_repo_id="RevanthGundala/003-pour-water-new20-only-globalstats",
    policy_pretrained_path="lerobot/pi05_base",
    policy_repo_id="RevanthGundala/pi05-pour-water-new20-from-base-globalstats",
    job_name="pi05_pour_water_new20_from_base_globalstats",
    instance_label="omx-pi05-new20-from-base-globalstats",
    steps=3000,
    compile_model=False,
)
