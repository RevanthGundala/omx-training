from configs.train.base import TrainConfig


config = TrainConfig(
    dataset_repo_id="RevanthGundala/003-pour-water-globalstats",
    policy_repo_id="RevanthGundala/pi05-pour-water-3k-globalstats",
    job_name="pi05_pour_water_3k_globalstats",
    instance_label="omx-pi05-3k-globalstats",
    steps=3000,
    compile_model=False,
)
