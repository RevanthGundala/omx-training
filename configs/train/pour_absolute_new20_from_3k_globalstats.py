from configs.train.base import TrainConfig


config = TrainConfig(
    dataset_repo_id="RevanthGundala/003-pour-water-new20-only-globalstats",
    policy_pretrained_path="RevanthGundala/pi05-pour-water-3k-globalstats",
    policy_repo_id="RevanthGundala/pi05-pour-water-new20-from-3k-globalstats",
    job_name="pi05_pour_water_new20_from_3k_globalstats",
    instance_label="omx-pi05-new20-from-3k-globalstats",
    steps=3000,
    compile_model=False,
)
