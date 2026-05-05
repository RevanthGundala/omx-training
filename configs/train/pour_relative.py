from configs.train.base import TrainConfig

config = TrainConfig(
    dataset_repo_id="RevanthGundala/004-pour-water-relative",
    policy_repo_id="RevanthGundala/pi05-pour-water-relative-3k",
    job_name="pi05_pour_water_relative",
    instance_label="omx-pi05-relative-training",
    # torch.compile CUDA Graph pool was eating 41GB on A100_SXM4 80GB and
    # OOM'ing the Adam optimizer. Disable compile for this run; ~2x slower
    # first step but reliably fits.
    compile_model=False,
)
