from configs.train.base import TrainConfig

# UMI-style relative actions: train on the ABSOLUTE dataset (003-pour-water).
# LeRobot's PI05 processor pipeline handles the conversion automatically:
#   train:  action -= state[obs]   (RelativeActionsProcessorStep)
#   infer:  action += state[obs]   (AbsoluteActionsProcessorStep)
# Gripper is kept absolute (open/close is naturally absolute).
#
# Prereq: dataset stats must be recomputed in relative space using:
#   uv run lerobot-edit-dataset \
#     --repo_id RevanthGundala/003-pour-water \
#     --operation.type recompute_stats \
#     --operation.relative_action true \
#     --operation.chunk_size 50 \
#     --operation.relative_exclude_joints "['gripper']" \
#     --push_to_hub true
config = TrainConfig(
    dataset_repo_id="RevanthGundala/003-pour-water",
    policy_repo_id="RevanthGundala/pi05-pour-water-relative-3k",
    job_name="pi05_pour_water_relative",
    instance_label="omx-pi05-relative-training",
    use_relative_actions=True,
    relative_exclude_joints=("gripper",),
    # torch.compile CUDA Graph pool was eating 41GB on A100_SXM4 80GB and
    # OOM'ing the Adam optimizer. Disable compile for this run; ~2x slower
    # first step but reliably fits.
    compile_model=False,
)
