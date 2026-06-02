# OMX Training

Standalone Python scripts for the OMX robot arm using the [LeRobot](https://github.com/ROBOTIS-GIT/lerobot) API.

## Setup

```bash
cd /Users/revanthgundala/projects/omx-training
uv sync
```

## Project Structure

```
calibration/     — Gripper & joint calibration scripts
control/         — Teleoperation
data/            — Recording, replay, visualization, dataset tools
deploy/          — Modal & Vast.ai cloud training/serving
diagnostics/     — Motor diagnostics & debugging
evaluation/      — Local policy evaluation (ACT, PI0.5)
training/        — Local training scripts (ACT, PI0.5)
utils/           — Shared config, robot helpers, control utils
tests/           — Smoke tests
rlt/             — Experimental RL modules
outputs/         — Checkpoints & Rerun logs
```

## Scripts

### 1. Teleoperate — `control/teleop.py`

Move the leader arm by hand and the follower mirrors your movements.

```bash
uv run python -m control.teleop
```

Press `Ctrl+C` to stop.

### 2. Record Data — `data/record.py`

Record teleoperation episodes to a local dataset.

```bash
uv run python -m data.record
```

- Press `→` (right arrow) to end an episode and start the next one
- Press `Ctrl+C` to stop recording entirely
- Data is saved locally to `~/.cache/huggingface/lerobot/<repo_id>/`

### 3. Visualize Data — `data/visualize.py`

View recorded episodes in the Rerun viewer (camera + joint plots side by side).

```bash
uv run python -m data.visualize
```

### 4. Train ACT Policy — `training/train.py`

Train locally on a CUDA/MPS machine:

```bash
uv run python -m training.train
```

### 5. Train on Vast.ai — `deploy/train_vastai.py`

Launch training on a Vast.ai GPU (for example an RTX 4090 at ~50 min, ~$0.40):

```bash
export VASTAI_API_KEY="your-key"  # from https://cloud.vast.ai/account/
export HF_TOKEN="your-token"      # from https://huggingface.co/settings/tokens
# Optional overrides when changing GPU families or images:
export OMX_GPU_NAME="RTX_4090"
export OMX_VAST_IMAGE="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"
uv run python -m deploy.train_vastai
```

The script automatically launches an instance, uploads `training/train.py`, runs
training, streams logs, and leaves the instance running so you can download the
checkpoints before destroying it.

If the cheapest offer is unhealthy, the launcher retries the next matching
offer and temporarily skips recently failed hosts using a cache at
`~/.cache/omx-training/vastai_failed_offers.json`. It also runs a CUDA
preflight before setup so an incompatible GPU/image combination fails fast.

### 6. Live Eval — `evaluation/eval.py`

Run a saved checkpoint directly on the follower arm:

```bash
uv run python -m evaluation.eval
```

By default it loads:

```bash
outputs/checkpoints/last/pretrained_model
```

and uses the live follower observation (joint state + camera) to predict the
next action and send it back to the robot in a closed loop.

If the checkpoint was trained without image inputs, `eval.py` now still opens
the front camera for monitoring and Rerun logging by default, but the policy
continues to act from joint state only. To make camera frames affect the
actions, the checkpoint itself must be trained with `observation.images.*`
inputs.

### 7. Smoke Test — `tests/smoke_test.py`

Run a lightweight preflight before spending on Vast.ai:

```bash
uv run python -m tests.smoke_test
```

This checks the dataset metadata against the Hugging Face repo tree, validates
the `training/train.py` config on CPU, and verifies that `deploy/train_vastai.py` still
generates the expected Vast onstart script.

To also run a real 1-step local training smoke test:

```bash
uv run python -m tests.smoke_test --train-step
```

### 8. ACT sim_transfer_cube benchmark

Use this path to check the custom `models/act` implementation against the
original ACT MuJoCo benchmark, not to simulate OMX.

Generate official ACT HDF5 demos from a checkout of
`https://github.com/tonyzhaozh/act`:

```bash
export ACT_REPO_DIR=/path/to/act
uv run python -m benchmarks.act_sim.generate \
  --dataset-dir data/benchmarks/act_sim_transfer_cube_scripted \
  --episodes 50
```

Then run the custom ACT trainer on the benchmark data:

```bash
uv run python -m models.act.train --profile sim_transfer_cube_reference --run-name seed0
```

To run the same reference profile on Modal, upload the downloaded/generated HDF5
episodes once, then launch training:

```bash
uv run modal run deploy/train_custom_act_modal.py --profile sim_transfer_cube_reference \
  --benchmark-dataset-dir data/benchmarks/act_sim_transfer_cube_scripted \
  --upload-benchmark-data
uv run modal run deploy/train_custom_act_modal.py --profile sim_transfer_cube_reference --run-name seed0
```

For a small wiring check, generate two episodes into
`data/benchmarks/act_sim_transfer_cube_smoke` and run:

```bash
uv run python -m models.act.train --profile sim_transfer_cube_smoke --dry-run
```

## Configuration

Each script has a **Configuration** section at the top of the file. Shared
settings live in `utils/config.py`. For `deploy/train_vastai.py`, you can also
override the target GPU and base image with `OMX_GPU_NAME` and `OMX_VAST_IMAGE`
instead of editing the file.

## Finding Your Serial Ports

```bash
ls /dev/tty.usbmodem*
```

The leader arm has motor IDs 1-6, the follower has IDs 11-16.
