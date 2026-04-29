# OMX Training

Standalone Python scripts for the OMX robot arm using the [LeRobot](https://github.com/ROBOTIS-GIT/lerobot) API.

## Setup

```bash
cd /Users/revanthgundala/projects/omx-training
uv sync
```

## Scripts

### 1. Teleoperate — `teleop.py`

Move the leader arm by hand and the follower mirrors your movements.

```bash
uv run python teleop.py
```

Press `Ctrl+C` to stop.

### 2. Record Data — `record.py`

Record teleoperation episodes to a local dataset.

```bash
uv run python record.py
```

- Press `→` (right arrow) to end an episode and start the next one
- Press `Ctrl+C` to stop recording entirely
- Data is saved locally to `~/.cache/huggingface/lerobot/<repo_id>/`

### 3. Visualize Data — `visualize.py`

View recorded episodes in the Rerun viewer (camera + joint plots side by side).

```bash
uv run python visualize.py
```

### 4. Train ACT Policy — `train.py`

Train locally on a CUDA/MPS machine:

```bash
python train.py
```

### 5. Train on Vast.ai — `train_vastai.py`

Launch training on a Vast.ai GPU (for example an RTX 4090 at ~50 min, ~$0.40):

```bash
export VASTAI_API_KEY="your-key"  # from https://cloud.vast.ai/account/
export HF_TOKEN="your-token"      # from https://huggingface.co/settings/tokens
# Optional overrides when changing GPU families or images:
export OMX_GPU_NAME="RTX_4090"
export OMX_VAST_IMAGE="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"
uv run python train_vastai.py
```

The script automatically launches an instance, uploads `train.py`, runs
training, streams logs, and leaves the instance running so you can download the
checkpoints before destroying it.

If the cheapest offer is unhealthy, the launcher retries the next matching
offer and temporarily skips recently failed hosts using a cache at
`~/.cache/omx-training/vastai_failed_offers.json`. It also runs a CUDA
preflight before setup so an incompatible GPU/image combination fails fast.

### 6. Live Eval — `eval.py`

Run a saved checkpoint directly on the follower arm:

```bash
uv run python eval.py
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

### 7. Smoke Test — `smoke_test.py`

Run a lightweight preflight before spending on Vast.ai:

```bash
uv run python smoke_test.py
```

This checks the dataset metadata against the Hugging Face repo tree, validates
the `train.py` config on CPU, and verifies that `train_vastai.py` still
generates the expected Vast onstart script.

To also run a real 1-step local training smoke test:

```bash
uv run python smoke_test.py --train-step
```

## Configuration

Each script has a **Configuration** section at the top of the file. For
`train_vastai.py`, you can also override the target GPU and base image with
`OMX_GPU_NAME` and `OMX_VAST_IMAGE` instead of editing the file.

## Finding Your Serial Ports

```bash
ls /dev/tty.usbmodem*
```

The leader arm has motor IDs 1-6, the follower has IDs 11-16.
