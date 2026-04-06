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

Launch training on a Vast.ai RTX 4090 (~50 min, ~$0.40):

```bash
export VASTAI_API_KEY="your-key"  # from https://cloud.vast.ai/account/
export HF_TOKEN="your-token"      # from https://huggingface.co/settings/tokens
uv run python train_vastai.py
```

The script automatically launches an instance, uploads `train.py`, runs
training, streams logs, and destroys the instance when done.

## Configuration

Each script has a **Configuration** section at the top of the file. Edit the
constants directly — no command-line arguments needed.

## Finding Your Serial Ports

```bash
ls /dev/tty.usbmodem*
```

The leader arm has motor IDs 1-6, the follower has IDs 11-16.
