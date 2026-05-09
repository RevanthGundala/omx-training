"""Run MolmoAct2 on recorded OMX LeRobot frames.

This does not connect to the robot. It checks whether MolmoAct2 can consume
this project's recorded camera/state observations and reports action shape/range
plus a rough comparison to recorded actions.

Example:
    modal run evaluation/offline_molmoact2_dataset_modal.py \
        --dataset-repo-id RevanthGundala/003-pour-water \
        --episodes 0,10,20 \
        --start-frame 0
"""

from __future__ import annotations

from pathlib import Path

import modal


DEFAULT_REPO_ID = "allenai/MolmoAct2-SO100_101"
DEFAULT_NORM_TAG = "so100_so101_molmoact2"
DEFAULT_DATASET_REPO_ID = "RevanthGundala/003-pour-water"
DEFAULT_TASK = "Pour water from one plastic bottle into another."

app = modal.App("omx-molmoact2-dataset-eval")
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "pillow",
        "numpy",
        "huggingface_hub",
        "requests",
        "accelerate",
        "einops",
        "lerobot[pi]",
        "av",
        "pandas",
    )
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=2400,
    region="us-west",
    secrets=[hf_secret],
)
def run_dataset_eval(
    dataset_repo_id: str,
    episodes: list[int],
    start_frame: int = 0,
    repo_id: str = DEFAULT_REPO_ID,
    norm_tag: str = DEFAULT_NORM_TAG,
    num_steps: int = 10,
) -> dict:
    import os
    import time

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    torch.set_float32_matmul_precision("high")

    def image_tensor_to_pil(tensor) -> Image.Image:
        arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    checkpoint_path = Path(snapshot_download(repo_id=repo_id, repo_type="model"))
    dataset_root = Path(snapshot_download(repo_id=dataset_repo_id, repo_type="dataset"))

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(checkpoint_path), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(checkpoint_path),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to("cuda").eval()
    load_seconds = time.perf_counter() - t0

    results = []
    for ep in episodes:
        ds = LeRobotDataset(dataset_repo_id, root=dataset_root, episodes=[ep], video_backend="pyav")
        if len(ds) <= start_frame:
            print(f"episode {ep}: too short, skipping")
            continue

        sample = ds[start_frame]
        images = []
        for cam in ("top", "wrist"):
            key = f"observation.images.{cam}"
            if key in sample:
                images.append(image_tensor_to_pil(sample[key]))
        if not images:
            print(f"episode {ep}: no top/wrist images, skipping")
            continue

        state = sample["observation.state"].detach().cpu().numpy().astype(np.float32)
        task = sample.get("task", DEFAULT_TASK)
        if not isinstance(task, str):
            task = DEFAULT_TASK

        start = time.perf_counter()
        with torch.inference_mode():
            out = model.predict_action(
                processor=processor,
                images=images,
                task=task,
                state=state,
                norm_tag=norm_tag,
                action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=num_steps,
                normalize_language=True,
                enable_cuda_graph=True,
            )
        inference_seconds = time.perf_counter() - start

        actions = out.actions
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        elif actions.ndim == 1:
            actions = actions.reshape(1, -1)

        n = min(actions.shape[0], len(ds) - start_frame)
        target = np.stack(
            [ds[start_frame + i]["action"].detach().cpu().numpy() for i in range(n)]
        ).astype(np.float32)

        comparable = actions.ndim == 2 and target.ndim == 2 and actions.shape[1] == target.shape[1]
        row = {
            "episode": ep,
            "start_frame": start_frame,
            "task": task,
            "image_count": len(images),
            "state_shape": list(state.shape),
            "action_shape": list(actions.shape),
            "target_shape": list(target.shape),
            "inference_seconds": round(inference_seconds, 3),
            "action_min": float(actions.min()),
            "action_max": float(actions.max()),
            "per_dim_min": actions.min(axis=0).round(4).tolist(),
            "per_dim_max": actions.max(axis=0).round(4).tolist(),
            "first_action": actions[0].round(4).tolist(),
            "recorded_first": target[0].round(4).tolist(),
            "comparable_to_recorded": comparable,
        }
        if comparable:
            pred_n = actions[:n]
            err = pred_n - target
            row["first_delta"] = (pred_n[0] - target[0]).round(4).tolist()
            row["mae"] = np.abs(err).mean(axis=0).round(4).tolist()
            row["max_abs"] = np.abs(err).max(axis=0).round(4).tolist()

        print(f"\nEP {ep}")
        for key, value in row.items():
            print(f"  {key}: {value}")
        results.append(row)

    return {
        "repo_id": repo_id,
        "norm_tag": norm_tag,
        "dataset_repo_id": dataset_repo_id,
        "episodes": episodes,
        "start_frame": start_frame,
        "load_seconds": round(load_seconds, 3),
        "results": results,
    }


@app.local_entrypoint()
def main(
    dataset_repo_id: str = DEFAULT_DATASET_REPO_ID,
    episodes: str = "0,10,20",
    start_frame: int = 0,
    repo_id: str = DEFAULT_REPO_ID,
    norm_tag: str = DEFAULT_NORM_TAG,
    num_steps: int = 10,
):
    episode_list = [int(part.strip()) for part in episodes.split(",") if part.strip()]
    result = run_dataset_eval.remote(
        dataset_repo_id=dataset_repo_id,
        episodes=episode_list,
        start_frame=start_frame,
        repo_id=repo_id,
        norm_tag=norm_tag,
        num_steps=num_steps,
    )
    print("\nSUMMARY")
    print(f"repo_id={result['repo_id']}")
    print(f"dataset={result['dataset_repo_id']}")
    print(f"load={result['load_seconds']:.3f}s")
    for row in result["results"]:
        print(
            f"ep={row['episode']} action_shape={row['action_shape']} "
            f"range=[{row['action_min']:.3f}, {row['action_max']:.3f}] "
            f"first_action={row['first_action']}"
        )
