"""Offline MolmoAct2 download/inference smoke test on Modal.

This does not connect to the robot. It loads an AllenAI MolmoAct2 checkpoint,
runs the model-card sample observation through ``predict_action``, and reports
the action shape/range plus inference latency.

Example:
    modal run evaluation/offline_molmoact2_modal.py
"""

from __future__ import annotations

import modal


DEFAULT_REPO_ID = "allenai/MolmoAct2-SO100_101"
DEFAULT_NORM_TAG = "so100_so101_molmoact2"

app = modal.App("omx-molmoact2-smoke")
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
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
    )
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
    region="us-west",
    secrets=[hf_secret],
)
def run_molmoact2_smoke(
    repo_id: str = DEFAULT_REPO_ID,
    norm_tag: str = DEFAULT_NORM_TAG,
    num_warmups: int = 2,
    num_steps: int = 10,
) -> dict:
    import os
    import time
    from pathlib import Path

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    torch.set_float32_matmul_precision("high")

    checkpoint_path = Path(snapshot_download(repo_id=repo_id, repo_type="model"))
    top_rgb = Image.open(checkpoint_path / "assets/sample_realsense_top_rgb.png").convert("RGB")
    side_rgb = Image.open(checkpoint_path / "assets/sample_realsense_side_rgb.png").convert("RGB")

    task = "Move the arm towards the lemon, grasp it, lift it up, and drop it into the red bowl."
    robot_state = np.array(
        [
            -0.52734375,
            189.140625,
            181.40625,
            60.64453125,
            -3.603515625,
            1.0971786975860596,
        ],
        dtype=np.float32,
    )

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(checkpoint_path), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(checkpoint_path),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to("cuda").eval()
    load_seconds = time.perf_counter() - t0

    def predict_once() -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        with torch.inference_mode():
            out = model.predict_action(
                processor=processor,
                images=[top_rgb, side_rgb],
                task=task,
                state=robot_state,
                norm_tag=norm_tag,
                action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=num_steps,
                normalize_language=True,
                enable_cuda_graph=True,
            )
        elapsed = time.perf_counter() - start
        actions = out.actions
        if hasattr(actions, "detach"):
            arr = actions.detach().cpu().numpy()
        else:
            arr = np.asarray(actions)
        return arr.astype(np.float32, copy=False), elapsed

    warmup_latencies = []
    for _ in range(num_warmups):
        _, elapsed = predict_once()
        warmup_latencies.append(elapsed)

    actions, inference_seconds = predict_once()
    flat = actions.reshape(-1, actions.shape[-1]) if actions.ndim > 1 else actions.reshape(1, -1)

    result = {
        "repo_id": repo_id,
        "checkpoint_path": str(checkpoint_path),
        "norm_tag": norm_tag,
        "load_seconds": round(load_seconds, 3),
        "warmup_seconds": [round(v, 3) for v in warmup_latencies],
        "inference_seconds": round(inference_seconds, 3),
        "action_shape": list(actions.shape),
        "action_dtype": str(actions.dtype),
        "action_min": float(actions.min()),
        "action_max": float(actions.max()),
        "action_mean": float(actions.mean()),
        "per_dim_min": flat.min(axis=0).round(4).tolist(),
        "per_dim_max": flat.max(axis=0).round(4).tolist(),
        "first_action": flat[0].round(4).tolist(),
    }

    print("MolmoAct2 smoke result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    return result


@app.local_entrypoint()
def main(
    repo_id: str = DEFAULT_REPO_ID,
    norm_tag: str = DEFAULT_NORM_TAG,
    num_warmups: int = 2,
    num_steps: int = 10,
):
    result = run_molmoact2_smoke.remote(
        repo_id=repo_id,
        norm_tag=norm_tag,
        num_warmups=num_warmups,
        num_steps=num_steps,
    )
    print("\nSUMMARY")
    print(f"repo_id={result['repo_id']}")
    print(f"action_shape={result['action_shape']} dtype={result['action_dtype']}")
    print(
        "range="
        f"[{result['action_min']:.4f}, {result['action_max']:.4f}] "
        f"mean={result['action_mean']:.4f}"
    )
    print(
        f"load={result['load_seconds']:.3f}s "
        f"warmups={result['warmup_seconds']} "
        f"inference={result['inference_seconds']:.3f}s"
    )
    print(f"first_action={result['first_action']}")
