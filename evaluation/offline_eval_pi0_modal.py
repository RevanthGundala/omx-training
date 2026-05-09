"""Offline PI0.5 checkpoint sanity check on recorded LeRobot episodes.

Runs a checkpoint on saved dataset start frames and compares the predicted first
action chunk against the recorded next actions. This does not connect to the
robot.

Example:
    modal run evaluation/offline_eval_pi0_modal.py \
        --checkpoint-repo-id RevanthGundala/pi05-pour-water-70-from-3k \
        --dataset-repo-id RevanthGundala/003-pour-water \
        --episodes 50,55,60,65 \
        --start-frame 0
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("omx-pi05-offline-eval")
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch",
        "torchvision",
        "lerobot[pi]",
        "numpy",
        "pandas",
        "safetensors",
        "av",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    region="us-west",
    secrets=[hf_secret],
)
def run_offline_eval(
    checkpoint_repo_id: str,
    dataset_repo_id: str,
    episodes: list[int],
    start_frame: int = 0,
    chunk_size: int = 50,
) -> dict:
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.utils import prepare_observation_for_inference
    from safetensors.torch import load_file

    def image_tensor_to_uint8_hwc(tensor):
        arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return arr

    def load_checkpoint_processor_stats(path: Path) -> dict:
        stats_file = path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        if not stats_file.exists():
            stats_file = path / "policy_preprocessor_step_2_normalizer_processor.safetensors"
        flat_stats = load_file(str(stats_file))
        nested: dict[str, dict] = {}
        for key, value in flat_stats.items():
            if "." not in key:
                continue
            feature, stat_name = key.rsplit(".", 1)
            nested.setdefault(feature, {})[stat_name] = value
        return nested

    device = torch.device("cuda")
    checkpoint_path = Path(snapshot_download(repo_id=checkpoint_repo_id, repo_type="model"))
    dataset_root = Path(snapshot_download(repo_id=dataset_repo_id, repo_type="dataset"))
    stats = load_checkpoint_processor_stats(checkpoint_path)
    action_q01 = torch.as_tensor(stats["action"]["q01"], dtype=torch.float32, device=device)
    action_q99 = torch.as_tensor(stats["action"]["q99"], dtype=torch.float32, device=device)

    ds_meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    policy = make_policy(
        cfg=PI05Config(
            pretrained_path=str(checkpoint_path),
            device="cuda",
            chunk_size=chunk_size,
            n_action_steps=chunk_size,
            rtc_config=RTCConfig(enabled=True, execution_horizon=40),
        ),
        ds_meta=ds_meta,
    )
    policy.eval()
    policy.to(device)

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint_path),
        dataset_stats=stats,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "normalizer_processor": {
                "stats": stats,
                "features": {
                    **policy.config.input_features,
                    **policy.config.output_features,
                },
                "norm_map": policy.config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )

    names = list(getattr(policy.config, "action_feature_names", []) or [])
    if not names:
        names = [f"dim_{i}" for i in range(len(action_q01))]

    print(f"checkpoint={checkpoint_repo_id}")
    print(f"dataset={dataset_repo_id}")
    print(f"episodes={episodes} start_frame={start_frame} chunk_size={chunk_size}")
    print(f"action_q01={action_q01.cpu().numpy().round(3).tolist()}")
    print(f"action_q99={action_q99.cpu().numpy().round(3).tolist()}")

    results = []
    for ep in episodes:
        ds = LeRobotDataset(dataset_repo_id, root=dataset_root, episodes=[ep], video_backend="pyav")
        if len(ds) <= start_frame + 1:
            print(f"episode {ep}: too short, skipping")
            continue

        sample = ds[start_frame]
        observation = {
            "observation.state": sample["observation.state"].detach().cpu().numpy().astype(np.float32)
        }
        for cam in ("wrist", "top"):
            key = f"observation.images.{cam}"
            if key in sample:
                observation[key] = image_tensor_to_uint8_hwc(sample[key])

        prepared = prepare_observation_for_inference(
            observation,
            device,
            task=sample.get("task", "Pour water from one plastic bottle into another."),
            robot_type="omx_follower",
        )
        prepared = preprocessor(prepared)

        policy.reset()
        normalized = policy.predict_action_chunk(
            prepared,
            prev_chunk_left_over=None,
            inference_delay=0,
        )
        denom = action_q99 - action_q01
        denom = torch.where(denom == 0, torch.tensor(1e-8, device=device), denom)
        pred = ((normalized + 1.0) * denom / 2.0 + action_q01).squeeze(0).detach().cpu().numpy()

        n = min(chunk_size, len(ds) - start_frame)
        target = np.stack([ds[start_frame + i]["action"].detach().cpu().numpy() for i in range(n)])
        pred_n = pred[:n]
        err = pred_n - target
        mae = np.abs(err).mean(axis=0)
        max_abs = np.abs(err).max(axis=0)
        first_delta = pred_n[0] - target[0]

        print(f"\nEP {ep} len={len(ds)} compare_n={n}")
        print("  recorded_first:", dict(zip(names, np.round(target[0], 2), strict=False)))
        print("  predicted_first:", dict(zip(names, np.round(pred_n[0], 2), strict=False)))
        print("  first_delta:", dict(zip(names, np.round(first_delta, 2), strict=False)))
        print("  mae:", dict(zip(names, np.round(mae, 2), strict=False)))
        print("  max_abs:", dict(zip(names, np.round(max_abs, 2), strict=False)))

        results.append(
            {
                "episode": ep,
                "length": len(ds),
                "compare_n": n,
                "recorded_first": target[0].round(4).tolist(),
                "predicted_first": pred_n[0].round(4).tolist(),
                "first_delta": first_delta.round(4).tolist(),
                "mae": mae.round(4).tolist(),
                "max_abs": max_abs.round(4).tolist(),
            }
        )

    return {
        "checkpoint_repo_id": checkpoint_repo_id,
        "dataset_repo_id": dataset_repo_id,
        "episodes": episodes,
        "start_frame": start_frame,
        "action_feature_names": names,
        "results": results,
    }


@app.local_entrypoint()
def main(
    checkpoint_repo_id: str,
    dataset_repo_id: str,
    episodes: str = "50,55,60,65",
    start_frame: int = 0,
    chunk_size: int = 50,
):
    episode_list = [int(part.strip()) for part in episodes.split(",") if part.strip()]
    result = run_offline_eval.remote(
        checkpoint_repo_id,
        dataset_repo_id,
        episode_list,
        start_frame,
        chunk_size,
    )
    print("\nSUMMARY")
    for row in result["results"]:
        print(
            f"ep={row['episode']} first_delta={row['first_delta']} "
            f"mae={row['mae']} max_abs={row['max_abs']}"
        )
