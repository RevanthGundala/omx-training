"""
serve_pi0_modal.py — Modal GPU inference server for PI0.5 finetuned policy.

Loads the finetuned PI0.5 checkpoint from the training volume and exposes
/predict, /reset, and /health endpoints.

Deploy:  modal deploy deploy/serve_pi0_modal.py
Dev:     modal serve deploy/serve_pi0_modal.py
"""

import modal

app = modal.App("omx-pi05-eval")

DATASET_REPO_ID = "RevanthGundala/003-pour-water"
CHECKPOINT_STEP = "003000"  # PI0.5 full finetune (3k steps, batch=32, ~163 epochs)

hf_secret = modal.Secret.from_name("huggingface")
vol = modal.Volume.from_name("omx-pi0-training-logs", create_if_missing=True)

pi05_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch",
        "torchvision",
        "lerobot[pi]",
        "fastapi[standard]",
        "numpy",
    )
    .pip_install("av")
)


@app.cls(
    image=pi05_image,
    gpu="A10G",
    scaledown_window=300,
    timeout=600,
    min_containers=1,
    volumes={"/workspace/outputs": vol},
    secrets=[hf_secret],
)
class Pi0Server:
    @modal.enter()
    def load_model(self):
        import os
        import torch
        from pathlib import Path
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
        vol.reload()

        checkpoint_path = Path(f"/workspace/outputs/checkpoints/{CHECKPOINT_STEP}/pretrained_model")
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                f"Available: {list(Path('/workspace/outputs').rglob('config.json'))}"
            )

        # Download dataset metadata for normalization stats
        dataset_root = Path("/tmp/dataset")
        snapshot_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            local_dir=dataset_root,
        )

        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        ds_meta = LeRobotDatasetMetadata(DATASET_REPO_ID, root=dataset_root)
        def load_checkpoint_processor_stats(path: Path) -> dict | None:
            stats_file = path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
            if not stats_file.exists():
                stats_file = path / "policy_preprocessor_step_2_normalizer_processor.safetensors"
            if not stats_file.exists():
                return None
            flat_stats = load_file(str(stats_file))
            nested: dict[str, dict] = {}
            for key, value in flat_stats.items():
                if "." not in key:
                    continue
                feature, stat_name = key.rsplit(".", 1)
                nested.setdefault(feature, {})[stat_name] = value
            return nested

        self.eval_stats = load_checkpoint_processor_stats(checkpoint_path) or ds_meta.stats
        self.stats_source = (
            "checkpoint_processor"
            if (checkpoint_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").exists()
            else "dataset_metadata"
        )
        print(f"Using normalization stats from {self.stats_source}.")

        print(f"Loading PI0.5 checkpoint from {checkpoint_path}...")
        self.device = torch.device("cuda")

        self.policy = make_policy(
            cfg=PI05Config(
                pretrained_path=str(checkpoint_path),
                device="cuda",
                chunk_size=50,
                n_action_steps=50,
                rtc_config=RTCConfig(enabled=True, execution_horizon=40),
            ),
            ds_meta=ds_meta,
        )
        self.policy.eval()
        self.policy.to(self.device)

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path="lerobot/pi05_base",
            dataset_stats=self.eval_stats,
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "normalizer_processor": {
                    "stats": self.eval_stats,
                    "features": {
                        **self.policy.config.input_features,
                        **self.policy.config.output_features,
                    },
                    "norm_map": self.policy.config.normalization_mapping,
                },
            },
            postprocessor_overrides={
                "unnormalizer_processor": {
                    "stats": self.eval_stats,
                    "features": self.policy.config.output_features,
                    "norm_map": self.policy.config.normalization_mapping,
                },
            },
        )

        # Cache action quantile stats on GPU for fast unnormalization
        action_stats = self.eval_stats["action"]
        self.action_q01 = torch.as_tensor(action_stats["q01"], dtype=torch.float32).to(self.device)
        self.action_q99 = torch.as_tensor(action_stats["q99"], dtype=torch.float32).to(self.device)

        # RTC state: track previous chunk for overlap guidance
        self.prev_chunk = None  # normalized actions from last prediction
        self.execution_horizon = self.policy.config.rtc_config.execution_horizon
        self.steps_since_predict = 0

        print("PI0.5 ready on GPU (RTC enabled, execution_horizon="
              f"{self.execution_horizon}).")

    @modal.fastapi_endpoint(method="POST")
    def predict(self, payload: dict):
        import numpy as np
        import torch
        import base64
        import cv2
        from copy import copy
        from lerobot.policies.utils import prepare_observation_for_inference

        def decode_image(b64_str, shape=None):
            img_bytes = base64.b64decode(b64_str)
            if shape is not None:
                return np.frombuffer(img_bytes, dtype=np.uint8).reshape(shape).copy()
            else:
                img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        observation = {}
        observation["observation.state"] = np.array(payload["state"], dtype=np.float32)

        for cam_name in ["wrist", "top"]:
            img_key = f"image_{cam_name}"
            shape_key = f"image_{cam_name}_shape"
            if img_key in payload:
                shape = payload.get(shape_key)
                img = decode_image(payload[img_key], shape)
                observation[f"observation.images.{cam_name}"] = img

        # RTC needs two different clocks:
        # - prev_steps_consumed: previous chunk index at observation/request time
        # - inference_delay: estimated actions consumed while inference is in flight
        inference_delay = int(payload.get(
            "inference_delay",
            payload.get("steps_executed", self.steps_since_predict),
        ))
        prev_steps_consumed = int(payload.get("prev_steps_consumed", inference_delay))

        # Preprocess once, run model once, unnormalize the full chunk
        observation = copy(observation)
        observation = prepare_observation_for_inference(
            observation, self.device,
            task=payload.get("task", "Pour water from one plastic bottle into another."),
            robot_type=payload.get("robot_type", "omx_follower"),
        )
        observation = self.preprocessor(observation)

        # Slice prev_chunk by actual steps_executed so leftover[0] aligns
        # temporally with new_chunk[0] (the action about to be executed now).
        prev_left_over = None
        if self.prev_chunk is not None:
            prev_left_over = self.prev_chunk[:, prev_steps_consumed:, :]

        # Single forward pass with RTC context
        # (predict_action_chunk has @torch.no_grad; RTC internally uses enable_grad)
        actions = self.policy.predict_action_chunk(
            observation,
            prev_chunk_left_over=prev_left_over,
            inference_delay=inference_delay,
        )
        # actions shape: (1, chunk_size, action_dim) — normalized

        # Store the full normalized chunk; we'll slice by actual delay at next call
        self.prev_chunk = actions.clone().detach()
        self.steps_since_predict = 0

        # Unnormalize the full chunk: QUANTILES inverse → (norm + 1) * (q99 - q01) / 2 + q01
        # Send ALL chunk_size actions so the client has a ~1.7s buffer at 30Hz.
        # execution_horizon controls RTC guidance weights, not output truncation.
        denom = self.action_q99 - self.action_q01
        denom = torch.where(denom == 0, torch.tensor(1e-8, device=denom.device), denom)
        actions_to_send = (actions + 1.0) * denom / 2.0 + self.action_q01

        # (1, chunk_size, action_dim) → (chunk_size, action_dim)
        return {
            "actions": actions_to_send.squeeze(0).cpu().numpy().tolist(),
            "debug": {
                "inference_delay": inference_delay,
                "prev_steps_consumed": prev_steps_consumed,
                "prev_chunk_exists": self.prev_chunk is not None,
                "prev_left_over_shape": list(prev_left_over.shape) if prev_left_over is not None else None,
            },
        }

    @modal.fastapi_endpoint(method="POST")
    def reset(self):
        self.policy.reset()
        self.prev_chunk = None
        self.steps_since_predict = 0
        return {"status": "ok"}

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {"status": "ready"}
