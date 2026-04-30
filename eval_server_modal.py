"""
eval_server_modal.py — Modal web endpoint for PI0.5 inference.

Loads the finetuned PI0.5 checkpoint on a GPU and exposes a FastAPI
endpoint that accepts camera images + joint state and returns actions.

Usage:
    # Deploy the server (keeps running until stopped):
    modal deploy eval_server_modal.py

    # Or run temporarily:
    modal serve eval_server_modal.py
"""

import modal

REMOTE_WORKSPACE = "/workspace"

hf_secret = modal.Secret.from_name("huggingface")

vol = modal.Volume.from_name("omx-pi0-training-logs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch",
        "torchvision",
        "lerobot[pi]",
        "fastapi[standard]",
    )
    .pip_install("av")
)

app = modal.App("omx-pi05-eval", image=image)

CHECKPOINT_DIR = "/workspace/outputs/run/checkpoints/010000/pretrained_model"
DATASET_REPO_ID = "RevanthGundala/002-pour-water"


@app.cls(
    gpu="A10G",
    timeout=3600,
    volumes={"/workspace/outputs": vol},
    secrets=[hf_secret],
    keep_warm=1,
    allow_concurrent_inputs=1,
)
class PI05Inference:
    @modal.enter()
    def load_model(self):
        import os
        import torch
        from pathlib import Path
        from huggingface_hub import snapshot_download

        os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

        vol.reload()

        print(f"Loading checkpoint from {CHECKPOINT_DIR}")
        checkpoint_path = Path(CHECKPOINT_DIR)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_DIR}")

        # Download dataset for stats (needed by normalizer)
        dataset_root = Path("/tmp/dataset")
        snapshot_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            local_dir=dataset_root,
        )

        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.configs.types import PolicyFeature, FeatureType

        # Load config and create policy
        ds_meta = LeRobotDatasetMetadata(DATASET_REPO_ID, root=dataset_root)

        self.policy = make_policy(
            cfg=PI05Config(
                pretrained_path=str(checkpoint_path),
                device="cuda",
                chunk_size=50,
                n_action_steps=50,
            ),
            ds_meta=ds_meta,
        )
        self.policy.eval()
        self.policy.to("cuda")

        # Create pre/post processors for normalization
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path="lerobot/pi05_base",
            dataset_stats=ds_meta.stats,
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "normalizer_processor": {
                    "stats": ds_meta.stats,
                    "features": {
                        **self.policy.config.input_features,
                        **self.policy.config.output_features,
                    },
                    "norm_map": self.policy.config.normalization_mapping,
                },
            },
            postprocessor_overrides={
                "unnormalizer_processor": {
                    "stats": ds_meta.stats,
                    "features": self.policy.config.output_features,
                    "norm_map": self.policy.config.normalization_mapping,
                },
            },
        )

        print("Model loaded and ready for inference!")

    @modal.fastapi_app()
    def web(self):
        from fastapi import FastAPI
        import torch
        import numpy as np
        import base64
        from pydantic import BaseModel

        web_app = FastAPI()

        class InferenceRequest(BaseModel):
            state: list[float]  # 6 joint positions
            wrist_image_b64: str  # base64-encoded JPEG
            top_image_b64: str  # base64-encoded JPEG
            task: str
            image_width: int = 640
            image_height: int = 480

        class InferenceResponse(BaseModel):
            actions: list[float]  # 6 joint positions

        @web_app.get("/health")
        def health():
            return {"status": "ok"}

        @web_app.post("/predict", response_model=InferenceResponse)
        def predict(req: InferenceRequest):
            import cv2

            # Decode images from base64 JPEG
            def decode_image(b64_str: str) -> np.ndarray:
                img_bytes = base64.b64decode(b64_str)
                img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img

            wrist_img = decode_image(req.wrist_image_b64)
            top_img = decode_image(req.top_image_b64)

            # Resize if needed
            if wrist_img.shape[:2] != (req.image_height, req.image_width):
                wrist_img = cv2.resize(wrist_img, (req.image_width, req.image_height))
            if top_img.shape[:2] != (req.image_height, req.image_width):
                top_img = cv2.resize(top_img, (req.image_width, req.image_height))

            # Build observation dict matching training format
            state_tensor = torch.tensor(req.state, dtype=torch.float32).unsqueeze(0)

            def img_to_tensor(img: np.ndarray) -> torch.Tensor:
                t = torch.from_numpy(img).float() / 255.0
                t = t.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
                return t

            obs_dict = {
                "observation.state": state_tensor,
                "observation.images.wrist": img_to_tensor(wrist_img),
                "observation.images.top": img_to_tensor(top_img),
                "task": [req.task],
            }

            # Preprocess (normalize, tokenize task, move to device)
            obs_dict = self.preprocessor(obs_dict)

            with torch.inference_mode():
                action = self.policy.select_action(obs_dict)

            # Postprocess (unnormalize)
            action_dict = {"action": action}
            action_dict = self.postprocessor(action_dict)
            action_values = action_dict["action"].squeeze(0).cpu().tolist()

            # Return only the first action step (of chunk_size=50)
            if isinstance(action_values[0], list):
                action_values = action_values[0]

            return InferenceResponse(actions=action_values[:6])

        return web_app
