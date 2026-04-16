"""
serve_pi0_modal.py — Modal GPU inference server for Pi0 policy.

Deploys Pi0 on an A10G GPU and exposes a /predict endpoint.
The local eval script sends observations and receives action chunks.

Deploy:  modal deploy serve_pi0_modal.py
Dev:     modal serve serve_pi0_modal.py
"""

import modal

app = modal.App("omx-pi0")

HF_REPO_ID = "lerobot/pi0"
TASK_NAME = "Pick up remote and place it onto the gray circle"
MODEL_CACHE = "/root/model-cache"


hf_secret = modal.Secret.from_name("huggingface")


def download_model():
    """Download Pi0 weights at image build time so they're baked into the image."""
    import os
    os.environ["HF_HOME"] = MODEL_CACHE
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    PI0Policy.from_pretrained(HF_REPO_ID)


pi0_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "lerobot[pi0] @ git+https://github.com/ROBOTIS-GIT/lerobot.git@feature-omx-devel",
        "fastapi[standard]",
        "numpy",
        "pytest",
    )
    .run_function(download_model, secrets=[hf_secret])
)


@app.cls(
    image=pi0_image,
    gpu="A10G",
    scaledown_window=300,
    timeout=600,
)
class Pi0Server:
    @modal.enter()
    def load_model(self):
        import os
        os.environ["HF_HOME"] = MODEL_CACHE
        import torch
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        # OMX follower has 6 joints, camera is 3-channel
        NUM_JOINTS = 6
        NUM_CHANNELS = 3
        dataset_stats = {
            "observation.state": {
                "mean": torch.zeros(NUM_JOINTS),
                "std": torch.ones(NUM_JOINTS),
            },
            "observation.images.camera0": {
                "mean": torch.zeros(NUM_CHANNELS, 1, 1),
                "std": torch.ones(NUM_CHANNELS, 1, 1),
            },
            "action": {
                "mean": torch.zeros(NUM_JOINTS),
                "std": torch.ones(NUM_JOINTS),
            },
        }

        print("Loading Pi0 from cache...")
        self.device = torch.device("cuda")
        self.policy = PI0Policy.from_pretrained(HF_REPO_ID, dataset_stats=dataset_stats)

        # Override input features to match our single-camera OMX setup
        from lerobot.configs.types import PolicyFeature, FeatureType
        self.policy.config.input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(NUM_JOINTS,)),
            "observation.images.camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        }
        self.policy.config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(NUM_JOINTS,)),
        }

        self.policy.config.device = "cuda"
        self.policy.to(self.device)
        self.policy.eval()
        self.policy.reset()
        print("Pi0 ready on GPU.")

    @modal.fastapi_endpoint(method="POST")
    def predict(self, payload: dict):
        import numpy as np
        import torch
        from lerobot.utils.control_utils import predict_action

        observation = {}
        state = np.array(payload["state"], dtype=np.float32)
        observation["observation.state"] = state

        if "image" in payload:
            import base64
            img_bytes = base64.b64decode(payload["image"])
            img_array = np.frombuffer(img_bytes, dtype=np.uint8).reshape(
                payload["image_shape"]
            ).copy()
            observation["observation.images.camera0"] = img_array.transpose(2, 0, 1).astype(np.float32) / 255.0

        action_values = predict_action(
            observation,
            self.policy,
            self.device,
            self.policy.config.use_amp,
            task=payload.get("task", TASK_NAME),
            robot_type=payload.get("robot_type", "omx_follower"),
        )

        return {"actions": action_values.cpu().numpy().tolist()}

    @modal.fastapi_endpoint(method="POST")
    def reset(self):
        self.policy.reset()
        return {"status": "ok"}

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {"status": "ready"}
