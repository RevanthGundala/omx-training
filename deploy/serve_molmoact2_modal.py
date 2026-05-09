"""Modal GPU inference server for AllenAI MolmoAct2.

This preserves the existing PI0.5 eval client's HTTP contract:
``POST /predict`` accepts state/task/base64 camera images and returns an action
chunk as ``{"actions": [[...], ...], "debug": {...}}``.

Deploy:
    modal deploy deploy/serve_molmoact2_modal.py

Dev:
    modal serve deploy/serve_molmoact2_modal.py
"""

from __future__ import annotations

import modal


DEFAULT_REPO_ID = "allenai/MolmoAct2-SO100_101"
DEFAULT_NORM_TAG = "so100_so101_molmoact2"
DEFAULT_ACTION_DIM = 6

app = modal.App("omx-molmoact2-eval")
hf_secret = modal.Secret.from_name("huggingface")

molmoact2_image = (
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
        "fastapi[standard]",
    )
)


@app.cls(
    image=molmoact2_image,
    gpu="A100-80GB",
    scaledown_window=300,
    timeout=900,
    min_containers=1,
    secrets=[hf_secret],
)
class MolmoAct2Server:
    @modal.enter()
    def load_model(self):
        import os
        import time
        from pathlib import Path

        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor

        os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
        torch.set_float32_matmul_precision("high")

        self.repo_id = os.environ.get("MOLMOACT2_REPO_ID", DEFAULT_REPO_ID)
        self.norm_tag = os.environ.get("MOLMOACT2_NORM_TAG", DEFAULT_NORM_TAG)
        self.num_steps = int(os.environ.get("MOLMOACT2_NUM_STEPS", "10"))
        self.action_dim = int(os.environ.get("MOLMOACT2_ACTION_DIM", str(DEFAULT_ACTION_DIM)))

        t0 = time.perf_counter()
        self.checkpoint_path = Path(snapshot_download(repo_id=self.repo_id, repo_type="model"))
        self.processor = AutoProcessor.from_pretrained(
            str(self.checkpoint_path),
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(self.checkpoint_path),
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to("cuda").eval()
        self.load_seconds = time.perf_counter() - t0

        print(
            "MolmoAct2 ready "
            f"repo={self.repo_id} norm_tag={self.norm_tag} "
            f"path={self.checkpoint_path} load={self.load_seconds:.3f}s"
        )

    @staticmethod
    def _decode_image(b64_str: str):
        import base64
        import io

        from PIL import Image

        img_bytes = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def _payload_images(self, payload: dict):
        images = []
        # SO100_101 says order is not fixed, but use a stable order for
        # reproducibility and to match existing OMX camera naming.
        for cam_name in ("top", "wrist"):
            img_key = f"image_{cam_name}"
            if img_key in payload:
                images.append(self._decode_image(payload[img_key]))
        if not images:
            raise ValueError("payload must include at least one image_top or image_wrist")
        return images

    @modal.fastapi_endpoint(method="POST")
    def predict(self, payload: dict):
        import time

        import numpy as np
        import torch

        if "state" not in payload:
            raise ValueError("payload missing required field: state")

        state = np.asarray(payload["state"], dtype=np.float32)
        if state.ndim != 1:
            raise ValueError(f"state must be 1-D, got shape {state.shape}")
        if state.shape[0] != self.action_dim:
            raise ValueError(
                f"MolmoAct2 {self.repo_id} expects {self.action_dim} state dims, "
                f"got {state.shape[0]}"
            )

        images = self._payload_images(payload)
        task = payload.get("task", "Pour water from one plastic bottle into another.")

        start = time.perf_counter()
        with torch.inference_mode():
            out = self.model.predict_action(
                processor=self.processor,
                images=images,
                task=task,
                state=state,
                norm_tag=self.norm_tag,
                action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=self.num_steps,
                normalize_language=True,
                enable_cuda_graph=True,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        actions = out.actions
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        elif actions.ndim == 1:
            actions = actions.reshape(1, -1)

        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(
                f"expected actions with shape (T, {self.action_dim}), got {actions.shape}"
            )

        return {
            "actions": actions.tolist(),
            "debug": {
                "model": "molmoact2",
                "repo_id": self.repo_id,
                "norm_tag": self.norm_tag,
                "num_steps": self.num_steps,
                "action_shape": list(actions.shape),
                "inference_ms": round(elapsed_ms, 1),
                "action_min": float(actions.min()),
                "action_max": float(actions.max()),
                # Present for compatibility with eval_pi0.py debug logging.
                "inference_delay": payload.get("inference_delay"),
                "prev_steps_consumed": payload.get("prev_steps_consumed"),
                "prev_chunk_exists": False,
                "prev_left_over_shape": None,
            },
        }

    @modal.fastapi_endpoint(method="POST")
    def reset(self):
        return {"status": "ok"}

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {
            "status": "ready",
            "model": "molmoact2",
            "repo_id": self.repo_id,
            "norm_tag": self.norm_tag,
            "load_seconds": round(self.load_seconds, 3),
        }
