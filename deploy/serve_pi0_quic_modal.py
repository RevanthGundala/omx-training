"""serve_pi0_quic_modal.py — Modal GPU PI0.5 server over QUIC.

Mirrors ``serve_pi0_modal.py`` but replaces the FastAPI HTTP endpoints
with a QUIC connection established via STUN + UDP hole punching.

Wire format: each request is a JSON-encoded blob carrying the same
fields as the HTTP ``predict`` payload (state, task, image_*, etc.)
plus an optional ``op`` field (``predict``, ``reset``, ``health``).
Response is a JSON-encoded blob with the same shape as the HTTP
response.

Usage:
  modal run --detach deploy/serve_pi0_quic_modal.py::serve \\
      --session-id my-omx-session
  python evaluation/eval_pi0_quic.py --session-id my-omx-session
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
CRATE_DIR = REPO_ROOT / "omx_quic"

DATASET_REPO_ID = "RevanthGundala/003-pour-water"
CHECKPOINT_STEP = "003000"

app = modal.App("omx-pi05-quic")

hf_secret = modal.Secret.from_name("huggingface")
vol = modal.Volume.from_name("omx-pi0-training-logs", create_if_missing=True)

pi05_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "curl", "build-essential", "pkg-config")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --default-toolchain stable --profile minimal",
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .pip_install(
        "torch",
        "torchvision",
        "lerobot[pi]",
        "numpy",
        "av",
        "maturin>=1.7,<2.0",
    )
    .add_local_dir(str(CRATE_DIR), remote_path="/build/omx_quic", copy=True)
    .run_commands(
        "cd /build/omx_quic && maturin build --release --out /build/wheels",
        "pip install /build/wheels/*.whl",
    )
)


@app.function(
    image=pi05_image,
    gpu="A10G",
    timeout=3600,
    region="us-west",
    volumes={"/workspace/outputs": vol},
    secrets=[hf_secret],
)
def serve(session_id: str, stun_server: str = "stun.l.google.com:19302") -> dict:
    """One-shot server: rendezvous with client, accept QUIC, run inference loop."""
    import os
    import time
    import base64

    import cv2
    import numpy as np
    import torch
    from copy import copy
    from huggingface_hub import snapshot_download

    import omx_quic
    from omx_quic import rendezvous

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    vol.reload()

    checkpoint_path = Path(
        f"/workspace/outputs/checkpoints/{CHECKPOINT_STEP}/pretrained_model"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

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
    from lerobot.policies.utils import prepare_observation_for_inference

    print(f"[server] loading PI0.5 from {checkpoint_path} ...")
    ds_meta = LeRobotDatasetMetadata(DATASET_REPO_ID, root=dataset_root)
    device = torch.device("cuda")
    policy = make_policy(
        cfg=PI05Config(
            pretrained_path=str(checkpoint_path),
            device="cuda",
            chunk_size=50,
            n_action_steps=50,
            rtc_config=RTCConfig(enabled=True, execution_horizon=40),
        ),
        ds_meta=ds_meta,
    )
    policy.eval()
    policy.to(device)

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path="lerobot/pi05_base",
        dataset_stats=ds_meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "normalizer_processor": {
                "stats": ds_meta.stats,
                "features": {
                    **policy.config.input_features,
                    **policy.config.output_features,
                },
                "norm_map": policy.config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": ds_meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )

    action_q01 = torch.as_tensor(
        ds_meta.stats["action"]["q01"], dtype=torch.float32
    ).to(device)
    action_q99 = torch.as_tensor(
        ds_meta.stats["action"]["q99"], dtype=torch.float32
    ).to(device)

    state = {
        "prev_chunk": None,
        "steps_since_predict": 0,
    }

    def decode_image(b64_str: str, shape=None):
        img_bytes = base64.b64decode(b64_str)
        if shape is not None:
            return np.frombuffer(img_bytes, dtype=np.uint8).reshape(shape).copy()
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def handle_predict(payload: dict) -> dict:
        observation = {}
        observation["observation.state"] = np.array(payload["state"], dtype=np.float32)
        for cam_name in ("wrist", "top"):
            img_key = f"image_{cam_name}"
            if img_key in payload:
                observation[f"observation.images.{cam_name}"] = decode_image(
                    payload[img_key], payload.get(f"image_{cam_name}_shape")
                )

        steps_executed = payload.get("steps_executed", state["steps_since_predict"])

        observation = copy(observation)
        observation = prepare_observation_for_inference(
            observation, device,
            task=payload.get("task", "Pour water from one plastic bottle into another."),
            robot_type=payload.get("robot_type", "omx_follower"),
        )
        observation = preprocessor(observation)

        prev_left_over = None
        if state["prev_chunk"] is not None:
            prev_left_over = state["prev_chunk"][:, steps_executed:, :]

        actions = policy.predict_action_chunk(
            observation,
            prev_chunk_left_over=prev_left_over,
            inference_delay=steps_executed,
        )
        state["prev_chunk"] = actions.clone().detach()
        state["steps_since_predict"] = 0

        denom = action_q99 - action_q01
        denom = torch.where(denom == 0, torch.tensor(1e-8, device=denom.device), denom)
        actions_to_send = (actions + 1.0) * denom / 2.0 + action_q01

        return {
            "actions": actions_to_send.squeeze(0).cpu().numpy().tolist(),
            "debug": {
                "inference_delay": steps_executed,
                "prev_chunk_exists": state["prev_chunk"] is not None,
                "prev_left_over_shape": list(prev_left_over.shape)
                if prev_left_over is not None
                else None,
            },
        }

    def handler(request_bytes: bytes) -> bytes:
        t0 = time.perf_counter()
        op = "?"
        ok = True
        try:
            payload = json.loads(request_bytes.decode("utf-8"))
            op = payload.get("op", "predict")
            if op == "predict":
                resp = handle_predict(payload)
            elif op == "reset":
                policy.reset()
                state["prev_chunk"] = None
                state["steps_since_predict"] = 0
                resp = {"status": "ok"}
            elif op == "health":
                resp = {"status": "ready"}
            else:
                resp = {"error": f"unknown op {op!r}"}
        except Exception as e:
            import traceback
            ok = False
            resp = {"error": str(e), "traceback": traceback.format_exc()}
        out = json.dumps(resp).encode("utf-8")
        dt_ms = (time.perf_counter() - t0) * 1000
        status = "OK" if ok else "ERR"
        print(f"  QUIC {op:<7} -> {status}  "
              f"(in={len(request_bytes)/1024:.1f}KB out={len(out)/1024:.1f}KB "
              f"exec={dt_ms:.1f}ms)", flush=True)
        return out

    print(f"[server] PI0.5 ready. Entering accept loop for session={session_id!r}")
    print(f"[server] (multiple eval runs can connect back-to-back without "
          f"reloading the model)")

    total_requests = 0
    session_count = 0

    while True:
        session_count += 1
        # Fresh socket / QuicServer per session: punched socket is
        # consumed by quinn after listen().
        server = omx_quic.QuicServer(session_id)
        print(f"\n[server] === session #{session_count} ===")
        print(f"[server] local UDP port: {server.local_port()}")

        pub_ip, pub_port = server.discover_public_address(stun_server)
        print(f"[server] public address: {pub_ip}:{pub_port}")

        pub_ip2, pub_port2 = server.discover_public_address(stun_server)
        if (pub_ip2, pub_port2) != (pub_ip, pub_port):
            raise RuntimeError(
                f"[server] symmetric NAT detected: STUN1={pub_ip}:{pub_port} "
                f"STUN2={pub_ip2}:{pub_port2}. Hole punching cannot work from "
                f"this Modal region/cloud. Try a different region= pin or cloud=."
            )
        print(f"[server] cone-NAT confirmed.")

        rendezvous.publish(session_id, "server", pub_ip, pub_port)
        print(f"[server] waiting for client (rendezvous dict 'omx-quic-rendezvous-{session_id}') ...")
        try:
            peer_ip, peer_port = rendezvous.wait_for_peer(
                session_id, "server", timeout_s=600.0,
            )
        except TimeoutError:
            print(f"[server] no client in 10min, exiting.")
            rendezvous.clear(session_id, "server")
            break
        print(f"[server] peer: {peer_ip}:{peer_port}")
        server.set_peer_address(peer_ip, peer_port)

        try:
            sent, received, elapsed = server.punch(timeout_s=15.0)
            print(f"[server] punch ok: sent={sent} received={received} elapsed={elapsed:.3f}s")
            print("[server] accepting QUIC connection ...")
            server.listen(timeout_s=30.0)
            print("[server] QUIC connection up. Reset RTC state. Serving requests.")
            # Reset RTC state for the new session.
            policy.reset()
            state["prev_chunk"] = None
            state["steps_since_predict"] = 0

            t0 = time.perf_counter()
            n = server.serve_forever(handler)
            dt = time.perf_counter() - t0
            print(f"[server] session #{session_count} closed: "
                  f"{n} requests in {dt:.1f}s")
            total_requests += int(n)
        except Exception as e:
            print(f"[server] session #{session_count} failed: {e}")
        finally:
            rendezvous.clear(session_id, "server")

        print(f"[server] looping back to accept next eval run...")

    return {"requests_handled": total_requests, "sessions": session_count}


@app.local_entrypoint()
def main(session_id: str = "omx-default", stun_server: str = "stun.l.google.com:19302"):
    print(f"Launching Modal QUIC server (session_id={session_id!r}). "
          "Run eval_pi0_quic.py with the same --session-id.")
    result = serve.remote(session_id, stun_server)
    print(f"\nDone: {result}")
