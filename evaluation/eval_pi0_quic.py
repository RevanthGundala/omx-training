"""eval_pi0_quic.py — Run PI0.5 live on OMX follower via QUIC remote inference.

Mirrors ``eval_pi0.py`` but uses ``omx_quic.QuicClient`` instead of HTTP.
Run ``deploy/serve_pi0_quic_modal.py`` first with the same ``--session-id``.
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time

import numpy as np
import rerun as rr

from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features

from utils.config import CAMERAS, FPS, JOINT_NAMES, TASK_NAME
from utils.control_utils import ensure_camera_size, maintain_fps
from utils.rerun_utils import init_rerun
from utils.robot_utils import create_follower, safe_disconnect

import omx_quic
from omx_quic import rendezvous

from evaluation.eval_pi0 import RTCActionQueue

START_DELAY_S = 3


def _build_follower():
    return create_follower(camera=True)


def _build_payload(observation_frame, observation, steps_executed: int) -> bytes:
    import cv2
    state = np.asarray(observation_frame["observation.state"]).tolist()
    payload = {
        "op": "predict",
        "state": state,
        "task": TASK_NAME,
        "robot_type": "omx_follower",
        "steps_executed": steps_executed,
    }
    for cam_name in CAMERAS:
        if cam_name in observation:
            img = observation[cam_name]
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            payload[f"image_{cam_name}"] = base64.b64encode(buf.tobytes()).decode("ascii")
    return json.dumps(payload).encode("utf-8")


def _connect_quic(session_id: str, stun_server: str) -> omx_quic.QuicClient:
    client = omx_quic.QuicClient(session_id)
    print(f"[client] local UDP port: {client.local_port()}")
    pub_ip, pub_port = client.discover_public_address(stun_server)
    print(f"[client] public address: {pub_ip}:{pub_port}")
    pub_ip2, pub_port2 = client.discover_public_address(stun_server)
    if (pub_ip2, pub_port2) != (pub_ip, pub_port):
        raise RuntimeError(
            f"[client] symmetric NAT detected: STUN1={pub_ip}:{pub_port} "
            f"STUN2={pub_ip2}:{pub_port2}. Hole punching cannot work from "
            "this network. Try a different network or a TURN relay."
        )
    rendezvous.publish(session_id, "client", pub_ip, pub_port)
    print(f"[client] waiting for server peer in rendezvous dict ...")
    peer_ip, peer_port = rendezvous.wait_for_peer(session_id, "client", timeout_s=300.0)
    print(f"[client] peer: {peer_ip}:{peer_port}")
    client.set_peer_address(peer_ip, peer_port)
    sent, received, elapsed = client.punch(timeout_s=15.0)
    print(f"[client] punch ok: sent={sent} received={received} elapsed={elapsed:.3f}s")
    print(f"[client] QUIC handshake ...")
    t0 = time.perf_counter()
    client.connect(timeout_s=30.0)
    print(f"[client] QUIC connected in {(time.perf_counter()-t0)*1000:.1f}ms")
    rendezvous.clear(session_id, "client")
    return client


def main():
    parser = argparse.ArgumentParser(
        description="PI0.5 eval on OMX follower over QUIC remote inference"
    )
    parser.add_argument("--session-id", type=str, required=True)
    parser.add_argument("--stun", type=str, default="stun.l.google.com:19302")
    args = parser.parse_args()

    print(f"Connecting QUIC for session={args.session_id!r} ...")
    quic_client = _connect_quic(args.session_id, args.stun)

    # Health probe.
    try:
        h = quic_client.request(json.dumps({"op": "health"}).encode("utf-8"), 10.0)
        print(f"[client] server health: {json.loads(h)}")
    except Exception as e:
        print(f"[client] health probe failed: {e}")

    follower = _build_follower()
    dataset_features = {
        **hw_to_dataset_features(follower.action_features, "action", use_video=False),
        **hw_to_dataset_features(follower.observation_features, "observation", use_video=False),
    }

    print("Connecting follower arm...")
    for attempt in range(1, 4):
        try:
            follower.connect(calibrate=False)
            break
        except (TimeoutError, RuntimeError) as e:
            print(f"  Camera connect attempt {attempt}/3 failed: {e}")
            if attempt == 3:
                raise
            if hasattr(follower, "disconnect"):
                follower.disconnect()
            follower = _build_follower()
            dataset_features = {
                **hw_to_dataset_features(follower.action_features, "action", use_video=False),
                **hw_to_dataset_features(follower.observation_features, "observation", use_video=False),
            }
            time.sleep(2)

    init_rerun("omx_eval_pi05_quic", save_rrd=True)

    try:
        quic_client.request(json.dumps({"op": "reset"}).encode("utf-8"), 10.0)
    except Exception:
        pass

    action_queue = RTCActionQueue()

    obs_lock = threading.Lock()
    latest_observation = None
    latest_observation_frame = None
    inference_running = threading.Event()
    stop_event = threading.Event()

    def _inference_loop():
        nonlocal latest_observation, latest_observation_frame
        while not stop_event.is_set():
            with obs_lock:
                obs = latest_observation
                obs_frame = latest_observation_frame
            if obs is not None and obs_frame is not None:
                break
            time.sleep(0.01)

        first_call = True
        last_round_trip_delay = 0
        while not stop_event.is_set():
            with obs_lock:
                obs_snapshot = {k: v for k, v in latest_observation.items()}
                frame_snapshot = latest_observation_frame
            steps_consumed_now = action_queue.steps_since_replace
            estimated_delay = steps_consumed_now + last_round_trip_delay
            if first_call:
                print("Sending first QUIC inference request ...")
            inference_running.set()
            try:
                req = _build_payload(frame_snapshot, obs_snapshot, estimated_delay)
                t0 = time.perf_counter()
                resp_bytes = quic_client.request(req, 30.0)
                rtt_ms = (time.perf_counter() - t0) * 1000
                data = json.loads(resp_bytes)
                if "error" in data:
                    print(f"\n⚠ Server error: {data['error']}")
                    continue
                dbg = data.get("debug", {})
                print(f"\n  [client] RTT={rtt_ms:6.1f}ms "
                      f"req={len(req)/1024:.1f}KB resp={len(resp_bytes)/1024:.1f}KB "
                      f"server_exec={dbg.get('inference_delay')}step_delay "
                      f"prev_chunk={dbg.get('prev_chunk_exists')}")
                actions = np.array(data["actions"], dtype=np.float32)
                actual_skip = action_queue.replace_atomic(actions)
                last_round_trip_delay = actual_skip
                if first_call:
                    print(f"First inference returned {len(actions)} actions. Robot active!")
                    first_call = False
            except Exception as e:
                print(f"\n⚠ Inference error: {e}")
            finally:
                inference_running.clear()

    inference_thread = threading.Thread(target=_inference_loop, daemon=True)
    inference_thread.start()

    try:
        print(f"Starting Pi0 QUIC eval in {START_DELAY_S}s. Press Ctrl+C to stop.")
        for remaining in range(START_DELAY_S, 0, -1):
            print(f"  {remaining}...")
            time.sleep(1)
        print("Running! Waiting for first inference...")

        run_start = time.perf_counter()
        step = 0
        frozen_joint_targets: dict[str, float] = {}
        debug_log = open("outputs/eval_quic_debug.csv", "w")

        while True:
            loop_start = time.perf_counter()
            observation = follower.get_observation()
            for cam_name in CAMERAS:
                if cam_name in observation:
                    ensure_camera_size(observation, key=cam_name)

            observation_frame = build_dataset_frame(
                dataset_features, observation, prefix="observation",
            )
            with obs_lock:
                latest_observation = observation
                latest_observation_frame = observation_frame

            action_values = action_queue.get()
            if action_values is None:
                maintain_fps(loop_start, FPS)
                continue

            action = {
                key: float(action_values[i])
                for i, key in enumerate(follower.action_features)
            }
            for name, target in frozen_joint_targets.items():
                action[f"{name}.pos"] = target

            sent_action = follower.send_action(action)

            loop_dt = time.perf_counter() - loop_start
            hz = 1.0 / loop_dt if loop_dt > 0 else float("inf")

            if step == 0:
                joint_headers = ",".join(JOINT_NAMES)
                debug_log.write(f"step,hz,qsize,{joint_headers}\n")
            vals = ",".join(f"{action_values[i]:.4f}" for i in range(len(JOINT_NAMES)))
            debug_log.write(f"{step},{hz:.1f},{action_queue.qsize()},{vals}\n")
            if step % 30 == 0:
                debug_log.flush()

            rr.set_time_sequence("step", step)
            rr.set_time_seconds("time", time.perf_counter() - run_start)
            rr.log("metrics/loop_hz", rr.Scalars(hz))
            for cam_name in CAMERAS:
                if cam_name in observation:
                    rr.log(f"camera/{cam_name}", rr.Image(observation[cam_name]))
            for i, name in enumerate(JOINT_NAMES):
                if "observation.state" in observation_frame:
                    state_val = observation_frame["observation.state"][i].item()
                    rr.log(f"joints/{name}/state", rr.Scalars(state_val))
                rr.log(f"joints/{name}/policy_action", rr.Scalars(float(action_values[i])))
                rr.log(f"joints/{name}/sent_action", rr.Scalars(sent_action[f"{name}.pos"]))

            step += 1
            queue_len = action_queue.qsize()
            inf_sym = "↻" if inference_running.is_set() else " "
            action_preview = "  |  ".join(
                f"{name}: {value:7.2f}" for name, value in sent_action.items()
            )
            print(f"Step {step:05d} | {hz:5.1f} Hz | Q:{queue_len:2d} {inf_sym}| {action_preview}", end="\r")
            maintain_fps(loop_start, FPS)

    except KeyboardInterrupt:
        print("\n\nStopping Pi0 QUIC eval...")
    finally:
        debug_log.close()
        print("Debug log saved to outputs/eval_quic_debug.csv")
        stop_event.set()
        inference_thread.join(timeout=5)
        try:
            quic_client.close()
        except Exception:
            pass
        safe_disconnect(follower)
        print("Disconnected. Done!")


if __name__ == "__main__":
    main()
