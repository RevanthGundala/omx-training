"""eval_pi0_quic.py — Run PI0.5 live on OMX follower via QUIC remote inference.

Mirrors ``eval_pi0.py`` but uses ``omx_quic.QuicClient`` instead of HTTP.
Run ``deploy/serve_pi0_quic_modal.py`` first with the same ``--session-id``.
"""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import threading
import time

import cv2
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
from evaluation.eval_pi0 import RTC_IDLE_SLEEP_S, RTC_QUEUE_REFILL_THRESHOLD

START_DELAY_S = 3


class EvalVideoRecorder:
    def __init__(self, run_dir: Path, fps: int):
        self.run_dir = run_dir
        self.fps = fps
        self.writers: dict[str, cv2.VideoWriter] = {}

    def write(self, observation: dict) -> None:
        for cam_name in CAMERAS:
            if cam_name not in observation:
                continue
            rgb = observation[cam_name]
            height, width = rgb.shape[:2]
            writer = self.writers.get(cam_name)
            if writer is None:
                path = self.run_dir / f"{cam_name}.mp4"
                writer = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(self.fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open eval video writer: {path}")
                self.writers[cam_name] = writer
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        for writer in self.writers.values():
            writer.release()
        self.writers.clear()


def transcode_eval_videos_for_browser(run_dir: Path) -> None:
    for camera in CAMERAS:
        path = run_dir / f"{camera}.mp4"
        if not path.exists():
            continue
        tmp = run_dir / f"{camera}.h264.tmp.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(path),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                str(tmp),
            ],
            check=True,
        )
        tmp.replace(path)


class EvalRunLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.requests_dir = run_dir / "inference_requests"
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._request_index = 0
        self._control_file = (run_dir / "control_steps.csv").open("w", newline="", encoding="utf-8")
        self._request_file = (run_dir / "inference_requests.csv").open("w", newline="", encoding="utf-8")
        self._control_writer: csv.DictWriter | None = None
        self._request_writer = csv.DictWriter(
            self._request_file,
            fieldnames=[
                "request_id",
                "step",
                "t_monotonic",
                "rtt_ms",
                "request_kb",
                "response_kb",
                "queue_before",
                "request_steps_consumed",
                "estimated_delay",
                "prev_steps_consumed",
                "actual_skip",
                "qsize_after",
                "server_inference_delay",
                "server_prev_steps_consumed",
                "server_prev_chunk_exists",
                *[f"first_action_{name}" for name in JOINT_NAMES],
            ],
        )
        self._request_writer.writeheader()

    def log_control_step(
        self,
        *,
        step: int,
        time_s: float,
        hz: float,
        qsize: int,
        observation_state: np.ndarray,
        policy_action: np.ndarray,
        sent_action: dict,
    ) -> None:
        row = {
            "step": step,
            "time_s": time_s,
            "hz": hz,
            "qsize": qsize,
        }
        for i, name in enumerate(JOINT_NAMES):
            row[f"state_{name}"] = float(observation_state[i])
            row[f"policy_action_{name}"] = float(policy_action[i])
            row[f"sent_action_{name}"] = float(sent_action[f"{name}.pos"])

        if self._control_writer is None:
            self._control_writer = csv.DictWriter(self._control_file, fieldnames=list(row))
            self._control_writer.writeheader()
        self._control_writer.writerow(row)
        if step % 30 == 0:
            self._control_file.flush()

    def log_inference_request(
        self,
        *,
        step: int,
        request_payload: bytes,
        response: dict,
        rtt_ms: float,
        queue_before: int,
        request_steps_consumed: int,
        estimated_delay: int,
        prev_steps_consumed: int,
        actual_skip: int,
        qsize_after: int,
    ) -> None:
        with self._lock:
            request_id = self._request_index
            self._request_index += 1

        request_dir = self.requests_dir / f"request-{request_id:04d}"
        request_dir.mkdir(parents=True, exist_ok=True)

        payload = json.loads(request_payload)
        for cam_name in CAMERAS:
            key = f"image_{cam_name}"
            if key in payload:
                image_path = request_dir / f"{cam_name}.jpg"
                image_path.write_bytes(base64.b64decode(payload[key]))
                payload[key] = image_path.name

        np.save(request_dir / "state.npy", np.asarray(payload["state"], dtype=np.float32))
        actions = np.asarray(response.get("actions", []), dtype=np.float32)
        np.save(request_dir / "actions.npy", actions)
        (request_dir / "request.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (request_dir / "response.json").write_text(json.dumps(response, indent=2), encoding="utf-8")

        debug = response.get("debug", {})
        row = {
            "request_id": request_id,
            "step": step,
            "t_monotonic": time.perf_counter(),
            "rtt_ms": rtt_ms,
            "request_kb": len(request_payload) / 1024,
            "response_kb": len(json.dumps(response).encode("utf-8")) / 1024,
            "queue_before": queue_before,
            "request_steps_consumed": request_steps_consumed,
            "estimated_delay": estimated_delay,
            "prev_steps_consumed": prev_steps_consumed,
            "actual_skip": actual_skip,
            "qsize_after": qsize_after,
            "server_inference_delay": debug.get("inference_delay"),
            "server_prev_steps_consumed": debug.get("prev_steps_consumed"),
            "server_prev_chunk_exists": debug.get("prev_chunk_exists"),
        }
        if actions.size:
            for i, name in enumerate(JOINT_NAMES):
                row[f"first_action_{name}"] = float(actions[0, i])
        with self._lock:
            self._request_writer.writerow(row)
            self._request_file.flush()

    def close(self) -> None:
        self._control_file.close()
        self._request_file.close()


def _build_follower():
    return create_follower(camera=True)


def _build_payload(
    observation_frame,
    observation,
    inference_delay: int,
    prev_steps_consumed: int,
) -> bytes:
    import cv2
    state = np.asarray(observation_frame["observation.state"]).tolist()
    payload = {
        "op": "predict",
        "state": state,
        "task": TASK_NAME,
        "robot_type": "omx_follower",
        # Keep steps_executed for older servers; newer servers distinguish
        # chunk alignment from future execution delay.
        "steps_executed": inference_delay,
        "inference_delay": inference_delay,
        "prev_steps_consumed": prev_steps_consumed,
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
    try:
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
        return client
    except Exception:
        try:
            client.close()
        except Exception:
            pass
        raise
    finally:
        rendezvous.clear(session_id, "client")


def main():
    parser = argparse.ArgumentParser(
        description="PI0.5 eval on OMX follower over QUIC remote inference"
    )
    parser.add_argument("--session-id", type=str, required=True)
    parser.add_argument("--stun", type=str, default="stun.l.google.com:19302")
    parser.add_argument(
        "--eval-output-root",
        type=Path,
        default=Path("outputs/eval_runs"),
        help="Directory where each eval run saves videos, metadata, and debug CSV.",
    )
    parser.add_argument(
        "--no-save-videos",
        action="store_true",
        help="Disable per-camera MP4 recording for this eval run.",
    )
    parser.add_argument("--notes", type=str, default="", help="Optional scene notes saved to metadata.json.")
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

    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.session_id}"
    run_dir = args.eval_output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "session_id": args.session_id,
                "stun": args.stun,
                "fps": FPS,
                "cameras": list(CAMERAS.keys()),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "outcome_label": "",
                "notes": args.notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    video_recorder = None if args.no_save_videos else EvalVideoRecorder(run_dir, FPS)
    run_logger = EvalRunLogger(run_dir)
    print(f"Eval artifacts will be saved under: {run_dir}")

    try:
        quic_client.request(json.dumps({"op": "reset"}).encode("utf-8"), 10.0)
    except Exception:
        pass

    action_queue = RTCActionQueue()

    obs_lock = threading.Lock()
    latest_observation = None
    latest_observation_frame = None
    latest_step = -1
    inference_running = threading.Event()
    stop_event = threading.Event()

    def _inference_loop():
        nonlocal latest_observation, latest_observation_frame, latest_step
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
            queue_size, request_steps_consumed = action_queue.request_snapshot()
            if not first_call and queue_size > RTC_QUEUE_REFILL_THRESHOLD:
                time.sleep(RTC_IDLE_SLEEP_S)
                continue

            with obs_lock:
                obs_snapshot = {k: v for k, v in latest_observation.items()}
                frame_snapshot = latest_observation_frame
                step_snapshot = latest_step
            estimated_delay = max(0, min(last_round_trip_delay, action_queue.chunk_size - 1))
            prev_steps_consumed = max(
                0, min(request_steps_consumed, action_queue.chunk_size - 1)
            )
            if first_call:
                print("Sending first QUIC inference request ...")
            inference_running.set()
            try:
                req = _build_payload(
                    frame_snapshot,
                    obs_snapshot,
                    estimated_delay,
                    prev_steps_consumed,
                )
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
                      f"server_delay={dbg.get('inference_delay')} "
                      f"server_prev_steps={dbg.get('prev_steps_consumed')} "
                      f"prev_chunk={dbg.get('prev_chunk_exists')}")
                actions = np.array(data["actions"], dtype=np.float32)
                actual_skip = action_queue.replace_after_request(actions, request_steps_consumed)
                run_logger.log_inference_request(
                    step=step_snapshot,
                    request_payload=req,
                    response=data,
                    rtt_ms=rtt_ms,
                    queue_before=queue_size,
                    request_steps_consumed=request_steps_consumed,
                    estimated_delay=estimated_delay,
                    prev_steps_consumed=prev_steps_consumed,
                    actual_skip=actual_skip,
                    qsize_after=action_queue.qsize(),
                )
                print(
                    f"  [rtc] prev_steps={prev_steps_consumed} "
                    f"est_delay={estimated_delay} actual_skip={actual_skip} "
                    f"qsize_after={action_queue.qsize()}"
                )
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

        while True:
            loop_start = time.perf_counter()
            observation = follower.get_observation()
            for cam_name in CAMERAS:
                if cam_name in observation:
                    ensure_camera_size(observation, key=cam_name)
            if video_recorder is not None:
                video_recorder.write(observation)

            observation_frame = build_dataset_frame(
                dataset_features, observation, prefix="observation",
            )
            with obs_lock:
                latest_observation = observation
                latest_observation_frame = observation_frame
                latest_step = step

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

            maintain_fps(loop_start, FPS)
            loop_dt = time.perf_counter() - loop_start
            hz = 1.0 / loop_dt if loop_dt > 0 else float("inf")
            observation_state = np.asarray(observation_frame["observation.state"], dtype=np.float32)
            run_logger.log_control_step(
                step=step,
                time_s=time.perf_counter() - run_start,
                hz=hz,
                qsize=action_queue.qsize(),
                observation_state=observation_state,
                policy_action=action_values,
                sent_action=sent_action,
            )

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

    except KeyboardInterrupt:
        print("\n\nStopping Pi0 QUIC eval...")
    finally:
        stop_event.set()
        try:
            quic_client.close()
        except Exception:
            pass
        inference_thread.join(timeout=5)
        run_logger.close()
        if video_recorder is not None:
            video_recorder.close()
            transcode_eval_videos_for_browser(run_dir)
        print(f"Eval artifacts saved to {run_dir}")
        safe_disconnect(follower)
        print("Disconnected. Done!")


if __name__ == "__main__":
    main()
