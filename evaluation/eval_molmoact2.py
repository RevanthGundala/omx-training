"""Dry-run-first MolmoAct2 eval client for the OMX follower.

This uses the same HTTP server contract as ``evaluation/eval_pi0.py`` but is
safe by default: it logs predicted MolmoAct2 actions and refuses to actuate
unless ``--actuate`` is passed. Actions are range-checked before any send.

Run the server first:
    modal serve deploy/serve_molmoact2_modal.py

Then dry-run:
    uv run python evaluation/eval_molmoact2.py --remote <predict-url>
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import rerun as rr

from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features

from evaluation.eval_pi0 import (
    RTCActionQueue,
    RTC_IDLE_SLEEP_S,
    _predict_remote,
)
from utils.config import CAMERAS, FPS, JOINT_NAMES
from utils.control_utils import ensure_camera_size, maintain_fps
from utils.rerun_utils import init_rerun
from utils.robot_utils import create_follower, safe_disconnect


START_DELAY_S = 3
DEFAULT_BODY_MIN = -100.0
DEFAULT_BODY_MAX = 100.0
DEFAULT_GRIPPER_MIN = 0.0
DEFAULT_GRIPPER_MAX = 100.0
MOLMOACT_QUEUE_REFILL_THRESHOLD = 10


def _build_follower():
    return create_follower(camera=True)


def _action_limits(action_features: dict) -> dict[str, tuple[float, float]]:
    limits = {}
    for key in action_features:
        if key == "gripper.pos":
            limits[key] = (DEFAULT_GRIPPER_MIN, DEFAULT_GRIPPER_MAX)
        else:
            limits[key] = (DEFAULT_BODY_MIN, DEFAULT_BODY_MAX)
    return limits


def _validate_action(action: dict[str, float], limits: dict[str, tuple[float, float]]):
    bad = []
    for key, value in action.items():
        lo, hi = limits[key]
        if not np.isfinite(value) or value < lo or value > hi:
            bad.append(f"{key}={value:.3f} outside [{lo:.1f}, {hi:.1f}]")
    if bad:
        raise ValueError("; ".join(bad))


def main():
    parser = argparse.ArgumentParser(description="MolmoAct2 eval on OMX follower")
    parser.add_argument("--remote", type=str, required=True, metavar="URL",
                        help="MolmoAct2 Modal server predict URL")
    parser.add_argument(
        "--actuate",
        action="store_true",
        help="Actually send actions to the robot. Without this, dry-run logs only.",
    )
    parser.add_argument(
        "--allow-out-of-range",
        action="store_true",
        help="Do not reject actions outside the default OMX percentage ranges.",
    )
    args = parser.parse_args()

    server_url = args.remote.rstrip("/")
    mode = "ACTUATING" if args.actuate else "DRY-RUN"
    print(f"Using MolmoAct2 remote server: {server_url}")
    print(f"Mode: {mode}")
    if args.actuate and args.allow_out_of_range:
        raise ValueError("--actuate and --allow-out-of-range cannot be combined")

    print("Warming up server (cold start can take several minutes)...")
    health_url = server_url.replace("-predict", "-health", 1)
    try:
        import requests

        health = requests.get(health_url, timeout=300)
        health.raise_for_status()
        print(f"Remote server is ready: {health.json()}")
    except Exception as e:
        print(f"Health check failed ({e}), proceeding anyway...")

    follower = _build_follower()
    dataset_features = {
        **hw_to_dataset_features(follower.action_features, "action", use_video=False),
        **hw_to_dataset_features(follower.observation_features, "observation", use_video=False),
    }
    limits = _action_limits(follower.action_features)

    print("Connecting follower arm/cameras...")
    for attempt in range(1, 4):
        try:
            follower.connect(calibrate=False)
            break
        except (TimeoutError, RuntimeError) as e:
            print(f"  Connect attempt {attempt}/3 failed: {e}")
            if attempt == 3:
                raise
            if hasattr(follower, "disconnect"):
                follower.disconnect()
            follower = _build_follower()
            dataset_features = {
                **hw_to_dataset_features(follower.action_features, "action", use_video=False),
                **hw_to_dataset_features(follower.observation_features, "observation", use_video=False),
            }
            limits = _action_limits(follower.action_features)
            time.sleep(2)

    init_rerun("omx_eval_molmoact2", save_rrd=True)

    reset_url = server_url.replace("-predict", "-reset", 1)
    try:
        import requests

        requests.post(reset_url, json={}, timeout=10)
    except Exception:
        pass

    action_queue = RTCActionQueue(chunk_size=30)
    latest_observation = None
    latest_observation_frame = None
    stop_event = False

    try:
        print(f"Starting MolmoAct2 {mode.lower()} in {START_DELAY_S}s. Press Ctrl+C to stop.")
        for remaining in range(START_DELAY_S, 0, -1):
            print(f"  {remaining}...")
            time.sleep(1)

        run_start = time.perf_counter()
        step = 0
        debug_log = open("outputs/eval_molmoact2_debug.csv", "w")
        joint_headers = ",".join(JOINT_NAMES)
        debug_log.write(f"step,hz,qsize,mode,range_ok,{joint_headers}\n")

        while True:
            loop_start = time.perf_counter()
            observation = follower.get_observation()
            for cam_name in CAMERAS:
                if cam_name in observation:
                    ensure_camera_size(observation, key=cam_name)

            latest_observation = observation
            latest_observation_frame = build_dataset_frame(
                dataset_features, observation, prefix="observation",
            )

            queue_size, request_steps_consumed = action_queue.request_snapshot()
            if queue_size <= MOLMOACT_QUEUE_REFILL_THRESHOLD:
                try:
                    actions = _predict_remote(
                        latest_observation_frame,
                        server_url,
                        latest_observation,
                        inference_delay=0,
                        prev_steps_consumed=request_steps_consumed,
                    )
                    if actions.ndim != 2 or actions.shape[1] != len(follower.action_features):
                        raise ValueError(
                            f"server returned actions shape {actions.shape}; "
                            f"expected (T, {len(follower.action_features)})"
                        )
                    action_queue.replace_after_request(actions, request_steps_consumed)
                except Exception as e:
                    print(f"\n⚠ Inference error: {e}")
                    time.sleep(RTC_IDLE_SLEEP_S)

            action_values = action_queue.get()
            if action_values is None:
                maintain_fps(loop_start, FPS)
                continue

            action = {
                key: float(action_values[i])
                for i, key in enumerate(follower.action_features)
            }

            range_ok = True
            try:
                if not args.allow_out_of_range:
                    _validate_action(action, limits)
            except ValueError as e:
                range_ok = False
                if args.actuate:
                    raise RuntimeError(f"Refusing to actuate unsafe MolmoAct2 action: {e}") from e
                print(f"\n⚠ Dry-run range warning: {e}")

            sent_action = None
            if args.actuate:
                sent_action = follower.send_action(action)

            maintain_fps(loop_start, FPS)
            loop_dt = time.perf_counter() - loop_start
            hz = 1.0 / loop_dt if loop_dt > 0 else float("inf")

            rr.set_time_sequence("step", step)
            rr.set_time_seconds("time", time.perf_counter() - run_start)
            rr.log("metrics/loop_hz", rr.Scalars(hz))
            rr.log("metrics/range_ok", rr.Scalars(1.0 if range_ok else 0.0))
            for cam_name in CAMERAS:
                if cam_name in observation:
                    rr.log(f"camera/{cam_name}", rr.Image(observation[cam_name]))
            for i, name in enumerate(JOINT_NAMES):
                rr.log(f"joints/{name}/policy_action", rr.Scalars(float(action_values[i])))
                if sent_action is not None:
                    rr.log(f"joints/{name}/sent_action", rr.Scalars(sent_action[f"{name}.pos"]))

            vals = ",".join(f"{action_values[i]:.4f}" for i in range(len(JOINT_NAMES)))
            debug_log.write(
                f"{step},{hz:.1f},{action_queue.qsize()},{mode},{int(range_ok)},{vals}\n"
            )
            if step % 30 == 0:
                debug_log.flush()

            action_preview = "  |  ".join(
                f"{name}: {value:7.2f}" for name, value in action.items()
            )
            print(
                f"Step {step:05d} | {hz:5.1f} Hz | Q:{action_queue.qsize():2d} "
                f"| range:{'ok' if range_ok else 'BAD'} | {action_preview}",
                end="\r",
            )
            step += 1

    except KeyboardInterrupt:
        print("\n\nStopping MolmoAct2 eval...")
    finally:
        stop_event = True
        if "debug_log" in locals():
            debug_log.close()
            print("Debug log saved to outputs/eval_molmoact2_debug.csv")
        safe_disconnect(follower)
        if stop_event:
            print("Disconnected. Done!")


if __name__ == "__main__":
    main()
