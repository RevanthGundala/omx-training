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
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np
import rerun as rr
from pynput import keyboard

from data.scene_assist import (
    DEFAULT_MODEL as SCENE_ASSIST_DEFAULT_MODEL,
    TARGET_OBJECTS,
    analyze_start_scene,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features

from utils.config import CAMERAS, FPS, JOINT_NAMES, TASK_NAME
from utils.control_utils import ensure_camera_size, maintain_fps
from utils.rerun_utils import init_rerun
from utils.robot_utils import create_follower, create_leader, safe_disconnect

import omx_quic
from omx_quic import rendezvous

from evaluation.eval_pi0 import RTCActionQueue
from evaluation.eval_pi0 import RTC_IDLE_SLEEP_S, RTC_QUEUE_REFILL_THRESHOLD

START_DELAY_S = 3
TAKEOVER_SOFT_START_DURATION_S = 2.0
DEFAULT_SCENE_DIVERSITY_DIR = Path("outputs/record_scene_diversity")
DEFAULT_SCENE_TARGET_COUNT = 4
DEFAULT_SCENE_TOP_CAMERA = "top"
DEFAULT_CORRECTION_DATASET_REPO_ID = "RevanthGundala/005-pour-water-dagger-corrections"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RED = "\033[91m"


def color(text: str, ansi: str) -> str:
    return f"{ansi}{text}{RESET}"


def _read_single_key(prompt: str, valid: set[str]) -> str:
    print(prompt, end="", flush=True)
    done = threading.Event()
    selected = {"value": ""}

    def on_press(key):
        value = None
        if key in (keyboard.Key.enter, keyboard.Key.right):
            value = "c"
        elif key == keyboard.Key.esc:
            value = "q"
        elif hasattr(key, "char") and key.char:
            value = key.char.lower()
        if value in valid:
            selected["value"] = value
            done.set()
            return False
        return None

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    done.wait()
    listener.stop()
    print()
    return selected["value"]


def _soft_start_takeover(follower, leader, stop_signal: threading.Event) -> None:
    print(color(
        f"\nAligning follower to leader over {TAKEOVER_SOFT_START_DURATION_S:.1f}s before takeover...",
        YELLOW,
    ))
    current = follower.get_observation()
    target = leader.get_action()
    steps = max(1, int(TAKEOVER_SOFT_START_DURATION_S * FPS))
    for step in range(1, steps + 1):
        if stop_signal.is_set():
            break
        loop_start = time.perf_counter()
        alpha = step / steps
        blended = {
            key: float(current.get(key, target[key]) + alpha * (target[key] - current.get(key, target[key])))
            for key in target
        }
        follower.send_action(blended)
        target = leader.get_action()
        maintain_fps(loop_start, FPS)
    print(color("Takeover alignment complete.", GREEN))


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


class DAggerCorrectionWriter:
    def __init__(
        self,
        *,
        repo_id: str,
        features: dict,
        run_dir: Path,
        scene_summary: dict | None,
        server_health: dict | None,
    ):
        self.repo_id = repo_id
        self.root = Path.home() / ".cache/huggingface/lerobot" / repo_id
        self.scene_summary = scene_summary
        self.server_health = server_health
        self.frame_count = 0
        self.total_frame_count = 0
        self.segment_index = 0
        self.saved_episode_indices: list[int] = []
        self.saved_episode_index: int | None = None
        self._closed = False

        self.dataset = self._create_or_resume(features)
        self._corrections_file = (run_dir / "dagger_corrections.csv").open(
            "w", newline="", encoding="utf-8",
        )
        self._corrections_writer = csv.DictWriter(
            self._corrections_file,
            fieldnames=[
                "correction_segment",
                "correction_frame",
                "total_correction_frame",
                "source_eval_step",
                "scene_label",
                "scene_bucket",
                *[f"policy_action_{name}" for name in JOINT_NAMES],
                *[f"expert_action_{name}" for name in JOINT_NAMES],
            ],
        )
        self._corrections_writer.writeheader()
        (run_dir / "dagger_metadata.json").write_text(
            json.dumps(
                {
                    "correction_dataset_repo_id": repo_id,
                    "correction_dataset_root": str(self.root),
                    "scene_preflight_summary": scene_summary,
                    "server_health": server_health,
                    "saved_episode_indices": [],
                    "current_segment_index": 0,
                    "current_segment_frames": 0,
                    "correction_frames": 0,
                },
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        self._metadata_path = run_dir / "dagger_metadata.json"

    def _create_or_resume(self, features: dict) -> LeRobotDataset:
        info_path = self.root / "meta" / "info.json"
        if info_path.exists():
            print(color(f"Resuming correction dataset: {self.root}", CYAN))
            return LeRobotDataset.resume(
                repo_id=self.repo_id,
                root=self.root,
                image_writer_processes=0,
                image_writer_threads=4,
            )
        if self.root.exists():
            backup_path = self.root.with_name(self.root.name + f"_backup_{int(time.time())}")
            print(color(f"Correction dataset dir has no info.json; backing up to {backup_path}", YELLOW))
            shutil.move(str(self.root), str(backup_path))
        print(color(f"Creating correction dataset: {self.repo_id}", CYAN))
        return LeRobotDataset.create(
            repo_id=self.repo_id,
            root=self.root,
            fps=FPS,
            robot_type="omx_follower",
            features=features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4,
        )

    def add_frame(
        self,
        *,
        observation: dict,
        expert_action: dict,
        policy_action: np.ndarray | None,
        source_eval_step: int,
    ) -> None:
        obs_frame = build_dataset_frame(self.dataset.features, observation, prefix="observation")
        action_frame = build_dataset_frame(self.dataset.features, expert_action, prefix="action")
        frame = {**obs_frame, **action_frame, "task": TASK_NAME}
        self.dataset.add_frame(frame)
        row = {
            "correction_segment": self.segment_index,
            "correction_frame": self.frame_count,
            "total_correction_frame": self.total_frame_count,
            "source_eval_step": source_eval_step,
            "scene_label": (self.scene_summary or {}).get("label"),
            "scene_bucket": (self.scene_summary or {}).get("bucket"),
        }
        if policy_action is not None:
            for i, name in enumerate(JOINT_NAMES):
                row[f"policy_action_{name}"] = float(policy_action[i])
        for name in JOINT_NAMES:
            row[f"expert_action_{name}"] = float(expert_action[f"{name}.pos"])
        self._corrections_writer.writerow(row)
        if self.frame_count % 30 == 0:
            self._corrections_file.flush()
        self.frame_count += 1
        self.total_frame_count += 1

    def save_episode(self) -> int | None:
        if self.frame_count <= 0:
            return None
        episode_index = int(self.dataset.num_episodes)
        self.dataset.save_episode(parallel_encoding=False)
        self.saved_episode_indices.append(episode_index)
        self.saved_episode_index = episode_index
        self.frame_count = 0
        self.segment_index += 1
        self._write_metadata()
        return episode_index

    def discard_episode(self) -> None:
        self.dataset.clear_episode_buffer()
        self.frame_count = 0
        self.saved_episode_index = None
        self.segment_index += 1
        self._write_metadata()

    def _write_metadata(self) -> None:
        self._metadata_path.write_text(
            json.dumps(
                {
                    "correction_dataset_repo_id": self.repo_id,
                    "correction_dataset_root": str(self.root),
                    "scene_preflight_summary": self.scene_summary,
                    "server_health": self.server_health,
                    "saved_episode_indices": self.saved_episode_indices,
                    "current_segment_index": self.segment_index,
                    "current_segment_frames": self.frame_count,
                    "correction_frames": self.total_frame_count,
                },
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._corrections_file.close()
        finalize = getattr(self.dataset, "finalize", None)
        if callable(finalize):
            finalize()
            print(color(
                f"Correction dataset finalized: {self.root} ({self.dataset.num_episodes} episodes)",
                GREEN,
            ))


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


def _scene_root_for_dataset(dataset_repo_id: str | None, override: Path | None) -> Path | None:
    if override is not None:
        return override
    if not dataset_repo_id:
        return None
    return DEFAULT_SCENE_DIVERSITY_DIR / dataset_repo_id.replace("/", "__")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_scene_preflight_artifacts(
    run_dir: Path,
    attempt_index: int,
    observation: dict,
    report: dict,
) -> None:
    preflight_dir = run_dir / "scene_preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    serializable_report = {key: value for key, value in report.items() if not key.startswith("_")}
    (preflight_dir / f"attempt-{attempt_index:02d}.json").write_text(
        json.dumps(serializable_report, indent=2, default=_json_default),
        encoding="utf-8",
    )
    for cam_name in CAMERAS:
        img = observation.get(cam_name)
        if isinstance(img, np.ndarray) and img.ndim == 3:
            cv2.imwrite(
                str(preflight_dir / f"attempt-{attempt_index:02d}_{cam_name}.jpg"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
    annotated = report.get("_annotated_rgb")
    if isinstance(annotated, np.ndarray):
        cv2.imwrite(
            str(preflight_dir / f"attempt-{attempt_index:02d}_top_yolo.jpg"),
            cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
        )


def _scene_distribution_summary(report: dict) -> dict:
    if "error" in report:
        return {
            "label": "unknown",
            "reason": str(report["error"]),
            "bucket": None,
            "bucket_count": 0,
            "target_count": DEFAULT_SCENE_TARGET_COUNT,
            "objects": {},
        }

    layout = report.get("layout", {})
    recommendation = report.get("recommendation", {})
    coverage = report.get("coverage", {})
    objects = report.get("objects", {})
    missing = layout.get("missing_targets", [])
    bucket_count = int(recommendation.get("bucket_count", 0))
    target_count = int(coverage.get("target_count_per_bucket", DEFAULT_SCENE_TARGET_COUNT))

    if missing:
        label = "unknown"
        reason = f"YOLO did not confidently find: {', '.join(missing)}."
    elif bucket_count <= 0:
        label = "out_of_distribution"
        reason = "This exact layout bucket has 0 training episodes."
    elif bucket_count < target_count:
        label = "thin_distribution"
        reason = f"This layout bucket has only {bucket_count}/{target_count} training episodes."
    else:
        label = "in_distribution"
        reason = f"This layout bucket has {bucket_count}/{target_count} training episodes."

    object_summary = {}
    for name in TARGET_OBJECTS:
        det = objects.get(name)
        if det:
            bucket = layout.get("object_buckets", {}).get(name, {})
            object_summary[name] = {
                "confidence": float(det["confidence"]),
                "center": [float(v) for v in det["center"]],
                "bucket": bucket,
            }
        else:
            object_summary[name] = None

    return {
        "label": label,
        "reason": reason,
        "bucket": layout.get("bucket"),
        "bucket_count": bucket_count,
        "target_count": target_count,
        "objects": object_summary,
    }


def _format_eval_scene_report(report: dict) -> str:
    summary = _scene_distribution_summary(report)
    label = str(summary["label"])
    status = label.replace("_", " ").upper()
    status_color = {
        "in_distribution": GREEN,
        "thin_distribution": YELLOW,
        "out_of_distribution": RED,
        "unknown": MAGENTA,
    }.get(label, RESET)

    target_bits = []
    for name in TARGET_OBJECTS:
        det = summary["objects"].get(name)
        if det:
            bucket = det.get("bucket", {})
            target_bits.append(
                f"{name}={bucket.get('x', '?')}/{bucket.get('y', '?')} "
                f"{float(det['confidence']):.2f}"
            )
        else:
            target_bits.append(f"{name}=missing")

    lines = [
        "",
        f"{CYAN}Scene check:{RESET} {color(status, status_color)}",
        f"  {DIM}{' | '.join(target_bits)} | bucket {summary['bucket_count']}/{summary['target_count']}{RESET}",
        f"  {DIM}bucket: {summary.get('bucket') or 'unknown'}{RESET}",
        f"  {color(summary['reason'], status_color)}",
    ]
    nearest = report.get("nearest_prior_episodes", [])[:3]
    if nearest:
        lines.append(
            f"  {DIM}nearest train starts: "
            + "; ".join(
                f"ep{item.get('episode_index')} sim={float(item.get('similarity', 0.0)):.2f}"
                for item in nearest
            )
            + RESET
        )
    return "\n".join(lines)


def _run_scene_preflight(
    *,
    follower,
    run_dir: Path,
    scene_root: Path | None,
    top_camera: str,
    model_path: str,
    min_confidence: float,
    target_count: int,
    min_episode_index: int,
) -> dict | None:
    if scene_root is None:
        print("\nScene check: SKIPPED (no dataset repo id available from server health)")
        return None
    if not scene_root.exists():
        print(f"\nScene check: SKIPPED (coverage directory not found: {scene_root})")
        return None

    attempt_index = 0
    latest_report = None
    while True:
        attempt_index += 1
        print(color("\nScene check: capturing start state...", CYAN))
        observation = follower.get_observation()
        top_frame = observation.get(top_camera)
        if not isinstance(top_frame, np.ndarray):
            report = {"error": f"missing {top_camera!r} camera frame"}
        else:
            try:
                report = analyze_start_scene(
                    top_frame,
                    episode_index=attempt_index,
                    scene_root=scene_root,
                    model_path=model_path,
                    min_confidence=min_confidence,
                    target_count=target_count,
                    min_episode_index=min_episode_index,
                )
            except Exception as e:
                report = {"error": str(e)}
        latest_report = report
        _write_scene_preflight_artifacts(run_dir, attempt_index, observation, report)
        print(_format_eval_scene_report(report))

        choice = _read_single_key(
            color("Press c/→ to continue inference, r to reset/recheck, q to quit: ", BOLD),
            {"c", "r", "q"},
        )
        if choice == "c":
            return latest_report
        if choice == "q":
            raise KeyboardInterrupt
        print(color("Reset the objects/robot state, then press r to recheck.", YELLOW))
        ready = _read_single_key(color("Press r when ready to recheck, q to quit: ", BOLD), {"r", "q"})
        if ready == "q":
            raise KeyboardInterrupt


def _connect_quic(session_id: str, stun_server: str) -> omx_quic.QuicClient:
    client = omx_quic.QuicClient(session_id)
    print(color(f"[client] local UDP port: {client.local_port()}", DIM))
    pub_ip, pub_port = client.discover_public_address(stun_server)
    print(color(f"[client] public address: {pub_ip}:{pub_port}", DIM))
    pub_ip2, pub_port2 = client.discover_public_address(stun_server)
    if (pub_ip2, pub_port2) != (pub_ip, pub_port):
        raise RuntimeError(
            f"[client] symmetric NAT detected: STUN1={pub_ip}:{pub_port} "
            f"STUN2={pub_ip2}:{pub_port2}. Hole punching cannot work from "
            "this network. Try a different network or a TURN relay."
        )
    rendezvous.publish(session_id, "client", pub_ip, pub_port)
    try:
        print(color("[client] waiting for server peer in rendezvous dict ...", DIM))
        peer_ip, peer_port = rendezvous.wait_for_peer(session_id, "client", timeout_s=300.0)
        print(color(f"[client] peer: {peer_ip}:{peer_port}", DIM))
        client.set_peer_address(peer_ip, peer_port)
        sent, received, elapsed = client.punch(timeout_s=15.0)
        print(color(f"[client] punch ok: sent={sent} received={received} elapsed={elapsed:.3f}s", GREEN))
        print(color("[client] QUIC handshake ...", DIM))
        t0 = time.perf_counter()
        client.connect(timeout_s=30.0)
        print(color(f"[client] QUIC connected in {(time.perf_counter()-t0)*1000:.1f}ms", GREEN))
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
    parser.add_argument(
        "--no-scene-check",
        action="store_true",
        help="Skip the YOLO start-scene in-distribution check before inference.",
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=None,
        help=(
            "Directory containing scene coverage JSONs. Defaults to "
            "outputs/record_scene_diversity/<server health dataset_repo_id>."
        ),
    )
    parser.add_argument("--scene-top-camera", type=str, default=DEFAULT_SCENE_TOP_CAMERA)
    parser.add_argument("--scene-model", type=str, default=SCENE_ASSIST_DEFAULT_MODEL)
    parser.add_argument("--scene-min-confidence", type=float, default=0.25)
    parser.add_argument("--scene-target-count", type=int, default=DEFAULT_SCENE_TARGET_COUNT)
    parser.add_argument("--scene-min-episode-index", type=int, default=0)
    parser.add_argument(
        "--enable-takeover",
        action="store_true",
        help="Enable leader-arm takeover controls without saving correction data.",
    )
    parser.add_argument(
        "--save-corrections",
        action="store_true",
        help=(
            "Save takeover frames as expert correction data in the default local "
            f"LeRobot dataset {DEFAULT_CORRECTION_DATASET_REPO_ID}."
        ),
    )
    parser.add_argument(
        "--correction-dataset-repo-id",
        type=str,
        default=None,
        help=(
            "Override the local LeRobot correction dataset repo id. Providing this "
            "also enables --save-corrections."
        ),
    )
    parser.add_argument("--notes", type=str, default="", help="Optional scene notes saved to metadata.json.")
    args = parser.parse_args()
    correction_repo_id = (
        args.correction_dataset_repo_id
        or (DEFAULT_CORRECTION_DATASET_REPO_ID if args.save_corrections else None)
    )
    save_corrections = correction_repo_id is not None
    takeover_enabled = args.enable_takeover or save_corrections

    print(color(f"Connecting QUIC for session={args.session_id!r} ...", CYAN))
    quic_client = _connect_quic(args.session_id, args.stun)

    # Health probe.
    health = None
    try:
        h = quic_client.request(json.dumps({"op": "health"}).encode("utf-8"), 10.0)
        health = json.loads(h)
        print(color("[client] server ready", GREEN))
        if isinstance(health, dict):
            print(
                color(
                    "  "
                    f"checkpoint={health.get('checkpoint_source')} | "
                    f"dataset={health.get('dataset_repo_id')} | "
                    f"stats={health.get('stats_source')} | "
                    f"relative={health.get('use_relative_actions')} | "
                    f"stats_ok={health.get('stats_global_check', {}).get('ok')}",
                    DIM,
                )
            )
    except Exception as e:
        print(color(f"[client] health probe failed: {e}", YELLOW))

    follower = _build_follower()
    dataset_features = {
        **hw_to_dataset_features(follower.action_features, "action", use_video=False),
        **hw_to_dataset_features(follower.observation_features, "observation", use_video=False),
    }
    correction_dataset_features = {
        **hw_to_dataset_features(follower.action_features, "action", use_video=True),
        **hw_to_dataset_features(follower.observation_features, "observation", use_video=True),
    }

    print(color("Connecting follower arm...", CYAN))
    for attempt in range(1, 4):
        try:
            follower.connect(calibrate=False)
            break
        except (TimeoutError, RuntimeError) as e:
            print(color(f"  Camera connect attempt {attempt}/3 failed: {e}", YELLOW))
            if attempt == 3:
                raise
            if hasattr(follower, "disconnect"):
                follower.disconnect()
            follower = _build_follower()
            dataset_features = {
                **hw_to_dataset_features(follower.action_features, "action", use_video=False),
                **hw_to_dataset_features(follower.observation_features, "observation", use_video=False),
            }
            correction_dataset_features = {
                **hw_to_dataset_features(follower.action_features, "action", use_video=True),
                **hw_to_dataset_features(follower.observation_features, "observation", use_video=True),
            }
            time.sleep(2)

    leader = None
    if takeover_enabled:
        print(color("Connecting leader arm for takeover...", CYAN))
        try:
            leader = create_leader()
            leader.connect()
        except Exception:
            safe_disconnect(follower)
            try:
                quic_client.close()
            except Exception:
                pass
            raise

    init_rerun("omx_eval_pi05_quic", save_rrd=True)

    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.session_id}"
    run_dir = args.eval_output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "session_id": args.session_id,
                "stun": args.stun,
                "server_health": health,
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
    print(color(f"Eval artifacts will be saved under: {run_dir}", CYAN))

    scene_report = None
    scene_summary = None
    if not args.no_scene_check:
        try:
            scene_report = _run_scene_preflight(
                follower=follower,
                run_dir=run_dir,
                scene_root=_scene_root_for_dataset(
                    health.get("dataset_repo_id") if isinstance(health, dict) else None,
                    args.scene_root,
                ),
                top_camera=args.scene_top_camera,
                model_path=args.scene_model,
                min_confidence=args.scene_min_confidence,
                target_count=args.scene_target_count,
                min_episode_index=args.scene_min_episode_index,
            )
        except KeyboardInterrupt:
            print("\nScene check cancelled before inference.")
            try:
                quic_client.close()
            except Exception:
                pass
            run_logger.close()
            if video_recorder is not None:
                video_recorder.close()
            print(f"Eval artifacts saved to {run_dir}")
            safe_disconnect(follower)
            if leader is not None:
                safe_disconnect(leader)
            print("Disconnected. Done!")
            return
        metadata_path = run_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["scene_preflight"] = (
            {key: value for key, value in scene_report.items() if not key.startswith("_")}
            if scene_report is not None
            else None
        )
        metadata["scene_preflight_summary"] = (
            _scene_distribution_summary(scene_report)
            if scene_report is not None
            else None
        )
        scene_summary = metadata["scene_preflight_summary"]
        if metadata["scene_preflight_summary"] is not None:
            label = metadata["scene_preflight_summary"]["label"]
            bucket = metadata["scene_preflight_summary"].get("bucket") or "unknown"
            print(color(f"Scene preflight label saved: {label} ({bucket})", CYAN))
        metadata_path.write_text(
            json.dumps(metadata, indent=2, default=_json_default),
            encoding="utf-8",
        )
    if save_corrections:
        metadata_path = run_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["correction_dataset_repo_id"] = correction_repo_id
        metadata["correction_dataset_root"] = str(
            Path.home() / ".cache/huggingface/lerobot" / correction_repo_id
        )
        metadata_path.write_text(
            json.dumps(metadata, indent=2, default=_json_default),
            encoding="utf-8",
        )

    correction_writer = None
    if save_corrections:
        try:
            correction_writer = DAggerCorrectionWriter(
                repo_id=correction_repo_id,
                features=correction_dataset_features,
                run_dir=run_dir,
                scene_summary=scene_summary,
                server_health=health if isinstance(health, dict) else None,
            )
        except Exception:
            run_logger.close()
            if video_recorder is not None:
                video_recorder.close()
            safe_disconnect(follower)
            if leader is not None:
                safe_disconnect(leader)
            try:
                quic_client.close()
            except Exception:
                pass
            raise

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
    stop_requested = threading.Event()
    restart_requested = threading.Event()
    takeover_requested = threading.Event()
    policy_requested = threading.Event()
    save_requested = threading.Event()
    discard_requested = threading.Event()
    restart_generation = 0
    control_mode = "policy"
    resume_policy_after_observation = False
    latest_policy_action = None

    def on_key_press(key):
        if key in (keyboard.Key.right, keyboard.Key.up):
            restart_requested.set()
        elif hasattr(key, "char") and key.char and key.char.lower() == "q":
            stop_requested.set()
        elif takeover_enabled and hasattr(key, "char") and key.char and key.char.lower() == "t":
            takeover_requested.set()
        elif takeover_enabled and hasattr(key, "char") and key.char and key.char.lower() == "p":
            policy_requested.set()
        elif hasattr(key, "char") and key.char and key.char.lower() == "s":
            save_requested.set()
            stop_requested.set()
        elif hasattr(key, "char") and key.char and key.char.lower() == "d":
            discard_requested.set()

    key_listener = keyboard.Listener(on_press=on_key_press)
    key_listener.start()

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
        seen_restart_generation = restart_generation
        while not stop_event.is_set():
            if control_mode == "takeover":
                time.sleep(RTC_IDLE_SLEEP_S)
                continue
            if seen_restart_generation != restart_generation:
                first_call = True
                last_round_trip_delay = 0
                seen_restart_generation = restart_generation
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
            request_generation = restart_generation
            if first_call:
                print(color("Sending first QUIC inference request ...", CYAN))
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
                    print(color(f"\nServer error: {data['error']}", RED))
                    continue
                dbg = data.get("debug", {})
                print(
                    color(
                        f"\n  [client] RTT={rtt_ms:6.1f}ms "
                        f"req={len(req)/1024:.1f}KB resp={len(resp_bytes)/1024:.1f}KB "
                        f"server_delay={dbg.get('inference_delay')} "
                        f"server_prev_steps={dbg.get('prev_steps_consumed')} "
                        f"prev_chunk={dbg.get('prev_chunk_exists')}",
                        DIM,
                    )
                )
                actions = np.array(data["actions"], dtype=np.float32)
                if request_generation != restart_generation:
                    continue
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
                print(color(
                    f"  [rtc] prev_steps={prev_steps_consumed} "
                    f"est_delay={estimated_delay} actual_skip={actual_skip} "
                    f"qsize_after={action_queue.qsize()}",
                    DIM,
                ))
                last_round_trip_delay = actual_skip
                if first_call:
                    print(color(f"First inference returned {len(actions)} actions. Robot active!", GREEN))
                    first_call = False
            except Exception as e:
                print(color(f"\nInference error: {e}", YELLOW))
            finally:
                inference_running.clear()

    inference_thread = threading.Thread(target=_inference_loop, daemon=True)
    inference_thread.start()

    try:
        print(color(
            f"Starting Pi0 QUIC eval in {START_DELAY_S}s. "
            + (
                "Controls: t takeover, p save segment→policy, →/↑ restart, "
                "s save/stop, d discard current segment, q quit."
                if takeover_enabled
                else "Controls: →/↑ restart policy from beginning, q stop/save."
            ),
            BOLD + BLUE,
        ))
        for remaining in range(START_DELAY_S, 0, -1):
            print(color(f"  {remaining}...", BLUE))
            time.sleep(1)
        print(color("Running! Waiting for first inference...", GREEN))

        run_start = time.perf_counter()
        step = 0

        while not stop_requested.is_set():
            loop_start = time.perf_counter()
            if takeover_requested.is_set():
                takeover_requested.clear()
                if leader is None:
                    print(color("\nTakeover unavailable: leader arm is not connected.", YELLOW))
                else:
                    control_mode = "takeover"
                    restart_generation += 1
                    action_queue = RTCActionQueue()
                    try:
                        quic_client.request(json.dumps({"op": "reset"}).encode("utf-8"), 10.0)
                    except Exception as e:
                        print(color(f"\nServer reset failed during takeover: {e}", YELLOW))
                    _soft_start_takeover(follower, leader, stop_requested)
                    takeover_msg = (
                        "\nTAKEOVER: leader controls follower. Recording expert corrections."
                        if correction_writer is not None
                        else "\nTAKEOVER: leader controls follower. Corrections are not being saved."
                    )
                    print(color(takeover_msg, MAGENTA))

            if policy_requested.is_set():
                policy_requested.clear()
                if control_mode == "takeover":
                    if correction_writer is not None and correction_writer.frame_count > 0:
                        episode_index = correction_writer.save_episode()
                        print(color(
                            f"\nSaved correction segment as episode {episode_index}; returning to policy.",
                            GREEN,
                        ))
                    resume_policy_after_observation = True
                    restart_generation += 1
                    action_queue = RTCActionQueue()
                    step = 0
                    run_start = time.perf_counter()
                    try:
                        quic_client.request(json.dumps({"op": "reset"}).encode("utf-8"), 10.0)
                    except Exception as e:
                        print(color(f"\nServer reset failed when returning to policy: {e}", YELLOW))
                    print(color("\nPOLICY: returned to policy control from current state.", CYAN))

            if discard_requested.is_set():
                discard_requested.clear()
                if correction_writer is not None and correction_writer.frame_count > 0:
                    correction_writer.discard_episode()
                    print(color("\nDiscarded current correction segment.", RED))

            if restart_requested.is_set():
                restart_requested.clear()
                if control_mode == "takeover":
                    print(color(
                        "\nRestart ignored during takeover. Use p to save→policy, d to discard, or s to save/stop.",
                        YELLOW,
                    ))
                    continue
                resume_policy_after_observation = control_mode == "takeover"
                control_mode = "policy"
                restart_generation += 1
                action_queue = RTCActionQueue()
                step = 0
                run_start = time.perf_counter()
                try:
                    quic_client.request(json.dumps({"op": "reset"}).encode("utf-8"), 10.0)
                except Exception as e:
                    print(color(f"\nServer reset failed during restart: {e}", YELLOW))
                print(color("\nRestarted eval policy state. Waiting for fresh inference...", MAGENTA))

            observation = follower.get_observation()
            for cam_name in CAMERAS:
                if cam_name in observation:
                    ensure_camera_size(observation, key=cam_name)
            if video_recorder is not None:
                video_recorder.write(observation)

            if resume_policy_after_observation:
                observation_frame = build_dataset_frame(
                    dataset_features, observation, prefix="observation",
                )
                with obs_lock:
                    latest_observation = observation
                    latest_observation_frame = observation_frame
                    latest_step = step
                control_mode = "policy"
                resume_policy_after_observation = False
                maintain_fps(loop_start, FPS)
                continue

            if control_mode == "takeover":
                if leader is None:
                    maintain_fps(loop_start, FPS)
                    continue
                expert_action = leader.get_action()
                sent_action = follower.send_action(expert_action)
                if correction_writer is not None:
                    correction_writer.add_frame(
                        observation=observation,
                        expert_action=sent_action,
                        policy_action=latest_policy_action,
                        source_eval_step=step,
                    )
                maintain_fps(loop_start, FPS)
                print(
                    color(
                        f"TAKEOVER | frames:{correction_writer.frame_count if correction_writer else 0:04d} "
                        f"saved:{len(correction_writer.saved_episode_indices) if correction_writer else 0:02d} "
                        "| p save→policy | s save/stop | d discard segment | q quit",
                        MAGENTA,
                    ),
                    end="\r",
                )
                continue

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
            latest_policy_action = action_values

            action = {
                key: float(action_values[i])
                for i, key in enumerate(follower.action_features)
            }

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
            print(
                color(
                    f"Step {step:05d} | {hz:5.1f} Hz | Q:{queue_len:2d} {inf_sym} "
                    + ("| t takeover | →/↑ restart | q stop/save" if takeover_enabled else "| →/↑ restart | q stop/save"),
                    GREEN,
                ),
                end="\r",
            )

        if correction_writer is not None:
            if save_requested.is_set() and correction_writer.frame_count > 0:
                episode_index = correction_writer.save_episode()
                print(color(
                    f"\nSaved correction episode {episode_index} "
                    f"to {correction_writer.root}",
                    GREEN,
                ))
            elif correction_writer.frame_count > 0:
                correction_writer.discard_episode()
                print(color("\nDiscarded unsaved correction segment.", YELLOW))

        print(color("\nStop requested. Saving eval artifacts...", YELLOW))

    except KeyboardInterrupt:
        print(color("\n\nStopping Pi0 QUIC eval...", YELLOW))
    finally:
        stop_event.set()
        try:
            key_listener.stop()
        except Exception:
            pass
        try:
            quic_client.close()
        except Exception:
            pass
        inference_thread.join(timeout=5)
        run_logger.close()
        if correction_writer is not None:
            if correction_writer.frame_count > 0:
                correction_writer.discard_episode()
                print(color("Discarded uncommitted correction segment.", YELLOW))
            correction_writer.close()
        if video_recorder is not None:
            video_recorder.close()
            transcode_eval_videos_for_browser(run_dir)
        print(color(f"Eval artifacts saved to {run_dir}", CYAN))
        safe_disconnect(follower)
        if leader is not None:
            safe_disconnect(leader)
        print(color("Disconnected. Done!", GREEN))


if __name__ == "__main__":
    main()
