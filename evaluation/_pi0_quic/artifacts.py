"""Run-artifact writers: event log, per-step CSV/Rerun bundle, video, transcode."""

from __future__ import annotations

import base64
import csv
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from utils.config import CAMERAS, JOINT_NAMES


def json_default(value):
    """JSON default for numpy scalars/arrays."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class EventLogger:
    def __init__(self, run_dir: Path):
        self._file = (run_dir / "event_log.jsonl").open("w", encoding="utf-8")
        self._lock = threading.Lock()

    def log(self, event: str, **fields) -> None:
        row = {
            "time_monotonic": time.perf_counter(),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
            **fields,
        }
        with self._lock:
            self._file.write(json.dumps(row, default=json_default) + "\n")
            self._file.flush()

    def close(self) -> None:
        self._file.close()


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
        self._takeover_file = (run_dir / "takeover_steps.csv").open("w", newline="", encoding="utf-8")
        self._control_writer: csv.DictWriter | None = None
        self._takeover_writer = csv.DictWriter(
            self._takeover_file,
            fieldnames=[
                "takeover_frame",
                "source_eval_step",
                "time_s",
                "loop_dt_s",
                "hz",
                "mode",
                "settling",
                "recording",
                "missing_joint_count",
                *[f"observation_{name}" for name in JOINT_NAMES],
                *[f"leader_action_{name}" for name in JOINT_NAMES],
                *[f"sent_action_{name}" for name in JOINT_NAMES],
            ],
        )
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
        self._takeover_writer.writeheader()
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

    def log_takeover_step(
        self,
        *,
        takeover_frame: int,
        source_eval_step: int,
        time_s: float,
        loop_dt_s: float,
        mode: str,
        observation: dict,
        leader_action: dict | None,
        sent_action: dict | None,
        settling: bool,
        recording: bool,
    ) -> None:
        row = {
            "takeover_frame": takeover_frame,
            "source_eval_step": source_eval_step,
            "time_s": time_s,
            "loop_dt_s": loop_dt_s,
            "hz": (1.0 / loop_dt_s) if loop_dt_s > 0 else float("inf"),
            "mode": mode,
            "settling": settling,
            "recording": recording,
            "missing_joint_count": 0,
        }
        missing = 0
        for name in JOINT_NAMES:
            key = f"{name}.pos"
            for prefix, values in (
                ("observation", observation),
                ("leader_action", leader_action or {}),
                ("sent_action", sent_action or {}),
            ):
                if key in values:
                    row[f"{prefix}_{name}"] = float(values[key])
                else:
                    row[f"{prefix}_{name}"] = ""
                    if prefix == "sent_action":
                        missing += 1
        row["missing_joint_count"] = missing
        self._takeover_writer.writerow(row)
        if takeover_frame % 30 == 0:
            self._takeover_file.flush()

    def close(self) -> None:
        self._control_file.close()
        self._request_file.close()
        self._takeover_file.close()
