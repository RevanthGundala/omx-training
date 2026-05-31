"""Pre-roll scene/object preflight check.

Captures a top-camera frame, runs the YOLO start-state analyzer against the
training-set coverage map, prints an in/thin/out-of-distribution verdict, and
loops on operator input until they continue (``c``/``→``), reset (``r``), or
quit (``q``).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np
from pynput import keyboard

from data.scene_assist import TARGET_OBJECTS, analyze_start_scene
from utils.config import CAMERAS

from .artifacts import json_default
from .colors import BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RESET, YELLOW, color


DEFAULT_SCENE_DIVERSITY_DIR = Path("outputs/record_scene_diversity")
DEFAULT_SCENE_TARGET_COUNT = 4
DEFAULT_SCENE_TOP_CAMERA = "top"


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


def scene_root_for_dataset(dataset_repo_id: str | None, override: Path | None) -> Path | None:
    if override is not None:
        return override
    if not dataset_repo_id:
        return None
    return DEFAULT_SCENE_DIVERSITY_DIR / dataset_repo_id.replace("/", "__")


def _write_scene_preflight_artifacts(
    run_dir: Path,
    attempt_index: int,
    observation: dict,
    report: dict,
) -> None:
    preflight_dir = run_dir / "scene_preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in report.items() if not k.startswith("_")}
    (preflight_dir / f"attempt-{attempt_index:02d}.json").write_text(
        json.dumps(serializable, indent=2, default=json_default),
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


def scene_distribution_summary(report: dict) -> dict:
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


def _format_scene_report(report: dict) -> str:
    summary = scene_distribution_summary(report)
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


def run_scene_preflight(
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
        print(_format_scene_report(report))

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
