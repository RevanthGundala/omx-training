"""Object-centric start-scene assistance for teleop collection."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TARGET_OBJECTS = ("bottle", "cup")
DEFAULT_MODEL = "yolo11n.pt"
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"

_YOLO_CACHE: dict[str, Any] = {}


def _load_yolo(model_path: str):
    if model_path not in _YOLO_CACHE:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for scene assistance. Install it with: uv pip install ultralytics"
            ) from exc
        _YOLO_CACHE[model_path] = YOLO(model_path)
    return _YOLO_CACHE[model_path]


def _bucket_axis(value: float, labels: tuple[str, str, str]) -> str:
    if value < 1.0 / 3.0:
        return labels[0]
    if value < 2.0 / 3.0:
        return labels[1]
    return labels[2]


def _distance_bucket(distance: float) -> str:
    if distance < 0.30:
        return "small"
    if distance < 0.55:
        return "medium"
    return "large"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def run_yolo_detections(
    rgb: np.ndarray,
    model_path: str = DEFAULT_MODEL,
    min_confidence: float = 0.25,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Run YOLO on an RGB image and return normalized detections plus an annotated RGB image."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {rgb.shape}")

    model = _load_yolo(model_path)
    result = model.predict(rgb, verbose=False, conf=min_confidence)[0]
    height, width = rgb.shape[:2]
    names = result.names
    detections: list[dict[str, Any]] = []

    if result.boxes is not None:
        for box in result.boxes:
            cls_index = int(box.cls[0].item())
            label = str(names.get(cls_index, cls_index))
            confidence = float(box.conf[0].item())
            if confidence < min_confidence:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            norm_box = [
                max(0.0, min(1.0, x1 / width)),
                max(0.0, min(1.0, y1 / height)),
                max(0.0, min(1.0, x2 / width)),
                max(0.0, min(1.0, y2 / height)),
            ]
            center = [(norm_box[0] + norm_box[2]) / 2.0, (norm_box[1] + norm_box[3]) / 2.0]
            size = [max(0.0, norm_box[2] - norm_box[0]), max(0.0, norm_box[3] - norm_box[1])]
            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "box": norm_box,
                    "center": center,
                    "size": size,
                }
            )

    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return detections, annotated_rgb


def select_target_objects(detections: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    for target in TARGET_OBJECTS:
        matches = [det for det in detections if det["label"] == target]
        if matches:
            objects[target] = max(matches, key=lambda det: det["confidence"])
    return objects


def build_layout(objects: dict[str, dict[str, Any]]) -> dict[str, Any]:
    missing = [name for name in TARGET_OBJECTS if name not in objects]
    object_buckets: dict[str, dict[str, str]] = {}
    for name, det in objects.items():
        cx, cy = det["center"]
        object_buckets[name] = {
            "x": _bucket_axis(cx, ("left", "center", "right")),
            "y": _bucket_axis(cy, ("upper", "middle", "lower")),
        }

    relation: dict[str, Any] = {}
    bucket = "uncertain"
    label = "uncertain"
    if not missing:
        bottle = objects["bottle"]["center"]
        cup = objects["cup"]["center"]
        dx = float(cup[0] - bottle[0])
        dy = float(cup[1] - bottle[1])
        distance = float(math.sqrt(dx * dx + dy * dy))
        relation = {
            "cup_relative_to_bottle": {
                "x": "right" if dx > 0.08 else "left" if dx < -0.08 else "aligned",
                "y": "lower" if dy > 0.08 else "upper" if dy < -0.08 else "aligned",
            },
            "center_distance": distance,
            "distance_bucket": _distance_bucket(distance),
        }
        b = object_buckets["bottle"]
        c = object_buckets["cup"]
        bucket = (
            f"bottle_{b['x']}_{b['y']}__"
            f"cup_{c['x']}_{c['y']}__"
            f"dist_{relation['distance_bucket']}"
        )
        label = (
            f"bottle={b['x']}/{b['y']}, "
            f"cup={c['x']}/{c['y']}, "
            f"distance={relation['distance_bucket']}"
        )

    return {
        "bucket": bucket,
        "label": label,
        "object_buckets": object_buckets,
        "relation": relation,
        "missing_targets": missing,
    }


def load_scene_reports(scene_root: Path, min_episode_index: int = 0) -> list[dict[str, Any]]:
    reports = []
    for path in sorted(scene_root.glob("episode-*.scene.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if int(report.get("episode_index", -1)) < min_episode_index:
            continue
        if report.get("layout", {}).get("bucket"):
            report["_path"] = str(path)
            reports.append(report)
    return reports


def summarize_coverage(reports: list[dict[str, Any]], target_count: int = 3) -> dict[str, Any]:
    counts = Counter(
        report.get("layout", {}).get("bucket")
        for report in reports
        if report.get("layout", {}).get("bucket") and report.get("layout", {}).get("bucket") != "uncertain"
    )
    buckets = [
        {
            "bucket": bucket,
            "count": count,
            "status": "overrepresented" if count >= target_count else "underrepresented",
        }
        for bucket, count in counts.most_common()
    ]
    return {
        "target_count_per_bucket": target_count,
        "total_scene_reports": len(reports),
        "known_bucket_count": len(counts),
        "buckets": buckets,
        "underrepresented_buckets": [item for item in buckets if item["count"] < target_count],
        "overrepresented_buckets": [item for item in buckets if item["count"] >= target_count],
    }


def _object_similarity(current: dict[str, Any], prior: dict[str, Any]) -> float | None:
    scores = []
    for name in TARGET_OBJECTS:
        cur = current.get(name)
        old = prior.get(name)
        if not cur or not old:
            continue
        cur_center = np.array(cur["center"], dtype=np.float32)
        old_center = np.array(old["center"], dtype=np.float32)
        cur_size = np.array(cur.get("size", [0.0, 0.0]), dtype=np.float32)
        old_size = np.array(old.get("size", [0.0, 0.0]), dtype=np.float32)
        center_score = max(0.0, 1.0 - float(np.linalg.norm(cur_center - old_center)) / 0.55)
        size_score = max(0.0, 1.0 - float(np.linalg.norm(cur_size - old_size)) / 0.50)
        scores.append(0.8 * center_score + 0.2 * size_score)
    if not scores:
        return None
    return float(np.mean(scores))


def nearest_layouts(current_objects: dict[str, Any], prior_reports: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    matches = []
    for report in prior_reports:
        score = _object_similarity(current_objects, report.get("objects", {}))
        if score is None:
            continue
        matches.append(
            {
                "episode_index": report.get("episode_index"),
                "bucket": report.get("layout", {}).get("bucket"),
                "label": report.get("layout", {}).get("label"),
                "similarity": score,
            }
        )
    return sorted(matches, key=lambda item: item["similarity"], reverse=True)[:limit]


def build_recommendation(
    layout: dict[str, Any],
    coverage: dict[str, Any],
    nearest: list[dict[str, Any]],
    target_count: int,
) -> dict[str, Any]:
    bucket = layout["bucket"]
    bucket_count = 0
    for item in coverage["buckets"]:
        if item["bucket"] == bucket:
            bucket_count = int(item["count"])
            break

    missing = layout["missing_targets"]
    nearest_score = nearest[0]["similarity"] if nearest else None
    warning = bool(missing) or bucket_count >= target_count

    messages = []
    if missing:
        messages.append(f"YOLO did not confidently find: {', '.join(missing)}.")
    elif bucket_count >= target_count:
        messages.append(f"This layout bucket already has {bucket_count} episodes (target {target_count}).")
    else:
        messages.append(f"This layout has {bucket_count} prior episodes; it is useful to record.")

    under = coverage.get("underrepresented_buckets", [])[:3]
    if under:
        messages.append(
            "Underrepresented existing buckets: "
            + "; ".join(f"{item['bucket']} ({item['count']}/{target_count})" for item in under)
        )
    elif not missing and bucket_count >= target_count:
        messages.append("Try moving the bottle or cup into a different left/center/right or upper/middle/lower region.")

    return {
        "warning": warning,
        "bucket_count": bucket_count,
        "nearest_similarity": nearest_score,
        "messages": messages,
    }


def analyze_start_scene(
    rgb: np.ndarray,
    episode_index: int,
    scene_root: Path,
    model_path: str = DEFAULT_MODEL,
    min_confidence: float = 0.25,
    target_count: int = 3,
    min_episode_index: int = 0,
) -> dict[str, Any]:
    detections, annotated = run_yolo_detections(rgb, model_path=model_path, min_confidence=min_confidence)
    objects = select_target_objects(detections)
    layout = build_layout(objects)
    prior_reports = load_scene_reports(scene_root, min_episode_index=min_episode_index)
    coverage = summarize_coverage(prior_reports, target_count=target_count)
    coverage["min_episode_index"] = min_episode_index
    nearest = nearest_layouts(objects, prior_reports)
    recommendation = build_recommendation(layout, coverage, nearest, target_count=target_count)
    return {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "episode_index": episode_index,
        "camera": "top",
        "model": model_path,
        "min_confidence": min_confidence,
        "objects": objects,
        "all_detections": detections,
        "layout": layout,
        "coverage": coverage,
        "nearest_prior_episodes": nearest,
        "recommendation": recommendation,
        "_annotated_rgb": annotated,
    }


def save_scene_artifacts(
    scene_root: Path,
    episode_index: int,
    observation: dict[str, Any],
    report: dict[str, Any],
    camera_names: list[str] | tuple[str, ...],
) -> None:
    scene_root.mkdir(parents=True, exist_ok=True)
    serializable_report = {key: value for key, value in report.items() if not key.startswith("_")}
    for cam_name in camera_names:
        img = observation.get(cam_name)
        if isinstance(img, np.ndarray) and img.ndim == 3:
            cv2.imwrite(
                str(scene_root / f"episode-{episode_index:04d}_{cam_name}_start.jpg"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
    annotated = report.get("_annotated_rgb")
    if isinstance(annotated, np.ndarray):
        cv2.imwrite(
            str(scene_root / f"episode-{episode_index:04d}_top_yolo.jpg"),
            cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
        )
    (scene_root / f"episode-{episode_index:04d}.scene.json").write_text(
        json.dumps(serializable_report, indent=2, default=_json_default),
        encoding="utf-8",
    )
    coverage_config = report.get("coverage", {})
    coverage = summarize_coverage(
        load_scene_reports(scene_root, min_episode_index=int(coverage_config.get("min_episode_index", 0))),
        target_count=coverage_config.get("target_count_per_bucket", 3),
    )
    coverage["min_episode_index"] = int(coverage_config.get("min_episode_index", 0))
    (scene_root / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")


def format_preflight_report(report: dict[str, Any]) -> str:
    if "error" in report:
        return f"\n{YELLOW}Scene check: UNKNOWN{RESET} {DIM}({report['error']}){RESET}"

    recommendation = report.get("recommendation", {})
    coverage = report.get("coverage", {})
    layout = report.get("layout", {})
    warning = recommendation.get("warning")
    status = f"{YELLOW}CHANGE SETUP?{RESET}" if warning else f"{GREEN}OK TO RECORD{RESET}"
    objects = report.get("objects", {})

    target_bits = []
    for name in TARGET_OBJECTS:
        det = objects.get(name)
        if det:
            bucket = layout.get("object_buckets", {}).get(name, {})
            target_bits.append(f"{name}={bucket.get('x', '?')}/{bucket.get('y', '?')} {det['confidence']:.2f}")
        else:
            target_bits.append(f"{name}=missing")

    bucket_count = recommendation.get("bucket_count", 0)
    target_count = coverage.get("target_count_per_bucket", 3)
    lines = [
        "",
        f"{CYAN}Scene check:{RESET} {status}",
        f"  {DIM}{' | '.join(target_bits)} | bucket {bucket_count}/{target_count}{RESET}",
    ]
    if warning:
        message = recommendation.get("messages", ["This setup may be overrepresented."])[0]
        lines.append(f"  {YELLOW}{message}{RESET}")
    return "\n".join(lines)


def format_coverage_summary(scene_root: Path, target_count: int = 3, min_episode_index: int = 0) -> str:
    coverage = summarize_coverage(
        load_scene_reports(scene_root, min_episode_index=min_episode_index),
        target_count=target_count,
    )
    lines = [
        "",
        f"Coverage summary: episodes {min_episode_index}+, {coverage['total_scene_reports']} reports, {coverage['known_bucket_count']} known buckets",
    ]
    for item in coverage["buckets"][:12]:
        lines.append(f"  {item['count']:2d}/{target_count}  {item['bucket']}  {item['status']}")
    if not coverage["buckets"]:
        lines.append("  No object-layout buckets recorded yet.")
    return "\n".join(lines)
