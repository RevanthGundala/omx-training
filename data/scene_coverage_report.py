"""Annotate and review object-layout coverage for teleop collection."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import cv2

from data.scene_assist import (
    DEFAULT_MODEL,
    analyze_start_scene,
    format_coverage_summary,
    load_scene_reports,
    save_scene_artifacts,
    summarize_coverage,
)
from utils.config import CAMERAS, FPS, RECORD_DATASET_REPO_ID


def _default_dataset_root(repo_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id


def _default_scene_root(repo_id: str) -> Path:
    return Path("outputs/record_scene_diversity") / repo_id.replace("/", "__")


def _load_episode_rows(dataset_root: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read LeRobot episode metadata") from exc

    episode_files = sorted((dataset_root / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No meta/episodes parquet files found under {dataset_root}")
    episodes = pd.concat([pd.read_parquet(path) for path in episode_files], ignore_index=True)
    return episodes.sort_values("episode_index").to_dict("records")


def _read_start_frame(dataset_root: Path, episode_row: dict[str, Any], camera_name: str) -> Any:
    prefix = f"videos/observation.images.{camera_name}"
    required = [f"{prefix}/chunk_index", f"{prefix}/file_index", f"{prefix}/from_timestamp"]
    if not all(key in episode_row for key in required):
        return None
    video_path = (
        dataset_root
        / "videos"
        / f"observation.images.{camera_name}"
        / f"chunk-{int(episode_row[f'{prefix}/chunk_index']):03d}"
        / f"file-{int(episode_row[f'{prefix}/file_index']):03d}.mp4"
    )
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    try:
        frame_index = max(0, int(float(episode_row[f"{prefix}/from_timestamp"]) * FPS))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def annotate_dataset_starts(
    dataset_root: Path,
    scene_root: Path,
    camera_name: str,
    model_path: str,
    min_confidence: float,
    target_count: int,
    min_episode_index: int,
    limit: int | None = None,
) -> int:
    scene_root.mkdir(parents=True, exist_ok=True)
    rows = _load_episode_rows(dataset_root)
    created = 0
    for row in rows[:limit]:
        episode_index = int(row["episode_index"])
        if episode_index < min_episode_index:
            continue
        if (scene_root / f"episode-{episode_index:04d}.scene.json").exists():
            continue

        observation = {}
        for cam_name in CAMERAS:
            frame = _read_start_frame(dataset_root, row, cam_name)
            if frame is not None:
                observation[cam_name] = frame
        top_frame = observation.get(camera_name)
        if top_frame is None:
            print(f"episode {episode_index:04d}: skipped, missing {camera_name} frame")
            continue

        report = analyze_start_scene(
            top_frame,
            episode_index=episode_index,
            scene_root=scene_root,
            model_path=model_path,
            min_confidence=min_confidence,
            target_count=target_count,
            min_episode_index=min_episode_index,
        )
        report["bootstrapped_from_dataset"] = True
        save_scene_artifacts(scene_root, episode_index, observation, report, tuple(CAMERAS.keys()))
        created += 1
        print(f"episode {episode_index:04d}: {report['layout']['label']}")
    return created


def _relative(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base.parent).as_posix()
    except ValueError:
        return path.as_posix()


def write_html_report(scene_root: Path, output: Path, target_count: int, min_episode_index: int) -> None:
    scene_root.mkdir(parents=True, exist_ok=True)
    reports = load_scene_reports(scene_root, min_episode_index=min_episode_index)
    coverage = summarize_coverage(reports, target_count=target_count)
    coverage["min_episode_index"] = min_episode_index
    coverage_path = scene_root / "coverage.json"
    coverage_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    rows = []
    for report in reports:
        episode = int(report.get("episode_index", -1))
        bucket = report.get("layout", {}).get("bucket", "uncertain")
        label = report.get("layout", {}).get("label", "uncertain")
        objects = report.get("objects", {})
        bottle = objects.get("bottle", {})
        cup = objects.get("cup", {})
        image = scene_root / f"episode-{episode:04d}_top_yolo.jpg"
        image_html = ""
        if image.exists():
            image_html = f'<img src="{html.escape(_relative(image, output))}" loading="lazy">'
        rows.append(
            "<tr>"
            f"<td>{episode}</td>"
            f"<td>{image_html}</td>"
            f"<td><code>{html.escape(bucket)}</code><br>{html.escape(label)}</td>"
            f"<td>bottle conf={float(bottle.get('confidence', 0.0)):.2f}<br>"
            f"cup conf={float(cup.get('confidence', 0.0)):.2f}</td>"
            f"<td>{html.escape(str(report.get('operator_decision', '')))}</td>"
            "</tr>"
        )

    bucket_rows = [
        "<tr>"
        f"<td><code>{html.escape(item['bucket'])}</code></td>"
        f"<td>{item['count']}</td>"
        f"<td>{html.escape(item['status'])}</td>"
        "</tr>"
        for item in coverage["buckets"]
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Teleop Scene Coverage</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 32px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f5f5f5; text-align: left; }}
    img {{ width: 280px; max-height: 210px; object-fit: contain; }}
    code {{ white-space: nowrap; }}
  </style>
</head>
<body>
  <h1>Teleop Scene Coverage</h1>
  <p>Episodes included: {min_episode_index}+ &nbsp; Target per bucket: {target_count}</p>
  <h2>Bucket counts</h2>
  <table>
    <thead><tr><th>Bucket</th><th>Count</th><th>Status</th></tr></thead>
    <tbody>{"\n".join(bucket_rows)}</tbody>
  </table>
  <h2>Episodes</h2>
  <table>
    <thead><tr><th>Episode</th><th>Top YOLO</th><th>Layout</th><th>Detections</th><th>Decision</th></tr></thead>
    <tbody>{"\n".join(rows)}</tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=RECORD_DATASET_REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--scene-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--camera", default="top")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--target-count", type=int, default=4)
    parser.add_argument("--min-episode-index", type=int, default=50)
    parser.add_argument("--no-annotate", action="store_true", help="Only report existing scene JSON files.")
    parser.add_argument("--limit", type=int, default=None, help="Annotate at most this many dataset episodes.")
    args = parser.parse_args()

    dataset_root = args.dataset_root or _default_dataset_root(args.repo_id)
    scene_root = args.scene_root or _default_scene_root(args.repo_id)
    output = args.output or scene_root / "coverage.html"

    if not args.no_annotate:
        created = annotate_dataset_starts(
            dataset_root=dataset_root,
            scene_root=scene_root,
            camera_name=args.camera,
            model_path=args.model,
            min_confidence=args.min_confidence,
            target_count=args.target_count,
            min_episode_index=args.min_episode_index,
            limit=args.limit,
        )
        print(f"Annotated {created} new start scenes.")

    write_html_report(scene_root, output, target_count=args.target_count, min_episode_index=args.min_episode_index)
    print(
        format_coverage_summary(
            scene_root,
            target_count=args.target_count,
            min_episode_index=args.min_episode_index,
        )
    )
    print(f"\nCoverage report: {output}")
    print(f"Coverage JSON:   {scene_root / 'coverage.json'}")


if __name__ == "__main__":
    main()
