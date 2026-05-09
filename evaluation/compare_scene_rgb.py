"""Compare RGB/luminance statistics between training and eval scene media.

Example:
    uv run python evaluation/compare_scene_rgb.py \
        --train-root outputs/datasets/003-pour-water-new20-only-globalstats \
        --eval-dir outputs/episode_gallery_all70_clips \
        --output-csv outputs/scene_rgb_compare.csv \
        --output-html outputs/scene_rgb_compare.html
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - exercised by users without opencv
    raise SystemExit("compare_scene_rgb.py requires OpenCV (cv2), already used by this repo's camera tools.") from exc


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
CAMERA_HINTS = ("top", "wrist")


@dataclass
class SourceStats:
    dataset: str
    camera: str
    path: str
    frames: int = 0
    pixels: int = 0
    sums: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    sums_sq: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    hist: np.ndarray = field(default_factory=lambda: np.zeros((4, 256), dtype=np.int64))
    dark_pixels: int = 0
    bright_pixels: int = 0

    def add_frame(self, rgb: np.ndarray, pixel_stride: int, dark_threshold: int, bright_threshold: int) -> None:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return
        pixels = rgb[::pixel_stride, ::pixel_stride].reshape(-1, 3).astype(np.float64)
        if pixels.size == 0:
            return

        luminance = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
        values = np.column_stack((pixels, luminance))

        self.frames += 1
        self.pixels += int(values.shape[0])
        self.sums += values.sum(axis=0)
        self.sums_sq += np.square(values).sum(axis=0)

        clipped = np.clip(np.rint(values), 0, 255).astype(np.uint8)
        for channel in range(4):
            self.hist[channel] += np.bincount(clipped[:, channel], minlength=256)
        self.dark_pixels += int(np.count_nonzero(luminance <= dark_threshold))
        self.bright_pixels += int(np.count_nonzero(luminance >= bright_threshold))

    def merge(self, other: "SourceStats") -> None:
        self.frames += other.frames
        self.pixels += other.pixels
        self.sums += other.sums
        self.sums_sq += other.sums_sq
        self.hist += other.hist
        self.dark_pixels += other.dark_pixels
        self.bright_pixels += other.bright_pixels

    def metrics(self) -> dict[str, float | int | str]:
        if self.pixels == 0:
            return {
                "dataset": self.dataset,
                "camera": self.camera,
                "source_path": self.path,
                "frames": self.frames,
                "pixels": self.pixels,
            }

        means = self.sums / self.pixels
        variances = np.maximum(self.sums_sq / self.pixels - np.square(means), 0.0)
        stds = np.sqrt(variances)
        qs = np.array([_hist_quantile(self.hist[channel], q) for channel in range(4) for q in (0.05, 0.50, 0.95)])

        return {
            "dataset": self.dataset,
            "camera": self.camera,
            "source_path": self.path,
            "frames": self.frames,
            "pixels": self.pixels,
            "mean_r": means[0],
            "mean_g": means[1],
            "mean_b": means[2],
            "std_r": stds[0],
            "std_g": stds[1],
            "std_b": stds[2],
            "lum_mean": means[3],
            "lum_std": stds[3],
            "r_q05": qs[0],
            "r_q50": qs[1],
            "r_q95": qs[2],
            "g_q05": qs[3],
            "g_q50": qs[4],
            "g_q95": qs[5],
            "b_q05": qs[6],
            "b_q50": qs[7],
            "b_q95": qs[8],
            "lum_q05": qs[9],
            "lum_q50": qs[10],
            "lum_q95": qs[11],
            "dark_fraction": self.dark_pixels / self.pixels,
            "bright_fraction": self.bright_pixels / self.pixels,
        }


def _hist_quantile(hist: np.ndarray, q: float) -> float:
    total = int(hist.sum())
    if total <= 0:
        return math.nan
    target = max(1, int(math.ceil(q * total)))
    return float(np.searchsorted(np.cumsum(hist), target))


def detect_camera(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    for part in parts:
        if part.startswith("observation.images."):
            return part.rsplit(".", 1)[-1]

    stem_tokens = path.stem.lower().replace("-", "_").split("_")
    for hint in CAMERA_HINTS:
        if hint in stem_tokens or any(part == hint or part.endswith(f"_{hint}") for part in parts):
            return hint
    return "unknown"


def discover_train_media(train_root: Path) -> list[Path]:
    videos_root = train_root / "videos"
    if videos_root.exists():
        media = [path for path in videos_root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    else:
        media = [path for path in train_root.rglob("*") if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS]
    return sorted(media)


def discover_eval_media(paths: Iterable[Path]) -> list[Path]:
    media: list[Path] = []
    for path in paths:
        if path.is_dir():
            media.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS)
        elif path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS:
            media.append(path)
        else:
            print(f"Skipping eval path with no supported media: {path}")
    return sorted(set(media))


def read_image_rgb(path: Path) -> np.ndarray | None:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def sample_video(
    path: Path,
    stats: SourceStats,
    max_frames: int,
    frame_stride: int,
    pixel_stride: int,
    dark_threshold: int,
    bright_threshold: int,
) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Warning: could not open video: {path}")
        return
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 0:
            step = max(1, frame_stride, total // max(1, max_frames)) if max_frames else max(1, frame_stride)
            indices = range(0, total, step)
            for sampled, frame_index in enumerate(indices):
                if max_frames and sampled >= max_frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, bgr = cap.read()
                if ok:
                    stats.add_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), pixel_stride, dark_threshold, bright_threshold)
        else:
            frame_index = 0
            sampled = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if frame_index % max(1, frame_stride) == 0:
                    stats.add_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), pixel_stride, dark_threshold, bright_threshold)
                    sampled += 1
                    if max_frames and sampled >= max_frames:
                        break
                frame_index += 1
    finally:
        cap.release()


def sample_media(
    dataset: str,
    media_paths: list[Path],
    max_frames_per_video: int,
    frame_stride: int,
    pixel_stride: int,
    dark_threshold: int,
    bright_threshold: int,
) -> tuple[list[SourceStats], dict[str, SourceStats]]:
    source_stats: list[SourceStats] = []
    aggregates: dict[str, SourceStats] = {}

    for path in media_paths:
        camera = detect_camera(path)
        stats = SourceStats(dataset=dataset, camera=camera, path=path.as_posix())
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            rgb = read_image_rgb(path)
            if rgb is not None:
                stats.add_frame(rgb, pixel_stride, dark_threshold, bright_threshold)
            else:
                print(f"Warning: could not read image: {path}")
        else:
            sample_video(path, stats, max_frames_per_video, frame_stride, pixel_stride, dark_threshold, bright_threshold)

        source_stats.append(stats)
        for key in (camera, "overall"):
            aggregates.setdefault(key, SourceStats(dataset=dataset, camera=key, path="")).merge(stats)

    return source_stats, aggregates


def compare_aggregates(
    train: dict[str, SourceStats],
    eval_: dict[str, SourceStats],
    rgb_delta_threshold: float,
    luminance_delta_threshold: float,
    fraction_delta_threshold: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for camera in sorted(set(train) & set(eval_), key=lambda value: (value != "overall", value)):
        train_metrics = train[camera].metrics()
        eval_metrics = eval_[camera].metrics()
        if train_metrics.get("pixels", 0) == 0 or eval_metrics.get("pixels", 0) == 0:
            continue

        train_rgb = np.array([train_metrics["mean_r"], train_metrics["mean_g"], train_metrics["mean_b"]], dtype=float)
        eval_rgb = np.array([eval_metrics["mean_r"], eval_metrics["mean_g"], eval_metrics["mean_b"]], dtype=float)
        rgb_delta = float(np.linalg.norm(eval_rgb - train_rgb))
        lum_delta = float(abs(float(eval_metrics["lum_mean"]) - float(train_metrics["lum_mean"])))
        dark_delta = float(abs(float(eval_metrics["dark_fraction"]) - float(train_metrics["dark_fraction"])))
        bright_delta = float(abs(float(eval_metrics["bright_fraction"]) - float(train_metrics["bright_fraction"])))
        flags = []
        if rgb_delta >= rgb_delta_threshold:
            flags.append("rgb")
        if lum_delta >= luminance_delta_threshold:
            flags.append("luminance")
        if max(dark_delta, bright_delta) >= fraction_delta_threshold:
            flags.append("dark/bright")

        rows.append(
            {
                "row_type": "comparison",
                "dataset": "eval_vs_train",
                "camera": camera,
                "source_path": "",
                "frames": int(eval_metrics["frames"]),
                "pixels": int(eval_metrics["pixels"]),
                "delta_rgb_norm": rgb_delta,
                "delta_lum_mean": lum_delta,
                "delta_dark_fraction": dark_delta,
                "delta_bright_fraction": bright_delta,
                "flag": ",".join(flags) if flags else "",
                "notes": (
                    f"train_lum={float(train_metrics['lum_mean']):.1f}; "
                    f"eval_lum={float(eval_metrics['lum_mean']):.1f}"
                ),
            }
        )
    return rows


def stats_rows(row_type: str, stats: Iterable[SourceStats]) -> list[dict[str, float | int | str]]:
    rows = []
    for item in stats:
        row = {"row_type": row_type, **item.metrics(), "flag": "", "notes": ""}
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = [
        "row_type",
        "dataset",
        "camera",
        "source_path",
        "frames",
        "pixels",
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
        "lum_mean",
        "lum_std",
        "r_q05",
        "r_q50",
        "r_q95",
        "g_q05",
        "g_q50",
        "g_q95",
        "b_q05",
        "b_q50",
        "b_q95",
        "lum_q05",
        "lum_q50",
        "lum_q95",
        "dark_fraction",
        "bright_fraction",
        "delta_rgb_norm",
        "delta_lum_mean",
        "delta_dark_fraction",
        "delta_bright_fraction",
        "flag",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row.get(key, "")) for key in fieldnames})


def _format_cell(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def write_html(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    summary = [row for row in rows if row["row_type"] in {"aggregate", "comparison"}]
    source = [row for row in rows if row["row_type"] == "source"]

    def table(title: str, table_rows: list[dict[str, float | int | str]], columns: list[str]) -> str:
        body = []
        for row in table_rows:
            cls = "flagged" if row.get("flag") else ""
            cells = "".join(f"<td>{html.escape(str(_format_cell(row.get(col, ''))))}</td>" for col in columns)
            body.append(f"<tr class='{cls}'>{cells}</tr>")
        headers = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
        return f"<h2>{html.escape(title)}</h2><table><thead><tr>{headers}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Scene RGB comparison</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 5px 7px; text-align: right; }}
    th:nth-child(-n+4), td:nth-child(-n+4) {{ text-align: left; }}
    th {{ background: #f4f4f4; position: sticky; top: 0; }}
    tr.flagged {{ background: #fff1f1; }}
    code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Scene RGB comparison</h1>
  <p>Flagged comparison rows exceeded one or more configured thresholds.</p>
  {table("Aggregate and comparison", summary, ["row_type", "dataset", "camera", "frames", "mean_r", "mean_g", "mean_b", "lum_mean", "lum_std", "dark_fraction", "bright_fraction", "delta_rgb_norm", "delta_lum_mean", "flag", "notes"])}
  {table("Per-source samples", source, ["dataset", "camera", "source_path", "frames", "mean_r", "mean_g", "mean_b", "lum_mean", "lum_q05", "lum_q50", "lum_q95", "dark_fraction", "bright_fraction"])}
</body>
</html>
""",
        encoding="utf-8",
    )


def print_summary(comparisons: list[dict[str, float | int | str]], train_media: int, eval_media: int) -> None:
    print(f"Sampled {train_media} train media files and {eval_media} eval media files.")
    if not comparisons:
        print("No matching train/eval camera labels to compare.")
        return
    print("\n=== Eval vs train visual deltas ===")
    print(f"{'camera':10s} {'rgbΔ':>8s} {'lumΔ':>8s} {'darkΔ':>8s} {'brightΔ':>8s}  flags")
    for row in comparisons:
        print(
            f"{str(row['camera']):10s} "
            f"{float(row['delta_rgb_norm']):8.1f} "
            f"{float(row['delta_lum_mean']):8.1f} "
            f"{float(row['delta_dark_fraction']):8.3f} "
            f"{float(row['delta_bright_fraction']):8.3f}  "
            f"{row.get('flag') or '-'}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True, help="LeRobot dataset root with videos/observation.images.*")
    parser.add_argument("--eval-dir", type=Path, action="append", default=[], help="Eval media directory; may be repeated")
    parser.add_argument("--eval-media", type=Path, nargs="*", default=[], help="Specific eval media files or directories")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/scene_rgb_compare.csv"))
    parser.add_argument("--output-html", type=Path, default=None, help="Optional HTML report path")
    parser.add_argument("--max-frames-per-video", type=int, default=60)
    parser.add_argument("--frame-stride", type=int, default=1, help="Minimum stride between sampled video frames")
    parser.add_argument("--pixel-stride", type=int, default=8, help="Sample every Nth pixel in x/y inside selected frames")
    parser.add_argument("--dark-threshold", type=int, default=32, help="Luminance <= this counts as dark")
    parser.add_argument("--bright-threshold", type=int, default=224, help="Luminance >= this counts as bright")
    parser.add_argument("--rgb-delta-threshold", type=float, default=35.0)
    parser.add_argument("--luminance-delta-threshold", type=float, default=30.0)
    parser.add_argument("--fraction-delta-threshold", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_inputs = [*args.eval_dir, *args.eval_media]
    if not eval_inputs:
        raise SystemExit("Provide at least one --eval-dir or --eval-media path.")
    if args.pixel_stride < 1 or args.frame_stride < 1:
        raise SystemExit("--pixel-stride and --frame-stride must be >= 1.")

    train_media = discover_train_media(args.train_root)
    eval_media = discover_eval_media(eval_inputs)
    if not train_media:
        raise FileNotFoundError(f"No supported train media found under {args.train_root}")
    if not eval_media:
        raise FileNotFoundError(f"No supported eval media found under: {', '.join(str(p) for p in eval_inputs)}")

    train_sources, train_aggregates = sample_media(
        "train",
        train_media,
        args.max_frames_per_video,
        args.frame_stride,
        args.pixel_stride,
        args.dark_threshold,
        args.bright_threshold,
    )
    eval_sources, eval_aggregates = sample_media(
        "eval",
        eval_media,
        args.max_frames_per_video,
        args.frame_stride,
        args.pixel_stride,
        args.dark_threshold,
        args.bright_threshold,
    )
    comparisons = compare_aggregates(
        train_aggregates,
        eval_aggregates,
        args.rgb_delta_threshold,
        args.luminance_delta_threshold,
        args.fraction_delta_threshold,
    )

    rows = [
        *stats_rows("aggregate", train_aggregates.values()),
        *stats_rows("aggregate", eval_aggregates.values()),
        *comparisons,
        *stats_rows("source", train_sources),
        *stats_rows("source", eval_sources),
    ]
    write_csv(args.output_csv, rows)
    if args.output_html is not None:
        write_html(args.output_html, rows)

    print_summary(comparisons, len(train_media), len(eval_media))
    print(f"\nWrote CSV: {args.output_csv}")
    if args.output_html is not None:
        print(f"Wrote HTML: {args.output_html}")


if __name__ == "__main__":
    main()
