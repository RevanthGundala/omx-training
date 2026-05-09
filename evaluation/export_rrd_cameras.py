"""Export camera streams from Rerun .rrd eval recordings to MP4 files.

Example:
    uv run python evaluation/export_rrd_cameras.py \
        --rrd-dir outputs/eval_runs \
        --output-dir outputs/eval_runs_videos
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from rerun import dataframe as rrd


def _cell_value(value):
    if isinstance(value, np.ndarray) and value.dtype == object and len(value) == 1:
        return value[0]
    return value


def _format_value(value) -> dict:
    value = _cell_value(value)
    if isinstance(value, dict):
        return value
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return dict(value)


def _image_from_cells(buffer_value, format_value) -> np.ndarray | None:
    if buffer_value is None or format_value is None:
        return None
    buffer = _cell_value(buffer_value)
    fmt = _format_value(format_value)
    width = int(fmt["width"])
    height = int(fmt["height"])
    arr = np.asarray(buffer, dtype=np.uint8)
    if arr.size == width * height * 3:
        return arr.reshape(height, width, 3)
    if arr.size == width * height * 4:
        return arr.reshape(height, width, 4)[:, :, :3]
    if arr.size == width * height:
        return np.repeat(arr.reshape(height, width, 1), 3, axis=2)
    raise ValueError(f"Unexpected image buffer size {arr.size} for {width}x{height}")


def export_recording(path: Path, output_dir: Path, fps: float) -> list[Path]:
    rec = rrd.load_recording(str(path))
    raw = rec.view(index="step", contents="camera/**").select().read_pandas()
    if raw.empty:
        return []

    cameras = []
    for column in raw.columns:
        name = str(column)
        if name.startswith("/camera/") and name.endswith(":Image:buffer"):
            cameras.append(name.removeprefix("/camera/").removesuffix(":Image:buffer"))
    cameras = sorted(set(cameras))

    run_dir = output_dir / path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for camera in cameras:
        buffer_col = f"/camera/{camera}:Image:buffer"
        format_col = f"/camera/{camera}:Image:format"
        frames = []
        for _, row in raw[[buffer_col, format_col]].dropna().iterrows():
            frames.append(_image_from_cells(row[buffer_col], row[format_col]))
        frames = [frame for frame in frames if frame is not None]
        if not frames:
            continue

        height, width = frames[0].shape[:2]
        out = run_dir / f"{camera}.mp4"
        writer = cv2.VideoWriter(
            str(out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {out}")
        try:
            for rgb in frames:
                if rgb.shape[:2] != (height, width):
                    rgb = cv2.resize(rgb, (width, height))
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        written.append(out)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rrd", type=Path, action="append", default=[], help="Specific .rrd recording; may be repeated.")
    parser.add_argument("--rrd-dir", type=Path, action="append", default=[], help="Directory to scan for .rrd recordings; may be repeated.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_runs_videos"))
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    paths = set(args.rrd)
    for directory in args.rrd_dir:
        paths.update(directory.glob("*.rrd"))
    recordings = sorted(path for path in paths if path.exists() and path.suffix == ".rrd")
    if not recordings:
        raise FileNotFoundError("No .rrd recordings found. Provide --rrd or --rrd-dir.")

    total = 0
    for recording in recordings:
        outputs = export_recording(recording, args.output_dir, args.fps)
        total += len(outputs)
        print(f"{recording}: wrote {len(outputs)} video(s)")
        for output in outputs:
            print(f"  {output}")
    print(f"Wrote {total} video(s) under {args.output_dir}")


if __name__ == "__main__":
    main()
