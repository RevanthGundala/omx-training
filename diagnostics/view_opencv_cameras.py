"""Live OpenCV camera viewer for checking camera indices after replugging.

Examples:
    uv run python diagnostics/view_opencv_cameras.py
    uv run python diagnostics/view_opencv_cameras.py --indices 0 1 2
    uv run python diagnostics/view_opencv_cameras.py --scan
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np

from utils.config import CAMERAS, CAMERA_HEIGHT, CAMERA_WIDTH


@dataclass
class CameraStream:
    label: str
    index: int
    cap: cv2.VideoCapture
    width: int
    height: int
    last_frame: np.ndarray | None = None
    last_ok: bool = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        help="Open these camera indices instead of the configured CAMERAS map.",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Probe indices 0..max-index first and print which ones return frames.",
    )
    parser.add_argument("--max-index", type=int, default=5, help="Highest index to probe with --scan.")
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def _open_camera(label: str, index: int, width: int, height: int, fps: int) -> CameraStream | None:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"{label}: index {index} did not open")
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return CameraStream(label=label, index=index, cap=cap, width=width, height=height)


def _probe_indices(max_index: int, width: int, height: int, fps: int) -> None:
    print("Scanning OpenCV camera indices...")
    for index in range(max_index + 1):
        stream = _open_camera(f"camera_{index}", index, width, height, fps)
        if stream is None:
            continue
        ok = False
        frame = None
        start = time.perf_counter()
        while time.perf_counter() - start < 2.0:
            ok, frame = stream.cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        if ok and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(
                f"camera_{index}: opened, frame={frame.shape[1]}x{frame.shape[0]}, "
                f"mean_gray={gray.mean():.1f}"
            )
        else:
            print(f"camera_{index}: opened, but no frame")
        stream.cap.release()


def _placeholder(stream: CameraStream) -> np.ndarray:
    frame = np.zeros((stream.height, stream.width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "NO FRAME",
        (max(10, stream.width // 2 - 90), stream.height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
    )
    return frame


def _annotate(stream: CameraStream, frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    status = "ok" if stream.last_ok else "no frame"
    text = f"{stream.label} | index={stream.index} | {status} | mean_gray={gray.mean():.1f}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return out


def _tile_frames(frames: list[np.ndarray]) -> np.ndarray:
    if len(frames) == 1:
        return frames[0]
    heights = [frame.shape[0] for frame in frames]
    target_h = min(heights)
    resized = []
    for frame in frames:
        scale = target_h / frame.shape[0]
        target_w = int(frame.shape[1] * scale)
        resized.append(cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA))
    return np.hstack(resized)


def main() -> int:
    args = _parse_args()

    if args.scan:
        _probe_indices(args.max_index, args.width, args.height, args.fps)

    if args.indices:
        camera_items = [(f"camera_{index}", index) for index in args.indices]
    else:
        camera_items = list(CAMERAS.items())

    streams = [
        stream
        for label, index in camera_items
        if (stream := _open_camera(label, index, args.width, args.height, args.fps)) is not None
    ]
    if not streams:
        print("No cameras opened.")
        return 1

    print("Press q or Esc in the OpenCV cameras window to quit.")
    try:
        while True:
            frames = []
            for stream in streams:
                ok, frame = stream.cap.read()
                if ok and frame is not None:
                    stream.last_ok = True
                    stream.last_frame = frame
                elif stream.last_frame is not None:
                    stream.last_ok = False
                    frame = stream.last_frame.copy()
                    cv2.putText(frame, "NO NEW FRAME", (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    stream.last_ok = False
                    frame = _placeholder(stream)
                frames.append(_annotate(stream, frame))

            cv2.imshow("OpenCV cameras", _tile_frames(frames))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        for stream in streams:
            stream.cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
