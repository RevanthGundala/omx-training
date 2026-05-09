"""Build a scrollable HTML gallery comparing training and eval episodes.

Examples:
    uv run python evaluation/make_train_eval_gallery.py \
        --train-root outputs/datasets/003-pour-water-new20-only-globalstats \
        --eval-dir outputs/eval_runs \
        --output outputs/train_eval_gallery.html \
        --clips-dir outputs/train_eval_gallery_clips

    uv run python evaluation/make_train_eval_gallery.py \
        --train-root outputs/datasets/003-pour-water-new20-only-globalstats \
        --eval-media outputs/eval_runs/run-001/top.mp4 outputs/eval_runs/run-001/wrist.mp4 \
        --output outputs/train_eval_gallery.html

Eval media grouping:
    Common video files (.mp4, .mov, .m4v, .webm) are discovered recursively under
    every --eval-dir and from each --eval-media path. In --eval-group-by auto
    mode, files in subdirectories are grouped by parent directory, while files
    directly under a supplied eval directory are grouped by filename stem after
    removing camera tokens such as "top" and "wrist". Use --eval-group-by stem
    or --eval-group-by parent to force one behavior.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
CAMERA_PRIORITY = {"top": 0, "wrist": 1}
CAMERA_TOKEN_RE = re.compile(r"(^|[-_.\s])(top|wrist)([-_.\s]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class VideoItem:
    label: str
    src: str
    title: str


@dataclass(frozen=True)
class EpisodeCard:
    source: str
    title: str
    subtitle: str
    anchor: str
    videos: tuple[VideoItem, ...]


def load_episode_metadata(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata parquet files found under {root}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True).sort_values("episode_index")


def training_cameras(episodes: pd.DataFrame) -> list[str]:
    cameras: set[str] = set()
    for column in episodes.columns:
        match = re.fullmatch(r"videos/observation\.images\.([^/]+)/chunk_index", column)
        if match:
            cameras.add(match.group(1))
    return sorted(cameras, key=lambda camera: (CAMERA_PRIORITY.get(camera, 99), camera))


def relative_or_uri(path: Path, output: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(output.resolve().parent).as_posix()
    except ValueError:
        return resolved.as_uri()


def rel_training_video(
    root: Path,
    output: Path,
    camera: str,
    chunk_index: int,
    file_index: int,
    start: float,
    end: float,
) -> str:
    video = root / "videos" / f"observation.images.{camera}" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
    return f"{relative_or_uri(video, output)}#t={start:.3f},{end:.3f}"


def make_clip(
    root: Path,
    clips_dir: Path,
    camera: str,
    episode: int,
    chunk_index: int,
    file_index: int,
    start: float,
    end: float,
) -> Path:
    source = root / "videos" / f"observation.images.{camera}" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
    clip = clips_dir / f"train-episode-{episode:03d}_{camera}.mp4"
    if clip.exists():
        return clip
    clip.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.001, end - start)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-movflags",
            "+faststart",
            str(clip),
        ],
        check=True,
    )
    return clip


def build_training_cards(root: Path, output: Path, clips_dir: Path | None, limit: int | None) -> list[EpisodeCard]:
    episodes = load_episode_metadata(root)
    cameras = training_cameras(episodes)
    if not cameras:
        raise ValueError(f"No video camera metadata columns found under {root}")
    if limit is not None:
        episodes = episodes.head(limit)

    cards: list[EpisodeCard] = []
    for _, row in episodes.iterrows():
        ep = int(row["episode_index"])
        length = int(row["length"])
        duration = length / 30.0
        videos: list[VideoItem] = []
        for camera in cameras:
            prefix = f"videos/observation.images.{camera}"
            chunk = int(row[f"{prefix}/chunk_index"])
            file_index = int(row[f"{prefix}/file_index"])
            start = float(row[f"{prefix}/from_timestamp"])
            end = float(row[f"{prefix}/to_timestamp"])
            if clips_dir is not None:
                video = make_clip(root, clips_dir, camera, ep, chunk, file_index, start, end)
                src = relative_or_uri(video, output)
            else:
                src = rel_training_video(root, output, camera, chunk, file_index, start, end)
            videos.append(VideoItem(label=camera, src=src, title=f"train episode {ep} {camera}"))
        cards.append(
            EpisodeCard(
                source="train",
                title=f"Train episode {ep}",
                subtitle=f"{length} frames · {duration:.1f}s",
                anchor=f"train-episode-{ep}",
                videos=tuple(videos),
            )
        )
    return cards


def discover_eval_files(eval_dirs: list[Path], eval_media: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for directory in eval_dirs:
        if not directory.exists():
            raise FileNotFoundError(f"Eval directory does not exist: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"--eval-dir must be a directory: {directory}")
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                files.add(path)
    for path in eval_media:
        if not path.exists():
            raise FileNotFoundError(f"Eval media path does not exist: {path}")
        if path.is_dir():
            for media in path.rglob("*"):
                if media.is_file() and media.suffix.lower() in VIDEO_EXTENSIONS:
                    files.add(media)
        elif path.suffix.lower() in VIDEO_EXTENSIONS:
            files.add(path)
    return sorted(files)


def infer_camera_label(path: Path, used_labels: set[str]) -> str:
    match = CAMERA_TOKEN_RE.search(path.stem)
    if match:
        return match.group(2).lower()
    label = path.stem
    suffix = 2
    while label in used_labels:
        label = f"{path.stem}-{suffix}"
        suffix += 1
    return label


def stem_episode_key(path: Path) -> str:
    stem = path.stem
    key = CAMERA_TOKEN_RE.sub(lambda match: match.group(1) if match.group(1) else match.group(3), stem)
    key = re.sub(r"[-_.\s]+", "-", key).strip("-_. ")
    return key or stem


def eval_group_key(path: Path, eval_dirs: list[Path], group_by: str) -> str:
    if group_by == "parent":
        return path.parent.as_posix()
    if group_by == "stem":
        return f"{path.parent.as_posix()}/{stem_episode_key(path)}"

    resolved_parent = path.parent.resolve()
    for directory in eval_dirs:
        resolved_dir = directory.resolve()
        try:
            rel_parent = resolved_parent.relative_to(resolved_dir)
        except ValueError:
            continue
        if rel_parent != Path("."):
            return path.parent.as_posix()
        return f"{path.parent.as_posix()}/{stem_episode_key(path)}"
    return f"{path.parent.as_posix()}/{stem_episode_key(path)}"


def build_eval_cards(eval_dirs: list[Path], eval_media: list[Path], output: Path, group_by: str) -> list[EpisodeCard]:
    files = discover_eval_files(eval_dirs, eval_media)
    if not files:
        searched = [str(path) for path in eval_dirs + eval_media]
        raise FileNotFoundError(
            "No eval media found. Expected files with extensions "
            f"{', '.join(sorted(VIDEO_EXTENSIONS))} under: {', '.join(searched) or '(none)'}"
        )

    groups: dict[str, list[Path]] = defaultdict(list)
    group_roots = eval_dirs + [path for path in eval_media if path.is_dir()]
    for path in files:
        groups[eval_group_key(path, group_roots, group_by)].append(path)

    cards: list[EpisodeCard] = []
    for index, (key, paths) in enumerate(sorted(groups.items()), start=1):
        used_labels: set[str] = set()
        videos: list[VideoItem] = []
        for path in sorted(paths, key=lambda item: (CAMERA_PRIORITY.get(infer_camera_label(item, set()), 99), item.name)):
            label = infer_camera_label(path, used_labels)
            used_labels.add(label)
            videos.append(VideoItem(label=label, src=relative_or_uri(path, output), title=path.name))
        display_key = Path(key).name or f"episode-{index:03d}"
        metadata = load_eval_metadata(Path(key))
        details = [display_key, f"{len(videos)} video{'s' if len(videos) != 1 else ''}"]
        if metadata.get("outcome_label"):
            details.append(f"outcome: {metadata['outcome_label']}")
        if metadata.get("notes"):
            details.append(f"notes: {metadata['notes']}")
        cards.append(
            EpisodeCard(
                source="eval",
                title=f"Eval {index:03d}",
                subtitle=" · ".join(details),
                anchor=f"eval-{index:03d}",
                videos=tuple(videos),
            )
        )
    return cards


def load_eval_metadata(group_path: Path) -> dict[str, str]:
    metadata_path = group_path / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {
        "outcome_label": str(data.get("outcome_label") or ""),
        "notes": str(data.get("notes") or ""),
    }


def render_card(card: EpisodeCard) -> str:
    videos = "\n".join(
        f"""
        <figure>
          <figcaption>{html.escape(video.label)}</figcaption>
          <video controls preload="metadata" src="{html.escape(video.src)}" title="{html.escape(video.title)}"></video>
        </figure>
        """
        for video in card.videos
    )
    return f"""
    <article class="card {html.escape(card.source)}" id="{html.escape(card.anchor)}">
      <header>
        <h3>{html.escape(card.title)}</h3>
        <span>{html.escape(card.subtitle)}</span>
      </header>
      <div class="buttons">
        <button onclick="playCard(this)">Play all</button>
        <button onclick="pauseCard(this)">Pause all</button>
        <a href="#{html.escape(card.anchor)}">link</a>
      </div>
      <div class="videos">
        {videos}
      </div>
    </article>
    """


def render_section(title: str, cards: list[EpisodeCard]) -> str:
    return f"""
    <section class="column">
      <h2>{html.escape(title)} <span>{len(cards)} episodes</span></h2>
      <div class="stack">
        {''.join(render_card(card) for card in cards)}
      </div>
    </section>
    """


def build_gallery(
    train_root: Path,
    eval_dirs: list[Path],
    eval_media: list[Path],
    output: Path,
    clips_dir: Path | None,
    eval_group_by: str,
    max_train_episodes: int | None,
) -> None:
    train_cards = build_training_cards(train_root, output, clips_dir, max_train_episodes)
    eval_cards = build_eval_cards(eval_dirs, eval_media, output, eval_group_by)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Train/eval episode gallery</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #111;
      color: #eee;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: #111;
    }}
    .sticky {{
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 12px 0 18px;
      background: linear-gradient(#111 80%, transparent);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    .hint {{
      color: #bbb;
      margin: 0;
    }}
    .legend {{
      display: flex;
      gap: 16px;
      margin-top: 12px;
      color: #ccc;
    }}
    .dot {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      margin-right: 6px;
      vertical-align: -1px;
    }}
    .train-dot {{ background: #3b82f6; }}
    .eval-dot {{ background: #f97316; }}
    .comparison {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(720px, 1fr));
      gap: 22px;
      align-items: start;
    }}
    .column h2 {{
      margin: 0 0 12px;
      font-size: 22px;
    }}
    .column h2 span {{
      color: #aaa;
      font-size: 14px;
      font-weight: 400;
    }}
    .stack {{
      display: grid;
      gap: 22px;
    }}
    .card {{
      background: #1c1c1f;
      border: 2px solid #333;
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
    }}
    .card.train {{ border-color: #244b91; }}
    .card.eval {{ border-color: #9a4a11; }}
    .card header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .card h3 {{
      margin: 0;
      font-size: 22px;
    }}
    .card header span {{
      color: #bbb;
      font-size: 14px;
    }}
    .buttons {{
      display: flex;
      gap: 8px;
      align-items: center;
      margin-bottom: 12px;
    }}
    button, a {{
      background: #2d2d33;
      color: #eee;
      border: 1px solid #555;
      border-radius: 8px;
      padding: 7px 10px;
      text-decoration: none;
      font-size: 14px;
      cursor: pointer;
    }}
    button:hover, a:hover {{ background: #3a3a43; }}
    .videos {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 12px;
    }}
    figure {{
      margin: 0;
    }}
    figcaption {{
      color: #ccc;
      font-size: 14px;
      margin-bottom: 5px;
    }}
    video {{
      display: block;
      width: 100%;
      max-height: 420px;
      background: #000;
      border-radius: 10px;
    }}
  </style>
</head>
<body>
  <div class="sticky">
    <h1>Train/eval episode gallery</h1>
    <p class="hint">Scroll each source column. Every card has independent controls; top/wrist videos are side-by-side when both are available.</p>
    <div class="legend">
      <span><span class="dot train-dot"></span>training dataset: {html.escape(train_root.as_posix())}</span>
      <span><span class="dot eval-dot"></span>eval media</span>
    </div>
  </div>
  <main class="comparison">
    {render_section("Training episodes", train_cards)}
    {render_section("Eval episodes", eval_cards)}
  </main>
  <script>
    function videosFor(button) {{
      return button.closest('.card').querySelectorAll('video');
    }}
    function playCard(button) {{
      videosFor(button).forEach((video) => video.play());
    }}
    function pauseCard(button) {{
      videosFor(button).forEach((video) => video.pause());
    }}
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-root", type=Path, required=True, help="LeRobot dataset root containing meta/episodes and videos.")
    parser.add_argument("--eval-dir", type=Path, action="append", default=[], help="Directory to recursively scan for eval videos. May be repeated.")
    parser.add_argument("--eval-media", type=Path, nargs="*", default=[], help="Explicit eval video files or directories to include.")
    parser.add_argument("--output", type=Path, default=Path("outputs/train_eval_gallery.html"))
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=None,
        help="If set, cut browser-friendly per-episode clips for training dataset videos with ffmpeg.",
    )
    parser.add_argument(
        "--eval-group-by",
        choices=("auto", "parent", "stem"),
        default="auto",
        help="How to group eval videos into episodes. Auto uses subdirectory parent groups and filename stem groups at eval-dir roots.",
    )
    parser.add_argument("--max-train-episodes", type=int, default=None, help="Optional limit for quick inspection galleries.")
    args = parser.parse_args()

    clips_dir = args.clips_dir
    if clips_dir is not None and not clips_dir.is_absolute():
        clips_dir = args.output.parent / clips_dir

    build_gallery(
        train_root=args.train_root,
        eval_dirs=args.eval_dir,
        eval_media=args.eval_media,
        output=args.output,
        clips_dir=clips_dir,
        eval_group_by=args.eval_group_by,
        max_train_episodes=args.max_train_episodes,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
