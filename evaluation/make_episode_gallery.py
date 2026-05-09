"""Build a scrollable HTML gallery for visually reviewing LeRobot episodes.

Example:
    uv run python evaluation/make_episode_gallery.py \
        --root outputs/datasets/003-pour-water-globalstats \
        --output outputs/episode_gallery.html
"""

from __future__ import annotations

import argparse
import html
import subprocess
from pathlib import Path

import pandas as pd


def load_episode_metadata(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata parquet files found under {root}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True).sort_values("episode_index")


def rel_video(root: Path, output: Path, camera: str, chunk_index: int, file_index: int, start: float, end: float) -> str:
    video = root / "videos" / f"observation.images.{camera}" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
    rel = video.resolve().relative_to(output.resolve().parent)
    return f"{rel.as_posix()}#t={start:.3f},{end:.3f}"


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
    clip = clips_dir / f"episode-{episode:03d}_{camera}.mp4"
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


def build_gallery(root: Path, output: Path, repo_id: str | None, clips_dir: Path | None) -> None:
    episodes = load_episode_metadata(root)
    cards: list[str] = []
    for _, row in episodes.iterrows():
        ep = int(row["episode_index"])
        length = int(row["length"])
        duration = length / 30.0
        tag = "new20" if ep >= 50 else "old50"
        videos = []
        for camera in ("top", "wrist"):
            chunk = int(row[f"videos/observation.images.{camera}/chunk_index"])
            file_index = int(row[f"videos/observation.images.{camera}/file_index"])
            start = float(row[f"videos/observation.images.{camera}/from_timestamp"])
            end = float(row[f"videos/observation.images.{camera}/to_timestamp"])
            if clips_dir is not None:
                video = make_clip(root, clips_dir, camera, ep, chunk, file_index, start, end)
                src = html.escape(video.resolve().relative_to(output.resolve().parent).as_posix())
            else:
                src = html.escape(rel_video(root, output, camera, chunk, file_index, start, end))
            videos.append(
                f"""
                <figure>
                  <figcaption>{camera}</figcaption>
                  <video controls preload="metadata" src="{src}"></video>
                </figure>
                """
            )
        cards.append(
            f"""
            <article class="card {tag}" id="episode-{ep}">
              <header>
                <h2>Episode {ep}</h2>
                <span>{tag} · {length} frames · {duration:.1f}s</span>
              </header>
              <div class="buttons">
                <button onclick="playCard(this)">Play both</button>
                <button onclick="pauseCard(this)">Pause both</button>
                <a href="#episode-{ep}">link</a>
              </div>
              <div class="videos">
                {''.join(videos)}
              </div>
            </article>
            """
        )

    title = repo_id or root.name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Episode gallery - {html.escape(title)}</title>
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
    .old-dot {{ background: #3b82f6; }}
    .new-dot {{ background: #f97316; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(720px, 1fr));
      gap: 22px;
      align-items: start;
    }}
    .card {{
      background: #1c1c1f;
      border: 2px solid #333;
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
    }}
    .card.old50 {{ border-color: #244b91; }}
    .card.new20 {{ border-color: #9a4a11; }}
    .card header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .card h2 {{
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
      grid-template-columns: 1fr 1fr;
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
    <h1>Episode gallery - {html.escape(title)}</h1>
    <p class="hint">Scroll the grid. Each episode has independent top/wrist video controls. Episodes 50-69 are the newer camera distribution.</p>
    <div class="legend">
      <span><span class="dot old-dot"></span>episodes 0-49</span>
      <span><span class="dot new-dot"></span>episodes 50-69</span>
    </div>
  </div>
  <main class="grid">
    {''.join(cards)}
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/episode_gallery.html"))
    parser.add_argument("--repo-id", default=None)
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=None,
        help="If set, cut browser-friendly per-episode clips and reference those instead of chunk videos.",
    )
    args = parser.parse_args()
    clips_dir = args.clips_dir
    if clips_dir is not None and not clips_dir.is_absolute():
        clips_dir = args.output.parent / clips_dir
    build_gallery(args.root, args.output, args.repo_id, clips_dir)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
