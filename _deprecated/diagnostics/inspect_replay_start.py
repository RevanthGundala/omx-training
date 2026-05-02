"""
Diagnostic for the failing replay.

Reads:
  - Live follower state (raw encoder ticks via Present_Position, AND normalized .pos)
  - Retrofitted dataset start state (current ~/.cache copy of 002-pour-water)
  - Pre-retrofit dataset start state (backup folder)
  - Active follower calibration (homing_offset / range_min / range_max / drive_mode)

Prints them side-by-side per joint so we can see which joint(s) the retrofit
math broke.

NO motion is sent. Read-only.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from utils.robot_utils import create_follower, safe_disconnect

CACHE = Path.home() / ".cache" / "huggingface" / "lerobot"
CURRENT_DS = CACHE / "RevanthGundala" / "002-pour-water"
BACKUPS = sorted(glob.glob(str(CACHE / "RevanthGundala" / "002-pour-water_pre_retrofit_*")))
CAL_PATH = CACHE / "calibration" / "robots" / "omx_follower" / "omx_follower_arm.json"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _read_first_row(parquet_path: Path):
    table = pq.read_table(parquet_path, columns=["episode_index", "action", "observation.state"])
    df = table.to_pandas()
    row0 = df[df["episode_index"] == 0].iloc[0]
    return list(row0["action"]), list(row0["observation.state"])


def _pct_to_enc(pct: float, cal: dict) -> int:
    rng_min = cal["range_min"]
    rng_max = cal["range_max"]
    homing = cal["homing_offset"]
    if cal.get("drive_mode", 0) == 1:
        pct = -pct
    half = (rng_max - rng_min) / 2.0
    mid = (rng_max + rng_min) / 2.0
    if cal.get("norm_mode", "RANGE_M100_100") == "RANGE_0_100":
        pp = rng_min + (pct / 100.0) * (rng_max - rng_min)
    else:
        pp = mid + (pct / 100.0) * half
    return int(round(pp + homing))


def main():
    if not BACKUPS:
        print("No pre-retrofit backup found.")
        return
    backup = Path(BACKUPS[-1])
    print(f"Using backup: {backup.name}\n")

    cur_action, cur_state = _read_first_row(CURRENT_DS / "data" / "chunk-000" / "file-000.parquet")
    bak_action, bak_state = _read_first_row(backup / "data" / "chunk-000" / "file-000.parquet")

    cal = json.loads(CAL_PATH.read_text())
    print(f"Active calibration: {CAL_PATH}")
    for j in JOINT_NAMES:
        c = cal.get(j, {})
        print(f"  {j:14s} homing={c.get('homing_offset'):>6} "
              f"range=[{c.get('range_min'):>6}, {c.get('range_max'):>6}] "
              f"drive_mode={c.get('drive_mode')}")
    print()

    # Live read
    follower = create_follower(camera=False)
    print("Connecting follower (calibrate=False)...")
    follower.connect(calibrate=False)
    try:
        live_norm = follower.get_observation()
        live_raw = follower.bus.sync_read("Present_Position", normalize=False)
    finally:
        safe_disconnect(follower)

    print()
    header = f"{'joint':14s} {'live_raw':>10} {'live_pct':>10} {'cur_state':>10} {'cur_act':>10} {'bak_state':>10} {'bak_act':>10}"
    print(header)
    print("-" * len(header))
    for i, j in enumerate(JOINT_NAMES):
        live_pct = float(live_norm.get(f"{j}.pos", float("nan")))
        live_r = int(live_raw.get(j, -1))
        cs = float(cur_state[i])
        ca = float(cur_action[i])
        bs = float(bak_state[i])
        ba = float(bak_action[i])
        print(f"{j:14s} {live_r:>10d} {live_pct:>10.2f} {cs:>10.2f} {ca:>10.2f} {bs:>10.2f} {ba:>10.2f}")

    print()
    print("Where the move-to-start would push each joint (pct delta):")
    for i, j in enumerate(JOINT_NAMES):
        live_pct = float(live_norm.get(f"{j}.pos", float("nan")))
        target = float(cur_state[i])
        delta = target - live_pct
        flag = "  <-- LARGE" if abs(delta) > 30 else ""
        print(f"  {j:14s} live={live_pct:7.2f}  target={target:7.2f}  delta={delta:+7.2f}{flag}")

    print()
    print("If `bak_state` represents the original physical pose at episode 0 start,")
    print("the live reading (in raw ticks) should match `bak_state` re-encoded under")
    print("the OLD calibration, OR `cur_state` re-encoded under the NEW calibration.")


if __name__ == "__main__":
    main()
