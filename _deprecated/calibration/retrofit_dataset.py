"""
retrofit_dataset.py — Remap recorded `.pos` percentages in a LeRobot dataset
from an old calibration to a new one.

Why this exists:
    See plan.md / repo memory. The 53 demos in 002-pour-water were recorded
    under a software calibration that compensated for a hardware misalignment
    (leader/follower shoulder_pan horns clocked ~95° apart). The hardware was
    fixed by rotating the leader's first-motor horn 180°; teleop now works on
    raw LeRobot defaults. Rather than re-record, we remap the existing
    percentages so they correspond to the same physical poses under the new
    calibration.

How:
    Both `action` and `observation.state` are physical commands/observations
    of the FOLLOWER (action is what gets sent to follower.send_action() at
    replay; state is what follower.get_observation() reported at record).
    So both columns are remapped through the FOLLOWER calibration only:
    pct_old → pp_old (via OLD follower range/mode) → enc = pp_old - homing_old
    → pp_new = enc + homing_new → pct_new (via NEW follower range/mode).
    Gripper inversion changes (per follower) are also handled.

    The leader's hardware change (180° horn rotation on shoulder_pan) does
    NOT enter the retrofit, because at replay time the leader is not in the
    loop — the action goes directly to the follower under whatever
    calibration the follower has today.

Outputs:
    - Rewrites action and observation.state in
      `<dataset_root>/data/chunk-*/file-*.parquet`. Atomic .tmp + rename.
    - Recomputes per-episode stats in `<dataset_root>/meta/episodes/...`.
    - Recomputes dataset-wide stats in `<dataset_root>/meta/stats.json`.

Usage:
    # Dry-run first to see deltas / out-of-range counts:
    python calibration/retrofit_dataset.py \
        --dataset-root ~/.cache/huggingface/lerobot/RevanthGundala/002-pour-water \
        --old-cal-dir ~/.cache/huggingface/lerobot/calibration_backup \
        --new-cal-dir ~/.cache/huggingface/lerobot/calibration \
        --dry-run

    # Apply:
    python calibration/retrofit_dataset.py \
        --dataset-root ... --old-cal-dir ... --new-cal-dir ... --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Per-joint normalization mode. Body joints use RANGE_M100_100 (-100..100),
# gripper uses RANGE_0_100 (0..100). See lerobot/robots/omx_follower.py.
JOINT_NORM_MODE = {
    "shoulder_pan": "M100_100",
    "shoulder_lift": "M100_100",
    "elbow_flex": "M100_100",
    "wrist_flex": "M100_100",
    "wrist_roll": "M100_100",
    "gripper": "RANGE_0_100",
}


def _pct_to_pp(pct: np.ndarray, rmin: float, rmax: float, mode: str) -> np.ndarray:
    if mode == "RANGE_0_100":
        return pct / 100.0 * (rmax - rmin) + rmin
    return (pct + 100.0) / 200.0 * (rmax - rmin) + rmin


def _pp_to_pct(pp: np.ndarray, rmin: float, rmax: float, mode: str) -> np.ndarray:
    if mode == "RANGE_0_100":
        return (pp - rmin) / (rmax - rmin) * 100.0
    return (pp - rmin) / (rmax - rmin) * 200.0 - 100.0


@dataclass
class JointCal:
    homing_offset: float
    range_min: float
    range_max: float

    @classmethod
    def from_dict(cls, d: dict) -> "JointCal":
        return cls(
            homing_offset=float(d["homing_offset"]),
            range_min=float(d["range_min"]),
            range_max=float(d["range_max"]),
        )


@dataclass
class ArmCal:
    """Per-arm calibration for the 6 OMX joints, plus gripper-inversion flag."""

    joints: dict[str, JointCal]
    gripper_invert: bool

    @classmethod
    def load(cls, cal_path: Path, inversion_flag: bool) -> "ArmCal":
        with cal_path.open() as f:
            data = json.load(f)
        joints = {name: JointCal.from_dict(data[name]) for name in JOINT_ORDER}
        return cls(joints=joints, gripper_invert=inversion_flag)


def _load_inversion(cal_dir: Path) -> dict[str, bool]:
    """Returns {'leader': bool, 'follower': bool}; False/False if file missing."""
    p = cal_dir / "omx_gripper_inversion.json"
    if not p.exists():
        return {"leader": False, "follower": False}
    with p.open() as f:
        d = json.load(f)
    return {"leader": bool(d.get("leader", False)), "follower": bool(d.get("follower", False))}


def _find_json(subdir: Path) -> Path:
    """Find the single calibration JSON in a per-arm subdirectory.

    LeRobot writes the file as `<config.id>.json`; we accept any *.json file
    so this works whether it's named `omx_leader_arm.json` (custom id) or
    `None.json` (id unset).
    """
    if not subdir.is_dir():
        raise FileNotFoundError(f"{subdir} does not exist")
    candidates = sorted(subdir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"no JSON found in {subdir}")
    if len(candidates) > 1:
        raise RuntimeError(f"multiple JSONs in {subdir}: {candidates}; specify explicitly")
    return candidates[0]


def load_calibrations(old_dir: Path, new_dir: Path) -> tuple[ArmCal, ArmCal, ArmCal, ArmCal]:
    old_inv = _load_inversion(old_dir)
    new_inv = _load_inversion(new_dir)
    old_leader = ArmCal.load(_find_json(old_dir / "teleoperators/omx_leader"), old_inv["leader"])
    old_follower = ArmCal.load(_find_json(old_dir / "robots/omx_follower"), old_inv["follower"])
    new_leader = ArmCal.load(_find_json(new_dir / "teleoperators/omx_leader"), new_inv["leader"])
    new_follower = ArmCal.load(_find_json(new_dir / "robots/omx_follower"), new_inv["follower"])
    return old_leader, old_follower, new_leader, new_follower


def remap_column(
    pct_old: np.ndarray,
    old: ArmCal,
    new: ArmCal,
    is_leader: bool,
    shoulder_pan_delta: float,
) -> tuple[np.ndarray, dict]:
    """Remap a (N, 6) array of percentages from old cal to new cal.

    For the leader (is_leader=True), shoulder_pan gets a ±2048 encoder shift
    to account for the 180° horn rotation. For the follower, no shift.

    Gripper inversion delta: if the old setup software-flipped gripper but the
    new doesn't (or vice versa), we apply a `100 - pct` flip to the recorded
    pct BEFORE recovering the encoder. Per `PatchedOmxLeader.get_action()`
    and `PatchedOmxFollower.get_observation()/send_action()`, the convention
    is: when invert is True, the user-facing pct is `100 - firmware_pct`.

    Returns (pct_new, stats_dict).
    """
    assert pct_old.ndim == 2 and pct_old.shape[1] == 6
    pct_new = np.empty_like(pct_old, dtype=np.float32)
    stats = {}

    for i, joint in enumerate(JOINT_ORDER):
        col = pct_old[:, i].astype(np.float64)
        oc = old.joints[joint]
        nc = new.joints[joint]
        mode = JOINT_NORM_MODE[joint]

        # Step 0 (gripper only): undo old software inversion to get firmware pct
        firmware_pct_old = col.copy()
        if joint == "gripper" and old.gripper_invert:
            firmware_pct_old = 100.0 - firmware_pct_old

        # Step 1: pct -> firmware present_position under old cal (mode-aware)
        pp_old = _pct_to_pp(firmware_pct_old, oc.range_min, oc.range_max, mode)
        # Step 2: present_position -> raw encoder
        enc_record = pp_old - oc.homing_offset

        # Step 3: account for hardware change (only leader shoulder_pan)
        if is_leader and joint == "shoulder_pan":
            enc_now = enc_record + shoulder_pan_delta
        else:
            enc_now = enc_record

        # Step 4: raw encoder -> firmware present_position under new cal
        pp_new = enc_now + nc.homing_offset
        firmware_pct_new = _pp_to_pct(pp_new, nc.range_min, nc.range_max, mode)

        # Step 5 (gripper only): re-apply new software inversion if configured
        final = firmware_pct_new
        if joint == "gripper" and new.gripper_invert:
            final = 100.0 - final

        pct_new[:, i] = final.astype(np.float32)

        lo_bound = 0.0 if mode == "RANGE_0_100" else -100.0
        n_below = int((final < lo_bound).sum())
        n_above = int((final > 100).sum())
        stats[joint] = {
            "old_min": float(col.min()),
            "old_max": float(col.max()),
            "new_min": float(final.min()),
            "new_max": float(final.max()),
            "delta_min": float((final - col).min()),
            "delta_max": float((final - col).max()),
            "n_below_-100": n_below,
            "n_above_+100": n_above,
            "n_total": int(len(col)),
        }

    return pct_new, stats


def _pretty_stats(label: str, stats: dict) -> None:
    print(f"\n  {label}:")
    print(
        f"    {'joint':<14} {'old_min':>10} {'old_max':>10} "
        f"{'new_min':>10} {'new_max':>10} {'<-100':>7} {'>+100':>7}"
    )
    for joint in JOINT_ORDER:
        s = stats[joint]
        flag = ""
        oor = s["n_below_-100"] + s["n_above_+100"]
        if oor > 0:
            pct = 100.0 * oor / s["n_total"]
            flag = f"  ⚠ {oor} ({pct:.1f}% oor)"
        print(
            f"    {joint:<14} {s['old_min']:>10.2f} {s['old_max']:>10.2f} "
            f"{s['new_min']:>10.2f} {s['new_max']:>10.2f} "
            f"{s['n_below_-100']:>7d} {s['n_above_+100']:>7d}{flag}"
        )


def _accumulate(global_stats: dict, file_stats: dict) -> None:
    for joint, s in file_stats.items():
        if joint not in global_stats:
            global_stats[joint] = {
                "old_min": s["old_min"],
                "old_max": s["old_max"],
                "new_min": s["new_min"],
                "new_max": s["new_max"],
                "n_below_-100": 0,
                "n_above_+100": 0,
                "n_total": 0,
            }
        g = global_stats[joint]
        g["old_min"] = min(g["old_min"], s["old_min"])
        g["old_max"] = max(g["old_max"], s["old_max"])
        g["new_min"] = min(g["new_min"], s["new_min"])
        g["new_max"] = max(g["new_max"], s["new_max"])
        g["n_below_-100"] += s["n_below_-100"]
        g["n_above_+100"] += s["n_above_+100"]
        g["n_total"] += s["n_total"]


def process_data_parquet(
    path: Path,
    old_leader: ArmCal,
    old_follower: ArmCal,
    new_leader: ArmCal,
    new_follower: ArmCal,
    shoulder_pan_delta: float,
    apply: bool,
    clamp: bool,
) -> tuple[dict, dict]:
    """Returns (action_stats, state_stats) dicts for this file."""
    table = pq.read_table(path)
    n = table.num_rows

    action_old = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    state_old = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)

    action_new, action_stats = remap_column(
        action_old, old_follower, new_follower, is_leader=False, shoulder_pan_delta=shoulder_pan_delta
    )
    state_new, state_stats = remap_column(
        state_old, old_follower, new_follower, is_leader=False, shoulder_pan_delta=shoulder_pan_delta
    )

    if clamp:
        # Body joints clamp to [-100, 100], gripper to [0, 100].
        for i, joint in enumerate(JOINT_ORDER):
            lo = 0.0 if JOINT_NORM_MODE[joint] == "RANGE_0_100" else -100.0
            action_new[:, i] = np.clip(action_new[:, i], lo, 100.0)
            state_new[:, i] = np.clip(state_new[:, i], lo, 100.0)

    if apply:
        # Rebuild the table with new action and observation.state arrays.
        new_action_arr = pa.FixedSizeListArray.from_arrays(
            pa.array(action_new.reshape(-1), type=pa.float32()), 6
        )
        new_state_arr = pa.FixedSizeListArray.from_arrays(
            pa.array(state_new.reshape(-1), type=pa.float32()), 6
        )
        cols = {}
        for name in table.column_names:
            if name == "action":
                cols[name] = new_action_arr
            elif name == "observation.state":
                cols[name] = new_state_arr
            else:
                cols[name] = table[name]
        new_table = pa.table(cols)
        # Preserve schema metadata (huggingface info) from the original file.
        new_table = new_table.replace_schema_metadata(table.schema.metadata)

        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(new_table, tmp)
        tmp.replace(path)
        if pq.read_table(path).num_rows != n:
            raise RuntimeError(f"rowcount mismatch after rewrite: {path}")

    return action_stats, state_stats


# -----------------------------------------------------------------------------
# Stats recomputation
# -----------------------------------------------------------------------------


def _stats_block(arr: np.ndarray) -> dict:
    """Compute the stats dict block expected by LeRobot v3.0 episode parquet.

    arr: (N, D)  -> returns dict with keys min/max/mean/std/count/q01/q10/q50/q90/q99
    Each value is a length-D list (or [int] for count).
    """
    return {
        "min": arr.min(axis=0).astype(np.float32).tolist(),
        "max": arr.max(axis=0).astype(np.float32).tolist(),
        "mean": arr.mean(axis=0).astype(np.float32).tolist(),
        "std": arr.std(axis=0).astype(np.float32).tolist(),
        "count": [int(arr.shape[0])],
        "q01": np.quantile(arr, 0.01, axis=0).astype(np.float32).tolist(),
        "q10": np.quantile(arr, 0.10, axis=0).astype(np.float32).tolist(),
        "q50": np.quantile(arr, 0.50, axis=0).astype(np.float32).tolist(),
        "q90": np.quantile(arr, 0.90, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).astype(np.float32).tolist(),
    }


def recompute_episode_stats(dataset_root: Path) -> None:
    """For each meta/episodes/chunk-*/file-*.parquet, recompute the
    stats/action/* and stats/observation.state/* columns from the (now
    retrofitted) data parquets. Other stat blocks are preserved verbatim.
    """
    print("\nRecomputing per-episode stats...")
    meta_dir = dataset_root / "meta" / "episodes"
    files = sorted(meta_dir.glob("chunk-*/file-*.parquet"))
    if not files:
        raise RuntimeError(f"no episode meta parquets under {meta_dir}")

    for ep_meta_path in files:
        ep_table = pq.read_table(ep_meta_path)
        # Map (chunk, file) -> data parquet -> per-episode action/state slices.
        # data_from_index/data_to_index are absolute frame indices across the
        # whole dataset. We recompute by loading the referenced data parquet.
        n_eps = ep_table.num_rows
        new_cols = {name: ep_table[name].to_pylist() for name in ep_table.column_names}

        # Cache loaded data parquets to avoid re-reading.
        data_cache: dict[Path, pa.Table] = {}
        for i in range(n_eps):
            chunk = ep_table["data/chunk_index"][i].as_py()
            fidx = ep_table["data/file_index"][i].as_py()
            data_path = dataset_root / "data" / f"chunk-{chunk:03d}" / f"file-{fidx:03d}.parquet"
            if data_path not in data_cache:
                data_cache[data_path] = pq.read_table(data_path)
            data = data_cache[data_path]

            from_idx = ep_table["dataset_from_index"][i].as_py()
            to_idx = ep_table["dataset_to_index"][i].as_py()
            # The data parquet's index column is global; slice rows whose index in [from, to).
            global_idx = np.asarray(data["index"].to_pylist())
            mask = (global_idx >= from_idx) & (global_idx < to_idx)
            action_slice = np.asarray(data["action"].to_pylist(), dtype=np.float32)[mask]
            state_slice = np.asarray(data["observation.state"].to_pylist(), dtype=np.float32)[mask]

            action_stats = _stats_block(action_slice)
            state_stats = _stats_block(state_slice)
            for k, v in action_stats.items():
                col = f"stats/action/{k}"
                new_cols[col][i] = v
            for k, v in state_stats.items():
                col = f"stats/observation.state/{k}"
                new_cols[col][i] = v

        # Rewrite parquet preserving schema (and metadata).
        new_table = pa.table({name: pa.array(new_cols[name], type=ep_table.schema.field(name).type)
                              for name in ep_table.column_names})
        new_table = new_table.replace_schema_metadata(ep_table.schema.metadata)
        tmp = ep_meta_path.with_suffix(ep_meta_path.suffix + ".tmp")
        pq.write_table(new_table, tmp)
        tmp.replace(ep_meta_path)
        print(f"  rewrote {ep_meta_path.name} ({n_eps} episodes)")


def recompute_dataset_stats(dataset_root: Path) -> None:
    """Update meta/stats.json action and observation.state blocks from the
    retrofitted data. Preserve every other block verbatim.
    """
    print("\nRecomputing dataset-wide meta/stats.json (action + state blocks)...")
    stats_path = dataset_root / "meta" / "stats.json"
    if not stats_path.exists():
        print(f"  (no stats.json at {stats_path}; skipping)")
        return
    with stats_path.open() as f:
        stats = json.load(f)

    actions = []
    states = []
    for p in sorted((dataset_root / "data").glob("chunk-*/file-*.parquet")):
        t = pq.read_table(p)
        actions.append(np.asarray(t["action"].to_pylist(), dtype=np.float32))
        states.append(np.asarray(t["observation.state"].to_pylist(), dtype=np.float32))
    actions = np.concatenate(actions, axis=0)
    states = np.concatenate(states, axis=0)

    stats["action"] = _stats_block(actions)
    stats["observation.state"] = _stats_block(states)

    with stats_path.open("w") as f:
        json.dump(stats, f, indent=4)
    print(f"  rewrote {stats_path}  (action N={actions.shape[0]}, state N={states.shape[0]})")


# -----------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Remap LeRobot dataset .pos values from old cal to new cal.")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--old-cal-dir", type=Path, required=True)
    p.add_argument("--new-cal-dir", type=Path, required=True)
    p.add_argument("--shoulder-pan-delta", type=float, default=2048.0,
                   help="Encoder-tick shift for leader shoulder_pan due to 180° horn rotation. ±2048.")
    p.add_argument("--clamp", action="store_true",
                   help="Clamp retrofitted percentages to [-100, 100].")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="Compute and print stats; do not write files.")
    g.add_argument("--apply", action="store_true",
                   help="Rewrite data parquets in place (atomic .tmp + rename) and recompute stats.")
    args = p.parse_args()

    print(f"Dataset root: {args.dataset_root}")
    print(f"Old cal dir:  {args.old_cal_dir}")
    print(f"New cal dir:  {args.new_cal_dir}")
    print(f"Shoulder-pan delta: {args.shoulder_pan_delta:+.0f} ticks (action only)")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}{'  (CLAMP)' if args.clamp else ''}")

    old_leader, old_follower, new_leader, new_follower = load_calibrations(
        args.old_cal_dir, args.new_cal_dir
    )
    print(f"Old gripper invert: leader={old_leader.gripper_invert} follower={old_follower.gripper_invert}")
    print(f"New gripper invert: leader={new_leader.gripper_invert} follower={new_follower.gripper_invert}")

    data_files = sorted((args.dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        sys.exit(f"no data parquets found under {args.dataset_root}/data/")

    global_action_stats: dict = {}
    global_state_stats: dict = {}

    for path in data_files:
        print(f"\nProcessing {path.relative_to(args.dataset_root)} ...")
        action_stats, state_stats = process_data_parquet(
            path, old_leader, old_follower, new_leader, new_follower,
            args.shoulder_pan_delta, apply=args.apply, clamp=args.clamp,
        )
        _accumulate(global_action_stats, action_stats)
        _accumulate(global_state_stats, state_stats)

    print("\n=== GLOBAL STATS (all data parquets) ===")
    _pretty_stats("ACTION (leader-derived)", global_action_stats)
    _pretty_stats("OBSERVATION.STATE (follower)", global_state_stats)

    if args.apply:
        recompute_episode_stats(args.dataset_root)
        recompute_dataset_stats(args.dataset_root)
        print("\n✓ Retrofit complete.")
    else:
        print("\n(dry-run; no files written. Re-run with --apply to commit.)")


if __name__ == "__main__":
    main()
