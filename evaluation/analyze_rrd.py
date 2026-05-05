#!/usr/bin/env python3
"""Headless analysis for OMX Rerun eval recordings.

Extracts joint Scalar streams from .rrd files and reports action ranges,
tracking lag, and jerk candidates without launching rerun-viewer.
"""

from __future__ import annotations

import argparse
import glob
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rerun import dataframe as rrd

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
KINDS = {
    "state": "state",
    "policy_action": "policy",
    "sent_action": "sent",
}


@dataclass
class EpisodeSummary:
    recording: str
    episode: int
    rows: int
    start_step: float | int | None
    end_step: float | int | None
    start_time: float | None
    end_time: float | None
    top_jerky_joint: str | None
    max_lag: float
    jerk_events: int


@dataclass
class JointEpisodeStats:
    recording: str
    episode: int
    joint: str
    sent_min: float
    sent_max: float
    sent_mean: float
    sent_std: float
    policy_min: float
    policy_max: float
    lag_max_abs: float
    lag_mean_abs: float
    policy_div_max_abs: float
    jerk_sigma: float
    jerk_events: int
    sustained_lag_runs: int


def expand_paths(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def scalarize(value):
    """Convert Rerun scalar component cells (often [x]) to float."""
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return np.nan
        return scalarize(value[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def time_to_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        epoch = pd.Timestamp("1970-01-01", tz=series.dt.tz)
        return (series - epoch).dt.total_seconds()
    return pd.to_numeric(series.map(scalarize), errors="coerce")


def find_component_columns(schema) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for col in schema.component_columns():
        entity = str(getattr(col, "entity_path", ""))
        component = str(getattr(col, "component", ""))
        if component != "Scalars:scalars":
            continue
        parts = entity.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "joints":
            continue
        joint, kind = parts[1], parts[2]
        if joint in JOINTS and kind in KINDS:
            mapping[f"{KINDS[kind]}_{joint}"] = f"{entity}:Scalars:scalars"
    return mapping


def read_joint_rows(path: Path) -> tuple[pd.DataFrame, list[str]]:
    rec = rrd.load_recording(str(path))
    schema = rec.schema()
    component_columns = find_component_columns(schema)
    notes: list[str] = []
    if not component_columns:
        return pd.DataFrame(), ["no joint Scalar columns found"]

    index_names = {str(c).split("timeline:")[-1].rstrip(")") for c in schema.index_columns()}
    index_name = "log_tick" if "log_tick" in index_names else "step"
    try:
        raw = rec.view(index=index_name, contents="joints/**").select().read_pandas()
    except Exception as exc:  # noqa: BLE001 - diagnostics should survive bad recordings
        return pd.DataFrame(), [f"failed reading dataframe view: {type(exc).__name__}: {exc}"]
    if raw.empty:
        return pd.DataFrame(), ["dataframe view is empty"]

    out = pd.DataFrame()
    out["_order"] = np.arange(len(raw), dtype=np.int64)
    if "log_tick" in raw:
        out["log_tick"] = pd.to_numeric(raw["log_tick"], errors="coerce")
    if "log_time" in raw:
        out["log_time"] = raw["log_time"]
    if "step" in raw:
        out["step"] = pd.to_numeric(raw["step"].map(scalarize), errors="coerce")
    else:
        notes.append("missing step timeline; using row order as step")
        out["step"] = np.arange(len(raw), dtype=np.int64)
    if "time" in raw:
        out["time_s"] = time_to_seconds(raw["time"])
    else:
        notes.append("missing time timeline")
        out["time_s"] = np.nan

    episode_col = next((c for c in raw.columns if str(c).endswith("episode_index:Scalars:scalars") or str(c) == "episode_index"), None)
    if episode_col is not None:
        out["episode_index"] = pd.to_numeric(raw[episode_col].map(scalarize), errors="coerce")

    present: list[str] = []
    for short_name, raw_name in component_columns.items():
        if raw_name in raw:
            out[short_name] = pd.to_numeric(raw[raw_name].map(scalarize), errors="coerce")
            present.append(short_name)
        else:
            notes.append(f"schema column not present in view: {raw_name}")

    if not present:
        return pd.DataFrame(), notes + ["no joint data columns present after read"]

    # Rerun's log_tick/log_time view is sparse because each rr.log call creates a row.
    # Collapse all rows that share the same custom step/time into one control-loop frame.
    group_cols = ["step"]
    if out["time_s"].notna().any():
        # Keep time in the key so repeated step values from reset episodes do not merge.
        out["_time_key"] = out["time_s"].round(9)
        group_cols.append("_time_key")
    if "episode_index" in out and out["episode_index"].notna().any():
        group_cols.append("episode_index")

    def first_valid(s: pd.Series):
        valid = s.dropna()
        return valid.iloc[0] if len(valid) else np.nan

    grouped = out.groupby(group_cols, dropna=False, sort=False).agg(first_valid).reset_index()
    grouped = grouped.sort_values("_order").reset_index(drop=True)
    if "_time_key" in grouped:
        grouped = grouped.drop(columns=["_time_key"])
    if "step" in grouped:
        grouped["step"] = grouped["step"].astype("Int64")
    grouped = grouped.set_index("step", drop=False)
    return grouped, notes


def true_runs(mask: pd.Series, min_len: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    vals = mask.fillna(False).to_numpy(dtype=bool)
    for i, value in enumerate(vals):
        if value and start is None:
            start = i
        if (not value or i == len(vals) - 1) and start is not None:
            end = i if value and i == len(vals) - 1 else i - 1
            if end - start + 1 >= min_len:
                runs.append((start, end))
            start = None
    return runs


def detect_episodes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values("_order").copy()
    boundary_reasons = ["start"]
    episode_ids = [0]
    current = 0
    prev = df.iloc[0]
    for _, row in df.iloc[1:].iterrows():
        reasons: list[str] = []
        step = row.get("step")
        prev_step = prev.get("step")
        time_s = row.get("time_s")
        prev_time_s = prev.get("time_s")
        if pd.notna(step) and pd.notna(prev_step) and step <= prev_step:
            reasons.append("step reset/non-monotonic")
        if pd.notna(time_s) and pd.notna(prev_time_s) and (time_s - prev_time_s) > 2.0:
            reasons.append(f"time gap {time_s - prev_time_s:.2f}s")
        if "episode_index" in df and pd.notna(row.get("episode_index")) and pd.notna(prev.get("episode_index")):
            if row.get("episode_index") != prev.get("episode_index"):
                reasons.append("episode_index changed")
        if reasons:
            current += 1
            boundary_reasons.append(", ".join(reasons))
        episode_ids.append(current)
        prev = row
    df["episode"] = episode_ids
    return df, boundary_reasons


def fmt_num(value: float | int | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def analyze_recording(path: Path, save_csv: bool, output_dir: Path) -> tuple[list[EpisodeSummary], list[JointEpisodeStats], list[str]]:
    print(f"\n=== {path} ===")
    df, notes = read_joint_rows(path)
    for note in notes:
        print(f"note: {note}")
    if df.empty:
        print("No analyzable joint frames.")
        return [], [], notes

    df, boundary_reasons = detect_episodes(df)
    print(f"frames: {len(df)} | episodes detected: {df['episode'].nunique()}")
    print("Episode boundaries:")
    for ep, group in df.groupby("episode", sort=True):
        reason = boundary_reasons[int(ep)] if int(ep) < len(boundary_reasons) else "unknown"
        print(
            f"  ep {int(ep):02d}: rows={len(group):4d} "
            f"steps={fmt_num(group['step'].iloc[0], 0)}..{fmt_num(group['step'].iloc[-1], 0)} "
            f"time={fmt_num(group['time_s'].iloc[0])}..{fmt_num(group['time_s'].iloc[-1])} "
            f"reason={reason}"
        )

    if save_csv:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{path.stem}.csv"
        csv_cols = [c for c in df.columns if not c.startswith("_") and c != "log_time"]
        df[csv_cols].to_csv(csv_path, index=False)
        print(f"saved CSV: {csv_path}")

    ep_summaries: list[EpisodeSummary] = []
    joint_stats: list[JointEpisodeStats] = []

    for ep, group in df.groupby("episode", sort=True):
        print(f"\nEpisode {int(ep):02d} joint stats:")
        print(
            "  joint            sent[min,max,mean,std]      "
            "lag|max mean|   policy-state|max|  jerkσ events sustainedLag"
        )
        ep_jerk_counts: dict[str, int] = {}
        ep_max_lag = 0.0
        jerk_rows: list[tuple[float, str, float, float, float]] = []

        for joint in JOINTS:
            sent_col = f"sent_{joint}"
            state_col = f"state_{joint}"
            policy_col = f"policy_{joint}"
            if sent_col not in group or group[sent_col].notna().sum() < 2:
                continue
            sent = group[sent_col]
            state = group[state_col] if state_col in group else pd.Series(np.nan, index=group.index)
            policy = group[policy_col] if policy_col in group else pd.Series(np.nan, index=group.index)
            lag = sent - state
            policy_div = policy - state
            jerk = sent.diff()
            jerk_sigma = float(jerk.std(skipna=True) or 0.0)
            threshold = 3.0 * jerk_sigma
            jerk_mask = jerk.abs() > threshold if threshold > 0 else pd.Series(False, index=group.index)
            jerk_events = int(jerk_mask.sum())
            sustained_runs = true_runs(lag.abs() > 5.0, min_len=10)
            ep_jerk_counts[joint] = jerk_events
            lag_max_abs = float(lag.abs().max(skipna=True)) if lag.notna().any() else float("nan")
            if not math.isnan(lag_max_abs):
                ep_max_lag = max(ep_max_lag, lag_max_abs)
            for pos in np.flatnonzero(jerk_mask.to_numpy(dtype=bool)):
                row = group.iloc[pos]
                jerk_rows.append((abs(float(jerk.iloc[pos])), joint, float(row.get("time_s", np.nan)), float(row.get("step", np.nan)), float(jerk.iloc[pos])))

            sent_std = float(sent.std(skipna=True) or 0.0)
            policy_min = float(policy.min(skipna=True)) if policy.notna().any() else float("nan")
            policy_max = float(policy.max(skipna=True)) if policy.notna().any() else float("nan")
            policy_div_max = float(policy_div.abs().max(skipna=True)) if policy_div.notna().any() else float("nan")
            lag_mean_abs = float(lag.abs().mean(skipna=True)) if lag.notna().any() else float("nan")
            stat = JointEpisodeStats(
                recording=path.name,
                episode=int(ep),
                joint=joint,
                sent_min=float(sent.min(skipna=True)),
                sent_max=float(sent.max(skipna=True)),
                sent_mean=float(sent.mean(skipna=True)),
                sent_std=sent_std,
                policy_min=policy_min,
                policy_max=policy_max,
                lag_max_abs=lag_max_abs,
                lag_mean_abs=lag_mean_abs,
                policy_div_max_abs=policy_div_max,
                jerk_sigma=jerk_sigma,
                jerk_events=jerk_events,
                sustained_lag_runs=len(sustained_runs),
            )
            joint_stats.append(stat)
            print(
                f"  {joint:<15} "
                f"[{fmt_num(stat.sent_min)},{fmt_num(stat.sent_max)},{fmt_num(stat.sent_mean)},{fmt_num(stat.sent_std)}] "
                f"{fmt_num(stat.lag_max_abs):>6} {fmt_num(stat.lag_mean_abs):>6} "
                f"{fmt_num(stat.policy_div_max_abs):>10} "
                f"{fmt_num(stat.jerk_sigma):>7} {stat.jerk_events:6d} {len(sustained_runs):12d}"
            )
            for start, end in sustained_runs[:5]:
                start_row = group.iloc[start]
                end_row = group.iloc[end]
                print(
                    f"    sustained lag >5: {joint} frames={end - start + 1} "
                    f"steps={fmt_num(start_row.get('step'), 0)}..{fmt_num(end_row.get('step'), 0)} "
                    f"time={fmt_num(start_row.get('time_s'))}..{fmt_num(end_row.get('time_s'))}"
                )

        top_joint = max(ep_jerk_counts, key=ep_jerk_counts.get) if ep_jerk_counts else None
        total_jerks = int(sum(ep_jerk_counts.values()))
        ep_summaries.append(
            EpisodeSummary(
                recording=path.name,
                episode=int(ep),
                rows=len(group),
                start_step=group["step"].iloc[0] if "step" in group else None,
                end_step=group["step"].iloc[-1] if "step" in group else None,
                start_time=float(group["time_s"].iloc[0]) if "time_s" in group and pd.notna(group["time_s"].iloc[0]) else None,
                end_time=float(group["time_s"].iloc[-1]) if "time_s" in group and pd.notna(group["time_s"].iloc[-1]) else None,
                top_jerky_joint=top_joint,
                max_lag=ep_max_lag,
                jerk_events=total_jerks,
            )
        )
        if jerk_rows:
            print("  Top jerk events:")
            for mag, joint, time_s, step, signed in sorted(jerk_rows, reverse=True)[:10]:
                print(f"    {joint:<15} step={fmt_num(step, 0):>5} time={fmt_num(time_s):>7}s Δsent={signed:+.2f}")
        else:
            print("  Top jerk events: none over 3σ")

    return ep_summaries, joint_stats, notes


def print_summary(ep_summaries: list[EpisodeSummary], joint_stats: list[JointEpisodeStats]) -> None:
    print("\n=== Summary table ===")
    if not ep_summaries:
        print("No episodes with analyzable joint data.")
        return
    print("recording                                           ep rows steps        top_jerky_joint max_lag jerk_events")
    for s in ep_summaries:
        print(
            f"{s.recording[:50]:<50} {s.episode:2d} {s.rows:4d} "
            f"{fmt_num(s.start_step, 0):>4}..{fmt_num(s.end_step, 0):<4} "
            f"{(s.top_jerky_joint or 'n/a'):<15} {fmt_num(s.max_lag):>7} {s.jerk_events:11d}"
        )
    if not joint_stats:
        return
    stats_df = pd.DataFrame([stat.__dict__ for stat in joint_stats])
    print("\nAcross episodes by joint:")
    agg = stats_df.groupby("joint").agg(
        episodes=("episode", "count"),
        jerk_events=("jerk_events", "sum"),
        episodes_with_jerk=("jerk_events", lambda s: int((s > 0).sum())),
        max_lag=("lag_max_abs", "max"),
        mean_lag=("lag_mean_abs", "mean"),
        policy_min=("policy_min", "min"),
        policy_max=("policy_max", "max"),
        sustained_lag_runs=("sustained_lag_runs", "sum"),
    ).reset_index()
    agg = agg.sort_values(["jerk_events", "max_lag"], ascending=False)
    print("joint            eps jerk_events eps_w_jerk max_lag mean_lag policy_range       sustainedLag")
    for _, row in agg.iterrows():
        print(
            f"{row['joint']:<15} {int(row['episodes']):3d} {int(row['jerk_events']):11d} "
            f"{int(row['episodes_with_jerk']):10d} {fmt_num(row['max_lag']):>7} {fmt_num(row['mean_lag']):>8} "
            f"[{fmt_num(row['policy_min'])},{fmt_num(row['policy_max'])}] {int(row['sustained_lag_runs']):12d}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rrd", nargs="+", help=".rrd path(s) or glob(s)")
    parser.add_argument("--no-csv", action="store_true", help="do not save extracted per-frame CSV files")
    parser.add_argument("--output-dir", default="outputs/analysis", help="directory for per-recording CSV files")
    args = parser.parse_args()

    paths = expand_paths(args.rrd)
    if not paths:
        parser.error("no input paths")
    all_ep_summaries: list[EpisodeSummary] = []
    all_joint_stats: list[JointEpisodeStats] = []
    for path in paths:
        if not path.exists():
            print(f"\n=== {path} ===\nnote: file does not exist")
            continue
        ep_summaries, joint_stats, _ = analyze_recording(path, save_csv=not args.no_csv, output_dir=Path(args.output_dir))
        all_ep_summaries.extend(ep_summaries)
        all_joint_stats.extend(joint_stats)
    print_summary(all_ep_summaries, all_joint_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
