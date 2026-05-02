"""Capture (leader_action, follower_observation) snapshots at user-chosen poses.

Workflow:
  1. Power on leader+follower, run this script.
  2. Move the leader by hand to a recognizable pose (e.g. "home", "above-cup",
     "pour-tilt"), label it, press Enter. Repeat.
  3. Press 'q' + Enter to stop and dump everything.

Each snapshot stores both pcts (LeRobot percentage) AND raw firmware
Present_Position so we can sanity-check the calibration math directly.

Output is written to diagnostics/reference_poses.json — diff that against
the retrofitted dataset frames at visually-similar poses to find any
remaining offset/sign error.
"""
import json
import time
from pathlib import Path

from utils.robot_utils import create_follower, create_leader, safe_disconnect


OUT_PATH = Path(__file__).parent / "reference_poses.json"


def _read_raw_pp(bus) -> dict:
    """Raw firmware Present_Position per joint (encoder + EEPROM homing_offset)."""
    return {name: int(v) for name, v in bus.sync_read("Present_Position", normalize=False).items()}


def main() -> None:
    print("Connecting leader + follower (calibrate=False, raw)...")
    leader = create_leader()
    follower = create_follower(camera=False)
    leader.connect(calibrate=False)
    follower.connect(calibrate=False)

    snapshots = []
    try:
        while True:
            label = input("\nLabel for this pose (or 'q' to quit): ").strip()
            if label.lower() in ("q", "quit", "exit"):
                break
            if not label:
                continue

            time.sleep(0.2)
            leader_action = leader.get_action()
            follower_obs = follower.get_observation()
            leader_pp = _read_raw_pp(leader.bus)
            follower_pp = _read_raw_pp(follower.bus)

            snap = {
                "label": label,
                "timestamp": time.time(),
                "leader_action_pct": {k: float(v) for k, v in leader_action.items()},
                "follower_obs_pct": {
                    k: float(v) for k, v in follower_obs.items() if k.endswith(".pos")
                },
                "leader_pp_raw": leader_pp,
                "follower_pp_raw": follower_pp,
            }
            snapshots.append(snap)

            print(f"  captured '{label}':")
            print("  joint            leader_pct   follower_pct   leader_pp   follower_pp")
            for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"):
                la = snap["leader_action_pct"].get(f"{j}.pos", float("nan"))
                fo = snap["follower_obs_pct"].get(f"{j}.pos", float("nan"))
                lp = snap["leader_pp_raw"].get(j, "—")
                fp = snap["follower_pp_raw"].get(j, "—")
                print(f"  {j:<15} {la:>10.2f}    {fo:>10.2f}     {lp:>8}    {fp:>8}")

    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        OUT_PATH.write_text(json.dumps(snapshots, indent=2))
        print(f"\nWrote {len(snapshots)} snapshots to {OUT_PATH}")
        safe_disconnect(leader)
        safe_disconnect(follower)


if __name__ == "__main__":
    main()
