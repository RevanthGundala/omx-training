"""Probe follower: read Operating_Mode + send a small relative move and check
whether the motor responds the way we expect.
"""
from __future__ import annotations

import time

from utils.robot_utils import create_follower, safe_disconnect


def main():
    follower = create_follower(camera=False)
    follower.connect(calibrate=False)
    try:
        op_mode = follower.bus.sync_read("Operating_Mode", normalize=False)
        drive = follower.bus.sync_read("Drive_Mode", normalize=False)
        torque = follower.bus.sync_read("Torque_Enable", normalize=False)
        present = follower.bus.sync_read("Present_Position", normalize=False)
        print(f"{'joint':14s} {'op_mode':>8} {'drive':>6} {'torque':>7} {'present':>8}")
        for j in ["shoulder_pan", "shoulder_lift", "elbow_flex",
                  "wrist_flex", "wrist_roll", "gripper"]:
            print(f"{j:14s} {op_mode[j]:>8d} {drive[j]:>6d} {torque[j]:>7d} {present[j]:>8d}")
        print("(op_mode: 3=Position, 4=Extended_Position, 5=Current_Position)")

        # Probe: send shoulder_pan 5° in each direction by writing Goal_Position
        # directly (skip normalization) so we see exactly what the motor does.
        sp_now = present["shoulder_pan"]
        target_a = sp_now + 200  # +200 ticks ≈ 17° (but small enough to be safe)
        target_b = sp_now - 200
        print(f"\nProbe: shoulder_pan currently at {sp_now}. Will pulse +200 → -200 → return.")
        print("Press Enter to continue (or Ctrl+C to abort)...")
        input()

        for label, tgt in [("+200", target_a), ("-200 from start", sp_now - 200), ("home", sp_now)]:
            follower.bus.write("Goal_Position", "shoulder_pan", tgt)
            time.sleep(1.5)
            actual = follower.bus.sync_read("Present_Position", normalize=False)["shoulder_pan"]
            err = actual - tgt
            print(f"  cmd {label:>15} target={tgt:>6d}  actual={actual:>6d}  err={err:+5d}")
    finally:
        safe_disconnect(follower)


if __name__ == "__main__":
    main()
