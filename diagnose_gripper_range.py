"""
diagnose_gripper_range.py — Read-only: print raw encoder min/max for each
gripper as you move it by hand. No motion is commanded, no calibration is
written. Use this to see how much of the 0..4095 encoder span each gripper
actually uses mechanically.

Procedure:
    1. Run the script.
    2. With torque off, move the LEADER gripper jaw all the way open, then all
       the way closed, several times. The min/max on screen will settle.
    3. Press ENTER to move on to the FOLLOWER gripper. Repeat.
    4. Press ENTER again to exit.

The printed numbers are *raw* encoder values (no homing offset, no normalization).
"""

import time

from robot_utils import create_follower, create_leader, safe_disconnect


def sweep(name: str, bus, motor: str = "gripper") -> tuple[int, int]:
    bus.disable_torque()
    print(f"\nMove the {name} {motor} fully open, then fully closed, several times.")
    print("Live raw encoder reading below. Press ENTER when min/max have stopped changing.")

    pos = bus.sync_read("Present_Position", [motor], normalize=False)[motor]
    raw_min = pos
    raw_max = pos

    import sys, select
    while True:
        pos = bus.sync_read("Present_Position", [motor], normalize=False)[motor]
        raw_min = min(raw_min, pos)
        raw_max = max(raw_max, pos)
        print(f"  raw={pos:>6}    min={raw_min:>6}    max={raw_max:>6}    span={raw_max - raw_min:>5}", end="\r")
        if select.select([sys.stdin], [], [], 0.05)[0]:
            sys.stdin.readline()
            break

    print()
    return raw_min, raw_max


def main() -> None:
    leader = create_leader()
    follower = create_follower(camera=False)

    leader.connect()
    follower.connect()
    try:
        l_min, l_max = sweep("LEADER", leader.bus)
        f_min, f_max = sweep("FOLLOWER", follower.bus)
    finally:
        safe_disconnect(leader)
        safe_disconnect(follower)

    print("\n=== Results ===")
    print(f"  LEADER   gripper raw range: [{l_min}, {l_max}]   span: {l_max - l_min}")
    print(f"  FOLLOWER gripper raw range: [{f_min}, {f_max}]   span: {f_max - f_min}")
    print(f"\nFor reference, the encoder resolution is 4096 ticks (full shaft revolution).")
    print("If the spans above are ~4000, my range hypothesis is wrong.")
    print("If the spans are much smaller (e.g. <1500), the gripper only uses a fraction")
    print("of the 0..4095 range and the calibration JSONs need range_min/range_max set")
    print("to these measured values for leader<->follower mapping to be consistent.")


if __name__ == "__main__":
    main()
