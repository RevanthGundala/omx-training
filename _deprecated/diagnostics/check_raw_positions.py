"""
check_raw_positions.py — Read raw encoder positions with homing offsets zeroed.

Pose both arms identically by hand, then run this script. It will:
1. Temporarily zero all homing offsets (EEPROM only, JSON untouched)
2. Read raw Present_Position from both arms
3. Show a comparison table
4. Restore original homing offsets to EEPROM

If any joint differs by ~2048 ticks (180°), that joint's horn is mounted
180° off between leader and follower.

Usage:
    uv run python diagnostics/check_raw_positions.py
"""

from utils.robot_utils import create_follower, create_leader, safe_disconnect


BODY_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def read_raw(bus, label: str) -> dict:
    """Zero homing offsets, read raw positions, return {motor: (raw_pos, original_offset)}."""
    bus.disable_torque()

    # Save current homing offsets
    original_offsets = {}
    for motor in BODY_JOINTS:
        original_offsets[motor] = bus.read("Homing_Offset", motor)

    # Zero homing offsets temporarily
    for motor in BODY_JOINTS:
        bus.write("Homing_Offset", motor, 0)

    # Read raw positions (no homing offset applied)
    raw_positions = bus.sync_read("Present_Position", BODY_JOINTS, normalize=False)

    # Restore original homing offsets
    for motor in BODY_JOINTS:
        bus.write("Homing_Offset", motor, original_offsets[motor])

    print(f"\n  {label} raw positions (homing_offset=0):")
    for motor in BODY_JOINTS:
        print(f"    {motor:<18} raw={raw_positions[motor]:>5}  (saved offset={original_offsets[motor]:>+6})")

    return {motor: raw_positions[motor] for motor in BODY_JOINTS}


def main():
    input(
        "Pose BOTH arms into the exact same physical position,\n"
        "then press ENTER to read raw positions..."
    )

    leader = create_leader()
    follower = create_follower(camera=False)

    leader.connect()
    follower.connect()

    try:
        leader_raw = read_raw(leader.bus, "Leader")
        follower_raw = read_raw(follower.bus, "Follower")

        print(f"\n  {'='*65}")
        print(f"  Comparison (both arms should be in the same physical pose):")
        print(f"  {'Joint':<18} {'Leader':>8} {'Follower':>8} {'Diff':>8} {'Note'}")
        print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*20}")

        for motor in BODY_JOINTS:
            l = leader_raw[motor]
            f = follower_raw[motor]
            diff = abs(l - f)
            # Check if diff is close to 2048 (half revolution = 180°)
            if 1800 < diff < 2300:
                note = "⚠️  ~180° OFF — reassemble horn!"
            elif diff > 500:
                note = "⚠️  Large difference"
            else:
                note = "✓ OK"
            print(f"  {motor:<18} {l:>8} {f:>8} {diff:>8} {note}")

        print(f"\n  Note: Leader uses XL330, Follower uses XL430/XL330.")
        print(f"  Full revolution = 4096 ticks. Half = 2048 ticks.")
        print(f"  Differences < 500 are normal (assembly tolerance).")
        print(f"  Differences ~2048 mean the horn is mounted 180° off.")

    finally:
        safe_disconnect(leader)
        safe_disconnect(follower)


if __name__ == "__main__":
    main()
