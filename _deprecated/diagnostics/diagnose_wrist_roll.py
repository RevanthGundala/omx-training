"""Quick diagnostic: read wrist_roll Present_Position and show where it falls in the calibration range."""

import json
from pathlib import Path

from utils.robot_utils import create_leader, create_follower, safe_disconnect

LEADER_CAL = Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/omx_leader/omx_leader_arm.json"
FOLLOWER_CAL = Path.home() / ".cache/huggingface/lerobot/calibration/robots/omx_follower/omx_follower_arm.json"


def read_wrist_roll_raw(bus):
    return int(bus.sync_read("Present_Position", ["wrist_roll"], normalize=False)["wrist_roll"])


def main():
    leader = create_leader()
    follower = create_follower(camera=False)

    print("Connecting leader...")
    leader.connect()
    print("Connecting follower...")
    follower.connect()

    l_raw = read_wrist_roll_raw(leader.bus)
    f_raw = read_wrist_roll_raw(follower.bus)

    l_cal = json.loads(LEADER_CAL.read_text())["wrist_roll"]
    f_cal = json.loads(FOLLOWER_CAL.read_text())["wrist_roll"]

    l_homing = l_cal["homing_offset"]
    f_homing = f_cal["homing_offset"]
    l_range = (l_cal["range_min"], l_cal["range_max"])
    f_range = (f_cal["range_min"], f_cal["range_max"])

    # Present_Position in Extended Position mode includes homing_offset
    # Raw encoder (absolute) = Present_Position - Homing_Offset
    l_abs = l_raw - l_homing
    f_abs = f_raw - f_homing

    def normalize(val, rmin, rmax):
        bounded = min(rmax, max(rmin, val))
        return ((bounded - rmin) / (rmax - rmin)) * 200 - 100

    l_norm = normalize(l_raw, *l_range)
    f_norm = normalize(f_raw, *f_range)

    # Ideal homing_offset: centers rest position at (range_min + range_max) / 2
    # Present_Position = actual + Homing_Offset, want Present_Position = center
    range_center = (l_range[0] + l_range[1]) / 2
    l_ideal_homing = int(range_center) - l_abs
    range_center_f = (f_range[0] + f_range[1]) / 2
    f_ideal_homing = int(range_center_f) - f_abs

    print(f"\n=== LEADER wrist_roll ===")
    print(f"  Present_Position (raw from motor): {l_raw}")
    print(f"  Absolute encoder (raw - homing):   {l_abs}")
    print(f"  Homing_Offset:                     {l_homing}")
    print(f"  Calibration range:                 [{l_range[0]}, {l_range[1]}]")
    print(f"  Normalized position:               {l_norm:.1f}  (target: near 0)")
    print(f"  Suggested homing_offset:           {l_ideal_homing}")

    print(f"\n=== FOLLOWER wrist_roll ===")
    print(f"  Present_Position (raw from motor): {f_raw}")
    print(f"  Absolute encoder (raw - homing):   {f_abs}")
    print(f"  Homing_Offset:                     {f_homing}")
    print(f"  Calibration range:                 [{f_range[0]}, {f_range[1]}]")
    print(f"  Normalized position:               {f_norm:.1f}  (target: near 0)")
    print(f"  Suggested homing_offset:           {f_ideal_homing}")

    print(f"\nIf normalized ≈ 0 at rest, calibration is centered correctly.")
    print(f"If normalized is far from 0, use the suggested homing_offset values.")

    safe_disconnect(follower)
    safe_disconnect(leader)


if __name__ == "__main__":
    main()
