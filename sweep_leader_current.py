"""One-shot diagnostic: find the leader gripper Goal_Current that lets the
trigger physically reach its calibrated range_max.

For each candidate current value we:
  1. Disable torque, write Current_Limit (EEPROM, requires torque off).
  2. Re-enable torque, write Goal_Current, set rest pose.
  3. Ask user to squeeze trigger fully and press ENTER.
  4. Read Present_Position; report how close to range_max we got.

When done, prints which setting reached closest to range_max so we can pick
the right Goal_Current to bake into PatchedOmxLeader.configure().
"""

import json
import time
from pathlib import Path

from lerobot.motors.dynamixel import OperatingMode

from robot_utils import create_leader, safe_disconnect

CAL_PATH = Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/omx_leader/omx_leader_arm.json"
CANDIDATES = [50, 30, 20, 10, 5, 0]


def main() -> None:
    cal = json.loads(CAL_PATH.read_text())["gripper"]
    range_min = cal["range_min"]
    range_max = cal["range_max"]
    print(f"Calibrated leader gripper range = [{range_min}, {range_max}]\n")

    leader = create_leader()
    leader.connect()
    bus = leader.bus

    results = []
    try:
        # Read back current state to confirm previous configure() applied
        cl = bus.sync_read("Current_Limit", ["gripper"], normalize=False)["gripper"]
        gc = bus.sync_read("Goal_Current", ["gripper"], normalize=False)["gripper"]
        print(f"After connect: Current_Limit={cl}  Goal_Current={gc}\n")

        for cur in CANDIDATES:
            print(f"\n--- Trying Goal_Current = {cur} mA ---")
            bus.disable_torque(["gripper"])
            bus.write("Current_Limit", "gripper", cur)
            bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)
            bus.enable_torque(["gripper"])
            bus.write("Goal_Current", "gripper", cur)
            time.sleep(0.5)
            input("  Squeeze the trigger as hard as you would in normal teleop, "
                  "hold it, and press ENTER... ")
            raw = bus.sync_read("Present_Position", ["gripper"], normalize=False)["gripper"]
            gap = range_max - raw
            pct = (raw - range_min) / (range_max - range_min) * 100
            print(f"  raw={raw}   gap_from_max={gap}   bus_pct={pct:.1f}%")
            results.append((cur, raw, gap))

    finally:
        safe_disconnect(leader)

    print("\n=== Summary ===")
    print(f"  range_max target = {range_max}")
    for cur, raw, gap in results:
        marker = "  <-- best" if abs(gap) == min(abs(g) for _, _, g in results) else ""
        print(f"  Goal_Current={cur:3d}  reached raw={raw}  gap={gap:+5d}{marker}")
    print("\nPick the smallest current that gets gap close to 0 (or slightly")
    print("negative, meaning trigger overshoots). I'll bake that into")
    print("PatchedOmxLeader.configure().")


if __name__ == "__main__":
    main()
