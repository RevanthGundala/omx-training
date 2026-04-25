"""
inspect_state.py — Print the current calibration state of both arms.

For each motor on each arm, prints:
  - Motor EEPROM: Homing_Offset, Drive_Mode, Min/Max_Position_Limit, raw Present_Position
  - Calibration JSON: homing_offset, drive_mode, range_min, range_max
  - Whether they agree (is_calibrated is the AND of these per-motor checks)
  - Computed normalized Present_Position

Read-only. Doesn't move motors, doesn't write anything.
"""

import json
from pathlib import Path

from robot_utils import create_follower, create_leader, safe_disconnect

CACHE = Path.home() / ".cache/huggingface/lerobot/calibration"
LEADER_CAL_PATH = CACHE / "teleoperators/omx_leader/omx_leader_arm.json"
FOLLOWER_CAL_PATH = CACHE / "robots/omx_follower/omx_follower_arm.json"


def dump(name: str, bus, json_path: Path) -> None:
    print(f"\n=== {name} ({json_path.name}) ===")
    motors = list(bus.motors)

    eeprom_homing = bus.sync_read("Homing_Offset", motors, normalize=False)
    eeprom_drive = bus.sync_read("Drive_Mode", motors, normalize=False)
    eeprom_min = bus.sync_read("Min_Position_Limit", motors, normalize=False)
    eeprom_max = bus.sync_read("Max_Position_Limit", motors, normalize=False)
    pos_raw = bus.sync_read("Present_Position", motors, normalize=False)
    try:
        pos_norm = bus.sync_read("Present_Position", motors)
    except Exception as exc:
        pos_norm = {m: f"err:{exc}" for m in motors}

    if json_path.exists():
        with json_path.open() as f:
            cal = json.load(f)
    else:
        cal = {}

    header = (
        f"{'motor':<14}"
        f"{'EEPROM':<48}"
        f"{'JSON':<48}"
        f"{'match':<6}"
        f"{'pos_raw':>8}"
        f"{'pos_norm':>10}"
    )
    print(header)
    print("-" * len(header))
    for m in motors:
        e = (
            f"hom={eeprom_homing[m]:>+5} "
            f"drv={eeprom_drive[m]} "
            f"min={eeprom_min[m]:>5} "
            f"max={eeprom_max[m]:>5}"
        )
        j = cal.get(m, {})
        if j:
            jstr = (
                f"hom={j['homing_offset']:>+5} "
                f"drv={j['drive_mode']} "
                f"min={j['range_min']:>5} "
                f"max={j['range_max']:>5}"
            )
            match = (
                eeprom_homing[m] == j["homing_offset"]
                and eeprom_drive[m] == j["drive_mode"]
                and eeprom_min[m] == j["range_min"]
                and eeprom_max[m] == j["range_max"]
            )
        else:
            jstr = "(no JSON entry)"
            match = False
        norm_str = f"{pos_norm[m]:.2f}" if isinstance(pos_norm[m], (int, float)) else str(pos_norm[m])
        print(f"{m:<14}{e:<48}{jstr:<48}{'OK' if match else 'BAD':<6}{pos_raw[m]:>8}{norm_str:>10}")

    print(f"  bus.is_calibrated = {bus.is_calibrated}")


def main() -> None:
    leader = create_leader()
    follower = create_follower(camera=False)
    leader.connect()
    follower.connect()
    try:
        dump("LEADER", leader.bus, LEADER_CAL_PATH)
        dump("FOLLOWER", follower.bus, FOLLOWER_CAL_PATH)
    finally:
        safe_disconnect(leader)
        safe_disconnect(follower)


if __name__ == "__main__":
    main()
