"""Read EEPROM Homing_Offset / Min/Max_Position_Limit and compare to JSON.

If EEPROM != JSON, replay/teleop will be off by the difference. Prints a fix
suggestion. Read-only by default; pass --apply to push JSON values into EEPROM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.robot_utils import create_follower, safe_disconnect

JSON_PATH = (
    Path.home()
    / ".cache/huggingface/lerobot/calibration/robots/omx_follower/omx_follower_arm.json"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Push JSON to EEPROM")
    args = ap.parse_args()

    js = json.loads(JSON_PATH.read_text())
    follower = create_follower(camera=False)
    follower.connect(calibrate=False)
    try:
        eeprom_homing = follower.bus.sync_read("Homing_Offset", normalize=False)
        eeprom_min = follower.bus.sync_read("Min_Position_Limit", normalize=False)
        eeprom_max = follower.bus.sync_read("Max_Position_Limit", normalize=False)
        present = follower.bus.sync_read("Present_Position", normalize=False)

        print(f"{'joint':14s} {'json_homing':>11} {'eeprom_homing':>14} "
              f"{'eeprom_min':>11} {'eeprom_max':>11} {'present':>9} {'actual_enc':>12}")
        any_diff = False
        for j in ["shoulder_pan", "shoulder_lift", "elbow_flex",
                  "wrist_flex", "wrist_roll", "gripper"]:
            jh = js[j]["homing_offset"]
            eh = eeprom_homing[j]
            em = eeprom_min[j]
            eM = eeprom_max[j]
            p = present[j]
            actual = p - eh
            flag = "" if jh == eh else "  <-- MISMATCH"
            if jh != eh:
                any_diff = True
            print(f"{j:14s} {jh:>11d} {eh:>14d} {em:>11d} {eM:>11d} "
                  f"{p:>9d} {actual:>12d}{flag}")

        if any_diff and not args.apply:
            print("\nEEPROM differs from JSON. Re-run with --apply to push JSON values "
                  "to EEPROM (this re-burns Homing_Offset and Min/Max_Position_Limit).")
        elif args.apply:
            from lerobot.motors.motors_bus import MotorCalibration
            print("\nWriting JSON calibration to EEPROM...")
            cal = {}
            for j, d in js.items():
                cal[j] = MotorCalibration(
                    id=d["id"],
                    drive_mode=d.get("drive_mode", 0),
                    homing_offset=d["homing_offset"],
                    range_min=max(0, d["range_min"]),
                    range_max=min(4095, d["range_max"]),
                )
            follower.bus.write_calibration(cal)
            print("done.")
    finally:
        safe_disconnect(follower)


if __name__ == "__main__":
    main()
