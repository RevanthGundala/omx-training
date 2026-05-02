"""center_arms.py — Stock-range cal compatible with 002-pour-water dataset.

This is a stripped-down align_follower.py that writes STOCK range [0, 4095]
instead of the wide [-2048, 6143] range. Use it before replaying the
002-pour-water dataset, since that dataset was recorded under stock range.

What it does:
    1. Pose BOTH arms at your natural "home" pose by hand.
    2. set_half_turn_homings() on each arm — sets motor Homing_Offset so
       Present_Position == 2047 at the current pose.
    3. Save cal JSON with range=[0, 4095] (stock).

Why stock range:
    Recording was done with stock range. For replay's pct -> physical encoder
    mapping to match recording, the cal range MUST match. Wider range means
    different PP for the same pct, which means different physical pose.

After running this, run replay.py with calibrate=False (default) and the
motor reference frame will match the recording's reference frame.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

from lerobot.motors.motors_bus import MotorCalibration

from utils.robot_utils import create_follower, create_leader, safe_disconnect

CACHE = Path.home() / ".cache/huggingface/lerobot/calibration"
LEADER_CAL_PATH = CACHE / "teleoperators/omx_leader/omx_leader_arm.json"
FOLLOWER_CAL_PATH = CACHE / "robots/omx_follower/omx_follower_arm.json"


def center(name: str, bus) -> dict:
    print(f"\nCentering {name} arm body joints (stock range)...")
    bus.disable_torque()
    body_motors = [m for m in bus.motors if m != "gripper"]
    offsets = bus.set_half_turn_homings(body_motors)
    cal = bus.read_calibration()

    for motor in body_motors:
        c = cal[motor]
        cal[motor] = MotorCalibration(
            id=c.id,
            drive_mode=c.drive_mode,
            homing_offset=c.homing_offset,
            range_min=0,
            range_max=4095,
        )

    for motor, off in offsets.items():
        c = cal[motor]
        print(f"  {motor:<16} homing={off:>+6}  range=[{c.range_min}, {c.range_max}]")
    return cal


def save(path: Path, cal: dict) -> None:
    """Save cal JSON, preserving existing gripper range."""
    cal_dict = {k: asdict(v) for k, v in cal.items()}
    if path.exists():
        with path.open("r") as f:
            existing = json.load(f)
        if "gripper" in existing and "gripper" in cal_dict:
            cal_dict["gripper"]["range_min"] = existing["gripper"]["range_min"]
            cal_dict["gripper"]["range_max"] = existing["gripper"]["range_max"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(cal_dict, f, indent=4)
    print(f"  saved {path}")


def main() -> None:
    input(
        "Pose BOTH arms at your natural 'home' / pour-water start pose\n"
        "(the same pose you used when recording 002-pour-water), then press ENTER..."
    )

    leader = create_leader()
    follower = create_follower(camera=False)

    leader.connect(calibrate=False)
    follower.connect(calibrate=False)
    try:
        l_cal = center("leader", leader.bus)
        f_cal = center("follower", follower.bus)
        save(LEADER_CAL_PATH, l_cal)
        save(FOLLOWER_CAL_PATH, f_cal)
        print("\nDone. Now run: uv run python data/replay.py")
    finally:
        safe_disconnect(leader)
        safe_disconnect(follower)


if __name__ == "__main__":
    main()
