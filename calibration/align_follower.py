"""
align_follower.py — Centre BOTH arms' encoders at the current (shared) pose.

Why this exists:
    The OMX shipped with factory per-unit Homing_Offset values that made each
    motor's encoder center sit roughly in the middle of its mechanical range.
    LeRobot's first-connect calibrate() overwrote those with Homing_Offset=0,
    which can leave a motor pinned to the edge of its 0..4095 encoder range
    (e.g. leader shoulder_pan reading raw=-2, which clamps normalized to -100).
    Once a motor is at the range edge, no follower-side offset can make teleop
    work — leader moves in one direction but the normalized value is already
    saturated.

What this script does:
    1. Pose BOTH arms into the same, comfortable middle shape by hand.
    2. Call bus.set_half_turn_homings() on each arm — this sets each motor's
       Homing_Offset so Present_Position == 2047 at the current pose. Both
       arms now report identical values at this neutral pose, with symmetric
       range on each side.
    3. Persist the new calibration to the JSON caches so the next connect
       sees is_calibrated == True and skips the default calibrate().

Run order:
    1. uv run python align_follower.py
    2. uv run python check_calibration.py   # verify diffs ~ 0
    3. uv run python teleop.py
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from utils.robot_utils import create_follower, create_leader, safe_disconnect

CACHE = Path.home() / ".cache/huggingface/lerobot/calibration"
LEADER_CAL_PATH = CACHE / "teleoperators/omx_leader/omx_leader_arm.json"
FOLLOWER_CAL_PATH = CACHE / "robots/omx_follower/omx_follower_arm.json"


def drive_follower_to_leader(leader, follower, duration_s=3.0, fps=30):
    """Read leader body joint positions and slowly drive the follower to match."""
    print("\nDriving follower to match leader pose...")
    current = follower.get_observation()
    target = leader.get_action()
    steps = max(1, int(duration_s * fps))

    for step in range(1, steps + 1):
        alpha = step / steps
        blended = {}
        for key in target:
            if key == "gripper.pos":
                continue
            blended[key] = current.get(key, 0) + alpha * (target.get(key, 0) - current.get(key, 0))
        # Keep gripper at current position
        if "gripper.pos" in target:
            blended["gripper.pos"] = current.get("gripper.pos", 0)
        follower.send_action(blended)
        time.sleep(1.0 / fps)

    print("  Follower matched to leader pose.")
    # Disable torque so set_half_turn_homings can run
    follower.bus.disable_torque()


def center(name: str, bus) -> dict:
    print(f"\nCentering {name} arm body joints...")
    bus.disable_torque()
    body_motors = [m for m in bus.motors if m != "gripper"]
    offsets = bus.set_half_turn_homings(body_motors)
    cal = bus.read_calibration()

    # Widen range for body joints so normalization doesn't clamp in Extended
    # Position mode.  set_half_turn_homings() centres each motor at 2047, but
    # the default range [0, 4095] only covers one revolution (±180°).  Joints
    # that can physically exceed ±180° from centre will saturate and the
    # follower will stop tracking.  Using [-2048, 6143] gives ±270° of
    # headroom.  _write_calibration_safe() still clamps EEPROM writes to
    # [0, 4095] (those registers are ignored in Extended Position mode anyway).
    from lerobot.motors.motors_bus import MotorCalibration

    for motor in body_motors:
        c = cal[motor]
        cal[motor] = MotorCalibration(
            id=c.id,
            drive_mode=c.drive_mode,
            homing_offset=c.homing_offset,
            range_min=-2048,
            range_max=6143,
        )

    for motor, off in offsets.items():
        c = cal[motor]
        print(f"  {motor:<16} homing={off:>+6}  range=[{c.range_min}, {c.range_max}]")
    return cal


def save(path: Path, cal: dict) -> None:
    """Save calibration to JSON, preserving existing gripper range values.

    The gripper range is software-only (not stored in EEPROM), so
    read_calibration() always returns range=[0, 4095] for it. We must
    preserve the values set by calibrate_gripper.py.
    """
    if path.exists():
        with path.open("r") as f:
            existing = json.load(f)
        if "gripper" in existing and "gripper" in cal:
            from dataclasses import asdict as _asdict
            cal_dict = {k: asdict(v) for k, v in cal.items()}
            cal_dict["gripper"]["range_min"] = existing["gripper"]["range_min"]
            cal_dict["gripper"]["range_max"] = existing["gripper"]["range_max"]
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                json.dump(cal_dict, f, indent=4)
            print(f"  saved {path} (gripper range preserved)")
            return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({k: asdict(v) for k, v in cal.items()}, f, indent=4)
    print(f"  saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align follower body joints to leader.")
    parser.add_argument(
        "--auto", action="store_true",
        help="Automatically drive the follower to match the leader's current pose, "
             "then calibrate both. Leader must already be posed.",
    )
    args = parser.parse_args()

    if not args.auto:
        input(
            "Pose BOTH arms into the same, comfortable middle shape\n"
            "(nothing at a joint limit), then press ENTER to continue..."
        )

    leader = create_leader()
    follower = create_follower(camera=False)

    leader.connect()
    follower.connect()
    try:
        if args.auto:
            drive_follower_to_leader(leader, follower)

        l_cal = center("leader", leader.bus)
        f_cal = center("follower", follower.bus)
        save(LEADER_CAL_PATH, l_cal)
        save(FOLLOWER_CAL_PATH, f_cal)
        print("\nDone. Re-run check_calibration.py to verify, then teleop.py.")
        print("(Gripper calibration is managed separately by calibrate_gripper.py.)")
    finally:
        safe_disconnect(leader)
        safe_disconnect(follower)


if __name__ == "__main__":
    main()
