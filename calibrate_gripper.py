"""
calibrate_gripper.py -- Calibrate gripper mechanical range. Manual, foolproof.

Why grippers need this when body joints don't
---------------------------------------------
- Body joints span most of the encoder's 0..4095 range; align_follower.py just
  centers them with Homing_Offset.
- The gripper jaws hit hard mechanical stops within a small slice of the
  encoder range (~1300 ticks of 4096). Without correct range_min/range_max,
  "60% open" doesn't mean the same physical state on both arms.

Approach (Option B: software-side gripper inversion)
----------------------------------------------------
We always set firmware Drive_Mode=0 on both grippers. Then range_min/range_max
get saved as the honest min and max of the encoder values you actually
measure. No firmware-frame mismatches.

LeRobot's percent formula always says: percent=0 at PP=range_min,
percent=100 at PP=range_max. So:
  - If your arm's OPEN position is at range_max (open is the high encoder
    value), percent=100 mechanically means OPEN. Good, no inversion.
  - If your arm's OPEN position is at range_min (open is the low encoder
    value), percent=100 mechanically means CLOSED. We mark this arm as
    "inverted" so PatchedOmxLeader / PatchedOmxFollower flip gripper.pos
    in software (gripper.pos = 100 - gripper.pos) on read/write.

The inversion flags are saved in a side file:
  ~/.cache/huggingface/lerobot/calibration/omx_gripper_inversion.json

Run order
---------
    uv run python align_follower.py
    uv run python calibrate_gripper.py
    uv run python check_calibration.py
    uv run python teleop.py
"""

import json
import time
from pathlib import Path

from robot_utils import (
    GRIPPER_INVERSION_PATH,
    create_follower,
    create_leader,
    safe_disconnect,
)

CACHE = Path.home() / ".cache/huggingface/lerobot/calibration"
LEADER_CAL_PATH = CACHE / "teleoperators/omx_leader/omx_leader_arm.json"
FOLLOWER_CAL_PATH = CACHE / "robots/omx_follower/omx_follower_arm.json"


def read_raw(bus, motor: str = "gripper") -> int:
    return bus.sync_read("Present_Position", [motor], normalize=False)[motor]


def prep_gripper(bus, label: str) -> None:
    """Set firmware Drive_Mode=0 and disable torque so the user can move freely.

    Drive_Mode must be set BEFORE measuring extremes so encoder readings are
    in the canonical (non-flipped) frame. Inversion happens in software in
    robot_utils.

    Note: Squeeze with NORMAL teleop force at the extreme prompts. Squeezing
    extra hard saves an unreachable range_max (the leader trigger has user
    effort variance ~85 ticks between "trying" and "casual" squeezes).
    """
    bus.disable_torque(["gripper"])
    bus.write("Drive_Mode", "gripper", 0)
    print(f"  {label} gripper: firmware Drive_Mode reset to 0, torque off.")


def main() -> None:
    leader = create_leader()
    follower = create_follower(camera=False)
    leader.connect()
    follower.connect()

    try:
        prep_gripper(leader.bus, "LEADER")
        prep_gripper(follower.bus, "FOLLOWER")

        input(
            "\n[1/4] Move the LEADER trigger so the jaws are fully OPEN (apart), "
            "hold it there, and press ENTER..."
        )
        leader_open_raw = read_raw(leader.bus)
        print(f"  leader open raw  = {leader_open_raw}")

        input(
            "\n[2/4] Now move the LEADER trigger so the jaws are fully CLOSED "
            "(together), hold it there, and press ENTER..."
        )
        leader_close_raw = read_raw(leader.bus)
        print(f"  leader close raw = {leader_close_raw}")

        if abs(leader_close_raw - leader_open_raw) < 500:
            raise RuntimeError(
                f"Leader gripper barely moved ({leader_open_raw} -> {leader_close_raw}). "
                "Did you hold the trigger at both extremes?"
            )

        input("\n[3/4] Move FOLLOWER gripper fully OPEN (jaws apart) by hand, then press ENTER...")
        follower_open_raw = read_raw(follower.bus)
        print(f"  follower open raw  = {follower_open_raw}")

        input("\n[4/4] Move FOLLOWER gripper fully CLOSED (jaws together) by hand, then press ENTER...")
        follower_close_raw = read_raw(follower.bus)
        print(f"  follower close raw = {follower_close_raw}")

        if abs(follower_close_raw - follower_open_raw) < 500:
            raise RuntimeError(
                f"Follower gripper barely moved ({follower_open_raw} -> {follower_close_raw}). "
                "Did you move both jaws to both extremes?"
            )

        # Honest min/max -- always the smaller and larger of the two raw reads.
        # Reset gripper homing_offset to 0 so PP == raw encoder (avoids the
        # large body-joint centering offsets that align_follower may have left).
        leader.bus.write("Homing_Offset", "gripper", 0)
        follower.bus.write("Homing_Offset", "gripper", 0)
        l_homing = 0
        f_homing = 0

        l_range_min = min(leader_open_raw, leader_close_raw)
        l_range_max = max(leader_open_raw, leader_close_raw)
        f_range_min = min(follower_open_raw, follower_close_raw)
        f_range_max = max(follower_open_raw, follower_close_raw)

        # Inversion flags: True if the OPEN position is at range_min (the low
        # encoder value). Then percent=100 (which lerobot computes at
        # range_max) mechanically means CLOSED, and we need to flip in software.
        # Must be computed BEFORE padding shifts the range boundaries.
        l_inverted = leader_open_raw == l_range_min
        f_inverted = follower_open_raw == f_range_min

        # Padding: during teleop the user can't quite reach the torque-off
        # extremes measured above (leader trigger has 30mA motor resistance).
        # We shrink the OPEN end inward so the user reaches 100% open easily,
        # and EXTEND the CLOSED end outward so the motor drives slightly past
        # the measured closed position (takes up mechanical backlash).
        l_span = l_range_max - l_range_min
        f_span = f_range_max - f_range_min
        OPEN_PAD_PCT = 0.03   # shrink open end inward 3%
        CLOSE_PAD_PCT = 0.01  # extend closed end outward 1%

        if l_inverted:
            # Leader: open=low (range_min), closed=high (range_max)
            l_range_min += int(l_span * OPEN_PAD_PCT)   # shrink open end
            l_range_max += int(l_span * CLOSE_PAD_PCT)  # extend closed end
        else:
            # Leader: open=high (range_max), closed=low (range_min)
            l_range_max -= int(l_span * OPEN_PAD_PCT)
            l_range_min -= int(l_span * CLOSE_PAD_PCT)

        if f_inverted:
            # Follower: open=low (range_min), closed=high (range_max)
            f_range_min += int(f_span * OPEN_PAD_PCT)
            f_range_max += int(f_span * CLOSE_PAD_PCT)
        else:
            # Follower: open=high (range_max), closed=low (range_min)
            f_range_max -= int(f_span * OPEN_PAD_PCT)
            f_range_min -= int(f_span * CLOSE_PAD_PCT)

        print("\nProposed calibration:")
        print(f"  LEADER   gripper: drive_mode=0, homing_offset=0, "
              f"range=[{l_range_min}, {l_range_max}], open_raw={leader_open_raw}, "
              f"sw_invert={l_inverted}")
        print(f"  FOLLOWER gripper: drive_mode=0, homing_offset=0, "
              f"range=[{f_range_min}, {f_range_max}], open_raw={follower_open_raw}, "
              f"sw_invert={f_inverted}")

        # ---------- Apply to motor EEPROM ----------
        leader.bus.write("Min_Position_Limit", "gripper", l_range_min)
        leader.bus.write("Max_Position_Limit", "gripper", l_range_max)
        follower.bus.write("Min_Position_Limit", "gripper", f_range_min)
        follower.bus.write("Max_Position_Limit", "gripper", f_range_max)

        # ---------- Save calibration JSONs ----------
        for path, rmin, rmax in [
            (LEADER_CAL_PATH, l_range_min, l_range_max),
            (FOLLOWER_CAL_PATH, f_range_min, f_range_max),
        ]:
            with path.open("r") as f:
                cal = json.load(f)
            cal["gripper"]["drive_mode"] = 0
            cal["gripper"]["homing_offset"] = 0
            cal["gripper"]["range_min"] = rmin
            cal["gripper"]["range_max"] = rmax
            with path.open("w") as f:
                json.dump(cal, f, indent=4)
            print(f"  saved {path}")

        # ---------- Save inversion side file ----------
        GRIPPER_INVERSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        with GRIPPER_INVERSION_PATH.open("w") as f:
            json.dump({"leader": l_inverted, "follower": f_inverted}, f, indent=4)
        print(f"  saved {GRIPPER_INVERSION_PATH}")

        # ---------- Verify on leader (after re-applying configure-style setup) ----------
        # With torque off, leader spring rests at OPEN. After Patched configure() runs
        # on next connect, software inversion (if any) will flip the read so that
        # OPEN reads as percent~=100. We verify the raw mapping here.
        leader.bus.disable_torque(["gripper"])
        time.sleep(1.0)
        raw_now = read_raw(leader.bus)
        # Raw -> "natural percent" without sw inversion:
        nat_pct = (raw_now - l_range_min) / (l_range_max - l_range_min) * 100
        # After sw inversion (if needed):
        eff_pct = 100 - nat_pct if l_inverted else nat_pct
        print(
            f"\n[verify] leader gripper raw={raw_now}  natural%={nat_pct:.1f}  "
            f"after_sw_invert%={eff_pct:.1f}  (expect ~100 if spring rests OPEN)"
        )

    finally:
        safe_disconnect(leader)
        safe_disconnect(follower)

    print("\nDone. Run check_calibration.py and then teleop.py.")


if __name__ == "__main__":
    main()
