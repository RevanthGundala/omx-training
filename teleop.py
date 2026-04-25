"""
teleop.py — Teleoperate the OMX follower arm using the OMX leader arm.

Move the leader arm by hand and the follower arm mirrors your movements in real time.
Press Ctrl+C to stop.
"""

import time

from config import TELEOP_FPS as FPS
from control_utils import maintain_fps
from robot_utils import create_follower, create_leader, safe_disconnect

SOFT_START_DURATION_S = 3.0
ALIGNMENT_THRESHOLD = 15.0


def assert_arms_aligned(leader, follower, threshold=ALIGNMENT_THRESHOLD):
    """Refuse to move follower if leader/follower disagree at the start pose.

    Skips the gripper because the leader gripper is a current-controlled trigger
    (different operating mode than the follower gripper) and intentionally does
    not read identically.
    """
    leader_pos = leader.get_action()
    follower_pos = follower.get_observation()
    bad = []
    for key in leader_pos:
        if key == "gripper.pos" or key not in follower_pos:
            continue
        diff = leader_pos[key] - follower_pos[key]
        if abs(diff) > threshold:
            bad.append(
                f"  {key}: leader={leader_pos[key]:+.1f}  "
                f"follower={follower_pos[key]:+.1f}  diff={diff:+.1f}"
            )
    if bad:
        raise RuntimeError(
            "Arms not aligned at start pose — refusing to move follower.\n"
            "Re-run `uv run python align_follower.py`.\n"
            "Offending joints:\n" + "\n".join(bad)
        )


def soft_start(follower, leader):
    """Gradually move follower to match leader position to prevent jerk/overload."""
    print(f"  Soft-starting: ramping follower to leader over {SOFT_START_DURATION_S}s...")
    current = follower.get_observation()
    target = leader.get_action()
    steps = max(1, int(SOFT_START_DURATION_S * FPS))

    for step in range(1, steps + 1):
        loop_start = time.perf_counter()
        alpha = step / steps
        blended = {
            key: current[key] + alpha * (target[key] - current[key])
            for key in target
        }
        follower.send_action(blended)
        target = leader.get_action()
        maintain_fps(loop_start, FPS)

    print("  Soft-start complete.")


LOG_PATH = "teleop_log.txt"


def main():
    leader = create_leader()
    follower = create_follower(camera=False)

    print("Connecting leader arm...")
    leader.connect()
    print("Connecting follower arm...")
    follower.connect()

    assert_arms_aligned(leader, follower)
    soft_start(follower, leader)
    print(f"Teleop running at {FPS} FPS. Move the leader arm! Press Ctrl+C to stop.")
    print(f"Logging to {LOG_PATH}\n")

    logf = open(LOG_PATH, "w")
    header = "wrist_roll_L,wrist_roll_F,grip_L,grip_F,raw_L,raw_F"
    logf.write(header + "\n")

    try:
        while True:
            start = time.perf_counter()
            action = leader.get_action()
            follower.send_action(action)
            obs = follower.get_observation()

            l_grip_raw = int(leader.bus.sync_read(
                "Present_Position", ["gripper"], normalize=False
            )["gripper"])
            f_grip_raw = int(follower.bus.sync_read(
                "Present_Position", ["gripper"], normalize=False
            )["gripper"])

            grip_l = action.get("gripper.pos", 0)
            grip_f = obs.get("gripper.pos", 0)
            wr_l = action.get("wrist_roll.pos", 0)
            wr_f = obs.get("wrist_roll.pos", 0)

            logf.write(f"{wr_l:.1f},{wr_f:.1f},{grip_l:.1f},{grip_f:.1f},{l_grip_raw},{f_grip_raw}\n")
            logf.flush()

            parts = []
            for name, val in action.items():
                f_val = obs.get(name)
                if f_val is None:
                    parts.append(f"{name} L={val:6.1f}")
                else:
                    parts.append(f"{name} L={val:6.1f} F={f_val:6.1f}")
            parts.append(f"grip_raw L={l_grip_raw:5d} F={f_grip_raw:5d}")
            print("  |  ".join(parts), end="\r", flush=True)

            maintain_fps(start, FPS)

    except KeyboardInterrupt:
        print("\n\nStopping teleoperation...")
    finally:
        logf.close()
        safe_disconnect(follower)
        safe_disconnect(leader)
        print(f"Disconnected. Log saved to {LOG_PATH}. Done!")


if __name__ == "__main__":
    main()
