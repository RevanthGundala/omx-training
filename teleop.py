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


def main():
    leader = create_leader()
    follower = create_follower(camera=False)

    print("Connecting leader arm...")
    leader.connect()
    print("Connecting follower arm...")
    follower.connect()

    soft_start(follower, leader)
    print(f"Teleop running at {FPS} FPS. Move the leader arm! Press Ctrl+C to stop.\n")

    try:
        while True:
            start = time.perf_counter()
            action = leader.get_action()
            follower.send_action(action)

            positions = [f"{name}: {val:7.2f}" for name, val in action.items()]
            print("  |  ".join(positions), end="\r")

            maintain_fps(start, FPS)

    except KeyboardInterrupt:
        print("\n\nStopping teleoperation...")
    finally:
        safe_disconnect(follower)
        safe_disconnect(leader)
        print("Disconnected. Done!")


if __name__ == "__main__":
    main()
