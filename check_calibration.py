"""
check_calibration.py — Hold leader and follower in the SAME physical pose,
then run this. If the numbers differ by more than ~5°, the arms are
calibrated differently and soft_start will drive the follower into a bad pose.

Read-only: connects, reads one snapshot, prints, disconnects. No motion.
"""

from robot_utils import create_follower, create_leader, safe_disconnect


def main():
    leader = create_leader()
    follower = create_follower(camera=False)

    leader.connect()
    follower.connect()

    try:
        leader_pos = leader.get_action()
        follower_pos = follower.get_observation()

        joints = [k for k in leader_pos if k in follower_pos]

        print(f"\n{'joint':<20}{'leader':>10}{'follower':>12}{'diff':>10}")
        print("-" * 52)
        for j in joints:
            L = leader_pos[j]
            F = follower_pos[j]
            print(f"{j:<20}{L:>10.2f}{F:>12.2f}{L - F:>+10.2f}")
        print()
    finally:
        safe_disconnect(leader)
        safe_disconnect(follower)


if __name__ == "__main__":
    main()
