"""
teleop.py — Teleoperate the OMX follower arm using the OMX leader arm.

Move the leader arm by hand and the follower arm mirrors your movements in real time.
Press Ctrl+C to stop.
"""

import time

from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
from lerobot.robots.omx_follower.omx_follower import OmxFollower
from lerobot.teleoperators.omx_leader.config_omx_leader import OmxLeaderConfig
from lerobot.teleoperators.omx_leader.omx_leader import OmxLeader

# ──────────────────────────────────────────────
# Configuration — edit these to match your setup
# ──────────────────────────────────────────────
LEADER_PORT = "/dev/tty.usbmodem1301"
FOLLOWER_PORT = "/dev/tty.usbmodem1201"
FPS = 60


def main():
    # ── 1. Create configs ──
    # id must match the calibration filename (e.g. "omx_leader_arm" → omx_leader_arm.json)
    leader_config = OmxLeaderConfig(port=LEADER_PORT, id="omx_leader_arm")
    follower_config = OmxFollowerConfig(port=FOLLOWER_PORT, id="omx_follower_arm")

    # ── 2. Instantiate hardware ──
    leader = OmxLeader(leader_config)
    follower = OmxFollower(follower_config)

    # ── 3. Connect (this also runs calibration if needed) ──
    print("Connecting leader arm...")
    leader.connect()
    print("Connecting follower arm...")
    follower.connect()
    print(f"Teleop running at {FPS} FPS. Move the leader arm! Press Ctrl+C to stop.\n")

    try:
        # ── 4. Teleop loop ──
        while True:
            start = time.perf_counter()

            # Read the leader arm's joint positions
            action = leader.get_action()

            # Send those positions to the follower arm
            follower.send_action(action)

            # Print current joint positions
            positions = [f"{name}: {val:7.2f}" for name, val in action.items()]
            print("  |  ".join(positions), end="\r")

            # Wait to maintain target FPS
            elapsed = time.perf_counter() - start
            sleep_time = (1.0 / FPS) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nStopping teleoperation...")

    finally:
        # ── 5. Clean disconnect ──
        follower.disconnect()
        leader.disconnect()
        print("Disconnected. Done!")


if __name__ == "__main__":
    main()
