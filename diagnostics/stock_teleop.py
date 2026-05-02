"""Stock LeRobot teleop — no cached calibration, no patches."""
import time
from lerobot.robots.omx_follower.omx_follower import OmxFollower
from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
from lerobot.teleoperators.omx_leader.omx_leader import OmxLeader
from lerobot.teleoperators.omx_leader.config_omx_leader import OmxLeaderConfig

leader = OmxLeader(OmxLeaderConfig(port="/dev/tty.usbmodem1301"))
follower = OmxFollower(OmxFollowerConfig(port="/dev/tty.usbmodem1401"))
leader.connect()
follower.connect()
print("Stock teleop running — Ctrl+C to stop")
try:
    while True:
        action = leader.get_action()
        follower.send_action(action)
        time.sleep(1 / 60)
except KeyboardInterrupt:
    pass
leader.disconnect()
follower.disconnect()
