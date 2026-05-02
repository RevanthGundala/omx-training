"""Hardware setup helpers for OMX leader/follower arms.

Returns stock LeRobot OmxLeader / OmxFollower with no patches. Calibration
is handled by stock LeRobot — either via factory defaults on first connect
or via the `lerobot-calibrate` CLI.

Recording targets a fresh dataset (RevanthGundala/003-pour-water). The
older RevanthGundala/002-pour-water dataset is abandoned (not retrofittable
across cal regimes).
"""

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
from lerobot.robots.omx_follower.omx_follower import OmxFollower
from lerobot.teleoperators.omx_leader.config_omx_leader import OmxLeaderConfig
from lerobot.teleoperators.omx_leader.omx_leader import OmxLeader

from utils.config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CAMERAS,
    FOLLOWER_PORT,
    FPS,
    LEADER_PORT,
)


def create_follower(
    port: str | None = None,
    camera: bool = True,
    fps: int | None = None,
    camera_width: int | None = None,
    camera_height: int | None = None,
) -> OmxFollower:
    """Create a stock OmxFollower with cameras attached."""
    port = port or FOLLOWER_PORT
    fps = fps or FPS
    cameras = {}
    if camera:
        for name, index in CAMERAS.items():
            cameras[name] = OpenCVCameraConfig(
                index_or_path=index,
                fps=fps,
                width=camera_width or CAMERA_WIDTH,
                height=camera_height or CAMERA_HEIGHT,
                warmup_s=5,
            )
    config = OmxFollowerConfig(port=port, id="omx_follower_arm", cameras=cameras)
    return OmxFollower(config)


def create_leader(port: str | None = None) -> OmxLeader:
    """Create a stock OmxLeader."""
    port = port or LEADER_PORT
    config = OmxLeaderConfig(port=port, id="omx_leader_arm")
    return OmxLeader(config)


def safe_disconnect(robot) -> None:
    """Disconnect a robot, suppressing hardware errors."""
    try:
        robot.disconnect()
    except RuntimeError:
        print("Warning: clean disconnect failed (motor hardware error). Power-cycling recommended.")
