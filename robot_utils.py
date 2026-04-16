"""Hardware setup helpers for OMX leader/follower arms."""

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
from lerobot.robots.omx_follower.omx_follower import OmxFollower
from lerobot.teleoperators.omx_leader.config_omx_leader import OmxLeaderConfig
from lerobot.teleoperators.omx_leader.omx_leader import OmxLeader

from config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
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
    """Create an OmxFollower with optional camera."""
    port = port or FOLLOWER_PORT
    fps = fps or FPS
    cameras = {}
    if camera:
        cameras["front"] = OpenCVCameraConfig(
            index_or_path=CAMERA_INDEX,
            fps=fps,
            width=camera_width or CAMERA_WIDTH,
            height=camera_height or CAMERA_HEIGHT,
        )
    config = OmxFollowerConfig(port=port, id="omx_follower_arm", cameras=cameras)
    return OmxFollower(config)


def create_leader(port: str | None = None) -> OmxLeader:
    """Create an OmxLeader."""
    port = port or LEADER_PORT
    config = OmxLeaderConfig(port=port, id="omx_leader_arm")
    return OmxLeader(config)


def safe_disconnect(robot) -> None:
    """Disconnect a robot, suppressing hardware errors."""
    try:
        robot.disconnect()
    except RuntimeError:
        print("Warning: clean disconnect failed (motor hardware error). Power-cycling recommended.")
