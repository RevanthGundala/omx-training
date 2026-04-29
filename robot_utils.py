"""Hardware setup helpers for OMX leader/follower arms."""

import json
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.motors.dynamixel import DriveMode, OperatingMode
from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
from lerobot.robots.omx_follower.omx_follower import OmxFollower
from lerobot.teleoperators.omx_leader.config_omx_leader import OmxLeaderConfig
from lerobot.teleoperators.omx_leader.omx_leader import OmxLeader

# Side-file storing software-side gripper inversion flags. We store these
# outside the per-arm calibration JSON because LeRobot's MotorCalibration
# dataclass has no field for them and we don't want to fork that schema.
# See calibrate_gripper.py docstring for why software inversion exists.
GRIPPER_INVERSION_PATH = (
    Path.home() / ".cache/huggingface/lerobot/calibration/omx_gripper_inversion.json"
)


def _load_gripper_inversion() -> dict:
    """Returns {"leader": bool, "follower": bool}. Defaults to False/False."""
    if not GRIPPER_INVERSION_PATH.exists():
        return {"leader": False, "follower": False}
    with GRIPPER_INVERSION_PATH.open("r") as f:
        data = json.load(f)
    return {
        "leader": bool(data.get("leader", False)),
        "follower": bool(data.get("follower", False)),
    }


class PatchedOmxLeader(OmxLeader):
    """OmxLeader that respects the gripper's calibration JSON and applies
    software-side gripper inversion if configured.

    Stock OmxLeader.configure() hard-pins gripper Drive_Mode=INVERTED (1) and
    Homing_Offset=100 on every connect, which makes it impossible to support a
    leader gripper whose motor was mounted with reversed clocking. This
    subclass writes the values from the JSON instead.

    In Option B we always write firmware Drive_Mode=0 (saved in JSON) and do
    open<->closed flipping in software here so the percent emitted matches the
    LeRobot convention: 0 = CLOSED, 100 = OPEN. (gripper_open_pos=60 means
    "rest at 60% open".)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gripper_invert = _load_gripper_inversion()["leader"]

    def calibrate(self) -> None:
        """Override stock calibrate() which wipes gripper calibration.

        Stock OmxLeader.calibrate() unconditionally sets gripper to
        drive_mode=1, homing_offset=100, range=[0, 4095] and saves to JSON.
        This destroys the gripper range calibration from calibrate_gripper.py.
        Our override writes the EEPROM to match whatever the JSON already
        contains, then saves — no destructive reset.
        """
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)
            self.bus.write("Drive_Mode", motor, DriveMode.NON_INVERTED.value)

        if self.calibration:
            # JSON already has calibration — push it to EEPROM so they match.
            self.bus.write_calibration(self.calibration)
        else:
            # First-ever connect, no JSON yet. Use stock defaults for body
            # joints, leave gripper at drive_mode=0 so calibrate_gripper.py
            # can set the range properly later.
            from lerobot.motors.motors_bus import MotorCalibration

            self.calibration = {}
            for motor, m in self.bus.motors.items():
                self.calibration[motor] = MotorCalibration(
                    id=m.id,
                    drive_mode=0,
                    homing_offset=0,
                    range_min=0,
                    range_max=4095,
                )
            self.bus.write_calibration(self.calibration)

        self._save_calibration()

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            if motor != "gripper":
                self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)
                self.bus.write("Drive_Mode", motor, DriveMode.NON_INVERTED.value)

        gripper_cal = self.calibration["gripper"]
        self.bus.write("Drive_Mode", "gripper", gripper_cal.drive_mode)
        self.bus.write("Homing_Offset", "gripper", gripper_cal.homing_offset)
        self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)
        # Lowered from stock 100mA -> 30mA so the user can easily squeeze the
        # trigger past the spring/motor restoring force and reach the full
        # calibrated range_max (otherwise the motor pushes back hard enough
        # that user input never sends gripper.pos below ~5% closed, and the
        # follower visibly stops short of fully-closed jaws).
        self.bus.write("Current_Limit", "gripper", 30)
        self.bus.write("Goal_Current", "gripper", 30)
        self.bus.enable_torque("gripper")
        if self.is_calibrated:
            # Goal_Position must be in the firmware frame (no software flip).
            # gripper_open_pos is a percent in the LeRobot convention (where
            # 100 = OPEN). With software inversion enabled, the firmware
            # percent that physically means OPEN is (100 - gripper_open_pos).
            firmware_pct = (
                100 - self.config.gripper_open_pos
                if self._gripper_invert
                else self.config.gripper_open_pos
            )
            self.bus.write("Goal_Position", "gripper", firmware_pct)

    def get_action(self) -> dict[str, float]:
        action = super().get_action()
        if self._gripper_invert and "gripper.pos" in action:
            action["gripper.pos"] = 100.0 - action["gripper.pos"]
        return action


class PatchedOmxFollower(OmxFollower):
    """OmxFollower with calibrate() override and optional software gripper inversion.

    Stock OmxFollower.calibrate() unconditionally resets ALL motors to
    drive_mode=0, homing_offset=0, range=[0, 4095] — destroying the gripper
    range calibration from calibrate_gripper.py. This subclass overrides
    calibrate() with the same JSON-preserving logic used in PatchedOmxLeader.

    The follower gripper motor on some units is clocked such that the encoder
    counts UP when the jaws CLOSE (open is at range_min). LeRobot always
    normalizes percent=100 at range_max, so without inversion sending
    "gripper.pos = 100" would close the jaws. We flip it on send/receive here
    so the LeRobot convention (0=CLOSED, 100=OPEN) holds end-to-end.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gripper_invert = _load_gripper_inversion()["follower"]

    def calibrate(self) -> None:
        """Override stock calibrate() which wipes gripper calibration.

        Same pattern as PatchedOmxLeader: push JSON → EEPROM if JSON exists,
        otherwise use factory defaults only on first-ever connect.
        """
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)
            self.bus.write("Drive_Mode", motor, DriveMode.NON_INVERTED.value)

        if self.calibration:
            self.bus.write_calibration(self.calibration)
        else:
            from lerobot.motors.motors_bus import MotorCalibration

            self.calibration = {}
            for motor, m in self.bus.motors.items():
                self.calibration[motor] = MotorCalibration(
                    id=m.id,
                    drive_mode=0,
                    homing_offset=0,
                    range_min=0,
                    range_max=4095,
                )
            self.bus.write_calibration(self.calibration)

        self._save_calibration()

    def configure(self) -> None:
        super().configure()

    def get_observation(self) -> dict:
        obs = super().get_observation()
        if self._gripper_invert and "gripper.pos" in obs:
            obs["gripper.pos"] = 100.0 - obs["gripper.pos"]
        return obs

    def send_action(self, action: dict) -> dict:
        if self._gripper_invert and "gripper.pos" in action:
            action = dict(action)
            action["gripper.pos"] = 100.0 - action["gripper.pos"]
        sent = super().send_action(action)
        if self._gripper_invert and "gripper.pos" in sent:
            sent = dict(sent)
            sent["gripper.pos"] = 100.0 - sent["gripper.pos"]
        return sent


from config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
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
    """Create a PatchedOmxFollower (applies software gripper inversion if configured)."""
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
            )
    config = OmxFollowerConfig(port=port, id="omx_follower_arm", cameras=cameras)
    return PatchedOmxFollower(config)


def create_leader(port: str | None = None) -> OmxLeader:
    """Create a PatchedOmxLeader (respects gripper drive_mode from JSON, applies sw inversion)."""
    port = port or LEADER_PORT
    config = OmxLeaderConfig(port=port, id="omx_leader_arm")
    return PatchedOmxLeader(config)


def safe_disconnect(robot) -> None:
    """Disconnect a robot, suppressing hardware errors."""
    try:
        robot.disconnect()
    except RuntimeError:
        print("Warning: clean disconnect failed (motor hardware error). Power-cycling recommended.")
