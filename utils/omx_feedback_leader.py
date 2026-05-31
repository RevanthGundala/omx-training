"""OMX leader feedback adapter for LeRobot DAgger-style handoff.

Mirrors the SO-101 leader's feedback shape (single ``sync_write`` of
``Goal_Position``) but uses Dynamixel ``Profile_Velocity`` and
``Profile_Acceleration`` so the servo firmware does the trapezoidal motion
profile. This avoids whipping the small XL330 wrist motors when the operator's
hand is far from the handoff target.
"""

from __future__ import annotations

from collections.abc import Callable

from lerobot.teleoperators.omx_leader.config_omx_leader import OmxLeaderConfig
from lerobot.teleoperators.omx_leader.omx_leader import OmxLeader

from utils.dynamixel_errors import assert_no_hw_errors, read_hw_errors


EventCallback = Callable[[str], None] | Callable[..., None]


class OmxFeedbackLeader(OmxLeader):
    """OMX leader exposing a SO-101-shaped feedback API.

    All body joints (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
    wrist_roll) are actuated during DAgger handoff. The gripper is left alone:
    it is configured in CURRENT_POSITION mode as a spring-loaded trigger and
    forcing a goal would fight the operator's finger.

    Motion is paced by the Dynamixel firmware via ``Profile_Velocity`` /
    ``Profile_Acceleration`` (written on ``enable_torque``, cleared on
    ``disable_torque``). This keeps ``send_feedback`` a one-line sync_write,
    matching the SO-101 leader.
    """

    FEEDBACK_MOTORS = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )

    # Dynamixel profile registers. Units (XL330):
    #   Profile_Velocity:     0.229 rev/min per LSB  →  30 ≈ 41 °/s
    #   Profile_Acceleration: 214.577 rev/min² per LSB → 20 ≈ smooth ramp
    # Tuned conservatively; adjust on real hardware if handoff feels too slow.
    PROFILE_VELOCITY = 30
    PROFILE_ACCELERATION = 20

    def __init__(self, config: OmxLeaderConfig, event_callback: EventCallback | None = None):
        super().__init__(config)
        self.event_callback = event_callback
        self.feedback_active = False

    @property
    def feedback_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.FEEDBACK_MOTORS}

    def _log(self, event: str, **fields) -> None:
        if self.event_callback is not None:
            self.event_callback(event, **fields)

    def read_feedback_hw_errors(self) -> dict[str, dict]:
        return read_hw_errors(self.bus, self.FEEDBACK_MOTORS)

    def preflight_feedback(self, *, label: str = "leader_feedback") -> dict:
        hw_errors = assert_no_hw_errors(self.bus, self.FEEDBACK_MOTORS, label=label)
        present = self.bus.sync_read("Present_Position", motors=self.FEEDBACK_MOTORS)
        result = {
            "label": label,
            "feedback_motors": list(self.FEEDBACK_MOTORS),
            "present_position": {motor: float(value) for motor, value in present.items()},
            "hardware_errors": hw_errors,
            "ok": True,
        }
        self._log("leader_feedback_preflight", **result)
        return result

    def _write_profile(self, velocity: int, acceleration: int) -> None:
        for motor in self.FEEDBACK_MOTORS:
            self.bus.write("Profile_Acceleration", motor, acceleration)
            self.bus.write("Profile_Velocity", motor, velocity)

    def enable_torque(self) -> None:
        self.preflight_feedback(label="leader_feedback_enable")
        present = self.bus.sync_read("Present_Position", motors=self.FEEDBACK_MOTORS)
        present = {motor: float(value) for motor, value in present.items()}
        # Seed Goal_Position before enabling torque so the motors don't lurch
        # against whatever stale goal sits in EEPROM.
        self.bus.sync_write("Goal_Position", present)
        self._write_profile(self.PROFILE_VELOCITY, self.PROFILE_ACCELERATION)
        self.bus.enable_torque(list(self.FEEDBACK_MOTORS))
        self.feedback_active = True
        self._log(
            "leader_feedback_enabled",
            present=present,
            profile_velocity=self.PROFILE_VELOCITY,
            profile_acceleration=self.PROFILE_ACCELERATION,
        )

    def disable_torque(self) -> None:
        self.bus.disable_torque(list(self.FEEDBACK_MOTORS))
        # Restore unlimited profile so non-feedback use (calibration, manual
        # jogging) is unaffected.
        try:
            self._write_profile(0, 0)
        except Exception:
            pass
        self.feedback_active = False
        self._log("leader_feedback_disabled")

    def send_feedback(self, feedback: dict[str, float]) -> None:
        goals = {
            key.removesuffix(".pos"): float(value)
            for key, value in feedback.items()
            if key.endswith(".pos") and key.removesuffix(".pos") in self.FEEDBACK_MOTORS
        }
        if not goals:
            return
        try:
            self.bus.sync_write("Goal_Position", goals)
            self._log("leader_feedback_sent", goals=goals)
        except Exception:
            try:
                self.disable_torque()
            finally:
                raise
