"""Dynamixel hardware-error helpers."""

from __future__ import annotations


HW_ERROR_BITS = {
    0: "Input Voltage",
    2: "Overheating",
    3: "Motor Encoder",
    4: "Electrical Shock",
    5: "Overload",
}


def decode_hw_error(value: int) -> list[str]:
    return [name for bit, name in HW_ERROR_BITS.items() if value & (1 << bit)]


def read_hw_errors(bus, motors: list[str] | tuple[str, ...] | None = None) -> dict[str, dict]:
    raw = bus.sync_read("Hardware_Error_Status", motors=motors, normalize=False)
    return {
        motor: {"value": int(value), "reasons": decode_hw_error(int(value))}
        for motor, value in raw.items()
    }


def faulted_hw_errors(hw_errors: dict[str, dict]) -> dict[str, dict]:
    return {motor: entry for motor, entry in hw_errors.items() if entry["value"]}


def assert_no_hw_errors(bus, motors: list[str] | tuple[str, ...] | None = None, *, label: str) -> dict[str, dict]:
    hw_errors = read_hw_errors(bus, motors)
    faulted = faulted_hw_errors(hw_errors)
    if faulted:
        raise RuntimeError(f"{label} motor hardware errors: {faulted}")
    return hw_errors
