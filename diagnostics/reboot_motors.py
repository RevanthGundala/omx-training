"""
reboot_motors.py — Reboot Dynamixel motors to clear hardware errors (e.g. Input Voltage Error).

Usage:
    uv run python reboot_motors.py [--port PORT]
"""

import argparse
import time

from dynamixel_sdk import PacketHandler, PortHandler

from utils.config import LEADER_PORT, FOLLOWER_PORT
from utils.dynamixel_errors import HW_ERROR_BITS

PORTS = [LEADER_PORT, FOLLOWER_PORT]
BAUDRATE = 1_000_000
PROTOCOL = 2.0
SCAN_RANGE = range(1, 20)
REBOOT_WAIT_S = 2

TORQUE_ENABLE_ADDR = 64
HW_ERROR_ADDR = 70
def scan_and_reboot(port: str) -> None:
    ph = PortHandler(port)
    if not ph.openPort():
        raise RuntimeError(f"Cannot open port {port}")
    ph.setBaudRate(BAUDRATE)
    pkt = PacketHandler(PROTOCOL)

    # Scan for motors
    print(f"Scanning {port} for motors...")
    motors: dict[int, int] = {}
    for mid in SCAN_RANGE:
        model, comm, _ = pkt.ping(ph, mid)
        if comm == 0:
            motors[mid] = model

    if not motors:
        print("No motors found.")
        ph.closePort()
        return

    print(f"Found {len(motors)} motor(s): {list(motors.keys())}\n")

    # Disable torque on all motors
    print("Disabling torque...")
    for mid in motors:
        pkt.write1ByteTxRx(ph, mid, TORQUE_ENABLE_ADDR, 0)
        print(f"  ID {mid}: torque OFF")
    print()

    # Check for hardware errors
    errored: list[int] = []
    for mid in motors:
        hw_err, _, _ = pkt.read1ByteTxRx(ph, mid, HW_ERROR_ADDR)
        if hw_err:
            reasons = [name for bit, name in HW_ERROR_BITS.items() if hw_err & (1 << bit)]
            print(f"  ID {mid}: HW Error = {hw_err} ({', '.join(reasons) or 'Unknown'})")
            errored.append(mid)
        else:
            print(f"  ID {mid}: OK")

    if not errored:
        print("\nAll motors healthy — no reboot needed.")
        ph.closePort()
        return

    # Reboot errored motors
    print(f"\nRebooting {len(errored)} motor(s): {errored}")
    for mid in errored:
        comm, _ = pkt.reboot(ph, mid)
        status = "sent" if comm == 0 else "FAILED"
        print(f"  Reboot ID {mid}: {status}")

    print(f"Waiting {REBOOT_WAIT_S}s for motors to restart...")
    time.sleep(REBOOT_WAIT_S)

    # Verify
    print("\nVerifying...")
    all_clear = True
    for mid in errored:
        hw_err, _, _ = pkt.read1ByteTxRx(ph, mid, HW_ERROR_ADDR)
        if hw_err:
            print(f"  ID {mid}: still has error ({hw_err})")
            all_clear = False
        else:
            print(f"  ID {mid}: cleared ✓")

    if all_clear:
        print("\nAll errors cleared!")
    else:
        print("\nSome errors persist — check your power supply (XL430 needs 9-14.8V).")

    ph.closePort()


if __name__ == "__main__":
    for p in PORTS:
        scan_and_reboot(p)
