# OMX Motor Setup & Calibration Guide

## Overview

This documents every problem we hit getting the ROBOTIS OpenManipulator-X (OMX) leader/follower arms working with LeRobot for teleoperation, what caused each problem, and how we fixed it.

**TL;DR:** LeRobot's stock calibration code has three destructive behaviors that break OMX gripper teleop. We wrote patched subclasses (`PatchedOmxLeader`, `PatchedOmxFollower`) and separate calibration scripts to work around them.

---

## Setup Run Order

After a fresh setup or if things get messed up, run these in order:

```bash
uv run python align_follower.py       # 1. Align body joints
uv run python calibrate_gripper.py    # 2. Calibrate gripper range + inversion
uv run python check_calibration.py    # 3. Verify (body joints ~0 diff)
uv run python teleop.py              # 4. Teleoperate
```

---

## Problem 1: Body Joint Misalignment

### Symptom
Leader and follower report different angles for the same physical pose. Teleop drives follower into table/limits.

### Root Cause
LeRobot's first-connect `calibrate()` writes `Homing_Offset=0` to every motor's EEPROM, permanently wiping the factory per-unit calibration values that ROBOTIS set to compensate for mounting angle variance between units.

### Fix: `align_follower.py`
1. Pose both arms identically by hand.
2. Calls `bus.set_half_turn_homings()` on **body joints only** (not gripper) — sets each motor's `Homing_Offset` so `Present_Position == 2047` at the current pose.
3. Saves to calibration JSON so `is_calibrated == True` on next connect.

**Key detail:** Gripper is explicitly excluded from `set_half_turn_homings()` because that function resets `range_min=0, range_max=4095`, destroying the gripper's separate range calibration.

---

## Problem 2: Gripper Range Not Calibrated

### Symptom
Body joints track perfectly, but follower gripper barely moves when leader trigger is squeezed.

### Root Cause
Body joints use most of the 0–4095 encoder range; centering with `Homing_Offset` is enough. But the gripper jaws only span ~1300 ticks (out of 4096). With the default `range_min=0, range_max=4095`, LeRobot's normalization formula (`percent = (raw - min) / (max - min) * 100`) maps the gripper's actual ~1300-tick physical range into a tiny ~30% slice of the 0–100% output. A full leader squeeze might only produce a 30% change → follower barely moves.

### Fix: `calibrate_gripper.py`
Manually measures the actual encoder positions at fully-open and fully-closed for both grippers, then writes those as `range_min` / `range_max`. Now 0% = jaws closed, 100% = jaws open (full range utilization).

---

## Problem 3: Gripper Encoder Direction (Software Inversion)

### Symptom
Gripper moves the wrong way, or calibration values become negative/nonsensical.

### Root Cause
On our unit, the leader gripper encoder counts **DOWN** when closing (open=high, close=low), but the follower encoder counts **UP** when closing (open=low, close=high). LeRobot always maps `percent=100` to `range_max` and `percent=0` to `range_min`. So without correction, "100% open" on leader maps to "100% closed" on follower.

### Why Not Use Firmware `Drive_Mode`?
We tried flipping the follower's firmware `Drive_Mode` to reverse encoder direction, but:
1. `DynamixelMotorsBus.apply_drive_mode = False` — LeRobot's software normalization **ignores** drive_mode on Dynamixel buses entirely. The field only affects firmware encoder direction.
2. Changing Drive_Mode after measuring extremes makes the saved `range_min/range_max` point to wrong physical positions (measured in old frame, interpreted in new frame).

### Fix: Software Inversion (Option B)
- Always set firmware `Drive_Mode=0` on both grippers (canonical encoder frame).
- Save honest `range_min = min(open_raw, close_raw)`, `range_max = max(open_raw, close_raw)`.
- Detect which arms need inversion: if `open_raw == range_min` (open is at the low end), the arm is "inverted."
- Save flags to `~/.cache/huggingface/lerobot/calibration/omx_gripper_inversion.json`.
- `PatchedOmxLeader.get_action()` and `PatchedOmxFollower.send_action()`/`get_observation()` flip `gripper.pos = 100 - gripper.pos` when the inversion flag is set.

**Our hardware:** `{"leader": true, "follower": false}`

---

## Problem 4: Stock `calibrate()` Wipes Everything

### Symptom
Gripper calibration works, but after any reconnect, values reset to `drive_mode=1, homing_offset=100, range=[0, 4095]`.

### Root Cause
Stock `OmxLeader.calibrate()` (line 87–112 in the LeRobot source) runs whenever `is_calibrated` returns `False` (any mismatch between JSON and motor EEPROM). It unconditionally resets the gripper to hardcoded defaults:
```python
drive_mode=1 (INVERTED), homing_offset=100, range_min=0, range_max=4095
```
This destroys our custom gripper range calibration.

### Fix: `PatchedOmxLeader.calibrate()` Override
Instead of resetting to hardcoded defaults, our override pushes whatever's already in the JSON to the EEPROM. If JSON has calibration → use it. If no JSON exists (first-ever connect) → use safe defaults with `drive_mode=0`.

---

## Problem 5: Stock `configure()` Hard-pins Gripper Settings

### Symptom
Even after saving correct gripper calibration, `Drive_Mode` and `Homing_Offset` revert on every connect.

### Root Cause
Stock `OmxLeader.configure()` (runs on every `connect()`) writes:
- `Drive_Mode = 1` (INVERTED) — hardcoded
- `Homing_Offset = 100` — hardcoded

This ignores whatever the calibration JSON says.

### Fix: `PatchedOmxLeader.configure()` Override
Reads `drive_mode` and `homing_offset` from `self.calibration["gripper"]` (the JSON) instead of hardcoding values. Whatever `calibrate_gripper.py` saved is what gets written.

---

## Problem 6: Leader Trigger Motor Resistance

### Symptom
Leader trigger at full squeeze reads `gripper.pos = 2.7%` instead of `0%`. Follower stops proportionally short of fully closed.

### Root Cause
The leader gripper motor runs in **current-controlled position mode** — it actively pushes the trigger back toward a rest position, acting like a spring. Stock `Goal_Current = 100mA` creates enough resistance that the user's hand can't push the trigger all the way to the calibrated extreme during normal teleop use.

Additionally, the `range_max` was measured during calibration with torque OFF (user could push freely to the absolute mechanical limit). During teleop with torque ON, the user physically can't reach that far.

### Fix
1. Reduced `Goal_Current` and `Current_Limit` from `100mA → 30mA` in `PatchedOmxLeader.configure()`. Trigger feels lighter, user can squeeze further.
2. Set `range_max` to the value the user actually reaches during normal teleop squeeze (~3040 instead of ~3187). Any squeeze past this gets clamped to 100% by LeRobot's normalization → follower goes to fully closed.

---

## Files Reference

### Scripts (run by user)

| Script | Purpose |
|---|---|
| `align_follower.py` | Align body joints (not gripper) between leader and follower |
| `calibrate_gripper.py` | Measure gripper range + direction, save calibration + inversion flags |
| `check_calibration.py` | Read-only snapshot comparing leader vs follower positions |
| `inspect_state.py` | Read-only deep diagnostic: EEPROM vs JSON for every motor |
| `teleop.py` | Main teleoperation loop |
| `diagnose_gripper_range.py` | Live encoder monitor to find raw gripper range by hand |
| `sweep_leader_current.py` | Sweep Goal_Current values to find usable trigger resistance |

### Core Module

| File | Purpose |
|---|---|
| `robot_utils.py` | `PatchedOmxLeader`, `PatchedOmxFollower`, `create_leader()`, `create_follower()`, gripper inversion loading |

### Calibration Data (on disk)

| Path | Contents |
|---|---|
| `~/.cache/huggingface/lerobot/calibration/teleoperators/omx_leader/omx_leader_arm.json` | Leader per-motor calibration (homing_offset, drive_mode, range_min, range_max) |
| `~/.cache/huggingface/lerobot/calibration/robots/omx_follower/omx_follower_arm.json` | Follower per-motor calibration |
| `~/.cache/huggingface/lerobot/calibration/omx_gripper_inversion.json` | Software inversion flags: `{"leader": true/false, "follower": true/false}` |

### Current Calibration Values (as of working state)

**Leader gripper:** `drive_mode=0, homing_offset=0, range_min=1459, range_max=3040, sw_invert=true`
**Follower gripper:** `drive_mode=0, homing_offset=0, range_min=1016, range_max=2279, sw_invert=false`

---

## Key Concepts

### LeRobot's Normalization Formula
```
percent = (Present_Position - range_min) / (range_max - range_min) × 100
```
- `percent=0` at `range_min`, `percent=100` at `range_max`
- Values outside the range are clamped
- On Dynamixel buses, `drive_mode` is **ignored** in software normalization (`apply_drive_mode = False`)

### `is_calibrated` Check
Compares JSON calibration values to motor EEPROM for all motors. Any mismatch → returns `False` → triggers `calibrate()`. This is why JSON and EEPROM must always be updated together — updating one without the other triggers a destructive recalibration.

### Current-Controlled Position Mode
The leader gripper motor has a position target (rest pose) but caps its torque at `Goal_Current` mA. This makes it act like a variable-stiffness spring: low current = soft trigger feel, high current = stiff. We use 30mA — enough to gently return the trigger when released, soft enough that the user can squeeze past it to reach the full range.

---

## Lesson Learned

If you ever set up a new OMX with LeRobot, **back up every motor's `Homing_Offset` EEPROM register before the first `connect()`**. LeRobot's `calibrate()` is a one-way door that wipes factory calibration permanently.
