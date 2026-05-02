# OMX Calibration: Why This Happens

## The short version

The two OMX arms ship pre-calibrated from ROBOTIS. That calibration is stored in each motor's EEPROM (specifically the `Homing_Offset` register at address 20), **not** in a file. The first time LeRobot connects to an arm without a matching JSON calibration file, its `bus.write_calibration()` method writes `homing_offset=0` into the motor EEPROM, **permanently overwriting ROBOTIS's per-unit factory values**.

After that point, the arms are no longer calibrated to each other. Matching pose → different reported angles. Teleop commanding the follower to the leader's reported angle drives the follower to a different physical pose, which in the worst case (shoulder_lift) means driving into the table and tripping Overload (HW error 32).

## Why the docs say "no calibration needed"

HuggingFace LeRobot docs and ROBOTIS docs both describe the ideal never-touched case:

1. Motor EEPROM has factory Homing_Offset values
2. The shipped calibration JSON approximately matches those values
3. On first connect, the two agree, and nothing gets wiped

Your arms fell out of this ideal state, most likely because:

- The ROBOTIS fork's shipped JSON had specific non-zero offsets that didn't match YOUR motors' factory values — so `write_calibration()` silently overwrote your factory EEPROM values with the JSON's values.
- Later, deleting the JSON + connecting via HF mainline wrote `homing_offset=0` into both arms' motors. Both arms are now at 0, but "0 for the leader" and "0 for the follower" correspond to different physical poses because the motors were mounted with unit-specific rotational offsets the factory calibration was hiding.

Neither HF nor ROBOTIS documents this failure mode. It's a real hole in both projects.

## Why shoulder_pan is 95° off specifically

The motors were mounted into the frames with arbitrary rotational offsets — ROBOTIS doesn't mechanically index the motors to a specific orientation, because the factory `Homing_Offset` EEPROM value is supposed to absorb that variance. Our motor shafts were clocked ~95° apart between leader and follower on joint 1, and the EEPROM compensation is gone. shoulder_lift, elbow_flex, wrist_flex happened to be mounted within a couple of degrees of each other — lucky. shoulder_pan and wrist_roll were less lucky.

## Recovery path

Since the factory EEPROM values are gone and not backed up anywhere recoverable:

1. **Measure the per-joint offset yourself.** Pose both arms identically by hand. Read both arms' raw encoder positions. The difference is the offset you need to apply as `homing_offset` on the follower.
2. **Save it to the follower calibration JSON.** `~/.cache/huggingface/lerobot/calibration/robots/omx_follower/omx_follower_arm.json`.
3. **Reconnect.** LeRobot writes the new offset into the motor EEPROM. Done.

That's what `calibration/align_follower.py` automates.

## Optional: contact ROBOTIS

ROBOTIS *may* keep per-unit calibration sheets tied to motor serial numbers. If they do, they could theoretically restore your factory `Homing_Offset` values. Worth asking in the Discord thread with @Enjin, but don't count on it.

## Lesson

If you ever set up a new OMX (or any pre-calibrated Dynamixel arm) with LeRobot, **back up the motor EEPROMs before first connect**. Read `Homing_Offset` (addr 20) from every motor and save it. `bus.write_calibration()` is a one-way door.
