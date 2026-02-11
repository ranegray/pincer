#!/usr/bin/env python
"""
Simple single-arm test to verify torque, acceleration, and PID limits on bus1.

Usage:
    python -m lerobot.robots.xlerobot.examples.test_single_arm_limits

What it does:
    1. Connects to bus1 only (left arm, 6 motors)
    2. Reads current positions
    3. Applies your chosen motor limits (torque, acceleration, PID)
    4. Enables torque so the arm holds position
    5. Gently nudges each joint by a small offset so you can feel/see the response
    6. Lets you manually push the arm to feel the torque resistance
    7. Disconnects cleanly

Safety:
    - Starts with LOW torque (200/1000) and LOW acceleration (10)
    - Nudge is only ±40 raw steps (~3.5 degrees)
    - Press Ctrl+C at any time to disable torque and exit
"""

import time
import signal
import sys

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


# ── CONFIG: Change these to test different limits ──────────────────────────
PORT = "/dev/ttyACM0"

# Torque: 0-1000 (1000 = max). Start low!
TORQUE_LIMIT = 200

# Acceleration: 0-254. Lower = slower ramp-up. 0 = use Maximum_Acceleration default.
ACCELERATION = 10

# PID: Lower P = softer, slower response. Default P=32, D=32, I=0
P_COEFF = 8
I_COEFF = 0
D_COEFF = 32

# How far to nudge each joint (raw steps, 4096 = full revolution, 40 ≈ 3.5°)
NUDGE_STEPS = 40

# How long to hold each nudge position (seconds)
HOLD_TIME = 1.5
# ───────────────────────────────────────────────────────────────────────────

ARM_MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

MOTOR_NAMES = list(ARM_MOTORS.keys())


def cleanup(bus: FeetechMotorsBus):
    """Disable torque and disconnect."""
    print("\nDisabling torque...")
    try:
        bus.disable_torque()
        bus.disconnect()
    except Exception:
        pass
    print("Done. Arm is free.")


def main():
    bus = FeetechMotorsBus(port=PORT, motors=ARM_MOTORS)

    # Ctrl+C handler for safe shutdown
    def signal_handler(sig, frame):
        cleanup(bus)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # ── Step 1: Connect ──
    print(f"Connecting to bus on {PORT}...")
    bus.connect()
    print("Connected!\n")

    # ── Step 2: Disable torque to configure ──
    bus.disable_torque()

    # ── Step 3: Apply limits to each motor ──
    print("Applying motor limits:")
    print(f"  Torque_Limit  = {TORQUE_LIMIT}/1000")
    print(f"  Acceleration  = {ACCELERATION}")
    print(f"  P={P_COEFF}  I={I_COEFF}  D={D_COEFF}")
    print()

    for name in MOTOR_NAMES:
        bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
        bus.write("Torque_Limit", name, TORQUE_LIMIT)
        bus.write("Acceleration", name, ACCELERATION)
        bus.write("P_Coefficient", name, P_COEFF)
        bus.write("I_Coefficient", name, I_COEFF)
        bus.write("D_Coefficient", name, D_COEFF)

    # ── Step 4: Read current positions ──
    positions = bus.sync_read("Present_Position", MOTOR_NAMES)
    print("Current positions (raw):")
    for name, pos in positions.items():
        print(f"  {name:16s} = {pos}")
    print()

    # ── Step 5: Enable torque (arm holds position) ──
    bus.enable_torque()
    print("Torque ENABLED - arm is now holding position.\n")

    # ── Step 6: Push test ──
    input(
        ">> Try pushing each joint gently to feel the resistance.\n"
        "   With low torque you should be able to overpower the motors easily.\n"
        "   Press ENTER when ready to start the nudge test..."
    )

    # ── Step 7: Nudge test ──
    print(f"\nNudge test: moving each joint ±{NUDGE_STEPS} steps (~{NUDGE_STEPS * 360 / 4096:.1f}°)")
    print("Watch how smoothly each joint moves.\n")

    for name in MOTOR_NAMES:
        home = positions[name]

        # Nudge forward
        target_fwd = home + NUDGE_STEPS
        print(f"  {name}: {home} -> {target_fwd} (forward)")
        bus.write("Goal_Position", name, target_fwd)
        time.sleep(HOLD_TIME)

        # Nudge back
        target_bwd = home - NUDGE_STEPS
        print(f"  {name}: {target_fwd} -> {target_bwd} (backward)")
        bus.write("Goal_Position", name, target_bwd)
        time.sleep(HOLD_TIME)

        # Return home
        print(f"  {name}: {target_bwd} -> {home} (home)")
        bus.write("Goal_Position", name, home)
        time.sleep(HOLD_TIME)
        print()

    # ── Step 8: Read load feedback ──
    print("Reading Present_Load after nudge test:")
    loads = bus.sync_read("Present_Load", MOTOR_NAMES)
    for name, load in loads.items():
        print(f"  {name:16s} load = {load}")
    print()

    # ── Step 9: Hold for manual testing ──
    input(
        ">> Arm is holding position. Push joints again to feel resistance.\n"
        "   Press ENTER to disable torque and exit..."
    )

    cleanup(bus)


if __name__ == "__main__":
    main()
